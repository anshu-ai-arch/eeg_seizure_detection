"""
scripts/cnn/run_glcm_5fold_cv.py

5-Fold Stratified Cross-Validation Runner for the Bonn EEG GLCM-CNN Branch.
Evaluates 8-channel GLCM matrices [batch_size, 8, 16, 16] using:
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- 90:10 inner stratified split on development folds for validation/early stopping
- Balanced class weights computed strictly on gradient-training subset
- Fresh model, optimizer, and scheduler initialization per fold
- Max 30 epochs, AdamW (lr=0.005, weight_decay=1e-4), ReduceLROnPlateau, Early Stopping (patience=5)
- Best checkpoint restoration before held-out test evaluation
- Fold-level mean/std metrics and pooled out-of-fold metrics
"""

import sys
import os
import time
import json
import copy
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cnn.model import BonnGLCMCNN
from scripts.cnn.config import GLCMModelConfig

logger = logging.getLogger("glcm_5fold")

CASE_SPECS = {
    "Case1_3Class": {
        "name": "Case 1: Three-Class (A vs B vs C+D, Exclude E)",
        "case_id": "Case1",
        "num_classes": 3,
        "subsets_included": ["A", "B", "C", "D"],
        "subset_to_label": {"A": 0, "B": 1, "C": 2, "D": 2},
        "class_names": ["A (healthy open)", "B (healthy closed)", "C+D (interictal)"],
        "expected_total": 400,
        "expected_test_per_fold": 80,
        "expected_dev_per_fold": 320,
        "expected_subtrain_per_fold": 288,
        "expected_val_per_fold": 32,
        "class_proportions": {0: 100, 1: 100, 2: 200},
    },
    "Case2_Binary": {
        "name": "Case 2: Binary (A+B+C+D vs E)",
        "case_id": "Case2",
        "num_classes": 2,
        "subsets_included": ["A", "B", "C", "D", "E"],
        "subset_to_label": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 1},
        "class_names": ["non_ictal (A+B+C+D)", "ictal (E)"],
        "expected_total": 500,
        "expected_test_per_fold": 100,
        "expected_dev_per_fold": 400,
        "expected_subtrain_per_fold": 360,
        "expected_val_per_fold": 40,
        "class_proportions": {0: 400, 1: 100},
    },
    "Case3_Binary": {
        "name": "Case 3: Binary (D vs E, Exclude A, B, C)",
        "case_id": "Case3",
        "num_classes": 2,
        "subsets_included": ["D", "E"],
        "subset_to_label": {"D": 0, "E": 1},
        "class_names": ["interictal (D)", "ictal (E)"],
        "expected_total": 200,
        "expected_test_per_fold": 40,
        "expected_dev_per_fold": 160,
        "expected_subtrain_per_fold": 144,
        "expected_val_per_fold": 16,
        "class_proportions": {0: 100, 1: 100},
    },
}


class MemoryCachedGLCMDataset(Dataset):
    """
    In-memory cached PyTorch Dataset for Bonn EEG GLCM matrices.
    Loads and validates each (8, 16, 16) float32 tensor with allow_pickle=False.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        matrices_dir: Path,
        cache_in_memory: bool = True,
    ):
        self.records = records
        self.matrices_dir = Path(matrices_dir)
        self.cache_in_memory = cache_in_memory
        self.cached_tensors: Dict[str, torch.Tensor] = {}

        if self.cache_in_memory:
            for r in records:
                rec_id = r["recording_id"]
                npy_path = self.matrices_dir / f"{rec_id}.npy"
                if not npy_path.exists():
                    raise FileNotFoundError(f"Missing matrix file: {npy_path}")
                arr = np.load(npy_path, allow_pickle=False)
                if arr.shape != (8, 16, 16):
                    raise ValueError(f"Invalid shape {arr.shape} in {npy_path}")
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Non-finite values in {npy_path}")
                if np.any(arr < 0.0):
                    raise ValueError(f"Negative probabilities in {npy_path}")
                self.cached_tensors[rec_id] = torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        rec_id = rec["recording_id"]
        label = rec["label"]

        if self.cache_in_memory:
            tensor = self.cached_tensors[rec_id]
        else:
            npy_path = self.matrices_dir / f"{rec_id}.npy"
            arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
            tensor = torch.from_numpy(arr)

        return tensor, label, rec_id


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def select_device(device_pref: str = "auto") -> torch.device:
    """Select best hardware device."""
    if device_pref == "mps" or (device_pref == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        try:
            t = torch.ones(2, 2, device="mps")
            _ = t + t
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS allocation failed: {e}. Falling back to CPU.")
            return torch.device("cpu")
    elif device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def compute_balanced_class_weights(
    labels: List[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute balanced class weights strictly on gradient-training subset labels:
    w_c = N_total / (C * N_c)
    """
    arr = np.array(labels)
    total_samples = len(arr)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        count = int(np.sum(arr == c))
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def generate_5fold_splits(
    manifest_records: List[Dict[str, Any]],
    case_key: str,
    outer_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate reproducible 5-fold stratified cross-validation splits.
    For each fold:
    - 4 folds (80%) for development
    - 1 fold (20%) for held-out test
    - Inner 90:10 stratified split of dev pool for validation
    """
    spec = CASE_SPECS[case_key]
    case_records = []

    for r in manifest_records:
        sub = r["subset"]
        if sub in spec["subsets_included"]:
            lbl = spec["subset_to_label"][sub]
            case_records.append({
                "recording_id": r["recording_id"],
                "filename": r["filename"],
                "subset": sub,
                "label": lbl,
            })

    case_records.sort(key=lambda x: x["recording_id"])
    total_count = len(case_records)

    if total_count != spec["expected_total"]:
        raise ValueError(f"Case {case_key} expected {spec['expected_total']} recordings, found {total_count}!")

    labels = np.array([r["label"] for r in case_records])
    indices = np.arange(total_count)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)

    folds = []
    all_test_indices = []

    for fold_idx, (dev_idx, test_idx) in enumerate(skf.split(indices, labels)):
        dev_idx = sorted(dev_idx.tolist())
        test_idx = sorted(test_idx.tolist())
        all_test_indices.extend(test_idx)

        # Inner split of dev pool: 90% sub-train, 10% validation
        dev_labels = labels[dev_idx]
        inner_seed = outer_seed + fold_idx * 17
        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=inner_seed)
        subtrain_local, val_local = next(sss_inner.split(dev_idx, dev_labels))

        subtrain_idx = sorted([dev_idx[i] for i in subtrain_local])
        val_idx = sorted([dev_idx[i] for i in val_local])

        # Verify pairwise disjointness
        s_tr, s_va, s_te = set(subtrain_idx), set(val_idx), set(test_idx)
        assert s_tr.isdisjoint(s_va), f"Fold {fold_idx}: Sub-train and validation overlap!"
        assert s_tr.isdisjoint(s_te), f"Fold {fold_idx}: Sub-train and test overlap!"
        assert s_va.isdisjoint(s_te), f"Fold {fold_idx}: Validation and test overlap!"
        assert len(s_tr) + len(s_va) + len(s_te) == total_count, f"Fold {fold_idx}: Partition count mismatch!"

        # Verify exact counts
        assert len(subtrain_idx) == spec["expected_subtrain_per_fold"]
        assert len(val_idx) == spec["expected_val_per_fold"]
        assert len(test_idx) == spec["expected_test_per_fold"]

        fold_data = {
            "fold_idx": fold_idx,
            "inner_seed": inner_seed,
            "counts": {
                "subtrain": len(subtrain_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
                "dev_pool": len(dev_idx),
            },
            "class_counts": {
                "subtrain": {int(c): int(np.sum(labels[subtrain_idx] == c)) for c in range(spec["num_classes"])},
                "validation": {int(c): int(np.sum(labels[val_idx] == c)) for c in range(spec["num_classes"])},
                "test": {int(c): int(np.sum(labels[test_idx] == c)) for c in range(spec["num_classes"])},
            },
            "subtrain_records": [case_records[i] for i in subtrain_idx],
            "val_records": [case_records[i] for i in val_idx],
            "test_records": [case_records[i] for i in test_idx],
            "subtrain_ids": [case_records[i]["recording_id"] for i in subtrain_idx],
            "val_ids": [case_records[i]["recording_id"] for i in val_idx],
            "test_ids": [case_records[i]["recording_id"] for i in test_idx],
        }
        folds.append(fold_data)

    # Verify coverage across all 5 folds
    assert len(all_test_indices) == total_count, "Outer test set size mismatch!"
    assert len(set(all_test_indices)) == total_count, "Duplicate test recordings across outer folds!"

    manifest = {
        "case_key": case_key,
        "case_name": spec["name"],
        "num_classes": spec["num_classes"],
        "total_recordings": total_count,
        "outer_seed": outer_seed,
        "folds": [
            {
                "fold_idx": f["fold_idx"],
                "inner_seed": f["inner_seed"],
                "counts": f["counts"],
                "class_counts": f["class_counts"],
                "test_ids": f["test_ids"],
                "val_ids": f["val_ids"],
                "subtrain_ids": f["subtrain_ids"],
            }
            for f in folds
        ],
    }

    return {"manifest": manifest, "folds": folds, "case_records": case_records}


def train_single_fold(
    case_key: str,
    fold_data: Dict[str, Any],
    matrices_dir: Path,
    fold_dir: Path,
    max_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 4,
    min_lr: float = 1e-6,
    fold_seed: int = 42,
    device: Optional[torch.device] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate an isolated single fold."""
    spec = CASE_SPECS[case_key]
    fold_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = select_device("auto")

    # Set seeds for full fold reproducibility
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(fold_seed)

    subtrain_ds = MemoryCachedGLCMDataset(fold_data["subtrain_records"], matrices_dir)
    val_ds = MemoryCachedGLCMDataset(fold_data["val_records"], matrices_dir)
    test_ds = MemoryCachedGLCMDataset(fold_data["test_records"], matrices_dir)

    train_loader = DataLoader(subtrain_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class weights strictly from gradient training subset
    subtrain_labels = [r["label"] for r in fold_data["subtrain_records"]]
    class_weights = compute_balanced_class_weights(subtrain_labels, spec["num_classes"], device)

    # Brand new model, optimizer, scheduler per fold
    model = BonnGLCMCNN(num_classes=spec["num_classes"]).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_duration_sec": [],
    }

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        ep_start = time.perf_counter()

        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_targets, _ in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_targets)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")

            loss.backward()

            for p in model.parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    raise RuntimeError(f"Non-finite gradients at epoch {epoch}")

            optimizer.step()

            train_loss_accum += float(loss.item()) * len(batch_targets)
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == batch_targets).sum().item())
            train_total += len(batch_targets)

        epoch_train_loss = train_loss_accum / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_targets, _ in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_targets = batch_targets.to(device)
                logits = model(batch_imgs)
                loss = criterion(logits, batch_targets)

                val_loss_accum += float(loss.item()) * len(batch_targets)
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == batch_targets).sum().item())
                val_total += len(batch_targets)

        epoch_val_loss = val_loss_accum / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        epoch_dur = time.perf_counter() - ep_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 5))
        history["train_acc"].append(round(epoch_train_acc, 5))
        history["val_loss"].append(round(epoch_val_loss, 5))
        history["val_acc"].append(round(epoch_val_acc, 5))
        history["lr"].append(current_lr)
        history["epoch_duration_sec"].append(round(epoch_dur, 3))

        # Scheduler step based on validation accuracy
        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            logger.info(f"[{case_key} Fold {fold_data['fold_idx']}] LR reduced from {current_lr:.6f} to {new_lr:.6f}")

        # Checkpoint selection (strictly greater than best_val_acc, ties keep first best)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if not smoke_test and epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"[{case_key} Fold {fold_data['fold_idx']}] Early stopping at epoch {epoch}. "
                f"Best Val Acc was {best_val_acc*100:.2f}% at epoch {best_epoch}."
            )
            break

    total_training_duration = time.perf_counter() - t_start

    # Restore best validation checkpoint
    model.load_state_dict(best_weights)

    # Save best checkpoint
    checkpoint_path = fold_dir / "best_checkpoint.pt"
    torch.save(
        {
            "case_key": case_key,
            "fold_idx": fold_data["fold_idx"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "model_state_dict": best_weights,
            "num_classes": spec["num_classes"],
            "class_weights": class_weights.cpu().numpy().tolist(),
            "fold_seed": fold_seed,
        },
        checkpoint_path,
    )

    # Evaluate held-out test fold ONCE
    model.eval()
    test_loss_accum = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_rec_ids = []

    with torch.no_grad():
        for batch_imgs, batch_targets, batch_rec_ids in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(batch_imgs)
            loss = criterion(logits, batch_targets)

            test_loss_accum += float(loss.item()) * len(batch_targets)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_rec_ids.extend(batch_rec_ids)

    test_loss = test_loss_accum / max(1, len(all_targets))
    test_acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(spec["num_classes"])))
    cls_report = classification_report(
        all_targets,
        all_preds,
        target_names=spec["class_names"],
        output_dict=True,
        zero_division=0,
    )

    # Binary sensitivity & specificity
    sensitivity = None
    specificity = None
    if spec["num_classes"] == 2:
        # Class 1 is positive (ictal), Class 0 is negative (non-ictal / interictal)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Save per-fold history
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    # Save per-fold predictions
    pred_data = {
        "recording_id": all_rec_ids,
        "true_label": all_targets,
        "true_class_name": [spec["class_names"][t] for t in all_targets],
        "predicted_label": all_preds,
        "predicted_class_name": [spec["class_names"][p] for p in all_preds],
        "correct": [int(t == p) for t, p in zip(all_targets, all_preds)],
    }
    for c in range(spec["num_classes"]):
        pred_data[f"prob_class_{c}"] = [round(pr[c], 6) for pr in all_probs]
    pd.DataFrame(pred_data).to_csv(fold_dir / "test_predictions.csv", index=False)

    fold_metrics = {
        "case_key": case_key,
        "fold_idx": fold_data["fold_idx"],
        "fold_seed": fold_seed,
        "device": str(device),
        "total_training_duration_sec": round(total_training_duration, 2),
        "avg_epoch_duration_sec": round(float(np.mean(history["epoch_duration_sec"])), 3),
        "total_epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 5),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "test_metrics": {
            "test_loss": round(float(test_loss), 5),
            "test_accuracy": round(float(test_acc), 5),
            "balanced_accuracy": round(float(bal_acc), 5),
            "macro_f1": round(float(macro_f1), 5),
            "macro_precision": round(float(macro_prec), 5),
            "macro_recall": round(float(macro_rec), 5),
            "sensitivity": round(float(sensitivity), 5) if sensitivity is not None else None,
            "specificity": round(float(specificity), 5) if specificity is not None else None,
            "positive_class": "1 (ictal)" if spec["num_classes"] == 2 else None,
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
        },
        "predictions_summary": pred_data,
    }

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)

    logger.info(
        f"[{case_key} Fold {fold_data['fold_idx']}] Test Acc: {test_acc*100:.2f}% | "
        f"Bal Acc: {bal_acc*100:.2f}% | Macro-F1: {macro_f1*100:.2f}% | "
        f"Best Epoch: {best_epoch} | Time: {total_training_duration:.2f}s"
    )

    return fold_metrics


def run_case_5fold(
    case_key: str,
    manifest_records: List[Dict[str, Any]],
    matrices_dir: Path,
    case_dir: Path,
    device: torch.device,
    max_epochs: int = 30,
    outer_seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute complete 5-fold CV for a given classification case."""
    spec = CASE_SPECS[case_key]
    case_dir.mkdir(parents=True, exist_ok=True)

    split_obj = generate_5fold_splits(manifest_records, case_key, outer_seed=outer_seed)
    with open(case_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_obj["manifest"], f, indent=2)

    fold_metrics_list = []
    all_oof_predictions = []

    for fold_idx, fold_data in enumerate(split_obj["folds"]):
        if smoke_test and fold_idx > 0:
            break

        fold_dir = case_dir / f"fold_{fold_idx}"
        fold_seed = outer_seed + fold_idx * 100
        logger.info(f"\n>>> Running {case_key} Fold {fold_idx}/4 (Seed: {fold_seed}) <<<")

        f_metrics = train_single_fold(
            case_key=case_key,
            fold_data=fold_data,
            matrices_dir=matrices_dir,
            fold_dir=fold_dir,
            max_epochs=max_epochs,
            fold_seed=fold_seed,
            device=device,
            smoke_test=smoke_test,
        )
        fold_metrics_list.append(f_metrics)

        # Collect OOF predictions
        preds_df = pd.read_csv(fold_dir / "test_predictions.csv")
        preds_df["fold"] = fold_idx
        all_oof_predictions.append(preds_df)

    if smoke_test:
        return {"smoke_test_fold0": fold_metrics_list[0]}

    # Concat all OOF predictions (100% of the dataset)
    oof_df = pd.concat(all_oof_predictions, ignore_index=True)
    oof_df.to_csv(case_dir / "all_folds_test_predictions.csv", index=False)

    # 1. Fold-level statistics (mean +/- sample standard deviation, ddof=1)
    accs = [m["test_metrics"]["test_accuracy"] for m in fold_metrics_list]
    bal_accs = [m["test_metrics"]["balanced_accuracy"] for m in fold_metrics_list]
    f1s = [m["test_metrics"]["macro_f1"] for m in fold_metrics_list]
    precs = [m["test_metrics"]["macro_precision"] for m in fold_metrics_list]
    recs = [m["test_metrics"]["macro_recall"] for m in fold_metrics_list]

    fold_stats = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)),
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs, ddof=1)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s, ddof=1)),
        "mean_macro_precision": float(np.mean(precs)),
        "std_macro_precision": float(np.std(precs, ddof=1)),
        "mean_macro_recall": float(np.mean(recs)),
        "std_macro_recall": float(np.std(recs, ddof=1)),
    }

    if spec["num_classes"] == 2:
        sens_list = [m["test_metrics"]["sensitivity"] for m in fold_metrics_list]
        spec_list = [m["test_metrics"]["specificity"] for m in fold_metrics_list]
        fold_stats["mean_sensitivity"] = float(np.mean(sens_list))
        fold_stats["std_sensitivity"] = float(np.std(sens_list, ddof=1))
        fold_stats["mean_specificity"] = float(np.mean(spec_list))
        fold_stats["std_specificity"] = float(np.std(spec_list, ddof=1))
        fold_stats["positive_class"] = "1 (ictal)"

    # 2. Pooled Out-Of-Fold metrics (calculated from concatenated OOF predictions)
    oof_targets = oof_df["true_label"].values
    oof_preds = oof_df["predicted_label"].values
    pooled_cm = confusion_matrix(oof_targets, oof_preds, labels=list(range(spec["num_classes"])))
    pooled_acc = accuracy_score(oof_targets, oof_preds)
    pooled_bal_acc = balanced_accuracy_score(oof_targets, oof_preds)
    pooled_f1 = f1_score(oof_targets, oof_preds, average="macro", zero_division=0)
    pooled_prec = precision_score(oof_targets, oof_preds, average="macro", zero_division=0)
    pooled_rec = recall_score(oof_targets, oof_preds, average="macro", zero_division=0)
    pooled_report = classification_report(
        oof_targets,
        oof_preds,
        target_names=spec["class_names"],
        output_dict=True,
        zero_division=0,
    )

    pooled_sens = None
    pooled_spec = None
    if spec["num_classes"] == 2:
        tn, fp, fn, tp = pooled_cm.ravel()
        pooled_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        pooled_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    pooled_metrics = {
        "pooled_accuracy": float(pooled_acc),
        "pooled_balanced_accuracy": float(pooled_bal_acc),
        "pooled_macro_f1": float(pooled_f1),
        "pooled_macro_precision": float(pooled_prec),
        "pooled_macro_recall": float(pooled_rec),
        "pooled_sensitivity": pooled_sens,
        "pooled_specificity": pooled_spec,
        "pooled_confusion_matrix": pooled_cm.tolist(),
        "pooled_classification_report": pooled_report,
    }

    # Build fold metrics summary table
    table_rows = []
    for f in fold_metrics_list:
        row = {
            "fold": f["fold_idx"],
            "seed": f["fold_seed"],
            "best_epoch": f["best_epoch"],
            "total_epochs": f["total_epochs_trained"],
            "best_val_acc": f["best_val_accuracy"],
            "test_acc": f["test_metrics"]["test_accuracy"],
            "balanced_acc": f["test_metrics"]["balanced_accuracy"],
            "macro_f1": f["test_metrics"]["macro_f1"],
            "macro_prec": f["test_metrics"]["macro_precision"],
            "macro_rec": f["test_metrics"]["macro_recall"],
            "duration_sec": f["total_training_duration_sec"],
        }
        if spec["num_classes"] == 2:
            row["sensitivity"] = f["test_metrics"]["sensitivity"]
            row["specificity"] = f["test_metrics"]["specificity"]
        table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(case_dir / "fold_metrics_table.csv", index=False)

    case_summary = {
        "case_key": case_key,
        "case_name": spec["name"],
        "num_classes": spec["num_classes"],
        "total_samples": spec["expected_total"],
        "fold_level_statistics": fold_stats,
        "pooled_oof_metrics": pooled_metrics,
        "individual_folds": fold_metrics_list,
    }

    with open(case_dir / "case_summary.json", "w", encoding="utf-8") as f:
        json.dump(case_summary, f, indent=2)

    logger.info(
        f"\n[{case_key} 5-Fold Summary] "
        f"Mean Acc: {fold_stats['mean_accuracy']*100:.2f}% +/- {fold_stats['std_accuracy']*100:.2f}% | "
        f"Pooled Acc: {pooled_acc*100:.2f}% | "
        f"Mean Macro-F1: {fold_stats['mean_macro_f1']*100:.2f}% +/- {fold_stats['std_macro_f1']*100:.2f}% | "
        f"Pooled Macro-F1: {pooled_f1*100:.2f}%"
    )

    return case_summary


def run_pipeline(
    manifest_path: Path,
    matrices_dir: Path,
    output_dir: Path,
    smoke_test_only: bool = False,
    device_pref: str = "auto",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / "execution.log")

    device = select_device(device_pref)
    logger.info(f"5-Fold CV execution started on device: {device}")
    logger.info(f"Manifest path: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["records"]

    if smoke_test_only:
        logger.info("=== STAGE 3: SMOKE TEST ON CASE 1 FOLD 0 (2 EPOCHS) ===")
        res = run_case_5fold(
            case_key="Case1_3Class",
            manifest_records=records,
            matrices_dir=matrices_dir,
            case_dir=output_dir / "smoke_test_case1",
            device=device,
            max_epochs=2,
            smoke_test=True,
        )
        logger.info("=== STAGE 3: SMOKE TEST COMPLETED SUCCESSFULLY ===")
        return res

    # Full 5-fold CV across all three cases
    logger.info("=== STAGE 4: COMPLETE 5-FOLD CV ACROSS ALL THREE CASES (15 FITS) ===")
    all_cases_summary = {}

    for case_key in ["Case1_3Class", "Case2_Binary", "Case3_Binary"]:
        logger.info(f"\n========================================================")
        logger.info(f"STARTING 5-FOLD CROSS-VALIDATION FOR: {case_key}")
        logger.info(f"========================================================")

        res = run_case_5fold(
            case_key=case_key,
            manifest_records=records,
            matrices_dir=matrices_dir,
            case_dir=output_dir / case_key,
            device=device,
            max_epochs=30,
            smoke_test=False,
        )
        all_cases_summary[case_key] = res

    summary_file = output_dir / "all_cases_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_cases_summary, f, indent=2)

    logger.info(f"\nAll 15 fold fits completed successfully! Summary written to {summary_file}")
    return all_cases_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 5-Fold CV on Bonn EEG GLCM CNN")
    parser.add_argument("--manifest", type=str, default="data/processed/glcm_matrices_cnn/glcm_manifest.json")
    parser.add_argument("--matrices", type=str, default="data/processed/glcm_matrices_cnn/matrices")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke test on Case 1 Fold 0 only")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    run_pipeline(
        manifest_path=Path(args.manifest),
        matrices_dir=Path(args.matrices),
        output_dir=Path(args.output),
        smoke_test_only=args.smoke_test,
        device_pref=args.device,
    )
