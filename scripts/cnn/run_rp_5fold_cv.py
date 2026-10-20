"""
scripts/cnn/run_rp_5fold_cv.py

5-Fold Stratified Cross-Validation Runner for Continuous Recurrence Plot (GRP) CNN Branch.
Evaluates continuous 128x128 GRP images [batch_size, 1, 128, 128] using:
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- 90:10 inner stratified split on development folds for validation and early stopping
- Balanced class weights computed strictly on gradient-training subset
- Fresh model (BonnModernRPCNN), optimizer, and scheduler initialization per fold
- Max 30 epochs, AdamW (lr=0.0005, weight_decay=1e-4), ReduceLROnPlateau, Early Stopping (patience=7)
- Tie-breaking early stopping: restores checkpoint with best val_acc (or lower val_loss on tie)
- Restores best checkpoint before held-out test evaluation
- Evaluates Case 1 (3-Class: A vs B vs C+D), Case 2 (Binary: A-D vs E), and Case 3 (Binary: D vs E)
- Computes individual fold metrics, mean/std, and pooled out-of-fold metrics
"""

import sys
import os
import time
import json
import copy
import logging
import argparse
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

from scripts.cnn.model import BonnModernRPCNN

logger = logging.getLogger("rp_5fold")

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


class MemoryCachedRPDataset(Dataset):
    """
    In-memory cached PyTorch Dataset for continuous 128x128 Bonn EEG Recurrence Plots.
    Loads and validates each (1, 128, 128) float32 tensor from .npy or .png.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        images_dir: Path,
        cache_in_memory: bool = True,
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.cache_in_memory = cache_in_memory
        self.cached_tensors: Dict[str, torch.Tensor] = {}

        if self.cache_in_memory:
            for r in records:
                rec_id = r["recording_id"]
                npy_path = self.images_dir / "npy" / f"{rec_id}.npy"
                if not npy_path.exists():
                    npy_path = self.images_dir / f"{rec_id}.npy"

                if npy_path.exists():
                    arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
                else:
                    # Fallback to PNG
                    from PIL import Image
                    png_path = self.images_dir / "images" / f"{rec_id}.png"
                    if not png_path.exists():
                        png_path = self.images_dir / f"{rec_id}.png"
                    if not png_path.exists():
                        raise FileNotFoundError(f"Missing RP file for {rec_id} in {self.images_dir}")
                    pil_img = Image.open(png_path).convert("L")
                    arr = np.array(pil_img, dtype=np.float32) / 255.0

                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)  # Shape (1, 128, 128)

                if arr.shape != (1, 128, 128):
                    raise ValueError(f"Invalid shape {arr.shape} for recording {rec_id}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"Non-finite values in recording {rec_id}")

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
            npy_path = self.images_dir / "npy" / f"{rec_id}.npy"
            arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
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


def run_single_fold_training(
    fold_idx: int,
    subtrain_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
    max_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    fold_seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any], int, float, Dict[str, List[float]]]:
    """
    Train a fresh BonnModernRPCNN on the sub-training subset with validation monitoring.
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    model = BonnModernRPCNN(num_classes=num_classes, dropout_rate=0.4).to(device)

    train_loader = DataLoader(
        subtrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_val_loss = float("inf")
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
    }

    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.item()) * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == batch_y).sum().item())
            train_total += len(batch_y)

        ep_train_loss = train_loss_accum / max(1, train_total)
        ep_train_acc = train_correct / max(1, train_total)

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss_accum += float(loss.item()) * len(batch_y)
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == batch_y).sum().item())
                val_total += len(batch_y)

        ep_val_loss = val_loss_accum / max(1, val_total)
        ep_val_acc = val_correct / max(1, val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(ep_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(ep_train_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)
        history["lr"].append(current_lr)

        # Early Stopping & Best Checkpoint check (composite: accuracy first, tie-break by loss)
        is_improved = False
        if ep_val_acc > best_val_acc:
            is_improved = True
        elif np.isclose(ep_val_acc, best_val_acc, atol=1e-5) and ep_val_loss < best_val_loss - 1e-4:
            is_improved = True

        if is_improved:
            best_val_acc = ep_val_acc
            best_val_loss = ep_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Fold {fold_idx}: Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (val_acc={best_val_acc:.4f}, val_loss={best_val_loss:.4f})")
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    meta = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["epoch"]),
        "early_stopped": epochs_no_improve >= early_stopping_patience,
    }
    return model, meta, best_epoch, best_val_acc, history


def evaluate_model_on_test(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Held-out test set evaluation returning exhaustive metrics and sample predictions.
    """
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_sample_ids = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y, batch_ids in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.numpy().tolist())
            all_sample_ids.extend(list(batch_ids))
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels_list = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0).tolist()
    r_per_class = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0).tolist()

    per_class = {}
    for c in range(num_classes):
        c_name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class[c_name] = {
            "class_index": c,
            "precision": float(p_per_class[c]),
            "recall": float(r_per_class[c]),
            "f1": float(f1_per_class[c]),
            "support": int(np.sum(y_true == c)),
        }

    eval_results = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "sample_ids": all_sample_ids,
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred],
        "y_probs": probs_arr.tolist(),
    }

    # Binary specifics
    if num_classes == 2:
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        eval_results["sensitivity"] = sens
        eval_results["specificity"] = spec
        eval_results["true_positives"] = tp
        eval_results["true_negatives"] = tn
        eval_results["false_positives"] = fp
        eval_results["false_negatives"] = fn

    return eval_results


def run_rp_5fold_benchmark(
    case_key: str,
    manifest_records: List[Dict[str, Any]],
    images_dir: Path,
    output_case_dir: Path,
    device: torch.device,
    max_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    n_splits: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute 5-fold cross-validation for a specific benchmark case on RP images.
    """
    spec = CASE_SPECS[case_key]
    case_name = spec["name"]
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]
    subsets_inc = spec["subsets_included"]
    sub_to_lbl = spec["subset_to_label"]

    logger.info(f"\n=======================================================")
    logger.info(f"Starting {case_name}")
    logger.info(f"Target: {num_classes} classes, subsets: {subsets_inc}")
    logger.info(f"=======================================================")

    output_case_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_case_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Filter records for this case
    case_records = []
    for r in manifest_records:
        sub = r["subset"]
        if sub in subsets_inc:
            item = dict(r)
            item["label"] = sub_to_lbl[sub]
            case_records.append(item)

    if len(case_records) != spec["expected_total"]:
        raise ValueError(
            f"{case_name}: Expected {spec['expected_total']} records, found {len(case_records)}"
        )

    case_records.sort(key=lambda x: x["recording_id"])
    rec_ids = [r["recording_id"] for r in case_records]
    labels = np.array([r["label"] for r in case_records], dtype=np.int64)

    logger.info(f"Total samples: {len(case_records)}. Class counts: {dict(pd.Series(labels).value_counts())}")

    # Load in-memory dataset
    logger.info(f"Loading and caching {len(case_records)} RP images in memory...")
    t_load_start = time.time()
    rp_dataset = MemoryCachedRPDataset(case_records, images_dir=images_dir, cache_in_memory=True)
    logger.info(f"Images cached in {time.time() - t_load_start:.2f}s.")

    # 5-fold Stratified split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    fold_splits = list(skf.split(rec_ids, labels))

    fold_results_list = []
    pooled_y_true = []
    pooled_y_pred = []
    pooled_ids = []
    pooled_probs = []

    for fold_idx, (dev_pos, test_pos) in enumerate(fold_splits):
        fold_seed = base_seed + fold_idx * 100
        fold_dir = output_case_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_records = [case_records[i] for i in test_pos]
        dev_records = [case_records[i] for i in dev_pos]

        # 90:10 subtrain / val split on development set
        dev_labels = np.array([r["label"] for r in dev_records])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=fold_seed)
        subtrain_pos, val_pos = next(sss.split(dev_records, dev_labels))

        subtrain_records = [dev_records[i] for i in subtrain_pos]
        val_records = [dev_records[i] for i in val_pos]

        # Verify zero overlap
        s_sub = set(r["recording_id"] for r in subtrain_records)
        s_val = set(r["recording_id"] for r in val_records)
        s_test = set(r["recording_id"] for r in test_records)
        assert len(s_sub.intersection(s_val)) == 0, f"Fold {fold_idx}: subtrain and val overlap!"
        assert len(s_sub.intersection(s_test)) == 0, f"Fold {fold_idx}: subtrain and test overlap!"
        assert len(s_val.intersection(s_test)) == 0, f"Fold {fold_idx}: val and test overlap!"

        # Save split manifest
        split_meta = {
            "fold_idx": fold_idx,
            "random_seed": fold_seed,
            "test_sample_ids": [r["recording_id"] for r in test_records],
            "subtrain_sample_ids": [r["recording_id"] for r in subtrain_records],
            "val_sample_ids": [r["recording_id"] for r in val_records],
            "test_count": len(test_records),
            "subtrain_count": len(subtrain_records),
            "val_count": len(val_records),
        }
        with open(manifests_dir / f"fold_{fold_idx}_split.json", "w") as f:
            json.dump(split_meta, f, indent=2)

        subtrain_ds = MemoryCachedRPDataset(subtrain_records, images_dir=images_dir, cache_in_memory=True)
        val_ds = MemoryCachedRPDataset(val_records, images_dir=images_dir, cache_in_memory=True)
        test_ds = MemoryCachedRPDataset(test_records, images_dir=images_dir, cache_in_memory=True)

        # Class weights from subtrain
        subtrain_lbls = [r["label"] for r in subtrain_records]
        class_weights = compute_balanced_class_weights(subtrain_lbls, num_classes=num_classes, device=device)

        logger.info(f"--- Fold {fold_idx} --- (Test: {len(test_records)}, Subtrain: {len(subtrain_records)}, Val: {len(val_records)})")
        t_fold_start = time.time()

        trained_model, train_meta, best_ep, best_vacc, history = run_single_fold_training(
            fold_idx=fold_idx,
            subtrain_dataset=subtrain_ds,
            val_dataset=val_ds,
            num_classes=num_classes,
            class_weights=class_weights,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            fold_seed=fold_seed,
        )

        fold_train_sec = time.time() - t_fold_start

        # Save checkpoint
        torch.save(trained_model.state_dict(), fold_dir / "best_checkpoint.pt")

        # Save history CSV
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(fold_dir / "training_history.csv", index=False)

        # Evaluate on held-out test
        test_eval = evaluate_model_on_test(
            model=trained_model,
            test_dataset=test_ds,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
        )

        # Save test predictions CSV
        pred_rows = []
        for sid, yt, yp, prb in zip(
            test_eval["sample_ids"], test_eval["y_true"], test_eval["y_pred"], test_eval["y_probs"]
        ):
            row = {
                "recording_id": sid,
                "true_label": yt,
                "predicted_label": yp,
                "is_correct": int(yt == yp),
            }
            for c_i, p_val in enumerate(prb):
                row[f"prob_class_{c_i}"] = p_val
            pred_rows.append(row)

        df_preds = pd.DataFrame(pred_rows)
        df_preds.to_csv(fold_dir / "test_predictions.csv", index=False)

        # Save fold metrics JSON
        fold_metrics = {
            "fold_idx": fold_idx,
            "random_seed": fold_seed,
            "test_accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_precision": test_eval["macro_precision"],
            "macro_recall": test_eval["macro_recall"],
            "macro_f1": test_eval["macro_f1"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "per_class": test_eval["per_class"],
            "best_epoch": best_ep,
            "best_val_accuracy": best_vacc,
            "best_val_loss": train_meta["best_val_loss"],
            "total_epochs_trained": train_meta["epochs_trained"],
            "early_stopped": train_meta["early_stopped"],
            "train_duration_sec": fold_train_sec,
        }
        if num_classes == 2:
            fold_metrics["sensitivity"] = test_eval["sensitivity"]
            fold_metrics["specificity"] = test_eval["specificity"]
            fold_metrics["true_positives"] = test_eval["true_positives"]
            fold_metrics["true_negatives"] = test_eval["true_negatives"]
            fold_metrics["false_positives"] = test_eval["false_positives"]
            fold_metrics["false_negatives"] = test_eval["false_negatives"]

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        fold_results_list.append(fold_metrics)

        # Accumulate pooled OOF
        pooled_ids.extend(test_eval["sample_ids"])
        pooled_y_true.extend(test_eval["y_true"])
        pooled_y_pred.extend(test_eval["y_pred"])
        pooled_probs.extend(test_eval["y_probs"])

        logger.info(
            f"Fold {fold_idx} Result: Test Acc={test_eval['accuracy']*100:.2f}%, "
            f"Bal Acc={test_eval['balanced_accuracy']*100:.2f}%, "
            f"Macro F1={test_eval['macro_f1']*100:.2f}%, "
            f"Best Epoch={best_ep}/{train_meta['epochs_trained']} in {fold_train_sec:.2f}s"
        )
        if num_classes == 2:
            logger.info(f"  -> Sensitivity (E)={test_eval['sensitivity']*100:.2f}%, Specificity={test_eval['specificity']*100:.2f}% (FP={test_eval['false_positives']}, FN={test_eval['false_negatives']})")

    # Pooled Metrics
    p_true = np.array(pooled_y_true)
    p_pred = np.array(pooled_y_pred)

    pooled_acc = float(accuracy_score(p_true, p_pred))
    pooled_bal_acc = float(balanced_accuracy_score(p_true, p_pred))
    pooled_macro_p = float(precision_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_macro_r = float(recall_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_macro_f1 = float(f1_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_cm = confusion_matrix(p_true, p_pred, labels=list(range(num_classes)))

    # Fold Summary Statistics (mean +- std)
    accs = [f["test_accuracy"] for f in fold_results_list]
    bal_accs = [f["balanced_accuracy"] for f in fold_results_list]
    f1s = [f["macro_f1"] for f in fold_results_list]

    summary = {
        "case_key": case_key,
        "case_name": case_name,
        "num_classes": num_classes,
        "n_samples": len(case_records),
        "pooled_metrics": {
            "accuracy": pooled_acc,
            "balanced_accuracy": pooled_bal_acc,
            "macro_precision": pooled_macro_p,
            "macro_recall": pooled_macro_r,
            "macro_f1": pooled_macro_f1,
            "confusion_matrix": pooled_cm.tolist(),
        },
        "mean_fold_metrics": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "balanced_accuracy_mean": float(np.mean(bal_accs)),
            "balanced_accuracy_std": float(np.std(bal_accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        },
        "fold_results": fold_results_list,
    }

    if num_classes == 2:
        sens_list = [f["sensitivity"] for f in fold_results_list]
        specs_list = [f["specificity"] for f in fold_results_list]
        summary["mean_fold_metrics"]["sensitivity_mean"] = float(np.mean(sens_list))
        summary["mean_fold_metrics"]["sensitivity_std"] = float(np.std(sens_list))
        summary["mean_fold_metrics"]["specificity_mean"] = float(np.mean(specs_list))
        summary["mean_fold_metrics"]["specificity_std"] = float(np.std(specs_list))

        p_tn, p_fp, p_fn, p_tp = int(pooled_cm[0, 0]), int(pooled_cm[0, 1]), int(pooled_cm[1, 0]), int(pooled_cm[1, 1])
        summary["pooled_metrics"]["sensitivity"] = float(p_tp / (p_tp + p_fn)) if (p_tp + p_fn) > 0 else 0.0
        summary["pooled_metrics"]["specificity"] = float(p_tn / (p_tn + p_fp)) if (p_tn + p_fp) > 0 else 0.0
        summary["pooled_metrics"]["total_false_positives"] = p_fp
        summary["pooled_metrics"]["total_false_negatives"] = p_fn

    with open(output_case_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=== {case_name} Complete ===")
    logger.info(f"Mean Accuracy: {np.mean(accs)*100:.2f}% +- {np.std(accs)*100:.2f}% | Pooled OOF: {pooled_acc*100:.2f}%")
    logger.info(f"Mean Macro-F1: {np.mean(f1s)*100:.2f}% +- {np.std(f1s)*100:.2f}% | Pooled OOF: {pooled_macro_f1*100:.2f}%")
    if num_classes == 2:
        logger.info(f"Mean Sens: {np.mean(sens_list)*100:.2f}%, Spec: {np.mean(specs_list)*100:.2f}%")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-Fold CV on Continuous Bonn EEG RP Images")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/rp_images_128/image_manifest.json",
        help="Path to image_manifest.json",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/processed/rp_images_128",
        help="Path to directory containing images/ and npy/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=["all", "Case1_3Class", "Case2_Binary", "Case3_Binary"],
        help="Specific case to evaluate or 'all'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Max epochs per fold (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Initial learning rate for AdamW (default: 0.0005)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'mps', 'cuda', 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = PROJECT_ROOT / "experiments" / f"results_cnn_rp_5fold_{timestamp}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(exp_dir / "experiment.log")

    logger.info(f"Initializing Bonn EEG Continuous RP 5-Fold Cross-Validation...")
    logger.info(f"Experiment Directory: {exp_dir}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}!")

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    records = manifest_data["records"]
    logger.info(f"Loaded manifest with {len(records)} recording records.")

    device = select_device(args.device)
    logger.info(f"Compute Device: {device}")

    cases_to_run = (
        ["Case1_3Class", "Case2_Binary", "Case3_Binary"]
        if args.case == "all"
        else [args.case]
    )

    all_summaries = {}
    for c_key in cases_to_run:
        c_dir = exp_dir / c_key
        summary = run_rp_5fold_benchmark(
            case_key=c_key,
            manifest_records=records,
            images_dir=Path(args.images_dir),
            output_case_dir=c_dir,
            device=device,
            max_epochs=args.epochs,
            learning_rate=args.lr,
        )
        all_summaries[c_key] = summary

    # Write consolidated summary
    with open(exp_dir / "consolidated_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"\n=======================================================")
    logger.info(f"ALL BENCHMARKS FINISHED SUCCESSFULLY!")
    logger.info(f"Results written to: {exp_dir}")
    logger.info(f"=======================================================\n")


if __name__ == "__main__":
    main()
