"""
scripts/cnn/run_glcm_experiments.py

Controlled execution engine for training Bonn EEG GLCM-CNN on an M1 Mac.
Supports:
- Stage C: Single-case smoke test (Case 1, 3 epochs)
- Stage D: Full training across Case 1, Case 2, Case 3 (up to 30 epochs)
- Stratified 80:20 outer split with 10% inner validation of the training pool
- Balanced class weights computed strictly on the gradient training subset
- Early stopping (patience=5) and ReduceLROnPlateau (factor=0.5, patience=4)
- Full metrics computation and artifact generation
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cnn.model import BonnGLCMCNN
from scripts.cnn.config import GLCMModelConfig

logger = logging.getLogger("glcm_runner")


CASE_SPECS = {
    "Case1_3Class": {
        "name": "Case 1: Three-Class (A vs B vs C+D, Exclude E)",
        "case_id": "Case1",
        "num_classes": 3,
        "subsets_included": ["A", "B", "C", "D"],
        "subset_to_label": {"A": 0, "B": 1, "C": 2, "D": 2},
        "class_names": ["A (healthy open)", "B (healthy closed)", "C+D (interictal)"],
        "expected_total": 400,
        "expected_pool": 320,
        "expected_test": 80,
        "expected_subtrain": 288,
        "expected_val": 32,
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
        "expected_pool": 400,
        "expected_test": 100,
        "expected_subtrain": 360,
        "expected_val": 40,
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
        "expected_pool": 160,
        "expected_test": 40,
        "expected_subtrain": 144,
        "expected_val": 16,
        "class_proportions": {0: 100, 1: 100},
    },
}


class MemoryCachedGLCMDataset(Dataset):
    """
    In-memory or lazy cached PyTorch Dataset for Bonn EEG GLCM matrices.
    Loads and validates each (8, 16, 16) float32 tensor strictly with allow_pickle=False.
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
                # Channel sum check
                channel_sums = arr.sum(axis=(1, 2))
                if np.max(np.abs(channel_sums - 1.0)) > 1e-4:
                    raise ValueError(f"Channel sums deviate from 1.0 in {npy_path}")
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
    """Select best available hardware backend (MPS, CUDA, or CPU)."""
    if device_pref == "mps" or (device_pref == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        try:
            test_t = torch.ones(2, 2, device="mps")
            _ = test_t + test_t
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS requested but failed test allocation: {e}. Falling back to CPU.")
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
    Compute balanced class weights exclusively from training labels:
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


def generate_case_splits(
    manifest_records: List[Dict[str, Any]],
    case_key: str,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate reproducible, stratified 80:20 split + 10% inner validation of training pool.
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

    # Sort deterministically by recording_id before splitting
    case_records.sort(key=lambda x: x["recording_id"])
    total_count = len(case_records)

    if total_count != spec["expected_total"]:
        raise ValueError(
            f"Case {case_key} expected {spec['expected_total']} recordings, found {total_count}!"
        )

    labels = np.array([r["label"] for r in case_records])
    rec_ids = np.array([r["recording_id"] for r in case_records])
    indices = np.arange(total_count)

    # 1. Outer Stratified 80:20 split
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_seed)
    train_pool_idx, test_idx = next(sss_outer.split(indices, labels))

    train_pool_idx = sorted(train_pool_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    # 2. Inner Stratified 90:10 split of training pool
    train_pool_labels = labels[train_pool_idx]
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=random_seed)
    subtrain_local_idx, val_local_idx = next(sss_inner.split(train_pool_idx, train_pool_labels))

    subtrain_idx = sorted([train_pool_idx[i] for i in subtrain_local_idx])
    val_idx = sorted([train_pool_idx[i] for i in val_local_idx])

    # Assert pairwise disjointness
    s_tr, s_va, s_te = set(subtrain_idx), set(val_idx), set(test_idx)
    assert s_tr.isdisjoint(s_va), f"{case_key}: Sub-train and validation overlap!"
    assert s_tr.isdisjoint(s_te), f"{case_key}: Sub-train and test overlap!"
    assert s_va.isdisjoint(s_te), f"{case_key}: Validation and test overlap!"
    assert len(s_tr) + len(s_va) + len(s_te) == total_count, f"{case_key}: Partition count mismatch!"

    # Verify counts
    assert len(subtrain_idx) == spec["expected_subtrain"], (
        f"{case_key}: subtrain count {len(subtrain_idx)} != {spec['expected_subtrain']}"
    )
    assert len(val_idx) == spec["expected_val"], (
        f"{case_key}: val count {len(val_idx)} != {spec['expected_val']}"
    )
    assert len(test_idx) == spec["expected_test"], (
        f"{case_key}: test count {len(test_idx)} != {spec['expected_test']}"
    )

    subtrain_records = [case_records[i] for i in subtrain_idx]
    val_records = [case_records[i] for i in val_idx]
    test_records = [case_records[i] for i in test_idx]

    split_manifest = {
        "case_key": case_key,
        "case_name": spec["name"],
        "random_seed": random_seed,
        "total_count": total_count,
        "counts": {
            "total": total_count,
            "training_pool": len(train_pool_idx),
            "gradient_training": len(subtrain_records),
            "inner_validation": len(val_records),
            "held_out_test": len(test_records),
        },
        "class_counts": {
            "subtrain": {int(c): int(np.sum(labels[subtrain_idx] == c)) for c in range(spec["num_classes"])},
            "validation": {int(c): int(np.sum(labels[val_idx] == c)) for c in range(spec["num_classes"])},
            "test": {int(c): int(np.sum(labels[test_idx] == c)) for c in range(spec["num_classes"])},
        },
        "partitions": {
            "gradient_training_ids": [r["recording_id"] for r in subtrain_records],
            "validation_ids": [r["recording_id"] for r in val_records],
            "test_ids": [r["recording_id"] for r in test_records],
        },
    }

    return {
        "manifest": split_manifest,
        "subtrain_records": subtrain_records,
        "val_records": val_records,
        "test_records": test_records,
    }


def train_case(
    case_key: str,
    splits: Dict[str, Any],
    matrices_dir: Path,
    output_dir: Path,
    max_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 4,
    min_lr: float = 1e-6,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate a single case.
    """
    spec = CASE_SPECS[case_key]
    case_dir = output_dir / case_key
    case_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = select_device("auto")

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(random_seed)

    subtrain_ds = MemoryCachedGLCMDataset(splits["subtrain_records"], matrices_dir)
    val_ds = MemoryCachedGLCMDataset(splits["val_records"], matrices_dir)
    test_ds = MemoryCachedGLCMDataset(splits["test_records"], matrices_dir)

    train_loader = DataLoader(subtrain_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class weights strictly from gradient training subset
    subtrain_labels = [r["label"] for r in splits["subtrain_records"]]
    class_weights = compute_balanced_class_weights(subtrain_labels, spec["num_classes"], device)

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

    logger.info(f"[{case_key}] Starting training on {device} (Max epochs: {max_epochs}, Batch size: {batch_size})...")
    training_start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_targets, _ in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)

            # Check finite loss and shapes in smoke test
            loss = criterion(logits, batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

            loss.backward()

            # Check finite gradients
            for p in model.parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    raise RuntimeError(f"Non-finite gradients encountered at epoch {epoch}")

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
        epoch_duration = time.perf_counter() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 5))
        history["train_acc"].append(round(epoch_train_acc, 5))
        history["val_loss"].append(round(epoch_val_loss, 5))
        history["val_acc"].append(round(epoch_val_acc, 5))
        history["lr"].append(current_lr)
        history["epoch_duration_sec"].append(round(epoch_duration, 3))

        logger.info(
            f"[{case_key}] Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_duration:.2f}s"
        )

        # Learning rate scheduler
        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            logger.info(f"[{case_key}] LR reduced from {current_lr:.6f} to {new_lr:.6f}")

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
                f"[{case_key}] Early stopping triggered at epoch {epoch}. "
                f"Best validation accuracy was {best_val_acc*100:.2f}% at epoch {best_epoch}."
            )
            break

    total_training_duration = time.perf_counter() - training_start_time

    # Restore best validation checkpoint
    model.load_state_dict(best_weights)

    # Save best checkpoint
    checkpoint_path = case_dir / "best_checkpoint.pt"
    torch.save(
        {
            "case_key": case_key,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "model_state_dict": best_weights,
            "num_classes": spec["num_classes"],
            "class_weights": class_weights.cpu().numpy().tolist(),
            "random_seed": random_seed,
        },
        checkpoint_path,
    )

    # Evaluate held-out test set ONCE after restoring best checkpoint
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

    logger.info(
        f"[{case_key}] Held-Out Test Evaluation: "
        f"Acc: {test_acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}% | "
        f"Macro-F1: {macro_f1*100:.2f}% | Best Epoch: {best_epoch}"
    )

    # Save artifacts
    # 1. Training history CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(case_dir / "training_history.csv", index=False)

    # 2. Test predictions CSV
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
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(case_dir / "test_predictions.csv", index=False)

    # 3. Metrics JSON
    metrics_summary = {
        "case_key": case_key,
        "case_name": spec["name"],
        "num_classes": spec["num_classes"],
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
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
        },
    }
    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def run_pipeline(
    manifest_path: Path,
    matrices_dir: Path,
    output_dir: Path,
    smoke_test_only: bool = False,
    device_pref: str = "auto",
) -> Dict[str, Any]:
    """Execute complete smoke test or full training pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / "execution.log")

    device = select_device(device_pref)
    logger.info(f"Execution started on device: {device}")
    logger.info(f"Loading manifest from: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["records"]

    if smoke_test_only:
        logger.info("=== STAGE C: RUNNING SMOKE TEST ON CASE 1 (3 EPOCHS) ===")
        splits = generate_case_splits(records, "Case1_3Class", random_seed=42)
        # Save split manifest
        with open(output_dir / "Case1_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(splits["manifest"], f, indent=2)

        metrics = train_case(
            case_key="Case1_3Class",
            splits=splits,
            matrices_dir=matrices_dir,
            output_dir=output_dir,
            max_epochs=3,
            device=device,
            smoke_test=True,
        )
        logger.info("=== STAGE C: SMOKE TEST COMPLETED SUCCESSFULLY ===")
        return {"smoke_test": metrics}

    # Full training on all 3 cases
    logger.info("=== STAGE D: FULL TRAINING ACROSS ALL THREE CASES ===")
    all_results = {}
    for case_key in ["Case1_3Class", "Case2_Binary", "Case3_Binary"]:
        logger.info(f"\n--- Preparing Splits for {case_key} ---")
        splits = generate_case_splits(records, case_key, random_seed=42)
        with open(output_dir / f"{case_key}_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(splits["manifest"], f, indent=2)

        res = train_case(
            case_key=case_key,
            splits=splits,
            matrices_dir=matrices_dir,
            output_dir=output_dir,
            max_epochs=30,
            device=device,
            smoke_test=False,
        )
        all_results[case_key] = res

    # Write summary report
    summary_path = output_dir / "all_cases_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All cases completed. Summary saved to {summary_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN on Bonn EEG GLCM matrices")
    parser.add_argument("--manifest", type=str, default="data/processed/glcm_matrices_cnn/glcm_manifest.json")
    parser.add_argument("--matrices", type=str, default="data/processed/glcm_matrices_cnn/matrices")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-epoch smoke test on Case 1 only")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    run_pipeline(
        manifest_path=Path(args.manifest),
        matrices_dir=Path(args.matrices),
        output_dir=Path(args.output),
        smoke_test_only=args.smoke_test,
        device_pref=args.device,
    )
