"""
scripts/cnn/run_hybrid_rp_5fold_cv.py

5-Fold Stratified Cross-Validation Runner for Multimodal Hybrid RP-CNN Branch.
Fuses:
1. Continuous 128x128 GRP recurrence plot images (Deep visual patterns via 4-block ConvNet)
2. Physiological dynamical and spectral scalars:
   - tau (AMI time delay)
   - m (FNN embedding dimension)
   - raw_std (Signal amplitude standard deviation)
   - alpha_ratio (Relative 8-13 Hz alpha band power)
   - theta_ratio (Relative 4-8 Hz theta band power)
   - delta_ratio (Relative 0.5-4 Hz delta band power)
   - gamma_ratio (Relative >30 Hz gamma band power)

Evaluates:
- Case 1: Three-Class (A vs B vs C+D, N=400)
- Case 2: Binary Seizure Detection (A+B+C+D vs E, N=500)
- Case 3: Binary Interictal vs Ictal (D vs E, N=200)

Guarantees:
- Strict fold isolation: StandardScaler fitted strictly on sub-training fold.
- Zero data leakage between partitions.
- Fresh model, optimizer, scheduler per fold.
- Tie-breaking early stopping with patience 7.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cnn.model import BonnHybridRPCNN

logger = logging.getLogger("hybrid_rp_5fold")

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
    },
}


def compute_spectral_scalars(raw_dir: Path) -> Dict[str, Dict[str, float]]:
    """Compute relative band powers and raw std for each recording."""
    logger.info("Computing physiological spectral features from raw EEG...")
    scalar_dict = {}
    subsets = ["A", "B", "C", "D", "E"]

    for s in subsets:
        s_dir = raw_dir / s
        for f in sorted(s_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith("."):
                rec_id = f.stem
                sig = np.loadtxt(f, dtype=np.float64)
                raw_std = float(np.std(sig))

                freqs, psd = welch(sig, fs=173.61, nperseg=256)
                tot = np.sum(psd) + 1e-12

                delta_p = float(np.sum(psd[(freqs >= 0.5) & (freqs < 4.0)]) / tot)
                theta_p = float(np.sum(psd[(freqs >= 4.0) & (freqs < 8.0)]) / tot)
                alpha_p = float(np.sum(psd[(freqs >= 8.0) & (freqs <= 13.0)]) / tot)
                beta_p = float(np.sum(psd[(freqs > 13.0) & (freqs <= 30.0)]) / tot)
                gamma_p = float(np.sum(psd[freqs > 30.0]) / tot)

                scalar_dict[rec_id] = {
                    "raw_std": raw_std,
                    "delta_ratio": delta_p,
                    "theta_ratio": theta_p,
                    "alpha_ratio": alpha_p,
                    "beta_ratio": beta_p,
                    "gamma_ratio": gamma_p,
                }
    logger.info(f"Spectral features computed for {len(scalar_dict)} recordings.")
    return scalar_dict


class HybridRPDataset(Dataset):
    """
    Dataset returning (image_tensor, scalar_vector, label, rec_id).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cached_tensors: Dict[str, torch.Tensor],
        scalar_matrix: np.ndarray,
    ):
        self.records = records
        self.cached_tensors = cached_tensors
        self.scalar_matrix = torch.from_numpy(scalar_matrix.astype(np.float32))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[idx]
        rec_id = rec["recording_id"]
        label = rec["label"]
        tensor_img = self.cached_tensors[rec_id]
        tensor_scalars = self.scalar_matrix[idx]
        return tensor_img, tensor_scalars, label, rec_id


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
    if device_pref == "mps" or (device_pref == "auto" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        try:
            t = torch.ones(2, 2, device="mps")
            _ = t + t
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    elif device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def compute_balanced_class_weights(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    arr = np.array(labels)
    total = len(arr)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        cnt = int(np.sum(arr == c))
        weights[c] = total / (num_classes * cnt) if cnt > 0 else 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_hybrid_single_fold_training(
    fold_idx: int,
    subtrain_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int,
    num_scalars: int,
    class_weights: torch.Tensor,
    device: torch.device,
    max_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    fold_seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any], int, float, Dict[str, List[float]]]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    model = BonnHybridRPCNN(num_classes=num_classes, num_scalars=num_scalars, dropout_rate=0.4).to(device)

    train_loader = DataLoader(
        subtrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for b_img, b_scal, b_y, _ in train_loader:
            b_img = b_img.to(device)
            b_scal = b_scal.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            logits = model(b_img, b_scal)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.item()) * len(b_y)
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == b_y).sum().item())
            train_total += len(b_y)

        ep_train_loss = train_loss_accum / max(1, train_total)
        ep_train_acc = train_correct / max(1, train_total)

        # Val
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for b_img, b_scal, b_y, _ in val_loader:
                b_img = b_img.to(device)
                b_scal = b_scal.to(device)
                b_y = b_y.to(device)

                logits = model(b_img, b_scal)
                loss = criterion(logits, b_y)

                val_loss_accum += float(loss.item()) * len(b_y)
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == b_y).sum().item())
                val_total += len(b_y)

        ep_val_loss = val_loss_accum / max(1, val_total)
        ep_val_acc = val_correct / max(1, val_total)

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(ep_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(ep_train_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)
        history["lr"].append(cur_lr)

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
            logger.info(f"Fold {fold_idx}: Early stopped at epoch {epoch}. Best epoch: {best_epoch} (val_acc={best_val_acc:.4f})")
            break

    model.load_state_dict(best_weights)
    meta = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["epoch"]),
        "early_stopped": epochs_no_improve >= early_stopping_patience,
    }
    return model, meta, best_epoch, best_val_acc, history


def evaluate_hybrid_on_test(
    model: nn.Module,
    test_dataset: Dataset,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
) -> Dict[str, Any]:
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_targets, all_sample_ids, all_probs = [], [], [], []

    with torch.no_grad():
        for b_img, b_scal, b_y, b_ids in test_loader:
            b_img = b_img.to(device)
            b_scal = b_scal.to(device)
            logits = model(b_img, b_scal)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(b_y.numpy().tolist())
            all_sample_ids.extend(list(b_ids))
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

    eval_results = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "sample_ids": all_sample_ids,
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred],
        "y_probs": probs_arr.tolist(),
    }

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


def run_hybrid_rp_benchmark(
    case_key: str,
    manifest_records: List[Dict[str, Any]],
    spectral_dict: Dict[str, Dict[str, float]],
    images_dir: Path,
    output_case_dir: Path,
    device: torch.device,
    max_epochs: int = 30,
    learning_rate: float = 0.0005,
    n_splits: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    spec = CASE_SPECS[case_key]
    case_name = spec["name"]
    num_classes = spec["num_classes"]
    class_names = spec["class_names"]
    subsets_inc = spec["subsets_included"]
    sub_to_lbl = spec["subset_to_label"]

    logger.info(f"\n=======================================================")
    logger.info(f"Starting HYBRID {case_name}")
    logger.info(f"Target: {num_classes} classes, subsets: {subsets_inc}")
    logger.info(f"=======================================================")

    output_case_dir.mkdir(parents=True, exist_ok=True)

    # Filter records
    case_records = []
    scalar_list = []
    for r in manifest_records:
        sub = r["subset"]
        if sub in subsets_inc:
            item = dict(r)
            item["label"] = sub_to_lbl[sub]
            case_records.append(item)

            rec_id = r["recording_id"]
            sp = spectral_dict[rec_id]
            # 7 physiological scalars
            sc_vec = [
                float(r["tau"]),
                float(r["m"]),
                float(sp["raw_std"]),
                float(sp["alpha_ratio"]),
                float(sp["theta_ratio"]),
                float(sp["delta_ratio"]),
                float(sp["gamma_ratio"]),
            ]
            scalar_list.append(sc_vec)

    raw_scalars_all = np.array(scalar_list, dtype=np.float32)

    case_records.sort(key=lambda x: x["recording_id"])
    rec_ids = [r["recording_id"] for r in case_records]
    labels = np.array([r["label"] for r in case_records], dtype=np.int64)

    # In-memory image caching
    logger.info(f"Caching {len(case_records)} RP images in memory...")
    cached_tensors = {}
    for r in case_records:
        rec_id = r["recording_id"]
        npy_path = images_dir / "npy" / f"{rec_id}.npy"
        if npy_path.exists():
            arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
        else:
            from PIL import Image
            png_path = images_dir / "images" / f"{rec_id}.png"
            pil_img = Image.open(png_path).convert("L")
            arr = np.array(pil_img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
        cached_tensors[rec_id] = torch.from_numpy(arr)

    # Stratified 5-fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    fold_splits = list(skf.split(rec_ids, labels))

    fold_results_list = []
    pooled_y_true, pooled_y_pred, pooled_ids = [], [], []

    for fold_idx, (dev_pos, test_pos) in enumerate(fold_splits):
        fold_seed = base_seed + fold_idx * 100
        fold_dir = output_case_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_records = [case_records[i] for i in test_pos]
        dev_records = [case_records[i] for i in dev_pos]

        dev_labels = np.array([r["label"] for r in dev_records])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=fold_seed)
        subtrain_pos, val_pos = next(sss.split(dev_records, dev_labels))

        subtrain_records = [dev_records[i] for i in subtrain_pos]
        val_records = [dev_records[i] for i in val_pos]

        # Scalar feature normalization strictly fitted on sub-training fold
        subtrain_indices_orig = [dev_pos[i] for i in subtrain_pos]
        val_indices_orig = [dev_pos[i] for i in val_pos]
        test_indices_orig = list(test_pos)

        scaler = StandardScaler()
        scaler.fit(raw_scalars_all[subtrain_indices_orig])

        scaled_subtrain = scaler.transform(raw_scalars_all[subtrain_indices_orig])
        scaled_val = scaler.transform(raw_scalars_all[val_indices_orig])
        scaled_test = scaler.transform(raw_scalars_all[test_indices_orig])

        subtrain_ds = HybridRPDataset(subtrain_records, cached_tensors, scaled_subtrain)
        val_ds = HybridRPDataset(val_records, cached_tensors, scaled_val)
        test_ds = HybridRPDataset(test_records, cached_tensors, scaled_test)

        subtrain_lbls = [r["label"] for r in subtrain_records]
        class_weights = compute_balanced_class_weights(subtrain_lbls, num_classes=num_classes, device=device)

        logger.info(f"--- Fold {fold_idx} --- (Test: {len(test_records)}, Subtrain: {len(subtrain_records)}, Val: {len(val_records)})")
        t_fold_start = time.time()

        trained_model, train_meta, best_ep, best_vacc, history = run_hybrid_single_fold_training(
            fold_idx=fold_idx,
            subtrain_dataset=subtrain_ds,
            val_dataset=val_ds,
            num_classes=num_classes,
            num_scalars=raw_scalars_all.shape[1],
            class_weights=class_weights,
            device=device,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            early_stopping_patience=7,
            fold_seed=fold_seed,
        )

        fold_train_sec = time.time() - t_fold_start

        torch.save(trained_model.state_dict(), fold_dir / "best_checkpoint.pt")
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

        test_eval = evaluate_hybrid_on_test(
            model=trained_model,
            test_dataset=test_ds,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
        )

        # Save predictions
        pred_rows = []
        for sid, yt, yp in zip(test_eval["sample_ids"], test_eval["y_true"], test_eval["y_pred"]):
            pred_rows.append({"recording_id": sid, "true_label": yt, "predicted_label": yp, "is_correct": int(yt == yp)})
        pd.DataFrame(pred_rows).to_csv(fold_dir / "test_predictions.csv", index=False)

        fold_metrics = {
            "fold_idx": fold_idx,
            "test_accuracy": test_eval["accuracy"],
            "balanced_accuracy": test_eval["balanced_accuracy"],
            "macro_precision": test_eval["macro_precision"],
            "macro_recall": test_eval["macro_recall"],
            "macro_f1": test_eval["macro_f1"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "best_epoch": best_ep,
            "best_val_accuracy": best_vacc,
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
        pooled_ids.extend(test_eval["sample_ids"])
        pooled_y_true.extend(test_eval["y_true"])
        pooled_y_pred.extend(test_eval["y_pred"])

        logger.info(
            f"Fold {fold_idx} Result: Test Acc={test_eval['accuracy']*100:.2f}%, "
            f"Bal Acc={test_eval['balanced_accuracy']*100:.2f}%, "
            f"Macro F1={test_eval['macro_f1']*100:.2f}%, Best Epoch={best_ep}/{train_meta['epochs_trained']} in {fold_train_sec:.2f}s"
        )
        if num_classes == 2:
            logger.info(f"  -> Sensitivity={test_eval['sensitivity']*100:.2f}%, Specificity={test_eval['specificity']*100:.2f}% (FP={test_eval['false_positives']}, FN={test_eval['false_negatives']})")

    p_true = np.array(pooled_y_true)
    p_pred = np.array(pooled_y_pred)

    pooled_acc = float(accuracy_score(p_true, p_pred))
    pooled_bal_acc = float(balanced_accuracy_score(p_true, p_pred))
    pooled_macro_p = float(precision_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_macro_r = float(recall_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_macro_f1 = float(f1_score(p_true, p_pred, average="macro", zero_division=0))
    pooled_cm = confusion_matrix(p_true, p_pred, labels=list(range(num_classes)))

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

    logger.info(f"\n=== HYBRID {case_name} Complete ===")
    logger.info(f"Mean Accuracy: {np.mean(accs)*100:.2f}% +- {np.std(accs)*100:.2f}% | Pooled OOF: {pooled_acc*100:.2f}%")
    logger.info(f"Mean Macro-F1: {np.mean(f1s)*100:.2f}% +- {np.std(f1s)*100:.2f}% | Pooled OOF: {pooled_macro_f1*100:.2f}%")
    if num_classes == 2:
        logger.info(f"Mean Sens: {np.mean(sens_list)*100:.2f}%, Spec: {np.mean(specs_list)*100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold CV on Hybrid RP-CNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_dir = PROJECT_ROOT / "experiments" / f"results_cnn_hybrid_rp_5fold_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(exp_dir / "experiment.log")

    logger.info("Initializing Bonn EEG HYBRID RP-CNN 5-Fold Benchmark...")
    logger.info(f"Experiment Directory: {exp_dir}")

    manifest_path = PROJECT_ROOT / "data" / "processed" / "rp_images_128" / "image_manifest.json"
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    records = manifest_data["records"]

    raw_dir = PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"
    spectral_dict = compute_spectral_scalars(raw_dir)

    device = select_device(args.device)
    logger.info(f"Compute Device: {device}")

    all_summaries = {}
    for c_key in ["Case1_3Class", "Case2_Binary", "Case3_Binary"]:
        c_dir = exp_dir / c_key
        summary = run_hybrid_rp_benchmark(
            case_key=c_key,
            manifest_records=records,
            spectral_dict=spectral_dict,
            images_dir=PROJECT_ROOT / "data" / "processed" / "rp_images_128",
            output_case_dir=c_dir,
            device=device,
            max_epochs=args.epochs,
            learning_rate=args.lr,
        )
        all_summaries[c_key] = summary

    with open(exp_dir / "consolidated_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info("\n=======================================================")
    logger.info("ALL HYBRID BENCHMARKS COMPLETED!")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("=======================================================\n")


if __name__ == "__main__":
    main()
