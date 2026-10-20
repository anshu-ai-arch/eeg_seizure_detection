"""
scripts/transformer/run_cbm_transformer_5fold_cv.py

5-Fold Cross-Validation Evaluation for Concept-Bottlenecked Gated GLCM Transformer on Bonn EEG.

Evaluates:
- Case 2: Binary Seizure Detection (A+B+C+D vs E, N=500)
- Case 1: Standard Three-Class (Healthy A+B vs Interictal C+D vs Ictal E, N=500)
- Case 3: Binary Interictal vs Ictal (D vs E, N=200)

Features:
- Leakage-free percentile scaling for 3 clinical concepts:
  * C1: Paroxysmal Spike Index (PSI)
  * C2: Hypersynchronous Rhythmic Power Surge (HPS)
  * C3: Phase-Space Texture / Recurrence Disruption (PRC)
- Multi-Task Loss: L_total = L_cross_entropy + lambda_concept * L_smooth_l1_concept
- Concept decision weights logging and per-sample clinical attribution export.
"""

import sys
import os
import time
import json
import copy
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transformer.cbm_model import BonnConceptGatedTransformer, CONCEPT_NAMES
from scripts.transformer.cbm_dataset import (
    RobustCBMDataset,
    extract_raw_concept_signals,
    fit_and_scale_concepts,
)

logger = logging.getLogger("cbm_transformer_5fold")

CASE_SPECS = {
    "Case1_3Class": {
        "name": "Case 1: Standard Three-Class (Healthy A+B vs Interictal C+D vs Ictal E)",
        "case_id": "Case1",
        "num_classes": 3,
        "subsets_included": ["A", "B", "C", "D", "E"],
        "subset_to_label": {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2},
        "class_names": ["Healthy (A+B)", "Interictal (C+D)", "Ictal (E)"],
        "expected_total": 500,
        "expected_test_per_fold": 100,
    },
    "Case2_Binary": {
        "name": "Case 2: Binary Seizure Detection (Non-ictal A+B+C+D vs Ictal E)",
        "case_id": "Case2",
        "num_classes": 2,
        "subsets_included": ["A", "B", "C", "D", "E"],
        "subset_to_label": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 1},
        "class_names": ["Non-ictal (A+B+C+D)", "Ictal (E)"],
        "expected_total": 500,
        "expected_test_per_fold": 100,
    },
    "Case3_Binary": {
        "name": "Case 3: Binary Interictal vs Ictal (D vs E, Exclude A, B, C)",
        "case_id": "Case3",
        "num_classes": 2,
        "subsets_included": ["D", "E"],
        "subset_to_label": {"D": 0, "E": 1},
        "class_names": ["Interictal (D)", "Ictal (E)"],
        "expected_total": 200,
        "expected_test_per_fold": 40,
    },
}


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def select_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def compute_balanced_class_weights(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(len(labels))
    weights = total / (float(num_classes) * np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_cbm_single_fold_training(
    fold_idx: int,
    subtrain_dataset: RobustCBMDataset,
    val_dataset: RobustCBMDataset,
    num_classes: int,
    num_cheatsheet: int,
    residual_indices: List[int],
    class_weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    max_epochs: int = 35,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4,
    lambda_concept: float = 0.25,
    early_stopping_patience: int = 8,
    fold_seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any], int, float, Dict[str, List[Any]]]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    model = BonnConceptGatedTransformer(
        num_classes=num_classes,
        num_cheatsheet=num_cheatsheet,
        num_concepts=3,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.2,
        residual_indices=residual_indices,
    ).to(device)

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
    criterion_task = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_concept = nn.BCELoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_task_loss": [],
        "train_concept_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    ictal_class_idx = num_classes - 1

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_task_accum = 0.0
        train_concept_accum = 0.0
        train_correct = 0
        train_total = 0

        for b_glcm, b_cheat, b_y, b_concepts, _ in train_loader:
            b_glcm = b_glcm.to(device)
            b_cheat = b_cheat.to(device)
            b_y = b_y.to(device)
            b_concepts = b_concepts.to(device)

            optimizer.zero_grad()
            logits, pred_concepts = model(b_glcm, b_cheat, return_concepts=True)

            task_loss = criterion_task(logits, b_y)
            concept_loss = criterion_concept(pred_concepts, b_concepts)

            # Physiological Sign Constraint: Ictal weights on clinical concepts MUST be non-negative
            ictal_w = model.concept_to_class.weight[ictal_class_idx, :]
            sign_loss = 3.0 * torch.sum(torch.relu(-ictal_w))

            loss = task_loss + lambda_concept * concept_loss + sign_loss

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.item()) * len(b_y)
            train_task_accum += float(task_loss.item()) * len(b_y)
            train_concept_accum += float(concept_loss.item()) * len(b_y)

            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == b_y).sum().item())
            train_total += len(b_y)

        ep_train_loss = train_loss_accum / max(1, train_total)
        ep_train_acc = train_correct / max(1, train_total)
        ep_train_task = train_task_accum / max(1, train_total)
        ep_train_concept = train_concept_accum / max(1, train_total)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for b_glcm, b_cheat, b_y, b_concepts, _ in val_loader:
                b_glcm = b_glcm.to(device)
                b_cheat = b_cheat.to(device)
                b_y = b_y.to(device)
                b_concepts = b_concepts.to(device)

                logits, pred_concepts = model(b_glcm, b_cheat, return_concepts=True)
                task_loss = criterion_task(logits, b_y)
                concept_loss = criterion_concept(pred_concepts, b_concepts)
                loss = task_loss + lambda_concept * concept_loss

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
        history["train_task_loss"].append(ep_train_task)
        history["train_concept_loss"].append(ep_train_concept)
        history["train_acc"].append(ep_train_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)
        history["lr"].append(cur_lr)

        # Save checkpoint if accuracy improves, or loss improves on tied accuracy
        if (ep_val_acc > best_val_acc) or (abs(ep_val_acc - best_val_acc) < 1e-5 and ep_val_loss < best_val_loss):
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


def evaluate_cbm_on_test(
    model: nn.Module,
    test_dataset: RobustCBMDataset,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
) -> Dict[str, Any]:
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_targets, all_sample_ids, all_probs = [], [], [], []
    all_pred_concepts, all_target_concepts = [], []

    with torch.no_grad():
        for b_glcm, b_cheat, b_y, b_concepts, b_ids in test_loader:
            b_glcm = b_glcm.to(device)
            b_cheat = b_cheat.to(device)
            logits, pred_concepts = model(b_glcm, b_cheat, return_concepts=True)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(b_y.numpy().tolist())
            all_sample_ids.extend(list(b_ids))
            all_probs.extend(probs.cpu().numpy().tolist())
            all_pred_concepts.extend(pred_concepts.cpu().numpy().tolist())
            all_target_concepts.extend(b_concepts.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    pred_c_arr = np.array(all_pred_concepts)
    true_c_arr = np.array(all_target_concepts)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    labels_list = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    # Concept errors & Pearson correlation
    concept_mae = np.mean(np.abs(pred_c_arr - true_c_arr), axis=0).tolist()
    concept_r = []
    for k in range(pred_c_arr.shape[1]):
        if np.std(pred_c_arr[:, k]) > 1e-6 and np.std(true_c_arr[:, k]) > 1e-6:
            r_val = float(np.corrcoef(pred_c_arr[:, k], true_c_arr[:, k])[0, 1])
        else:
            r_val = 0.0
        concept_r.append(round(r_val, 4))

    eval_results = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "concept_mae": concept_mae,
        "concept_r": concept_r,
        "sample_ids": all_sample_ids,
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred],
        "pred_concepts": pred_c_arr.tolist(),
        "true_concepts": true_c_arr.tolist(),
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


def run_cbm_benchmark(
    case_key: str,
    manifest_records: List[Dict[str, Any]],
    cheatsheet_df: pd.DataFrame,
    feature_cols: List[str],
    residual_indices: List[int],
    tensors_dir: Path,
    output_case_dir: Path,
    device: torch.device,
    max_epochs: int = 35,
    learning_rate: float = 0.0005,
    lambda_concept: float = 1.0,
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
    logger.info(f"Starting CONCEPT-BOTTLENECKED TRANSFORMER {case_name}")
    logger.info(f"Target: {num_classes} classes, subsets: {subsets_inc}, lambda_concept: {lambda_concept}")
    logger.info(f"=======================================================")

    output_case_dir.mkdir(parents=True, exist_ok=True)

    rec_to_cheat_row = {r["recording_id"]: r for _, r in cheatsheet_df.iterrows()}

    case_records = []
    cheatsheet_vectors = []
    for r in manifest_records:
        sub = r["subset"]
        if sub in subsets_inc:
            item = dict(r)
            item["label"] = sub_to_lbl[sub]
            case_records.append(item)

            rec_id = r["recording_id"]
            c_row = rec_to_cheat_row[rec_id]
            vec = [float(c_row[col]) for col in feature_cols]
            cheatsheet_vectors.append(vec)

    raw_cheatsheet_all = np.array(cheatsheet_vectors, dtype=np.float32)

    case_records.sort(key=lambda x: x["recording_id"])
    rec_ids = [r["recording_id"] for r in case_records]
    labels = np.array([r["label"] for r in case_records], dtype=np.int64)

    # Build sub-dataframe for concepts
    case_df = cheatsheet_df[cheatsheet_df["recording_id"].isin(rec_ids)].copy()
    case_df.sort_values(by="recording_id", inplace=True)
    raw_c1, raw_c2, raw_c3 = extract_raw_concept_signals(case_df)
    raw_concepts_all = (raw_c1, raw_c2, raw_c3)

    # Cache GLCM tensors in memory
    logger.info(f"Caching {len(case_records)} Robust GLCM tensors [4, 64, 64] in memory...")
    cached_tensors = {}
    for r in case_records:
        rec_id = r["recording_id"]
        t_path = tensors_dir / f"{rec_id}.npy"
        arr = np.load(t_path, allow_pickle=False).astype(np.float32)
        cached_tensors[rec_id] = torch.from_numpy(arr)

    # Stratified 5-Fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    fold_splits = list(skf.split(rec_ids, labels))

    fold_results_list = []
    pooled_y_true, pooled_y_pred, pooled_ids = [], [], []
    pooled_c_pred, pooled_c_true = [], []

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

        # Cheatsheet normalization strictly fitted on sub-training fold
        subtrain_indices_orig = [dev_pos[i] for i in subtrain_pos]
        val_indices_orig = [dev_pos[i] for i in val_pos]
        test_indices_orig = list(test_pos)

        scaler = StandardScaler()
        scaler.fit(raw_cheatsheet_all[subtrain_indices_orig])

        scaled_subtrain = scaler.transform(raw_cheatsheet_all[subtrain_indices_orig])
        scaled_val = scaler.transform(raw_cheatsheet_all[val_indices_orig])
        scaled_test = scaler.transform(raw_cheatsheet_all[test_indices_orig])

        # Concept scaling strictly fitted on sub-training fold
        subtrain_raw_concepts = (
            raw_concepts_all[0][subtrain_indices_orig],
            raw_concepts_all[1][subtrain_indices_orig],
            raw_concepts_all[2][subtrain_indices_orig],
        )
        val_raw_concepts = (
            raw_concepts_all[0][val_indices_orig],
            raw_concepts_all[1][val_indices_orig],
            raw_concepts_all[2][val_indices_orig],
        )
        test_raw_concepts = (
            raw_concepts_all[0][test_indices_orig],
            raw_concepts_all[1][test_indices_orig],
            raw_concepts_all[2][test_indices_orig],
        )

        c_subtrain, (c_val, c_test) = fit_and_scale_concepts(
            subtrain_raw=subtrain_raw_concepts,
            other_raw_list=[val_raw_concepts, test_raw_concepts],
        )

        subtrain_ds = RobustCBMDataset(subtrain_records, cached_tensors, scaled_subtrain, c_subtrain)
        val_ds = RobustCBMDataset(val_records, cached_tensors, scaled_val, c_val)
        test_ds = RobustCBMDataset(test_records, cached_tensors, scaled_test, c_test)

        subtrain_lbls = [r["label"] for r in subtrain_records]
        class_weights = compute_balanced_class_weights(subtrain_lbls, num_classes=num_classes, device=device)

        logger.info(f"--- Fold {fold_idx} --- (Test: {len(test_records)}, Subtrain: {len(subtrain_records)}, Val: {len(val_records)})")
        t_fold_start = time.time()

        trained_model, train_meta, best_ep, best_vacc, history = run_cbm_single_fold_training(
            fold_idx=fold_idx,
            subtrain_dataset=subtrain_ds,
            val_dataset=val_ds,
            num_classes=num_classes,
            num_cheatsheet=raw_cheatsheet_all.shape[1],
            residual_indices=residual_indices,
            class_weights=class_weights,
            device=device,
            batch_size=16,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            lambda_concept=lambda_concept,
            early_stopping_patience=8,
            fold_seed=fold_seed,
        )

        fold_train_sec = time.time() - t_fold_start

        torch.save(trained_model.state_dict(), fold_dir / "best_checkpoint.pt")
        pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

        # Save concept decision weights
        concept_weights = trained_model.get_concept_weights()
        with open(fold_dir / "concept_decision_weights.json", "w", encoding="utf-8") as f:
            json.dump({
                "weights": concept_weights["weights"].tolist(),
                "bias": concept_weights["bias"].tolist(),
                "concept_names": concept_weights["concept_names"],
            }, f, indent=2)

        test_eval = evaluate_cbm_on_test(
            model=trained_model,
            test_dataset=test_ds,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
        )

        # Save predictions with concept scorecards
        pred_rows = []
        for sid, yt, yp, pc, tc in zip(
            test_eval["sample_ids"],
            test_eval["y_true"],
            test_eval["y_pred"],
            test_eval["pred_concepts"],
            test_eval["true_concepts"],
        ):
            pred_rows.append({
                "recording_id": sid,
                "true_label": yt,
                "predicted_label": yp,
                "is_correct": int(yt == yp),
                "C1_spike_pred": round(pc[0], 4),
                "C2_surge_pred": round(pc[1], 4),
                "C3_texture_pred": round(pc[2], 4),
                "C1_spike_true": round(tc[0], 4),
                "C2_surge_true": round(tc[1], 4),
                "C3_texture_true": round(tc[2], 4),
            })
        pd.DataFrame(pred_rows).to_csv(fold_dir / "test_predictions.csv", index=False)

        fold_metrics = {
            "fold": fold_idx,
            "seed": fold_seed,
            "best_epoch": best_ep,
            "best_val_acc": best_vacc,
            "test_accuracy": test_eval["accuracy"],
            "test_balanced_accuracy": test_eval["balanced_accuracy"],
            "test_macro_precision": test_eval["macro_precision"],
            "test_macro_recall": test_eval["macro_recall"],
            "test_macro_f1": test_eval["macro_f1"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "concept_mae": test_eval["concept_mae"],
            "train_time_sec": round(fold_train_sec, 2),
        }

        if num_classes == 2:
            fold_metrics["sensitivity"] = test_eval["sensitivity"]
            fold_metrics["specificity"] = test_eval["specificity"]
            fold_metrics["tp"] = test_eval["true_positives"]
            fold_metrics["tn"] = test_eval["true_negatives"]
            fold_metrics["fp"] = test_eval["false_positives"]
            fold_metrics["fn"] = test_eval["false_negatives"]
            logger.info(
                f"Fold {fold_idx} TEST: Acc={test_eval['accuracy']:.4f}, Sens={test_eval['sensitivity']:.4f}, "
                f"Spec={test_eval['specificity']:.4f}, Macro-F1={test_eval['macro_f1']:.4f}, "
                f"Concept MAE={[round(x, 3) for x in test_eval['concept_mae']]} ({fold_train_sec:.1f}s)"
            )
        else:
            logger.info(
                f"Fold {fold_idx} TEST: Acc={test_eval['accuracy']:.4f}, BalAcc={test_eval['balanced_accuracy']:.4f}, "
                f"Macro-F1={test_eval['macro_f1']:.4f}, Concept MAE={[round(x, 3) for x in test_eval['concept_mae']]} ({fold_train_sec:.1f}s)"
            )

        with open(fold_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=2)

        fold_results_list.append(fold_metrics)
        pooled_y_true.extend(test_eval["y_true"])
        pooled_y_pred.extend(test_eval["y_pred"])
        pooled_ids.extend(test_eval["sample_ids"])
        pooled_c_pred.extend(test_eval["pred_concepts"])
        pooled_c_true.extend(test_eval["true_concepts"])

    # Pooled evaluation
    pooled_acc = float(accuracy_score(pooled_y_true, pooled_y_pred))
    pooled_bal_acc = float(balanced_accuracy_score(pooled_y_true, pooled_y_pred))
    pooled_macro_p = float(precision_score(pooled_y_true, pooled_y_pred, average="macro", zero_division=0))
    pooled_macro_r = float(recall_score(pooled_y_true, pooled_y_pred, average="macro", zero_division=0))
    pooled_macro_f1 = float(f1_score(pooled_y_true, pooled_y_pred, average="macro", zero_division=0))
    pooled_cm = confusion_matrix(pooled_y_true, pooled_y_pred, labels=list(range(num_classes)))

    pooled_c_pred_arr = np.array(pooled_c_pred)
    pooled_c_true_arr = np.array(pooled_c_true)
    pooled_concept_mae = np.mean(np.abs(pooled_c_pred_arr - pooled_c_true_arr), axis=0).tolist()

    acc_list = [f["test_accuracy"] for f in fold_results_list]
    bal_acc_list = [f["test_balanced_accuracy"] for f in fold_results_list]
    f1_list = [f["test_macro_f1"] for f in fold_results_list]

    summary = {
        "case_key": case_key,
        "case_name": case_name,
        "num_classes": num_classes,
        "n_samples": len(case_records),
        "mean_accuracy": float(np.mean(acc_list)),
        "std_accuracy": float(np.std(acc_list, ddof=1)),
        "mean_balanced_accuracy": float(np.mean(bal_acc_list)),
        "std_balanced_accuracy": float(np.std(bal_acc_list, ddof=1)),
        "mean_macro_f1": float(np.mean(f1_list)),
        "std_macro_f1": float(np.std(f1_list, ddof=1)),
        "pooled_accuracy": pooled_acc,
        "pooled_balanced_accuracy": pooled_bal_acc,
        "pooled_macro_precision": pooled_macro_p,
        "pooled_macro_recall": pooled_macro_r,
        "pooled_macro_f1": pooled_macro_f1,
        "pooled_confusion_matrix": pooled_cm.tolist(),
        "pooled_concept_mae": pooled_concept_mae,
        "concept_names": CONCEPT_NAMES,
        "folds": fold_results_list,
    }

    if num_classes == 2:
        sens_list = [f["sensitivity"] for f in fold_results_list]
        spec_list = [f["specificity"] for f in fold_results_list]
        summary["mean_sensitivity"] = float(np.mean(sens_list))
        summary["std_sensitivity"] = float(np.std(sens_list, ddof=1))
        summary["mean_specificity"] = float(np.mean(spec_list))
        summary["std_specificity"] = float(np.std(spec_list, ddof=1))

        tn, fp, fn, tp = int(pooled_cm[0, 0]), int(pooled_cm[0, 1]), int(pooled_cm[1, 0]), int(pooled_cm[1, 1])
        summary["pooled_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        summary["pooled_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        summary["pooled_tp"] = tp
        summary["pooled_tn"] = tn
        summary["pooled_fp"] = fp
        summary["pooled_fn"] = fn

        logger.info(f"\n=======================================================")
        logger.info(
            f"SUMMARY {case_key}: Mean Acc = {summary['mean_accuracy']*100:.2f}% +/- {summary['std_accuracy']*100:.2f}%, "
            f"Mean Sens = {summary['mean_sensitivity']*100:.2f}%, Mean Spec = {summary['mean_specificity']*100:.2f}%, "
            f"Concept MAE = {[round(x, 3) for x in pooled_concept_mae]}"
        )
        logger.info(f"Pooled CM:\n{pooled_cm}")
        logger.info(f"=======================================================")
    else:
        logger.info(f"\n=======================================================")
        logger.info(
            f"SUMMARY {case_key}: Mean Acc = {summary['mean_accuracy']*100:.2f}% +/- {summary['std_accuracy']*100:.2f}%, "
            f"Mean BalAcc = {summary['mean_balanced_accuracy']*100:.2f}%, Mean Macro-F1 = {summary['mean_macro_f1']*100:.2f}%, "
            f"Concept MAE = {[round(x, 3) for x in pooled_concept_mae]}"
        )
        logger.info(f"Pooled CM:\n{pooled_cm}")
        logger.info(f"=======================================================")

    with open(output_case_dir / "case_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 5-fold CV for Concept-Bottlenecked Gated Transformer on Bonn EEG")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--cases", nargs="+", default=["Case2_Binary", "Case1_3Class", "Case3_Binary"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lambda-concept", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "experiments" / f"results_cbm_transformer_5fold_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(out_dir / "benchmark.log")
    device = select_device(args.device)
    logger.info(f"Concept-Bottlenecked Gated Transformer 5-Fold Benchmark launched on device: {device}")
    logger.info(f"Output directory: {out_dir}")

    # Load dataset manifest
    data_dir = PROJECT_ROOT / "data" / "processed" / "robust_glcm"
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    records = manifest_data["records"]
    logger.info(f"Loaded {len(records)} records from manifest.")

    # Load Cheatsheet features table
    cheatsheet_path = data_dir / "cheatsheet_features.csv"
    df_cheat = pd.read_csv(cheatsheet_path)
    feature_cols = [c for c in df_cheat.columns if c not in ("recording_id", "subset")]
    logger.info(f"Loaded Cheatsheet features table: {len(df_cheat)} rows, {len(feature_cols)} feature columns.")

    # Residual indices: global_log_raw_std, global_log_energy, global_log_line_length
    residual_names = ["global_log_raw_std", "global_log_energy", "global_log_line_length"]
    residual_indices = [feature_cols.index(col) for col in residual_names if col in feature_cols]
    logger.info(f"Identified residual indices: {residual_indices} for columns: {[feature_cols[i] for i in residual_indices]}")

    tensors_dir = data_dir / "tensors"

    all_case_summaries = {}
    for case_key in args.cases:
        case_out = out_dir / case_key
        summary = run_cbm_benchmark(
            case_key=case_key,
            manifest_records=records,
            cheatsheet_df=df_cheat,
            feature_cols=feature_cols,
            residual_indices=residual_indices,
            tensors_dir=tensors_dir,
            output_case_dir=case_out,
            device=device,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            lambda_concept=args.lambda_concept,
            n_splits=5,
            base_seed=args.seed,
        )
        all_case_summaries[case_key] = summary

    # Save master summary across all cases
    with open(out_dir / "all_cases_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_case_summaries, f, indent=2)

    logger.info("\n=======================================================")
    logger.info("ALL BENCHMARK CASES COMPLETE SUCCESSFULLY!")
    logger.info(f"Master summary saved to: {out_dir / 'all_cases_summary.json'}")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()
