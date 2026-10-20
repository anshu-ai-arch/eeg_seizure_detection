"""
scripts/transformer/run_sequential_cbm_test.py

Sequential Concept Bottleneck Model (S-CBM) 5-Fold Evaluation on Case 2 (Binary Seizure Detection).

Defeats the "Concept Leakage via Backpropagation" critique (Reviewer #2 / #3):
- Stage 1: Trains Transformer backbone + Concept Bottleneck EXCLUSIVELY on Concept Loss (L_task = 0).
           Zero seizure label gradients ever reach the backbone.
- Stage 2: FREEZES the entire Transformer backbone and concept projection layers.
           Trains strictly the linear decision head (concept_to_class) on seizure labels.
- Evaluation: Tests on unseen test folds to prove whether independent concepts yield high accuracy.
"""

import sys
import os
import time
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transformer.cbm_model import BonnConceptGatedTransformer, CONCEPT_NAMES
from scripts.transformer.cbm_dataset import (
    RobustCBMDataset,
    extract_raw_concept_signals,
    fit_and_scale_concepts,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("sequential_cbm")


def run_sequential_cbm_fold(
    fold_idx: int,
    subtrain_ds: RobustCBMDataset,
    val_ds: RobustCBMDataset,
    test_ds: RobustCBMDataset,
    num_cheatsheet: int,
    residual_indices: List[int],
    device: torch.device,
    stage1_epochs: int = 25,
    batch_size: int = 16,
    fold_seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    train_loader = DataLoader(subtrain_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(fold_seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BonnConceptGatedTransformer(
        num_classes=2,
        num_cheatsheet=num_cheatsheet,
        num_concepts=3,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.2,
        residual_indices=residual_indices,
    ).to(device)

    crit_bce = nn.BCELoss()

    # =========================================================================
    # STAGE 1: PURE CONCEPT LEARNING (L_task = 0, Seizure labels COMPLETELY WITHHELD)
    # =========================================================================
    logger.info(f"--- Fold {fold_idx}: STAGE 1 (Pure Concept Pre-training, Zero Task Loss) ---")
    opt_stage1 = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched_stage1 = ReduceLROnPlateau(opt_stage1, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_closs = float("inf")
    best_stage1_weights = None

    for ep in range(1, stage1_epochs + 1):
        model.train()
        train_closs_accum = 0.0
        n_samples = 0
        for b_glcm, b_cheat, _, b_c, _ in train_loader:
            b_glcm, b_cheat, b_c = b_glcm.to(device), b_cheat.to(device), b_c.to(device)

            opt_stage1.zero_grad()
            _, pred_c = model(b_glcm, b_cheat, return_concepts=True)
            c_loss = crit_bce(pred_c, b_c)
            c_loss.backward()
            opt_stage1.step()

            train_closs_accum += float(c_loss.item()) * len(b_c)
            n_samples += len(b_c)

        train_closs = train_closs_accum / n_samples

        # Validation on concept loss
        model.eval()
        val_closs_accum = 0.0
        n_val = 0
        with torch.no_grad():
            for b_glcm, b_cheat, _, b_c, _ in val_loader:
                b_glcm, b_cheat, b_c = b_glcm.to(device), b_cheat.to(device), b_c.to(device)
                _, pred_c = model(b_glcm, b_cheat, return_concepts=True)
                c_loss = crit_bce(pred_c, b_c)
                val_closs_accum += float(c_loss.item()) * len(b_c)
                n_val += len(b_c)
        val_closs = val_closs_accum / n_val
        sched_stage1.step(val_closs)

        if val_closs < best_val_closs:
            best_val_closs = val_closs
            best_stage1_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_stage1_weights)
    logger.info(f"Fold {fold_idx}: Stage 1 Best Val Concept Loss = {best_val_closs:.4f}")

    # =========================================================================
    # STAGE 2: STRICT FREEZING + EXTRACT PREDICTED CONCEPTS
    # =========================================================================
    logger.info(f"--- Fold {fold_idx}: STAGE 2 (Strictly Freeze Backbone & Train Linear Decision Head) ---")
    
    # 1. Freeze entire model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    def get_features_and_labels(loader):
        all_pred_c, all_true_c, all_y = [], [], []
        with torch.no_grad():
            for b_glcm, b_cheat, b_y, b_c, _ in loader:
                b_glcm, b_cheat = b_glcm.to(device), b_cheat.to(device)
                _, pred_c = model(b_glcm, b_cheat, return_concepts=True)
                all_pred_c.extend(pred_c.cpu().numpy())
                all_true_c.extend(b_c.numpy())
                all_y.extend(b_y.numpy())
        return np.array(all_pred_c), np.array(all_true_c), np.array(all_y)

    train_c_preds, train_c_true, train_y = get_features_and_labels(train_loader)
    val_c_preds, val_c_true, val_y = get_features_and_labels(val_loader)
    test_c_preds, test_c_true, test_y = get_features_and_labels(test_loader)

    # Concept Alignment Metrics on Unseen Test Fold
    concept_maes = np.mean(np.abs(test_c_preds - test_c_true), axis=0).tolist()
    concept_rs = [float(pearsonr(test_c_preds[:, k], test_c_true[:, k])[0]) for k in range(3)]
    concept_rhos = [float(spearmanr(test_c_preds[:, k], test_c_true[:, k])[0]) for k in range(3)]

    # 2. Train Pure Linear Decision Head (Logistic Regression with class balancing)
    clf = LogisticRegression(class_weight="balanced", C=1.0, solver="lbfgs", random_state=fold_seed)
    # Combine subtrain + val predicted concepts to train linear head, exactly as standard CV protocol
    dev_c_preds = np.vstack([train_c_preds, val_c_preds])
    dev_y = np.concatenate([train_y, val_y])
    clf.fit(dev_c_preds, dev_y)

    test_preds = clf.predict(test_c_preds)
    test_probs = clf.predict_proba(test_c_preds)[:, 1]

    acc = accuracy_score(test_y, test_preds)
    sens = recall_score(test_y, test_preds, pos_label=1)
    spec = recall_score(test_y, test_preds, pos_label=0)
    f1 = f1_score(test_y, test_preds, average="macro")
    cm = confusion_matrix(test_y, test_preds)

    logger.info(
        f"Fold {fold_idx} S-CBM RESULT -> Acc: {acc*100:.2f}%, Sens: {sens*100:.2f}%, Spec: {spec*100:.2f}%, F1: {f1*100:.2f}% | "
        f"Concept r: {[round(x, 3) for x in concept_rs]}, MAE: {[round(x, 3) for x in concept_maes]} | "
        f"Linear Weights: {[round(x, 3) for x in clf.coef_[0]]}, Intercept: {round(clf.intercept_[0], 3)}"
    )

    return {
        "fold": fold_idx,
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "f1": f1,
        "cm": cm.tolist(),
        "weights": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "concept_rs": concept_rs,
        "concept_rhos": concept_rhos,
        "concept_maes": concept_maes,
        "test_y": test_y.tolist(),
        "test_preds": test_preds.tolist(),
        "test_probs": test_probs.tolist(),
        "test_c_preds": test_c_preds.tolist(),
        "test_c_true": test_c_true.tolist(),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Launching Sequential CBM (S-CBM) 5-Fold Verification on device: {device}")

    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "robust_glcm"
    with open(DATA_DIR / "dataset_manifest.json") as f:
        manifest = json.load(f)["records"]

    df_cheat = pd.read_csv(DATA_DIR / "cheatsheet_features.csv")
    feature_cols = [c for c in df_cheat.columns if c not in ("recording_id", "subset")]
    res_names = ["global_log_raw_std", "global_log_energy", "global_log_line_length"]
    res_indices = [feature_cols.index(c) for c in res_names]

    rec_to_cheat_row = {r["recording_id"]: r for _, r in df_cheat.iterrows()}
    case_records, cheatsheet_vectors = [], []
    for r in manifest:
        item = dict(r)
        item["label"] = 1 if r["subset"] == "E" else 0
        case_records.append(item)
        rec_id = r["recording_id"]
        c_row = rec_to_cheat_row[rec_id]
        cheatsheet_vectors.append([float(c_row[col]) for col in feature_cols])

    raw_cheatsheet_all = np.array(cheatsheet_vectors, dtype=np.float32)
    case_records.sort(key=lambda x: x["recording_id"])
    rec_ids = [r["recording_id"] for r in case_records]
    labels = np.array([r["label"] for r in case_records], dtype=np.int64)

    case_df = df_cheat[df_cheat["recording_id"].isin(rec_ids)].sort_values("recording_id")
    raw_concepts_all = extract_raw_concept_signals(case_df)

    cached_tensors = {r: torch.from_numpy(np.load(DATA_DIR / "tensors" / f"{r}.npy").astype(np.float32)) for r in rec_ids}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(rec_ids, labels))

    fold_results = []
    pooled_y_true, pooled_y_pred = [], []

    for fold_idx, (dev_pos, test_pos) in enumerate(fold_splits):
        fold_seed = 42 + fold_idx * 100
        dev_records = [case_records[i] for i in dev_pos]
        test_records = [case_records[i] for i in test_pos]
        dev_labels = np.array([r["label"] for r in dev_records])

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=fold_seed)
        subtrain_pos, val_pos = next(sss.split(dev_records, dev_labels))
        subtrain_indices_orig = [dev_pos[i] for i in subtrain_pos]
        val_indices_orig = [dev_pos[i] for i in val_pos]
        test_indices_orig = list(test_pos)

        scaler = StandardScaler()
        scaler.fit(raw_cheatsheet_all[subtrain_indices_orig])
        scaled_subtrain = scaler.transform(raw_cheatsheet_all[subtrain_indices_orig])
        scaled_val = scaler.transform(raw_cheatsheet_all[val_indices_orig])
        scaled_test = scaler.transform(raw_cheatsheet_all[test_indices_orig])

        subtrain_raw_concepts = (raw_concepts_all[0][subtrain_indices_orig], raw_concepts_all[1][subtrain_indices_orig], raw_concepts_all[2][subtrain_indices_orig])
        val_raw_concepts = (raw_concepts_all[0][val_indices_orig], raw_concepts_all[1][val_indices_orig], raw_concepts_all[2][val_indices_orig])
        test_raw_concepts = (raw_concepts_all[0][test_indices_orig], raw_concepts_all[1][test_indices_orig], raw_concepts_all[2][test_indices_orig])

        c_subtrain, (c_val, c_test) = fit_and_scale_concepts(subtrain_raw_concepts, [val_raw_concepts, test_raw_concepts])

        subtrain_ds = RobustCBMDataset([dev_records[i] for i in subtrain_pos], cached_tensors, scaled_subtrain, c_subtrain)
        val_ds = RobustCBMDataset([dev_records[i] for i in val_pos], cached_tensors, scaled_val, c_val)
        test_ds = RobustCBMDataset(test_records, cached_tensors, scaled_test, c_test)

        res = run_sequential_cbm_fold(
            fold_idx=fold_idx,
            subtrain_ds=subtrain_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            num_cheatsheet=raw_cheatsheet_all.shape[1],
            residual_indices=res_indices,
            device=device,
            stage1_epochs=25,
            batch_size=16,
            fold_seed=fold_seed,
        )
        fold_results.append(res)
        pooled_y_true.extend(res["test_y"])
        pooled_y_pred.extend(res["test_preds"])

    print("\n" + "=" * 88)
    print("FINAL SUMMARY: PURE SEQUENTIAL CONCEPT BOTTLENECK (S-CBM) 5-FOLD BENCHMARK")
    print("=" * 88)
    accs = [r["acc"] for r in fold_results]
    sens = [r["sens"] for r in fold_results]
    specs = [r["spec"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    weights_all = np.array([r["weights"] for r in fold_results])

    print(f"S-CBM MEAN ACCURACY:    {np.mean(accs)*100:.2f}% +/- {np.std(accs, ddof=1)*100:.2f}%")
    print(f"S-CBM MEAN SENSITIVITY: {np.mean(sens)*100:.2f}% +/- {np.std(sens, ddof=1)*100:.2f}%")
    print(f"S-CBM MEAN SPECIFICITY: {np.mean(specs)*100:.2f}% +/- {np.std(specs, ddof=1)*100:.2f}%")
    print(f"S-CBM MEAN F1-SCORE:    {np.mean(f1s)*100:.2f}% +/- {np.std(f1s, ddof=1)*100:.2f}%")
    
    print("\nFold Breakdown:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']*100:.2f}%, Sens={r['sens']*100:.2f}%, Spec={r['spec']*100:.2f}%, "
              f"Weights: C1={r['weights'][0]:.2f}, C2={r['weights'][1]:.2f}, C3={r['weights'][2]:.2f}")

    pooled_acc = accuracy_score(pooled_y_true, pooled_y_pred)
    pooled_sens = recall_score(pooled_y_true, pooled_y_pred, pos_label=1)
    pooled_spec = recall_score(pooled_y_true, pooled_y_pred, pos_label=0)
    pooled_cm = confusion_matrix(pooled_y_true, pooled_y_pred)
    
    print(f"\nPOOLED ACCURACY:        {pooled_acc*100:.2f}%")
    print(f"POOLED SENSITIVITY:     {pooled_sens*100:.2f}% ({pooled_cm[1,1]}/{pooled_cm[1,0]+pooled_cm[1,1]} Seizures Detected)")
    print(f"POOLED SPECIFICITY:     {pooled_spec*100:.2f}% ({pooled_cm[0,0]}/{pooled_cm[0,0]+pooled_cm[0,1]} Non-Seizures Detected)")
    print(f"\nPOOLED CONFUSION MATRIX:\n{pooled_cm}")
    print(f"MEAN LINEAR WEIGHTS:    C1={np.mean(weights_all[:,0]):.3f}, C2={np.mean(weights_all[:,1]):.3f}, C3={np.mean(weights_all[:,2]):.3f}")
    print("=" * 88)

    # Save results to json for paper/documentation
    summary_path = PROJECT_ROOT / "docs" / "sequential_cbm_5fold_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs, ddof=1)),
                "mean_sens": float(np.mean(sens)),
                "std_sens": float(np.std(sens, ddof=1)),
                "mean_spec": float(np.mean(specs)),
                "std_spec": float(np.std(specs, ddof=1)),
                "pooled_acc": float(pooled_acc),
                "pooled_sens": float(pooled_sens),
                "pooled_spec": float(pooled_spec),
                "pooled_cm": pooled_cm.tolist(),
                "mean_weights": np.mean(weights_all, axis=0).tolist(),
                "folds": fold_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved complete S-CBM results to {summary_path}")


if __name__ == "__main__":
    main()
