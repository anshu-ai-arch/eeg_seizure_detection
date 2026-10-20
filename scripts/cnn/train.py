"""
scripts/cnn/train.py

Training engine for the Bonn EEG CNN branch.
Implements:
- Balanced class weight computation from sub-training fold only.
- AdamW optimizer (lr=0.005, weight_decay=0.0001).
- ReduceLROnPlateau scheduler (factor=0.5, patience=4, min_lr=1e-6).
- Early stopping (patience=5, monitor='val_accuracy', strict improvement, best model restore).
- Outer fold evaluation and artifact logging for GLCM multi-channel inputs.
"""

import copy
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from scripts.cnn.config import GLCMExperimentConfig, CNNTrainingConfig
from scripts.cnn.model import BonnGLCMCNN, BonnRecurrenceCNN
from scripts.cnn.dataset import BonnGLCMDataset
from scripts.cnn.evaluate import evaluate_model_on_loader, compute_classification_metrics


logger = logging.getLogger("cnn_train")


def compute_balanced_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute balanced class weights exclusively from training labels:
    w_c = N_total / (C * N_c)
    """
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        count = int(np.sum(labels == c))
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
        else:
            weights[c] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def split_outer_train_into_subtrain_and_val(
    outer_train_indices: List[int],
    dataset_labels: np.ndarray,
    val_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Deterministically split outer training fold into sub-training and validation subsets.
    Uses StratifiedShuffleSplit to ensure class distribution is preserved.
    """
    sub_labels = dataset_labels[outer_train_indices]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed)
    
    train_sub_pos, val_sub_pos = next(sss.split(outer_train_indices, sub_labels))
    sub_train_indices = [outer_train_indices[i] for i in train_sub_pos]
    val_indices = [outer_train_indices[i] for i in val_sub_pos]

    # Verify disjointness
    assert len(set(sub_train_indices).intersection(set(val_indices))) == 0, (
        "Sub-training and validation partitions overlap!"
    )
    return sub_train_indices, val_indices


def train_single_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    training_cfg: CNNTrainingConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a single fold with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_cfg.lr_scheduler_factor,
        patience=training_cfg.lr_scheduler_patience,
        min_lr=training_cfg.min_lr,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    current_lr = training_cfg.learning_rate

    for epoch in range(1, training_cfg.max_epochs + 1):
        # --- Training Phase ---
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
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.item()) * len(batch_targets)
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == batch_targets).sum().item())
            train_total += len(batch_targets)

        epoch_train_loss = train_loss_accum / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # --- Validation Phase ---
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

        # Record metrics
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            break

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        # Update Scheduler (based on val_accuracy)
        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            logger.info(f"Epoch {epoch}: LR reduced from {current_lr:.6f} to {new_lr:.6f}")
            current_lr = new_lr

        # Early Stopping check (strictly greater than best_val_acc, ties do not count)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= training_cfg.early_stopping_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best validation accuracy was {best_val_acc:.4f} at epoch {best_epoch}."
            )
            break

    # Restore best checkpoint
    model.load_state_dict(best_weights)

    summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "total_epochs_trained": len(history["train_loss"]),
        "stopped_early": epochs_no_improve >= training_cfg.early_stopping_patience,
        "history": history,
    }
    return model, summary


def run_nested_cv_glcm_case(
    dataset: BonnGLCMDataset,
    case_name: str,
    output_dir: Path,
    config: Optional[GLCMExperimentConfig] = None,
    pos_label: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run 10-fold nested CV for a specific classification case using multi-channel GLCMs.
    Reconstructs exact outer folds matching the baseline SVM protocol.
    """
    if config is None:
        config = GLCMExperimentConfig()

    output_dir = Path(output_dir)
    case_dir = output_dir / case_name
    chk_dir = case_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.training.device in ["auto", "cuda"]
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and config.training.device in ["auto", "mps"]
        else "cpu"
    )
    logger.info(f"Running {case_name} on device: {device}")

    all_labels = dataset.get_labels()
    all_sample_ids = dataset.get_sample_ids()
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    # Outer 10-fold split: StratifiedKFold seed 42
    skf = StratifiedKFold(
        n_splits=config.n_outer_splits,
        shuffle=True,
        random_state=config.training.random_seed,
    )

    outer_predictions: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    fold_summaries: List[Dict[str, Any]] = []

    start_time_all = time.perf_counter()

    for fold_idx, (out_train_idx, out_test_idx) in enumerate(skf.split(indices, all_labels)):
        fold_start = time.perf_counter()
        logger.info(f"--- {case_name} Fold {fold_idx}/9 ---")

        # Inner split: 90% sub-train, 10% validation
        sub_train_idx, val_idx = split_outer_train_into_subtrain_and_val(
            outer_train_indices=out_train_idx.tolist(),
            dataset_labels=all_labels,
            val_ratio=config.validation_split_ratio,
            random_seed=config.training.random_seed,
        )

        # Assert pairwise disjointness
        s_tr, s_va, s_te = set(sub_train_idx), set(val_idx), set(out_test_idx)
        assert len(s_tr.intersection(s_va)) == 0, f"Fold {fold_idx}: sub_train and val overlap!"
        assert len(s_tr.intersection(s_te)) == 0, f"Fold {fold_idx}: sub_train and test overlap!"
        assert len(s_va.intersection(s_te)) == 0, f"Fold {fold_idx}: val and test overlap!"

        # Dataloaders
        train_sub = Subset(dataset, sub_train_idx)
        val_sub = Subset(dataset, val_idx)
        test_sub = Subset(dataset, out_test_idx.tolist())

        train_loader = DataLoader(train_sub, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=config.training.batch_size, shuffle=False)
        test_loader = DataLoader(test_sub, batch_size=config.training.batch_size, shuffle=False)

        # Compute balanced class weights strictly from sub-train labels
        sub_train_labels = all_labels[sub_train_idx]
        class_weights = compute_balanced_class_weights(
            sub_train_labels,
            num_classes=dataset.num_classes,
            device=device,
        )

        # Initialize fresh model per fold
        torch.manual_seed(config.training.random_seed + fold_idx)
        model = BonnGLCMCNN(num_classes=dataset.num_classes, config=config.model).to(device)

        # Train fold
        trained_model, train_summary = train_single_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            training_cfg=config.training,
            device=device,
        )

        # Evaluate on held-out outer test fold
        unique_classes = sorted(list(dataset.label_map.values()))
        eval_metrics, fold_preds = evaluate_model_on_loader(
            model=trained_model,
            loader=test_loader,
            device=device,
            labels=unique_classes,
            pos_label=pos_label,
        )

        fold_time = time.perf_counter() - fold_start
        eval_metrics["outer_fold"] = fold_idx
        eval_metrics["runtime_seconds"] = fold_time
        eval_metrics["best_epoch"] = train_summary["best_epoch"]
        eval_metrics["best_val_accuracy"] = train_summary["best_val_accuracy"]
        fold_metrics.append(eval_metrics)

        # Add sample metadata to fold predictions
        for p in fold_preds:
            p["outer_fold"] = fold_idx
        outer_predictions.extend(fold_preds)

        # Save fold checkpoint state
        chk_file = chk_dir / f"fold_{fold_idx}.json"
        fold_state = {
            "outer_fold_idx": fold_idx,
            "metrics": eval_metrics,
            "training_summary": train_summary,
            "partitions": {
                "sub_train_sample_ids": [all_sample_ids[i] for i in sub_train_idx],
                "val_sample_ids": [all_sample_ids[i] for i in val_idx],
                "outer_test_sample_ids": [all_sample_ids[i] for i in out_test_idx],
            },
        }
        with open(chk_file, "w", encoding="utf-8") as f:
            json.dump(fold_state, f, indent=2)

    total_runtime = time.perf_counter() - start_time_all

    # Aggregate pooled metrics across all out-of-fold predictions
    y_true_all = np.array([p["y_true"] for p in outer_predictions])
    y_pred_all = np.array([p["y_pred"] for p in outer_predictions])
    pooled_metrics = compute_classification_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        labels=unique_classes,
        pos_label=pos_label,
    )

    # Mean and standard deviation across folds
    mean_fold_metrics = {
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
        "mean_macro_precision": float(np.mean([m["macro_precision"] for m in fold_metrics])),
        "std_macro_precision": float(np.std([m["macro_precision"] for m in fold_metrics])),
        "mean_macro_recall": float(np.mean([m["macro_recall"] for m in fold_metrics])),
        "std_macro_recall": float(np.std([m["macro_recall"] for m in fold_metrics])),
        "mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "std_macro_f1": float(np.std([m["macro_f1"] for m in fold_metrics])),
    }

    if pos_label is not None and len(unique_classes) == 2:
        mean_fold_metrics["mean_sensitivity"] = float(np.mean([m["sensitivity"] for m in fold_metrics]))
        mean_fold_metrics["std_sensitivity"] = float(np.std([m["sensitivity"] for m in fold_metrics]))
        mean_fold_metrics["mean_specificity"] = float(np.mean([m["specificity"] for m in fold_metrics]))
        mean_fold_metrics["std_specificity"] = float(np.std([m["specificity"] for m in fold_metrics]))

    final_results = {
        "case_name": case_name,
        "input_representation": "GLCM_matrices [8, 16, 16]",
        "class_mapping": dataset.label_map,
        "pos_label_index": pos_label,
        "n_samples": n_samples,
        "total_runtime_seconds": total_runtime,
        "pooled_metrics": pooled_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "fold_metrics": fold_metrics,
        "outer_predictions": outer_predictions,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    results_file = case_dir / f"{case_name}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Completed {case_name}: mean acc = {mean_fold_metrics['mean_accuracy']:.4f} +/- {mean_fold_metrics['std_accuracy']:.4f}")
    return final_results
