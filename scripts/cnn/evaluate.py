"""
scripts/cnn/evaluate.py

Evaluation metrics and inference utilities for Bonn EEG CNN models.
Computes accuracy, macro-F1, per-class precision/recall, sensitivity/specificity,
and confusion matrices aligned with the baseline nested-CV format.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    pos_label: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute standard classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels.
    labels : list of int, optional
        Unique class labels to consider in confusion matrix.
    pos_label : int, optional
        Positive class label for binary sensitivity/specificity.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))
    else:
        labels = [int(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    result = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": {
            int(lbl): {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i]),
            }
            for i, lbl in enumerate(labels)
        },
    }

    # Binary specific metrics
    if len(labels) == 2 and pos_label is not None:
        neg_label = [l for l in labels if l != pos_label][0]
        cm_binary = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
        tn, fp, fn, tp = cm_binary.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        result["sensitivity"] = sensitivity
        result["specificity"] = specificity
        result["tp"] = int(tp)
        result["tn"] = int(tn)
        result["fp"] = int(fp)
        result["fn"] = int(fn)

    return result


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: Optional[List[int]] = None,
    pos_label: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation on a DataLoader and return metrics plus sample-level prediction records.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    sample_records = []

    with torch.no_grad():
        for batch_imgs, batch_targets, batch_rec_ids in loader:
            batch_imgs = batch_imgs.to(device)
            logits = model(batch_imgs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            preds_cpu = preds.cpu().numpy()
            targets_cpu = batch_targets.numpy()
            probs_cpu = probs.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_targets.extend(targets_cpu)
            all_probs.extend(probs_cpu)

            for i in range(len(batch_rec_ids)):
                sample_records.append({
                    "sample_id": batch_rec_ids[i],
                    "y_true": int(targets_cpu[i]),
                    "y_pred": int(preds_cpu[i]),
                    "probabilities": [float(p) for p in probs_cpu[i]],
                    "is_correct": bool(targets_cpu[i] == preds_cpu[i]),
                })

    metrics = compute_classification_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        labels=labels,
        pos_label=pos_label,
    )
    return metrics, sample_records
