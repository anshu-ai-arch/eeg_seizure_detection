"""
scripts/transformer/cbm_dataset.py

PyTorch Dataset and Concept Targets Generator for Concept-Bottlenecked Transformer.
Strictly ensures zero data leakage by computing concept scaling statistics on training folds only.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RobustCBMDataset(Dataset):
    """
    Dataset returning:
    - glcm_tensor: [4, 64, 64] float32 tensor
    - cheatsheet_vector: [num_cheatsheet] float32 vector
    - label: int class index
    - concept_targets: [3] float32 target concept vector in [0, 1]
    - recording_id: str identifier
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cached_tensors: Dict[str, torch.Tensor],
        cheatsheet_matrix: np.ndarray,
        concept_targets: np.ndarray,
    ):
        self.records = records
        self.cached_tensors = cached_tensors
        self.cheatsheet_matrix = torch.from_numpy(cheatsheet_matrix.astype(np.float32))
        self.concept_targets = torch.from_numpy(concept_targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, str]:
        rec = self.records[idx]
        rec_id = rec["recording_id"]
        label = rec["label"]
        tensor_glcm = self.cached_tensors[rec_id]
        tensor_cheat = self.cheatsheet_matrix[idx]
        tensor_concept = self.concept_targets[idx]
        return tensor_glcm, tensor_cheat, label, tensor_concept, rec_id


def extract_raw_concept_signals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the raw electrophysiological quantities corresponding to the 3 clinical concepts:
    1. Paroxysmal Spike Index: global_log_line_length
    2. Hypersynchronous Rhythmic Power Surge: global_log_energy + 3.0 * global_theta_ratio
    3. Phase-Space Recurrence Disruption: log1p(mode0_contrast)
    """
    raw_c1 = df["global_log_line_length"].values.astype(np.float64)
    raw_c2 = df["global_log_energy"].values.astype(np.float64)
    raw_c3 = np.log1p(df["mode0_contrast"].values).astype(np.float64)
    return raw_c1, raw_c2, raw_c3


def fit_and_scale_concepts(
    subtrain_raw: Tuple[np.ndarray, np.ndarray, np.ndarray],
    other_raw_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Fits percentile bounds [p1, p99] strictly on sub-training fold, then scales all arrays into [0, 1].
    Guarantees absolute leakage-free concept targets.
    """
    bounds = []
    scaled_subtrain_cols = []

    for i in range(3):
        col_train = subtrain_raw[i]
        p1 = float(np.percentile(col_train, 1))
        p99 = float(np.percentile(col_train, 99))
        denom = p99 - p1 if (p99 - p1) > 1e-8 else 1.0
        bounds.append((p1, denom))

        col_scaled = np.clip((col_train - p1) / denom, 0.0, 1.0)
        scaled_subtrain_cols.append(col_scaled)

    scaled_subtrain = np.stack(scaled_subtrain_cols, axis=1)

    scaled_others = []
    for other_raw in other_raw_list:
        other_cols = []
        for i in range(3):
            p1, denom = bounds[i]
            col = other_raw[i]
            col_scaled = np.clip((col - p1) / denom, 0.0, 1.0)
            other_cols.append(col_scaled)
        scaled_others.append(np.stack(other_cols, axis=1))

    return scaled_subtrain, scaled_others
