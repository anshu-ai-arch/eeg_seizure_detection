"""
scripts/transformer/dataset.py

PyTorch Dataset for Robust GLCM Tensors and Domain Cheatsheet Vectors.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class RobustGLCMDataset(Dataset):
    """
    Dataset returning (glcm_tensor [4, 64, 64], cheatsheet_vector [69], label, recording_id).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cached_tensors: Dict[str, torch.Tensor],
        cheatsheet_matrix: np.ndarray,
    ):
        self.records = records
        self.cached_tensors = cached_tensors
        self.cheatsheet_matrix = torch.from_numpy(cheatsheet_matrix.astype(np.float32))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[idx]
        rec_id = rec["recording_id"]
        label = rec["label"]
        tensor_glcm = self.cached_tensors[rec_id]
        tensor_cheat = self.cheatsheet_matrix[idx]
        return tensor_glcm, tensor_cheat, label, rec_id
