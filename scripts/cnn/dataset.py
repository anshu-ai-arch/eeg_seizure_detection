"""
scripts/cnn/dataset.py

PyTorch Dataset and DataLoader abstractions for Bonn EEG CNN inputs:
1. BonnGLCMDataset: Primary dataset loading multi-channel GLCM matrices [8, 16, 16].
2. BonnRPDataset: Retained dataset loading 128x128 continuous recurrence plot images [1, 128, 128].
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BonnGLCMDataset(Dataset):
    """
    PyTorch Dataset loading normalized multi-channel GLCM matrices [8, 16, 16].
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        matrix_dir: Path,
        label_key: str = "case1_label",
        label_map: Optional[Dict[str, int]] = None,
    ):
        """
        Parameters
        ----------
        samples : list of dict
            Metadata records containing at minimum 'recording_id' and the label field.
        matrix_dir : Path
            Directory containing '<recording_id>.npy' or relative paths from manifest.
        label_key : str
            Field name for the label ('case1_label', 'case2_label', 'case3_label').
        label_map : dict, optional
            Mapping from string labels to integer classes {lbl: int}.
        """
        self.matrix_dir = Path(matrix_dir)
        self.label_key = label_key

        # Filter valid samples (e.g. exclude NaN in Case 3)
        self.valid_samples = []
        for s in samples:
            val = s.get(label_key)
            if val is not None and str(val).strip() != "" and str(val).lower() != "nan":
                self.valid_samples.append(s)

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid samples found with label key '{label_key}'!")

        # Establish deterministic alphabetical label mapping if not provided
        if label_map is None:
            unique_labels = sorted(list(set(s[self.label_key] for s in self.valid_samples)))
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self.valid_samples[idx]
        rec_id = record["recording_id"]

        # Locate matrix file (.npy)
        npy_path = self.matrix_dir / f"{rec_id}.npy"
        if not npy_path.exists():
            alt_path = self.matrix_dir / "matrices" / f"{rec_id}.npy"
            if alt_path.exists():
                npy_path = alt_path
            else:
                raise FileNotFoundError(f"GLCM matrix not found for recording {rec_id} at {npy_path}")

        # Load matrix of shape (8, 16, 16) float32
        mat_arr = np.load(npy_path).astype(np.float32)
        if mat_arr.shape != (8, 16, 16):
            raise ValueError(f"Expected shape (8, 16, 16), got {mat_arr.shape} in {npy_path}")

        tensor_mat = torch.from_numpy(mat_arr)
        label_str = record[self.label_key]
        label_int = self.label_map[label_str]

        return tensor_mat, label_int, rec_id

    def get_labels(self) -> np.ndarray:
        return np.array([self.label_map[s[self.label_key]] for s in self.valid_samples], dtype=np.int64)

    def get_sample_ids(self) -> List[str]:
        return [s["recording_id"] for s in self.valid_samples]


class BonnRPDataset(Dataset):
    """
    PyTorch Dataset loading continuous 128x128 recurrence plot images [1, 128, 128].
    Retained for backward compatibility.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_dir: Path,
        label_key: str = "case1_label",
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_key = label_key

        self.valid_samples = []
        for s in samples:
            val = s.get(label_key)
            if val is not None and str(val).strip() != "" and str(val).lower() != "nan":
                self.valid_samples.append(s)

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid samples found with label key '{label_key}'!")

        if label_map is None:
            unique_labels = sorted(list(set(s[self.label_key] for s in self.valid_samples)))
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self.valid_samples[idx]
        rec_id = record["recording_id"]

        png_path = self.image_dir / f"{rec_id}.png"
        if not png_path.exists():
            alt_path = self.image_dir / "images" / f"{rec_id}.png"
            if alt_path.exists():
                png_path = alt_path
            else:
                raise FileNotFoundError(f"RP image not found for recording {rec_id} at {png_path}")

        pil_img = Image.open(png_path).convert("L")
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        tensor_img = torch.from_numpy(img_arr).unsqueeze(0)

        label_str = record[self.label_key]
        label_int = self.label_map[label_str]

        return tensor_img, label_int, rec_id

    def get_labels(self) -> np.ndarray:
        return np.array([self.label_map[s[self.label_key]] for s in self.valid_samples], dtype=np.int64)

    def get_sample_ids(self) -> List[str]:
        return [s["recording_id"] for s in self.valid_samples]
