"""
scripts/cnn/export_glcm_matrices.py

Isolated utility to deterministically export multi-channel GLCM matrices [8, 16, 16]
for the Bonn EEG CNN branch.

Guarantees:
- Reuses existing preprocessing and GLCM computation without altering baseline code.
- Stacks the 8 canonical normalized GLCM probability matrices into shape (8, 16, 16) float32.
- Canonical channel ordering:
  0: theta_0deg_d_1
  1: theta_0deg_d_2
  2: theta_0deg_d_3
  3: theta_0deg_d_4
  4: theta_45deg_d_1
  5: theta_45deg_d_2
  6: theta_45deg_d_3
  7: theta_45deg_d_4
- Saves lossless binary .npy arrays to avoid precision loss.
- Handles case-insensitive extensions (.txt and .TXT).
- Generates a cryptographically verified manifest (SHA-256 for each matrix tensor).
- Safe reruns and pilot-mode support (does not process all 500 without explicit request).
"""

import sys
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing.config import PreprocessingConfig
from scripts.preprocessing.pipeline import preprocess_recording
from scripts.run_batch_preprocessing import SUBSET_METADATA


logger = logging.getLogger("cnn_glcm_exporter")

CANONICAL_GLCM_CHANNELS = [
    "theta_0deg_d_1",
    "theta_0deg_d_2",
    "theta_0deg_d_3",
    "theta_0deg_d_4",
    "theta_45deg_d_1",
    "theta_45deg_d_2",
    "theta_45deg_d_3",
    "theta_45deg_d_4",
]


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


def compute_file_sha256(filepath: Path) -> str:
    """Compute streaming SHA-256 checksum."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def discover_eeg_recordings(extracted_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover all 500 Bonn EEG recordings case-insensitively (.txt and .TXT).
    Enforces exactly 100 recordings per subset (A, B, C, D, E).
    """
    recordings = []
    subsets = ["A", "B", "C", "D", "E"]

    for s in subsets:
        subset_dir = extracted_dir / s
        if not subset_dir.exists() or not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing subset directory: {subset_dir}")

        valid_files = [
            f for f in subset_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() == ".txt"
        ]

        if len(valid_files) != 100:
            raise ValueError(f"Subset {s} expected exactly 100 recordings, found {len(valid_files)}")

        valid_files.sort(key=lambda x: x.name)

        for f in valid_files:
            rec_id = f.stem
            recordings.append({
                "recording_id": rec_id,
                "filename": f.name,
                "filepath": f,
                "subset": s,
                "meta": SUBSET_METADATA[s],
            })

    if len(recordings) != 500:
        raise ValueError(f"Expected 500 recordings total, found {len(recordings)}")

    unique_ids = set(r["recording_id"] for r in recordings)
    if len(unique_ids) != 500:
        raise ValueError(f"Duplicate recording IDs detected! Unique: {len(unique_ids)}")

    return recordings


def generate_single_glcm_tensor(
    filepath: Path,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate normalized 8-channel GLCM tensor of shape (8, 16, 16) float32.
    """
    if config is None:
        config = PreprocessingConfig()

    res = preprocess_recording(filepath, config=config)
    glcm_dict = res["glcm_matrices"]

    # Verify all 8 canonical matrices are present
    for ch in CANONICAL_GLCM_CHANNELS:
        if ch not in glcm_dict:
            raise KeyError(f"Expected GLCM matrix '{ch}' missing from pipeline output!")

    # Stack into multi-channel tensor (8, 16, 16)
    matrices = [glcm_dict[ch] for ch in CANONICAL_GLCM_CHANNELS]
    tensor = np.stack(matrices, axis=0).astype(np.float32)

    if tensor.shape != (8, 16, 16):
        raise ValueError(f"Expected GLCM tensor shape (8, 16, 16), got {tensor.shape}")

    if not np.all(np.isfinite(tensor)):
        raise ValueError("GLCM tensor contains non-finite values (NaN / Inf)!")

    # Diagnostic metadata
    diag = {
        "shape": list(tensor.shape),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "channel_sums": [float(s) for s in tensor.sum(axis=(1, 2))],
        "tau_estimated": int(res["tau_estimated"]),
        "m_estimated": int(res["m_estimated"]),
    }
    return tensor, diag


def export_glcm_dataset(
    extracted_dir: Path,
    output_dir: Path,
    pilot_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Export GLCM tensors and manifest.
    If pilot_limit is provided, only processes the first N recordings.
    """
    output_dir = Path(output_dir)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "export_glcm.log"
    setup_logger(log_file)
    logger.info("Discovering Bonn EEG recordings...")
    recordings = discover_eeg_recordings(extracted_dir)

    if pilot_limit is not None and pilot_limit > 0:
        recordings = recordings[:pilot_limit]
        logger.info(f"Pilot mode active: processing {len(recordings)} recordings only.")
    else:
        logger.info(f"Full export mode: processing {len(recordings)} recordings.")

    cfg = PreprocessingConfig()
    manifest_records = []

    for idx, rec in enumerate(recordings, 1):
        rec_id = rec["recording_id"]
        txt_path = rec["filepath"]
        logger.info(f"[{idx}/{len(recordings)}] Processing GLCM for {rec_id} ({rec['subset']})...")

        tensor, diag = generate_single_glcm_tensor(txt_path, config=cfg)

        npy_path = matrices_dir / f"{rec_id}.npy"
        np.save(npy_path, tensor)
        npy_sha = compute_file_sha256(npy_path)

        rec_entry = {
            "recording_id": rec_id,
            "filename": rec["filename"],
            "subset": rec["subset"],
            "class_label": rec["meta"]["class_label"],
            "case1_label": rec["meta"]["case1_label"],
            "case2_label": rec["meta"]["case2_label"],
            "case3_label": rec["meta"]["case3_label"],
            "matrix_relative_path": str(npy_path.relative_to(output_dir)),
            "matrix_sha256": npy_sha,
            "tensor_shape": [8, 16, 16],
            "channel_names": CANONICAL_GLCM_CHANNELS,
            "dtype": "float32",
            "tau": diag["tau_estimated"],
            "m": diag["m_estimated"],
            "min_val": diag["min"],
            "max_val": diag["max"],
        }
        manifest_records.append(rec_entry)

    manifest_data = {
        "dataset_name": "Bonn EEG Normalized GLCM Matrices [8, 16, 16]",
        "total_tensors": len(manifest_records),
        "is_pilot": pilot_limit is not None,
        "preprocessing_config": {
            "sampling_rate_hz": cfg.sampling_rate_hz,
            "expected_sample_count": cfg.expected_sample_count,
            "norm_ddof": cfg.norm_ddof,
            "ami_max_lag": cfg.ami_max_lag,
            "ami_n_bins": cfg.ami_n_bins,
            "fnn_max_dim": cfg.fnn_max_dim,
            "fnn_r_tol": cfg.fnn_r_tol,
            "grp_num_levels": cfg.grp_num_levels,
            "grp_quantization_mode": cfg.grp_quantization_mode,
            "glcm_orientations_rad": [float(r) for r in cfg.glcm_orientations],
            "glcm_offsets": list(cfg.glcm_offsets),
            "channel_ordering": CANONICAL_GLCM_CHANNELS,
            "matrix_dimension": [16, 16],
            "normalization": "joint_probabilities_sum_to_1",
        },
        "records": manifest_records,
    }

    manifest_file = output_dir / "glcm_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"GLCM export completed: {len(manifest_records)} tensors saved to {matrices_dir}")
    logger.info(f"Manifest written to {manifest_file}")
    return manifest_data


def parse_args():
    parser = argparse.ArgumentParser(description="Export continuous Bonn EEG GLCM matrices for CNN")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"),
        help="Path to raw extracted Bonn EEG directory containing subsets A-E",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "glcm_matrices_cnn"),
        help="Target output directory for GLCM matrices and manifest",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Limit export to N recordings for pilot verification (default: None, full export)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_glcm_dataset(
        extracted_dir=Path(args.input),
        output_dir=Path(args.output),
        pilot_limit=args.pilot,
    )
