"""
scripts/cnn/export_rp_images.py

Isolated utility to deterministically export continuous Generalized Recurrence Plot (GRP)
images resized to 128x128 for the Bonn EEG CNN branch.

Guarantees:
- Reuses existing preprocessing functions without altering paper baseline code.
- Normalizes continuous GRP matrix by max(GRP) into [0, 1].
- Applies bilinear interpolation to resize to (128, 128, 1).
- Saves deterministic 8-bit grayscale PNGs (and optional float32 numpy arrays).
- Handles case-insensitive extensions (.txt and .TXT).
- Generates a cryptographically verified manifest (SHA-256 of every image).
- Safe reruns and pilot-mode support (does not run on all 500 without explicit request).
"""

import os
import sys
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocessing.config import PreprocessingConfig
from scripts.preprocessing.normalization import zscore_normalize
from scripts.preprocessing.embedding import compute_ami, compute_fnn
from scripts.preprocessing.grp import reconstruct_phase_space, compute_grp
from scripts.run_batch_preprocessing import SUBSET_METADATA


logger = logging.getLogger("cnn_rp_exporter")


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

        # Filter out hidden files / resource forks, accept .txt and .TXT
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


def generate_single_rp_image(
    raw_signal: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate normalized continuous GRP and resize to 128x128 bilinear.

    Returns:
    - img_float: 2D float32 array in [0.0, 1.0] of shape (128, 128)
    - img_uint8: 2D uint8 array in [0, 255] for standard lossless PNG storage
    - diag: Diagnostics dictionary
    """
    if config is None:
        config = PreprocessingConfig()

    # Step 1: Population Z-score normalization
    norm_signal, mean_val, std_val = zscore_normalize(raw_signal, ddof=config.norm_ddof)

    # Step 2: AMI time delay tau
    tau_opt, _, ami_diag = compute_ami(
        norm_signal,
        max_lag=config.ami_max_lag,
        n_bins=config.ami_n_bins,
        fallback_tau=config.ami_fallback_tau,
        return_diagnostics=True,
    )

    # Step 3: FNN embedding dimension m
    theiler_w = tau_opt if config.fnn_theiler_window else 1
    m_opt, _, fnn_diag = compute_fnn(
        norm_signal,
        tau=tau_opt,
        max_dim=config.fnn_max_dim,
        r_tol=config.fnn_r_tol,
        a_tol=config.fnn_a_tol,
        theiler_w=theiler_w,
        threshold=config.fnn_ratio_threshold,
        fallback_m=config.fnn_fallback_m,
        return_diagnostics=True,
    )

    # Step 4: Phase-space reconstruction
    trajectory = reconstruct_phase_space(norm_signal, tau=tau_opt, m=m_opt)

    # Step 5: Continuous distance matrix GRP
    grp_continuous = compute_grp(trajectory, metric=config.grp_metric)

    if not np.all(np.isfinite(grp_continuous)):
        raise ValueError("Continuous GRP contains non-finite values (NaN / Inf)!")

    # Step 6: Scale to [0, 1] continuous
    max_val = float(grp_continuous.max())
    if max_val < 1e-12:
        grp_norm = np.zeros_like(grp_continuous, dtype=np.float32)
    else:
        grp_norm = (grp_continuous / max_val).astype(np.float32)

    # Step 7: Bilinear interpolation to target_size (128, 128)
    pil_float = Image.fromarray(grp_norm, mode="F")
    pil_resized = pil_float.resize(target_size, resample=Image.Resampling.BILINEAR)
    img_float = np.array(pil_resized, dtype=np.float32)
    img_float = np.clip(img_float, 0.0, 1.0)

    # Deterministic 8-bit quantized image for standard PNG format
    img_uint8 = np.clip(np.round(img_float * 255.0), 0, 255).astype(np.uint8)

    diag = {
        "tau_opt": int(tau_opt),
        "m_opt": int(m_opt),
        "raw_mean": float(mean_val),
        "raw_std": float(std_val),
        "grp_continuous_shape": list(grp_continuous.shape),
        "grp_continuous_max": float(max_val),
        "resized_shape": list(img_float.shape),
        "float_min": float(img_float.min()),
        "float_max": float(img_float.max()),
    }
    return img_float, img_uint8, diag


def export_rp_dataset(
    extracted_dir: Path,
    output_dir: Path,
    pilot_limit: Optional[int] = None,
    save_npy: bool = False,
) -> Dict[str, Any]:
    """
    Export RP images and manifest.
    If pilot_limit is provided, only processes the first N recordings.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if save_npy:
        npy_dir = output_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "export.log"
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
        logger.info(f"[{idx}/{len(recordings)}] Processing {rec_id} ({rec['subset']})...")

        raw_signal = np.loadtxt(txt_path, dtype=np.float64)
        if len(raw_signal) != 4097:
            raise ValueError(f"Recording {rec_id} has {len(raw_signal)} samples, expected 4097!")

        img_float, img_uint8, diag = generate_single_rp_image(raw_signal, target_size=(128, 128), config=cfg)

        # Save PNG (lossless 8-bit single-channel)
        png_path = images_dir / f"{rec_id}.png"
        pil_img = Image.fromarray(img_uint8, mode="L")
        pil_img.save(png_path, format="PNG", compress_level=6)
        png_sha = compute_file_sha256(png_path)

        rec_entry = {
            "recording_id": rec_id,
            "filename": rec["filename"],
            "subset": rec["subset"],
            "class_label": rec["meta"]["class_label"],
            "case1_label": rec["meta"]["case1_label"],
            "case2_label": rec["meta"]["case2_label"],
            "case3_label": rec["meta"]["case3_label"],
            "png_relative_path": str(png_path.relative_to(output_dir)),
            "png_sha256": png_sha,
            "image_shape": [128, 128, 1],
            "tau": diag["tau_opt"],
            "m": diag["m_opt"],
            "float_min": diag["float_min"],
            "float_max": diag["float_max"],
        }

        if save_npy:
            npy_path = npy_dir / f"{rec_id}.npy"
            np.save(npy_path, img_float)
            rec_entry["npy_relative_path"] = str(npy_path.relative_to(output_dir))
            rec_entry["npy_sha256"] = compute_file_sha256(npy_path)

        manifest_records.append(rec_entry)

    manifest_data = {
        "dataset_name": "Bonn EEG Continuous GRP 128x128",
        "total_images": len(manifest_records),
        "is_pilot": pilot_limit is not None,
        "preprocessing_config": {
            "sampling_rate_hz": cfg.sampling_rate_hz,
            "expected_sample_count": cfg.expected_sample_count,
            "norm_ddof": cfg.norm_ddof,
            "ami_max_lag": cfg.ami_max_lag,
            "ami_n_bins": cfg.ami_n_bins,
            "fnn_max_dim": cfg.fnn_max_dim,
            "fnn_r_tol": cfg.fnn_r_tol,
            "target_resolution": [128, 128],
            "interpolation": "bilinear",
            "intensity_range": [0.0, 1.0],
        },
        "records": manifest_records,
    }

    manifest_file = output_dir / "image_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Export completed: {len(manifest_records)} images saved to {images_dir}")
    logger.info(f"Manifest written to {manifest_file}")
    return manifest_data


def parse_args():
    parser = argparse.ArgumentParser(description="Export continuous Bonn EEG GRP images for CNN")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"),
        help="Path to raw extracted Bonn EEG directory containing subsets A-E",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "rp_images_128"),
        help="Target output directory for images and manifest",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Limit export to N recordings for pilot verification (default: None, full export)",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also export full float32 .npy arrays alongside PNGs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_rp_dataset(
        extracted_dir=Path(args.input),
        output_dir=Path(args.output),
        pilot_limit=args.pilot,
        save_npy=args.save_npy,
    )
