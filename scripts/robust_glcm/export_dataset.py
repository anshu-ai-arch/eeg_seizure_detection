"""
scripts/robust_glcm/export_dataset.py

Batch Preprocessing and Dataset Exporter for the Robust GLCM & IMF Pipeline.

Pipeline:
1. Discovers 500 raw Bonn EEG recordings across subsets A, B, C, D, E.
2. Decomposes each signal into K=4 sorted modes via Variational Mode Decomposition (VMD).
3. Applies Outlier-Immune Median/IQR Robust Scaling and Quantile-Based Equi-Density Digitization (64 levels).
4. Generates multi-scale temporal GLCM texture tensors [4, 64, 64].
5. Extracts comprehensive domain Cheatsheet feature vectors (energy, entropy, ZCR, Haralick descriptors).
6. Exports float32 .npy tensors, cheatsheet CSV, visual sample PNGs, and cryptographic JSON manifest.

Completely isolated from existing CNN / baseline artifacts.
"""

import sys
import os
import time
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.robust_glcm.config import RobustGLCMConfig
from scripts.robust_glcm.decomposition import decompose_eeg_signal
from scripts.robust_glcm.glcm import generate_multimode_texture_tensor

logger = logging.getLogger("robust_glcm.export")


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


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def discover_bonn_recordings(raw_dir: Path) -> List[Dict[str, Any]]:
    """Discovers all valid EEG recordings across subsets A-E."""
    subsets = ["A", "B", "C", "D", "E"]
    records = []

    for s in subsets:
        s_dir = raw_dir / s
        if not s_dir.exists():
            continue
        for f in sorted(s_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith("."):
                rec_id = f.stem
                records.append({
                    "recording_id": rec_id,
                    "subset": s,
                    "raw_path": str(f.resolve()),
                    "filename": f.name,
                })

    records.sort(key=lambda x: x["recording_id"])
    return records


def compute_global_spectral_features(sig: np.ndarray, fs: float = 173.61) -> Dict[str, float]:
    """Computes Welch PSD relative power ratios across standard neural frequency bands."""
    freqs, psd = welch(sig, fs=fs, nperseg=256)
    tot = np.sum(psd) + 1e-12

    delta_p = float(np.sum(psd[(freqs >= 0.5) & (freqs < 4.0)]) / tot)
    theta_p = float(np.sum(psd[(freqs >= 4.0) & (freqs < 8.0)]) / tot)
    alpha_p = float(np.sum(psd[(freqs >= 8.0) & (freqs <= 13.0)]) / tot)
    beta_p = float(np.sum(psd[(freqs > 13.0) & (freqs <= 30.0)]) / tot)
    gamma_p = float(np.sum(psd[freqs > 30.0]) / tot)

    return {
        "global_delta_ratio": delta_p,
        "global_theta_ratio": theta_p,
        "global_alpha_ratio": alpha_p,
        "global_beta_ratio": beta_p,
        "global_gamma_ratio": gamma_p,
    }


def save_visual_sample(
    tensor: np.ndarray,
    out_png: Path,
    recording_id: str,
    subset: str,
    center_freqs: List[float],
) -> None:
    """Saves a multi-mode GLCM visualization as PNG for human inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_modes = tensor.shape[0]
    fig, axes = plt.subplots(1, num_modes, figsize=(4 * num_modes, 3.8), constrained_layout=True)
    if num_modes == 1:
        axes = [axes]

    for k in range(num_modes):
        im = axes[k].imshow(tensor[k], cmap="viridis", origin="lower", aspect="auto")
        axes[k].set_title(f"Mode {k} ({center_freqs[k]:.1f} Hz)", fontsize=11, fontweight="bold")
        axes[k].set_xlabel("Transition State (t+d)", fontsize=9)
        axes[k].set_ylabel("Current State (t)", fontsize=9)
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    fig.suptitle(f"Robust GLCM Texture Cube: {recording_id} (Subset {subset})", fontsize=13, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def export_robust_glcm_dataset(
    config: Optional[RobustGLCMConfig] = None,
    raw_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    save_visuals_count: int = 25,
) -> Dict[str, Any]:
    """
    Executes full robust GLCM dataset export across Bonn EEG recordings.
    """
    cfg = config or RobustGLCMConfig()
    base_out = cfg.output_base_dir
    tensors_dir = base_out / "tensors"
    visuals_dir = base_out / "visuals"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(base_out / "export_preprocessing.log")
    logger.info("=================================================================")
    logger.info("STARTING ROBUST GLCM & IMF PREPROCESSING PIPELINE")
    logger.info(f"Decomposition: {cfg.decomposition_method.upper()} (K={cfg.num_modes} modes)")
    logger.info(f"Quantization: Robust Median/IQR + Quantile Binning ({cfg.num_levels} levels)")
    logger.info(f"Temporal Lags: {cfg.glcm_distances}")
    logger.info(f"Output Directory: {base_out}")
    logger.info("=================================================================")

    actual_raw_dir = raw_dir or PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"
    recordings = discover_bonn_recordings(actual_raw_dir)
    logger.info(f"Discovered {len(recordings)} Bonn EEG recordings.")

    if limit is not None and limit > 0:
        recordings = recordings[:limit]
        logger.info(f"Applying limit: processing {len(recordings)} recordings.")

    manifest_records = []
    cheatsheet_rows = []
    t_start = time.time()

    for idx, rec in enumerate(recordings, start=1):
        rec_id = rec["recording_id"]
        subset = rec["subset"]
        raw_f = Path(rec["raw_path"])

        sig = np.loadtxt(raw_f, dtype=np.float64)
        n_samples = len(sig)

        # 1. Decompose into K sorted modes
        modes, center_freqs = decompose_eeg_signal(sig, cfg)

        # 2. Generate multi-mode GLCM texture tensor [K, num_levels, num_levels]
        #    and mode-level cheatsheet features
        texture_tensor, mode_cheatsheets = generate_multimode_texture_tensor(
            modes=modes,
            num_levels=cfg.num_levels,
            distances=cfg.glcm_distances,
            symmetric=cfg.glcm_symmetric,
            normalized=cfg.glcm_normalized,
        )

        # 3. Save tensor as float32 .npy
        tensor_path = tensors_dir / f"{rec_id}.npy"
        np.save(tensor_path, texture_tensor, allow_pickle=False)
        tensor_hash = compute_sha256(tensor_path)

        # 4. Global recording features
        glob_spectral = compute_global_spectral_features(sig, fs=cfg.sampling_rate)
        glob_energy = float(np.mean(sig ** 2))
        glob_std = float(np.std(sig))
        glob_line_len = float(np.mean(np.abs(np.diff(sig))))

        # 5. Build Cheatsheet row
        row = {
            "recording_id": rec_id,
            "subset": subset,
            "global_raw_std": glob_std,
            "global_log_raw_std": float(np.log(glob_std + 1e-6)),
            "global_energy": glob_energy,
            "global_log_energy": float(np.log(glob_energy + 1e-6)),
            "global_line_length": glob_line_len,
            "global_log_line_length": float(np.log(glob_line_len + 1e-6)),
        }
        row.update(glob_spectral)

        for m_idx in range(cfg.num_modes):
            m_feats = mode_cheatsheets[m_idx]
            cf = center_freqs[m_idx]
            row[f"mode{m_idx}_center_freq_hz"] = cf
            for k_name, val in m_feats.items():
                row[f"mode{m_idx}_{k_name}"] = val

        cheatsheet_rows.append(row)

        # 6. Save visual sample PNG for first N recordings
        visual_rel_path = None
        if idx <= save_visuals_count or (idx % 100 == 0):
            png_path = visuals_dir / f"{rec_id}_modes.png"
            save_visual_sample(texture_tensor, png_path, rec_id, subset, center_freqs)
            visual_rel_path = str(png_path.relative_to(base_out))

        # 7. Record manifest item
        rec_manifest = {
            "recording_id": rec_id,
            "subset": subset,
            "original_filename": rec["filename"],
            "raw_samples": n_samples,
            "tensor_shape": list(texture_tensor.shape),
            "tensor_relative_path": f"tensors/{rec_id}.npy",
            "tensor_sha256": tensor_hash,
            "visual_relative_path": visual_rel_path,
            "mode_center_frequencies_hz": center_freqs,
            "matrix_diagonal_fraction": float(np.mean([np.trace(texture_tensor[m]) for m in range(cfg.num_modes)])),
        }
        manifest_records.append(rec_manifest)

        if idx % 50 == 0 or idx == len(recordings):
            elapsed = time.time() - t_start
            logger.info(f"Processed [{idx}/{len(recordings)}] recordings in {elapsed:.1f}s ({elapsed/idx:.3f}s/rec)")

    # Save Cheatsheet DataFrame
    cheatsheet_df = pd.DataFrame(cheatsheet_rows)
    cheatsheet_csv_path = base_out / "cheatsheet_features.csv"
    cheatsheet_df.to_csv(cheatsheet_csv_path, index=False)
    logger.info(f"Saved Cheatsheet features table with {len(cheatsheet_df.columns)} columns to {cheatsheet_csv_path}")

    # Build and save master manifest
    manifest_data = {
        "dataset_name": "Bonn_EEG_Robust_GLCM_Decomposed",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_recordings": len(manifest_records),
        "configuration": {
            "decomposition_method": cfg.decomposition_method,
            "num_modes": cfg.num_modes,
            "num_levels": cfg.num_levels,
            "glcm_distances": list(cfg.glcm_distances),
            "glcm_symmetric": cfg.glcm_symmetric,
            "glcm_normalized": cfg.glcm_normalized,
            "sampling_rate": cfg.sampling_rate,
        },
        "cheatsheet_csv": "cheatsheet_features.csv",
        "cheatsheet_columns": list(cheatsheet_df.columns),
        "records": manifest_records,
    }

    manifest_json_path = base_out / "dataset_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved dataset manifest to {manifest_json_path}")

    total_time = time.time() - t_start
    logger.info(f"Dataset export completed in {total_time:.2f}s ({total_time/len(recordings):.3f}s/recording).")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Export Robust GLCM & IMF Dataset for EEG")
    parser.add_argument("--method", type=str, default="vmd", choices=["vmd", "emd"])
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--levels", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recordings for smoke testing")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_p = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "processed" / "robust_glcm"
    cfg = RobustGLCMConfig(
        decomposition_method=args.method,
        num_modes=args.modes,
        num_levels=args.levels,
        output_base_dir=out_p,
    )

    export_robust_glcm_dataset(config=cfg, limit=args.limit)


if __name__ == "__main__":
    main()
