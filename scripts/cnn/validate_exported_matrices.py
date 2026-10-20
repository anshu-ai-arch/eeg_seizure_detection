"""
scripts/cnn/validate_exported_matrices.py

Independent, read-only validator for the full 500-recording GLCM dataset export.
Validates:
- Exactly 500 .npy files in matrices/
- Exactly 500 manifest records in glcm_manifest.json
- SHA-256 hash match for all 500 files
- Shape (8, 16, 16), dtype float32, all finite, non-negative, <= 1.0
- Channel sums approx 1.0 (atol=1e-5)
- Exact class distribution across subsets A, B, C, D, E (100 each)
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_export(
    export_dir: Path,
    expected_count: int = 500,
    channel_sum_atol: float = 1e-5,
) -> dict:
    export_dir = Path(export_dir)
    matrices_dir = export_dir / "matrices"
    manifest_path = export_dir / "glcm_manifest.json"

    assert export_dir.exists(), f"Export dir does not exist: {export_dir}"
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
    assert matrices_dir.exists(), f"Matrices dir missing: {matrices_dir}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    assert len(records) == expected_count, f"Manifest records count {len(records)} != {expected_count}"

    npy_files = list(matrices_dir.glob("*.npy"))
    assert len(npy_files) == expected_count, f"Found {len(npy_files)} .npy files != {expected_count}"

    subsets_count = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    max_channel_sum_dev = 0.0
    global_min_val = 1.0
    global_max_val = 0.0

    for idx, r in enumerate(records, 1):
        rec_id = r["recording_id"]
        sub = r["subset"]
        subsets_count[sub] += 1

        npy_path = matrices_dir / f"{rec_id}.npy"
        assert npy_path.exists(), f"Missing file: {npy_path}"

        # SHA-256 verification
        actual_sha = compute_sha256(npy_path)
        assert actual_sha == r["matrix_sha256"], f"SHA mismatch for {rec_id}!"

        # Array verification (allow_pickle=False)
        arr = np.load(npy_path, allow_pickle=False)
        assert arr.shape == (8, 16, 16), f"Wrong shape {arr.shape} for {rec_id}"
        assert arr.dtype == np.float32, f"Wrong dtype {arr.dtype} for {rec_id}"
        assert np.all(np.isfinite(arr)), f"Non-finite values in {rec_id}"
        assert not np.any(arr < 0.0), f"Negative values in {rec_id}"
        assert not np.any(arr > 1.0), f"Values > 1.0 in {rec_id}"

        # Channel sums
        ch_sums = arr.sum(axis=(1, 2))
        devs = np.abs(ch_sums - 1.0)
        max_dev = float(np.max(devs))
        if max_dev > max_channel_sum_dev:
            max_channel_sum_dev = max_dev
        assert max_dev <= channel_sum_atol, f"Channel sum dev {max_dev} > {channel_sum_atol} in {rec_id}"

        global_min_val = min(global_min_val, float(np.min(arr)))
        global_max_val = max(global_max_val, float(np.max(arr)))

    for s, c in subsets_count.items():
        assert c == 100, f"Subset {s} count {c} != 100"

    summary = {
        "export_dir": str(export_dir),
        "total_recordings_validated": len(records),
        "total_npy_files": len(npy_files),
        "subset_distribution": subsets_count,
        "global_min_value": float(global_min_val),
        "global_max_value": float(global_max_val),
        "max_channel_sum_deviation_from_1": float(max_channel_sum_dev),
        "channel_sum_tolerance_used": float(channel_sum_atol),
        "all_sha256_matched": True,
        "all_shapes_exact_8_16_16": True,
        "all_dtypes_float32": True,
        "all_values_finite_and_valid_probabilities": True,
        "status": "PASS",
    }
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/processed/glcm_matrices_cnn")
    args = parser.parse_args()

    res = validate_export(Path(args.dir))
    print(json.dumps(res, indent=2))
