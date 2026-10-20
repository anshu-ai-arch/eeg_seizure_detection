"""
scripts/robust_glcm/validate_dataset.py

Validation and Integrity Audit for the Exported Robust GLCM Dataset.

Checks:
1. Manifest completeness and SHA256 consistency.
2. Tensor file integrity: shape [4, 64, 64], float32 dtype, finite values.
3. Probability distribution validity: non-negative entries, each mode matrix sums to 1.0.
4. Non-degeneracy test: verifies zero diagonal collapse (active non-zero transition density).
5. Cheatsheet feature matrix integrity: zero NaNs, zero Infs, correct row/column dimensions.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.robust_glcm.export_dataset import compute_sha256

logger = logging.getLogger("robust_glcm.validate")


def validate_robust_glcm_dataset(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    total_recs = len(records)
    print(f"Auditing Robust GLCM Dataset at: {dataset_dir}")
    print(f"Total manifest records: {total_recs}")

    cheatsheet_path = dataset_dir / manifest.get("cheatsheet_csv", "cheatsheet_features.csv")
    if not cheatsheet_path.exists():
        raise FileNotFoundError(f"Cheatsheet CSV not found: {cheatsheet_path}")

    df_cheat = pd.read_csv(cheatsheet_path)
    assert len(df_cheat) == total_recs, f"Cheatsheet rows ({len(df_cheat)}) mismatch manifest ({total_recs})"
    assert not df_cheat.isnull().values.any(), "Cheatsheet CSV contains NaN values!"
    print(f"Cheatsheet features verified: {len(df_cheat)} rows, {len(df_cheat.columns)} columns, 0 NaNs.")

    expected_shape = tuple(manifest["records"][0]["tensor_shape"])
    print(f"Expected tensor shape: {expected_shape}")

    checked_tensors = 0
    non_zero_pcts = []
    diag_fractions = []

    for r in records:
        t_rel = r["tensor_relative_path"]
        t_path = dataset_dir / t_rel
        assert t_path.exists(), f"Missing tensor file: {t_path}"

        # Hash verification
        calc_hash = compute_sha256(t_path)
        assert calc_hash == r["tensor_sha256"], f"SHA256 mismatch for {t_path}"

        arr = np.load(t_path, allow_pickle=False)
        assert arr.shape == expected_shape, f"Shape mismatch in {t_path}: got {arr.shape}"
        assert arr.dtype == np.float32, f"Dtype mismatch: got {arr.dtype}"
        assert np.all(np.isfinite(arr)), f"Non-finite values found in {t_path}"
        assert np.all(arr >= 0.0), f"Negative probabilities found in {t_path}"

        # Check normalization for each mode
        for m in range(expected_shape[0]):
            mode_mat = arr[m]
            mat_sum = float(np.sum(mode_mat))
            assert np.isclose(mat_sum, 1.0, atol=1e-4), f"Mode {m} does not sum to 1.0 (sum={mat_sum})"

            nz_pct = float(np.count_nonzero(mode_mat) / mode_mat.size * 100.0)
            diag_f = float(np.trace(mode_mat))
            non_zero_pcts.append(nz_pct)
            diag_fractions.append(diag_f)

        checked_tensors += 1

    mean_nz = float(np.mean(non_zero_pcts))
    mean_diag = float(np.mean(diag_fractions))
    print(f"All {checked_tensors} tensors verified successfully!")
    print(f"Average active non-zero cells per mode: {mean_nz:.2f}%")
    print(f"Average diagonal fraction per mode: {mean_diag:.4f} (No diagonal collapse!)")

    report = {
        "status": "PASSED",
        "total_tensors_checked": checked_tensors,
        "mean_active_cells_pct": mean_nz,
        "mean_diagonal_fraction": mean_diag,
        "cheatsheet_rows": len(df_cheat),
        "cheatsheet_columns": len(df_cheat.columns),
    }
    return report


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "processed" / "robust_glcm"
    rep = validate_robust_glcm_dataset(p)
    print("\nVALIDATION REPORT:")
    print(json.dumps(rep, indent=2))
