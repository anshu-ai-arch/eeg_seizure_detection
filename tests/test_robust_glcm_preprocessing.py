"""
tests/test_robust_glcm_preprocessing.py

Unit test suite for the Robust GLCM & IMF Preprocessing Pipeline.
Tests:
- Variational Mode Decomposition (VMD) frequency sorting and shapes.
- Outlier immunity of Robust Median/IQR Quantile Binning.
- Multi-distance GLCM symmetry, non-negativity, and normalization.
- Vectorized Haralick texture descriptor computation.
- Cheatsheet Feature Vector extraction and finiteness.
"""

import numpy as np
import pytest

from scripts.robust_glcm.config import RobustGLCMConfig
from scripts.robust_glcm.decomposition import decompose_vmd, decompose_eeg_signal
from scripts.robust_glcm.glcm import (
    robust_quantile_quantize,
    compute_single_1d_glcm,
    compute_multidistance_glcm,
    compute_haralick_descriptors,
    extract_cheatsheet_features,
    generate_multimode_texture_tensor,
)


def test_vmd_decomposition():
    """Verify VMD extracts 4 modes strictly sorted in ascending order of center frequency."""
    t = np.linspace(0, 1, 500)
    # Composite signal: 2 Hz (Delta), 10 Hz (Alpha), 25 Hz (Beta), 50 Hz (Gamma)
    sig = (
        1.0 * np.sin(2 * np.pi * 2 * t)
        + 0.8 * np.sin(2 * np.pi * 10 * t)
        + 0.6 * np.sin(2 * np.pi * 25 * t)
        + 0.4 * np.sin(2 * np.pi * 50 * t)
    )

    modes, freqs = decompose_vmd(sig, num_modes=4, sampling_rate=173.61)

    assert modes.shape == (4, 500)
    assert len(freqs) == 4
    # Ensure modes are strictly sorted by center frequency
    for i in range(len(freqs) - 1):
        assert freqs[i] <= freqs[i + 1], f"Frequencies not sorted: {freqs}"


def test_robust_quantile_quantization_outlier_immunity():
    """Verify that a massive outlier spike does NOT collapse normal brain waves into a single bin."""
    np.random.seed(42)
    # 256 normal samples with std=15.0
    sig = np.random.normal(loc=0.0, scale=15.0, size=256)
    # Add a massive voltage spike (+1000 uV)
    sig[50] = 1000.0
    sig[100] = -1000.0

    discrete = robust_quantile_quantize(sig, num_levels=64)

    assert discrete.shape == (256,)
    assert discrete.min() >= 0
    assert discrete.max() <= 63

    # Check that normal values are NOT all crammed into 1 or 2 bins
    unique_levels = len(np.unique(discrete))
    assert unique_levels >= 40, f"Outlier caused severe bin collapse: only {unique_levels} unique bins used!"


def test_single_1d_glcm_properties():
    """Verify GLCM matrix symmetry, probability normalization, and shape."""
    discrete = np.random.randint(0, 64, size=300, dtype=np.int32)
    glcm = compute_single_1d_glcm(discrete, num_levels=64, distance=1, symmetric=True, normalized=True)

    assert glcm.shape == (64, 64)
    assert np.all(glcm >= 0.0)
    assert np.isclose(np.sum(glcm), 1.0, atol=1e-5)
    # Check exact symmetry
    assert np.allclose(glcm, glcm.T, atol=1e-6)


def test_multidistance_glcm():
    """Verify multi-distance GLCM fuses multiple temporal lags properly."""
    discrete = np.random.randint(0, 64, size=500, dtype=np.int32)
    fused = compute_multidistance_glcm(discrete, num_levels=64, distances=(1, 2, 4, 8), symmetric=True, normalized=True)

    assert fused.shape == (64, 64)
    assert np.all(fused >= 0.0)
    assert np.isclose(np.sum(fused), 1.0, atol=1e-5)
    assert np.allclose(fused, fused.T, atol=1e-6)


def test_haralick_descriptors_finiteness():
    """Verify Haralick descriptors produce finite, valid metrics."""
    discrete = np.random.randint(0, 64, size=500, dtype=np.int32)
    glcm = compute_single_1d_glcm(discrete, num_levels=64, distance=1, symmetric=True, normalized=True)

    h = compute_haralick_descriptors(glcm)

    assert isinstance(h, dict)
    expected_keys = {"contrast", "dissimilarity", "homogeneity", "energy_asm", "correlation"}
    assert expected_keys.issubset(h.keys())
    for k, v in h.items():
        assert np.isfinite(v), f"Haralick descriptor {k} is not finite: {v}"
    assert 0.0 <= h["homogeneity"] <= 1.0
    assert -1.0 <= h["correlation"] <= 1.0


def test_cheatsheet_features():
    """Verify domain Cheatsheet feature vector contains physical energy, line length, ZCR, entropy."""
    raw_sig = np.random.normal(loc=0.0, scale=30.0, size=500)
    glcm = np.ones((64, 64), dtype=np.float32) / (64 * 64)

    feats = extract_cheatsheet_features(raw_sig, glcm)

    for k, v in feats.items():
        assert np.isfinite(v), f"Cheatsheet feature {k} is NaN or Inf: {v}"
    assert feats["energy"] > 0.0
    assert feats["line_length"] > 0.0
    assert 0.0 <= feats["zero_crossing_rate"] <= 1.0
    assert feats["shannon_entropy"] > 0.0


def test_multimode_texture_tensor_generation():
    """Verify [4, 64, 64] texture cube generation from 4 decomposed modes."""
    modes = np.random.normal(0, 10, size=(4, 500)).astype(np.float32)

    tensor, cheatsheets = generate_multimode_texture_tensor(
        modes=modes,
        num_levels=64,
        distances=(1, 2, 4, 8),
    )

    assert tensor.shape == (4, 64, 64)
    assert tensor.dtype == np.float32
    assert len(cheatsheets) == 4
    for m in range(4):
        assert np.isclose(np.sum(tensor[m]), 1.0, atol=1e-5)
