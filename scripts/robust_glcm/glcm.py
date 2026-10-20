"""
scripts/robust_glcm/glcm.py

Robust Quantization, Multi-Distance GLCM Generation, and Domain Cheatsheet Vector Extraction.

Features:
1. Robust Scaling via Median and Interquartile Range (IQR) - completely immune to extreme amplitude spikes.
2. Quantile-Based Equi-Density Digitization - guarantees active utilization of all discrete levels.
3. Multi-Distance Temporal GLCM [K_modes, num_levels, num_levels] mapping transitions across multiple scales.
4. Comprehensive Cheatsheet Feature Vector (Physical Energy, Line Length, ZCR, Entropy, and Haralick Descriptors).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def robust_quantile_quantize(
    signal: np.ndarray,
    num_levels: int = 64,
    iqr_epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Robust Scaling and Quantile Binning Pipeline to map continuous EEG signals
    into strict GLCM index integers (0 to num_levels - 1) without amplitude resolution loss.

    Steps:
    1. Robust Centering & Scaling: (x - median) / IQR. Outliers outside 25th-75th percentiles
       cannot compress the internal dynamic range.
    2. Quantile-Based Equi-Density Digitization: Bin edges are placed at exact sample percentiles,
       guaranteeing that every bin receives a dense, balanced allocation of transitions.
    """
    sig = np.asarray(signal, dtype=np.float64).flatten()
    n = len(sig)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    # 1. Robust Scaling via Median and Interquartile Range (IQR)
    q25, q75 = np.percentile(sig, [25.0, 75.0])
    iqr = q75 - q25
    if iqr < iqr_epsilon:
        # Fallback to standard deviation if IQR is near zero (e.g. flat line)
        sig_std = np.std(sig)
        iqr = sig_std if sig_std > iqr_epsilon else 1.0

    scaled = (sig - np.median(sig)) / iqr

    # 2. Quantile-Based Equi-Density Bin Edges
    percentiles = np.linspace(0.0, 100.0, num_levels + 1)
    bins = np.percentile(scaled, percentiles)

    # Force inclusion of the maximum boundary point
    bins[-1] += 1e-5

    # Check for duplicate bin edges (can happen if signal has repeated values)
    unique_bins = np.unique(bins)
    if len(unique_bins) < num_levels + 1:
        # Interpolate linearly across unique range to avoid degenerate collapsed bins
        bins = np.linspace(scaled.min(), scaled.max() + 1e-5, num_levels + 1)

    # Digitise continuous signal points into strict integer bin assignments (0 to num_levels - 1)
    discrete = np.digitize(scaled, bins) - 1
    discrete = np.clip(discrete, 0, num_levels - 1).astype(np.int32)

    return discrete


def compute_single_1d_glcm(
    discrete_seq: np.ndarray,
    num_levels: int = 64,
    distance: int = 1,
    symmetric: bool = True,
    normalized: bool = True,
) -> np.ndarray:
    """
    Computes a 2D Gray-Level Co-occurrence Matrix (GLCM) from a 1D discrete sequence.

    Args:
        discrete_seq: 1D array of integers in [0, num_levels - 1].
        num_levels: Size of the GLCM matrix (num_levels x num_levels).
        distance: Temporal lag offset (in samples).
        symmetric: If True, adds forward (t -> t+d) and backward (t+d -> t) transitions.
        normalized: If True, normalizes matrix so sum(P_ij) = 1.0 (probabilities).

    Returns:
        glcm: [num_levels, num_levels] float32 array.
    """
    glcm = np.zeros((num_levels, num_levels), dtype=np.float32)
    n = len(discrete_seq)

    if n <= distance:
        return glcm

    rows = discrete_seq[:-distance]
    cols = discrete_seq[distance:]

    # Fast 2D bincount accumulation
    flat_indices = rows * num_levels + cols
    counts = np.bincount(flat_indices, minlength=num_levels * num_levels)
    glcm = counts.reshape((num_levels, num_levels)).astype(np.float32)

    if symmetric:
        glcm = glcm + glcm.T

    if normalized:
        total = np.sum(glcm)
        if total > 0.0:
            glcm /= total

    return glcm


def compute_multidistance_glcm(
    discrete_seq: np.ndarray,
    num_levels: int = 64,
    distances: Tuple[int, ...] = (1, 2, 4, 8),
    symmetric: bool = True,
    normalized: bool = True,
) -> np.ndarray:
    """
    Computes multi-scale GLCM matrices across a set of temporal distances
    and averages them into a unified multi-scale texture map.

    Returns:
        fused_glcm: [num_levels, num_levels] float32 array.
    """
    matrices = []
    for d in distances:
        m = compute_single_1d_glcm(
            discrete_seq=discrete_seq,
            num_levels=num_levels,
            distance=d,
            symmetric=symmetric,
            normalized=normalized,
        )
        matrices.append(m)

    fused = np.mean(matrices, axis=0).astype(np.float32)
    if normalized:
        tot = np.sum(fused)
        if tot > 0.0:
            fused /= tot
    return fused


def compute_haralick_descriptors(glcm: np.ndarray) -> Dict[str, float]:
    """
    Vectorized computation of classic Haralick texture descriptors from a normalized GLCM matrix:
    1. Contrast: Sum((i - j)^2 * P_ij)
    2. Dissimilarity: Sum(|i - j| * P_ij)
    3. Homogeneity: Sum(P_ij / (1 + (i - j)^2))
    4. Angular Second Moment (ASM / Energy): Sum(P_ij^2)
    5. Correlation: Linear gray-level dependency across transition pairs.
    """
    num_levels = glcm.shape[0]
    i_grid, j_grid = np.indices((num_levels, num_levels), dtype=np.float64)
    diff = i_grid - j_grid

    p = glcm.astype(np.float64)
    p_sum = np.sum(p)
    if p_sum > 0.0:
        p = p / p_sum

    contrast = float(np.sum((diff ** 2) * p))
    dissimilarity = float(np.sum(np.abs(diff) * p))
    homogeneity = float(np.sum(p / (1.0 + (diff ** 2))))
    energy_asm = float(np.sum(p ** 2))

    # Mean and variance along marginals
    mu_i = np.sum(i_grid * p)
    mu_j = np.sum(j_grid * p)
    var_i = np.sum(((i_grid - mu_i) ** 2) * p)
    var_j = np.sum(((j_grid - mu_j) ** 2) * p)
    std_ij = np.sqrt(var_i * var_j) + 1e-12

    correlation = float(np.sum((i_grid - mu_i) * (j_grid - mu_j) * p) / std_ij)

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy_asm": energy_asm,
        "correlation": correlation,
    }


def extract_cheatsheet_features(
    raw_signal: np.ndarray,
    glcm_matrix: np.ndarray,
    sampling_rate: float = 173.61,
) -> Dict[str, float]:
    """
    Extracts parallel domain-specific metadata features directly from the raw mode and its GLCM:
    1. Physical amplitude: raw_std, log_raw_std
    2. Physical power: energy, log_energy
    3. Waveform complexity: line_length, log_line_length, zero_crossing_rate
    4. Transition randomness: shannon_entropy
    5. Haralick texture descriptors: contrast, dissimilarity, homogeneity, energy_asm, correlation
    """
    sig = np.asarray(raw_signal, dtype=np.float64).flatten()
    n = len(sig)

    raw_std = float(np.std(sig))
    energy = float(np.mean(sig ** 2)) if n > 0 else 0.0
    line_len = float(np.mean(np.abs(np.diff(sig)))) if n > 1 else 0.0
    zcr = float(np.sum(np.diff(np.signbit(sig)))) / max(1, n)

    log_raw_std = float(np.log(raw_std + 1e-6))
    log_energy = float(np.log(energy + 1e-6))
    log_line_length = float(np.log(line_len + 1e-6))

    # 2D Shannon Entropy from GLCM transition distribution
    p = glcm_matrix.flatten().astype(np.float64)
    p = p[p > 0.0]
    shannon_entropy = float(-np.sum(p * np.log2(p + 1e-12))) if len(p) > 0 else 0.0

    # Haralick texture features
    haralick = compute_haralick_descriptors(glcm_matrix)

    feats = {
        "raw_std": raw_std,
        "log_raw_std": log_raw_std,
        "energy": energy,
        "log_energy": log_energy,
        "line_length": line_len,
        "log_line_length": log_line_length,
        "zero_crossing_rate": zcr,
        "shannon_entropy": shannon_entropy,
    }
    feats.update(haralick)
    return feats


def generate_multimode_texture_tensor(
    modes: np.ndarray,
    num_levels: int = 64,
    distances: Tuple[int, ...] = (1, 2, 4, 8),
    symmetric: bool = True,
    normalized: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Generates a [K_modes, num_levels, num_levels] texture tensor from K decomposed signal modes,
    and extracts a parallel list of domain Cheatsheet feature dictionaries (one per mode).

    Returns:
        texture_tensor: [K, num_levels, num_levels] float32 array.
        cheatsheet_list: List of K feature dictionaries.
    """
    num_modes = modes.shape[0]
    texture_tensor = np.zeros((num_modes, num_levels, num_levels), dtype=np.float32)
    cheatsheet_list = []

    for k in range(num_modes):
        raw_mode = modes[k]
        # 1. Outlier-immune robust quantile quantization
        discrete = robust_quantile_quantize(raw_mode, num_levels=num_levels)
        # 2. Multi-distance temporal GLCM fusion
        glcm_mat = compute_multidistance_glcm(
            discrete_seq=discrete,
            num_levels=num_levels,
            distances=distances,
            symmetric=symmetric,
            normalized=normalized,
        )
        texture_tensor[k] = glcm_mat

        # 3. Domain Cheatsheet Vector extraction
        feats = extract_cheatsheet_features(raw_mode, glcm_mat)
        cheatsheet_list.append(feats)

    return texture_tensor, cheatsheet_list
