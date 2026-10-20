"""
scripts/robust_glcm/decomposition.py

Signal Decomposition Engine for EEG:
Supports:
1. Variational Mode Decomposition (VMD) - Dragomiretskiy & Zosso (2014)
   Convex, band-limited optimization with zero mode-mixing.
2. Empirical Mode Decomposition (EMD) - Huang et al. (1998)
   Adaptive sifting decomposition into Intrinsic Mode Functions (IMFs).

Guarantees:
- Output modes are strictly sorted by central frequency (Mode 0 = lowest/Delta to Mode K-1 = highest/Gamma).
- Constant output shape [K, N] across all input recordings.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np

from scripts.robust_glcm.config import RobustGLCMConfig

logger = logging.getLogger("robust_glcm.decomposition")


def decompose_vmd(
    signal: np.ndarray,
    num_modes: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    dc: int = 0,
    init: int = 1,
    tol: float = 1e-7,
    sampling_rate: float = 173.61,
) -> Tuple[np.ndarray, List[float]]:
    """
    Decomposes a 1D EEG signal using Variational Mode Decomposition (VMD).

    Returns:
        sorted_modes: [num_modes, N] float32 array, sorted from lowest to highest center frequency.
        center_freqs_hz: List of K center frequencies in Hertz.
    """
    try:
        from vmdpy import VMD
    except ImportError as e:
        raise ImportError("vmdpy package is required. Install via `pip install vmdpy`.") from e

    # Ensure 1D float64 for numerical precision during VMD optimization
    sig_f64 = np.asarray(signal, dtype=np.float64).flatten()
    orig_len = len(sig_f64)

    # VMD execution
    u, u_hat, omega = VMD(
        f=sig_f64,
        alpha=alpha,
        tau=tau,
        K=num_modes,
        DC=dc,
        init=init,
        tol=tol,
    )

    # Center frequencies at the final iteration (normalized 0 to 0.5 where 0.5 = Nyquist)
    final_omegas = omega[-1]  # shape: [K]
    center_freqs_hz = (final_omegas * sampling_rate).tolist()

    # Sort modes in ascending order of center frequency (Mode 0 = Delta/slowest, Mode K-1 = Gamma/fastest)
    sort_idx = np.argsort(final_omegas)
    sorted_modes = u[sort_idx].astype(np.float32)
    sorted_freqs = [float(center_freqs_hz[i]) for i in sort_idx]

    # If VMD padded or cropped by 1 sample due to FFT even/odd lengths, ensure exact length match
    if sorted_modes.shape[1] < orig_len:
        pad_width = orig_len - sorted_modes.shape[1]
        sorted_modes = np.pad(sorted_modes, ((0, 0), (0, pad_width)), mode="edge")
    elif sorted_modes.shape[1] > orig_len:
        sorted_modes = sorted_modes[:, :orig_len]

    return sorted_modes, sorted_freqs


def decompose_emd(
    signal: np.ndarray,
    num_modes: int = 4,
    sampling_rate: float = 173.61,
) -> Tuple[np.ndarray, List[float]]:
    """
    Decomposes a 1D EEG signal using Empirical Mode Decomposition (EMD).

    Returns:
        sorted_modes: [num_modes, N] float32 array, sorted from lowest to highest dominant frequency.
        center_freqs_hz: Estimated dominant frequencies in Hertz.
    """
    try:
        from PyEMD import EMD
    except ImportError as e:
        raise ImportError("PyEMD package is required. Install via `pip install EMD-signal`.") from e

    sig_f64 = np.asarray(signal, dtype=np.float64).flatten()
    orig_len = len(sig_f64)

    emd = EMD()
    imfs = emd(sig_f64)  # shape: [M, N]
    n_imfs = imfs.shape[0]

    # Select first num_modes IMFs or pad with zeros if fewer
    if n_imfs >= num_modes:
        selected = imfs[:num_modes]
    else:
        pad_count = num_modes - n_imfs
        zero_imfs = np.zeros((pad_count, orig_len), dtype=np.float64)
        selected = np.vstack([imfs, zero_imfs])

    # Estimate dominant frequency of each IMF via zero-crossing rate
    est_freqs = []
    for i in range(num_modes):
        imf = selected[i]
        zc = np.sum(np.diff(np.signbit(imf)))
        freq = float(zc * sampling_rate / (2.0 * max(1, orig_len)))
        est_freqs.append(freq)

    sort_idx = np.argsort(est_freqs)
    sorted_modes = selected[sort_idx].astype(np.float32)
    sorted_freqs = [float(est_freqs[i]) for i in sort_idx]

    return sorted_modes, sorted_freqs


def decompose_eeg_signal(
    signal: np.ndarray,
    config: Optional[RobustGLCMConfig] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Unified entry point for EEG decomposition based on configuration.
    """
    cfg = config or RobustGLCMConfig()

    if cfg.decomposition_method.lower() == "vmd":
        return decompose_vmd(
            signal=signal,
            num_modes=cfg.num_modes,
            alpha=cfg.vmd_alpha,
            tau=cfg.vmd_tau,
            dc=cfg.vmd_dc,
            init=cfg.vmd_init,
            tol=cfg.vmd_tol,
            sampling_rate=cfg.sampling_rate,
        )
    elif cfg.decomposition_method.lower() == "emd":
        return decompose_emd(
            signal=signal,
            num_modes=cfg.num_modes,
            sampling_rate=cfg.sampling_rate,
        )
    else:
        raise ValueError(f"Unknown decomposition method: {cfg.decomposition_method}. Expected 'vmd' or 'emd'.")
