"""
scripts/robust_glcm/config.py

Configuration dataclasses for the Robust GLCM & IMF Preprocessing Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass(frozen=True)
class RobustGLCMConfig:
    """
    Configuration parameters for EEG decomposition, quantization, GLCM construction,
    and metadata extraction.
    """
    # Signal sampling rate (Hz)
    sampling_rate: float = 173.61

    # Decomposition parameters
    decomposition_method: str = "vmd"  # "vmd" or "emd"
    num_modes: int = 4                 # 4 IMFs / modes
    vmd_alpha: float = 2000.0          # Bandwidth constraint for VMD
    vmd_tau: float = 0.0               # Noise-slack parameter
    vmd_dc: int = 0                    # DC component retention (0 = no DC drift mode)
    vmd_init: int = 1                  # 1 = uniform frequency distribution initialization
    vmd_tol: float = 1e-7              # Convergence tolerance

    # Quantization parameters
    num_levels: int = 64               # 64 discrete levels (0 to 63)
    clip_outliers: bool = True
    iqr_epsilon: float = 1e-7          # Numerical stability epsilon for zero-IQR segments

    # GLCM construction parameters
    glcm_distances: Tuple[int, ...] = (1, 2, 4, 8)  # Multi-scale temporal lags
    glcm_symmetric: bool = True                     # Forward + Backward time transition symmetry
    glcm_normalized: bool = True                    # Normalized to sum = 1.0 (probabilities)

    # Output paths
    output_base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "robust_glcm"
    )

    # Reproducibility
    random_seed: int = 42
