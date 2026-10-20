"""
scripts/robust_glcm

Robust Signal-to-Texture (STI) Preprocessing Pipeline for EEG:
- Variational Mode Decomposition (VMD) & Empirical Mode Decomposition (EMD)
- Outlier-Immune Median/IQR Robust Scaling
- Quantile-Based Equi-Density Digitization (64 levels)
- Multi-Distance / Multi-Scale GLCM Generation [4, 64, 64]
- Domain-Specific Metadata & Haralick Cheatsheet Feature Vectors
"""

__version__ = "1.0.0"
