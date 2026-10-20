"""
scripts/transformer

1D Gated Transformer Architecture for EEG Signal-to-Texture Classification:
- Markov State Tokenization of Multi-Mode GLCM Tensors [4, 64, 64]
- 2-Layer Pre-LN Multi-Head Self-Attention Transformer Encoder (d_model=32, nhead=4)
- Domain Cheatsheet Vector Feature Projection (69 features)
- Dynamic Sigmoid Gated Late Fusion
- Direct Physical Amplitude/Energy Residual Logit Connection
"""

__version__ = "1.0.0"
