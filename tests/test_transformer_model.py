"""
tests/test_transformer_model.py

Unit test suite for the 1D Gated GLCM Transformer architecture.
"""

import torch
import numpy as np
import pytest

from scripts.transformer.model import Bonn1DGatedGLCMTransformer
from scripts.transformer.dataset import RobustGLCMDataset


def test_transformer_forward_and_output_shapes():
    """Verify forward pass output dimensions for binary and 3-class settings."""
    model_3c = Bonn1DGatedGLCMTransformer(num_classes=3, num_cheatsheet=67, d_model=32, nhead=4, num_layers=2)
    b_glcm = torch.randn(4, 4, 64, 64)
    b_cheat = torch.randn(4, 67)

    # Logits
    logits = model_3c(b_glcm, b_cheat)
    assert logits.shape == (4, 3)

    # Probs
    probs = model_3c(b_glcm, b_cheat, return_probs=True)
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Binary model
    model_2c = Bonn1DGatedGLCMTransformer(num_classes=2, num_cheatsheet=67, d_model=32, nhead=4, num_layers=2)
    logits_2c = model_2c(b_glcm, b_cheat)
    assert logits_2c.shape == (4, 2)


def test_transformer_backward_pass():
    """Verify gradients propagate through Transformer layers, Cheatsheet MLP, Gating, and Residual head."""
    model = Bonn1DGatedGLCMTransformer(num_classes=3, num_cheatsheet=67, d_model=32, nhead=4, num_layers=2)
    b_glcm = torch.randn(2, 4, 64, 64)
    b_cheat = torch.randn(2, 67)

    logits = model(b_glcm, b_cheat)
    loss = logits.sum()
    loss.backward()

    assert model.token_proj.weight.grad is not None, "Token projection gradient missing"
    assert model.pos_emb.grad is not None, "Positional embedding gradient missing"
    assert model.cheat_fc.weight.grad is not None, "Cheatsheet FC gradient missing"
    assert model.gate_fc.weight.grad is not None, "Gating network gradient missing"
    assert model.fc_residual.weight.grad is not None, "Residual shortcut gradient missing"
    assert model.fc_out.weight.grad is not None, "Classifier head gradient missing"


def test_transformer_eval_mode_determinism():
    """Verify that in eval() mode, identical inputs yield identical outputs (dropout disabled)."""
    model = Bonn1DGatedGLCMTransformer(num_classes=3, num_cheatsheet=67, d_model=32)
    model.eval()

    b_glcm = torch.randn(2, 4, 64, 64)
    b_cheat = torch.randn(2, 67)

    with torch.no_grad():
        out1 = model(b_glcm, b_cheat)
        out2 = model(b_glcm, b_cheat)

    assert torch.allclose(out1, out2, atol=1e-6), "Eval mode non-deterministic!"


def test_robust_glcm_dataset():
    """Verify dataset indexing and tensor shapes."""
    records = [
        {"recording_id": "REC001", "label": 0},
        {"recording_id": "REC002", "label": 1},
    ]
    cached = {
        "REC001": torch.randn(4, 64, 64),
        "REC002": torch.randn(4, 64, 64),
    }
    cheat_mat = np.random.randn(2, 67).astype(np.float32)

    ds = RobustGLCMDataset(records, cached, cheat_mat)
    assert len(ds) == 2

    t_glcm, t_cheat, lbl, rid = ds[0]
    assert t_glcm.shape == (4, 64, 64)
    assert t_cheat.shape == (67,)
    assert lbl == 0
    assert rid == "REC001"
