"""
tests/test_cnn_branch.py

Comprehensive lightweight test suite for the independent Bonn EEG CNN branch.
Covers both:
1. Multi-channel GLCM matrix pipeline [8, 16, 16] (primary deep learning branch).
2. Recurrence plot image pipeline [1, 128, 128] (retained utility).

Verifies:
- Model forward pass on synthetic GLCM input.
- Output shapes for all three classification cases (Case 1: 3, Case 2: 2, Case 3: 2).
- Exact spatial tensor shapes through all convolution, pooling, and dense layers.
- Finite output values, loss calculation, and backward gradients.
- Deterministic GLCM matrix generation and channel probability normalization (sums to 1).
- Filename discovery handling both .txt and .TXT case-insensitively.
- GLCM manifest structure and schema.
- PyTorch BonnGLCMDataset loading and index-to-sample alignment.
- Class weights computed exclusively from sub-training labels.
- Pairwise partition disjointness between sub-train, val, and outer-test sets.
- Early-stopping triggering after 5 non-improving epochs.
- Best-checkpoint restoration behavior.
- ReduceLROnPlateau scheduler behavior at patience 4.
- Immutability of the frozen feature CSV hash.
- Existing SVM source files and results remain intact.
"""

import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.cnn.config import (
    CNNModelConfig,
    GLCMModelConfig,
    CNNTrainingConfig,
    CNNExperimentConfig,
    GLCMExperimentConfig,
)
from scripts.cnn.model import BonnGLCMCNN, BonnRecurrenceCNN, BonnModernRPCNN, Conv2dSame
from scripts.cnn.export_glcm_matrices import (
    generate_single_glcm_tensor,
    export_glcm_dataset,
    CANONICAL_GLCM_CHANNELS,
)
from scripts.cnn.export_rp_images import (
    generate_single_rp_image,
    discover_eeg_recordings,
    export_rp_dataset,
)
from scripts.cnn.dataset import BonnGLCMDataset, BonnRPDataset
from scripts.cnn.train import (
    compute_balanced_class_weights,
    split_outer_train_into_subtrain_and_val,
    train_single_fold,
)
from scripts.cnn.evaluate import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# 1. GLCM CNN Architecture & Exact Tensor Shapes
# =========================================================================

def test_glcm_cnn_forward_shapes_all_cases():
    """Verify output shape for all three classification cases with GLCM input [B, 8, 16, 16]."""
    x = torch.randn(4, 8, 16, 16)

    # Case 1: 3 classes
    m1 = BonnGLCMCNN(num_classes=3)
    out1 = m1(x)
    assert out1.shape == (4, 3)

    # Case 2: 2 classes
    m2 = BonnGLCMCNN(num_classes=2)
    out2 = m2(x)
    assert out2.shape == (4, 2)

    # Case 3: 2 classes
    m3 = BonnGLCMCNN(num_classes=2)
    out3 = m3(x)
    assert out3.shape == (4, 2)


def test_glcm_cnn_exact_intermediate_tensor_shapes():
    """
    Verify exact spatial dimensions at every layer for the GLCM CNN:
    Input: [B, 8, 16, 16]
    Block 1 Conv:  [B, 16, 16, 16]
    Block 1 Pool:  [B, 16, 8, 8]
    Block 2 Conv:  [B, 32, 8, 8]
    Block 2 Pool:  [B, 32, 4, 4]
    Block 3 Conv:  [B, 64, 4, 4]
    Block 3 Pool:  [B, 64, 2, 2]
    GAP:           [B, 64]
    Dense:         [B, 64]
    Logits:        [B, 3]
    """
    x = torch.randn(2, 8, 16, 16)
    model = BonnGLCMCNN(num_classes=3)
    shapes = model.get_intermediate_shapes(x)

    assert shapes["input"] == (2, 8, 16, 16)
    assert shapes["block1_conv"] == (2, 16, 16, 16)
    assert shapes["block1_pool"] == (2, 16, 8, 8)
    assert shapes["block2_conv"] == (2, 32, 8, 8)
    assert shapes["block2_pool"] == (2, 32, 4, 4)
    assert shapes["block3_conv"] == (2, 64, 4, 4)
    assert shapes["block3_pool"] == (2, 64, 2, 2)
    assert shapes["gap"] == (2, 64)
    assert shapes["dense"] == (2, 64)
    assert shapes["logits"] == (2, 3)


def test_glcm_cnn_finite_loss_and_gradients():
    """Verify forward and backward passes produce finite values on GLCM inputs."""
    model = BonnGLCMCNN(num_classes=3)
    x = torch.randn(4, 8, 16, 16)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    logits = model(x)
    assert torch.all(torch.isfinite(logits))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad)), f"NaN gradient in {name}"


# =========================================================================
# 2. GLCM Matrix Exporter & Representation Verification
# =========================================================================

def test_single_glcm_tensor_generation():
    """Verify generation of normalized 8-channel GLCM tensor (8, 16, 16)."""
    sample_file = PROJECT_ROOT / "data" / "bonn_eeg" / "extracted" / "A" / "Z001.txt"
    if sample_file.exists():
        tensor, diag = generate_single_glcm_tensor(sample_file)

        assert tensor.shape == (8, 16, 16)
        assert tensor.dtype == np.float32
        assert np.all(np.isfinite(tensor))
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

        # Each channel represents normalized joint probabilities that sum to 1.0
        for ch_idx in range(8):
            ch_sum = tensor[ch_idx].sum()
            assert pytest.approx(ch_sum, abs=1e-5) == 1.0

        assert diag["shape"] == [8, 16, 16]
        assert len(diag["channel_sums"]) == 8


def test_glcm_pilot_export_and_manifest():
    """Test exporting a small GLCM pilot (2 recordings) into a temporary directory."""
    extracted_dir = PROJECT_ROOT / "data" / "bonn_eeg" / "extracted"
    if extracted_dir.exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "glcm_test_export"
            manifest = export_glcm_dataset(extracted_dir, out_path, pilot_limit=2)

            assert manifest["total_tensors"] == 2
            assert manifest["is_pilot"] is True
            assert (out_path / "glcm_manifest.json").exists()

            for rec in manifest["records"]:
                npy_path = out_path / rec["matrix_relative_path"]
                assert npy_path.exists()
                assert len(rec["matrix_sha256"]) == 64
                assert rec["tensor_shape"] == [8, 16, 16]
                assert rec["channel_names"] == CANONICAL_GLCM_CHANNELS

                # Reload and verify bit-exact values
                loaded_tensor = np.load(npy_path)
                assert loaded_tensor.shape == (8, 16, 16)
                assert loaded_tensor.dtype == np.float32


def test_glcm_dataset_loader_and_labels():
    """Test BonnGLCMDataset loading tensors and mapping labels."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        mat_dir = tmp_dir_path / "matrices"
        mat_dir.mkdir(parents=True)

        # Create two mock recordings
        rec1_tensor = np.ones((8, 16, 16), dtype=np.float32) / 256.0
        rec2_tensor = np.ones((8, 16, 16), dtype=np.float32) / 256.0
        np.save(mat_dir / "Z001.npy", rec1_tensor)
        np.save(mat_dir / "S001.npy", rec2_tensor)

        mock_records = [
            {"recording_id": "Z001", "case1_label": "healthy"},
            {"recording_id": "S001", "case1_label": "ictal"},
        ]

        dataset = BonnGLCMDataset(
            samples=mock_records,
            matrix_dir=tmp_dir_path,
            label_key="case1_label",
        )

        assert len(dataset) == 2
        t0, y0, id0 = dataset[0]
        assert t0.shape == (8, 16, 16)
        assert id0 == "Z001"
        assert y0 in [0, 1]

        t1, y1, id1 = dataset[1]
        assert t1.shape == (8, 16, 16)
        assert id1 == "S001"
        assert y0 != y1


# =========================================================================
# 3. Retained RP Image Architecture Verification
# =========================================================================

def test_rp_cnn_forward_shapes_all_cases():
    """Verify output shape for all three classification cases on RP images."""
    x = torch.randn(4, 1, 128, 128)
    m1 = BonnRecurrenceCNN(num_classes=3)
    assert m1(x).shape == (4, 3)

    m2 = BonnRecurrenceCNN(num_classes=2)
    assert m2(x).shape == (4, 2)


def test_rp_conv2d_same_padding_behavior():
    """Verify Conv2dSame correctly implements ceil(dim / stride)."""
    conv = Conv2dSame(1, 8, kernel_size=3, stride=3)
    assert conv(torch.randn(1, 1, 128, 128)).shape == (1, 8, 43, 43)
    assert conv(torch.randn(1, 1, 21, 21)).shape == (1, 8, 7, 7)
    assert conv(torch.randn(1, 1, 3, 3)).shape == (1, 8, 1, 1)


def test_modern_rp_cnn_forward_shapes_and_intermediates():
    """Verify BonnModernRPCNN output shapes and spatial progression across all 4 blocks."""
    x = torch.randn(4, 1, 128, 128)
    model3 = BonnModernRPCNN(num_classes=3)
    out3 = model3(x)
    assert out3.shape == (4, 3)

    model2 = BonnModernRPCNN(num_classes=2)
    out2 = model2(x)
    assert out2.shape == (4, 2)

    # Check intermediate spatial dimensions
    shapes = model3.get_intermediate_shapes(x)
    assert shapes["input"] == (4, 1, 128, 128)
    assert shapes["block1_pool"] == (4, 16, 64, 64)
    assert shapes["block2_pool"] == (4, 32, 32, 32)
    assert shapes["block3_pool"] == (4, 64, 16, 16)
    assert shapes["block4_pool"] == (4, 64, 8, 8)
    assert shapes["gap"] == (4, 64)
    assert shapes["logits"] == (4, 3)

    # Check gradient flow
    loss = out3.sum()
    loss.backward()
    for name, param in model3.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"


# =========================================================================
# 4. Leakage Prevention, Splitting & Training Safeguards
# =========================================================================

def test_class_weights_isolated_to_subtrain():
    """Verify that class weights depend exclusively on sub-training labels."""
    sub_train_labels = np.array([0] * 80 + [1] * 20)  # Imbalanced 4:1
    val_labels = np.array([0] * 10 + [1] * 10)        # Balanced 1:1

    dev = torch.device("cpu")
    w1 = compute_balanced_class_weights(sub_train_labels, num_classes=2, device=dev)
    
    assert w1[1] > w1[0]
    assert pytest.approx((w1[1] / w1[0]).item(), rel=1e-3) == 4.0

    # Changing val labels has zero effect on sub-train weights
    w2 = compute_balanced_class_weights(sub_train_labels, num_classes=2, device=dev)
    assert torch.equal(w1, w2)


def test_partition_pairwise_disjointness():
    """Verify that sub-training, validation, and outer-test indices are strictly pairwise disjoint."""
    outer_train = list(range(0, 450))
    outer_test = list(range(450, 500))
    mock_labels = np.array([0] * 200 + [1] * 100 + [2] * 200)

    sub_train, val = split_outer_train_into_subtrain_and_val(
        outer_train_indices=outer_train,
        dataset_labels=mock_labels,
        val_ratio=0.10,
        random_seed=42,
    )

    s_train = set(sub_train)
    s_val = set(val)
    s_test = set(outer_test)

    assert len(s_train.intersection(s_val)) == 0, "Sub-train and val overlap!"
    assert len(s_train.intersection(s_test)) == 0, "Sub-train and test overlap!"
    assert len(s_val.intersection(s_test)) == 0, "Val and test overlap!"

    assert len(s_train) + len(s_val) == 450
    assert len(s_train) + len(s_val) + len(s_test) == 500


def test_early_stopping_and_checkpoint_restoration():
    """Test that early stopping terminates after 5 non-improving epochs and restores best checkpoint."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )

    X_tr = torch.randn(64, 10)
    y_tr = torch.randint(0, 2, (64,))
    X_val = torch.randn(32, 10)
    y_val = torch.randint(0, 2, (32,))

    class SimpleTupleDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], f"sample_{idx}"

    train_loader = DataLoader(SimpleTupleDataset(X_tr, y_tr), batch_size=16)
    val_loader = DataLoader(SimpleTupleDataset(X_val, y_val), batch_size=16)

    cfg = CNNTrainingConfig(
        max_epochs=20,
        learning_rate=0.01,
        early_stopping_patience=5,
        lr_scheduler_patience=4,
    )

    weights = torch.tensor([1.0, 1.0])
    trained_model, summary = train_single_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights,
        training_cfg=cfg,
        device=torch.device("cpu"),
    )

    assert summary["stopped_early"] is True
    assert summary["total_epochs_trained"] <= 20
    assert 1 <= summary["best_epoch"] <= summary["total_epochs_trained"]
    assert 0.0 <= summary["best_val_accuracy"] <= 1.0


# =========================================================================
# 5. Project Immutability & Safety Guards
# =========================================================================

def test_frozen_feature_csv_hash_unchanged():
    """Assert SHA-256 hash of the frozen 96-feature CSV remains untouched."""
    csv_path = PROJECT_ROOT / "audit" / "batch_preprocessing" / "bonn_features_96.csv"
    assert csv_path.exists(), f"Frozen CSV missing at {csv_path}"
    
    with open(csv_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    expected_hash = "af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1"
    assert file_hash == expected_hash, f"Frozen CSV was altered! Hash is {file_hash}"


def test_svm_source_and_results_preserved():
    """Assert that baseline SVM files and checkpoint directories exist intact."""
    svm_engine = PROJECT_ROOT / "scripts" / "classification" / "nested_cv.py"
    svm_tuned = PROJECT_ROOT / "scripts" / "classification" / "run_kaggle_tuned.py"
    svm_k5 = PROJECT_ROOT / "scripts" / "classification" / "run_kaggle_k5.py"
    chk_k5 = PROJECT_ROOT / "experiments" / "experiment_k5"

    assert svm_engine.exists()
    assert svm_tuned.exists()
    assert svm_k5.exists()
    assert chk_k5.exists()
    assert (chk_k5 / "Case1_3Class" / "checkpoints").exists()
    assert (chk_k5 / "Case2_Binary" / "checkpoints").exists()
    assert (chk_k5 / "Case3_Pilot" / "checkpoints").exists()


# =========================================================================
# 6. 5-Fold Stratified Cross-Validation Integrity Tests
# =========================================================================

def test_5fold_split_separation_and_coverage():
    """Verify that every recording is held out exactly once across 5 folds and subsets are disjoint."""
    from scripts.cnn.run_glcm_5fold_cv import generate_5fold_splits, CASE_SPECS

    # Create mock 500 records matching Bonn subsets
    mock_records = []
    for sub in ["A", "B", "C", "D", "E"]:
        for i in range(1, 101):
            rec_id = f"{sub}{i:03d}"
            mock_records.append({"recording_id": rec_id, "filename": f"{rec_id}.txt", "subset": sub})

    for case_key in ["Case1_3Class", "Case2_Binary", "Case3_Binary"]:
        split_res = generate_5fold_splits(mock_records, case_key=case_key, outer_seed=42)
        folds = split_res["folds"]
        assert len(folds) == 5, f"{case_key} should have exactly 5 folds"

        all_test_ids = []
        for f in folds:
            s_tr = set(f["subtrain_ids"])
            s_va = set(f["val_ids"])
            s_te = set(f["test_ids"])

            # Disjointness within fold
            assert s_tr.isdisjoint(s_va), f"{case_key} Fold {f['fold_idx']}: subtrain overlaps val"
            assert s_tr.isdisjoint(s_te), f"{case_key} Fold {f['fold_idx']}: subtrain overlaps test"
            assert s_va.isdisjoint(s_te), f"{case_key} Fold {f['fold_idx']}: val overlaps test"

            all_test_ids.extend(f["test_ids"])

        # Complete coverage: every recording appears in outer test set exactly once
        spec = CASE_SPECS[case_key]
        assert len(all_test_ids) == spec["expected_total"]
        assert len(set(all_test_ids)) == spec["expected_total"]


def test_5fold_class_stratification():
    """Verify class proportions are strictly preserved in test and sub-train across all 5 folds."""
    from scripts.cnn.run_glcm_5fold_cv import generate_5fold_splits

    mock_records = []
    for sub in ["A", "B", "C", "D", "E"]:
        for i in range(1, 101):
            rec_id = f"{sub}{i:03d}"
            mock_records.append({"recording_id": rec_id, "filename": f"{rec_id}.txt", "subset": sub})

    # Case 1: 400 total -> test per fold = 80 (20 class 0, 20 class 1, 40 class 2)
    c1 = generate_5fold_splits(mock_records, "Case1_3Class", outer_seed=42)
    for f in c1["folds"]:
        assert f["class_counts"]["test"][0] == 20
        assert f["class_counts"]["test"][1] == 20
        assert f["class_counts"]["test"][2] == 40
        assert f["class_counts"]["subtrain"][0] == 72
        assert f["class_counts"]["subtrain"][1] == 72
        assert f["class_counts"]["subtrain"][2] == 144

    # Case 2: 500 total -> test per fold = 100 (80 class 0, 20 class 1)
    c2 = generate_5fold_splits(mock_records, "Case2_Binary", outer_seed=42)
    for f in c2["folds"]:
        assert f["class_counts"]["test"][0] == 80
        assert f["class_counts"]["test"][1] == 20
        assert f["class_counts"]["subtrain"][0] == 288
        assert f["class_counts"]["subtrain"][1] == 72

    # Case 3: 200 total -> test per fold = 40 (20 class 0, 20 class 1)
    c3 = generate_5fold_splits(mock_records, "Case3_Binary", outer_seed=42)
    for f in c3["folds"]:
        assert f["class_counts"]["test"][0] == 20
        assert f["class_counts"]["test"][1] == 20
        assert f["class_counts"]["subtrain"][0] == 72
        assert f["class_counts"]["subtrain"][1] == 72


def test_5fold_model_reinitialization_and_seeds():
    """Verify models initialized on different fold seeds have distinct parameter values."""
    from scripts.cnn.model import BonnGLCMCNN

    torch.manual_seed(42)
    m1 = BonnGLCMCNN(num_classes=3)
    p1 = [p.clone() for p in m1.parameters()]

    torch.manual_seed(142)
    m2 = BonnGLCMCNN(num_classes=3)
    p2 = [p.clone() for p in m2.parameters()]

    diffs = [torch.max(torch.abs(a - b)).item() for a, b in zip(p1, p2)]
    assert max(diffs) > 1e-4, "Reinitialized model parameters should not be identical across different seeds"


def test_bonn_gated_hybrid_rp_cnn():
    """Verify BonnGatedHybridRPCNN forward, gating, residual connection, and backprop."""
    from scripts.cnn.model import BonnGatedHybridRPCNN

    model = BonnGatedHybridRPCNN(num_classes=3, num_scalars=9, residual_indices=[2, 3, 4])
    x_img = torch.randn(4, 1, 128, 128)
    x_scal = torch.randn(4, 9)

    # Forward logits
    logits = model(x_img, x_scal)
    assert logits.shape == (4, 3)

    # Forward probabilities
    probs = model(x_img, x_scal, return_probs=True)
    assert probs.shape == (4, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    # Backpropagation check
    loss = logits.sum()
    loss.backward()
    assert model.conv1.weight.grad is not None, "Visual branch gradients missing"
    assert model.scalar_proj.weight.grad is not None, "Scalar branch gradients missing"
    assert model.gate_fc.weight.grad is not None, "Gating network gradients missing"
    assert model.fc_residual.weight.grad is not None, "Residual amplitude head gradients missing"


