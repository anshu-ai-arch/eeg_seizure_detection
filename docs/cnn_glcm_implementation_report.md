# Bonn EEG Research Project — GLCM-Based CNN Implementation Report

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** IMPLEMENTATION COMPLETE & VERIFIED — PENDING PILOT APPROVAL  
**Branch:** `fix/bonn-replication-audit`  
**Repository:** `anshu-ai-arch/eeg`

---

## 1. Executive Summary

In response to the revised requirement, the independent Convolutional Neural Network (CNN) branch has been successfully converted to receive **actual multi-channel GLCM matrices as input** (`[batch_size, 8, 16, 16]`), replacing the recurrence plot image representation and avoiding any reliance on the 96-dimensional scalar Haralick feature vector.

### Core Integrity Guarantees
1. **Zero Impact on Baseline Pipeline:** The existing GLCM + MCFS + RBF-SVM replication pipeline, frozen feature CSV, and all historical nested-CV fold checkpoints remain 100% untouched.
2. **Frozen Feature CSV Hash Immutability:** SHA-256 of `audit/batch_preprocessing/bonn_features_96.csv`:
   - **Before Implementation:** `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`
   - **After Implementation:**  `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1` (100% verified match).
3. **SVM Checkpoints Intact:** Checkpoints in `experiments/experiment_k5/` for all three benchmark cases (`Case1_3Class`, `Case2_Binary`, `Case3_Pilot`) remain intact.
4. **No Full Run Executed:** Neither full-dataset GLCM matrix export (500 recordings) nor full 10-fold nested-CV training was executed.

---

## 2. Audit of the Existing GLCM Implementation

The full preprocessing and feature extraction chain was traced through actual source code:

$$\text{EEG Recording} \xrightarrow{\text{Z-score}} \text{Norm Signal} \xrightarrow{\text{AMI/FNN}} (\tau, m) \xrightarrow{\text{Takens}} \text{Trajectory} \xrightarrow{\text{Euclidean}} \text{GRP} \xrightarrow{\text{Quantize}} \text{Quantized GRP} \xrightarrow{\text{GLCM}} 8 \times (16 \times 16) \xrightarrow{\text{Haralick}} 96 \text{ features}$$

### Verified GLCM Properties:
* **Constructing Function:** `scripts/preprocessing/glcm.py:L30-L95` (`compute_single_glcm`) and `scripts/preprocessing/glcm.py:L192-L224` (`extract_all_glcm_features`).
* **Haralick Extraction Function:** `scripts/preprocessing/glcm.py:L97-L190` (`extract_haralick_12`).
* **Number of Matrices per Recording:** Exactly **8 matrices**.
* **Matrix Spatial Dimensions:** Exactly **$16 \times 16$** (quantization levels $K=16$, integer bins $0..15$).
* **Orientations ($\theta$) and Offsets ($d$):**
  - Orientations: $\theta \in \{0.0, \pi/4\}$ radians ($0^\circ$ and $45^\circ$).
  - Offsets: $d \in \{1, 2, 3, 4\}$.
* **Symmetry & Normalization:**
  - **Symmetric:** Pairs accumulated symmetrically ($P = P + P^T$, `glcm.py:L83`).
  - **Normalized:** Normalized by sum ($p = P / \sum P$), yielding valid 2D probability density distributions where $\sum_{i,j} p(i, j) = 1.0$.
* **Data Type & Value Range:** `float64` in preprocessing, converted to `float32` for PyTorch tensors. Values are non-negative probabilities strictly in $[0.0, 1.0]$.
* **96 Features Composition:** Exactly 8 GLCM matrices $\times$ 12 Haralick texture features ($8 \times 12 = 96$).
* **Storage Status:** In the baseline batch preprocessing (`scripts/run_batch_preprocessing.py`), the 8 matrices were created in memory, used to extract the 96 features, and discarded without being saved to disk.

---

## 3. CNN Input Representation: Multi-Channel GLCM Tensor

The CNN receives the 8 actual normalized GLCM probability matrices stacked into a multi-channel tensor:

$$\text{Input Tensor Shape:} \quad [8, 16, 16]$$
$$\text{Batch Shape:} \quad [\text{batch\_size}, 8, 16, 16]$$

### Canonical Channel Ordering:
| Channel Index | Configuration Key | Orientation ($\theta$) | Distance Offset ($d$) | Interpretation |
|---|---|---|---|---|
| **0** | `theta_0deg_d_1` | $0^\circ$ (horizontal) | $d=1$ | Nearest horizontal co-occurrences |
| **1** | `theta_0deg_d_2` | $0^\circ$ (horizontal) | $d=2$ | Distance-2 horizontal co-occurrences |
| **2** | `theta_0deg_d_3` | $0^\circ$ (horizontal) | $d=3$ | Distance-3 horizontal co-occurrences |
| **3** | `theta_0deg_d_4` | $0^\circ$ (horizontal) | $d=4$ | Distance-4 horizontal co-occurrences |
| **4** | `theta_45deg_d_1` | $45^\circ$ (diagonal) | $d=1$ | Nearest diagonal co-occurrences |
| **5** | `theta_45deg_d_2` | $45^\circ$ (diagonal) | $d=2$ | Distance-2 diagonal co-occurrences |
| **6** | `theta_45deg_d_3` | $45^\circ$ (diagonal) | $d=3$ | Distance-3 diagonal co-occurrences |
| **7** | `theta_45deg_d_4` | $45^\circ$ (diagonal) | $d=4$ | Distance-4 diagonal co-occurrences |

---

## 4. Adapted CNN Architecture & Spatial Dimensions

### Technical Justification for Stride Adaptation
The previous recurrence plot architecture used convolution stride 3 on $128 \times 128$ images. Applying convolution stride 3 to a small $16 \times 16$ GLCM matrix causes immediate spatial collapse:
* Block 1 Conv (stride 3): $\lceil 16 / 3 \rceil = 6 \to$ MaxPool ($2 \times 2$): $3 \times 3$.
* Block 2 Conv (stride 3): $\lceil 3 / 3 \rceil = 1 \to$ MaxPool ($2 \times 2$): **$\mathbf{0 \times 0}$ (COLLAPSE & CRASH)**.

### Adapted Architecture (`BonnGLCMCNN` in `scripts/cnn/model.py`)
To preserve the 3-block convolutional design without spatial collapse or artificial matrix resizing, the convolutions adapt to standard **`stride=1, padding=1`**:

| Block & Layer | Filters / Pool | Kernel | Stride | Padding | Output Tensor Shape (Channels-First) |
|---|---|---|---|---|---|
| **Input** | — | — | — | — | `[B, 8, 16, 16]` |
| **Block 1: Conv2D** | 16 | $3 \times 3$ | **1** | 1 (same) | `[B, 16, 16, 16]` |
| **Block 1: BN + ReLU** | 16 | — | — | — | `[B, 16, 16, 16]` |
| **Block 1: MaxPool2D** | 16 | $2 \times 2$ | **2** | valid | `[B, 16, 8, 8]` |
| **Block 2: Conv2D** | 32 | $3 \times 3$ | **1** | 1 (same) | `[B, 32, 8, 8]` |
| **Block 2: BN + ReLU** | 32 | — | — | — | `[B, 32, 8, 8]` |
| **Block 2: MaxPool2D** | 32 | $2 \times 2$ | **2** | valid | `[B, 32, 4, 4]` |
| **Block 3: Conv2D** | 64 | $3 \times 3$ | **1** | 1 (same) | `[B, 64, 4, 4]` |
| **Block 3: BN + ReLU** | 64 | — | — | — | `[B, 64, 4, 4]` |
| **Block 3: MaxPool2D** | 64 | $2 \times 2$ | **2** | valid | `[B, 64, 2, 2]` |
| **Global Avg Pooling** | — | — | — | — | `[B, 64]` |
| **Dense (64, ReLU)** | 64 | — | — | — | `[B, 64]` |
| **Dropout (0.4)** | — | — | — | — | `[B, 64]` |
| **Dense Output (Logits)** | $C$ | — | — | — | `[B, num_classes]` (Case 1: 3, Case 2: 2, Case 3: 2) |

---

## 5. Fixed Training Hyperparameters

All required training hyperparameters remain exactly as specified:
* **Optimizer:** AdamW (`learning_rate=0.005`, `weight_decay=0.0001`).
* **Batch Size:** 16.
* **Maximum Epochs:** 50.
* **Early Stopping:** Patience $= 5$ epochs, strictly improving validation accuracy ($\Delta = 0.0$, ties are not improvements), best checkpoint restored.
* **LR Scheduler:** `ReduceLROnPlateau` (mode='max', factor $= 0.5$, patience $= 4$ epochs, `min_lr=1e-6`).
* **Loss Function:** Weighted Cross-Entropy with balanced class weights computed **strictly from the sub-training fold**.
* **Random Seed:** 42.
* **Augmentation:** Disabled.

---

## 6. Files Created and Modified

All changes are strictly confined to the isolated deep learning branch:

| File Path | Status | Purpose |
|---|---|---|
| `scripts/cnn/export_glcm_matrices.py` | **New** | Deterministic exporter creating multi-channel `.npy` files `[8, 16, 16]` and JSON manifest. |
| `scripts/cnn/config.py` | **Modified** | Added `GLCMModelConfig` (`in_channels=8`, `input_size=(16, 16)`, `conv_stride=1`) and `GLCMExperimentConfig`. |
| `scripts/cnn/model.py` | **Modified** | Implemented `BonnGLCMCNN` adapted to `[B, 8, 16, 16]`; retained `BonnRecurrenceCNN`. |
| `scripts/cnn/dataset.py` | **Modified** | Implemented `BonnGLCMDataset` loading `.npy` matrix tensors; retained `BonnRPDataset`. |
| `scripts/cnn/train.py` | **Modified** | Added `run_nested_cv_glcm_case` supporting 10-fold nested CV outer fold execution for GLCM tensors. |
| `tests/test_cnn_branch.py` | **Modified** | Added unit tests for GLCM shapes, tensor generation, channel sums, pilot export, and dataset loading. |
| `docs/cnn_glcm_implementation_report.md` | **New** | This verification report. |

---

## 7. Test Results and Verification

All 13 unit tests in `tests/test_cnn_branch.py` and all 74 tests across the entire repository passed cleanly under `/opt/anaconda3/bin/pytest`:

```bash
/opt/anaconda3/bin/pytest tests/test_cnn_branch.py -v
```
**Results:** `13 passed in 6.45s`

```bash
/opt/anaconda3/bin/pytest tests/ -v
```
**Results:** `74 passed in 9.46s (100% passing across entire workspace)`

### Specific Verifications Passed:
* `test_glcm_cnn_forward_shapes_all_cases`: Output shapes verified for Case 1 (`[B, 3]`), Case 2 (`[B, 2]`), Case 3 (`[B, 2]`).
* `test_glcm_cnn_exact_intermediate_tensor_shapes`: Traced all layer shapes (`[B, 8, 16, 16] -> [B, 16, 8, 8] -> [B, 32, 4, 4] -> [B, 64, 2, 2] -> [B, 64] -> [B, C]`).
* `test_glcm_cnn_finite_loss_and_gradients`: Finite loss and gradients verified with backpropagation.
* `test_single_glcm_tensor_generation`: Verified on `Z001.txt` that each channel sums to 1.0 within tolerance and values are non-negative probabilities.
* `test_glcm_pilot_export_and_manifest`: Verified export of 2 pilot recordings into `.npy` tensors with valid SHA-256 hashes in manifest.
* `test_glcm_dataset_loader_and_labels`: Tested `BonnGLCMDataset` loading and label assignment.
* `test_class_weights_isolated_to_subtrain`: Class weights strictly derived from sub-training split only.
* `test_partition_pairwise_disjointness`: Zero overlap between sub-training, validation, and outer-test partitions.
* `test_early_stopping_and_checkpoint_restoration`: Restores best epoch checkpoint.
* `test_frozen_feature_csv_hash_unchanged`: Confirmed `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`.
* `test_svm_source_and_results_preserved`: Verified baseline SVM files and checkpoints exist intact.

---

## 8. Command for the Next Approved Small Pilot Stage

To export the first 5 recordings into GLCM matrix tensors for inspection before approving the full 500-recording run, execute:

```bash
python scripts/cnn/export_glcm_matrices.py \
    --input data/bonn_eeg/extracted \
    --output data/processed/glcm_matrices_cnn_pilot \
    --pilot 5
```

---

**Execution is halted here as instructed. Please let me know when you are ready to approve the 5-recording GLCM pilot export.**
