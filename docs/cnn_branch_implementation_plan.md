# Research-Grade Implementation Plan: Independent CNN Branch for Bonn EEG Recurrence Plots

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** PLANNING ONLY — PENDING USER APPROVAL  
**Repository:** `anshu-ai-arch/eeg`  
**Target Branch:** `fix/bonn-replication-audit` (or dedicated branch `feat/cnn-rp-branch`)

---

## 1. Executive Summary

This document establishes a rigorous, step-by-step implementation plan for adding an independent Convolutional Neural Network (CNN) classification branch to the Bonn EEG research project. The CNN directly classifies continuous Generalized Recurrence Plots (GRPs) resized to $128 \times 128 \times 1$, providing a deep learning counterpart to the existing classical paper-replication pipeline (Z-score normalization $\to$ AMI time-delay $\to$ Kennel FNN dimension $\to$ Takens trajectory $\to$ continuous GRP $\to$ 16-level quantization $\to$ 96 Haralick GLCM features $\to$ supervised MCFS $\to$ RBF-SVM).

### Key Audit Findings & Direct Answers
1. **Pipeline Independence:** The existing paper-replication pipeline, frozen feature CSV (`audit/batch_preprocessing/bonn_features_96.csv`), and all previous nested-CV SVM checkpoints/results (`experiments/experiment_k5/`, `audit/classification/`) will remain 100% untouched and unaffected.
2. **RP Image Storage:** The existing pipeline **does not save RP images to disk** (confirmed in `scripts/run_batch_preprocessing.py:L19`: *"Intermediate distance matrices are discarded immediately; never persisted to disk"*). However, because preprocessing is 100% deterministic, continuous GRPs can be exported into a lightweight, versioned image directory ($128 \times 128 \times 1$, float32/PNG) using the existing preprocessing functions without altering baseline feature calculations.
3. **Exact Split Reproducibility:** Stratified 10-fold outer cross-validation splits stored in `experiments/experiment_k5/*/checkpoints/fold_*.json` were verified against `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` and **matched 100% (10/10 folds across Case 1, Case 2, and Case 3)**. The CNN branch can reuse these exact outer train/test splits.
4. **CRITICAL ARCHITECTURAL BLOCKER IDENTIFIED:**  
   The requested CNN configuration specifies three blocks with **convolution stride 3 and pooling stride 2**. Mathematical derivation proves that applying Conv2D stride 3 followed by MaxPooling2D stride 2 reduces the spatial dimension from $128 \times 128 \to 43 \times 43 \to 21 \times 21 \to 7 \times 7 \to 3 \times 3 \to 1 \times 1$ after Block 3 Conv2D. **Applying Block 3 MaxPooling2D (pool size $2 \times 2$, stride 2) on a $1 \times 1$ tensor causes spatial collapse to $0 \times 0$**, which triggers fatal runtime exceptions in both PyTorch (`RuntimeError: Calculated output size: (64x0x0). Output size is too small`) and TensorFlow/Keras (`ValueError: One of the dimensions in the output is <= 0`). Furthermore, PyTorch does not support `padding='same'` for strided convolutions. This issue is flagged prominently in Section 4 with concrete resolution options for user decision.

---

## 2. Comprehensive Repository Audit

### A. Recurrence Plot Generation
* **Generating Functions:**
  - `scripts/preprocessing/normalization.py` $\to$ `zscore_normalize(signal, ddof=0)`
  - `scripts/preprocessing/embedding.py` $\to$ `compute_ami(signal, max_lag=50, n_bins=32, fallback_tau=10)`
  - `scripts/preprocessing/embedding.py` $\to$ `compute_fnn(signal, tau, max_dim=10, r_tol=15.0, a_tol=2.0, theiler_w=tau, threshold=0.01)`
  - `scripts/preprocessing/grp.py` $\to$ `reconstruct_phase_space(signal, tau, m)` produces trajectory matrix $(N_t, m)$ where $N_t = 4097 - (m - 1)\tau$.
  - `scripts/preprocessing/grp.py` $\to$ `compute_grp(trajectory, metric="euclidean")` computes pairwise Euclidean distance matrix $\text{GRP}(i, j) = \|X(i) - X(j)\|_2$.
* **Representation & Data Type:**
  - `compute_grp` returns a continuous 2D symmetric float32 array of shape $(N_t, N_t)$, with values in $[0, \max(\text{GRP})]$ (e.g., for recording `Z001`: $N_t = 4061$, shape $(4061, 4061)$, $\min=0.0000, \max=9.8614, \text{mean}=2.9892$).
* **Quantization Decoupling:**
  - The continuous distance matrix $\text{GRP}$ is computed **before** `quantize_grp` is called. For the CNN branch, $\text{GRP}_{\text{norm}} = \text{GRP} / \max(\text{GRP})$ maps continuous values directly to $[0, 1]$ float32.
* **Disk Persistence Status:**
  - Neither raw continuous distance matrices nor quantized matrices are persisted to disk. Batch preprocessing discards them immediately after extracting 96 scalar GLCM features to conserve RAM.
* **Minimal Safe Export Solution:**
  - Create an isolated utility `scripts/cnn/export_rp_images.py` that calls `zscore_normalize`, `compute_ami`, `compute_fnn`, `reconstruct_phase_space`, and `compute_grp`, scales by $\max(\text{GRP})$, resizes to $128 \times 128$ using bilinear interpolation, and saves the output to `data/processed/rp_images_128/` (as `.npy` float32 or single-channel 16-bit/8-bit PNG). This requires zero changes to `scripts/preprocessing/`.

### B. Dataset & Recording Identity
* **Recordings Directory:** `data/bonn_eeg/extracted/` contains subsets `A/`, `B/`, `C/`, `D/`, `E/` with exactly 100 recordings each.
* **Filename Inconsistency Discovered:**
  - Subsets `A`, `B`, `D`, `E` use lowercase `.txt` (e.g., `Z001.txt`, `O001.txt`, `F001.txt`, `S001.txt`).
  - **Subset `C` uses uppercase `.TXT` (`N001.TXT` through `N100.TXT`)**.
  - All file-discovery functions must use `f.suffix.lower() == ".txt"` to prevent missing subset C.
* **Sample Count:** All 500 files contain exactly 4,097 integer ASCII samples at 173.61 Hz.
* **Recording Identifiers & Prefixes:**
  - Subset A: `Z001` – `Z100` (Surface EEG, Healthy, Eyes Open)
  - Subset B: `O001` – `O100` (Surface EEG, Healthy, Eyes Closed)
  - Subset C: `N001` – `N100` (Intracranial, Interictal, Hippocampal Formation)
  - Subset D: `F001` – `F100` (Intracranial, Interictal, Epileptogenic Zone)
  - Subset E: `S001` – `S100` (Intracranial, Ictal Seizure)
* **Alignment with Frozen Feature CSV:**
  - `audit/batch_preprocessing/bonn_features_96.csv` contains 500 rows. The primary key `recording_id` maps 1:1 with file stems.

### C. Existing Cross-Validation Verification
* **Integrity of Outer Splits:**
  - Evaluated `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` across all 3 benchmark cases.
  - Verified against stored checkpoints in `experiments/experiment_k5/`:
    - **Case 1 (3-class):** 10/10 outer folds match saved `outer_test_sample_ids` exactly.
    - **Case 2 (Binary):** 10/10 outer folds match saved `outer_test_sample_ids` exactly.
    - **Case 3 (Binary Pilot):** 10/10 outer folds match saved `outer_test_sample_ids` exactly.
  - The CNN evaluation can guarantee 100% split alignment with the RBF-SVM baseline.

### D. Environment & Dependencies
* **Local Python Environment (`/opt/anaconda3/bin/python`):**
  - `torch`: 2.12.0 (Installed)
  - `torchvision`: 0.27.0 (Installed)
  - `PIL` (Pillow): 12.0.0 (Installed)
  - `scikit-learn`: 1.5.0+ (Installed)
  - `pandas`: 2.2.3 (Installed)
  - `numpy`: 2.3.5 (Installed)
  - `tensorflow` / `keras`: **NOT installed** locally.
* **Kaggle Environment:**
  - Both PyTorch and TensorFlow/Keras are pre-installed in Kaggle GPU environments.
  - PyTorch 2.x is recommended as the primary backend to maintain seamless compatibility between local verification tests and Kaggle execution.

### E. Repository Safety Status
* Git branch: `fix/bonn-replication-audit` (clean working tree except untracked `paper.pdf`).
* Commit: `203d387ab458e857d34cc1131bf80fbe18dd3c9b`.
* Remote: `origin` at `https://github.com/anshu-ai-arch/eeg.git`.

---

## 3. Fixed CNN Configuration & Training Specifications

| Component | Setting / Hyperparameter | Notes & Constraints |
|---|---|---|
| **Input Data** | Continuous GRP before GLCM quantization | Scaled $\text{GRP} / \max(\text{GRP}) \in [0, 1]$ |
| **Input Resolution** | $128 \times 128 \times 1$ | Bilinear interpolation |
| **Augmentation** | None | No rotations, flips, or jitter initially |
| **Random Seed** | `42` | Set for NumPy, Python random, and PyTorch/TF |
| **CNN Backbone** | 3 Convolutional Blocks | Conv2D + BatchNorm + ReLU + MaxPool2D |
| **Classification Head** | GAP + Dense(64, ReLU) + Dropout(0.4) + Dense(C, Softmax) | Output units: Case 1 = 3, Case 2 = 2, Case 3 = 2 |
| **Optimizer** | AdamW | Initial learning rate $= 0.005$, weight decay $= 0.0001$ |
| **Batch Size** | 16 | Fixed batch size |
| **Max Epochs** | 50 | Early stopping may terminate sooner |
| **Early Stopping** | Patience $= 5$ epochs | Monitor: Validation Accuracy (higher is better) |
| **LR Scheduler** | ReduceLROnPlateau | Factor $= 0.5$, Patience $= 4$ epochs, Min LR $= 10^{-6}$ |
| **Loss Function** | Weighted Cross-Entropy | Class weights computed from training fold only |

---

## 4. Exact Tensor Shape Derivation & Architectural Blocker Analysis

### Layer-by-Layer Spatial Dimension Calculations
Standard convolutional arithmetic dictates:
$$\text{Output Size (same)} = \left\lceil \frac{\text{Input Size}}{\text{Stride}} \right\rceil, \quad \text{Output Size (valid)} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size}}{\text{Stride}} \right\rfloor + 1$$

Tracing an input tensor of spatial dimensions $128 \times 128 \times 1$:

| Layer / Operation | Filters / Pool | Kernel | Stride | Padding | Expected Tensor Shape (Channels-First) | Calculation / Derivation |
|---|---|---|---|---|---|---|
| **Input** | — | — | — | — | `[B, 1, 128, 128]` | Target RP input |
| **Block 1: Conv2D** | 16 | $3 \times 3$ | **3** | same | `[B, 16, 43, 43]` | $\lceil 128 / 3 \rceil = 43$ |
| **Block 1: BatchNorm + ReLU** | 16 | — | — | — | `[B, 16, 43, 43]` | Elementwise |
| **Block 1: MaxPool2D** | 16 | $2 \times 2$ | **2** | valid | `[B, 16, 21, 21]` | $\lfloor (43 - 2)/2 \rfloor + 1 = 21$ |
| **Block 2: Conv2D** | 32 | $3 \times 3$ | **3** | same | `[B, 32, 7, 7]` | $\lceil 21 / 3 \rceil = 7$ |
| **Block 2: BatchNorm + ReLU** | 32 | — | — | — | `[B, 32, 7, 7]` | Elementwise |
| **Block 2: MaxPool2D** | 32 | $2 \times 2$ | **2** | valid | `[B, 32, 3, 3]` | $\lfloor (7 - 2)/2 \rfloor + 1 = 3$ |
| **Block 3: Conv2D** | 64 | $3 \times 3$ | **3** | same | `[B, 64, 1, 1]` | $\lceil 3 / 3 \rceil = 1$ |
| **Block 3: BatchNorm + ReLU** | 64 | — | — | — | `[B, 64, 1, 1]` | Elementwise |
| **Block 3: MaxPool2D** | 64 | $2 \times 2$ | **2** | valid | **`[B, 64, 0, 0]` (CRASH)** | $\lfloor (1 - 2)/2 \rfloor + 1 = 0$ |
| **Global Average Pooling** | — | — | — | — | Undefined | Cannot pool over $0 \times 0$ grid |
| **Dense (64, ReLU)** | 64 | — | — | — | `[B, 64]` | Fully connected |
| **Dropout (0.4)** | — | — | — | — | `[B, 64]` | Regularization |
| **Dense + Softmax** | $C$ | — | — | — | `[B, C]` | $C \in \{3, 2, 2\}$ |

---

### Critical Findings & Required User Decisions

#### Finding 1: Block 3 MaxPool2D Collapses to $0 \times 0$
Because convolution stride is 3, Block 3 Conv2D downsamples the $3 \times 3$ feature map directly to $1 \times 1$. A $2 \times 2$ MaxPool with stride 2 cannot operate on a $1 \times 1$ window:
* **PyTorch Crash:** `RuntimeError: Given input size: (64x1x1). Calculated output size: (64x0x0). Output size is too small`
* **TensorFlow/Keras Crash:** `ValueError: One of the dimensions in the output is <= 0 due to downsampling in max_pooling2d`

**Resolution Options (Requires User Selection):**
* **Option A (Preserve 3-Block Stride 3, Omit Block 3 Pooling):**  
  Keep Conv2D stride 3 for all 3 blocks, but omit `MaxPooling2D` in Block 3. Block 3 outputs `[B, 64, 1, 1]`, which enters Global Average Pooling cleanly as `[B, 64]`.
* **Option B (Standard ConvNet Strides: Conv Stride 1, Pool Stride 2):**  
  Use standard `stride=1` for Conv2D in all blocks and let MaxPooling2D handle downsampling ($128 \to 64 \to 32 \to 16$). Block 3 outputs $16 \times 16 \times 64$, pooled by GAP to $64$.
* **Option C (Pool with Padding='same'):**  
  If Block 3 MaxPool uses `padding='same'`, $\lceil 1 / 2 \rceil = 1$, preserving $1 \times 1 \times 64$.

#### Finding 2: PyTorch Does Not Support `padding='same'` for Strided Convolutions
In PyTorch, `nn.Conv2d(..., stride=3, padding='same')` throws:
`ValueError: padding='same' is not supported for strided convolutions`.
* **Resolution:** In PyTorch, to emulate TensorFlow's `same` padding for stride 3:
  - Input 128: Add asymmetric pad `(0, 1, 0, 1)` (total pad 1) before Conv2D $\to$ Output is 43.
  - Input 21: Pad `(0, 0, 0, 0)` (total pad 0) $\to$ Output is 7.
  - Input 3: Pad `(0, 0, 0, 0)` (total pad 0) $\to$ Output is 1.

#### Finding 3: Interaction Between Early Stopping (Patience 5) and ReduceLROnPlateau (Patience 4)
* When validation accuracy plateaus starting at epoch $k$:
  - Epoch $k+1$: No improvement (ES count = 1, LR count = 1).
  - Epoch $k+2$: No improvement (ES count = 2, LR count = 2).
  - Epoch $k+3$: No improvement (ES count = 3, LR count = 3).
  - Epoch $k+4$: No improvement (ES count = 4, LR count = 4 $\to$ **LR reduced by 0.5** to $0.0025$).
  - Epoch $k+5$: Model trains with reduced LR. If validation accuracy does not strictly exceed epoch $k$, **ES count reaches 5 $\to$ Early Stopping terminates training immediately**.
* **Impact:** The reduced learning rate only gets **exactly 1 epoch** to demonstrate improvement before training halts.
* **Ties & Minimum Delta:**  
  Ties ($val\_acc_t == best\_val\_acc$) are treated strictly as **non-improvements** with minimum delta $\Delta = 0.0$. The best checkpoint is NOT overwritten on ties (preserving the earliest checkpoint).

---

## 5. Classification Cases & Label Alignment

All labels align 100% with the existing dataset schema in `audit/batch_preprocessing/bonn_features_96.csv`:

| Benchmark Case | Description | Subsets & Classes Included | Total Samples | Class Mapping (Alphabetical / Canonical) | Positive Class |
|---|---|---|---|---|---|
| **Case 1** | 3-Class | Healthy (A+B) vs Interictal (C+D) vs Ictal (E) | 500 | `healthy`: 0 (200)<br>`ictal`: 1 (100)<br>`interictal`: 2 (200) | Multi-class (Macro-F1) |
| **Case 2** | Binary | Non-Ictal (A+B+C+D) vs Ictal (E) | 500 | `ictal`: 0 (100)<br>`non_ictal`: 1 (400) | `ictal` (Index 0) |
| **Case 3** | Pilot Binary | Interictal (D) vs Ictal (E) | 200 | `ictal`: 0 (100)<br>`interictal`: 1 (100) | `ictal` (Index 0) |

*(Note: Subsets A, B, C are marked `NaN` in `case3_label` and are excluded from Case 3, leaving exactly 200 samples).*

---

## 6. Evaluation Protocol & Split Leakage Prevention

To ensure a scientifically valid comparison against the RBF-SVM baseline:

```
Full Dataset (500 recordings / 200 for Case 3)
   │
   ├── Stratified 10-Fold Outer Split (Seed 42, exactly matching SVM checkpoints)
   │      │
   │      ├── Outer Test Fold (10% = 50 samples / 20 samples) ──── HELD OUT COMPLETELY
   │      │     (Never seen by scaler, weights, training, or early stopping)
   │      │
   │      └── Outer Training Fold (90% = 450 samples / 180 samples)
   │             │
   │             ├── Inner Stratified Validation Split (10% of outer train = 45 samples)
   │             │     (Monitored strictly for ReduceLROnPlateau & Early Stopping)
   │             │
   │             └── Sub-Training Partition (90% of outer train = 405 samples)
   │                   │
   │                   ├── 1. Compute Balanced Class Weights w_c exclusively here
   │                   └── 2. Train CNN with AdamW + Weighted Cross-Entropy
   │
   └── Restore Best Validation Checkpoint ──► Evaluate on Outer Test Fold (Out-of-Fold Metrics)
```

### Strict Isolation Rules
1. **Zero Normalization Leakage:** Continuous recurrence plots are normalized using per-recording maximum distance $\text{GRP} / \max(\text{GRP})$. No dataset-wide mean or variance is computed across samples.
2. **Zero Outer Test Influence:** Class weights, batch normalization running statistics, gradient updates, and early stopping checkpoint selection NEVER observe outer test fold samples.
3. **Deterministic Seed Control:** Random seed `42` fixed for PyTorch (`torch.manual_seed`), CUDA deterministic mode (`torch.backends.cudnn.deterministic = True`), NumPy (`np.random.seed`), and Python `random`.
4. **Metrics Reported:**
   - Accuracy, Macro-F1, Balanced Accuracy.
   - Per-class Precision, Recall, Specificity.
   - Confusion Matrix.
   - Fold-level Mean $\pm$ Standard Deviation across the 10 outer folds.
   - Full training history: Loss curves, validation accuracy, best epoch, and runtime.

---

## 7. Proposed Implementation File Layout

The CNN branch will be introduced into a dedicated, isolated namespace `scripts/cnn/` without modifying any existing classification or preprocessing files:

```
phase_sync_eeg/
├── scripts/
│   ├── classification/               # UNTOUCHED (nested_cv.py, mcfs.py, run_kaggle_*.py)
│   ├── preprocessing/                  # UNTOUCHED (grp.py, embedding.py, pipeline.py, etc.)
│   └── cnn/                           # NEW DEDICATED PACKAGE
│       ├── __init__.py
│       ├── config.py                  # Dataclass holding all CNN hyperparameters
│       ├── export_rp_images.py        # Deterministic GRP extraction & 128x128 resizing
│       ├── dataset.py                 # PyTorch Dataset & DataLoader with split isolation
│       ├── model.py                   # 3-Block CNN architecture with padding resolution
│       ├── train.py                   # Per-fold trainer, early stopping, LR scheduler, loss
│       └── run_kaggle_cnn.py          # 10-fold nested CV Kaggle runner with atomic checkpoints
├── tests/
│   ├── test_cnn_architecture.py      # NEW: Shape tests, forward/backward pass, GAP
│   ├── test_cnn_leakage.py           # NEW: Split isolation, class weight validation
│   └── test_cnn_pipeline.py          # NEW: End-to-end synthetic 2-fold dry run
└── experiments/
    ├── experiment_k5/                 # UNTOUCHED: Previous SVM baseline checkpoints
    └── experiment_cnn/                # NEW: Dedicated CNN fold checkpoints & results JSON
```

---

## 8. Storage, Compute, and Environment Assessment

### Storage Footprint
* **128 × 128 RP Images:**
  - Uncompressed float32: $128 \times 128 \times 4\text{ bytes} = 65,536\text{ bytes} = 64\text{ KB}$ per image.
  - All 500 recordings: $500 \times 64\text{ KB} = \mathbf{32.0\text{ MB}}$ total.
  - If saved as 8-bit grayscale PNG: $\sim 8\text{ KB}$ per image $\to \mathbf{4.0\text{ MB}}$ total.
* **Checkpoint Storage:**
  - Model weights per fold (3 blocks + Dense 64): $\sim 65,000$ parameters $\approx 260\text{ KB}$ per `.pt` file.
  - 10 folds $\times$ 3 cases: $30 \times 260\text{ KB} = \mathbf{7.8\text{ MB}}$ total.
* **Verdict:** Highly lightweight. Storage overhead is negligible ($< 50\text{ MB}$ total).

### Compute & Training Runtime Estimate
* Total samples per outer fold: 405 training samples.
* Batch size 16: $\approx 26$ batches per epoch.
* Max 50 epochs: $\approx 1,300$ batch iterations per fold.
* On Kaggle GPU (T4 / P100):
  - Each epoch takes $\sim 0.15$ seconds.
  - 50 epochs take $\sim 7.5$ seconds per fold.
  - 10 outer folds take $\sim 75$ seconds ($\approx 1.25$ minutes) per case.
  - All 3 cases take $\sim \mathbf{3.5\text{ to }4.0\text{ minutes}}$ total on GPU.
* Even on local Mac CPU (M-series Apple Silicon), each case will run in under 8 minutes.

---

## 9. Comprehensive Testing Plan (To Be Run Before Full Experiment)

Before executing the full 10-fold CV on Kaggle, the following lightweight unit tests must pass:

1. **`test_cnn_image_export_and_shape`**:
   - Verify continuous GRP generation for a synthetic 4097-sample signal.
   - Confirm bilinear resizing produces exactly shape `(128, 128, 1)`.
   - Confirm intensity values are strictly within $[0.0, 1.0]$.
2. **`test_cnn_tensor_forward_shapes`**:
   - Verify forward pass on batch size 4 for Case 1 (`[4, 3]`), Case 2 (`[4, 2]`), and Case 3 (`[4, 2]`).
   - Confirm spatial dimensions after each convolutional and pooling block match the resolved specification.
3. **`test_cnn_split_isolation_and_exact_match`**:
   - Compare outer train and test indices generated by the CNN runner against saved `experiments/experiment_k5/Case1_3Class/checkpoints/fold_0.json`.
   - Assert zero intersection between training, validation, and test sets.
4. **`test_cnn_class_weights_training_only`**:
   - Compute class weights on sub-training split; verify that validation and test labels do not alter weight values.
5. **`test_cnn_early_stopping_and_checkpoint_restoration`**:
   - Simulate a plateau where validation accuracy stays flat for 5 epochs.
   - Assert that training terminates at epoch 5 and that weights from the best epoch are restored.
6. **`test_cnn_lr_scheduler_step`**:
   - Confirm learning rate drops by 0.5 at epoch 4 of plateau.
7. **`test_baseline_pipeline_immutability`**:
   - Assert SHA-256 of `audit/batch_preprocessing/bonn_features_96.csv` remains exactly `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`.
   - Assert `experiments/experiment_k5/` contents are unchanged.

---

## 10. Ordered Implementation Stages with Explicit Approval Checkpoints

```
[STAGE 1: Architectural Resolution & User Approval]  ◄── CURRENT STAGE
  - User reviews Block 3 MaxPool spatial collapse and selects Option A, B, or C.
  - User confirms PyTorch backend preference.
  ▼
[STAGE 2: Isolated RP Image Export Utility]
  - Create `scripts/cnn/export_rp_images.py`.
  - Export 128x128 continuous GRPs into `data/processed/rp_images_128/`.
  - Validate image checksums and recording ID matching.
  ▼
[STAGE 3: CNN Model & Unit Tests]
  - Implement `scripts/cnn/model.py`, `scripts/cnn/dataset.py`, `scripts/cnn/config.py`.
  - Implement and pass `tests/test_cnn_architecture.py`.
  ▼
[STAGE 4: Fold Trainer & Leakage Verification]
  - Implement `scripts/cnn/train.py` with early stopping and AdamW.
  - Run synthetic 2-fold verification test to confirm checkpointing and metric calculation.
  ▼
[STAGE 5: Full Kaggle Runner Preparation]
  - Implement `scripts/cnn/run_kaggle_cnn.py` with external checkpointing and resume support.
  - Provide tested Kaggle notebook commands for execution.
```

---

## 11. Final Statement & Required User Decisions

**Audit Verdict:**  
The repository audit **strongly supports implementation of the CNN branch** with high feasibility and minimal resource overhead, subject to resolving the Block 3 spatial collapse.

### Decisions Required from the User Prior to Implementation:
1. **Block 3 MaxPool Resolution:**  
   Which resolution do you approve?
   - **Option A (Recommended):** Omit `MaxPooling2D` in Block 3. Conv2D stride 3 downsamples to $1 \times 1 \times 64$, which flows directly into Global Average Pooling ($64$).
   - **Option B:** Change Conv2D strides from 3 to 1 across all blocks, letting MaxPooling2D perform downsampling ($128 \to 64 \to 32 \to 16$).
2. **Framework Backend:**  
   Confirm **PyTorch 2.x** as the primary implementation backend (since PyTorch 2.12.0 is already installed and verified in your local environment, and pre-installed on Kaggle).
