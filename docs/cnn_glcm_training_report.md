# Bonn EEG Research Project — GLCM-CNN Training Report (M1 Mac Controlled Execution)

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** COMPLETE — ALL STAGES EXECUTED & AUDITED  
**Branch:** `fix/bonn-replication-audit`  
**Hardware Environment:** Apple Silicon M1 (MacBook Air, 8 GB RAM, PyTorch 2.12.0 with MPS backend)  
**Experiment Root:** `experiments/results_cnn_glcm_80_20_30ep_20261013_131200/`  

---

## 1. Executive Summary

This report documents the end-to-end execution of the **independent Bonn EEG GLCM-CNN branch** using actual 8-channel GLCM matrix tensors of shape `[8, 16, 16]`. 

All 5 execution stages (Preflight, Full Matrix Preparation, Smoke Test, Full 3-Case Training, and Metrics/Artifact Generation) completed cleanly within the strict resource constraints of an 8 GB M1 Mac.

### Core Execution Highlights:
1. **Zero Impact on Baseline Pipeline:** The SVM replication pipeline, historical 10-fold CV checkpoints in `experiments/experiment_k5/`, and the frozen feature CSV (`audit/batch_preprocessing/bonn_features_96.csv`, SHA-256 `af8d3d2f...b9d1`) remain 100% untouched.
2. **Mac Resource Safety:** Matrix export required ~565 MB RAM and 6.3 MB disk space (16 GiB free). Full training on Apple Silicon MPS completed in **10.0 seconds total** across all three classification cases.
3. **Reproducibility:** A fixed, stratified 80:20 outer split and 10% inner validation of the training pool was applied independently to each case with random seed 42.

---

## 2. Stage-by-Stage Verification Summary

| Stage | Description | Measured Duration | Resource Footprint | Result Status |
|---|---|---|---|---|
| **Stage A: Preflight** | Branch audit, disk check, MPS hardware test, 13 CNN unit tests | ~8.0s | Clean git tree, 17 GiB free disk, MPS available | **PASS** |
| **Stage B: Matrix Export** | Full 500-recording sequential export `[8, 16, 16]` to `data/processed/glcm_matrices_cnn/` | 8m 38s (~1.04s/rec) | ~565 MB RAM, 6.3 MB disk, 0% swap | **PASS (500/500 validated)** |
| **Stage C: Smoke Test** | Case 1 on 80:20 split, 3 epochs on MPS device | 3.4s | 125 KB checkpoint, 100% finite loss & grads | **PASS** |
| **Stage D: Full Training** | Cases 1, 2, and 3 (up to 30 epochs each on MPS) | 10.0s total | Peak RAM < 600 MB, GPU acceleration via `mps:0` | **PASS (3/3 models)** |
| **Stage E: Artifacts** | Split manifests, logs, checkpoints, predictions, metrics | Instantaneous | Compact JSON/CSV files (~375 KB total checkpoints) | **PASS** |

---

## 3. Dataset Splits and Partition Counts

All splits were constructed with `StratifiedShuffleSplit(random_state=42)`:
* **Outer Split:** 80% training pool, 20% held-out test set.
* **Inner Split:** 10% of the training pool held out for validation and early stopping.
* **Proportions:** ~72% gradient training, ~8% inner validation, 20% held-out testing.
* **Integrity:** Sub-training, validation, and test subsets are strictly pairwise disjoint ($S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$).

### Exact Partition Counts:

| Case | Subsets & Classes | Total Recordings | Training Pool (80%) | Gradient Training (~72%) | Inner Validation (~8%) | Held-Out Test (20%) |
|---|---|---|---|---|---|---|
| **Case 1 (3-Class)** | A=0 (100), B=1 (100), C+D=2 (200), E excluded | **400** | 320 | **288** | **32** | **80** |
| **Case 2 (Binary)** | A+B+C+D=0 (400), E=1 (100) | **500** | 400 | **360** | **40** | **100** |
| **Case 3 (Binary)** | D=0 (100), E=1 (100), A,B,C excluded | **200** | 160 | **144** | **16** | **40** |

### Class-Wise Distribution across Partitions:

* **Case 1:**
  * Sub-train ($N=288$): Class 0 = 72, Class 1 = 72, Class 2 = 144
  * Validation ($N=32$): Class 0 = 8, Class 1 = 8, Class 2 = 16
  * Test ($N=80$): Class 0 = 20, Class 1 = 20, Class 2 = 40
* **Case 2:**
  * Sub-train ($N=360$): Class 0 = 288, Class 1 = 72
  * Validation ($N=40$): Class 0 = 32, Class 1 = 8
  * Test ($N=100$): Class 0 = 80, Class 1 = 20
* **Case 3:**
  * Sub-train ($N=144$): Class 0 = 72, Class 1 = 72
  * Validation ($N=16$): Class 0 = 8, Class 1 = 8
  * Test ($N=40$): Class 0 = 20, Class 1 = 20

---

## 4. Training Performance and Test Metrics

### Comparative Summary Table:

| Metric | Case 1: Three-Class (A vs B vs C+D) | Case 2: Binary (A+B+C+D vs E) | Case 3: Binary (D vs E) |
|---|---|---|---|
| **Input Shape** | `[B, 8, 16, 16]` | `[B, 8, 16, 16]` | `[B, 8, 16, 16]` |
| **Output Classes** | 3 | 2 | 2 |
| **Total Epochs Trained** | 11 (early stopped) | 10 (early stopped) | 6 (early stopped) |
| **Best Epoch** | **Epoch 6** | **Epoch 5** | **Epoch 1** |
| **Best Validation Accuracy** | 84.38% | 97.50% | 50.00% |
| **Held-Out Test Accuracy** | **75.00%** | **90.00%** | **50.00%** |
| **Balanced Accuracy** | 69.17% | 86.25% | 50.00% |
| **Macro-F1 Score** | 65.75% | 84.93% | 33.33% |
| **Macro-Precision** | 69.50% | 83.80% | 25.00% |
| **Macro-Recall** | 69.17% | 86.25% | 50.00% |
| **Training Duration** | 3.66s | 5.36s | 0.99s |
| **Average Epoch Time** | 0.332s | 0.536s | 0.165s |

---

## 5. Detailed Case Evaluations and Confusion Matrices

### Case 1: Three-Class Classification (A vs B vs C+D)
* **Best Model Selected:** Epoch 6 (Val Acc: 84.38%, Val Loss: 0.5013)
* **Test Accuracy:** 75.00% (60/80 correct)
* **Confusion Matrix:**
  $$\begin{pmatrix} 5 & 15 & 0 \\ 2 & 18 & 0 \\ 2 & 1 & 37 \end{pmatrix}$$
  * Row 0: True A ($N=20$) $\to$ 5 predicted as A, 15 predicted as B, 0 predicted as C+D
  * Row 1: True B ($N=20$) $\to$ 2 predicted as A, 18 predicted as B, 0 predicted as C+D
  * Row 2: True C+D ($N=40$) $\to$ 2 predicted as A, 1 predicted as B, 37 predicted as C+D
* **Per-Class Metrics:**
  * **A (healthy open):** Precision = 55.56%, Recall = 25.00%, F1 = 34.48%
  * **B (healthy closed):** Precision = 52.94%, Recall = 90.00%, F1 = 66.67%
  * **C+D (interictal):** Precision = **100.00%**, Recall = **92.50%**, F1 = **96.10%**

---

### Case 2: Binary Classification (A+B+C+D vs E)
* **Best Model Selected:** Epoch 5 (Val Acc: 97.50%, Val Loss: 0.1990)
* **Test Accuracy:** 90.00% (90/100 correct)
* **Confusion Matrix:**
  $$\begin{pmatrix} 74 & 6 \\ 4 & 16 \end{pmatrix}$$
  * Row 0: True Non-Ictal ($N=80$) $\to$ 74 predicted as non-ictal, 6 as ictal (Specificity = 92.50%)
  * Row 1: True Ictal ($N=20$) $\to$ 4 predicted as non-ictal, 16 as ictal (Sensitivity = 80.00%)
* **Per-Class Metrics:**
  * **Non-Ictal (A+B+C+D):** Precision = 94.87%, Recall = 92.50%, F1 = 93.67%
  * **Ictal (E):** Precision = 72.73%, Recall = 80.00%, F1 = 76.19%

---

### Case 3: Binary Classification (D vs E)
* **Best Model Selected:** Epoch 1 (Val Acc: 50.00%, Val Loss: 1.0677)
* **Test Accuracy:** 50.00% (20/40 correct)
* **Confusion Matrix:**
  $$\begin{pmatrix} 0 & 20 \\ 0 & 20 \end{pmatrix}$$
  * Row 0: True Interictal D ($N=20$) $\to$ 0 predicted as D, 20 predicted as E (Recall = 0.0%)
  * Row 1: True Ictal E ($N=20$) $\to$ 0 predicted as D, 20 predicted as E (Recall = 100.0%)
* **Per-Class Metrics:**
  * **Interictal (D):** Precision = 0.00%, Recall = 0.00%, F1 = 0.00%
  * **Ictal (E):** Precision = 50.00%, Recall = 100.00%, F1 = 66.67%

---

## 6. Analysis, Warnings, and Limitations

### 1. Case 3 Behavior (Model Collapse)
In Case 3 (small sample size: 144 gradient-training samples, 16 validation samples), the initial learning rate ($0.005$) caused the validation accuracy to tie at $50.0\%$ (random chance / majority prediction) across the first 5 epochs. Under the strict improvement rule ($\Delta > 0.0$, ties keep the earliest checkpoint), Epoch 1 was retained, and early stopping was triggered at Epoch 6. 

On the held-out test set, the Epoch 1 weights predicted Class 1 for all samples, resulting in 50% accuracy. This indicates that for small focal datasets, $lr=0.005$ without warm-up is too high or requires a different learning rate schedule.

### 2. Comparability with Existing SVM Baseline
**These CNN results are NOT directly comparable to the historical SVM baseline.**
* **Protocol Discrepancy:** The SVM baseline used **10-fold nested cross-validation** with an inner hyperparameter grid search for MCFS feature count $k \in \{5, 10, 15, 20\}$ and SVM $\gamma \in \{10^{-4}, \dots, 10^1\}$, evaluated across all 500 recordings pooled. In contrast, this CNN run evaluated a **single 80:20 train/test split**.
* **Case 1 Definition Discrepancy:** In the SVM replication pipeline, Case 1 was formulated as $A+B$ (healthy, 200) vs $C+D$ (interictal, 200) vs $E$ (ictal, 100), totaling 500 recordings. Here, Case 1 was specified as $A$ (100) vs $B$ (100) vs $C+D$ (200) with $E$ excluded (400 recordings total).
* **Feature Representation Discrepancy:** The SVM operates on scalar Haralick feature summaries selected via Laplacian graph feature selection (MCFS), whereas the CNN operates directly on continuous spatial joint probability distributions ($8 \times 16 \times 16$).

---

## 7. Artifact Inventory

All generated artifacts are preserved in dedicated directories without overwriting prior pilot files:

### Directory Structure:
```
experiments/results_cnn_glcm_80_20_30ep_20261013_131200/
├── config.json                               # Full run configuration
├── all_cases_summary.json                    # Consolidated metrics across all cases
├── execution.log                             # Epoch-by-epoch training log
├── Case1_3Class_split_manifest.json          # Recording IDs and labels for Case 1
├── Case2_Binary_split_manifest.json          # Recording IDs and labels for Case 2
├── Case3_Binary_split_manifest.json          # Recording IDs and labels for Case 3
├── Case1_3Class/
│   ├── best_checkpoint.pt                   # Best model PyTorch checkpoint (125 KB)
│   ├── metrics.json                          # Test metrics & classification report
│   ├── test_predictions.csv                 # Per-recording test predictions & probabilities
│   └── training_history.csv                 # Per-epoch loss, accuracy, and lr
├── Case2_Binary/
│   ├── best_checkpoint.pt                   # Best model PyTorch checkpoint (125 KB)
│   ├── metrics.json                          # Test metrics & classification report
│   ├── test_predictions.csv                 # Per-recording test predictions & probabilities
│   └── training_history.csv                 # Per-epoch loss, accuracy, and lr
└── Case3_Binary/
    ├── best_checkpoint.pt                   # Best model PyTorch checkpoint (125 KB)
    ├── metrics.json                          # Test metrics & classification report
    ├── test_predictions.csv                 # Per-recording test predictions & probabilities
    └── training_history.csv                 # Per-epoch loss, accuracy, and lr
```

### Git Working Tree Status:
* No large checkpoints have been pushed to Git.
* Checkpoints are strictly retained locally (125 KB each $\times$ 3 = ~375 KB total).
* Tracked repository files remain 100% clean.

---

## 8. Conclusion

The Bonn EEG GLCM-CNN branch is fully verified on real multi-channel GLCM matrices. The M1 Mac handled the entire full-dataset preparation and GPU-accelerated training pipeline safely in under 10 minutes total runtime.
