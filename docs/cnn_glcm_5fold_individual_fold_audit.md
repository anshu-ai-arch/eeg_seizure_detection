# Bonn EEG Research Project — 5-Fold GLCM-CNN Individual Fold Audit Report

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** COMPLETE — STRICTLY READ-ONLY AUDIT OF ALL 15 CASE-FOLD FITS  
**Experiment Root:** `experiments/results_cnn_glcm_5fold_20261013_132700/`  
**Reference Aggregate Report:** `docs/cnn_glcm_5fold_report.md`  
**Branch:** `fix/bonn-replication-audit`  

---

## 1. Executive Summary & Audit Scope

This document provides a comprehensive, read-only audit of **all 15 individual fold models** (3 classification cases $\times$ 5 folds) trained on continuous 8-channel GLCM matrix tensors of shape `[batch_size, 8, 16, 16]`.

Every individual fold's saved predictions, epoch-wise training logs, confusion matrices, checkpoint metadata, and split manifests were audited directly from disk. All evaluation metrics were independently recomputed from raw per-recording prediction CSVs.

### Summary Verdict by Case:
* **Case 1 (Three-Class, A vs B vs C+D, $N=400$):**
  * **Individual Fold Accuracies:** Fold 0: `77.50%`, Fold 1: `76.25%`, Fold 2: `73.75%`, Fold 3: `82.50%`, Fold 4: `61.25%`.
  * **Behavior:** Folds 0–3 demonstrated strong, stable classification (73.75%–82.50%). Fold 4 underperformed (61.25%) due to misclassifying 15 C+D samples as A. C+D (epileptic interictal) consistently achieved high precision (90.9%–100.0%) across all folds.
* **Case 2 (Binary, A+B+C+D vs E, $N=500$):**
  * **Individual Fold Accuracies:** Fold 0: `89.00%`, Fold 1: `93.00%`, Fold 2: `91.00%`, Fold 3: `88.00%`, Fold 4: `96.00%`.
  * **Behavior:** Highly consistent across all 5 folds (std = 3.21%). Specificity on non-ictal recordings was exceptional (90.0%–100.0%, 385/400 total correct). Sensitivity on seizure activity ranged from 45.0% (Fold 0) to 85.0% (Fold 4).
* **Case 3 (Binary, D vs E, $N=200$):**
  * **Individual Fold Accuracies:** Fold 0: `50.00%`, Fold 1: `50.00%`, Fold 2: `50.00%`, Fold 3: `50.00%`, Fold 4: `50.00%`.
  * **Behavior:** **Complete class collapse in all 5 folds.** Every fold predicted 100% of samples as Class 1 (ictal E). Training logs prove that validation accuracy tied at 50.0% from Epoch 1, causing early stopping to terminate at Epoch 6 and restore an un-converged Epoch 1 checkpoint in every fit.

---

## 2. Case 1: Three-Class Classification (A vs B vs C+D)

* **Class 0:** Subset A (healthy volunteers, eyes open)
* **Class 1:** Subset B (healthy volunteers, eyes closed)
* **Class 2:** Subsets C & D (epileptic interictal formation & zone)
* **Excluded:** Subset E (ictal)
* **Total Recordings:** 400 (100 A, 100 B, 200 C+D)
* **Test Samples per Fold:** Exactly 80 (20 A, 20 B, 40 C+D)

### 2.1 Fold-by-Fold Performance Table

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std |
|---|---|---|---|---|---|---|
| **Random Seed** | 42 | 142 | 242 | 342 | 442 | — |
| **Test Recordings ($N$)** | 80 | 80 | 80 | 80 | 80 | **80** |
| **Correct Predictions** | 62 / 80 | 61 / 80 | 59 / 80 | 66 / 80 | 49 / 80 | **59.4 / 80** |
| **Test Accuracy** | **77.50%** | **76.25%** | **73.75%** | **82.50%** | **61.25%** | **74.25% $\pm$ 7.94%** |
| **Balanced Accuracy** | 73.33% | 68.33% | 70.00% | 77.50% | 60.83% | **70.00% $\pm$ 6.21%** |
| **Macro Precision** | 73.71% | 71.88% | 68.63% | 84.52% | 59.90% | **71.73% $\pm$ 8.91%** |
| **Macro Recall** | 73.33% | 68.33% | 70.00% | 77.50% | 60.83% | **70.00% $\pm$ 6.21%** |
| **Macro-F1 Score** | **73.27%** | **68.21%** | **66.59%** | **77.54%** | **57.23%** | **68.57% $\pm$ 7.67%** |
| **Best Val Epoch** | Epoch 6 | Epoch 12 | Epoch 7 | Epoch 6 | Epoch 4 | **7.0 $\pm$ 3.0** |
| **Best Val Accuracy** | 84.38% | 81.25% | 68.75% | 71.88% | 65.62% | **74.38% $\pm$ 8.16%** |
| **Total Epochs Trained** | 11 | 17 | 12 | 11 | 9 | **12.0 $\pm$ 3.0** |
| **Early Stopping Epoch** | Epoch 11 | Epoch 17 | Epoch 12 | Epoch 11 | Epoch 9 | — |
| **Training Duration** | 3.57s | 4.85s | 3.43s | 3.23s | 2.57s | **3.53s $\pm$ 0.83s** |

### 2.2 Confusion Matrices (Class Order: 0: A, 1: B, 2: C+D)

* **Fold 0 Confusion Matrix:**
  $$\begin{pmatrix} 13 & 5 & 2 \\ 7 & 13 & 0 \\ 4 & 0 & 36 \end{pmatrix}$$
  * Predicted distribution: Class 0 = 24, Class 1 = 18, Class 2 = 38
* **Fold 1 Confusion Matrix:**
  $$\begin{pmatrix} 13 & 3 & 4 \\ 12 & 8 & 0 \\ 0 & 0 & 40 \end{pmatrix}$$
  * Predicted distribution: Class 0 = 25, Class 1 = 11, Class 2 = 44
* **Fold 2 Confusion Matrix:**
  $$\begin{pmatrix} 6 & 14 & 0 \\ 1 & 19 & 0 \\ 5 & 1 & 34 \end{pmatrix}$$
  * Predicted distribution: Class 0 = 12, Class 1 = 34, Class 2 = 34
* **Fold 3 Confusion Matrix:**
  $$\begin{pmatrix} 17 & 0 & 3 \\ 10 & 10 & 0 \\ 1 & 0 & 39 \end{pmatrix}$$
  * Predicted distribution: Class 0 = 28, Class 1 = 10, Class 2 = 42
* **Fold 4 Confusion Matrix:**
  $$\begin{pmatrix} 5 & 15 & 0 \\ 1 & 19 & 0 \\ 15 & 0 & 25 \end{pmatrix}$$
  * Predicted distribution: Class 0 = 21, Class 1 = 34, Class 2 = 25

### 2.3 Per-Class Precision, Recall, and F1 across Folds

| Fold | Class 0 (A: Healthy Open) | Class 1 (B: Healthy Closed) | Class 2 (C+D: Interictal) |
|---|---|---|---|
| **Fold 0** | Prec: 54.17%, Rec: 65.00%, F1: 59.09% | Prec: 72.22%, Rec: 65.00%, F1: 68.42% | Prec: 94.74%, Rec: 90.00%, F1: 92.31% |
| **Fold 1** | Prec: 52.00%, Rec: 65.00%, F1: 57.78% | Prec: 72.73%, Rec: 40.00%, F1: 51.61% | Prec: 90.91%, Rec: 100.00%, F1: 95.24% |
| **Fold 2** | Prec: 50.00%, Rec: 30.00%, F1: 37.50% | Prec: 55.88%, Rec: 95.00%, F1: 70.37% | Prec: 100.00%, Rec: 85.00%, F1: 91.89% |
| **Fold 3** | Prec: 60.71%, Rec: 85.00%, F1: 70.83% | Prec: 100.00%, Rec: 50.00%, F1: 66.67% | Prec: 92.86%, Rec: 97.50%, F1: 95.12% |
| **Fold 4** | Prec: 23.81%, Rec: 25.00%, F1: 24.39% | Prec: 55.88%, Rec: 95.00%, F1: 70.37% | Prec: 100.00%, Rec: 62.50%, F1: 76.92% |
| **Averaged** | **Prec: 48.14%, Rec: 54.00%, F1: 49.92%** | **Prec: 71.34%, Rec: 69.00%, F1: 65.49%** | **Prec: 95.70%, Rec: 87.00%, F1: 90.30%** |

### 2.4 Training and Validation Dynamics (Case 1)
* **Training Loss Convergence:** In all 5 folds, training loss decreased monotonically from $\sim 0.85$ at Epoch 1 to $\sim 0.51–0.62$ at early stopping.
* **Validation Loss Behavior:** Validation loss reached its minimum at the best epoch ($\sim 0.52–0.81$), but dramatically overfitted thereafter (reaching $7.4–12.7$ at stopping epoch).
* **Early Stopping Verification:** Worked strictly as configured with patience 5. The checkpoint restoration mechanism successfully reverted the model from the overfitted stopping state to the best validation epoch state.

---

## 3. Case 2: Binary Classification (A+B+C+D vs E)

* **Class 0 (Negative):** Subsets A, B, C, D (non-ictal, 400 recordings)
* **Class 1 (Positive):** Subset E (ictal seizure activity, 100 recordings)
* **Total Recordings:** 500
* **Test Samples per Fold:** Exactly 100 (80 Non-Ictal, 20 Ictal)

### 3.1 Fold-by-Fold Performance Table

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std |
|---|---|---|---|---|---|---|
| **Random Seed** | 42 | 142 | 242 | 342 | 442 | — |
| **Test Recordings ($N$)** | 100 | 100 | 100 | 100 | 100 | **100** |
| **Correct Predictions** | 89 / 100 | 93 / 100 | 91 / 100 | 88 / 100 | 96 / 100 | **91.4 / 100** |
| **Test Accuracy** | **89.00%** | **93.00%** | **91.00%** | **88.00%** | **96.00%** | **91.40% $\pm$ 3.21%** |
| **Balanced Accuracy** | 72.50% | 86.25% | 85.00% | 85.00% | 91.88% | **84.12% $\pm$ 7.09%** |
| **Macro Precision** | 93.96% | 91.11% | 86.39% | 80.70% | 95.39% | **89.51% $\pm$ 6.01%** |
| **Macro Recall** | 72.50% | 86.25% | 85.00% | 85.00% | 91.88% | **84.12% $\pm$ 7.09%** |
| **Macro-F1 Score** | **77.82%** | **88.39%** | **85.67%** | **82.52%** | **93.50%** | **85.58% $\pm$ 5.92%** |
| **Sensitivity (Ictal E)** | **45.00%** | **75.00%** | **75.00%** | **80.00%** | **85.00%** | **72.00% $\pm$ 15.65%** |
| **Specificity (Non-Ictal A-D)** | **100.00%** | **97.50%** | **95.00%** | **90.00%** | **98.75%** | **96.25% $\pm$ 3.95%** |
| **True Positives (TP)** | 9 | 15 | 15 | 16 | 17 | **14.4 $\pm$ 3.2** |
| **True Negatives (TN)** | 80 | 78 | 76 | 72 | 79 | **77.0 $\pm$ 3.2** |
| **False Positives (FP - False Alarms)** | **0** | **2** | **4** | **8** | **1** | **3.0 $\pm$ 3.2 (Total: 15)** |
| **False Negatives (FN - Missed Seizures)**| **11** | **5** | **5** | **4** | **3** | **5.6 $\pm$ 3.2 (Total: 28)** |
| **Best Val Epoch** | Epoch 8 | Epoch 4 | Epoch 7 | Epoch 11 | Epoch 7 | **7.4 $\pm$ 2.5** |
| **Best Val Accuracy** | 90.00% | 95.00% | 95.00% | 95.00% | 95.00% | **94.00% $\pm$ 2.24%** |
| **Total Epochs Trained** | 13 | 9 | 12 | 16 | 12 | **12.4 $\pm$ 2.5** |
| **Training Duration** | 4.92s | 3.19s | 4.29s | 5.71s | 4.26s | **4.47s $\pm$ 0.93s** |

### 3.2 Confusion Matrices (Class Order: 0: Non-Ictal, 1: Ictal)

* **Fold 0 Confusion Matrix:**
  $$\begin{pmatrix} 80 & 0 \\ 11 & 9 \end{pmatrix}$$
  * Predicted distribution: Non-Ictal = 91, Ictal = 9 (0 false alarms, 11 missed seizures)
* **Fold 1 Confusion Matrix:**
  $$\begin{pmatrix} 78 & 2 \\ 5 & 15 \end{pmatrix}$$
  * Predicted distribution: Non-Ictal = 83, Ictal = 17 (2 false alarms, 5 missed seizures)
* **Fold 2 Confusion Matrix:**
  $$\begin{pmatrix} 76 & 4 \\ 5 & 15 \end{pmatrix}$$
  * Predicted distribution: Non-Ictal = 81, Ictal = 19 (4 false alarms, 5 missed seizures)
* **Fold 3 Confusion Matrix:**
  $$\begin{pmatrix} 72 & 8 \\ 4 & 16 \end{pmatrix}$$
  * Predicted distribution: Non-Ictal = 76, Ictal = 24 (8 false alarms, 4 missed seizures)
* **Fold 4 Confusion Matrix:**
  $$\begin{pmatrix} 79 & 1 \\ 3 & 17 \end{pmatrix}$$
  * Predicted distribution: Non-Ictal = 82, Ictal = 18 (1 false alarm, 3 missed seizures)

### 3.3 Clinical Error Distribution & Trade-offs
* **False Alarms (FP):** Across 400 total non-ictal test recordings, only **15** were falsely flagged as seizures ($3.75\%$).
* **Missed Seizures (FN):** Across 100 total ictal seizure recordings, **28** were missed ($28.0\%$). Fold 0 was highly conservative (0 false alarms but 11 missed seizures), whereas Fold 4 achieved an optimal clinical balance (1 false alarm, 3 missed seizures $\to 85.0\%$ sensitivity, $98.75\%$ specificity).

---

## 4. Case 3: Binary Classification (D vs E, Exclude A, B, C)

* **Class 0 (Negative):** Subset D (interictal epileptogenic zone, 100 recordings)
* **Class 1 (Positive):** Subset E (ictal seizure activity, 100 recordings)
* **Total Recordings:** 200
* **Test Samples per Fold:** Exactly 40 (20 Interictal D, 20 Ictal E)

### 4.1 Fold-by-Fold Performance Table

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std |
|---|---|---|---|---|---|---|
| **Random Seed** | 42 | 142 | 242 | 342 | 442 | — |
| **Test Recordings ($N$)** | 40 | 40 | 40 | 40 | 40 | **40** |
| **Correct Predictions** | 20 / 40 | 20 / 40 | 20 / 40 | 20 / 40 | 20 / 40 | **20.0 / 40** |
| **Test Accuracy** | **50.00%** | **50.00%** | **50.00%** | **50.00%** | **50.00%** | **50.00% $\pm$ 0.00%** |
| **Balanced Accuracy** | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | **50.00% $\pm$ 0.00%** |
| **Macro Precision** | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% | **25.00% $\pm$ 0.00%** |
| **Macro Recall** | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | **50.00% $\pm$ 0.00%** |
| **Macro-F1 Score** | **33.33%** | **33.33%** | **33.33%** | **33.33%** | **33.33%** | **33.33% $\pm$ 0.00%** |
| **Sensitivity (Ictal E)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00% $\pm$ 0.00%** |
| **Specificity (Interictal D)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00% $\pm$ 0.00%** |
| **True Positives (TP)** | 20 | 20 | 20 | 20 | 20 | **20.0** |
| **True Negatives (TN)** | 0 | 0 | 0 | 0 | 0 | **0.0** |
| **False Positives (FP)** | **20** | **20** | **20** | **20** | **20** | **20.0 (Total: 100)** |
| **False Negatives (FN)** | **0** | **0** | **0** | **0** | **0** | **0.0 (Total: 0)** |
| **Best Val Epoch** | Epoch 1 | Epoch 1 | Epoch 1 | Epoch 1 | Epoch 1 | **1.0 $\pm$ 0.0** |
| **Best Val Accuracy** | 50.00% | 50.00% | 50.00% | 50.00% | 50.00% | **50.00% $\pm$ 0.00%** |
| **Total Epochs Trained** | 6 | 6 | 6 | 6 | 6 | **6.0 $\pm$ 0.0** |
| **Early Stopping Epoch** | Epoch 6 | Epoch 6 | Epoch 6 | Epoch 6 | Epoch 6 | — |
| **Training Duration** | 0.97s | 0.82s | 0.82s | 0.81s | 0.82s | **0.85s $\pm$ 0.07s** |

### 4.2 Confusion Matrices (Class Order: 0: Interictal D, 1: Ictal E)

All 5 folds produced an identical confusion matrix:
$$\begin{pmatrix} 0 & 20 \\ 0 & 20 \end{pmatrix}$$
* **Predicted Distribution in all 5 folds:** Class 0 (D) = **0**, Class 1 (E) = **40**.
* **Total Out-of-Fold Predictions ($N=200$):**
  $$\begin{pmatrix} 0 & 100 \\ 0 & 100 \end{pmatrix}$$

### 4.3 Root Cause Analysis of Case 3 Failure (Verified Evidence)
1. **Verification of Prediction Files:** Every single row in `test_predictions.csv` across all 5 folds has `predicted_label = 1` and `prob_class_1 > 0.5`.
2. **Epoch Logs Evidence:**
   * In all 5 folds, training loss dropped from $\sim 0.45$ at Epoch 1 to $\sim 0.22$ at Epoch 6.
   * However, validation loss increased rapidly (e.g. Fold 0: $0.86 \to 4.26$; Fold 1: $0.89 \to 3.26$; Fold 3: $0.98 \to 5.14$).
   * Validation accuracy remained pegged at exactly `50.00%` across Epochs 1, 2, 3, 4, 5, and 6.
3. **Mechanics of Checkpoint Restoration:**
   * The early-stopping criterion strictly requires validation accuracy to *exceed* the best validation accuracy ($\Delta > 0.0$).
   * Because epochs 2 through 6 tied with epoch 1 at $50.00\%$, Epoch 1 remained the selected checkpoint.
   * After 5 epochs without improvement ($1 + 5 = 6$), training stopped, and Epoch 1 weights were restored.
   * At Epoch 1, the model has barely begun gradient descent and is heavily biased toward Class 1 logits, producing uniform Class 1 predictions.

---

## 5. Fold Integrity and Split Manifest Audit

Every split manifest was checked against all raw prediction files:

| Verification Item | Requirement | Audit Result | Status |
|---|---|---|---|
| **Outer Test Set Exclusivity** | Every recording in test set of exactly 1 fold | Verified on all 400 (Case 1), 500 (Case 2), 200 (Case 3) samples | **PASS** |
| **Partition Disjointness** | $S_{\text{train}} \cap S_{\text{val}} = \emptyset$, $S_{\text{train}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$ | Confirmed across all 15 folds (0 overlaps detected) | **PASS** |
| **Class Stratification** | Exact integer proportions preserved | 20/20/40 test in Case 1; 80/20 in Case 2; 20/20 in Case 3 across all folds | **PASS** |
| **Sample Summation** | $\sum N_{\text{test}} = N_{\text{total}}$ | Case 1: $5 \times 80 = 400$; Case 2: $5 \times 100 = 500$; Case 3: $5 \times 40 = 200$ | **PASS** |
| **Artifact Completeness** | Checkpoints, logs, manifests, predictions exist | 15/15 fits contain all 4 required files (60 files total) | **PASS** |

---

## 6. Independent Metric Recalculation Audit

All metrics were recalculated independently using `scikit-learn` from raw `test_predictions.csv` files and compared against `metrics.json`:

* **Accuracy Mismatch Count:** **0** (Exact match across all 15 folds)
* **Balanced Accuracy Mismatch Count:** **0** (Exact match across all 15 folds)
* **Macro-F1 Mismatch Count:** **0** (Exact match across all 15 folds)
* **Confusion Matrix Mismatch Count:** **0** (Exact match across all 15 folds)
* **Discrepancies Found:** **None.** Every metric reported in `docs/cnn_glcm_5fold_report.md` matches the underlying raw predictions bit-for-bit.

---

## 7. Conclusions & Research Recommendations

1. **Well-Performing Folds:**
   * **Case 2 Folds 1, 2, and 4:** Demonstrated outstanding seizure classification (**91.0%–96.0% accuracy, 85.0%–93.5% Macro-F1**). Fold 4 achieved **96.00% accuracy** with 85.0% sensitivity and 98.75% specificity.
   * **Case 1 Fold 3:** Achieved **82.50% accuracy** (77.54% Macro-F1) on 3-class classification with 97.5% recall on epileptic interictal activity.
2. **Unstable Folds:**
   * **Case 1 Fold 4:** Experienced an atypical drop to **61.25% accuracy** due to 15 interictal C+D samples being misclassified as healthy A.
   * **Case 2 Fold 0:** Highly conservative model with 100% specificity (0 false alarms) but only 45% sensitivity (11 missed seizures).
3. **Failed / Collapsed Case:**
   * **Case 3 (All Folds):** Complete class collapse at 50% accuracy due to optimization dynamics on small sample size ($N_{\text{subtrain}}=144$) under $lr=0.005$.
   * **Mandatory Fix for Future Experiments:** A lower learning rate ($10^{-4}$ to $10^{-3}$) with learning rate warm-up and validation tie-breaking (e.g. using validation loss rather than accuracy as tie-breaker) is strictly required before Case 3 can learn discriminative features.
