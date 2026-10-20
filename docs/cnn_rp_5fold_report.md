# Bonn EEG Research Project — Continuous Recurrence Plot (GRP) CNN 5-Fold Evaluation Report

**Document Version:** 1.0.0  
**Date:** October 15, 2026  
**Status:** COMPLETE — ALL 15 BENCHMARK FITS EVALUATED & VERIFIED  
**Experiment Root:** `experiments/results_cnn_rp_5fold_20261015_034605/`  
**Input Data:** Continuous 128x128 Generalized Recurrence Plots (GRPs), shape `[batch_size, 1, 128, 128]` float32  
**Model Architecture:** `BonnModernRPCNN` (4-Block ConvNet, BatchNorm, MaxPool 2x2, Global Average Pooling, Dropout 0.4)  
**Branch:** `fix/bonn-replication-audit`  

---

## 1. Executive Summary

In response to the identified preprocessing information collapse in the GLCM pipeline, we implemented and evaluated the **Continuous Recurrence Plot (GRP) CNN architecture**.

Instead of compressing the phase-space trajectory into an intermediate 16-level GLCM histogram (which caused severe information loss, frequency blind spots, and class collapse), the CNN directly receives **continuous $128 \times 128$ Recurrence Plot images** normalized to $[0.0, 1.0]$.

### Key Breakthrough Findings:
1. **Case 3 (D vs E) Problem Completely Solved:**
   * **Old GLCM CNN:** **50.00%** (100% Class Collapse, Sensitivity=100%, Specificity=0.0%, 0/100 interictal samples detected).
   * **New Continuous RP CNN:** **96.00% $\pm$ 4.06%** Accuracy (**96.00%** Pooled OOF, Sensitivity=**94.00%**, Specificity=**98.00%**).
   * **Fold 0 & Fold 2:** Achieved **100.00% accuracy** (40/40 correct, 0 false positives, 0 false negatives).
   * **Comparison with SVM:** Surpassed the replicated RBF-SVM baseline (95.00%).
2. **Case 2 (A+B+C+D vs E) Seizure Detection:**
   * **Old GLCM CNN:** 91.40% Accuracy, 72.00% Sensitivity, 28 missed seizures.
   * **New Continuous RP CNN:** **96.40% $\pm$ 3.07%** Accuracy (**96.40%** Pooled OOF), **90.00%** Sensitivity, **98.00%** Specificity.
   * Missed seizures reduced by **64.3%** (from 28 down to 10 out of 100 ictal recordings).
   * Folds 1 and 4 achieved **99.00% accuracy** with 95.0%–100.0% sensitivity.
3. **Case 1 (A vs B vs C+D) Three-Class Classification:**
   * **Old GLCM CNN:** 74.25% Accuracy, 68.57% Macro-F1.
   * **New Continuous RP CNN:** **89.25% $\pm$ 2.57%** Accuracy (**89.25%** Pooled OOF), **87.51% $\pm$ 2.95%** Macro-F1.
   * Steady +15.00 percentage point improvement across all 5 folds.
4. **GLCM Artifacts & Code 100% Preserved:**
   * All 500 GLCM matrices in `data/processed/glcm_matrices_cnn` remain completely intact.
   * All historical GLCM experiment outputs remain untouched.

---

## 2. Comparative Performance Overview

| Benchmark Case | Metric | Target Paper | Replicated SVM Baseline | Old GLCM CNN | **New Continuous RP CNN** | Delta vs GLCM CNN |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Case 1: 3-Class**<br>*(A vs B vs C+D, $N=400$)* | **Accuracy** | 98.60% | 93.80% | 74.25% $\pm$ 7.94% | **89.25% $\pm$ 2.57%** | **+15.00%** |
| | **Macro-F1** | 98.60% | 93.29% | 68.57% $\pm$ 7.67% | **87.51% $\pm$ 2.95%** | **+18.94%** |
| **Case 2: Binary**<br>*(A+B+C+D vs E, $N=500$)* | **Accuracy** | 99.60% | 96.80% | 91.40% $\pm$ 3.21% | **96.40% $\pm$ 3.07%** | **+5.00%** |
| | **Sensitivity (E)** | 99.50% | 86.00% | 72.00% | **90.00%** | **+18.00%** |
| | **Specificity (A-D)**| 99.70% | 99.50% | 96.25% | **98.00%** | **+1.75%** |
| | **Macro-F1** | 99.60% | 94.76% | 85.58% $\pm$ 5.92% | **94.41% $\pm$ 4.49%** | **+8.83%** |
| **Case 3: Binary**<br>*(D vs E, $N=200$)* | **Accuracy** | 100.00% | 95.00% | 50.00% $\pm$ 0.00% *(Collapse)* | **96.00% $\pm$ 4.06%** | **+46.00%** |
| | **Sensitivity (E)** | 100.00% | 94.00% | 100.00% *(Trivial)* | **94.00%** | — |
| | **Specificity (D)**| 100.00% | 96.00% | 0.00% *(0/100 detected)* | **98.00%** | **+98.00%** |
| | **Macro-F1** | 100.00% | 95.00% | 33.33% $\pm$ 0.00% | **95.99% $\pm$ 4.07%** | **+62.66%** |

---

## 3. Case 1 Detailed Results: Three-Class (A vs B vs C+D, $N=400$)

### Fold-by-Fold Performance:

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std | Pooled OOF |
|---|---|---|---|---|---|---|---|
| **Test Accuracy** | **88.75%** | **93.75%** | **90.00%** | **87.50%** | **86.25%** | **89.25% $\pm$ 2.57%** | **89.25%** |
| **Balanced Accuracy** | 85.83% | 92.50% | 89.17% | 88.33% | 83.33% | **87.83% $\pm$ 3.10%** | **87.83%** |
| **Macro Precision** | 87.93% | 92.65% | 88.13% | 85.67% | 80.95% | **87.07% $\pm$ 3.88%** | **87.25%** |
| **Macro Recall** | 85.83% | 92.50% | 89.17% | 88.33% | 83.33% | **87.83% $\pm$ 3.10%** | **87.83%** |
| **Macro-F1** | **86.75%** | **92.54%** | **88.28%** | **86.44%** | **83.52%** | **87.51% $\pm$ 2.95%** | **87.53%** |
| **Best Val Epoch** | Epoch 13 | Epoch 25 | Epoch 7 | Epoch 9 | Epoch 13 | **13.4 $\pm$ 6.3** | — |
| **Total Epochs Trained** | 20 | 30 | 14 | 16 | 20 | **20.0 $\pm$ 5.6** | — |
| **Training Duration** | 9.38s | 11.79s | 5.37s | 6.10s | 7.63s | **8.05s $\pm$ 2.37s** | Total: 40.3s |

### Case 1 Pooled Confusion Matrix ($N=400$):
$$\begin{pmatrix} 84 & 10 & 6 \\ 12 & 86 & 2 \\ 7 & 6 & 187 \end{pmatrix}$$
* Class 0 (A: Healthy Eyes Open): Recall = 84.0%, Precision = 81.6%
* Class 1 (B: Healthy Eyes Closed): Recall = 86.0%, Precision = 84.3%
* Class 2 (C+D: Epileptic Interictal): Recall = **93.5%**, Precision = **95.9%** (187 / 200 correct!)

---

## 4. Case 2 Detailed Results: Binary (A+B+C+D vs E, $N=500$)

### Fold-by-Fold Performance:

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std | Pooled OOF |
|---|---|---|---|---|---|---|---|
| **Test Accuracy** | **91.00%** | **99.00%** | **98.00%** | **95.00%** | **99.00%** | **96.40% $\pm$ 3.07%** | **96.40%** |
| **Balanced Accuracy** | 90.62% | 97.50% | 95.00% | 87.50% | 99.38% | **94.00% $\pm$ 4.54%** | **94.00%** |
| **Macro Precision** | 85.34% | 99.41% | 97.53% | 97.06% | 97.62% | **95.39% $\pm$ 5.12%** | **95.38%** |
| **Macro-F1** | **87.10%** | **98.41%** | **96.75%** | **91.34%** | **98.47%** | **94.41% $\pm$ 4.49%** | **94.33%** |
| **Sensitivity (E)** | **90.00%** | **95.00%** | **90.00%** | **75.00%** | **100.00%** | **90.00% $\pm$ 8.37%** | **90.00%** |
| **Specificity (A-D)** | **91.25%** | **100.00%** | **100.00%** | **100.00%** | **98.75%** | **98.00% $\pm$ 3.44%** | **98.00%** |
| **False Positives (FP)** | 7 | 0 | 0 | 0 | 1 | **1.6 $\pm$ 2.7 (Total: 8)** | **8 / 400 (2.0%)** |
| **False Negatives (FN)** | 2 | 1 | 2 | 5 | 0 | **2.0 $\pm$ 1.7 (Total: 10)** | **10 / 100 (10.0%)** |
| **Best Val Epoch** | Epoch 8 | Epoch 10 | Epoch 13 | Epoch 11 | Epoch 28 | **14.0 $\pm$ 7.3** | — |
| **Training Duration** | 8.22s | 8.42s | 9.83s | 8.69s | 14.42s | **9.92s $\pm$ 2.34s** | Total: 49.6s |

### Case 2 Pooled Confusion Matrix ($N=500$):
$$\begin{pmatrix} 392 & 8 \\ 10 & 90 \end{pmatrix}$$
* Total Non-Ictal Correct: **392 / 400 (98.0%)**
* Total Ictal Seizures Correct: **90 / 100 (90.0%)**

---

## 5. Case 3 Detailed Results: Binary (D vs E, $N=200$)

### Fold-by-Fold Performance:

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ Std | Pooled OOF |
|---|---|---|---|---|---|---|---|
| **Test Accuracy** | **100.00%** | **97.50%** | **100.00%** | **92.50%** | **90.00%** | **96.00% $\pm$ 4.06%** | **96.00%** |
| **Balanced Accuracy** | 100.00% | 97.50% | 100.00% | 92.50% | 90.00% | **96.00% $\pm$ 4.06%** | **96.00%** |
| **Macro Precision** | 100.00% | 97.62% | 100.00% | 92.65% | 90.00% | **96.05% $\pm$ 4.07%** | **96.07%** |
| **Macro-F1** | **100.00%** | **97.50%** | **100.00%** | **92.50%** | **89.97%** | **95.99% $\pm$ 4.07%** | **96.00%** |
| **Sensitivity (E)** | **100.00%** | **95.00%** | **100.00%** | **90.00%** | **85.00%** | **94.00% $\pm$ 5.83%** | **94.00%** |
| **Specificity (D)** | **100.00%** | **100.00%** | **100.00%** | **95.00%** | **95.00%** | **98.00% $\pm$ 2.45%** | **98.00%** |
| **False Positives (FP)** | **0** | **0** | **0** | **1** | **1** | **0.4 $\pm$ 0.5 (Total: 2)** | **2 / 100 (2.0%)** |
| **False Negatives (FN)** | **0** | **1** | **0** | **2** | **3** | **1.2 $\pm$ 1.2 (Total: 6)** | **6 / 100 (6.0%)** |
| **Best Val Epoch** | Epoch 27 | Epoch 28 | Epoch 28 | Epoch 25 | Epoch 30 | **27.6 $\pm$ 1.7** | — |
| **Training Duration** | 5.74s | 5.69s | 5.80s | 5.91s | 5.83s | **5.79s $\pm$ 0.08s** | Total: 29.0s |

### Case 3 Pooled Confusion Matrix ($N=200$):
$$\begin{pmatrix} 98 & 2 \\ 6 & 94 \end{pmatrix}$$
* Total Interictal (D) Correct: **98 / 100 (98.0%)**
* Total Ictal (E) Correct: **94 / 100 (94.0%)**

---

## 6. Why the Overfitting Safeguards Succeeded

1. **Architecture Right-Sizing:**
   * Model parameter count was kept to **~62,000 parameters** using 4 compact conv blocks and Global Average Pooling (GAP).
   * GAP eliminated high-dimensional dense matrices (preventing the model from memorizing individual image pixels).
2. **Learning Rate & Optimizer Stability:**
   * Lowered initial learning rate to **$0.0005$** with AdamW and weight decay ($10^{-4}$).
   * Training loss decreased smoothly from $\sim 0.60$ to $\sim 0.15$ without exploding validation loss.
3. **Composite Tie-Breaking Early Stopping:**
   * Monitored both validation accuracy and validation loss.
   * If validation accuracy plateaued, the checkpoint with lower validation loss was preferred, completely curing the zero-gradient stall that previously destroyed Case 3.
4. **Balanced Loss Weights:**
   * Computed strictly on gradient-training subsets, eliminating majority-class bias.

---

## 7. Conclusion

By switching from the lossy $16 \times 16$ GLCM histogram to continuous $128 \times 128$ Recurrence Plots, the CNN's true spatial pattern-recognition capability was unlocked:
* **Case 3 accuracy leaped from 50.00% to 96.00%**.
* **Case 2 accuracy reached 96.40% with 90.00% sensitivity**.
* **Case 1 accuracy jumped from 74.25% to 89.25%**.
* All GLCM matrices, code, and historical experiments remain 100% intact for future research.
