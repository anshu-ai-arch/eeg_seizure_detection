# Bonn EEG Research Project — 5-Fold Stratified Cross-Validation Report (GLCM-CNN)

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** COMPLETE — 15 MODEL FITS EXECUTED & AUDITED  
**Branch:** `fix/bonn-replication-audit`  
**Hardware Environment:** Apple Silicon M1 (MacBook Air, 8 GB RAM, PyTorch 2.12.0 with MPS backend)  
**Experiment Root:** `experiments/results_cnn_glcm_5fold_20261013_132700/`  

---

## 1. Executive Summary

This report documents the rigorous evaluation of the **Bonn EEG multi-channel GLCM-CNN branch** across **5-fold stratified cross-validation**. The model consumes continuous 8-channel GLCM matrix tensors of shape `[batch_size, 8, 16, 16]`.

All 15 fold fits (5 folds $\times$ 3 classification cases) completed with strict partition disjointness, zero test data leakage, independent fold reinitialization, and reproducible seeds.

### Key Empirical Findings:
1. **Case 1 (Three-Class, A vs B vs C+D, $N=400$):**
   * **Fold-level Mean $\pm$ Std:** Accuracy = **$74.25\% \pm 7.94\%$**, Balanced Accuracy = **$70.00\% \pm 6.21\%$**, Macro-F1 = **$68.57\% \pm 7.67\%$**.
   * **Pooled Out-of-Fold (OOF) Accuracy:** **$74.25\%$** (297/400).
   * **Class Dynamics:** Strongest discrimination on epileptic interictal activity (Subset C+D: Recall = 87.0%, Precision = 95.1%, F1 = 90.9%). Moderate confusion between healthy eyes-open (A) and eyes-closed (B).
2. **Case 2 (Binary, A+B+C+D vs E, $N=500$):**
   * **Fold-level Mean $\pm$ Std:** Accuracy = **$91.40\% \pm 3.21\%$**, Balanced Accuracy = **$84.12\% \pm 7.09\%$**, Macro-F1 = **$85.58\% \pm 5.92\%$**.
   * **Pooled OOF Accuracy:** **$91.40\%$** (457/500), Specificity = **$96.25\%$** (385/400), Sensitivity = **$72.00\%$** (72/100).
3. **Case 3 (Binary, D vs E, $N=200$):**
   * **Fold-level Mean $\pm$ Std:** Accuracy = **$50.00\% \pm 0.00\%$**, Macro-F1 = **$33.33\% \pm 0.00\%$**.
   * **Class Collapse:** Complete collapse across all 5 folds. The model predicted 100% of samples as Class 1 (ictal E), yielding 0% sensitivity on interictal D and 100% sensitivity on ictal E.

---

## 2. Experimental Protocol & Partition Integrity

### Cross-Validation Design:
* **Outer Splits:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
* **Outer Test Sets:** Exactly 1 fold (20%) held out per fit. Every recording in the dataset is tested exactly once.
* **Inner Splits:** 90% of development portion used for gradient training, 10% for inner validation and checkpoint selection (`StratifiedShuffleSplit` with deterministic fold-specific seed).
* **Isolation Guarantee:** $S_{\text{subtrain}} \cap S_{\text{val}} = \emptyset$, $S_{\text{subtrain}} \cap S_{\text{test}} = \emptyset$, $S_{\text{val}} \cap S_{\text{test}} = \emptyset$. Zero test leakage.
* **Class Weights:** Computed strictly from the sub-training subset ($w_c = N_{\text{subtrain}} / (C \cdot N_{\text{subtrain}, c})$). Never computed from validation or test subsets.
* **Model Independence:** Every fold instantiated a completely fresh `BonnGLCMCNN` model, AdamW optimizer, and ReduceLROnPlateau scheduler. No state carried over.

### Exact Partition Counts:

| Case | Total Samples | Test per Fold (20%) | Development Pool (80%) | Sub-Train (~72%) | Inner Val (~8%) | Sub-Train Class Proportions | Test Class Proportions |
|---|---|---|---|---|---|---|---|
| **Case 1 (3-Class)** | 400 | **80** | 320 | **288** | **32** | 72 / 72 / 144 | 20 / 20 / 40 |
| **Case 2 (Binary)** | 500 | **100** | 400 | **360** | **40** | 288 / 72 | 80 / 20 |
| **Case 3 (Binary)** | 200 | **40** | 160 | **144** | **16** | 72 / 72 | 20 / 20 |

---

## 3. Case 1: Three-Class Classification (A vs B vs C+D, Exclude E)

### Per-Fold Performance:

| Fold Index | Seed | Best Epoch | Total Epochs Trained | Best Val Acc | Test Accuracy | Balanced Acc | Macro-F1 | Macro-Precision | Macro-Recall | Duration |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 0** | 42 | 6 | 11 | 84.38% | **77.50%** | 73.33% | 73.27% | 73.71% | 73.33% | 3.57s |
| **Fold 1** | 142 | 12 | 17 | 81.25% | **76.25%** | 68.33% | 68.21% | 71.88% | 68.33% | 4.85s |
| **Fold 2** | 242 | 7 | 12 | 68.75% | **73.75%** | 70.00% | 66.59% | 68.63% | 70.00% | 3.43s |
| **Fold 3** | 342 | 6 | 11 | 71.88% | **82.50%** | 77.50% | 77.54% | 84.52% | 77.50% | 3.23s |
| **Fold 4** | 442 | 4 | 9 | 65.62% | **61.25%** | 60.83% | 57.23% | 59.90% | 60.83% | 2.57s |
| **Mean $\pm$ Std** | — | **7.0** | **12.0** | **74.38% $\pm$ 8.16%** | **74.25% $\pm$ 7.94%** | **70.00% $\pm$ 6.21%** | **68.57% $\pm$ 7.67%** | **71.73% $\pm$ 8.91%** | **70.00% $\pm$ 6.21%** | **3.53s** |

### Pooled Out-of-Fold Confusion Matrix ($N=400$):
$$\begin{pmatrix}
54 & 37 & 9 \\
31 & 69 & 0 \\
25 & 1 & 174
\end{pmatrix}$$

* **Row 0: True A ($N=100$):** 54 predicted as A, 37 as B, 9 as C+D (Recall: 54.0%, Precision: 49.1%, F1: 51.4%)
* **Row 1: True B ($N=100$):** 31 predicted as A, 69 as B, 0 as C+D (Recall: 69.0%, Precision: 64.5%, F1: 66.7%)
* **Row 2: True C+D ($N=200$):** 25 predicted as A, 1 as B, 174 as C+D (Recall: 87.0%, Precision: 95.1%, F1: 90.9%)
* **Pooled OOF Accuracy:** **$74.25\%$**
* **Pooled Macro-F1:** **$69.65\%$**

---

## 4. Case 2: Binary Classification (A+B+C+D vs E)

### Per-Fold Performance:

| Fold Index | Seed | Best Epoch | Total Epochs | Best Val Acc | Test Accuracy | Balanced Acc | Macro-F1 | Sensitivity (Ictal) | Specificity (Non-Ictal) | Duration |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 0** | 42 | 8 | 13 | 90.00% | **89.00%** | 72.50% | 77.82% | 45.00% | 100.00% | 4.92s |
| **Fold 1** | 142 | 4 | 9 | 95.00% | **93.00%** | 86.25% | 88.39% | 75.00% | 97.50% | 3.19s |
| **Fold 2** | 242 | 7 | 12 | 95.00% | **91.00%** | 85.00% | 85.67% | 75.00% | 95.00% | 4.29s |
| **Fold 3** | 342 | 11 | 16 | 95.00% | **88.00%** | 85.00% | 82.52% | 80.00% | 90.00% | 5.71s |
| **Fold 4** | 442 | 7 | 12 | 95.00% | **96.00%** | 91.88% | 93.50% | 85.00% | 98.75% | 4.26s |
| **Mean $\pm$ Std** | — | **7.4** | **12.4** | **94.00% $\pm$ 2.24%** | **91.40% $\pm$ 3.21%** | **84.12% $\pm$ 7.09%** | **85.58% $\pm$ 5.92%** | **72.00% $\pm$ 15.65%** | **96.25% $\pm$ 3.95%** | **4.47s** |

### Pooled Out-of-Fold Confusion Matrix ($N=500$):
$$\begin{pmatrix}
385 & 15 \\
28 & 72
\end{pmatrix}$$

* **Non-Ictal ($N=400$):** Precision = 93.22%, Recall = 96.25% (Specificity), F1 = 94.71%
* **Ictal ($N=100$):** Precision = 82.76%, Recall = 72.00% (Sensitivity), F1 = 77.01%
* **Pooled OOF Accuracy:** **$91.40\%$**
* **Pooled Macro-F1:** **$85.86\%$**

---

## 5. Case 3: Binary Classification (D vs E, Exclude A, B, C)

### Per-Fold Performance:

| Fold Index | Seed | Best Epoch | Total Epochs | Best Val Acc | Test Accuracy | Balanced Acc | Macro-F1 | Sensitivity (Ictal E) | Specificity (Interictal D) | Duration |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fold 0** | 42 | 1 | 6 | 50.00% | **50.00%** | 50.00% | 33.33% | 100.00% | 0.00% | 0.97s |
| **Fold 1** | 142 | 1 | 6 | 50.00% | **50.00%** | 50.00% | 33.33% | 100.00% | 0.00% | 0.82s |
| **Fold 2** | 242 | 1 | 6 | 50.00% | **50.00%** | 50.00% | 33.33% | 100.00% | 0.00% | 0.82s |
| **Fold 3** | 342 | 1 | 6 | 50.00% | **50.00%** | 50.00% | 33.33% | 100.00% | 0.00% | 0.81s |
| **Fold 4** | 442 | 1 | 6 | 50.00% | **50.00%** | 50.00% | 33.33% | 100.00% | 0.00% | 0.82s |
| **Mean $\pm$ Std** | — | **1.0** | **6.0** | **50.00% $\pm$ 0.00%** | **50.00% $\pm$ 0.00%** | **50.00% $\pm$ 0.00%** | **33.33% $\pm$ 0.00%** | **100.00% $\pm$ 0.00%** | **0.00% $\pm$ 0.00%** | **0.85s** |

### Pooled Out-of-Fold Confusion Matrix ($N=200$):
$$\begin{pmatrix}
0 & 100 \\
0 & 100
\end{pmatrix}$$

### Detailed Analysis of Case 3 Class Collapse:
Case 3 evaluates interictal focal epileptogenic zone activity (Subset D) versus ictal seizure activity (Subset E). Across all 5 folds, the model collapsed entirely into predicting Class 1 (ictal E):
1. **Sample Size & Batch Dynamics:** The development pool for Case 3 contains only 160 samples (144 gradient-training, 16 validation). With a batch size of 16, each epoch consists of only 9 gradient steps.
2. **Learning Rate & Optimization Pathology:** The initial learning rate of `0.005` (fixed by protocol) caused the validation accuracy to tie at exactly `50.00%` across the first 5 epochs. Under the strict improvement requirement ($\Delta > 0.0$, ties retain the earliest best), Epoch 1 was selected.
3. **Early Stopping Restoring Trivial Solution:** Because no validation epoch exceeded 50.00% accuracy, early stopping was triggered at Epoch 6 in every fold ($1 + 5 = 6$). Restoring Epoch 1 weights reverted the network to an un-converged state where all logits favored Class 1.

---

## 6. Comparison with Single 80:20 Split and Historical Baseline

| Benchmark Dimension | Case 1 (3-Class) | Case 2 (Binary) | Case 3 (Binary) | Evaluation Protocol |
|---|---|---|---|---|
| **Earlier Single 80:20 Split (Held-Out Test)** | 75.00% | 90.00% | 50.00% | Single outer 80:20 split (1 seed) |
| **5-Fold CV: Mean $\pm$ Std** | **$74.25\% \pm 7.94\%$** | **$91.40\% \pm 3.21\%$** | **$50.00\% \pm 0.00\%$** | Stratified 5-Fold CV (Outer test mean) |
| **5-Fold CV: Pooled OOF Accuracy** | **$74.25\%$** (297/400) | **$91.40\%$** (457/500) | **$50.00\%$** (100/200) | Concatenated out-of-fold predictions |
| **Historical SVM Baseline (Nested 10-Fold CV)** | 93.80% | 96.80% | 95.00% | Nested 10-Fold CV + Haralick (96) + MCFS |

### Essential Differences from the SVM Baseline (Non-Comparable):
1. **Cross-Validation Scope:** The SVM baseline used **nested 10-fold cross-validation** (10 outer folds with an inner grid search over feature count $k \in \{5, 10, 15, 20\}$ and RBF parameter $\gamma$). This experiment evaluated **non-nested 5-fold cross-validation**.
2. **Case 1 Definition:** In the historical SVM benchmark, Case 1 was defined as $A+B$ (200) vs $C+D$ (200) vs $E$ (100) totaling 500 recordings. The current CNN Case 1 is defined as $A$ (100) vs $B$ (100) vs $C+D$ (200) with $E$ excluded (400 recordings total).
3. **Input Representation & Feature Selection:** The SVM pipeline extracted 96 scalar Haralick features and applied unsupervised Laplacian graph feature selection (MCFS) to select between 5 and 20 optimal features. The CNN pipeline operates on raw $8 \times 16 \times 16$ 2D spatial probability distributions without feature selection.

---

## 7. Artifact Directory Structure

The complete 5-fold cross-validation outputs are stored under:
`experiments/results_cnn_glcm_5fold_20261013_132700/`

```
experiments/results_cnn_glcm_5fold_20261013_132700/
├── config.json                               # Configuration schema
├── all_cases_summary.json                    # Consolidated metrics across all 15 fits
├── execution.log                             # Complete execution log
├── Case1_3Class/
│   ├── split_manifest.json                   # Recording IDs and fold indices
│   ├── fold_metrics_table.csv                # Tabular metrics for folds 0-4
│   ├── all_folds_test_predictions.csv        # Concatenated out-of-fold predictions (N=400)
│   ├── case_summary.json                     # Case-level mean/std and pooled metrics
│   ├── fold_0/ .. fold_4/
│   │   ├── best_checkpoint.pt               # PyTorch model weights (~125 KB)
│   │   ├── test_predictions.csv             # Fold test set predictions & probabilities
│   │   ├── training_history.csv             # Epoch-by-epoch loss, acc, lr
│   │   └── metrics.json                      # Comprehensive fold metrics
├── Case2_Binary/
│   ├── split_manifest.json, fold_metrics_table.csv, all_folds_test_predictions.csv (N=500)
│   └── fold_0/ .. fold_4/ [checkpoints, predictions, history, metrics]
└── Case3_Binary/
    ├── split_manifest.json, fold_metrics_table.csv, all_folds_test_predictions.csv (N=200)
    └── fold_0/ .. fold_4/ [checkpoints, predictions, history, metrics]
```

### Git Repository State:
* Frozen feature CSV hash: `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1` (verified immutable).
* Historical SVM checkpoints and prior single-split CNN experiments remain untouched.
* Checkpoints are retained locally (15 checkpoints $\times$ 125 KB $\approx$ 1.88 MB total). Zero checkpoints pushed to GitHub.
* All 77 unit tests in the repository pass.

---

## 8. Conclusion

Stratified 5-fold cross-validation provides an unbiased, variance-quantified baseline for the multi-channel GLCM-CNN on the Bonn EEG dataset:
* The CNN demonstrates competitive performance on the 500-recording binary task (**$91.40\% \pm 3.21\%$**) and the 400-recording three-class task (**$74.25\% \pm 7.94\%$**).
* On the 200-recording focal task (Case 3), the fixed optimization hyperparameters ($lr=0.005$) lead to severe optimization failure and class collapse (**$50.00\%$**), indicating that learning rate tuning or warm-up is mandatory for smaller EEG subsets.
