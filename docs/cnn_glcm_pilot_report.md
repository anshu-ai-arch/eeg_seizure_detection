# Bonn EEG Research Project — 5-Recording Real-Data GLCM Pilot Verification Report

**Document Version:** 1.0.0  
**Date:** October 13, 2026  
**Status:** VERIFICATION COMPLETE — VERDICT: PASS  
**Branch:** `fix/bonn-replication-audit`  
**Repository:** `anshu-ai-arch/eeg`  

---

## 1. Executive Summary & Final Verdict

The approved **5-recording real-data GLCM export pilot** was executed and independently audited from the raw Bonn EEG dataset. All generated multi-channel `.npy` tensors and the accompanying JSON manifest were validated using a standalone, read-only verification procedure.

### Final Verdict:
$$\mathbf{PASS\ —\ READY\ FOR\ CNN\ SMOKE\ TEST}$$

Every safety constraint, shape specification, numerical property, channel order mapping, and immutability check passed with zero errors, zero warnings, and zero data discrepancies.

---

## 2. Command Executed and Python Environment

### Executed Command:
```bash
PATH="/opt/anaconda3/bin:$PATH" python scripts/cnn/export_glcm_matrices.py \
  --input data/bonn_eeg/extracted \
  --output data/processed/glcm_matrices_cnn_pilot \
  --pilot 5
```

* **Execution Working Directory:** `/Users/anshutiwari123/phase_sync_eeg` (project root).
* **Execution Duration:** 11.2 seconds (~2.2s per recording including phase-space trajectory reconstruction, continuous GRP calculation, $K=16$ quantization, and GLCM tensor stacking).
* **Exit Code:** `0` (clean exit).

### Python Runtime Environment:
* **Interpreter:** `/opt/anaconda3/bin/python` (Python 3.13.9)
* **Core Libraries:**
  * NumPy: `2.3.5`
  * SciPy: `1.16.3`
  * PyTorch: `2.12.0`
  * Pandas: `2.2.3`
  * Pytest: `8.4.2`

---

## 3. Selected Recording IDs and Labels

The exporter deterministically selected the first five recordings according to alphabetical ordering within subset A:

| Index | Recording ID | File Name | Raw Source Path | Subset | Bonn Class | Case 1 Label (3-Class) | Case 2 Label (Binary) | Case 3 Label (Focal) |
|---|---|---|---|---|---|---|---|---|
| **1** | `Z001` | `Z001.txt` | `data/bonn_eeg/extracted/A/Z001.txt` | A | `healthy` | `healthy` | `non_ictal` | *None (excluded)* |
| **2** | `Z002` | `Z002.txt` | `data/bonn_eeg/extracted/A/Z002.txt` | A | `healthy` | `healthy` | `non_ictal` | *None (excluded)* |
| **3** | `Z003` | `Z003.txt` | `data/bonn_eeg/extracted/A/Z003.txt` | A | `healthy` | `healthy` | `non_ictal` | *None (excluded)* |
| **4** | `Z004` | `Z004.txt` | `data/bonn_eeg/extracted/A/Z004.txt` | A | `healthy` | `healthy` | `non_ictal` | *None (excluded)* |
| **5** | `Z005` | `Z005.txt` | `data/bonn_eeg/extracted/A/Z005.txt` | A | `healthy` | `healthy` | `non_ictal` | *None (excluded)* |

*Note on Case 3:* As defined by the ground-truth benchmark protocol, Case 3 evaluates only interictal focal epilepsy (Subset D) versus ictal (Subset E). Recordings from Subset A (healthy) are not included in Case 3, and their `case3_label` is appropriately set to `null` (`None`).

---

## 4. Exported File Count and Manifest Status

* **Designated Output Directory:** `data/processed/glcm_matrices_cnn_pilot/`
* **Total Exported Recordings:** Exactly **5**.
* **Total `.npy` Tensor Files:** Exactly **5** (in `data/processed/glcm_matrices_cnn_pilot/matrices/`).
* **Individual `.npy` File Size:** Exactly **8,320 bytes** each:
  $$\text{Payload} = 8 \times 16 \times 16 \times 4 \text{ bytes (float32)} = 8,192 \text{ bytes}$$
  $$\text{NumPy Header} = 128 \text{ bytes}$$
  $$\text{Total File Size} = 8,192 + 128 = 8,320 \text{ bytes}$$
* **Manifest Path:** `data/processed/glcm_matrices_cnn_pilot/glcm_manifest.json` (4,817 bytes).
* **Export Log:** `data/processed/glcm_matrices_cnn_pilot/export_glcm.log` (703 bytes).
* **Uniqueness & Confinement:**
  * Every recording appears exactly once in the manifest and on disk.
  * All output files reside strictly within the designated pilot directory.
  * Zero duplicate entries, zero missing files, and zero extraneous files detected.

---

## 5. Tensor Shapes, Dtypes, and Numerical Integrity

Each `.npy` file was loaded independently using `np.load(..., allow_pickle=False)`:

| Recording ID | Shape | Dtype | Finite | Min Value | Max Value | Negative Values? | Values > 1.0? | Max Deviation of Channel Sums from 1.0 |
|---|---|---|---|---|---|---|---|---|
| `Z001` | `(8, 16, 16)` | `float32` | **True** | `0.00000000` | `0.16184680` | **False** | **False** | $1.19 \times 10^{-7}$ |
| `Z002` | `(8, 16, 16)` | `float32` | **True** | `0.00000000` | `0.14667721` | **False** | **False** | $5.96 \times 10^{-8}$ |
| `Z003` | `(8, 16, 16)` | `float32` | **True** | `0.00000000` | `0.13385411` | **False** | **False** | $5.96 \times 10^{-8}$ |
| `Z004` | `(8, 16, 16)` | `float32` | **True** | `0.00000000` | `0.10804662` | **False** | **False** | $5.96 \times 10^{-8}$ |
| `Z005` | `(8, 16, 16)` | `float32` | **True** | `0.00000000` | `0.16968328` | **False** | **False** | $5.96 \times 10^{-8}$ |

### Channel Sum Audit:
* **Tolerance Tested:** $\text{atol} = 1 \times 10^{-5}$.
* **Observed Maximum Deviation:** $1.19 \times 10^{-7}$ (governed strictly by IEEE 754 single-precision float summation limits).
* Every individual channel in every exported tensor represents a strictly valid 2D discrete probability distribution:
  $$\forall c \in \{0, \dots, 7\}: \quad \sum_{i=1}^{16} \sum_{j=1}^{16} P_c(i, j) \approx 1.0$$

---

## 6. Channel Order and Independent GLCM Recomputation

### Canonical Channel Specification:
The canonical 8-channel layout in `scripts/cnn/export_glcm_matrices.py` directly mirrors `scripts/preprocessing/glcm.py`:
0. `theta_0deg_d_1` ($\theta = 0^\circ, d = 1$)
1. `theta_0deg_d_2` ($\theta = 0^\circ, d = 2$)
2. `theta_0deg_d_3` ($\theta = 0^\circ, d = 3$)
3. `theta_0deg_d_4` ($\theta = 0^\circ, d = 4$)
4. `theta_45deg_d_1` ($\theta = 45^\circ, d = 1$)
5. `theta_45deg_d_2` ($\theta = 45^\circ, d = 2$)
6. `theta_45deg_d_3` ($\theta = 45^\circ, d = 3$)
7. `theta_45deg_d_4` ($\theta = 45^\circ, d = 4$)

### Numerical Recomputation Comparison:
Recordings `Z001` and `Z002` were preprocessed independently directly from their raw `.txt` source files using the baseline `preprocess_recording` function:

| Recording ID | Recomputed Shape | Exported Shape | Max Absolute Difference | Mean Absolute Difference | Strict Bitwise Array Equality |
|---|---|---|---|---|---|
| `Z001` | `(8, 16, 16)` | `(8, 16, 16)` | $\mathbf{0.00 \times 10^0}$ | $\mathbf{0.00 \times 10^0}$ | **True (`np.array_equal`)** |
| `Z002` | `(8, 16, 16)` | `(8, 16, 16)` | $\mathbf{0.00 \times 10^0}$ | $\mathbf{0.00 \times 10^0}$ | **True (`np.array_equal`)** |

The exported multi-channel tensors are bit-for-bit identical to freshly recomputed baseline matrices.

---

## 7. SHA-256 Checksum Verification

Every `.npy` file on disk was independently hashed and compared against the recorded manifest entry:

| Recording ID | File Path | Manifest Recorded SHA-256 | Disk Computed SHA-256 | Checksum Status |
|---|---|---|---|---|
| `Z001` | `matrices/Z001.npy` | `bf20ab3fd22f3ffed61d165c5490232dcb0fe73f102d6a20b7e7a4505310a6f1` | `bf20ab3fd22f3ffed61d165c5490232dcb0fe73f102d6a20b7e7a4505310a6f1` | **MATCH** |
| `Z002` | `matrices/Z002.npy` | `62bb195c1856b8b316e2bf6737e2b6b250f904fcd582caae0813d551cc6eee0c` | `62bb195c1856b8b316e2bf6737e2b6b250f904fcd582caae0813d551cc6eee0c` | **MATCH** |
| `Z003` | `matrices/Z003.npy` | `18fdd206dde10c35be60048f5fd489da850a4405648742745a82439722fef5ba` | `18fdd206dde10c35be60048f5fd489da850a4405648742745a82439722fef5ba` | **MATCH** |
| `Z004` | `matrices/Z004.npy` | `5f83656ebc5b70988e0458a766e89a2b87a6ca0df89b33e28f8b6914bc72659b` | `5f83656ebc5b70988e0458a766e89a2b87a6ca0df89b33e28f8b6914bc72659b` | **MATCH** |
| `Z005` | `matrices/Z005.npy` | `fc970eb20a92165cfa523bb426d37f740c845b9fd1d887c7cd528e3e5d632377` | `fc970eb20a92165cfa523bb426d37f740c845b9fd1d887c7cd528e3e5d632377` | **MATCH** |

---

## 8. Frozen Baseline CSV and Repository Immutability

### Frozen Feature CSV:
* **Path:** `audit/batch_preprocessing/bonn_features_96.csv`
* **Recorded Baseline SHA-256:** `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`
* **Current Post-Pilot SHA-256:**  `af8d3d2f1c401e73fe2120bb823b370423b08cb02d55ce2432d3121813a9b9d1`
* **Status:** **100% UNCHANGED / IMMUTABLE**.

### Repository Working Tree & SVM Artifacts:
* **Current Git Branch:** `fix/bonn-replication-audit`
* **Tracked Changes (`git diff --stat`):** 0 files changed, 0 insertions, 0 deletions.
* **SVM Baseline Source Files:** `scripts/classification/run_kaggle_tuned.py`, `scripts/classification/mcfs.py`, etc., are completely untouched.
* **Historical Nested-CV Checkpoints:** Checkpoints in `experiments/experiment_k5/` remain completely intact.
* **Repository Test Suite:** All 74 unit tests pass (`pytest tests/ -v`).

---

## 9. Warnings, Failures, Limitations, or Deviations

* **Failures:** None.
* **Warnings:** None.
* **Deviations:** None. The exporter strictly processed 5 recordings as specified by `--pilot 5`, and did not trigger a full 500-recording export.
* **Verification Boundary:** As instructed, model training, nested cross-validation, and full-dataset export were **not** initiated.

---

## 10. Conclusion and Next Steps

The 5-recording real-data GLCM pilot export is successfully validated. The multi-channel GLCM tensor exporter is verified to produce lossless `[8, 16, 16]` float32 representations directly compatible with `BonnGLCMCNN` and `BonnGLCMDataset`.

**Execution is halted. Ready for user review and explicit approval before initiating the CNN smoke test.**
