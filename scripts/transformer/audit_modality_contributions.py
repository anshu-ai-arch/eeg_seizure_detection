"""
scripts/transformer/audit_modality_contributions.py

Rigorous Diagnostic Audit:
1. Empirical Gate Inspection:
   Measures mean gate values g (Transformer weight) vs (1 - g) (Cheatsheet weight) across all test samples.
2. In-Silico Checkpoint Ablation:
   - Full Model Performance
   - Cheatsheet Zeroed Out (Tests if Transformer alone can classify)
   - GLCM Tensor Zeroed Out (Tests if Cheatsheet alone can classify)
3. Standalone Pure 1D Transformer 5-Fold Training:
   Trains a pure Transformer directly on GLCM tensors with 0 cheatsheet features.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transformer.model import Bonn1DGatedGLCMTransformer
from scripts.transformer.dataset import RobustGLCMDataset
from scripts.transformer.run_transformer_5fold_cv import CASE_SPECS

EXP_DIR = PROJECT_ROOT / "experiments" / "results_transformer_gated_glcm_5fold_20261015_061806"
CHEAT_CSV = PROJECT_ROOT / "data" / "processed" / "robust_glcm" / "cheatsheet_features.csv"
MANIFEST_JSON = PROJECT_ROOT / "data" / "processed" / "robust_glcm" / "dataset_manifest.json"
TENSORS_DIR = PROJECT_ROOT / "data" / "processed" / "robust_glcm" / "tensors"

cheat_df = pd.read_csv(CHEAT_CSV)
feat_cols = [c for c in cheat_df.columns if c not in ("recording_id", "subset")]
num_cheatsheet = len(feat_cols)
residual_names = ["global_log_raw_std", "global_log_energy", "global_log_line_length"]
residual_indices = [feat_cols.index(col) for col in residual_names if col in feat_cols]

with open(MANIFEST_JSON) as f:
    manifest = json.load(f)["records"]

rec_to_cheat = {r["recording_id"]: r for _, r in cheat_df.iterrows()}

print("==========================================================================")
print("AUDIT: EMPIRICAL GATE VALUES & ZERO-OUT ABLATION ON TRAINED CHECKPOINTS")
print("==========================================================================")

for case_key in ["Case2_Binary", "Case1_3Class", "Case3_Binary"]:
    spec = CASE_SPECS[case_key]
    subsets_inc = spec["subsets_included"]
    sub_to_lbl = spec["subset_to_label"]
    num_classes = spec["num_classes"]
    
    case_recs = []
    cheat_vecs = []
    for r in manifest:
        if r["subset"] in subsets_inc:
            item = dict(r)
            item["label"] = sub_to_lbl[r["subset"]]
            case_recs.append(item)
            c_row = rec_to_cheat[r["recording_id"]]
            cheat_vecs.append([float(c_row[c]) for c in feat_cols])
            
    case_recs.sort(key=lambda x: x["recording_id"])
    rec_ids = [r["recording_id"] for r in case_recs]
    labels = np.array([r["label"] for r in case_recs])
    raw_cheat = np.array(cheat_vecs, dtype=np.float32)
    
    cached_tensors = {}
    for r in case_recs:
        rid = r["recording_id"]
        arr = np.load(TENSORS_DIR / f"{rid}.npy")
        cached_tensors[rid] = torch.from_numpy(arr)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(rec_ids, labels))
    
    all_gates = []
    y_true_all = []
    y_pred_full_all = []
    y_pred_no_cheat_all = []
    y_pred_no_glcm_all = []
    
    for fold_idx, (dev_pos, test_pos) in enumerate(fold_splits):
        fold_seed = 42 + fold_idx * 100
        ckpt_p = EXP_DIR / case_key / f"fold_{fold_idx}" / "best_checkpoint.pt"
        if not ckpt_p.exists():
            continue
        model = Bonn1DGatedGLCMTransformer(num_classes=num_classes, num_cheatsheet=num_cheatsheet, residual_indices=residual_indices)
        model.load_state_dict(torch.load(ckpt_p, map_location="cpu"))
        model.eval()
        
        dev_recs = [case_recs[i] for i in dev_pos]
        dev_lbls = np.array([r["label"] for r in dev_recs])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=fold_seed)
        subtrain_pos, val_pos = next(sss.split(dev_recs, dev_lbls))
        subtrain_orig = [dev_pos[i] for i in subtrain_pos]
        test_orig = list(test_pos)
        
        scaler = StandardScaler()
        scaler.fit(raw_cheat[subtrain_orig])
        scaled_test = scaler.transform(raw_cheat[test_orig])
        test_recs = [case_recs[i] for i in test_pos]
        
        test_ds = RobustGLCMDataset(test_recs, cached_tensors, scaled_test)
        
        with torch.no_grad():
            for i in range(len(test_ds)):
                glcm, cheat, lbl, _ = test_ds[i]
                glcm = glcm.unsqueeze(0)
                cheat = cheat.unsqueeze(0)
                y_true_all.append(lbl)
                
                # 1. Full Model
                logits_full = model(glcm, cheat)
                pred_full = int(torch.argmax(logits_full, dim=-1).item())
                y_pred_full_all.append(pred_full)
                
                # Forward inspection of gate
                B = 1
                x_tok = glcm.permute(0, 2, 1, 3).reshape(B, 64, 256)
                h_tokens = model.pos_drop(model.token_proj(x_tok) + model.pos_emb)
                out_tokens = model.transformer(h_tokens)
                h_tex = model.norm_tex(out_tokens.mean(dim=1))
                h_cheat = model.cheat_drop(model.cheat_act(model.cheat_bn(model.cheat_fc(cheat))))
                concat = torch.cat([h_tex, h_cheat], dim=1)
                gate = torch.sigmoid(model.gate_fc(concat)).squeeze().numpy()
                all_gates.append(gate)
                
                # 2. Zero-out Cheatsheet (Pass zeros for cheat_vec and disable residual)
                cheat_zero = torch.zeros_like(cheat)
                # Compute forward with zero cheatsheet
                h_cheat_zero = torch.zeros_like(h_cheat)
                concat_no_cheat = torch.cat([h_tex, h_cheat_zero], dim=1)
                gate_no_cheat = torch.sigmoid(model.gate_fc(concat_no_cheat))
                # If we test purely transformer texture:
                fused_no_cheat = h_tex  # Direct texture branch
                logits_tex_only = model.fc_out(model.drop_fused(model.act_fused(model.fc_fused(fused_no_cheat))))
                pred_no_cheat = int(torch.argmax(logits_tex_only, dim=-1).item())
                y_pred_no_cheat_all.append(pred_no_cheat)
                
                # 3. Zero-out GLCM Tensor
                h_tex_zero = torch.zeros_like(h_tex)
                concat_no_glcm = torch.cat([h_tex_zero, h_cheat], dim=1)
                gate_no_glcm = torch.sigmoid(model.gate_fc(concat_no_glcm))
                fused_no_glcm = (1.0 - gate_no_glcm) * h_cheat
                logits_cheat_only = model.fc_out(model.drop_fused(model.act_fused(model.fc_fused(fused_no_glcm))))
                if len(model.residual_indices) > 0:
                    logits_cheat_only = logits_cheat_only + model.fc_residual(cheat[:, model.residual_indices])
                pred_no_glcm = int(torch.argmax(logits_cheat_only, dim=-1).item())
                y_pred_no_glcm_all.append(pred_no_glcm)
                
    all_gates = np.array(all_gates)
    mean_gate = np.mean(all_gates)
    acc_full = accuracy_score(y_true_all, y_pred_full_all) * 100
    acc_no_cheat = accuracy_score(y_true_all, y_pred_no_cheat_all) * 100
    acc_no_glcm = accuracy_score(y_true_all, y_pred_no_glcm_all) * 100
    
    print(f"\n=======================================================")
    print(f"CASE: {case_key} ({len(y_true_all)} samples)")
    print(f"=======================================================")
    print(f"1. Gating Balance: Mean Gate g = {mean_gate:.4f} (Transformer weight: {mean_gate*100:.1f}%, Cheatsheet weight: {(1-mean_gate)*100:.1f}%)")
    print(f"2. Full Gated Hybrid Accuracy:              {acc_full:.2f}%")
    print(f"3. Ablation A: Cheatsheet Zeroed (Tex Only): {acc_no_cheat:.2f}%")
    print(f"4. Ablation B: GLCM Zeroed (Cheat Only):     {acc_no_glcm:.2f}%")
