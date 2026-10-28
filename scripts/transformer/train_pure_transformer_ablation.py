"""
scripts/transformer/train_pure_transformer_ablation.py

Trains:
1. Pure 1D Transformer (GLCM only, 0 cheatsheet features, 0 residual connections).
2. Pure Cheatsheet MLP (Cheatsheet only, 0 GLCM).
Compares with Full Gated Hybrid.
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
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transformer.dataset import RobustGLCMDataset
from scripts.transformer.run_transformer_5fold_cv import CASE_SPECS, select_device, compute_balanced_class_weights

# -------------------------------------------------------------
# Standalone Model A: Pure 1D Transformer (No Cheatsheet)
# -------------------------------------------------------------
class Pure1DGLCMTransformer(nn.Module):
    def __init__(self, num_classes=2, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.token_proj = nn.Linear(256, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm_tex = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, glcm_tensor, cheatsheet_vec=None):
        B = glcm_tensor.shape[0]
        x_tok = glcm_tensor.permute(0, 2, 1, 3).reshape(B, 64, 256)
        h_tokens = self.pos_drop(self.token_proj(x_tok) + self.pos_emb)
        out_tokens = self.transformer(h_tokens)
        h_tex = self.norm_tex(out_tokens.mean(dim=1))
        return self.classifier(h_tex)


# -------------------------------------------------------------
# Standalone Model B: Pure Cheatsheet MLP (No GLCM)
# -------------------------------------------------------------
class PureCheatsheetMLP(nn.Module):
    def __init__(self, num_classes=2, num_cheatsheet=67, d_model=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cheatsheet, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, glcm_tensor, cheatsheet_vec):
        return self.net(cheatsheet_vec)


def train_and_eval_5fold(model_type, case_key="Case2_Binary", max_epochs=35, device=torch.device("cpu")):
    spec = CASE_SPECS[case_key]
    subsets_inc = spec["subsets_included"]
    sub_to_lbl = spec["subset_to_label"]
    num_classes = spec["num_classes"]

    cheat_df = pd.read_csv(PROJECT_ROOT / "data/processed/robust_glcm/cheatsheet_features.csv")
    feat_cols = [c for c in cheat_df.columns if c not in ("recording_id", "subset")]
    with open(PROJECT_ROOT / "data/processed/robust_glcm/dataset_manifest.json") as f:
        manifest = json.load(f)["records"]

    rec_to_cheat = {r["recording_id"]: r for _, r in cheat_df.iterrows()}

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
        arr = np.load(PROJECT_ROOT / f"data/processed/robust_glcm/tensors/{rid}.npy")
        cached_tensors[rid] = torch.from_numpy(arr)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(rec_ids, labels))

    pooled_y_true, pooled_y_pred = [], []
    fold_accs = []

    for fold_idx, (dev_pos, test_pos) in enumerate(fold_splits):
        fold_seed = 42 + fold_idx * 100
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        dev_recs = [case_recs[i] for i in dev_pos]
        dev_lbls = np.array([r["label"] for r in dev_recs])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=fold_seed)
        subtrain_pos, val_pos = next(sss.split(dev_recs, dev_lbls))
        subtrain_orig = [dev_pos[i] for i in subtrain_pos]
        val_orig = [dev_pos[i] for i in val_pos]
        test_orig = list(test_pos)

        scaler = StandardScaler()
        scaler.fit(raw_cheat[subtrain_orig])
        scaled_subtrain = scaler.transform(raw_cheat[subtrain_orig])
        scaled_val = scaler.transform(raw_cheat[val_orig])
        scaled_test = scaler.transform(raw_cheat[test_orig])

        subtrain_ds = RobustGLCMDataset([dev_recs[i] for i in subtrain_pos], cached_tensors, scaled_subtrain)
        val_ds = RobustGLCMDataset([dev_recs[i] for i in val_pos], cached_tensors, scaled_val)
        test_ds = RobustGLCMDataset([case_recs[i] for i in test_pos], cached_tensors, scaled_test)

        if model_type == "pure_transformer":
            model = Pure1DGLCMTransformer(num_classes=num_classes).to(device)
        else:
            model = PureCheatsheetMLP(num_classes=num_classes, num_cheatsheet=len(feat_cols)).to(device)

        subtrain_lbls = [r["label"] for r in subtrain_ds.records]
        class_weights = compute_balanced_class_weights(subtrain_lbls, num_classes=num_classes, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

        train_loader = DataLoader(subtrain_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        best_val_loss = float("inf")
        best_state = None

        for ep in range(1, max_epochs + 1):
            model.train()
            for bx_glcm, bx_cheat, by, _ in train_loader:
                bx_glcm = bx_glcm.to(device)
                bx_cheat = bx_cheat.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                logits = model(bx_glcm, bx_cheat)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            # Val
            model.eval()
            v_losses, v_corr, v_tot = [], 0, 0
            with torch.no_grad():
                for bx_glcm, bx_cheat, by, _ in val_loader:
                    bx_glcm = bx_glcm.to(device)
                    bx_cheat = bx_cheat.to(device)
                    by = by.to(device)
                    logits = model(bx_glcm, bx_cheat)
                    loss = criterion(logits, by)
                    v_losses.append(loss.item())
                    v_corr += (torch.argmax(logits, dim=-1) == by).sum().item()
                    v_tot += len(by)
            v_loss = np.mean(v_losses)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate on test
        model.load_state_dict(best_state)
        model.eval()
        t_corr, t_tot = 0, 0
        with torch.no_grad():
            for bx_glcm, bx_cheat, by, _ in test_loader:
                bx_glcm = bx_glcm.to(device)
                bx_cheat = bx_cheat.to(device)
                logits = model(bx_glcm, bx_cheat)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                pooled_y_true.extend(by.numpy())
                pooled_y_pred.extend(preds)
                t_corr += (preds == by.numpy()).sum()
                t_tot += len(by)

        fold_acc = t_corr / t_tot
        fold_accs.append(fold_acc)

    pooled_acc = accuracy_score(pooled_y_true, pooled_y_pred) * 100
    mean_acc = np.mean(fold_accs) * 100
    std_acc = np.std(fold_accs) * 100
    return pooled_acc, mean_acc, std_acc


if __name__ == "__main__":
    device = select_device("auto")
    print(f"Running Ablation on Device: {device}\n")

    for case in ["Case2_Binary", "Case1_3Class", "Case3_Binary"]:
        print(f"===========================================================")
        print(f"BENCHMARK ABLATION: {case}")
        print(f"===========================================================")
        # 1. Pure Transformer (GLCM only, 0 cheatsheet)
        t_start = time.time()
        p_acc, m_acc, s_acc = train_and_eval_5fold("pure_transformer", case_key=case, max_epochs=35, device=device)
        print(f"1. Pure 1D Transformer (GLCM Only):  Pooled Acc = {p_acc:.2f}%, Mean = {m_acc:.2f}% ± {s_acc:.2f}% ({time.time() - t_start:.1f}s)")

        # 2. Pure Cheatsheet MLP (Cheatsheet only, 0 GLCM)
        t_start = time.time()
        p_acc2, m_acc2, s_acc2 = train_and_eval_5fold("pure_cheatsheet", case_key=case, max_epochs=35, device=device)
        print(f"2. Pure Cheatsheet MLP (0 GLCM):     Pooled Acc = {p_acc2:.2f}%, Mean = {m_acc2:.2f}% ± {s_acc2:.2f}% ({time.time() - t_start:.1f}s)")
