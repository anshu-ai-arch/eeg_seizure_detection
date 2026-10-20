"""
scripts/transformer/model.py

1D Gated Transformer Architecture for Robust GLCM EEG Classification.

Features:
1. Markov Amplitude State Tokenizer: Maps multi-mode GLCM [B, 4, 64, 64] -> sequence of 64 amplitude states [B, 64, 256] -> projected to [B, 64, d_model].
2. Micro-Transformer Encoder: 2-layer Pre-LN Multi-Head Self-Attention (d_model=32, nhead=4, d_ff=64) learning cross-amplitude state dependencies.
3. Cheatsheet MLP: Lightweight projection of 69 domain features (energy, entropy, ZCR, Haralick metrics) -> [B, d_model].
4. Balanced Sigmoid Gated Late Fusion: Dynamic gating between texture attention embedding and scalar cheatsheet vector.
5. Direct Physical Residual Connection: Amplitude and energy metrics shortcut directly to logits to eliminate false alarms.
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bonn1DGatedGLCMTransformer(nn.Module):
    """
    1D Gated Transformer for Multimodal EEG Classification.
    Total parameter budget: ~33,000 parameters (anti-overfitting design for small N).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_cheatsheet: int = 69,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        residual_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        # Default residual indices: global_log_raw_std (1), global_log_energy (3), global_log_line_length (5)
        self.residual_indices = residual_indices if residual_indices is not None else [1, 3, 5]

        # Path A: Markov State Tokenizer (4 modes x 64 transition values = 256 inputs per state)
        self.token_proj = nn.Linear(256, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.pos_drop = nn.Dropout(p=0.1)

        # 2-Layer Pre-LN Transformer Encoder
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

        # Path B: Cheatsheet MLP
        self.cheat_fc = nn.Linear(num_cheatsheet, d_model)
        self.cheat_bn = nn.BatchNorm1d(d_model)
        self.cheat_act = nn.GELU()
        self.cheat_drop = nn.Dropout(p=0.3)

        # Gated Late Fusion Module
        self.gate_fc = nn.Linear(d_model * 2, d_model)

        # Classifier Head
        self.fc_fused = nn.Linear(d_model, d_model)
        self.act_fused = nn.GELU()
        self.drop_fused = nn.Dropout(p=0.4)
        self.fc_out = nn.Linear(d_model, num_classes)

        # Direct Residual Logit Connection from physical amplitude/energy metrics
        self.fc_residual = nn.Linear(len(self.residual_indices), num_classes)

    def forward(
        self,
        glcm_tensor: torch.Tensor,
        cheatsheet_vec: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            glcm_tensor: [B, 4, 64, 64] float32 tensor of multi-mode GLCMs.
            cheatsheet_vec: [B, num_cheatsheet] float32 vector of domain features.
            return_probs: If True, returns softmax probabilities.
        """
        B = glcm_tensor.shape[0]

        # 1. Tokenize: [B, 4, 64, 64] -> [B, 64 states, 4*64=256]
        # Transpose mode and row indices so each of the 64 states has 4*64 transition features
        x_tok = glcm_tensor.permute(0, 2, 1, 3).reshape(B, 64, 256)
        h_tokens = self.pos_drop(self.token_proj(x_tok) + self.pos_emb)

        # 2. Transformer Encoder pass
        out_tokens = self.transformer(h_tokens)  # [B, 64, d_model]
        # Global mean pooling over the 64 amplitude states
        h_tex = self.norm_tex(out_tokens.mean(dim=1))  # [B, d_model]

        # 3. Cheatsheet MLP pass
        h_cheat = self.cheat_drop(self.cheat_act(self.cheat_bn(self.cheat_fc(cheatsheet_vec))))  # [B, d_model]

        # 4. Gated Late Fusion
        concat = torch.cat([h_tex, h_cheat], dim=1)  # [B, 2*d_model]
        gate = torch.sigmoid(self.gate_fc(concat))    # [B, d_model]
        fused = gate * h_tex + (1.0 - gate) * h_cheat  # [B, d_model]

        # 5. Main classification path
        fused_hidden = self.drop_fused(self.act_fused(self.fc_fused(fused)))
        main_logits = self.fc_out(fused_hidden)  # [B, num_classes]

        # 6. Direct physical amplitude/energy residual shortcut
        if len(self.residual_indices) > 0:
            res_input = cheatsheet_vec[:, self.residual_indices]
            res_logits = self.fc_residual(res_input)
            logits = main_logits + res_logits
        else:
            logits = main_logits

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits
