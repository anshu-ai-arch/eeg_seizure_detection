"""
scripts/transformer/cbm_model.py

Concept-Bottlenecked Gated Transformer Architecture for Robust GLCM EEG Classification.

Scientific Innovation:
1. Markov Amplitude State Tokenizer: Maps multi-mode GLCM [B, 4, 64, 64] -> sequence of 64 amplitude states [B, 64, 256] -> projected to [B, 64, d_model].
2. Micro-Transformer Encoder: 2-layer Pre-LN Multi-Head Self-Attention (d_model=32, nhead=4, d_ff=64) learning cross-amplitude state dependencies.
3. Cheatsheet MLP: Lightweight projection of domain features -> [B, d_model].
4. Balanced Sigmoid Gated Late Fusion: Dynamic gating between texture attention embedding and scalar cheatsheet vector -> [B, d_model].
5. Intrinsic Concept Bottleneck Layer: Maps fused representation to 3 clinically grounded electrophysiological concepts:
   - C1: Paroxysmal Spike Index (PSI)
   - C2: Hypersynchronous Rhythmic Power Surge (HPS)
   - C3: Phase-Space Dynamic Contrast / Texture Disruption (PRC)
6. Linear Interpretable Decision Head: Directly computes class logits as a linear combination of clinical concepts.
7. Direct Physical Residual Connection: Safeguards against "Concept Bottleneck Penalty" by shortcutting physical amplitude metrics to logits.
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


CONCEPT_NAMES = [
    "C1_Paroxysmal_Spike_Index",
    "C2_Hypersynchronous_Power_Surge",
    "C3_Phase_Space_Texture_Disruption",
]


class BonnConceptGatedTransformer(nn.Module):
    """
    Concept-Bottlenecked 1D Gated Transformer for Multimodal EEG Classification.
    Total parameter budget: ~33,000 parameters (anti-overfitting design for small N).
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_cheatsheet: int = 69,
        num_concepts: int = 3,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        residual_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_concepts = num_concepts
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

        # Concept Bottleneck Module
        # Step 1: Bottleneck projection to clinical concepts
        self.concept_proj = nn.Linear(d_model, num_concepts)
        
        # Step 2: Linear interpretable decision head from concepts to class logits
        self.concept_to_class = nn.Linear(num_concepts, num_classes)

        # Direct Residual Logit Connection from physical amplitude/energy metrics
        self.fc_residual = nn.Linear(len(self.residual_indices), num_classes)

    def forward(
        self,
        glcm_tensor: torch.Tensor,
        cheatsheet_vec: torch.Tensor,
        return_concepts: bool = False,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        Args:
            glcm_tensor: [B, 4, 64, 64] float32 tensor of multi-mode GLCMs.
            cheatsheet_vec: [B, num_cheatsheet] float32 vector of domain features.
            return_concepts: If True, returns (logits, concepts).
            return_probs: If True, returns softmax probabilities instead of logits.

        Returns:
            logits (or probs), [optional: concepts [B, num_concepts] in [0, 1]]
        """
        B = glcm_tensor.shape[0]

        # 1. Tokenize: [B, 4, 64, 64] -> [B, 64 states, 4*64=256]
        x_tok = glcm_tensor.permute(0, 2, 1, 3).reshape(B, 64, 256)
        h_tokens = self.pos_drop(self.token_proj(x_tok) + self.pos_emb)

        # 2. Transformer Encoder pass
        out_tokens = self.transformer(h_tokens)  # [B, 64, d_model]
        h_tex = self.norm_tex(out_tokens.mean(dim=1))  # [B, d_model]

        # 3. Cheatsheet MLP pass
        h_cheat = self.cheat_drop(self.cheat_act(self.cheat_bn(self.cheat_fc(cheatsheet_vec))))  # [B, d_model]

        # 4. Gated Late Fusion
        concat = torch.cat([h_tex, h_cheat], dim=1)  # [B, 2*d_model]
        gate = torch.sigmoid(self.gate_fc(concat))    # [B, d_model]
        fused = gate * h_tex + (1.0 - gate) * h_cheat  # [B, d_model]

        # 5. Concept Bottleneck Projection
        # Continuous bounded concept activations in [0, 1]
        concepts = torch.sigmoid(self.concept_proj(fused))  # [B, num_concepts]

        # 6. Linear Concept-to-Class Decision
        concept_logits = self.concept_to_class(concepts)  # [B, num_classes]

        # 7. Direct physical amplitude/energy residual shortcut
        if len(self.residual_indices) > 0:
            res_input = cheatsheet_vec[:, self.residual_indices]
            res_logits = self.fc_residual(res_input)
            logits = concept_logits + res_logits
        else:
            logits = concept_logits

        out = F.softmax(logits, dim=-1) if return_probs else logits

        if return_concepts:
            return out, concepts
        return out

    def get_concept_weights(self) -> Dict[str, Any]:
        """
        Extract the linear decision weights from concepts to classes.
        Enables doctors to inspect how each concept contributes to the diagnosis.
        """
        W = self.concept_to_class.weight.detach().cpu().numpy()  # [num_classes, num_concepts]
        b = self.concept_to_class.bias.detach().cpu().numpy()    # [num_classes]
        return {
            "weights": W,
            "bias": b,
            "concept_names": CONCEPT_NAMES[:self.num_concepts],
        }
