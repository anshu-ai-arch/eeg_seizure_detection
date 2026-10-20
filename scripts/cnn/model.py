"""
scripts/cnn/model.py

PyTorch CNN architectures for Bonn EEG classification:
1. BonnGLCMCNN: Primary model taking multi-channel GLCM matrices [batch_size, 8, 16, 16] as input.
   Uses stride 1 convolutions with padding 1 and 2x2 max pooling across 3 blocks.
2. BonnRecurrenceCNN: Retained model taking continuous RP images [batch_size, 1, 128, 128] as input.
"""

import math
from typing import Dict, Tuple, Optional, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.cnn.config import CNNModelConfig, GLCMModelConfig


class Conv2dSame(nn.Module):
    """
    Conv2D layer with explicit TensorFlow-compatible 'same' padding for strided convolutions.
    Calculates asymmetric padding dynamically so output dimension is ceil(in_dim / stride).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        out_h = math.ceil(h / self.stride)
        out_w = math.ceil(w / self.stride)
        pad_h = max(0, (out_h - 1) * self.stride + self.kernel_size - h)
        pad_w = max(0, (out_w - 1) * self.stride + self.kernel_size - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x_padded)


class BonnGLCMCNN(nn.Module):
    """
    Adapted 3-Block Convolutional Neural Network for multi-channel GLCM matrices.

    Input: [batch_size, 8, 16, 16] (8 normalized 16x16 GLCM probability matrices)
    - Block 1: Conv2D(8 -> 16, k=3, s=1, pad=1) -> BN -> ReLU -> MaxPool2D(2, s=2)  => [B, 16, 8, 8]
    - Block 2: Conv2D(16 -> 32, k=3, s=1, pad=1) -> BN -> ReLU -> MaxPool2D(2, s=2) => [B, 32, 4, 4]
    - Block 3: Conv2D(32 -> 64, k=3, s=1, pad=1) -> BN -> ReLU -> MaxPool2D(2, s=2) => [B, 64, 2, 2]
    - Head: GlobalAvgPool -> Dense(64, ReLU) -> Dropout(0.4) -> Dense(num_classes)
    """

    def __init__(self, num_classes: int = 3, config: Optional[GLCMModelConfig] = None):
        super().__init__()
        self.num_classes = num_classes
        self.config = config or GLCMModelConfig()

        in_ch = self.config.in_channels
        f1, f2, f3 = self.config.conv_filters

        # Block 1: 16x16 -> 16x16 -> 8x8
        self.conv1 = nn.Conv2d(in_ch, f1, kernel_size=self.config.kernel_size, stride=self.config.conv_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=self.config.pool_size, stride=self.config.pool_stride)

        # Block 2: 8x8 -> 8x8 -> 4x4
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=self.config.kernel_size, stride=self.config.conv_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=self.config.pool_size, stride=self.config.pool_stride)

        # Block 3: 4x4 -> 4x4 -> 2x2
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=self.config.kernel_size, stride=self.config.conv_stride, padding=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=self.config.pool_size, stride=self.config.pool_stride)

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(f3, self.config.dense_units)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.fc_out = nn.Linear(self.config.dense_units, self.num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Returns raw logits by default (for CrossEntropyLoss compatibility),
        or probabilities via softmax if return_probs is True.
        """
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc_out(x)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

    def get_intermediate_shapes(self, x: torch.Tensor) -> Dict[str, Tuple[int, ...]]:
        """
        Diagnostic helper to trace exact tensor shapes through every layer.
        """
        shapes = {"input": tuple(x.shape)}
        x_c1 = self.conv1(x)
        shapes["block1_conv"] = tuple(x_c1.shape)
        x_p1 = self.pool1(self.relu1(self.bn1(x_c1)))
        shapes["block1_pool"] = tuple(x_p1.shape)

        x_c2 = self.conv2(x_p1)
        shapes["block2_conv"] = tuple(x_c2.shape)
        x_p2 = self.pool2(self.relu2(self.bn2(x_c2)))
        shapes["block2_pool"] = tuple(x_p2.shape)

        x_c3 = self.conv3(x_p2)
        shapes["block3_conv"] = tuple(x_c3.shape)
        x_p3 = self.pool3(self.relu3(self.bn3(x_c3)))
        shapes["block3_pool"] = tuple(x_p3.shape)

        x_gap = self.gap(x_p3).flatten(1)
        shapes["gap"] = tuple(x_gap.shape)
        x_fc = self.fc1(x_gap)
        shapes["dense"] = tuple(x_fc.shape)
        out = self.fc_out(self.dropout(self.relu_fc(x_fc)))
        shapes["logits"] = tuple(out.shape)
        return shapes


class BonnRecurrenceCNN(nn.Module):
    """
    3-Block Convolutional Neural Network for Bonn EEG Recurrence Plots [1, 128, 128].
    Retained for backward compatibility with RP image experiments.
    """

    def __init__(self, num_classes: int = 3, config: Optional[CNNModelConfig] = None):
        super().__init__()
        self.num_classes = num_classes
        self.config = config or CNNModelConfig()

        # Block 1
        self.conv1 = Conv2dSame(
            self.config.in_channels,
            self.config.conv_filters[0],
            kernel_size=self.config.kernel_size,
            stride=self.config.conv_stride,
        )
        self.bn1 = nn.BatchNorm2d(self.config.conv_filters[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(
            kernel_size=self.config.pool_size,
            stride=self.config.pool_stride,
        )

        # Block 2
        self.conv2 = Conv2dSame(
            self.config.conv_filters[0],
            self.config.conv_filters[1],
            kernel_size=self.config.kernel_size,
            stride=self.config.conv_stride,
        )
        self.bn2 = nn.BatchNorm2d(self.config.conv_filters[1])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(
            kernel_size=self.config.pool_size,
            stride=self.config.pool_stride,
        )

        # Block 3 (MaxPooling intentionally omitted to avoid 1x1 -> 0x0 collapse)
        self.conv3 = Conv2dSame(
            self.config.conv_filters[1],
            self.config.conv_filters[2],
            kernel_size=self.config.kernel_size,
            stride=self.config.conv_stride,
        )
        self.bn3 = nn.BatchNorm2d(self.config.conv_filters[2])
        self.relu3 = nn.ReLU(inplace=True)

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.config.conv_filters[2], self.config.dense_units)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.fc_out = nn.Linear(self.config.dense_units, self.num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        # Block 1: 128x128 -> 43x43 -> 21x21
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2: 21x21 -> 7x7 -> 3x3
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3: 3x3 -> 1x1 (no pooling)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Head: 1x1 -> 64 -> 64 -> num_classes
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc_out(x)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

    def get_intermediate_shapes(self, x: torch.Tensor) -> Dict[str, Tuple[int, ...]]:
        shapes = {"input": tuple(x.shape)}
        x_c1 = self.conv1(x)
        shapes["block1_conv"] = tuple(x_c1.shape)
        x_p1 = self.pool1(self.relu1(self.bn1(x_c1)))
        shapes["block1_pool"] = tuple(x_p1.shape)

        x_c2 = self.conv2(x_p1)
        shapes["block2_conv"] = tuple(x_c2.shape)
        x_p2 = self.pool2(self.relu2(self.bn2(x_c2)))
        shapes["block2_pool"] = tuple(x_p2.shape)

        x_c3 = self.conv3(x_p2)
        shapes["block3_conv"] = tuple(x_c3.shape)
        x_b3 = self.relu3(self.bn3(x_c3))
        shapes["block3_out"] = tuple(x_b3.shape)

        x_gap = self.gap(x_b3).flatten(1)
        shapes["gap"] = tuple(x_gap.shape)
        x_fc = self.fc1(x_gap)
        shapes["dense"] = tuple(x_fc.shape)
        out = self.fc_out(self.dropout(self.relu_fc(x_fc)))
        shapes["logits"] = tuple(out.shape)
        return shapes


class BonnModernRPCNN(nn.Module):
    """
    Modern 4-Block Convolutional Neural Network for continuous Bonn EEG Recurrence Plots [1, 128, 128].

    Input: [batch_size, 1, 128, 128]
    Features:
    - Stride 1 convolutions with padding 1 across all 4 blocks (preserves fine-grained recurrence textures without skipping pixels).
    - Batch Normalization after every conv layer for stable convergence.
    - 2x2 Max Pooling: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8.
    - Global Average Pooling (GAP) eliminates spatial dimension, reducing parameter count.
    - Dense(64, 32) -> ReLU -> Dropout(0.4) -> Dense(32, num_classes).
    - Total parameters: ~62,000 (lightweight, designed to prevent overfitting on small N).
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 16x16 -> 8x8
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(32, self.num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))

        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        logits = self.fc_out(x)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits

    def get_intermediate_shapes(self, x: torch.Tensor) -> Dict[str, Tuple[int, ...]]:
        shapes = {"input": tuple(x.shape)}
        x_c1 = self.conv1(x)
        shapes["block1_conv"] = tuple(x_c1.shape)
        x_p1 = self.pool1(self.relu1(self.bn1(x_c1)))
        shapes["block1_pool"] = tuple(x_p1.shape)

        x_c2 = self.conv2(x_p1)
        shapes["block2_conv"] = tuple(x_c2.shape)
        x_p2 = self.pool2(self.relu2(self.bn2(x_c2)))
        shapes["block2_pool"] = tuple(x_p2.shape)

        x_c3 = self.conv3(x_p2)
        shapes["block3_conv"] = tuple(x_c3.shape)
        x_p3 = self.pool3(self.relu3(self.bn3(x_c3)))
        shapes["block3_pool"] = tuple(x_p3.shape)

        x_c4 = self.conv4(x_p3)
        shapes["block4_conv"] = tuple(x_c4.shape)
        x_p4 = self.pool4(self.relu4(self.bn4(x_c4)))
        shapes["block4_pool"] = tuple(x_p4.shape)

        x_gap = self.gap(x_p4).flatten(1)
        shapes["gap"] = tuple(x_gap.shape)
        x_fc = self.fc1(x_gap)
        shapes["dense"] = tuple(x_fc.shape)
        out = self.fc_out(self.dropout(self.relu_fc(x_fc)))
        shapes["logits"] = tuple(out.shape)
        return shapes


class BonnHybridRPCNN(nn.Module):
    """
    Hybrid Convolutional Neural Network fusing:
    1. Deep Visual Branch: 4-Block ConvNet on continuous Recurrence Plot images [B, 1, 128, 128] -> 64 features.
    2. Dynamical & Spectral Scalar Branch: Linear projection on physiological scalars (tau, m, alpha_ratio, etc.) [B, num_scalars] -> 16 features.
    3. Multimodal Fusion Head: Concatenates visual + scalar features -> Dense(80, 32) -> Dropout(0.4) -> Logits.
    """

    def __init__(self, num_classes: int = 3, num_scalars: int = 7, dropout_rate: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.num_scalars = num_scalars

        # Visual Branch (same 4-block ConvNet as BonnModernRPCNN)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Scalar Branch
        self.scalar_fc = nn.Linear(num_scalars, 16)
        self.scalar_bn = nn.BatchNorm1d(16)
        self.scalar_relu = nn.ReLU(inplace=True)

        # Multimodal Fusion Head (64 visual + 16 scalar = 80)
        self.fc_fused = nn.Linear(64 + 16, 32)
        self.relu_fused = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(32, self.num_classes)

    def forward(self, img: torch.Tensor, scalars: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        # Visual pass
        x_img = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        x_img = self.pool2(self.relu2(self.bn2(self.conv2(x_img))))
        x_img = self.pool3(self.relu3(self.bn3(self.conv3(x_img))))
        x_img = self.pool4(self.relu4(self.bn4(self.conv4(x_img))))
        feat_vis = self.gap(x_img).flatten(1)  # [B, 64]

        # Scalar pass
        feat_scalar = self.scalar_relu(self.scalar_bn(self.scalar_fc(scalars)))  # [B, 16]

        # Fusion pass
        fused = torch.cat([feat_vis, feat_scalar], dim=1)  # [B, 80]
        fused = self.relu_fused(self.fc_fused(fused))
        fused = self.dropout(fused)
        logits = self.fc_out(fused)

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits


class BonnGatedHybridRPCNN(nn.Module):
    """
    Gated Hybrid Recurrence Plot CNN featuring:
    1. Deep Visual Branch: 4-Block ConvNet on continuous Recurrence Plot images [B, 1, 128, 128]
       -> GAP -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> [B, 32] visual representation.
    2. Dynamical & Spectral Scalar Branch: Linear(num_scalars, 32) -> BatchNorm1d(32) -> ReLU -> [B, 32].
    3. Balanced Gating Mechanism:
       Computes gating coefficients g = Sigmoid(Linear(64, 64)([h_vis, h_scalar])).
       Applies elementwise gating: g_vis * h_vis (32 dims) and g_scal * h_scalar (32 dims).
       Fused representation: Linear(64, 32) -> ReLU -> Dropout(dropout_rate) -> Linear(32, num_classes).
    4. Direct Residual Amplitude/Energy Logit Connection:
       Linear(len(residual_indices), num_classes) directly projects physical signal amplitude/energy
       metrics (e.g. log_raw_std, log_energy, log_line_length) to classification logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_scalars: int = 9,
        residual_indices: Optional[List[int]] = None,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_scalars = num_scalars
        self.residual_indices = residual_indices if residual_indices is not None else [2, 3, 4]

        # Visual Branch (4-Block ConvNet)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Balanced 32-dim visual projection
        self.vis_proj = nn.Linear(64, 32)
        self.vis_bn = nn.BatchNorm1d(32)
        self.vis_relu = nn.ReLU(inplace=True)

        # Balanced 32-dim scalar projection
        self.scalar_proj = nn.Linear(num_scalars, 32)
        self.scalar_bn = nn.BatchNorm1d(32)
        self.scalar_relu = nn.ReLU(inplace=True)

        # Gating network: generates 64-dim gates for balanced visual & scalar feature modulation
        self.gate_fc = nn.Linear(64, 64)

        # Multimodal Fused Classifier
        self.fc_fused = nn.Linear(64, 32)
        self.relu_fused = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(32, self.num_classes)

        # Direct Residual Logit Connection from amplitude & energy scalars
        self.fc_residual = nn.Linear(len(self.residual_indices), self.num_classes)

    def forward(self, img: torch.Tensor, scalars: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        # Visual pass
        x = self.pool1(self.relu1(self.bn1(self.conv1(img))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        feat_vis_raw = self.gap(x).flatten(1)  # [B, 64]
        h_vis = self.vis_relu(self.vis_bn(self.vis_proj(feat_vis_raw)))  # [B, 32]

        # Scalar pass
        h_scal = self.scalar_relu(self.scalar_bn(self.scalar_proj(scalars)))  # [B, 32]

        # Gated fusion
        combined = torch.cat([h_vis, h_scal], dim=1)  # [B, 64]
        gates = torch.sigmoid(self.gate_fc(combined))  # [B, 64]
        g_vis = gates[:, :32]
        g_scal = gates[:, 32:]

        fused = torch.cat([g_vis * h_vis, g_scal * h_scal], dim=1)  # [B, 64]
        fused = self.relu_fused(self.fc_fused(fused))  # [B, 32]
        fused = self.dropout(fused)
        main_logits = self.fc_out(fused)  # [B, num_classes]

        # Residual physical amplitude connection directly to logits
        if len(self.residual_indices) > 0:
            amp_scalars = scalars[:, self.residual_indices]
            res_logits = self.fc_residual(amp_scalars)
            logits = main_logits + res_logits
        else:
            logits = main_logits

        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits



