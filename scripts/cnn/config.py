"""
scripts/cnn/config.py

Configuration parameters for the independent CNN branch on Bonn EEG data.
Supports both:
- GLCM multi-channel matrix representation [8, 16, 16] (primary CNN branch).
- Recurrence plot image representation [1, 128, 128] (retained utility).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class CNNModelConfig:
    """CNN Architecture configuration for RP images."""
    in_channels: int = 1
    input_size: Tuple[int, int] = (128, 128)
    conv_filters: Tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    conv_stride: int = 3
    pool_size: int = 2
    pool_stride: int = 2
    dense_units: int = 64
    dropout_rate: float = 0.4


@dataclass
class GLCMModelConfig:
    """CNN Architecture configuration for multi-channel GLCM matrices."""
    in_channels: int = 8
    input_size: Tuple[int, int] = (16, 16)
    conv_filters: Tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3
    conv_stride: int = 1  # Stride 1 preserves spatial validity on 16x16 input
    pool_size: int = 2
    pool_stride: int = 2
    dense_units: int = 64
    dropout_rate: float = 0.4


@dataclass
class CNNTrainingConfig:
    """CNN Training and Optimization configuration."""
    batch_size: int = 16
    max_epochs: int = 50
    learning_rate: float = 0.005
    weight_decay: float = 0.0001
    optimizer_name: str = "AdamW"
    
    # Early Stopping
    early_stopping_patience: int = 5
    early_stopping_monitor: str = "val_accuracy"
    early_stopping_min_delta: float = 0.0  # Strict improvement required
    
    # Learning Rate Scheduler
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    min_lr: float = 1e-6
    
    # Seed & Hardware
    random_seed: int = 42
    device: str = "auto"  # 'auto', 'cuda', 'mps', or 'cpu'
    num_workers: int = 0  # Safe default for portable DataLoader


@dataclass
class CNNExperimentConfig:
    """Paths and high-level experiment configuration for RP images."""
    model: CNNModelConfig = field(default_factory=CNNModelConfig)
    training: CNNTrainingConfig = field(default_factory=CNNTrainingConfig)
    
    # Project paths (relative to repo root)
    raw_data_dir: Path = Path("data/bonn_eeg/extracted")
    images_dir: Path = Path("data/processed/rp_images_128")
    manifest_path: Path = Path("data/processed/rp_images_128/image_manifest.json")
    feature_csv_path: Path = Path("audit/batch_preprocessing/bonn_features_96.csv")
    output_dir: Path = Path("experiments/cnn_bonn")
    
    # 10-fold nested CV outer splits
    n_outer_splits: int = 10
    validation_split_ratio: float = 0.10  # 10% of outer training fold for inner validation


@dataclass
class GLCMExperimentConfig:
    """Paths and experiment configuration for multi-channel GLCM matrices."""
    model: GLCMModelConfig = field(default_factory=GLCMModelConfig)
    training: CNNTrainingConfig = field(default_factory=CNNTrainingConfig)
    
    # Project paths (relative to repo root)
    raw_data_dir: Path = Path("data/bonn_eeg/extracted")
    matrices_dir: Path = Path("data/processed/glcm_matrices_cnn")
    manifest_path: Path = Path("data/processed/glcm_matrices_cnn/glcm_manifest.json")
    feature_csv_path: Path = Path("audit/batch_preprocessing/bonn_features_96.csv")
    output_dir: Path = Path("experiments/cnn_glcm_bonn")
    
    # 10-fold nested CV outer splits
    n_outer_splits: int = 10
    validation_split_ratio: float = 0.10  # 10% of outer training fold for inner validation
