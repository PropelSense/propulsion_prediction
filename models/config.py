"""
Configuration settings for propulsion power prediction.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_dir: str = "../data/synthetic_data"
    train_file: str = "train.csv"
    dev_in_file: str = "dev_in.csv"
    dev_out_file: str = "dev_out.csv"
    index_column: str = "time_id"
    target_column: str = "power"
    
    # Features to use
    base_features: List[str] = field(default_factory=lambda: [
        "draft_aft_telegram",
        "draft_fore_telegram",
        "stw",
        "diff_speed_overground",
        "awind_vcomp_provider",
        "awind_ucomp_provider",
        "rcurrent_vcomp",
        "rcurrent_ucomp",
        "comb_wind_swell_wave_height",
        "timeSinceDryDock"
    ])
    
    # Validation split ratio (from training data)
    val_split: float = 0.1
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model-related configuration."""
    # MLP architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Ensemble settings
    n_ensemble_members: int = 5
    mc_dropout_samples: int = 50
    
    # Tree model settings
    rf_n_estimators: int = 200
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Output directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# Default configuration instance
default_config = Config()

