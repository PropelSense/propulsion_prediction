"""
Propulsion Power Prediction Models

This package provides tools for predicting propeller shaft power
based on vessel operational and environmental conditions.

Modules:
- config: Configuration settings
- data_module: Data loading and preprocessing
- baseline_models: Traditional ML baselines (Linear, RF, XGBoost)
- neural_models: Neural network models (MLP, Gaussian MLP)
- uncertainty: Uncertainty quantification (Ensemble, MC Dropout)
- train: Training script
- evaluate: Evaluation script
"""

from .config import Config, DataConfig, ModelConfig, default_config
from .data_module import PropulsionDataModule, FeatureEngineer, DataScaler, TargetScaler
from .baseline_models import (
    MeanBaseline,
    LinearModel,
    RandomForestModel,
    create_baseline_model,
    save_model,
    load_model
)
from .neural_models import MLP, GaussianMLP, NeuralNetworkTrainer, create_mlp
from .uncertainty import DeepEnsemble, MCDropout, compute_uncertainty_metrics

__all__ = [
    # Config
    'Config', 'DataConfig', 'ModelConfig', 'default_config',
    # Data
    'PropulsionDataModule', 'FeatureEngineer', 'DataScaler', 'TargetScaler',
    # Baseline models
    'MeanBaseline', 'LinearModel', 'RandomForestModel',
    'create_baseline_model', 'save_model', 'load_model',
    # Neural models
    'MLP', 'GaussianMLP', 'NeuralNetworkTrainer', 'create_mlp',
    # Uncertainty
    'DeepEnsemble', 'MCDropout', 'compute_uncertainty_metrics'
]

