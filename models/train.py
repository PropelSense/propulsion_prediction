"""
Training script for propulsion power prediction models.

Usage:
    python train.py --model mlp
    python train.py --model ensemble --n_members 5
    python train.py --model xgboost
    python train.py --model all
"""
import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, default_config
from data_module import PropulsionDataModule
from baseline_models import (
    create_baseline_model, 
    MeanBaseline, 
    LinearModel, 
    RandomForestModel,
    save_model
)
from neural_models import create_mlp, NeuralNetworkTrainer
from uncertainty import DeepEnsemble


def train_baseline(model_type: str, data_module: PropulsionDataModule, config: Config) -> dict:
    """
    Train a baseline model.
    
    Args:
        model_type: Type of baseline ('mean', 'linear', 'ridge', 'rf', 'xgboost')
        data_module: Data module with loaded data
        config: Configuration
        
    Returns:
        Dictionary with model and training info
    """
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*50}")
    
    X_train, y_train = data_module.get_train_data()
    X_val, y_val = data_module.get_val_data()
    
    model = create_baseline_model(model_type, config.model)
    
    # XGBoost can use validation data for early stopping
    if model_type == 'xgboost':
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)
    
    # Validation performance
    val_pred = model.predict(X_val)
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    
    print(f"Validation MAE (scaled): {val_mae:.6f}")
    print(f"Validation RMSE (scaled): {val_rmse:.6f}")
    
    # Convert to original scale for interpretability
    val_pred_orig = data_module.inverse_transform_predictions(val_pred)
    val_true_orig = data_module.get_raw_targets('val')
    val_mae_kw = np.mean(np.abs(val_pred_orig - val_true_orig))
    
    print(f"Validation MAE (kW): {val_mae_kw:.2f}")
    
    # Save model
    model_path = os.path.join(config.checkpoint_dir, f"{model_type}_model.joblib")
    save_model(model, model_path)
    
    return {
        'model': model,
        'model_type': model_type,
        'val_mae_scaled': val_mae,
        'val_rmse_scaled': val_rmse,
        'val_mae_kw': val_mae_kw,
        'model_path': model_path
    }


def train_mlp(data_module: PropulsionDataModule, config: Config, gaussian: bool = False) -> dict:
    """
    Train a single MLP model.
    
    Args:
        data_module: Data module with loaded data
        config: Configuration
        gaussian: Whether to use Gaussian output for aleatoric uncertainty
        
    Returns:
        Dictionary with model and training info
    """
    model_name = "Gaussian MLP" if gaussian else "MLP"
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    X_train, y_train = data_module.get_train_data()
    X_val, y_val = data_module.get_val_data()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.model.random_seed)
    np.random.seed(config.model.random_seed)
    
    model = create_mlp(
        input_dim=data_module.n_features,
        config=config.model,
        gaussian_output=gaussian
    )
    
    trainer = NeuralNetworkTrainer(
        model=model,
        learning_rate=config.model.learning_rate,
        batch_size=config.model.batch_size,
        max_epochs=config.model.max_epochs,
        early_stopping_patience=config.model.early_stopping_patience
    )
    
    history = trainer.train(X_train, y_train, X_val, y_val, verbose=True)
    
    # Validation performance
    val_pred = trainer.predict(X_val)
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    
    print(f"\nFinal Validation MAE (scaled): {val_mae:.6f}")
    print(f"Final Validation RMSE (scaled): {val_rmse:.6f}")
    
    # Convert to original scale
    val_pred_orig = data_module.inverse_transform_predictions(val_pred)
    val_true_orig = data_module.get_raw_targets('val')
    val_mae_kw = np.mean(np.abs(val_pred_orig - val_true_orig))
    
    print(f"Final Validation MAE (kW): {val_mae_kw:.2f}")
    
    # Save model
    model_suffix = "gaussian_mlp" if gaussian else "mlp"
    model_path = os.path.join(config.checkpoint_dir, f"{model_suffix}_model.pt")
    trainer.save(model_path)
    
    return {
        'model': model,
        'trainer': trainer,
        'model_type': model_name,
        'history': history,
        'val_mae_scaled': val_mae,
        'val_rmse_scaled': val_rmse,
        'val_mae_kw': val_mae_kw,
        'model_path': model_path
    }


def train_ensemble(data_module: PropulsionDataModule, config: Config) -> dict:
    """
    Train a deep ensemble.
    
    Args:
        data_module: Data module with loaded data
        config: Configuration
        
    Returns:
        Dictionary with ensemble and training info
    """
    print(f"\n{'='*50}")
    print(f"Training Deep Ensemble ({config.model.n_ensemble_members} members)")
    print(f"{'='*50}")
    
    X_train, y_train = data_module.get_train_data()
    X_val, y_val = data_module.get_val_data()
    
    ensemble = DeepEnsemble(
        input_dim=data_module.n_features,
        n_members=config.model.n_ensemble_members,
        config=config.model,
        gaussian_output=False  # Can set to True for combined uncertainty
    )
    
    histories = ensemble.train(X_train, y_train, X_val, y_val, verbose=True)
    
    # Validation performance with uncertainty
    val_pred, val_uncertainty = ensemble.get_total_uncertainty(X_val)
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    
    print(f"\n{'='*50}")
    print("Ensemble Results")
    print(f"{'='*50}")
    print(f"Validation MAE (scaled): {val_mae:.6f}")
    print(f"Validation RMSE (scaled): {val_rmse:.6f}")
    print(f"Mean uncertainty (std): {np.mean(np.sqrt(val_uncertainty)):.6f}")
    
    # Convert to original scale
    val_pred_orig = data_module.inverse_transform_predictions(val_pred)
    val_true_orig = data_module.get_raw_targets('val')
    val_mae_kw = np.mean(np.abs(val_pred_orig - val_true_orig))
    
    print(f"Validation MAE (kW): {val_mae_kw:.2f}")
    
    # Save ensemble
    ensemble_dir = os.path.join(config.checkpoint_dir, "ensemble")
    ensemble.save(ensemble_dir)
    
    return {
        'ensemble': ensemble,
        'model_type': 'Deep Ensemble',
        'histories': histories,
        'val_mae_scaled': val_mae,
        'val_rmse_scaled': val_rmse,
        'val_mae_kw': val_mae_kw,
        'mean_uncertainty': float(np.mean(np.sqrt(val_uncertainty))),
        'model_path': ensemble_dir
    }


def train_all(data_module: PropulsionDataModule, config: Config) -> dict:
    """
    Train all model types and compare results.
    
    Returns:
        Dictionary with all results
    """
    results = {}
    
    # Baselines
    for model_type in ['mean', 'linear', 'rf']:
        results[model_type] = train_baseline(model_type, data_module, config)
    
    # Try XGBoost if available
    try:
        results['xgboost'] = train_baseline('xgboost', data_module, config)
    except ImportError:
        print("\nSkipping XGBoost (not installed)")
    
    # Neural networks
    results['mlp'] = train_mlp(data_module, config, gaussian=False)
    results['gaussian_mlp'] = train_mlp(data_module, config, gaussian=True)
    
    # Ensemble
    results['ensemble'] = train_ensemble(data_module, config)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Val MAE (scaled)':<18} {'Val MAE (kW)':<15}")
    print("-" * 70)
    
    for name, result in results.items():
        mae_scaled = result.get('val_mae_scaled', float('nan'))
        mae_kw = result.get('val_mae_kw', float('nan'))
        print(f"{name:<20} {mae_scaled:<18.6f} {mae_kw:<15.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train propulsion prediction models")
    parser.add_argument(
        '--model', 
        type=str, 
        default='mlp',
        choices=['mean', 'linear', 'rf', 'xgboost', 'mlp', 'gaussian_mlp', 'ensemble', 'all'],
        help='Model type to train'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--n_members',
        type=int,
        default=5,
        help='Number of ensemble members'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Training batch size'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    config.model.n_ensemble_members = args.n_members
    config.model.max_epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.random_seed = args.seed
    config.data.random_seed = args.seed
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("=" * 70)
    print("PROPULSION POWER PREDICTION - TRAINING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"Random seed: {config.model.random_seed}")
    print("=" * 70)
    
    # Load and prepare data
    print("\nLoading data...")
    data_module = PropulsionDataModule(config.data)
    data_module.setup()
    
    # Save scalers for later inference
    data_module.save_scalers(config.checkpoint_dir)
    
    # Train selected model(s)
    if args.model == 'all':
        results = train_all(data_module, config)
    elif args.model in ['mean', 'linear', 'rf', 'xgboost']:
        results = {args.model: train_baseline(args.model, data_module, config)}
    elif args.model == 'mlp':
        results = {'mlp': train_mlp(data_module, config, gaussian=False)}
    elif args.model == 'gaussian_mlp':
        results = {'gaussian_mlp': train_mlp(data_module, config, gaussian=True)}
    elif args.model == 'ensemble':
        results = {'ensemble': train_ensemble(data_module, config)}
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': args.model,
        'config': {
            'data_dir': config.data.data_dir,
            'n_features': data_module.n_features,
            'feature_columns': data_module.feature_columns,
            'random_seed': config.model.random_seed
        },
        'results': {}
    }
    
    for name, result in results.items():
        summary['results'][name] = {
            'val_mae_scaled': float(result.get('val_mae_scaled', float('nan'))),
            'val_mae_kw': float(result.get('val_mae_kw', float('nan'))),
            'model_path': result.get('model_path', '')
        }
    
    summary_path = os.path.join(config.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

