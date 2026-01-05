"""
Evaluation script for propulsion power prediction models.

Evaluates trained models on validation, dev_in, and dev_out sets.
Computes prediction metrics and uncertainty quality metrics.

Usage:
    python evaluate.py --model mlp
    python evaluate.py --model ensemble --with_uncertainty
    python evaluate.py --model all
"""
import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, default_config
from data_module import PropulsionDataModule
from baseline_models import load_model, RandomForestModel
from neural_models import create_mlp, NeuralNetworkTrainer, MLP, GaussianMLP
from uncertainty import DeepEnsemble, MCDropout, compute_uncertainty_metrics, calibration_analysis

# Import assessment utilities from parent
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.assessment import get_performance_metric, get_model_errors


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_module: PropulsionDataModule,
    split_name: str
) -> dict:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        y_true: True values (scaled)
        y_pred: Predicted values (scaled)
        data_module: Data module for inverse transform
        split_name: Name of the data split
        
    Returns:
        Dictionary of metrics
    """
    # Scaled metrics
    mae_scaled = np.mean(np.abs(y_pred - y_true))
    rmse_scaled = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Original scale metrics (kW)
    y_true_orig = data_module.inverse_transform_predictions(y_true)
    y_pred_orig = data_module.inverse_transform_predictions(y_pred)
    
    mae_kw = np.mean(np.abs(y_pred_orig - y_true_orig))
    rmse_kw = np.sqrt(np.mean((y_pred_orig - y_true_orig) ** 2))
    
    # MAPE (using original scale, avoiding division by zero)
    mask = np.abs(y_true_orig) > 100  # Only compute MAPE for power > 100 kW
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
    else:
        mape = float('nan')
    
    return {
        'split': split_name,
        'mae_scaled': float(mae_scaled),
        'rmse_scaled': float(rmse_scaled),
        'mae_kw': float(mae_kw),
        'rmse_kw': float(rmse_kw),
        'mape_percent': float(mape),
        'n_samples': len(y_true)
    }


def evaluate_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    split_name: str
) -> dict:
    """
    Evaluate uncertainty quality.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates (std)
        split_name: Name of data split
        
    Returns:
        Dictionary of uncertainty metrics
    """
    errors = np.abs(y_pred - y_true)
    
    metrics = compute_uncertainty_metrics(errors, uncertainty)
    metrics['split'] = split_name
    metrics['mean_uncertainty'] = float(np.mean(uncertainty))
    metrics['std_uncertainty'] = float(np.std(uncertainty))
    
    # Calibration analysis
    calib = calibration_analysis(y_true, y_pred, uncertainty)
    metrics['calibration'] = {
        'bin_mean_uncertainty': calib['bin_mean_uncertainty'].tolist(),
        'bin_mean_error': calib['bin_mean_error'].tolist()
    }
    
    return metrics


def load_trained_model(model_type: str, config: Config, n_features: int):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: Type of model to load
        config: Configuration
        n_features: Number of input features
        
    Returns:
        Loaded model/trainer
    """
    checkpoint_dir = config.checkpoint_dir
    
    if model_type in ['mean', 'linear', 'ridge', 'rf', 'xgboost']:
        model_path = os.path.join(checkpoint_dir, f"{model_type}_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        return load_model(model_path)
    
    elif model_type == 'mlp':
        import torch
        model_path = os.path.join(checkpoint_dir, "mlp_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = create_mlp(n_features, config.model, gaussian_output=False)
        trainer = NeuralNetworkTrainer(model)
        trainer.load(model_path)
        return trainer
    
    elif model_type == 'gaussian_mlp':
        import torch
        model_path = os.path.join(checkpoint_dir, "gaussian_mlp_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = create_mlp(n_features, config.model, gaussian_output=True)
        trainer = NeuralNetworkTrainer(model)
        trainer.load(model_path)
        return trainer
    
    elif model_type == 'ensemble':
        ensemble_dir = os.path.join(checkpoint_dir, "ensemble")
        if not os.path.exists(ensemble_dir):
            raise FileNotFoundError(f"Ensemble not found: {ensemble_dir}")
        
        # Load metadata
        metadata = np.load(os.path.join(ensemble_dir, "metadata.npy"), allow_pickle=True).item()
        
        # Recreate ensemble and load members
        ensemble = DeepEnsemble(
            input_dim=n_features,
            n_members=metadata['n_members'],
            config=config.model,
            gaussian_output=metadata['gaussian_output']
        )
        
        import torch
        for i in range(metadata['n_members']):
            model = create_mlp(n_features, config.model, gaussian_output=metadata['gaussian_output'])
            trainer = NeuralNetworkTrainer(model)
            trainer.load(os.path.join(ensemble_dir, f"member_{i}.pt"))
            ensemble.models.append(model)
            ensemble.trainers.append(trainer)
        
        ensemble.is_trained = True
        return ensemble
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(
    model_type: str,
    data_module: PropulsionDataModule,
    config: Config,
    with_uncertainty: bool = False
) -> dict:
    """
    Evaluate a single model on all data splits.
    
    Args:
        model_type: Type of model to evaluate
        data_module: Data module with loaded data
        config: Configuration
        with_uncertainty: Whether to evaluate uncertainty
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = load_trained_model(model_type, config, data_module.n_features)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None
    
    results = {
        'model_type': model_type,
        'predictions': {},
        'uncertainty': {} if with_uncertainty else None
    }
    
    # Evaluate on each split
    splits = {
        'val': data_module.get_val_data(),
        'dev_in': data_module.get_dev_in_data(),
        'dev_out': data_module.get_dev_out_data()
    }
    
    for split_name, (X, y_true) in splits.items():
        print(f"\n--- {split_name} ---")
        
        # Get predictions
        if model_type == 'ensemble':
            y_pred, uncertainty = model.get_total_uncertainty(X)
            uncertainty = np.sqrt(uncertainty)  # Convert variance to std
        elif hasattr(model, 'predict_with_uncertainty') and with_uncertainty:
            if isinstance(model, RandomForestModel):
                y_pred, uncertainty = model.predict_with_uncertainty(X)
            elif isinstance(model, NeuralNetworkTrainer):
                if isinstance(model.model, GaussianMLP):
                    y_pred, var = model.predict_with_uncertainty(X)
                    uncertainty = np.sqrt(var)
                else:
                    # Use MC Dropout for regular MLP
                    mc = MCDropout(model.model, n_samples=config.model.mc_dropout_samples)
                    y_pred, uncertainty = mc.predict_with_uncertainty(X)
            else:
                y_pred = model.predict(X)
                uncertainty = None
        else:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X)
            uncertainty = None
        
        # Compute prediction metrics
        pred_metrics = evaluate_predictions(y_true, y_pred, data_module, split_name)
        results['predictions'][split_name] = pred_metrics
        
        print(f"  MAE (scaled): {pred_metrics['mae_scaled']:.6f}")
        print(f"  MAE (kW): {pred_metrics['mae_kw']:.2f}")
        print(f"  RMSE (kW): {pred_metrics['rmse_kw']:.2f}")
        print(f"  MAPE: {pred_metrics['mape_percent']:.2f}%")
        
        # Compute uncertainty metrics if available
        if uncertainty is not None and with_uncertainty:
            unc_metrics = evaluate_uncertainty(y_true, y_pred, uncertainty, split_name)
            results['uncertainty'][split_name] = unc_metrics
            
            print(f"  Mean uncertainty (std): {unc_metrics['mean_uncertainty']:.6f}")
            print(f"  Error-uncertainty correlation: {unc_metrics['spearman_correlation']:.4f}")
    
    # Compute shift gap (performance drop from in-domain to out-of-domain)
    if 'dev_in' in results['predictions'] and 'dev_out' in results['predictions']:
        mae_in = results['predictions']['dev_in']['mae_kw']
        mae_out = results['predictions']['dev_out']['mae_kw']
        shift_gap = mae_out - mae_in
        shift_gap_percent = (shift_gap / mae_in) * 100 if mae_in > 0 else float('nan')
        
        results['shift_analysis'] = {
            'mae_in_domain': mae_in,
            'mae_out_domain': mae_out,
            'shift_gap_kw': shift_gap,
            'shift_gap_percent': shift_gap_percent
        }
        
        print(f"\n--- Shift Analysis ---")
        print(f"  Dev-in MAE: {mae_in:.2f} kW")
        print(f"  Dev-out MAE: {mae_out:.2f} kW")
        print(f"  Shift gap: {shift_gap:.2f} kW ({shift_gap_percent:.1f}% increase)")
    
    return results


def evaluate_all_models(
    data_module: PropulsionDataModule,
    config: Config,
    with_uncertainty: bool = False
) -> dict:
    """
    Evaluate all available trained models.
    
    Returns:
        Dictionary with all results
    """
    model_types = ['mean', 'linear', 'rf', 'xgboost', 'mlp', 'gaussian_mlp', 'ensemble']
    
    all_results = {}
    
    for model_type in model_types:
        result = evaluate_model(model_type, data_module, config, with_uncertainty)
        if result is not None:
            all_results[model_type] = result
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<15} {'Val MAE':<12} {'Dev-in MAE':<12} {'Dev-out MAE':<12} {'Shift Gap':<12}")
    print("-" * 80)
    
    for model_type, result in all_results.items():
        val_mae = result['predictions'].get('val', {}).get('mae_kw', float('nan'))
        dev_in_mae = result['predictions'].get('dev_in', {}).get('mae_kw', float('nan'))
        dev_out_mae = result['predictions'].get('dev_out', {}).get('mae_kw', float('nan'))
        shift_gap = result.get('shift_analysis', {}).get('shift_gap_kw', float('nan'))
        
        print(f"{model_type:<15} {val_mae:<12.2f} {dev_in_mae:<12.2f} {dev_out_mae:<12.2f} {shift_gap:<12.2f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate propulsion prediction models")
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['mean', 'linear', 'rf', 'xgboost', 'mlp', 'gaussian_mlp', 'ensemble', 'all'],
        help='Model type to evaluate'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--with_uncertainty',
        action='store_true',
        help='Evaluate uncertainty quality (for models that support it)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    print("=" * 70)
    print("PROPULSION POWER PREDICTION - EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data directory: {config.data.data_dir}")
    print(f"With uncertainty: {args.with_uncertainty}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data_module = PropulsionDataModule(config.data)
    data_module.setup()
    
    # Evaluate
    if args.model == 'all':
        results = evaluate_all_models(data_module, config, args.with_uncertainty)
    else:
        result = evaluate_model(args.model, data_module, config, args.with_uncertainty)
        results = {args.model: result} if result else {}
    
    # Save results
    output_path = args.output or os.path.join(config.output_dir, 'evaluation_results.json')
    
    # Prepare JSON-serializable results
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for model_type, result in results.items():
        if result is not None:
            json_results['models'][model_type] = {
                'predictions': result['predictions'],
                'shift_analysis': result.get('shift_analysis', {})
            }
            if result.get('uncertainty'):
                # Simplify uncertainty results for JSON
                json_results['models'][model_type]['uncertainty'] = {
                    split: {
                        'spearman_correlation': m['spearman_correlation'],
                        'mean_uncertainty': m['mean_uncertainty']
                    }
                    for split, m in result['uncertainty'].items()
                }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

