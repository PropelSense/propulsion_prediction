"""
Uncertainty quantification methods for propulsion power prediction.

Includes:
- Deep Ensemble
- MC Dropout
- Uncertainty metrics and analysis
"""
import numpy as np
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from copy import deepcopy

from neural_models import MLP, GaussianMLP, NeuralNetworkTrainer, create_mlp
from config import ModelConfig, default_config


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains multiple neural networks with different random initializations.
    Epistemic uncertainty is estimated from disagreement between ensemble members.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_members: int = 5,
        config: ModelConfig = None,
        gaussian_output: bool = False
    ):
        """
        Args:
            input_dim: Number of input features
            n_members: Number of ensemble members
            config: Model configuration
            gaussian_output: If True, each member outputs mean+variance
        """
        self.input_dim = input_dim
        self.n_members = n_members
        self.config = config or default_config.model
        self.gaussian_output = gaussian_output
        
        self.models: List[nn.Module] = []
        self.trainers: List[NeuralNetworkTrainer] = []
        self.is_trained = False
    
    def _create_member(self, seed: int) -> Tuple[nn.Module, NeuralNetworkTrainer]:
        """Create a single ensemble member with a specific random seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = create_mlp(
            input_dim=self.input_dim,
            config=self.config,
            gaussian_output=self.gaussian_output
        )
        
        trainer = NeuralNetworkTrainer(
            model=model,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            max_epochs=self.config.max_epochs,
            early_stopping_patience=self.config.early_stopping_patience
        )
        
        return model, trainer
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> List[dict]:
        """
        Train all ensemble members.
        
        Each member is trained with a different random seed for initialization
        and data shuffling, which creates diversity in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Print progress
            
        Returns:
            List of training histories for each member
        """
        base_seed = self.config.random_seed
        histories = []
        
        for i in range(self.n_members):
            seed = base_seed + i * 1000  # Different seed for each member
            
            if verbose:
                print(f"\n--- Training ensemble member {i+1}/{self.n_members} (seed={seed}) ---")
            
            model, trainer = self._create_member(seed)
            
            history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                verbose=verbose
            )
            
            self.models.append(model)
            self.trainers.append(trainer)
            histories.append(history)
        
        self.is_trained = True
        return histories
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (mean across ensemble).
        
        Returns:
            Mean predictions from all ensemble members
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")
        
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0)
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of:
            - mean: Mean prediction across ensemble
            - epistemic: Epistemic uncertainty (variance across members)
            - aleatoric: Aleatoric uncertainty (if gaussian_output=True)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")
        
        if self.gaussian_output:
            means = []
            variances = []
            
            for trainer in self.trainers:
                mean, var = trainer.predict_with_uncertainty(X)
                means.append(mean)
                variances.append(var)
            
            means = np.array(means)
            variances = np.array(variances)
            
            # Ensemble mean prediction
            mean_pred = np.mean(means, axis=0)
            
            # Epistemic uncertainty: variance of means across ensemble
            epistemic = np.var(means, axis=0)
            
            # Aleatoric uncertainty: mean of individual variances
            aleatoric = np.mean(variances, axis=0)
            
            return mean_pred, epistemic, aleatoric
        else:
            predictions = self._get_all_predictions(X)
            
            mean_pred = np.mean(predictions, axis=0)
            epistemic = np.var(predictions, axis=0)
            
            return mean_pred, epistemic, None
    
    def get_total_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and total uncertainty.
        
        Returns:
            Tuple of (predictions, total_uncertainty)
        """
        mean, epistemic, aleatoric = self.predict_with_uncertainty(X)
        
        if aleatoric is not None:
            total = epistemic + aleatoric
        else:
            total = epistemic
        
        return mean, total
    
    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all ensemble members."""
        predictions = []
        for trainer in self.trainers:
            pred = trainer.predict(X)
            predictions.append(pred)
        return np.array(predictions)
    
    def save(self, directory: str):
        """Save all ensemble members."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for i, trainer in enumerate(self.trainers):
            path = os.path.join(directory, f"member_{i}.pt")
            trainer.save(path)
        
        # Save ensemble metadata
        metadata = {
            'n_members': self.n_members,
            'input_dim': self.input_dim,
            'gaussian_output': self.gaussian_output
        }
        np.save(os.path.join(directory, "metadata.npy"), metadata)
        print(f"Ensemble saved to {directory}")


class MCDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Keeps dropout active at inference time and runs multiple forward passes
    to estimate uncertainty from the variance of predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        device: str = None
    ):
        """
        Args:
            model: Trained MLP model (must have dropout layers)
            n_samples: Number of forward passes for uncertainty estimation
            device: Computation device
        """
        self.model = model
        self.n_samples = n_samples
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
    
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty using MC Dropout.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (mean predictions, uncertainty as std)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        predictions = []
        
        # Run multiple forward passes with dropout enabled
        self._enable_dropout()
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                if isinstance(self.model, GaussianMLP):
                    pred, _ = self.model(X_tensor)
                else:
                    pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (mean of MC samples)."""
        mean, _ = self.predict_with_uncertainty(X)
        return mean


def compute_uncertainty_metrics(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    threshold: float = None
) -> dict:
    """
    Compute metrics for uncertainty quality.
    
    Args:
        errors: Prediction errors (absolute values)
        uncertainties: Uncertainty estimates
        threshold: Error threshold for acceptable predictions
        
    Returns:
        Dictionary of uncertainty metrics
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Correlation between uncertainty and errors
    spearman_corr, spearman_p = spearmanr(uncertainties, errors)
    pearson_corr, pearson_p = pearsonr(uncertainties, errors)
    
    # Compute rejection curve AUC
    # Sort by uncertainty (high to low)
    sorted_indices = np.argsort(uncertainties)[::-1]
    sorted_errors = errors[sorted_indices]
    
    # Compute cumulative mean error as we reject high-uncertainty samples
    n = len(errors)
    retention_fractions = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(sorted_errors[::-1])[::-1] / np.arange(n, 0, -1)
    
    # Area under rejection curve (lower is better for good uncertainty)
    rejection_auc = np.trapz(cumulative_errors, retention_fractions)
    
    # Oracle AUC (if we had perfect uncertainty = actual errors)
    oracle_sorted = np.sort(errors)[::-1]
    oracle_cumulative = np.cumsum(oracle_sorted[::-1])[::-1] / np.arange(n, 0, -1)
    oracle_auc = np.trapz(oracle_cumulative, retention_fractions)
    
    # Random baseline AUC
    random_auc = np.mean(errors)
    
    metrics = {
        'spearman_correlation': spearman_corr,
        'pearson_correlation': pearson_corr,
        'rejection_auc': rejection_auc,
        'oracle_auc': oracle_auc,
        'random_auc': random_auc,
        'normalized_auc': (rejection_auc - oracle_auc) / (random_auc - oracle_auc + 1e-10)
    }
    
    return metrics


def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Analyze calibration of uncertainty estimates.
    
    For well-calibrated uncertainty, samples in the p-th percentile of
    uncertainty should have p% of their errors within a certain threshold.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates (std)
        n_bins: Number of bins for analysis
        
    Returns:
        Calibration analysis results
    """
    errors = np.abs(y_true - y_pred)
    
    # Sort by uncertainty
    sorted_idx = np.argsort(uncertainty)
    sorted_errors = errors[sorted_idx]
    sorted_uncertainty = uncertainty[sorted_idx]
    
    # Bin samples by uncertainty
    bin_edges = np.linspace(0, len(errors), n_bins + 1, dtype=int)
    
    results = {
        'bin_mean_uncertainty': [],
        'bin_mean_error': [],
        'bin_rmse': [],
        'bin_count': []
    }
    
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        bin_errors = sorted_errors[start:end]
        bin_unc = sorted_uncertainty[start:end]
        
        results['bin_mean_uncertainty'].append(np.mean(bin_unc))
        results['bin_mean_error'].append(np.mean(bin_errors))
        results['bin_rmse'].append(np.sqrt(np.mean(bin_errors ** 2)))
        results['bin_count'].append(len(bin_errors))
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = np.random.randn(1000, 10).astype(np.float32)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(1000).astype(np.float32) * 0.5
    
    print("Testing Deep Ensemble...")
    ensemble = DeepEnsemble(input_dim=10, n_members=3)
    ensemble.train(X[:700], y[:700], X[700:850], y[700:850], verbose=False)
    
    mean, epistemic, _ = ensemble.predict_with_uncertainty(X[850:])
    print(f"Ensemble MAE: {np.mean(np.abs(mean - y[850:])):.4f}")
    print(f"Mean epistemic uncertainty: {np.mean(np.sqrt(epistemic)):.4f}")
    
    # Test MC Dropout
    print("\nTesting MC Dropout...")
    model = create_mlp(input_dim=10)
    trainer = NeuralNetworkTrainer(model, max_epochs=30)
    trainer.train(X[:700], y[:700], X[700:850], y[700:850], verbose=False)
    
    mc = MCDropout(model, n_samples=30)
    mc_mean, mc_std = mc.predict_with_uncertainty(X[850:])
    print(f"MC Dropout MAE: {np.mean(np.abs(mc_mean - y[850:])):.4f}")
    print(f"Mean MC std: {np.mean(mc_std):.4f}")

