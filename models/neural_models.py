"""
Neural network models for propulsion power prediction.

Includes:
- Basic MLP
- MLP with Gaussian output (for aleatoric uncertainty)
- MC Dropout wrapper
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import ModelConfig, default_config


class MLP(nn.Module):
    """
    Multi-layer perceptron for regression.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (single output for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> Callable:
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class GaussianMLP(nn.Module):
    """
    MLP that outputs both mean and variance for Gaussian likelihood.
    This models aleatoric (data) uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        min_variance: float = 1e-6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.min_variance = min_variance
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for hidden_dim in hidden_layers[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Last hidden layer
        last_hidden = hidden_layers[-1] if hidden_layers else input_dim
        self.last_hidden = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            activation_fn(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # Separate heads for mean and variance
        self.mean_head = nn.Linear(last_hidden, 1)
        self.var_head = nn.Linear(last_hidden, 1)
    
    def _get_activation(self, name: str) -> Callable:
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }
        return activations.get(name, nn.ReLU)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and variance.
        
        Returns:
            Tuple of (mean, variance) tensors
        """
        features = self.feature_extractor(x)
        hidden = self.last_hidden(features)
        
        mean = self.mean_head(hidden).squeeze(-1)
        # Use softplus to ensure positive variance
        var = nn.functional.softplus(self.var_head(hidden).squeeze(-1)) + self.min_variance
        
        return mean, var


class NeuralNetworkTrainer:
    """
    Trainer class for neural network models.
    Handles training loop, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> dict:
        """
        Train the model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        is_gaussian = isinstance(self.model, GaussianMLP)
        
        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                if is_gaussian:
                    mean, var = self.model(X_batch)
                    # Negative log-likelihood for Gaussian
                    loss = torch.mean(
                        0.5 * torch.log(var) + 0.5 * ((y_batch - mean) ** 2) / var
                    )
                else:
                    pred = self.model(X_batch)
                    loss = nn.functional.mse_loss(pred, y_batch)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(X_batch)
            
            train_loss /= len(X_train)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self._evaluate(val_loader, is_gaussian)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }
    
    def _evaluate(self, dataloader: DataLoader, is_gaussian: bool) -> float:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                if is_gaussian:
                    mean, var = self.model(X_batch)
                    loss = torch.mean(
                        0.5 * torch.log(var) + 0.5 * ((y_batch - mean) ** 2) / var
                    )
                else:
                    pred = self.model(X_batch)
                    loss = nn.functional.mse_loss(pred, y_batch)
                
                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)
        
        return total_loss / total_samples
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, GaussianMLP):
                mean, _ = self.model(X_tensor)
                return mean.cpu().numpy()
            else:
                return self.model(X_tensor).cpu().numpy()
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty (for GaussianMLP).
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not isinstance(self.model, GaussianMLP):
            raise ValueError("predict_with_uncertainty requires GaussianMLP model")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            mean, var = self.model(X_tensor)
            return mean.cpu().numpy(), var.cpu().numpy()
    
    def save(self, path: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Model loaded from {path}")


def create_mlp(
    input_dim: int,
    config: ModelConfig = None,
    gaussian_output: bool = False
) -> nn.Module:
    """
    Factory function to create MLP models.
    
    Args:
        input_dim: Number of input features
        config: Model configuration
        gaussian_output: If True, create GaussianMLP for uncertainty
        
    Returns:
        MLP or GaussianMLP model
    """
    config = config or default_config.model
    
    if gaussian_output:
        return GaussianMLP(
            input_dim=input_dim,
            hidden_layers=config.hidden_layers,
            dropout_rate=config.dropout_rate,
            activation=config.activation
        )
    else:
        return MLP(
            input_dim=input_dim,
            hidden_layers=config.hidden_layers,
            dropout_rate=config.dropout_rate,
            activation=config.activation
        )


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = np.random.randn(1000, 10).astype(np.float32)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(1000).astype(np.float32) * 0.1
    
    model = create_mlp(input_dim=10)
    trainer = NeuralNetworkTrainer(model, max_epochs=50)
    
    history = trainer.train(
        X[:800], y[:800],
        X[800:], y[800:],
        verbose=True
    )
    
    preds = trainer.predict(X[800:])
    print(f"\nTest MAE: {np.mean(np.abs(preds - y[800:])):.4f}")

