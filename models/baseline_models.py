"""
Baseline models for propulsion power prediction.

Includes:
- Mean baseline
- Linear regression
- Random Forest
- XGBoost
"""
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. XGBModel will not be available.")

from config import ModelConfig, default_config


class MeanBaseline(BaseEstimator, RegressorMixin):
    """
    Simple baseline that predicts the mean of training targets.
    Useful as a sanity check - any real model should beat this.
    """
    
    def __init__(self):
        self.mean_: float = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MeanBaseline':
        """Compute mean from training data."""
        self.mean_ = float(np.mean(y))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean for all samples."""
        return np.full(len(X), self.mean_)


class LinearModel(BaseEstimator, RegressorMixin):
    """
    Linear regression baseline with optional L2 regularization.
    """
    
    def __init__(self, alpha: float = 0.0):
        """
        Args:
            alpha: L2 regularization strength. 0 = no regularization.
        """
        self.alpha = alpha
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """Fit linear model on training data."""
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    @property
    def coef_(self) -> np.ndarray:
        """Access model coefficients."""
        return self.model.coef_
    
    @property
    def intercept_(self) -> float:
        """Access model intercept."""
        return self.model.intercept_


class RandomForestModel(BaseEstimator, RegressorMixin):
    """
    Random Forest regressor wrapper with sensible defaults.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Fit random forest on training data."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        Uncertainty is computed as variance across individual trees.
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])
        
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)
        
        return mean_pred, std_pred
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Access feature importances."""
        return self.model.feature_importances_


class XGBModel(BaseEstimator, RegressorMixin):
    """
    XGBoost regressor wrapper with early stopping support.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = 10,
        n_jobs: int = -1
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.model = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'XGBModel':
        """
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
        """
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0
        )
        
        if X_val is not None and y_val is not None and self.early_stopping_rounds:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    @property
    def feature_importances_(self) -> np.ndarray:
        """Access feature importances."""
        return self.model.feature_importances_


def create_baseline_model(
    model_type: str,
    config: ModelConfig = None
) -> BaseEstimator:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: One of 'mean', 'linear', 'ridge', 'rf', 'xgboost'
        config: Model configuration
        
    Returns:
        Model instance
    """
    config = config or default_config.model
    
    if model_type == 'mean':
        return MeanBaseline()
    
    elif model_type == 'linear':
        return LinearModel(alpha=0.0)
    
    elif model_type == 'ridge':
        return LinearModel(alpha=1.0)
    
    elif model_type == 'rf':
        return RandomForestModel(
            n_estimators=config.rf_n_estimators,
            random_state=config.random_seed
        )
    
    elif model_type == 'xgboost':
        return XGBModel(
            n_estimators=config.xgb_n_estimators,
            learning_rate=config.xgb_learning_rate,
            max_depth=config.xgb_max_depth,
            random_state=config.random_seed
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model: BaseEstimator, path: str):
    """Save model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> BaseEstimator:
    """Load model from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 10)
    y_dummy = np.sum(X_dummy[:, :3], axis=1) + np.random.randn(1000) * 0.1
    
    for model_type in ['mean', 'linear', 'rf']:
        model = create_baseline_model(model_type)
        model.fit(X_dummy[:800], y_dummy[:800])
        preds = model.predict(X_dummy[800:])
        print(f"{model_type}: MAE = {np.mean(np.abs(preds - y_dummy[800:])):.4f}")

