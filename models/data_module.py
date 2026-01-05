"""
Data loading and preprocessing module.

IMPORTANT: All transformations are fit ONLY on training data to prevent data leakage.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from config import DataConfig, default_config


class FeatureEngineer:
    """
    Handles feature engineering transformations.
    All derived features are computed using the same logic for train and test,
    but any statistics needed are computed only from training data.
    """
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit on training data (currently stateless, but kept for consistency)."""
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df = df.copy()
        
        # Speed cubed - power relationship is roughly cubic with speed
        if 'stw' in df.columns:
            df['stw_cubed'] = df['stw'] ** 3
            df['stw_squared'] = df['stw'] ** 2
        
        # Draft features
        if 'draft_aft_telegram' in df.columns and 'draft_fore_telegram' in df.columns:
            df['mean_draft'] = (df['draft_aft_telegram'] + df['draft_fore_telegram']) / 2
            df['trim'] = df['draft_aft_telegram'] - df['draft_fore_telegram']
        
        # Wind magnitude and direction
        if 'awind_vcomp_provider' in df.columns and 'awind_ucomp_provider' in df.columns:
            df['wind_magnitude'] = np.sqrt(
                df['awind_vcomp_provider']**2 + df['awind_ucomp_provider']**2
            )
            df['wind_angle'] = np.arctan2(
                df['awind_ucomp_provider'], df['awind_vcomp_provider']
            )
        
        # Current magnitude and direction
        if 'rcurrent_vcomp' in df.columns and 'rcurrent_ucomp' in df.columns:
            df['current_magnitude'] = np.sqrt(
                df['rcurrent_vcomp']**2 + df['rcurrent_ucomp']**2
            )
            df['current_angle'] = np.arctan2(
                df['rcurrent_ucomp'], df['rcurrent_vcomp']
            )
        
        # Interaction: speed with wind resistance
        if 'stw' in df.columns and 'wind_magnitude' in df.columns:
            df['speed_wind_interaction'] = df['stw'] * df['wind_magnitude']
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class DataScaler:
    """
    Handles data normalization with careful separation of train/test.
    
    CRITICAL: fit() must only be called on training data!
    """
    
    def __init__(self, time_feature: str = 'timeSinceDryDock'):
        self.scaler = StandardScaler()
        self.time_feature = time_feature
        self.time_max = 4324320  # ~8 years in minutes
        self.fitted = False
        self.feature_names: List[str] = []
    
    def fit(self, X: pd.DataFrame) -> 'DataScaler':
        """
        Fit scaler on training data ONLY.
        
        Args:
            X: Training features (NOT including target)
        """
        self.feature_names = list(X.columns)
        
        # Separate time feature for special scaling
        features_to_scale = [f for f in self.feature_names if f != self.time_feature]
        
        if features_to_scale:
            self.scaler.fit(X[features_to_scale])
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features as numpy array
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fit before transform. Call fit() first.")
        
        X = X.copy()
        features_to_scale = [f for f in self.feature_names if f != self.time_feature]
        
        result = pd.DataFrame(index=X.index)
        
        # Standard scale most features
        if features_to_scale:
            scaled = self.scaler.transform(X[features_to_scale])
            for i, fname in enumerate(features_to_scale):
                result[fname] = scaled[:, i]
        
        # Special scaling for time feature
        if self.time_feature in self.feature_names:
            result[self.time_feature] = X[self.time_feature] / self.time_max
        
        # Reorder to match original feature order
        result = result[self.feature_names]
        
        return result.values
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on data and transform it."""
        return self.fit(X).transform(X)
    
    def save(self, path: str):
        """Save scaler to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'time_feature': self.time_feature,
            'time_max': self.time_max,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'DataScaler':
        """Load scaler from disk."""
        data = joblib.load(path)
        instance = cls(time_feature=data['time_feature'])
        instance.scaler = data['scaler']
        instance.time_max = data['time_max']
        instance.feature_names = data['feature_names']
        instance.fitted = data['fitted']
        return instance


class TargetScaler:
    """Separate scaler for target variable."""
    
    def __init__(self):
        self.mean: float = 0.0
        self.std: float = 1.0
        self.fitted = False
    
    def fit(self, y: np.ndarray) -> 'TargetScaler':
        """Fit on training targets ONLY."""
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        self.fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Scale target values."""
        if not self.fitted:
            raise RuntimeError("TargetScaler must be fit before transform.")
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return y_scaled * self.std + self.mean
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)
    
    def save(self, path: str):
        joblib.dump({'mean': self.mean, 'std': self.std, 'fitted': self.fitted}, path)
    
    @classmethod
    def load(cls, path: str) -> 'TargetScaler':
        data = joblib.load(path)
        instance = cls()
        instance.mean = data['mean']
        instance.std = data['std']
        instance.fitted = data['fitted']
        return instance


class PropulsionDataModule:
    """
    Main data module for propulsion prediction.
    
    Handles loading, preprocessing, and splitting data with NO data leakage.
    All transformations are fit exclusively on training data.
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or default_config.data
        
        self.feature_engineer = FeatureEngineer()
        self.feature_scaler = DataScaler()
        self.target_scaler = TargetScaler()
        
        # Data storage
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.dev_in_df: Optional[pd.DataFrame] = None
        self.dev_out_df: Optional[pd.DataFrame] = None
        
        self.feature_columns: List[str] = []
        self.is_setup = False
    
    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file and set index."""
        df = pd.read_csv(filepath)
        df.set_index(self.config.index_column, inplace=True, drop=True)
        df.sort_index(inplace=True)
        return df
    
    def setup(self):
        """
        Load and preprocess all data.
        
        The preprocessing pipeline:
        1. Load raw data
        2. Apply feature engineering (same transformations for all splits)
        3. Split training data into train/val
        4. Fit scalers on training split ONLY
        5. Transform all splits using fitted scalers
        """
        # Load raw data
        train_path = os.path.join(self.config.data_dir, self.config.train_file)
        dev_in_path = os.path.join(self.config.data_dir, self.config.dev_in_file)
        dev_out_path = os.path.join(self.config.data_dir, self.config.dev_out_file)
        
        print("Loading datasets...")
        full_train_df = self._load_csv(train_path)
        self.dev_in_df = self._load_csv(dev_in_path)
        self.dev_out_df = self._load_csv(dev_out_path)
        
        print(f"  Train: {len(full_train_df)} samples")
        print(f"  Dev-in: {len(self.dev_in_df)} samples")
        print(f"  Dev-out: {len(self.dev_out_df)} samples")
        
        # Feature engineering (same logic applied to all)
        print("Applying feature engineering...")
        full_train_df = self.feature_engineer.fit_transform(full_train_df)
        self.dev_in_df = self.feature_engineer.transform(self.dev_in_df)
        self.dev_out_df = self.feature_engineer.transform(self.dev_out_df)
        
        # Determine feature columns (exclude target)
        self.feature_columns = [
            col for col in full_train_df.columns 
            if col != self.config.target_column
        ]
        print(f"  Using {len(self.feature_columns)} features")
        
        # Split training data into train/validation
        # IMPORTANT: This split happens BEFORE fitting scalers
        print(f"Splitting train into train/val ({1-self.config.val_split:.0%}/{self.config.val_split:.0%})...")
        train_idx, val_idx = train_test_split(
            full_train_df.index,
            test_size=self.config.val_split,
            random_state=self.config.random_seed
        )
        
        self.train_df = full_train_df.loc[train_idx]
        self.val_df = full_train_df.loc[val_idx]
        
        print(f"  Train split: {len(self.train_df)} samples")
        print(f"  Val split: {len(self.val_df)} samples")
        
        # Fit scalers on TRAINING SPLIT ONLY
        print("Fitting scalers on training data only...")
        train_features = self.train_df[self.feature_columns]
        train_target = self.train_df[self.config.target_column].values
        
        self.feature_scaler.fit(train_features)
        self.target_scaler.fit(train_target)
        
        self.is_setup = True
        print("Data module setup complete.")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scaled training data."""
        self._check_setup()
        X = self.feature_scaler.transform(self.train_df[self.feature_columns])
        y = self.target_scaler.transform(self.train_df[self.config.target_column].values)
        return X, y
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scaled validation data."""
        self._check_setup()
        X = self.feature_scaler.transform(self.val_df[self.feature_columns])
        y = self.target_scaler.transform(self.val_df[self.config.target_column].values)
        return X, y
    
    def get_dev_in_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scaled dev-in data."""
        self._check_setup()
        X = self.feature_scaler.transform(self.dev_in_df[self.feature_columns])
        y = self.target_scaler.transform(self.dev_in_df[self.config.target_column].values)
        return X, y
    
    def get_dev_out_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scaled dev-out data."""
        self._check_setup()
        X = self.feature_scaler.transform(self.dev_out_df[self.feature_columns])
        y = self.target_scaler.transform(self.dev_out_df[self.config.target_column].values)
        return X, y
    
    def get_raw_targets(self, split: str) -> np.ndarray:
        """Get unscaled target values for a split."""
        self._check_setup()
        if split == 'train':
            return self.train_df[self.config.target_column].values
        elif split == 'val':
            return self.val_df[self.config.target_column].values
        elif split == 'dev_in':
            return self.dev_in_df[self.config.target_column].values
        elif split == 'dev_out':
            return self.dev_out_df[self.config.target_column].values
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def inverse_transform_predictions(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale (kW)."""
        return self.target_scaler.inverse_transform(y_scaled)
    
    def _check_setup(self):
        if not self.is_setup:
            raise RuntimeError("DataModule not set up. Call setup() first.")
    
    def save_scalers(self, directory: str):
        """Save fitted scalers for later use."""
        os.makedirs(directory, exist_ok=True)
        self.feature_scaler.save(os.path.join(directory, 'feature_scaler.joblib'))
        self.target_scaler.save(os.path.join(directory, 'target_scaler.joblib'))
        print(f"Scalers saved to {directory}")
    
    def load_scalers(self, directory: str):
        """Load previously fitted scalers."""
        self.feature_scaler = DataScaler.load(
            os.path.join(directory, 'feature_scaler.joblib')
        )
        self.target_scaler = TargetScaler.load(
            os.path.join(directory, 'target_scaler.joblib')
        )
        print(f"Scalers loaded from {directory}")
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return len(self.feature_columns)


if __name__ == "__main__":
    # Quick test
    dm = PropulsionDataModule()
    dm.setup()
    
    X_train, y_train = dm.get_train_data()
    X_val, y_val = dm.get_val_data()
    
    print(f"\nTrain shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Feature columns: {dm.feature_columns}")

