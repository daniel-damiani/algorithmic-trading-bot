"""
Time Series Model Base Class

Base class for time series models like LSTM, GRU, etc.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import structlog
import joblib
from pathlib import Path

from .base_model import BaseModel, ModelConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class TimeSeriesConfig(ModelConfig):
    """Configuration specific to time series models"""
    
    # Time series specific parameters
    sequence_length: int = 60  # Number of timesteps to look back
    stride: int = 1  # Step size for sliding window
    forecast_horizon: int = 1  # Number of timesteps to predict
    
    # Preprocessing
    scaling_method: str = "standard"  # "standard", "minmax", or "none"
    detrend: bool = False
    handle_missing: str = "interpolate"  # "interpolate", "forward_fill", "drop"
    
    # Feature engineering
    add_time_features: bool = True  # Add hour, day, month features
    add_technical_indicators: bool = True
    add_lag_features: bool = True
    lag_periods: List[int] = None
    
    # Model architecture hints
    hidden_size: int = 256  # Increased default
    num_layers: int = 3  # More layers
    bidirectional: bool = True  # Use bidirectional LSTM
    
    def __post_init__(self):
        super().__post_init__()
        if self.lag_periods is None:
            self.lag_periods = [1, 5, 10, 20]
        self.model_type = ModelType.PRICE_PREDICTION


class TimeSeriesModel(BaseModel):
    """Base class for time series models"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__(config)
        self.config: TimeSeriesConfig = config
        self.scaler = None
        self.target_scaler = None  # Separate scaler for targets
        self.feature_columns = []
        self.target_columns = []
        
        # Initialize scalers based on config
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
            self.target_scaler = StandardScaler()  # Use StandardScaler for targets
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series models
        
        Args:
            data: Input data (n_samples, n_features)
            targets: Target data (n_samples, n_targets)
            
        Returns:
            Tuple of (sequences, target_sequences)
        """
        n_samples = len(data)
        seq_length = self.config.sequence_length
        stride = self.config.stride
        
        # Calculate number of sequences
        n_sequences = (n_samples - seq_length - self.config.forecast_horizon + 1) // stride
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {seq_length + self.config.forecast_horizon} samples")
        
        # Create sequences
        sequences = []
        target_sequences = [] if targets is not None else None
        
        for i in range(0, n_sequences * stride, stride):
            # Input sequence
            seq = data[i:i + seq_length]
            sequences.append(seq)
            
            # Target sequence
            if targets is not None:
                if self.config.forecast_horizon == 1:
                    target = targets[i + seq_length]
                else:
                    target = targets[i + seq_length:i + seq_length + self.config.forecast_horizon]
                target_sequences.append(target)
        
        sequences = np.array(sequences)
        if target_sequences is not None:
            target_sequences = np.array(target_sequences)
        
        return sequences, target_sequences
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to dataframe"""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found for time features")
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop intermediate columns
        df.drop(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter'], 
                axis=1, inplace=True)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Add lagged features"""
        # Create all new columns at once to avoid fragmentation
        new_columns = {}
        
        for col in columns:
            if col in df.columns:
                for lag in self.config.lag_periods:
                    new_columns[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                # Add rolling statistics
                new_columns[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
                new_columns[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
                new_columns[f'{col}_rolling_mean_30'] = df[col].rolling(window=30).mean()
                new_columns[f'{col}_rolling_std_30'] = df[col].rolling(window=30).std()
        
        # Concatenate all new columns at once
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        
        return df
    
    def prepare_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare time series data"""
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        df = data.copy()
        
        # Handle missing values
        if self.config.handle_missing == "interpolate":
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.config.handle_missing == "forward_fill":
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif self.config.handle_missing == "drop":
            df = df.dropna()
        
        # Add features if configured
        if self.config.add_time_features and 'timestamp' in df.columns:
            df = self.add_time_features(df)
        
        # Add lag features for numeric columns
        if self.config.add_lag_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.add_lag_features(df, numeric_cols)
        
        # Drop any remaining NaN values created by lagging
        df = df.dropna()
        
        # Store feature columns
        if is_training:
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['timestamp', 'symbol']]
            if labels is not None:
                if isinstance(labels, pd.Series):
                    self.target_columns = [labels.name]
                elif isinstance(labels, pd.DataFrame):
                    self.target_columns = labels.columns.tolist()
        
        # Select features
        feature_data = df[self.feature_columns].values
        
        # Scale features
        if self.scaler is not None:
            if is_training:
                feature_data = self.scaler.fit_transform(feature_data)
            else:
                feature_data = self.scaler.transform(feature_data)
        
        # Prepare labels
        if labels is not None:
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                labels = labels.values
            # Ensure labels are aligned with data
            labels = labels[-len(feature_data):]
            
            # Scale targets if scaler is available
            if self.target_scaler is not None:
                # Reshape if needed
                labels_reshaped = labels.reshape(-1, 1) if len(labels.shape) == 1 else labels
                
                if is_training:
                    labels = self.target_scaler.fit_transform(labels_reshaped)
                else:
                    labels = self.target_scaler.transform(labels_reshaped)
                
                # Flatten if originally 1D
                if len(labels.shape) > 1 and labels.shape[1] == 1:
                    labels = labels.flatten()
        
        # Create sequences
        sequences, target_sequences = self.create_sequences(feature_data, labels)
        
        logger.debug("Data prepared",
                    n_sequences=len(sequences),
                    sequence_shape=sequences.shape,
                    n_features=len(self.feature_columns))
        
        return sequences, target_sequences
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if self.target_scaler is None or self.config.scaling_method == "none":
            return predictions
        
        # Use target scaler for inverse transformation
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        unscaled = self.target_scaler.inverse_transform(predictions)
        
        # Return original shape
        if unscaled.shape[1] == 1:
            return unscaled.flatten()
        return unscaled
    
    def create_train_val_split(
        self, 
        sequences: np.ndarray, 
        targets: np.ndarray,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create time series aware train/validation split"""
        n_samples = len(sequences)
        split_idx = int(n_samples * (1 - val_size))
        
        train_X = sequences[:split_idx]
        train_y = targets[:split_idx]
        val_X = sequences[split_idx:]
        val_y = targets[split_idx:]
        
        return train_X, train_y, val_X, val_y
    
    def get_lookback_requirement(self) -> int:
        """Get minimum number of historical points needed"""
        base_lookback = self.config.sequence_length
        
        if self.config.add_lag_features and self.config.lag_periods:
            base_lookback += max(self.config.lag_periods)
        
        # Add buffer for rolling features
        if self.config.add_lag_features:
            base_lookback += 30  # For 30-day rolling features
        
        return base_lookback
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save time series model with scalers"""
        # Call parent save method
        base_path = super().save(path)
        
        # Save scalers separately
        if self.scaler is not None:
            scaler_path = base_path.with_suffix('.scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
        if self.target_scaler is not None:
            target_scaler_path = base_path.with_suffix('.target_scaler.pkl')
            joblib.dump(self.target_scaler, target_scaler_path)
            
        # Save additional time series specific data
        ts_data_path = base_path.with_suffix('.ts_data.pkl')
        joblib.dump({
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }, ts_data_path)
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'TimeSeriesModel':
        """Load time series model with scalers"""
        # Call parent load method
        model_instance = super().load(path)
        
        # Load scalers
        scaler_path = path.with_suffix('.scaler.pkl')
        if scaler_path.exists():
            model_instance.scaler = joblib.load(scaler_path)
            
        target_scaler_path = path.with_suffix('.target_scaler.pkl')
        if target_scaler_path.exists():
            model_instance.target_scaler = joblib.load(target_scaler_path)
            
        # Load time series specific data
        ts_data_path = path.with_suffix('.ts_data.pkl')
        if ts_data_path.exists():
            ts_data = joblib.load(ts_data_path)
            model_instance.feature_columns = ts_data['feature_columns']
            model_instance.target_columns = ts_data['target_columns']
        
        return model_instance