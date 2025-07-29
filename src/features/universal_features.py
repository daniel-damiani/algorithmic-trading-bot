"""
Universal Feature Engineering Pipeline

Creates a standardized feature set that all models can use,
combining the best features from LSTM, XGBoost, and CNN approaches.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import talib
import structlog
from dataclasses import dataclass

logger = structlog.get_logger(__name__)


@dataclass
class UniversalFeatureConfig:
    """Configuration for universal feature engineering"""
    
    # Technical indicators
    add_moving_averages: bool = True
    ma_periods: List[int] = None
    add_rsi: bool = True
    rsi_periods: List[int] = None
    add_macd: bool = True
    add_bollinger_bands: bool = True
    bb_periods: List[int] = None
    add_atr: bool = True
    atr_periods: List[int] = None
    add_adx: bool = True
    add_volume_indicators: bool = True
    
    # Price action features
    add_price_ratios: bool = True
    add_volatility: bool = True
    volatility_periods: List[int] = None
    add_returns: bool = True
    return_periods: List[int] = None
    
    # Time-based features
    add_time_features: bool = True
    add_cyclical_encoding: bool = True
    
    # Lag features
    add_lag_features: bool = True
    lag_periods: List[int] = None
    add_rolling_stats: bool = True
    rolling_windows: List[int] = None
    
    # Market microstructure
    add_microstructure: bool = True
    add_garch_volatility: bool = True
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50]  # Reduced from [5, 10, 20, 50, 100, 200]
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21]  # Reduced from [14, 21, 28]
        if self.bb_periods is None:
            self.bb_periods = [20]  # Reduced from [20, 50]
        if self.atr_periods is None:
            self.atr_periods = [14]  # Reduced from [14, 21]
        if self.volatility_periods is None:
            self.volatility_periods = [5, 10, 20, 50]  # Reduced from [5, 10, 20, 50, 100, 200]
        if self.return_periods is None:
            self.return_periods = [5, 10, 20]  # Reduced from [5, 10, 20, 50, 100, 200]
        if self.lag_periods is None:
            self.lag_periods = [1, 5, 10]  # Reduced from [1, 5, 10, 20]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 30]


class UniversalFeatureGenerator:
    """Universal feature generator for all model types"""
    
    def __init__(self, config: UniversalFeatureConfig = None):
        self.config = config if config else UniversalFeatureConfig()
        self.feature_names = []
        
    def generate_features(
        self, 
        data: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """Generate all features from OHLCV data"""
        
        if not {'open', 'high', 'low', 'close', 'volume'}.issubset(data.columns):
            raise ValueError("Data must contain OHLCV columns")
        
        logger.info("Generating universal features", 
                   n_samples=len(data),
                   columns=list(data.columns))
        
        df = data.copy()
        
        # Start with OHLCV data
        features = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add technical indicators
        if self.config.add_moving_averages:
            features = self._add_moving_averages(features, df)
            
        if self.config.add_rsi:
            features = self._add_rsi(features, df)
            
        if self.config.add_macd:
            features = self._add_macd(features, df)
            
        if self.config.add_bollinger_bands:
            features = self._add_bollinger_bands(features, df)
            
        if self.config.add_atr:
            features = self._add_atr(features, df)
            
        if self.config.add_adx:
            features = self._add_adx(features, df)
            
        if self.config.add_volume_indicators:
            features = self._add_volume_indicators(features, df)
        
        # Add price action features
        if self.config.add_price_ratios:
            features = self._add_price_ratios(features, df)
            
        if self.config.add_volatility:
            features = self._add_volatility(features, df)
            
        if self.config.add_returns:
            features = self._add_returns(features, df)
        
        # Add time features
        if self.config.add_time_features and 'timestamp' in df.columns:
            features = self._add_time_features(features, df)
        
        # Add lag features
        if self.config.add_lag_features:
            features = self._add_lag_features(features)
        
        # Add microstructure features
        if self.config.add_microstructure:
            features = self._add_microstructure_features(features, df)
        
        # Add GARCH volatility
        if self.config.add_garch_volatility:
            features = self._add_garch_volatility(features, df)
        
        # Store feature names for consistent ordering
        if is_training:
            self.feature_names = [col for col in features.columns 
                                if col not in ['timestamp', 'symbol']]
        
        # Handle NaN values more intelligently
        # Only drop rows where critical features are NaN
        critical_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Drop rows where any critical feature is NaN
        features = features.dropna(subset=critical_features)
        
        # For other features, forward fill then backward fill to preserve data
        features = features.ffill().bfill()
        
        # As a last resort, fill remaining NaN with 0
        features = features.fillna(0)
        
        # Defragment DataFrame to improve performance
        features = features.copy()
        
        logger.info("Feature generation completed",
                   n_features=len(features.columns),
                   n_samples=len(features))
        
        return features
    
    def _add_moving_averages(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        for period in self.config.ma_periods:
            # Simple Moving Average
            features[f'sma_{period}'] = talib.SMA(data['close'].values.astype(np.float64), timeperiod=period)
            
            # Exponential Moving Average
            features[f'ema_{period}'] = talib.EMA(data['close'].values.astype(np.float64), timeperiod=period)
            
            # Price to SMA ratio
            features[f'sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
        
        return features
    
    def _add_rsi(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicators"""
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = talib.RSI(data['close'].values.astype(np.float64), timeperiod=period)
        
        return features
    
    def _add_macd(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators"""
        macd, macd_signal, macd_hist = talib.MACD(data['close'].values.astype(np.float64))
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_diff'] = macd_hist
        
        return features
    
    def _add_bollinger_bands(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        for period in self.config.bb_periods:
            upper, middle, lower = talib.BBANDS(
                data['close'].values.astype(np.float64), 
                timeperiod=period, 
                nbdevup=2, 
                nbdevdn=2
            )
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = (upper - lower) / middle
            features[f'bb_position_{period}'] = (data['close'] - lower) / (upper - lower)
        
        return features
    
    def _add_atr(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range"""
        for period in self.config.atr_periods:
            features[f'atr_{period}'] = talib.ATR(
                data['high'].values.astype(np.float64),
                data['low'].values.astype(np.float64),
                data['close'].values.astype(np.float64),
                timeperiod=period
            )
        
        return features
    
    def _add_adx(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add ADX (Average Directional Index)"""
        features['adx'] = talib.ADX(
            data['high'].values.astype(np.float64),
            data['low'].values.astype(np.float64),
            data['close'].values.astype(np.float64),
            timeperiod=14
        )
        
        return features
    
    def _add_volume_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        # On Balance Volume
        features['obv'] = talib.OBV(data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64))
        
        # Volume Weighted Average Price
        features['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Volume moving averages
        for period in [5, 10, 20]:
            features[f'volume_ma_{period}'] = talib.SMA(data['volume'].values.astype(np.float64), timeperiod=period)
            features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_ma_{period}']
        
        return features
    
    def _add_price_ratios(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio features"""
        features['hl_spread'] = (data['high'] - data['low']) / data['close']
        features['hl_spread_ma'] = talib.SMA(features['hl_spread'].values.astype(np.float64), timeperiod=20)
        
        # Price position within daily range
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        return features
    
    def _add_volatility(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        returns = data['close'].pct_change()
        
        for period in self.config.volatility_periods:
            features[f'volatility_{period}'] = returns.rolling(window=period).std()
            features[f'skew_{period}'] = returns.rolling(window=period).skew()
            # Use kurt() instead of kurtosis() for pandas compatibility
            features[f'kurtosis_{period}'] = returns.rolling(window=period).apply(lambda x: x.kurtosis(), raw=False)
        
        return features
    
    def _add_returns(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add return features"""
        for period in self.config.return_periods:
            features[f'return_{period}'] = data['close'].pct_change(periods=period)
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' not in data.columns:
            return features
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if self.config.add_cyclical_encoding:
            # Cyclical encoding for time features
            features['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            features['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            features['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
            features['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        else:
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['month'] = df['timestamp'].dt.month
        
        return features
    
    def _add_lag_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        # Select key columns for lagging (avoid lagging all features to prevent explosion)
        key_columns = ['close', 'volume', 'rsi_14', 'macd', 'bb_position_20']
        lag_columns = [col for col in key_columns if col in features.columns]
        
        for col in lag_columns:
            for lag in self.config.lag_periods:
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)
                
            if self.config.add_rolling_stats:
                for window in self.config.rolling_windows:
                    features[f'{col}_rolling_mean_{window}'] = features[col].rolling(window=window).mean()
                    features[f'{col}_rolling_std_{window}'] = features[col].rolling(window=window).std()
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Amihud illiquidity ratio
        features['amihud_illiquidity'] = abs(data['close'].pct_change()) / data['volume']
        
        # Kyle's lambda (price impact)
        returns = data['close'].pct_change()
        volume_imbalance = data['volume'] / data['volume'].rolling(window=20).mean()
        features['kyle_lambda'] = abs(returns) / volume_imbalance
        
        # Volatility ratio
        short_vol = returns.rolling(window=5).std()
        long_vol = returns.rolling(window=20).std()
        features['volatility_ratio'] = short_vol / long_vol
        
        return features
    
    def _add_garch_volatility(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add GARCH volatility (simplified)"""
        returns = data['close'].pct_change()
        
        # Simplified GARCH(1,1) volatility estimate
        alpha = 0.1
        beta = 0.8
        
        garch_vol = np.zeros(len(returns))
        garch_vol[0] = returns.std()
        
        for i in range(1, len(returns)):
            garch_vol[i] = np.sqrt(
                alpha * returns.iloc[i-1]**2 + 
                beta * garch_vol[i-1]**2 + 
                (1 - alpha - beta) * returns.var()
            )
        
        features['garch_volatility'] = garch_vol
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def transform_for_model(
        self, 
        features: pd.DataFrame, 
        model_type: str
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform features for specific model type"""
        
        if model_type.lower() == 'lstm':
            # LSTM needs features in the order it was trained
            return features[self.feature_names] if self.feature_names else features
            
        elif model_type.lower() == 'xgboost':
            # XGBoost can handle the features as-is
            return features[self.feature_names] if self.feature_names else features
            
        elif model_type.lower() == 'cnn':
            # CNN needs OHLCV data for chart generation
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            if 'timestamp' in features.columns:
                base_columns.append('timestamp')
            return features[base_columns]
            
        else:
            return features[self.feature_names] if self.feature_names else features