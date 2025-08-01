"""
Model Training Pipeline

Orchestrates the training of all models in the QuantumSentiment system.
Handles data preparation, model training, validation, and ensemble creation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import structlog
import json
import joblib
from abc import ABC, abstractmethod

from ..models import (
    PriceLSTM, PriceLSTMConfig,
    ChartPatternCNN, ChartPatternConfig,
    MarketRegimeXGBoost, MarketRegimeConfig,
    FinBERT, FinBERTConfig,
    StackedEnsemble, StackedEnsembleConfig,
    BaseModel
)
from ..features.universal_features import UniversalFeatureGenerator, UniversalFeatureConfig

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline"""
    
    # Data configuration
    train_start_date: str = "2020-01-01"
    train_end_date: str = "2023-12-31"
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training configuration
    parallel_training: bool = True
    max_workers: int = 4
    use_gpu: bool = True
    random_seed: int = 42
    
    # Model selection
    train_lstm: bool = True
    train_cnn: bool = True
    train_xgboost: bool = True
    train_finbert: bool = False  # Set to False by default due to heavy compute
    train_ensemble: bool = True
    
    # Output configuration
    model_save_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoint_interval: int = 10  # Save checkpoint every N epochs
    save_best_only: bool = True
    
    # Performance thresholds
    min_accuracy: float = 0.55
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.2
    
    # Monitoring
    track_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'sharpe_ratio', 'max_drawdown'
    ])
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = "validation_f1"
    
    def __post_init__(self):
        self.model_save_dir = Path(self.model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True, parents=True)


class DataPreprocessor:
    """Handles data preprocessing for different model types"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Initialize universal feature generator
        self.feature_generator = UniversalFeatureGenerator(UniversalFeatureConfig())
        
    def prepare_price_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare price data for LSTM training using universal features"""
        # Sort by timestamp
        data = raw_data.sort_values('timestamp').copy()
        
        # Generate universal features
        features = self.feature_generator.generate_features(data, is_training=True)
        
        # Transform features specifically for LSTM
        features = self.feature_generator.transform_for_model(features, 'lstm')
        
        # Create targets - use log returns for better statistical properties
        # Calculate forward log returns (next period return)
        close_prices = data['close']
        log_prices = np.log(close_prices)
        targets = log_prices.diff().shift(-1)  # Forward log returns
        
        # Alternative: Use smoothed returns for less noise
        # targets = close_prices.pct_change(5).shift(-5) / 5  # 5-period average return
        
        # Clip extreme values to reduce impact of outliers
        targets = targets.clip(lower=-0.02, upper=0.02)  # Clip to ±2% log returns
        
        # Align features and targets by matching indices
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index].dropna()
        
        # Final alignment after dropping NaN targets
        features = features.loc[targets.index]
        
        # Log the final data shape
        logger.info(f"Prepared LSTM data: features shape={features.shape}, targets shape={targets.shape}")
        
        return features, targets
    
    def prepare_chart_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare chart data for CNN training using universal features"""
        # Sort by timestamp
        data = raw_data.sort_values('timestamp').copy()
        
        # Generate universal features
        features = self.feature_generator.generate_features(data, is_training=True)
        
        # Transform features specifically for CNN (returns OHLCV data)
        chart_data = self.feature_generator.transform_for_model(features, 'cnn')
        
        # Create pattern labels using the same pattern generation logic
        patterns = self._generate_pattern_labels(data)
        
        # Align chart data and patterns
        common_index = chart_data.index.intersection(patterns.index)
        chart_data = chart_data.loc[common_index]
        patterns = patterns.loc[common_index]
        
        return chart_data, patterns
    
    def prepare_regime_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for market regime classification using universal features"""
        # Sort by timestamp
        data = raw_data.sort_values('timestamp').copy()
        
        # Generate universal features
        features = self.feature_generator.generate_features(data, is_training=True)
        
        # Transform features specifically for XGBoost
        regime_data = self.feature_generator.transform_for_model(features, 'xgboost')
        
        # Generate regime labels based on market conditions
        regime_labels = self._generate_regime_labels(data, features)
        
        # Align labels with features
        common_index = regime_data.index.intersection(regime_labels.index)
        regime_data = regime_data.loc[common_index]
        regime_labels = regime_labels.loc[common_index]
        
        return regime_data, regime_labels
    
    def _generate_regime_labels(self, price_data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Generate market regime labels based on price movements and volatility"""
        
        # Calculate returns and volatility
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        trend = returns.rolling(window=20).mean()
        
        # Calculate momentum
        momentum = price_data['close'].pct_change(10)
        
        # Define regime classes
        # 0: Bear market (negative trend, high volatility)
        # 1: Bull market (positive trend, low volatility)  
        # 2: Volatile/Uncertain (high volatility)
        # 3: Ranging/Consolidation (low volatility, no trend)
        
        regimes = pd.Series(index=price_data.index, dtype=int)
        
        # Fill NaN values first
        volatility = volatility.fillna(volatility.median())
        trend = trend.fillna(0)
        
        # Initialize with default regime
        regimes = regimes.fillna(0)  # Start with all bull
        
        # Classification logic with relaxed thresholds
        bull_mask = (trend > 0.0001) & (volatility < volatility.quantile(0.75))
        bear_mask = (trend < -0.0001) & (volatility > volatility.quantile(0.25))
        volatile_mask = volatility > volatility.quantile(0.75)
        consolidation_mask = (volatility < volatility.quantile(0.25)) & (abs(trend) < 0.0001)
        
        # Assign regimes (XGBoost expects 0-based classes)
        regimes.loc[bull_mask] = 0  # Bull
        regimes.loc[bear_mask] = 1  # Bear
        regimes.loc[volatile_mask] = 2  # Volatile
        regimes.loc[consolidation_mask] = 3  # Consolidation
        
        # Ensure we have at least 2 different regimes
        unique_regimes = regimes.unique()
        if len(unique_regimes) < 2:
            logger.warning(f"Only {len(unique_regimes)} regimes found, forcing diversity")
            # Force some samples into different regimes
            n_samples = len(regimes)
            if n_samples > 10:
                regimes.iloc[:n_samples//4] = 0  # First quarter bull
                regimes.iloc[n_samples//4:n_samples//2] = 1  # Second quarter bear
                regimes.iloc[n_samples//2:3*n_samples//4] = 2  # Third quarter volatile
                regimes.iloc[3*n_samples//4:] = 3  # Fourth quarter consolidation
        
        # Ensure integer type
        regimes = regimes.astype(int)
        
        logger.info("Generated regime labels", 
                   distribution=regimes.value_counts().to_dict(),
                   total_samples=len(regimes))
        
        return regimes
    
    def prepare_sentiment_data(self, text_data: List[str], sentiment_labels: List[str]) -> Tuple[List[str], np.ndarray]:
        """Prepare text data for sentiment analysis"""
        # Clean text data
        cleaned_texts = [self._clean_text(text) for text in text_data]
        
        # Encode sentiment labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        encoded_labels = np.array([label_map.get(label, 1) for label in sentiment_labels])
        
        return cleaned_texts, encoded_labels
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_pattern_labels(self, data: pd.DataFrame) -> pd.Series:
        """Generate pattern labels based on actual price movements"""
        patterns = []
        
        # Define core pattern types focused on the most detectable patterns
        core_pattern_types = [
            'uptrend',      # Clear upward movement
            'downtrend',    # Clear downward movement
            'sideways',     # Range-bound movement
            'volatile',     # High volatility periods
            'breakout_up',  # Upward breakout
            'breakout_down' # Downward breakout
        ]
        
        logger.info("Generating pattern labels", data_length=len(data))
        
        # Calculate price features
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        # Generate patterns based on price action
        for i in range(len(data)):
            if i < 20:  # Need some history
                patterns.append('sideways')
                continue
                
            # Get recent window of data
            window_size = min(20, i)
            recent_data = data.iloc[i-window_size:i+1]
                
            returns = recent_data['close'].pct_change()
            volatility = returns.std()
            trend = returns.mean()
            momentum = returns.rolling(5).mean().iloc[-1] if len(returns) >= 5 else 0
            
            # Price levels for pattern detection
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            
            # Clear pattern detection logic
            pattern_scores = {}
            
            # Simple but clear pattern detection based on price action
            # Calculate simple metrics
            price_change = (closes[-1] - closes[0]) / closes[0] if len(closes) > 0 and closes[0] != 0 else 0
            avg_range = np.mean(highs - lows) if len(highs) > 0 else 0
            price_std = np.std(closes) if len(closes) > 1 else 0
            
            # Detect clear patterns
            if price_change > 0.01 and trend > 0.001:  # 1% up move
                pattern_scores['uptrend'] = 0.8
            elif price_change < -0.01 and trend < -0.001:  # 1% down move
                pattern_scores['downtrend'] = 0.8
            elif abs(price_change) < 0.005 and price_std < avg_range * 0.5:  # Less than 0.5% move
                pattern_scores['sideways'] = 0.8
            elif price_std > avg_range * 1.5:  # High volatility
                pattern_scores['volatile'] = 0.7
            
            # Breakout patterns
            if len(closes) >= 10:
                recent_high = np.max(closes[-10:-1])
                recent_low = np.min(closes[-10:-1])
                if closes[-1] > recent_high * 1.005:  # Break above recent high
                    pattern_scores['breakout_up'] = 0.9
                elif closes[-1] < recent_low * 0.995:  # Break below recent low
                    pattern_scores['breakout_down'] = 0.9
            
            # Add some randomization to avoid always selecting the same patterns
            # Use weighted random selection instead of always taking the max
            if pattern_scores:
                # Normalize scores to probabilities
                total_score = sum(pattern_scores.values())
                if total_score > 0:
                    probabilities = {k: v/total_score for k, v in pattern_scores.items()}
                    pattern_names = list(probabilities.keys())
                    pattern_probs = list(probabilities.values())
                    selected_pattern = np.random.choice(pattern_names, p=pattern_probs)
                else:
                    selected_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
                patterns.append(selected_pattern)
            else:
                # Default to sideways if no clear pattern
                patterns.append('sideways')
        
        # Ensure we have exactly the right number of patterns
        if len(patterns) > len(data):
            patterns = patterns[:len(data)]
        elif len(patterns) < len(data):
            # Pad with most common pattern if still short
            most_common = pd.Series(patterns).mode()[0] if patterns else 'consolidation'
            while len(patterns) < len(data):
                patterns.append(most_common)
        
        # Now create series with matching length
        pattern_series = pd.Series(patterns, index=data.index)
        pattern_counts = pattern_series.value_counts()
        
        # Ensure minimum representation of each pattern for training
        min_samples = 50
        for pattern_type in core_pattern_types:
            if pattern_counts.get(pattern_type, 0) < min_samples:
                # Find indices to replace
                most_common = pattern_counts.index[0]
                n_to_replace = min(min_samples - pattern_counts.get(pattern_type, 0), 
                                 pattern_counts[most_common] // 4)
                if n_to_replace > 0:
                    indices = pattern_series[pattern_series == most_common].index[:n_to_replace]
                    for idx in indices:
                        pattern_series.loc[idx] = pattern_type
        
        # Log final distribution
        final_counts = pattern_series.value_counts()
        logger.info("Pattern distribution", counts=final_counts.to_dict())
        
        return pattern_series
    
    def _detect_head_shoulders(self, highs, lows):
        """Simple head and shoulders pattern detection"""
        if len(highs) < 5:
            return False
        # Look for: low, high, higher high, high, low pattern
        mid = len(highs) // 2
        if mid < 2 or mid >= len(highs) - 2:
            return False
        # Check for head higher than shoulders
        return (highs[mid] > highs[mid-1] * 1.01 and 
                highs[mid] > highs[mid+1] * 1.01 and
                highs[mid-1] > highs[mid-2] and
                highs[mid+1] > highs[mid+2])
    
    def _detect_inverse_head_shoulders(self, highs, lows):
        """Simple inverse head and shoulders pattern detection"""
        if len(lows) < 5:
            return False
        mid = len(lows) // 2
        if mid < 2 or mid >= len(lows) - 2:
            return False
        # Check for head lower than shoulders
        return (lows[mid] < lows[mid-1] * 0.99 and 
                lows[mid] < lows[mid+1] * 0.99 and
                lows[mid-1] < lows[mid-2] and
                lows[mid+1] < lows[mid+2])
    
    def _detect_triangle(self, highs, lows):
        """Simple triangle pattern detection"""
        if len(highs) < 3:
            return False
        # Converging highs and lows
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        # More lenient threshold for triangle detection
        high_range = max(highs) - min(highs)
        low_range = max(lows) - min(lows)
        converging = abs(high_slope + low_slope) < 0.02
        narrowing = high_range < highs[0] * 0.05 and low_range < lows[0] * 0.05
        return converging or narrowing
    
    def _detect_flag(self, closes, trend):
        """Simple flag pattern detection"""
        if len(closes) < 5:
            return False
        recent_trend = (closes[-1] - closes[-5]) / closes[-5]
        return abs(recent_trend) > 0.02 and abs(trend) > 0.001
    
    def _detect_wedge(self, highs, lows):
        """Simple wedge pattern detection"""
        if len(highs) < 3:
            return False
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        return (high_slope * low_slope) > 0  # Both slopes same direction
    
    def _detect_double_top(self, highs):
        """Simple double top detection"""
        if len(highs) < 5:
            return False
        peaks = []
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        return len(peaks) >= 2 and abs(peaks[-1][1] - peaks[-2][1]) < 0.01
    
    def _detect_double_bottom(self, lows):
        """Simple double bottom detection"""
        if len(lows) < 5:
            return False
        troughs = []
        for i in range(1, len(lows)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        return len(troughs) >= 2 and abs(troughs[-1][1] - troughs[-2][1]) < 0.01
    
    def _detect_channel(self, highs, lows):
        """Simple channel detection"""
        if len(highs) < 3:
            return False
        high_std = np.std(highs)
        low_std = np.std(lows)
        return high_std < 0.02 and low_std < 0.02
    
    def _detect_rectangle(self, highs, lows):
        """Simple rectangle pattern detection"""
        if len(highs) < 4:
            return False
        high_range = max(highs) - min(highs)
        low_range = max(lows) - min(lows)
        return high_range < 0.02 and low_range < 0.02
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Basic text cleaning
        import re
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


class ModelTrainingPipeline:
    """Main training pipeline for all models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.trained_models: Dict[str, BaseModel] = {}
        self.training_results: Dict[str, Dict[str, Any]] = {}
        self.text_data: Optional[Dict[str, Any]] = None
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
    def train_all_models(
        self,
        price_data: pd.DataFrame,
        text_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, BaseModel]:
        """Train all models in the pipeline"""
        
        logger.info("Starting model training pipeline")
        
        # Store text data for later use
        self.text_data = text_data
        
        # Prepare data splits
        train_data, val_data, test_data = self._create_data_splits(price_data)
        
        # Define training tasks
        training_tasks = []
        
        if self.config.train_lstm:
            training_tasks.append(('PriceLSTM', self._train_lstm, train_data, val_data))
        
        if self.config.train_cnn:
            training_tasks.append(('ChartPatternCNN', self._train_cnn, train_data, val_data))
        
        if self.config.train_xgboost:
            training_tasks.append(('MarketRegimeXGBoost', self._train_xgboost, train_data, val_data))
        
        if self.config.train_finbert and text_data:
            training_tasks.append(('FinBERT', self._train_finbert, text_data, None))
        
        # Train models
        if self.config.parallel_training and len(training_tasks) > 1:
            self._train_models_parallel(training_tasks)
        else:
            self._train_models_sequential(training_tasks)
        
        # Train ensemble if requested
        if self.config.train_ensemble and len(self.trained_models) > 1:
            ensemble = self._train_ensemble(train_data, val_data)
            self.trained_models['StackedEnsemble'] = ensemble
        
        # Validate all models
        self._validate_models(test_data)
        
        # Note: Model saving is handled by ModelPersistence in train_models.py
        # to avoid double-saving and ensure proper versioning/metadata
        
        logger.info("Model training pipeline completed",
                   models_trained=list(self.trained_models.keys()),
                   total_models=len(self.trained_models))
        
        return self.trained_models
    
    def _create_data_splits(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits"""
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Calculate split indices
        total_len = len(data)
        test_size = int(total_len * self.config.test_split)
        val_size = int(total_len * self.config.validation_split)
        train_size = total_len - test_size - val_size
        
        # Ensure minimum sizes for each split
        min_train_size = 2000
        min_val_size = 500
        min_test_size = 500
        
        if train_size < min_train_size or val_size < min_val_size or test_size < min_test_size:
            logger.warning(f"Data splits too small. Adjusting splits for {total_len} samples")
            # Use 70/20/10 split for small datasets
            train_size = int(total_len * 0.7)
            val_size = int(total_len * 0.2)
            test_size = total_len - train_size - val_size
        
        # Create splits (maintaining temporal order)
        train_data = data.iloc[:train_size].copy()
        val_data = data.iloc[train_size:train_size + val_size].copy()
        test_data = data.iloc[train_size + val_size:].copy()
        
        logger.info("Data splits created",
                   train_samples=len(train_data),
                   val_samples=len(val_data),
                   test_samples=len(test_data))
        
        return train_data, val_data, test_data
    
    def _train_models_parallel(self, training_tasks: List[Tuple]):
        """Train models in parallel"""
        
        logger.info("Training models in parallel", max_workers=self.config.max_workers)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit training tasks
            future_to_model = {}
            for model_name, train_func, train_data, val_data in training_tasks:
                future = executor.submit(train_func, train_data, val_data)
                future_to_model[future] = model_name
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model, results = future.result()
                    self.trained_models[model_name] = model
                    self.training_results[model_name] = results
                    logger.info(f"Model {model_name} training completed")
                except Exception as e:
                    logger.error(f"Model {model_name} training failed", error=str(e))
    
    def _train_models_sequential(self, training_tasks: List[Tuple]):
        """Train models sequentially"""
        
        logger.info("Training models sequentially")
        
        for model_name, train_func, train_data, val_data in training_tasks:
            try:
                logger.info(f"Starting training: {model_name}")
                model, results = train_func(train_data, val_data)
                self.trained_models[model_name] = model
                self.training_results[model_name] = results
                logger.info(f"Model {model_name} training completed")
            except Exception as e:
                import traceback
                logger.error(f"Model {model_name} training failed", 
                           error=str(e),
                           traceback=traceback.format_exc())
    
    def _train_lstm(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[PriceLSTM, Dict]:
        """Train LSTM model"""
        
        # Prepare data
        train_features, train_targets = self.preprocessor.prepare_price_data(train_data)
        val_features, val_targets = self.preprocessor.prepare_price_data(val_data)
        
        # Create config with improved settings
        config = PriceLSTMConfig(
            sequence_length=60,  # 60 time steps
            lstm_hidden_size=128,
            lstm_layers=2,
            lstm_dropout=0.2,  # Use lstm_dropout instead of dropout
            epochs=100,
            batch_size=32,  # Smaller batch size for better gradient updates
            learning_rate=0.0005,  # Lower learning rate for stability
            early_stopping_patience=self.config.early_stopping_patience,
            save_path=self.config.model_save_dir / "lstm",
            use_external_features=True,  # We're using UniversalFeatureGenerator
            add_lag_features=False,  # Disable duplicate lag features
            add_time_features=False,  # Disable duplicate time features
            add_rolling_features=False,  # Disable duplicate rolling features
            scaling_method="standard",  # Use standard scaling for better LSTM training
            target_scaling_method="standard",  # Scale targets for stable gradients
            weight_decay=0.0001,  # Add L2 regularization
            gradient_clip=0.5,  # Use gradient_clip instead of gradient_clip_val
            use_attention=True,  # Enable attention mechanism
            attention_heads=4
        )
        
        # Create and train model
        model = PriceLSTM(config)
        
        validation_data = (val_features, val_targets) if val_data is not None else None
        history = model.train(train_features, train_targets, validation_data=validation_data)
        
        return model, history
    
    def _train_cnn(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[ChartPatternCNN, Dict]:
        """Train CNN model"""
        
        # Prepare data
        train_charts, train_patterns = self.preprocessor.prepare_chart_data(train_data)
        val_charts, val_patterns = self.preprocessor.prepare_chart_data(val_data)
        
        # Create config
        config = ChartPatternConfig(
            chart_height=64,
            chart_width=128,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=self.config.early_stopping_patience,
            save_path=self.config.model_save_dir / "cnn"
        )
        
        # Create and train model
        model = ChartPatternCNN(config)
        
        validation_data = (val_charts, val_patterns) if val_data is not None else None
        history = model.train(train_charts, train_patterns, validation_data=validation_data)
        
        return model, history
    
    def _train_xgboost(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[MarketRegimeXGBoost, Dict]:
        """Train XGBoost model"""
        
        # Prepare data
        train_features, train_labels = self.preprocessor.prepare_regime_data(train_data)
        val_features, val_labels = self.preprocessor.prepare_regime_data(val_data)
        
        # Create config
        config = MarketRegimeConfig(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=50,
            save_path=self.config.model_save_dir / "xgboost"
        )
        
        # Create and train model
        model = MarketRegimeXGBoost(config)
        
        validation_data = (val_features, val_labels) if val_data is not None else None
        history = model.train(train_features, train_labels, validation_data=validation_data)
        
        return model, history
    
    def _train_finbert(self, text_data: Dict[str, Any], val_data: Optional[Dict]) -> Tuple[FinBERT, Dict]:
        """Train FinBERT model"""
        
        # Extract text and labels
        texts = text_data.get('texts', [])
        labels = text_data.get('labels', [])
        
        # Prepare data
        train_texts, train_labels = self.preprocessor.prepare_sentiment_data(texts, labels)
        
        # Create config
        config = FinBERTConfig(
            epochs=5,  # Small number for transformers
            batch_size=16,
            learning_rate=2e-5,
            save_path=self.config.model_save_dir / "finbert"
        )
        
        # Create and train model
        model = FinBERT(config)
        
        # Split data for validation
        split_idx = int(len(train_texts) * 0.8)
        validation_data = (train_texts[split_idx:], train_labels[split_idx:]) if len(train_texts) > 10 else None
        
        history = model.train(
            train_texts[:split_idx], 
            train_labels[:split_idx], 
            validation_data=validation_data
        )
        
        return model, history
    
    def _extract_text_features_for_timestamps(self, price_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract text features aligned with price data timestamps"""
        if not self.text_data or 'texts' not in self.text_data:
            return None
        
        # For now, create synthetic text features based on available text data
        # In a real implementation, you would:
        # 1. Match text data timestamps with price data timestamps
        # 2. Aggregate sentiment scores for each time period
        # 3. Create features like: avg_sentiment, text_volume, etc.
        
        # Create a DataFrame with same index as price data
        text_features = pd.DataFrame(index=price_data.index)
        
        # Calculate overall sentiment distribution from text data
        labels = self.text_data.get('labels', [])
        if labels:
            sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            for label in labels:
                if label in sentiment_counts:
                    sentiment_counts[label] += 1
            
            total = sum(sentiment_counts.values())
            if total > 0:
                # Create sentiment probability features
                text_features['text_positive_prob'] = sentiment_counts['positive'] / total
                text_features['text_neutral_prob'] = sentiment_counts['neutral'] / total
                text_features['text_negative_prob'] = sentiment_counts['negative'] / total
                text_features['text_volume'] = len(labels) / len(price_data)  # Normalized text volume
                
                # Add sentiment score (positive - negative)
                text_features['text_sentiment_score'] = (
                    text_features['text_positive_prob'] - text_features['text_negative_prob']
                )
            else:
                # Default neutral features
                text_features['text_positive_prob'] = 0.33
                text_features['text_neutral_prob'] = 0.34
                text_features['text_negative_prob'] = 0.33
                text_features['text_volume'] = 0.0
                text_features['text_sentiment_score'] = 0.0
        else:
            return None
            
        return text_features
    
    def _train_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> StackedEnsemble:
        """Train ensemble model using standardized features"""
        
        logger.info("Training ensemble model with standardized features")
        
        # Create config
        config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            save_path=self.config.model_save_dir / "ensemble",
            use_probabilities=False,  # Use predictions only for stability
            generate_disagreement_features=True,
            generate_confidence_features=False,  # Disable if no probabilities
            include_original_features=False  # Start simple
        )
        
        # Create ensemble
        ensemble = StackedEnsemble(config)
        
        # Add base models
        for model_name, model in self.trained_models.items():
            ensemble.add_model(model_name, model)
        
        # Prepare data with universal features for ensemble training
        # Generate features once and transform for each model type
        train_data_sorted = train_data.sort_values('timestamp').copy()
        val_data_sorted = val_data.sort_values('timestamp').copy()
        
        # Generate universal features
        train_features = self.preprocessor.feature_generator.generate_features(train_data_sorted, is_training=True)
        val_features = self.preprocessor.feature_generator.generate_features(val_data_sorted, is_training=False)
        
        # Create data dictionary with features for each model type
        train_model_data = {
            'PriceLSTM': self.preprocessor.feature_generator.transform_for_model(train_features, 'lstm'),
            'ChartPatternCNN': self.preprocessor.feature_generator.transform_for_model(train_features, 'cnn'),
            'MarketRegimeXGBoost': self.preprocessor.feature_generator.transform_for_model(train_features, 'xgboost')
        }
        
        val_model_data = {
            'PriceLSTM': self.preprocessor.feature_generator.transform_for_model(val_features, 'lstm'),
            'ChartPatternCNN': self.preprocessor.feature_generator.transform_for_model(val_features, 'cnn'),
            'MarketRegimeXGBoost': self.preprocessor.feature_generator.transform_for_model(val_features, 'xgboost')
        }
        
        # If FinBERT is available and we have text data, extract text features
        if 'FinBERT' in self.trained_models and self.text_data:
            # Extract text features aligned with price data
            train_text_features = self._extract_text_features_for_timestamps(train_data_sorted)
            val_text_features = self._extract_text_features_for_timestamps(val_data_sorted)
            
            if train_text_features is not None:
                # For FinBERT, we'll use the text features as a proxy
                # The ensemble will use these features to understand text sentiment impact
                train_model_data['FinBERT'] = train_text_features
                val_model_data['FinBERT'] = val_text_features
            else:
                logger.warning("Could not extract text features for FinBERT in ensemble")
                # Let the ensemble handle missing FinBERT data gracefully
                train_model_data['FinBERT'] = pd.DataFrame(index=train_features.index)
                val_model_data['FinBERT'] = pd.DataFrame(index=val_features.index)
        
        
        # Create simple synthetic labels for ensemble training (binary classification)
        # In practice, you'd use actual trading signals or regime classifications
        returns = train_data_sorted['close'].pct_change()
        train_labels = (returns > returns.median()).astype(int)
        
        val_returns = val_data_sorted['close'].pct_change()
        val_labels = (val_returns > val_returns.median()).astype(int)
        
        # Align labels with features
        common_train_idx = train_features.index.intersection(train_labels.index)
        train_labels = train_labels.loc[common_train_idx]
        
        common_val_idx = val_features.index.intersection(val_labels.index)
        val_labels = val_labels.loc[common_val_idx]
        
        # Train ensemble with model-specific data
        validation_data = (val_model_data, val_labels) if val_data is not None else None
        history = ensemble.train(train_model_data, train_labels, validation_data=validation_data)
        
        self.training_results['StackedEnsemble'] = history
        
        return ensemble
    
    def _validate_models(self, test_data: pd.DataFrame):
        """Validate all trained models"""
        
        logger.info("Validating trained models")
        
        for model_name, model in self.trained_models.items():
            try:
                # Check if we have enough data for time series models
                if model_name == 'PriceLSTM' and hasattr(model.config, 'sequence_length'):
                    min_samples_needed = model.config.sequence_length + model.config.forecast_horizon
                    if len(test_data) < min_samples_needed:
                        logger.warning(f"Skipping validation for {model_name} - insufficient test data",
                                     test_samples=len(test_data),
                                     min_needed=min_samples_needed)
                        continue
                
                # Prepare test data based on model type using universal features
                if model_name == 'PriceLSTM':
                    test_features, test_targets = self.preprocessor.prepare_price_data(test_data)
                    metrics = model.evaluate(test_features, test_targets)
                elif model_name == 'ChartPatternCNN':
                    test_charts, test_patterns = self.preprocessor.prepare_chart_data(test_data)
                    metrics = model.evaluate(test_charts, test_patterns)
                elif model_name == 'MarketRegimeXGBoost':
                    test_features, test_labels = self.preprocessor.prepare_regime_data(test_data)
                    # Now we have labels from _generate_regime_labels
                    if test_labels is not None:
                        metrics = model.evaluate(test_features, test_labels)
                    else:
                        logger.info(f"Skipping validation for {model_name} - no test labels available")
                        continue
                elif model_name == 'StackedEnsemble':
                    # For ensemble, prepare model-specific data
                    test_data_sorted = test_data.sort_values('timestamp').copy()
                    test_features = self.preprocessor.feature_generator.generate_features(test_data_sorted, is_training=False)
                    
                    test_model_data = {
                        'PriceLSTM': self.preprocessor.feature_generator.transform_for_model(test_features, 'lstm'),
                        'ChartPatternCNN': self.preprocessor.feature_generator.transform_for_model(test_features, 'cnn'),
                        'MarketRegimeXGBoost': self.preprocessor.feature_generator.transform_for_model(test_features, 'xgboost')
                    }
                    
                    # Add FinBERT text features if available
                    if 'FinBERT' in self.trained_models and self.text_data:
                        test_text_features = self._extract_text_features_for_timestamps(test_data_sorted)
                        if test_text_features is not None:
                            test_model_data['FinBERT'] = test_text_features
                        else:
                            test_model_data['FinBERT'] = pd.DataFrame(index=test_features.index)
                    
                    # Create test labels (binary classification)
                    returns = test_data_sorted['close'].pct_change()
                    test_labels = (returns > returns.median()).astype(int)
                    
                    # Align labels with features
                    common_idx = test_features.index.intersection(test_labels.index)
                    test_labels = test_labels.loc[common_idx]
                    
                    metrics = model.evaluate(test_model_data, test_labels)
                else:
                    continue  # Skip models that don't have test data
                
                # Store validation metrics
                if model_name not in self.training_results:
                    self.training_results[model_name] = {}
                self.training_results[model_name]['test_metrics'] = metrics
                
                logger.info(f"Model {model_name} validation completed", metrics=metrics)
                
            except Exception as e:
                logger.error(f"Model {model_name} validation failed", error=str(e))
    
    def _save_models(self):
        """Save all trained models"""
        
        logger.info("Saving trained models")
        
        for model_name, model in self.trained_models.items():
            try:
                model_path = self.config.model_save_dir / model_name
                model.save(model_path)
                logger.info(f"Model {model_name} saved to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model {model_name}", error=str(e))
        
        # Save training results
        results_path = self.config.model_save_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.training_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Training results saved", path=str(results_path))
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        
        summary = {
            'models_trained': list(self.trained_models.keys()),
            'training_config': {
                'parallel_training': self.config.parallel_training,
                'max_workers': self.config.max_workers,
                'random_seed': self.config.random_seed
            },
            'results': {}
        }
        
        for model_name, results in self.training_results.items():
            summary['results'][model_name] = {
                'training_completed': True,
                'test_metrics': results.get('test_metrics', {}),
                'best_score': results.get('best_score', None)
            }
        
        return summary