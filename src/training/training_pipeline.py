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
        # and clip extreme values to reduce noise
        targets = np.log(data['close'] / data['close'].shift(1)).shift(-1)
        targets = targets.clip(lower=-0.05, upper=0.05)  # Clip to ±5% moves
        
        # Align features and targets by matching indices
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index].dropna()
        
        # Final alignment after dropping NaN targets
        features = features.loc[targets.index]
        
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
        
        # XGBoost model will auto-generate regime labels
        return regime_data, None
    
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
        """Generate simple pattern labels for training"""
        # This is a simplified approach - in practice, patterns would be labeled by experts
        # or detected using more sophisticated algorithms
        patterns = []
        
        # For CNN training, we'll use a simpler approach that ensures diversity
        # by using percentile-based categorization
        
        logger.info("Generating pattern labels", data_length=len(data))
        
        # Pre-calculate all features for the entire dataset
        all_volatilities = []
        all_trends = []
        all_rsis = []
        
        for i in range(len(data)):
            if i < 20:
                patterns.append('no_pattern')
                continue
                
            recent_data = data.iloc[i-20:i]
            returns = recent_data['close'].pct_change()
            volatility = returns.std()
            trend = returns.mean()
            rsi = self._calculate_rsi(recent_data['close'])
            
            all_volatilities.append(volatility)
            all_trends.append(trend)
            all_rsis.append(rsi.iloc[-1])
        
        # Calculate percentiles for better distribution
        volatility_percentiles = np.percentile(all_volatilities, [20, 40, 60, 80])
        trend_percentiles = np.percentile(all_trends, [25, 50, 75])
        rsi_percentiles = np.percentile(all_rsis, [30, 70])
        
        logger.info(f"Feature percentiles - volatility: {volatility_percentiles}, "
                   f"trend: {trend_percentiles}, rsi: {rsi_percentiles}")
        
        # Now assign patterns based on percentiles
        idx = 0
        for i in range(len(data)):
            if i < 20:
                continue  # Already added 'no_pattern'
                
            volatility = all_volatilities[idx]
            trend = all_trends[idx]
            rsi_val = all_rsis[idx]
            idx += 1
            
            # Use a decision tree approach for pattern assignment
            if rsi_val < rsi_percentiles[0]:
                patterns.append('oversold')
            elif rsi_val > rsi_percentiles[1]:
                patterns.append('overbought')
            elif volatility > volatility_percentiles[3]:  # Top 20% volatility
                if trend > trend_percentiles[2]:
                    patterns.append('volatile_bull')
                elif trend < trend_percentiles[0]:
                    patterns.append('volatile_bear')
                else:
                    patterns.append('high_volatility')
            elif volatility < volatility_percentiles[0]:  # Bottom 20% volatility
                patterns.append('sideways')
            elif trend > trend_percentiles[2]:  # Top 25% trend
                patterns.append('bull_flag')
            elif trend < trend_percentiles[0]:  # Bottom 25% trend
                patterns.append('bear_flag')
            elif volatility > volatility_percentiles[2]:  # 60-80% volatility
                if trend > trend_percentiles[1]:
                    patterns.append('rising')
                else:
                    patterns.append('falling')
            else:
                # Distribute remaining into neutral categories
                if i % 3 == 0:
                    patterns.append('neutral')
                elif i % 3 == 1:
                    patterns.append('consolidation')
                else:
                    patterns.append('ranging')
        
        # Log pattern distribution
        pattern_counts = pd.Series(patterns).value_counts()
        logger.info("Pattern distribution", counts=pattern_counts.to_dict())
        
        return pd.Series(patterns, index=data.index)
    
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
        
        # Create config
        config = PriceLSTMConfig(
            sequence_length=60,  # 60 time steps
            lstm_hidden_size=128,
            epochs=100,
            batch_size=64,
            learning_rate=0.001,
            early_stopping_patience=self.config.early_stopping_patience,
            save_path=self.config.model_save_dir / "lstm"
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
    
    def _train_ensemble(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> StackedEnsemble:
        """Train ensemble model using standardized features"""
        
        logger.info("Training ensemble model with standardized features")
        
        # Create config
        config = StackedEnsembleConfig(
            meta_learner_type="xgboost",
            save_path=self.config.model_save_dir / "ensemble"
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
                    # Skip evaluation if no labels (XGBoost auto-generates labels)
                    if test_labels is None:
                        logger.info(f"Skipping validation for {model_name} - no test labels available")
                        continue
                    metrics = model.evaluate(test_features, test_labels)
                elif model_name == 'StackedEnsemble':
                    # For ensemble, prepare model-specific data
                    test_data_sorted = test_data.sort_values('timestamp').copy()
                    test_features = self.preprocessor.feature_generator.generate_features(test_data_sorted, is_training=False)
                    
                    test_model_data = {
                        'PriceLSTM': self.preprocessor.feature_generator.transform_for_model(test_features, 'lstm'),
                        'ChartPatternCNN': self.preprocessor.feature_generator.transform_for_model(test_features, 'cnn'),
                        'MarketRegimeXGBoost': self.preprocessor.feature_generator.transform_for_model(test_features, 'xgboost')
                    }
                    
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