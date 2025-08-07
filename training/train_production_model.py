#!/usr/bin/env python3
"""
Train a production-grade XGBoost model using the current feature pipeline.

This script:
1. Loads historical market data
2. Generates features using the SAME pipeline as trading/backtesting
3. Creates labels for supervised learning
4. Trains an XGBoost classifier
5. Saves the model in the correct format
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from src.configuration import load_config
from src.database.database import DatabaseManager
from src.features.feature_pipeline import FeaturePipeline, FeatureConfig
from src.models.xgboost.market_regime_xgboost import MarketRegimeXGBoost, MarketRegimeConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_labels(df: pd.DataFrame, lookahead_hours: int = 24) -> pd.Series:
    """
    Create binary classification labels based on future price movement.
    
    1 = Bullish (price goes up)
    0 = Bearish (price goes down or sideways)
    """
    # Calculate future returns
    future_returns = df['close'].pct_change(lookahead_hours).shift(-lookahead_hours)
    
    # Create binary labels
    # Threshold for considering a move significant (0.5%)
    threshold = 0.005
    labels = (future_returns > threshold).astype(int)
    
    return labels


def generate_features_for_symbol(
    symbol: str,
    data_path: Path,
    feature_pipeline: FeaturePipeline,
    min_bars: int = 100
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate features for a single symbol using the feature pipeline"""
    
    # Find data file
    symbol_dir = data_path / symbol
    if not symbol_dir.exists():
        logger.warning(f"No data directory for {symbol}")
        return None, None
    
    # Load market data
    csv_files = list(symbol_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files for {symbol}")
        return None, None
    
    # Use the first CSV file found
    market_data = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
    
    if len(market_data) < min_bars:
        logger.warning(f"Insufficient data for {symbol}: {len(market_data)} bars")
        return None, None
    
    all_features = []
    all_labels = []
    
    # Generate features using rolling windows
    window_size = 100  # Need at least 100 bars for indicators
    step_size = 1  # Move forward 1 hour at a time
    
    logger.info(f"Processing {symbol} with {len(market_data)} bars...")
    
    for i in tqdm(range(window_size, len(market_data) - 24, step_size), desc=symbol, leave=False):
        # Get window of data
        window_data = market_data.iloc[i-window_size:i].copy()
        
        # Create dummy sentiment data (neutral)
        sentiment_df = pd.DataFrame([{
            'timestamp': window_data.index[-1],
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'volume': 0,
            'source': 'dummy'
        }])
        sentiment_df.set_index('timestamp', inplace=True)
        
        try:
            # Generate features using the pipeline
            feature_result = feature_pipeline.generate_features(
                symbol=symbol,
                market_data=window_data,
                sentiment_data=sentiment_df
            )
            
            features_dict = feature_result.get('features', {})
            
            if features_dict:
                # Add raw price features that might be needed
                features_dict['close'] = window_data.iloc[-1]['close']
                features_dict['open'] = window_data.iloc[-1]['open']
                features_dict['high'] = window_data.iloc[-1]['high']
                features_dict['low'] = window_data.iloc[-1]['low']
                features_dict['volume'] = window_data.iloc[-1]['volume']
                
                # Add symbol as categorical feature
                features_dict['symbol'] = symbol
                
                # Create label for this point
                label = 1 if i + 24 < len(market_data) and \
                           market_data.iloc[i + 24]['close'] > market_data.iloc[i]['close'] * 1.005 else 0
                
                all_features.append(features_dict)
                all_labels.append(label)
                
        except Exception as e:
            logger.debug(f"Feature generation error at index {i}: {e}")
            continue
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        labels_series = pd.Series(all_labels)
        
        logger.info(f"Generated {len(features_df)} samples for {symbol}")
        logger.info(f"Label distribution: {labels_series.value_counts().to_dict()}")
        
        return features_df, labels_series
    else:
        logger.warning(f"No features generated for {symbol}")
        return None, None


def prepare_training_data(data_path: Path, config) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare all training data"""
    
    # Initialize database manager with database URL
    db_url = getattr(config.database, 'url', 'sqlite:///data/quantum_sentiment.db')
    db_manager = DatabaseManager(db_url)
    
    # Create feature config (same as in main.py)
    feature_config = FeatureConfig(
        enable_technical=True,
        enable_sentiment=True,
        enable_fundamental=True,
        enable_market_structure=True
    )
    
    # Initialize feature pipeline (same as used in trading)
    feature_pipeline = FeaturePipeline(feature_config, db_manager)
    
    all_features = []
    all_labels = []
    
    # Process each symbol
    for symbol_dir in data_path.iterdir():
        if symbol_dir.is_dir():
            symbol = symbol_dir.name
            
            features, labels = generate_features_for_symbol(
                symbol, data_path, feature_pipeline
            )
            
            if features is not None and labels is not None:
                all_features.append(features)
                all_labels.append(labels)
    
    if all_features:
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Total features: {X.shape[1]}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        return X, y
    else:
        raise ValueError("No training data generated!")


def train_xgboost_model(X: pd.DataFrame, y: pd.Series, config) -> xgb.XGBClassifier:
    """Train XGBoost model with the exact feature set"""
    
    # Handle categorical features
    if 'symbol' in X.columns:
        # Convert symbol to numeric code
        X['symbol_code'] = pd.Categorical(X['symbol']).codes
        X = X.drop('symbol', axis=1)
    
    # Remove any non-numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)
    
    logger.info(f"Training with {X.shape[1]} features")
    logger.info(f"Feature names sample: {list(X.columns[:10])}")
    
    # Split data (time-based split for financial data)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create XGBoost model with comprehensive parameters
    model = xgb.XGBClassifier(
        n_estimators=2000,  # More trees for better performance
        max_depth=8,        # Deeper trees
        learning_rate=0.05, # Lower learning rate for better generalization
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # Faster training
        early_stopping_rounds=50,
        verbosity=1
    )
    
    # Train with early stopping
    logger.info("Training XGBoost model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bearish', 'Bullish']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Save scaler with the model
    model.scaler = scaler
    model.feature_names = list(X.columns)
    model.n_features = X.shape[1]
    
    return model, scaler, list(X.columns)


def save_model(model, scaler, feature_names, output_dir: Path):
    """Save model in the format expected by the trading system"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / "MarketRegimeXGBoost" / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    model_path = model_dir / "model.xgb"
    model.save_model(str(model_path))
    logger.info(f"Saved XGBoost model to {model_path}")
    
    # Save scaler
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    features_path = model_dir / "features.json"
    with open(features_path, 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'model_type': 'XGBClassifier',
            'n_classes': 2,
            'classes': [0, 1],
            'class_names': ['bearish', 'bullish']
        }, f, indent=2)
    logger.info(f"Saved feature info to {features_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'MarketRegimeXGBoost',
        'version': '3.0.0',
        'timestamp': timestamp,
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'n_features': len(feature_names),
        'training_samples': model.n_features_in_,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Create a dummy config file for compatibility
    config_path = model_dir / "config.json"
    config_data = {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'learning_rate': model.learning_rate,
        'model_type': 'regime_classification',
        'n_classes': 2
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Model saved to {model_dir}")
    return model_dir


def main():
    """Main training function"""
    
    print("=" * 80)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Set paths
    data_path = Path("data/training/production")
    output_dir = Path("models")
    
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist. Run download script first!")
    
    # Step 1: Prepare training data
    print("\n📊 Preparing training data...")
    X, y = prepare_training_data(data_path, config)
    
    # Step 2: Train model
    print("\n🧠 Training XGBoost model...")
    model, scaler, feature_names = train_xgboost_model(X, y, config)
    
    # Step 3: Save model
    print("\n💾 Saving model...")
    model_dir = save_model(model, scaler, feature_names, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print(f"📁 Model saved to: {model_dir}")
    print("=" * 80)
    print("\n🚀 To use this model:")
    print("1. The model is automatically saved in the correct format")
    print("2. It will be loaded next time you run the trading bot")
    print("3. Test it with: python backtest.py --start-date 2024-01-01 --end-date 2024-02-01")


if __name__ == "__main__":
    main()