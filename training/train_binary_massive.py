#!/usr/bin/env python3
"""
Binary classification approach with massive dataset - predict up/down.
Simpler problem that might achieve higher accuracy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import structlog
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from src.features.universal_features import UniversalFeatureGenerator

logger = structlog.get_logger()

def load_massive_data(data_path: str, n_symbols: int = 40) -> pd.DataFrame:
    """Load high-quality symbols from massive dataset."""
    
    logger.info(f"Loading massive dataset: {n_symbols} best symbols")
    
    # Load the massive dataset
    df = pd.read_parquet(data_path)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df.drop('date', axis=1, inplace=True)
    
    # Quality scoring: prioritize liquid, volatile symbols
    symbol_stats = df.groupby('symbol').agg({
        'close': ['count', 'std', 'mean'],
        'volume': ['mean', 'std'],
        'high': 'mean',
        'low': 'mean'
    }).round(4)
    
    symbol_stats.columns = ['_'.join(col).strip() for col in symbol_stats.columns]
    
    # Advanced quality score
    symbol_stats['volatility_score'] = symbol_stats['close_std'] / symbol_stats['close_mean']
    symbol_stats['volume_consistency'] = symbol_stats['volume_mean'] / (symbol_stats['volume_std'] + 1)
    symbol_stats['data_completeness'] = symbol_stats['close_count']
    
    # Combined quality score
    symbol_stats['quality_score'] = (
        symbol_stats['data_completeness'] * 0.3 +        # Data availability
        symbol_stats['volatility_score'] * 1000 * 0.4 +  # Price volatility
        np.log1p(symbol_stats['volume_mean']) * 0.3       # Liquidity
    )
    
    # Select best symbols
    top_symbols = symbol_stats.nlargest(n_symbols, 'quality_score').index.tolist()
    df = df[df['symbol'].isin(top_symbols)]
    
    # Clean and prepare
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    df = df[[col for col in required_columns if col in df.columns]]
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"Quality dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced trading features."""
    
    logger.info("Creating advanced trading features...")
    
    # Start with universal features
    feature_gen = UniversalFeatureGenerator()
    features_df = feature_gen.generate_features(df)
    
    # Add symbol and timestamp back if missing
    if 'symbol' not in features_df.columns:
        features_df['symbol'] = df['symbol'].values
    if 'timestamp' not in features_df.columns:
        features_df['timestamp'] = df['timestamp'].values
    
    # Add advanced features per symbol
    for symbol in df['symbol'].unique():
        mask = features_df['symbol'] == symbol
        symbol_data = features_df[mask].copy().sort_values('timestamp')
        
        if len(symbol_data) > 100:
            # Price momentum features
            for period in [3, 5, 8, 13, 21, 34]:
                symbol_data[f'momentum_{period}'] = symbol_data['close'].pct_change(period)
                symbol_data[f'volatility_{period}'] = symbol_data['close'].pct_change().rolling(period).std()
            
            # Volume features
            symbol_data['volume_sma_10'] = symbol_data['volume'].rolling(10).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_10']
            symbol_data['price_volume_trend'] = (symbol_data['volume'] * symbol_data['close'].pct_change()).cumsum()
            
            # Support/Resistance levels
            symbol_data['resistance'] = symbol_data['high'].rolling(20).max()
            symbol_data['support'] = symbol_data['low'].rolling(20).min()
            symbol_data['distance_to_resistance'] = (symbol_data['close'] - symbol_data['resistance']) / symbol_data['close']
            symbol_data['distance_to_support'] = (symbol_data['close'] - symbol_data['support']) / symbol_data['close']
            
            # Trend strength
            symbol_data['trend_strength'] = symbol_data['close'].rolling(10).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) == 10 else 0
            )
            
            # Market regime indicators
            symbol_data['high_volatility_regime'] = (
                symbol_data['volatility_20'] > symbol_data['volatility_20'].rolling(50).quantile(0.8)
            ).astype(int)
            
            # Update main dataframe
            feature_cols = [col for col in symbol_data.columns if col not in ['timestamp', 'symbol']]
            for col in feature_cols:
                if col in symbol_data.columns:
                    features_df.loc[mask, col] = symbol_data[col].values
    
    # Forward fill and fill remaining NaN
    features_df = features_df.groupby('symbol').fillna(method='ffill').fillna(0)
    
    logger.info(f"Advanced features created: {features_df.shape[1]} total columns")
    return features_df

def create_binary_targets(df: pd.DataFrame, horizon: int = 3) -> np.ndarray:
    """Create binary up/down targets with specified horizon."""
    
    logger.info(f"Creating binary targets with {horizon}-day horizon")
    
    targets = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy().sort_values('timestamp')
        
        if len(symbol_data) > horizon:
            # Future returns
            future_returns = symbol_data['close'].pct_change(horizon).shift(-horizon)
            
            # Binary classification: 1 if price goes up, 0 if down
            # Use a small threshold to avoid noise
            threshold = 0.005  # 0.5% threshold
            binary_targets = (future_returns > threshold).astype(int)
            
            targets.extend(binary_targets.values)
        else:
            targets.extend([0] * len(symbol_data))
    
    targets = np.array(targets[:len(df)])
    
    # Log class distribution
    unique, counts = np.unique(targets[~np.isnan(targets)], return_counts=True)
    logger.info(f"Binary target distribution: {dict(zip(unique, counts))}")
    
    return targets

def train_binary_model(X_train, X_val, y_train, y_val):
    """Train optimized binary XGBoost classifier."""
    
    logger.info("Training binary XGBoost classifier...")
    
    # Optimized parameters for binary classification
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1500,        # More trees
        max_depth=6,              # Moderate depth
        learning_rate=0.03,       # Lower learning rate
        subsample=0.85,           # High subsample
        colsample_bytree=0.85,    # High feature sampling
        min_child_weight=5,       # Prevent overfitting
        reg_alpha=0.05,          # L1 regularization
        reg_lambda=0.1,          # L2 regularization
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
        eval_metric='logloss'
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Generate predictions
    val_pred = model.predict(X_val)
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred_proba)
    
    logger.info(f"Binary classification results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUC-ROC: {auc:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    logger.info(f"Top 10 feature indices: {top_features}")
    
    return model, accuracy, f1, auc

def main():
    parser = argparse.ArgumentParser(description="Binary classification with massive dataset")
    parser.add_argument("--data", type=str, default="data/massive/massive_training_data.parquet")
    parser.add_argument("--symbols", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon in days")
    parser.add_argument("--target", type=float, default=0.65)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("🎯 BINARY CLASSIFICATION TRADING SYSTEM")
    print("="*70)
    print(f"Target Accuracy: {args.target:.1%}")
    print(f"Prediction Horizon: {args.horizon} days")
    print("="*70)
    
    try:
        # Load high-quality data
        data = load_massive_data(args.data, args.symbols)
        print(f"📊 Dataset: {len(data):,} rows, {data['symbol'].nunique()} symbols")
        
        # Create advanced features
        features = create_advanced_features(data)
        
        # Create binary targets using original data
        targets = create_binary_targets(data, args.horizon)
        
        # Prepare feature matrix
        feature_cols = [col for col in features.columns 
                       if col not in ['timestamp', 'symbol'] and not col.startswith('Unnamed')]
        X = features[feature_cols].fillna(0).values
        y = targets
        
        # Remove invalid targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📈 Features: {X.shape[1]}, Valid samples: {X.shape[0]}")
        print(f"📊 Class balance: Up={np.sum(y==1)}, Down={np.sum(y==0)}")
        
        # Time-aware split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"🔄 Training: {len(X_train):,}, Validation: {len(X_val):,}")
        
        # Train binary model
        model, accuracy, f1, auc = train_binary_model(X_train, X_val, y_train, y_val)
        
        # Results
        print(f"\n{'='*60}")
        print("📊 BINARY CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"F1 Score: {f1:.4f} ({f1:.1%})")
        print(f"AUC-ROC: {auc:.4f} ({auc:.1%})")
        
        if accuracy >= args.target:
            print("🎯 TARGET ACHIEVED!")
            status = "SUCCESS"
        elif accuracy >= 0.60:
            print("✅ EXCELLENT PROGRESS")
            status = "EXCELLENT"
        elif accuracy >= 0.55:
            print("✅ GOOD PROGRESS")
            status = "GOOD"
        else:
            print("⚠️ NEEDS IMPROVEMENT")
            status = "IMPROVEMENT"
        
        print("="*60)
        
        # Update experiment log
        experiment_text = f"""
### Experiment #binary_massive - Binary Classification ({datetime.now().strftime('%Y-%m-%d %H:%M')})
- **Data**: {len(data):,} rows, {data['symbol'].nunique()} symbols (quality-filtered massive dataset)
- **Features**: {X.shape[1]} advanced trading features
- **Target**: Binary up/down prediction ({args.horizon}-day horizon)  
- **Model**: Optimized XGBoost Binary Classifier (1500 trees, depth 6)
- **Results**:
  - Accuracy: {accuracy:.4f} ({accuracy:.1%})
  - F1 Score: {f1:.4f} ({f1:.1%})
  - AUC-ROC: {auc:.4f} ({auc:.1%})
  - Target: {args.target:.1%}
- **Status**: {'🎯 TARGET ACHIEVED!' if status == 'SUCCESS' else '🌟 Excellent Progress' if status == 'EXCELLENT' else '✅ Good Progress' if status == 'GOOD' else '⚠️ Needs Improvement'}
"""
        
        with open("EXPERIMENT_LOG.md", "a") as f:
            f.write(experiment_text)
        
        return 0 if accuracy >= args.target else 1
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())