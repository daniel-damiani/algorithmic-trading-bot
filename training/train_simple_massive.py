#!/usr/bin/env python3
"""
Simple direct training with massive dataset - no complex pipeline.
"""

import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import structlog
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

_features = importlib.util.spec_from_file_location(
    "universal_features",
    PROJECT_ROOT / "src" / "features" / "universal_features.py",
)
_universal_features = importlib.util.module_from_spec(_features)
_features.loader.exec_module(_universal_features)
UniversalFeatureGenerator = _universal_features.UniversalFeatureGenerator

logger = structlog.get_logger()

def load_massive_data(data_path: str, n_symbols: int = 30) -> pd.DataFrame:
    """Load data from massive dataset."""
    
    logger.info(f"Loading massive dataset: {n_symbols} symbols")
    
    # Load the massive dataset
    df = pd.read_parquet(data_path)
    
    # Fix column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Normalize timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
        df.drop('date', axis=1, inplace=True)
    
    # Select top symbols by data completeness
    symbol_stats = df.groupby('symbol').agg({
        'close': ['count', 'std'],
        'volume': 'mean'
    }).round(4)
    
    # Flatten column names
    symbol_stats.columns = ['_'.join(col).strip() for col in symbol_stats.columns]
    
    # Score symbols by data quality
    symbol_stats['score'] = (
        symbol_stats['close_count'] * 0.5 +  # More data points
        symbol_stats['close_std'] * 100 +    # More volatility (price movement)
        np.log1p(symbol_stats['volume_mean']) * 0.1  # Decent volume
    )
    
    # Select top symbols
    top_symbols = symbol_stats.nlargest(n_symbols, 'score').index.tolist()
    df = df[df['symbol'].isin(top_symbols)]
    
    # Clean data
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    df = df[[col for col in required_columns if col in df.columns]]
    df = df.dropna().reset_index(drop=True)
    
    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"Final dataset: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features for better performance."""
    
    logger.info("Generating enhanced features...")
    
    # Use universal features but preserve symbol
    feature_gen = UniversalFeatureGenerator()
    features_df = feature_gen.generate_features(df)
    
    # Add symbol column back if missing
    if 'symbol' not in features_df.columns:
        features_df['symbol'] = df['symbol'].values
    
    # Add more sophisticated features
    for symbol in df['symbol'].unique():
        mask = features_df['symbol'] == symbol
        symbol_data = features_df[mask].copy()
        
        if len(symbol_data) > 50:
            # Enhanced technical indicators
            symbol_data['rsi_divergence'] = (
                symbol_data['rsi_14'].diff() * symbol_data['close'].pct_change()
            )
            
            # Volatility regime
            symbol_data['vol_regime'] = pd.cut(
                symbol_data['volatility_20'], 
                bins=5, 
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
            
            # Price acceleration
            symbol_data['price_accel'] = symbol_data['close'].pct_change().diff()
            
            # Volume-price trend
            symbol_data['vpt'] = (
                symbol_data['volume'] * symbol_data['close'].pct_change()
            ).cumsum()
            
            # Update main dataframe
            features_df.loc[mask, 'rsi_divergence'] = symbol_data['rsi_divergence']
            features_df.loc[mask, 'vol_regime'] = symbol_data['vol_regime']
            features_df.loc[mask, 'price_accel'] = symbol_data['price_accel']
            features_df.loc[mask, 'vpt'] = symbol_data['vpt']
    
    # Fill any remaining NaN values
    features_df = features_df.ffill().fillna(0)
    
    logger.info(f"Enhanced features created: {features_df.shape[1]} columns")
    return features_df

def create_better_targets(df: pd.DataFrame) -> np.ndarray:
    """Create better prediction targets."""
    
    targets = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        
        if len(symbol_data) > 10:
            # Future returns (5-day forward)
            future_returns = symbol_data['close'].pct_change(5).shift(-5)
            
            # Create more balanced classes
            conditions = [
                future_returns <= -0.03,    # Strong sell (-1)
                (future_returns > -0.03) & (future_returns <= -0.01),  # Sell (0)
                (future_returns > -0.01) & (future_returns <= 0.01),   # Hold (1)
                (future_returns > 0.01) & (future_returns <= 0.03),    # Buy (2)
                future_returns > 0.03       # Strong buy (3)
            ]
            
            symbol_targets = np.select(conditions, [0, 1, 2, 3, 4], default=2)
            targets.extend(symbol_targets)
        else:
            targets.extend([2] * len(symbol_data))  # Default to hold
    
    return np.array(targets[:len(df)])

def train_enhanced_model(X_train, X_val, y_train, y_val):
    """Train enhanced XGBoost model with scikit-learn interface."""
    
    logger.info("Training enhanced XGBoost model...")
    
    # Create XGBoost classifier with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=1000,        # More trees
        max_depth=8,              # Deeper trees
        learning_rate=0.05,       # Lower learning rate for stability
        subsample=0.8,            # Prevent overfitting
        colsample_bytree=0.8,     # Feature sampling
        min_child_weight=3,       # Prevent overfitting
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        random_state=42,         # Reproducibility
        n_jobs=-1,               # Use all cores
        early_stopping_rounds=50, # More patience
        eval_metric='mlogloss'   # Multi-class log loss
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict and evaluate
    val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='weighted')
    
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    logger.info(f"Validation F1 score: {f1:.4f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    logger.info(f"Top features by importance: {np.argsort(feature_importance)[-10:]}")
    
    return model, accuracy, f1

def main():
    parser = argparse.ArgumentParser(description="Simple massive dataset training")
    parser.add_argument("--data", type=str, default="data/massive/massive_training_data.parquet")
    parser.add_argument("--symbols", type=int, default=50)
    parser.add_argument("--target", type=float, default=0.65)
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("🚀 ENHANCED MASSIVE TRAINING SYSTEM")
    print("="*70)
    print(f"Target Accuracy: {args.target:.1%}")
    print("="*70)
    
    try:
        # Load massive data
        data = load_massive_data(args.data, args.symbols)
        print(f"📊 Dataset: {len(data):,} rows, {data['symbol'].nunique()} symbols")
        
        # Create enhanced features
        features = create_enhanced_features(data)
        
        # Create better targets
        targets = create_better_targets(features)
        
        # Remove non-feature columns
        feature_cols = [col for col in features.columns 
                       if col not in ['timestamp', 'symbol'] and not col.startswith('Unnamed')]
        X = features[feature_cols].values
        y = targets
        
        # Remove samples with NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📈 Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"📊 Target distribution: {np.bincount(y.astype(int))}")
        
        # Train-validation split (time-aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"🔄 Training: {len(X_train):,}, Validation: {len(X_val):,}")
        
        # Train enhanced model
        model, accuracy, f1 = train_enhanced_model(X_train, X_val, y_train, y_val)

        from training.model_io import save_massive_training_model

        model_dir = save_massive_training_model(
            model=model,
            feature_names=feature_cols,
            metrics={"accuracy": float(accuracy), "f1": float(f1)},
        )
        print(f"Model saved to: {model_dir}")
        print("Backtest: python backtest.py --start-date 2024-06-01 --end-date 2024-09-30")
        
        # Results
        print(f"\n{'='*60}")
        print("📊 RESULTS")
        print("="*60)
        print(f"Validation Accuracy: {accuracy:.4f} ({accuracy:.1%})")
        print(f"Validation F1 Score: {f1:.4f} ({f1:.1%})")
        
        if accuracy >= args.target:
            print("🎯 TARGET ACHIEVED!")
            status = "SUCCESS"
        elif accuracy >= 0.60:
            print("✅ GOOD PROGRESS")
            status = "PROGRESS"
        elif accuracy >= 0.50:
            print("⚠️ IMPROVEMENT NEEDED")
            status = "IMPROVEMENT"
        else:
            print("❌ POOR PERFORMANCE")
            status = "POOR"
        
        print("="*60)
        
        # Update experiment log
        experiment_text = f"""
### Experiment #enhanced_massive - Enhanced Massive Training ({datetime.now().strftime('%Y-%m-%d %H:%M')})
- **Data**: {len(data):,} rows, {data['symbol'].nunique()} symbols (massive dataset with quality filtering)
- **Features**: {X.shape[1]} enhanced features (technical + volume + momentum)
- **Target**: 5-class future returns prediction (5-day horizon)
- **Model**: Enhanced XGBoost (1000 trees, depth 8, regularized)
- **Results**:
  - Validation Accuracy: {accuracy:.4f} ({accuracy:.1%})
  - Validation F1 Score: {f1:.4f} ({f1:.1%})
  - Target: {args.target:.1%}
- **Status**: {'🎯 TARGET ACHIEVED!' if status == 'SUCCESS' else '✅ Good Progress' if status == 'PROGRESS' else '⚠️ Improvement Needed' if status == 'IMPROVEMENT' else '❌ Poor Performance'}
"""
        
        with open(PROJECT_ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write(experiment_text)
        
        return 0 if accuracy >= args.target else 1
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())