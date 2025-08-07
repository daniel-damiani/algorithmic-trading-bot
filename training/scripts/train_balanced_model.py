#!/usr/bin/env python3
"""
Quick balanced model training using Alpaca historical data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_simple_features(df):
    """Create simple technical features that work"""
    features = pd.DataFrame(index=df.index)
    
    # Price returns
    features['return_1h'] = df['close'].pct_change(1)
    features['return_6h'] = df['close'].pct_change(6)
    features['return_24h'] = df['close'].pct_change(24)
    
    # Simple moving averages
    features['sma_ratio_10'] = df['close'] / df['close'].rolling(10).mean()
    features['sma_ratio_50'] = df['close'] / df['close'].rolling(50).mean()
    
    # Volatility
    features['volatility'] = df['close'].pct_change().rolling(24).std()
    
    # Volume
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    return features

def main():
    print("=" * 60)
    print("BALANCED MODEL TRAINING")
    print("=" * 60)
    
    # Load Alpaca historical data
    data_files = list(Path('data/historical').rglob('*Hour*.csv'))
    print(f"Found {len(data_files)} data files")
    
    all_X = []
    all_y = []
    
    for file in data_files[:5]:  # Use first 5 symbols
        print(f"Processing {file.parent.name}...")
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        
        if len(df) < 100:
            continue
            
        # Generate simple features
        features = prepare_simple_features(df)
        
        # Create balanced labels (predict next hour)
        returns = df['close'].pct_change(1).shift(-1)
        
        # Use ZERO threshold for balanced classes
        labels = (returns > 0).astype(int)
        
        # Drop NaN
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) > 0:
            all_X.append(features)
            all_y.append(labels)
    
    # Combine all data
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Features: {list(X.columns)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    print(f"Bullish ratio: {y.mean():.2%}")
    
    # Fill any remaining NaN
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train balanced XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=1.0,  # Balanced
        random_state=42,
        eval_metric='logloss'
    )
    
    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nTrain accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Test predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bearish', 'Bullish']))
    
    print("\nSample predictions (first 10):")
    for i in range(min(10, len(y_proba))):
        print(f"  Bearish: {y_proba[i,0]:.3f}, Bullish: {y_proba[i,1]:.3f}")
    
    # Save model
    model_dir = Path('models/SimpleBalanced')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(model_dir / 'model.xgb'))
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    
    with open(model_dir / 'features.json', 'w') as f:
        json.dump({
            'features': list(X.columns),
            'n_features': len(X.columns)
        }, f, indent=2)
    
    print(f"\n✅ Model saved to {model_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()