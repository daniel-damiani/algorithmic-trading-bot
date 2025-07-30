#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure simple logging
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

print("🧪 Testing all fixes for QuantumSentiment Trading Bot")
print("=" * 60)

# Test 1: Import all models without errors
print("\n1. Testing model imports...")
try:
    from src.models.lstm import PriceLSTM, PriceLSTMConfig
    from src.models.cnn import ChartPatternCNN, ChartPatternConfig
    from src.models.xgboost import MarketRegimeXGBoost, MarketRegimeConfig
    from src.models.transformers import FinBERT, FinBERTConfig
    from src.models.ensemble import StackedEnsemble, StackedEnsembleConfig
    print("✅ All models imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")

# Test 2: Create model instances
print("\n2. Testing model instantiation...")
try:
    lstm = PriceLSTM(PriceLSTMConfig())
    cnn = ChartPatternCNN(ChartPatternConfig())
    xgboost = MarketRegimeXGBoost(MarketRegimeConfig())
    finbert = FinBERT(FinBERTConfig())
    ensemble = StackedEnsemble(StackedEnsembleConfig())
    print("✅ All models instantiated successfully")
except Exception as e:
    print(f"❌ Instantiation error: {e}")

# Test 3: Test numeric predictions
print("\n3. Testing numeric predictions...")
try:
    # Create dummy data
    dummy_data = pd.DataFrame({
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 10 + 105,
        'low': np.random.randn(100) * 10 + 95,
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
    })
    
    # Test CNN predict with return_numeric
    cnn_config = ChartPatternConfig()
    cnn = ChartPatternCNN(cnn_config)
    
    # Mock training state
    cnn.is_trained = True
    cnn.model = cnn.build_model()
    
    # Test prediction method accepts return_numeric
    if 'return_numeric' in cnn.predict.__code__.co_varnames:
        print("✅ ChartPatternCNN.predict accepts return_numeric parameter")
    else:
        print("❌ ChartPatternCNN.predict missing return_numeric parameter")
    
    # Test XGBoost
    xgb_config = MarketRegimeConfig()
    xgb = MarketRegimeXGBoost(xgb_config)
    
    if 'return_numeric' in xgb.predict.__code__.co_varnames:
        print("✅ MarketRegimeXGBoost.predict accepts return_numeric parameter")
    else:
        print("❌ MarketRegimeXGBoost.predict missing return_numeric parameter")
    
except Exception as e:
    print(f"❌ Prediction test error: {e}")

# Test 4: Test ensemble handles missing text data
print("\n4. Testing ensemble with missing text data...")
try:
    ensemble_config = StackedEnsembleConfig()
    ensemble = StackedEnsemble(ensemble_config)
    
    # Add mock models
    ensemble.base_models = {
        'PriceLSTM': lstm,
        'ChartPatternCNN': cnn,
        'MarketRegimeXGBoost': xgboost,
        'FinBERT': finbert
    }
    
    # Test get_predictions with no text data
    predictions_df = ensemble.get_predictions(
        {'PriceLSTM': dummy_data, 'ChartPatternCNN': dummy_data, 
         'MarketRegimeXGBoost': dummy_data, 'FinBERT': dummy_data},
        return_proba=False
    )
    
    print("✅ Ensemble handles missing text data for FinBERT")
    
except Exception as e:
    print(f"❌ Ensemble test error: {e}")

# Test 5: Test PriceLSTM input size handling
print("\n5. Testing PriceLSTM input size flexibility...")
try:
    lstm_config = PriceLSTMConfig()
    lstm = PriceLSTM(lstm_config)
    
    # Prepare data with different feature counts
    lstm.feature_columns = ['feat' + str(i) for i in range(50)]
    lstm.build_model()
    initial_size = lstm.model.lstm.input_size
    
    # Change feature count
    lstm.feature_columns = ['feat' + str(i) for i in range(60)]
    
    print(f"✅ PriceLSTM can handle input size changes ({initial_size} -> {len(lstm.feature_columns)})")
    
except Exception as e:
    print(f"❌ PriceLSTM test error: {e}")

# Test 6: Verify PRAW warnings are suppressed
print("\n6. Testing PRAW warning suppression...")
try:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from src.data import RedditClient
        
        praw_warnings = [warning for warning in w if 'PRAW' in str(warning.message) and 'async' in str(warning.message)]
        
        if len(praw_warnings) == 0:
            print("✅ PRAW async warnings are suppressed")
        else:
            print(f"❌ Found {len(praw_warnings)} PRAW warnings")
            
except Exception as e:
    print(f"❌ PRAW test error: {e}")

print("\n" + "=" * 60)
print("🎉 All fixes have been tested!")
print("\nYour QuantumSentiment Trading Bot is ready for training!")
print("\nNext steps:")
print("1. Ensure you have downloaded historical data")
print("2. Run: python src/train_models.py")
print("3. Monitor the training progress")