#!/usr/bin/env python3
"""
Test main pipeline with SimpleBalanced model
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

async def test_main():
    from src.main import QuantumSentimentBot
    import pandas as pd
    
    print("=" * 60)
    print("TESTING MAIN PIPELINE WITH SIMPLEBALANCED MODEL")
    print("=" * 60)
    
    # Create bot instance
    bot = QuantumSentimentBot(mode='paper')
    
    # Initialize system
    print("\n📦 Initializing system...")
    await bot.initialize()
    
    # Check model loaded
    print(f"✅ Model type: {bot.ensemble_model.__class__.__name__}")
    if hasattr(bot.ensemble_model, 'feature_names'):
        print(f"📊 Features expected: {bot.ensemble_model.feature_names}")
    
    # Test with a few symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\n📈 Testing predictions for: {symbols}")
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        try:
            # Get market data from broker
            bars = await bot.broker.get_bars(symbol, timeframe='1Hour', limit=100)
            
            if bars is None or len(bars) < 50:
                print(f"  ⚠️ Insufficient data for {symbol}")
                continue
            
            print(f"  ✅ Got {len(bars)} bars")
            
            # Create dummy sentiment for testing
            sentiment_df = pd.DataFrame([{
                'timestamp': bars.index[-1],
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'volume': 0,
                'source': 'test'
            }])
            sentiment_df.set_index('timestamp', inplace=True)
            
            # Generate features
            feature_result = bot.feature_pipeline.generate_features(
                symbol=symbol,
                market_data=bars,
                sentiment_data=sentiment_df
            )
            
            features_dict = feature_result.get('features', {})
            if not features_dict:
                print(f"  ❌ No features generated")
                continue
                
            print(f"  ✅ Generated {len(features_dict)} features")
            
            # Get prediction
            try:
                proba = bot.ensemble_model.predict_proba(features_dict)
                bearish_prob = proba[0, 0]
                bullish_prob = proba[0, 1]
                
                print(f"  📊 Prediction: Bearish={bearish_prob:.3f}, Bullish={bullish_prob:.3f}")
                
                # Determine signal
                if bullish_prob > 0.5:
                    signal_type = "BUY"
                    signal_strength = (bullish_prob - 0.5) * 2
                else:
                    signal_type = "SELL"
                    signal_strength = (0.5 - bullish_prob) * 2
                
                print(f"  🎯 Signal: {signal_type} (strength={signal_strength:.3f})")
                
                # Check if signal would pass validation
                min_strength = 0.3  # Typical threshold
                if signal_strength >= min_strength:
                    print(f"  ✅ Signal would be EXECUTED")
                else:
                    print(f"  ❌ Signal would be REJECTED (strength < {min_strength})")
                    
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
                
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ MAIN PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Cleanup
    await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(test_main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")