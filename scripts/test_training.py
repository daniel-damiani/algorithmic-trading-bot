#!/usr/bin/env python3
"""
Test Training Script

Tests the training pipeline with adaptive configurations.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import structlog

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import AlpacaClient
from src.training import ModelTrainingPipeline, TrainingConfig
from src.configuration import load_config

# Configure logging with both print and structlog
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def test_training():
    """Test the training pipeline with current data"""
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}, continuing with test")
    
    # Initialize Alpaca client
    import os
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_API_SECRET'):
        logger.error("Missing Alpaca API credentials!")
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file")
        logger.error("You can get these from: https://alpaca.markets/")
        return
    
    try:
        alpaca = AlpacaClient()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return
    
    # Get a reasonable amount of data for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    
    print("Testing training pipeline")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    logger.info("Testing training pipeline")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get data for a single symbol first
    symbol = "AAPL"
    print(f"Fetching data for {symbol}")
    logger.info(f"Fetching data for {symbol}")
    
    try:
        # Get daily data
        daily_data = alpaca.get_bars(
            symbol=symbol,
            timeframe="Day",
            start=start_date,
            end=end_date
        )
        
        if daily_data is None or daily_data.empty:
            logger.error("No data retrieved")
            return
        
        logger.info(f"Retrieved {len(daily_data)} daily bars")
        
        # Check data quality
        logger.info("Data sample:")
        logger.info(f"First row: {daily_data.iloc[0].to_dict()}")
        logger.info(f"Last row: {daily_data.iloc[-1].to_dict()}")
        logger.info(f"Columns: {daily_data.columns.tolist()}")
        
        # Create training config with adaptive settings
        training_config = TrainingConfig(
            train_lstm=True,
            train_cnn=True,
            train_xgboost=True,
            train_finbert=False,  # Skip for quick test
            train_ensemble=True,
            parallel_training=False,  # Sequential for debugging
            early_stopping_patience=10
        )
        
        # Initialize training pipeline
        pipeline = ModelTrainingPipeline(training_config)
        
        # Train models
        logger.info("Starting model training")
        trained_models = pipeline.train_all_models(
            price_data=daily_data,
            text_data=None  # No text data for this test
        )
        
        logger.info(f"Training completed. Models trained: {list(trained_models.keys())}")
        
        # Get training summary
        summary = pipeline.get_training_summary()
        logger.info("Training summary:", summary=summary)
        
    except Exception as e:
        logger.error("Training failed", error=str(e), exc_info=True)
        raise


def check_data_availability():
    """Check how much data is available"""
    
    import os
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_API_SECRET'):
        logger.error("Missing Alpaca API credentials!")
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file")
        return
    
    try:
        alpaca = AlpacaClient()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return
        
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    
    print("Checking data availability for symbols:")
    logger.info("Checking data availability for symbols")
    
    for symbol in symbols:
        try:
            # Check different timeframes
            for timeframe in ["Day", "Hour"]:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)  # 1 year back
                
                data = alpaca.get_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                
                if data is not None and not data.empty:
                    print(f"✅ {symbol} - {timeframe}: {len(data)} bars available")
                    logger.info(f"{symbol} - {timeframe}: {len(data)} bars available")
                else:
                    print(f"❌ {symbol} - {timeframe}: No data available")
                    logger.warning(f"{symbol} - {timeframe}: No data available")
                    
        except Exception as e:
            print(f"❌ Error checking {symbol}: {e}")
            logger.error(f"Error checking {symbol}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--check-data", action="store_true", help="Check data availability")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal data")
    
    args = parser.parse_args()
    
    if args.check_data:
        check_data_availability()
    else:
        test_training()