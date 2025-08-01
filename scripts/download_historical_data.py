#!/usr/bin/env python3
"""
Download Historical Data Script

Downloads comprehensive historical data for all symbols in the watchlist.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import structlog
from typing import List, Dict, Optional
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import AlpacaClient
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


def download_historical_data(
    symbols: List[str],
    start_date: str = "2020-01-01",  # 4+ years of data
    end_date: Optional[str] = None,
    timeframes: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for specified symbols
    
    Args:
        symbols: List of stock symbols
        start_date: Start date for data download
        end_date: End date (defaults to today)
        timeframes: List of timeframes to download
        
    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if timeframes is None:
        timeframes = ["1Day", "1Hour", "15Min", "5Min"]  # All timeframes now working
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Starting historical data download",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframes=timeframes)
    
    # Initialize Alpaca client
    try:
        alpaca = AlpacaClient()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        raise
    
    all_data = {}
    
    for symbol in symbols:
        logger.info(f"Downloading data for {symbol}")
        symbol_data = {}
        
        for timeframe in timeframes:
            try:
                logger.info(f"  Fetching {timeframe} data...")
                
                # Convert timeframe to Alpaca format
                alpaca_timeframe = timeframe
                if timeframe == "1Day":
                    alpaca_timeframe = "Day"
                elif timeframe == "1Hour":
                    alpaca_timeframe = "Hour"
                elif timeframe == "15Min":
                    alpaca_timeframe = "15Min"
                elif timeframe == "5Min":
                    alpaca_timeframe = "5Min"
                
                # Fetch historical bars
                from datetime import datetime
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                bars = alpaca.get_bars(
                    symbol=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_dt,
                    end=end_dt
                )
                
                if bars is not None and not bars.empty:
                    logger.info(f"  Downloaded {len(bars)} bars for {timeframe}")
                    symbol_data[timeframe] = bars
                    
                    # Save to CSV for backup
                    output_dir = project_root / "data" / "historical" / symbol
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
                    output_path = output_dir / filename
                    bars.to_csv(output_path)
                    logger.info(f"  Saved to {output_path}")
                else:
                    logger.warning(f"  No data received for {timeframe}")
                
                # Rate limiting
                time.sleep(0.2)  # Small delay between requests
                
            except Exception as e:
                logger.error(f"  Error downloading {timeframe} data: {e}")
                continue
        
        all_data[symbol] = symbol_data
        
        # Longer delay between symbols
        time.sleep(1)
    
    return all_data


def download_supplementary_data(symbols: List[str]) -> None:
    """Download supplementary data like company info, fundamentals"""
    logger.info("Downloading supplementary data")
    
    alpaca = AlpacaClient()
    
    for symbol in symbols:
        try:
            # Get latest quote
            quote = alpaca.get_latest_quote(symbol)
            if quote:
                logger.info(f"Latest quote for {symbol}: {quote}")
            
            # Get latest trade  
            trade = alpaca.get_latest_trade(symbol)
            if trade:
                logger.info(f"Latest trade for {symbol}: {trade}")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error getting supplementary data for {symbol}: {e}")


def verify_data_quality(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Verify downloaded data quality"""
    logger.info("\nData Quality Report:")
    
    for symbol, timeframe_data in data.items():
        logger.info(f"\n{symbol}:")
        
        for timeframe, df in timeframe_data.items():
            if df is not None and not df.empty:
                # Check for missing data
                missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
                
                # Date range
                start = df.index.min()
                end = df.index.max()
                
                logger.info(f"  {timeframe}:")
                logger.info(f"    Rows: {len(df)}")
                logger.info(f"    Date range: {start} to {end}")
                logger.info(f"    Missing data: {missing_pct.to_dict()}")
                
                # Check for data gaps
                if timeframe == "1Day":
                    expected_days = pd.bdate_range(start, end)
                    actual_days = df.index
                    missing_days = expected_days.difference(actual_days)
                    if len(missing_days) > 0:
                        logger.warning(f"    Missing {len(missing_days)} trading days")


def main():
    """Main function"""
    # Check for API credentials
    import os
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_API_SECRET'):
        logger.error("Missing Alpaca API credentials!")
        logger.error("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file")
        logger.error("You can get these from: https://alpaca.markets/")
        return
    
    # Load configuration
    try:
        config = load_config("config/config.yaml")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    
    # Use extended symbol list for better training data
    symbols = [
        # Mega caps
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "XOM", "DIS",
        # Large caps
        "CVX", "ABBV", "KO", "PEP", "MRK", "WMT", "PFE", "TMO", "CSCO",
        "VZ", "CMCSA", "ADBE", "NFLX", "INTC", "WFC", "CRM", "AMD", "BA",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV",
        # High volatility/popular
        "GME", "AMC", "PLTR", "SOFI", "RIVN", "LCID", "NIO"
    ]
    
    # Download parameters
    start_date = "2023-01-01"  # 2+ years of focused data
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 50)
    print("Historical Data Download Script")
    print("=" * 50)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    
    logger.info("=" * 50)
    logger.info("Historical Data Download Script")
    logger.info("=" * 50)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    try:
        # Download historical data
        data = download_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframes=["1Day", "1Hour", "15Min"]  # All major timeframes
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("This might be due to:")
        logger.error("1. Invalid API credentials")
        logger.error("2. Network connectivity issues")
        logger.error("3. Alpaca API limits or downtime")
        return
    
    # Verify data quality
    verify_data_quality(data)
    
    # Download supplementary data
    download_supplementary_data(symbols)
    
    print("\n" + "=" * 50)
    print("Download complete!")
    
    # Summary statistics
    total_bars = 0
    for symbol_data in data.values():
        for df in symbol_data.values():
            if df is not None:
                total_bars += len(df)
    
    print(f"Total bars downloaded: {total_bars:,}")
    print("Data saved to: data/historical/")
    
    logger.info("\n" + "=" * 50)
    logger.info("Download complete!")
    logger.info(f"Total bars downloaded: {total_bars:,}")
    logger.info("Data saved to: data/historical/")


if __name__ == "__main__":
    main()