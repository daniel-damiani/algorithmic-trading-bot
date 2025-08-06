#!/usr/bin/env python3
"""
Download MASSIVE historical data for production-grade model training.
Target: Maximum data for best model performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path
import structlog
import argparse
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = structlog.get_logger()

# Quality symbols - mix of sectors for robustness
TECH_GIANTS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "ORCL", "CRM", "ADBE", "NFLX", "INTC", "AMD", "QCOM", "TXN"]
FINANCIALS = ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "TFC", "COF", "BK", "STT"]
HEALTHCARE = ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "CVS", "DHR", "MDT", "BMY", "AMGN", "GILD", "SYK", "BSX"]
CONSUMER = ["WMT", "PG", "KO", "PEP", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "TJX", "DG", "MDLZ", "CL"]
INDUSTRIALS = ["BA", "CAT", "HON", "UPS", "LMT", "RTX", "DE", "MMM", "GE", "FDX", "EMR", "ITW", "NOC", "ETN", "CSX"]
ENERGY = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD", "OXY", "KMI", "WMB", "DVN", "HAL", "BKR"]
ETFS = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "GLD", "SLV", "TLT", "XLF", "XLE", "XLK", "XLV", "XLI"]

# Combine all symbols
ALL_SYMBOLS = TECH_GIANTS + FINANCIALS + HEALTHCARE + CONSUMER + INDUSTRIALS + ENERGY + ETFS

def download_symbol_data(symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """Download data for a single symbol with retries."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {symbol}... (attempt {attempt + 1})")
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Reset index to have timestamp as column
            df.reset_index(inplace=True)
            if 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
            elif 'datetime' in df.columns:
                df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            
            logger.info(f"✅ {symbol}: {len(df)} rows downloaded")
            return df
            
        except Exception as e:
            logger.warning(f"Error downloading {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download {symbol} after {max_retries} attempts")
                return pd.DataFrame()
    
    return pd.DataFrame()

def download_massive_dataset(
    symbols: List[str],
    years: int = 5,
    interval: str = "1h",
    max_workers: int = 10
) -> pd.DataFrame:
    """Download massive dataset with parallel processing."""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    logger.info(f"Downloading {years} years of data for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Interval: {interval}")
    
    all_data = []
    failed_symbols = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_symbol = {
            executor.submit(download_symbol_data, symbol, start_date.strftime('%Y-%m-%d'), 
                          end_date.strftime('%Y-%m-%d'), interval): symbol
            for symbol in symbols
        }
        
        # Process completed downloads
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if not df.empty:
                    all_data.append(df)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Exception for {symbol}: {e}")
                failed_symbols.append(symbol)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total rows downloaded: {len(combined_df):,}")
        logger.info(f"Successful symbols: {len(symbols) - len(failed_symbols)}")
        logger.info(f"Failed symbols: {len(failed_symbols)}")
        
        if failed_symbols:
            logger.warning(f"Failed to download: {', '.join(failed_symbols)}")
        
        return combined_df
    else:
        logger.error("No data downloaded!")
        return pd.DataFrame()

def save_dataset(df: pd.DataFrame, output_dir: Path):
    """Save dataset with metadata."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet for efficiency
    data_file = output_dir / "massive_training_data.parquet"
    df.to_parquet(data_file, engine='pyarrow', compression='snappy')
    logger.info(f"Data saved to {data_file}")
    
    # Save metadata
    metadata = {
        "total_rows": len(df),
        "symbols": df['symbol'].unique().tolist(),
        "n_symbols": df['symbol'].nunique(),
        "date_range": {
            "start": str(df['timestamp'].min()),
            "end": str(df['timestamp'].max())
        },
        "columns": df.columns.tolist(),
        "download_date": datetime.now().isoformat(),
        "file_size_mb": data_file.stat().st_size / (1024 * 1024)
    }
    
    metadata_file = output_dir / "data_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MASSIVE DATA DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total rows: {metadata['total_rows']:,}")
    print(f"Symbols: {metadata['n_symbols']}")
    print(f"Date range: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
    print(f"File size: {metadata['file_size_mb']:.2f} MB")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Download massive historical data for training")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (default: use all)")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")
    parser.add_argument("--interval", type=str, default="1h", 
                       choices=["1h", "1d", "5m", "15m", "30m"], 
                       help="Data interval (default: 1h)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--output", type=str, default="data/massive", help="Output directory")
    
    args = parser.parse_args()
    
    # Select symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = ALL_SYMBOLS
    
    logger.info(f"Starting massive data download for {len(symbols)} symbols...")
    
    # Download data
    df = download_massive_dataset(
        symbols=symbols,
        years=args.years,
        interval=args.interval,
        max_workers=args.workers
    )
    
    if not df.empty:
        # Save dataset
        output_dir = Path(args.output)
        save_dataset(df, output_dir)
        
        # Create a sample for quick testing
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_file = output_dir / "sample_data.parquet"
        sample_df.to_parquet(sample_file)
        logger.info(f"Sample data ({sample_size} rows) saved to {sample_file}")
    else:
        logger.error("Download failed - no data to save")
        sys.exit(1)

if __name__ == "__main__":
    main()