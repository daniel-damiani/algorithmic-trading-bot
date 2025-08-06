#!/usr/bin/env python3
"""
Script to download quality historical data using the existing infrastructure
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.configuration import load_config
from src.data.data_fetcher import DataFetcher
from src.data.alpaca_client import AlpacaClient
import structlog

logger = structlog.get_logger(__name__)

# High-quality symbol selection
QUALITY_SYMBOLS = {
    'mega_caps': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'DIS'
    ],
    'large_caps': [
        'CVX', 'ABBV', 'KO', 'PEP', 'MRK', 'WMT', 'PFE', 'TMO', 'CSCO',
        'VZ', 'CMCSA', 'ADBE', 'NFLX', 'INTC', 'WFC', 'CRM', 'AMD', 'BA'
    ],
    'sector_leaders': [
        'GS', 'MS', 'C',  # Financials
        'CVS', 'CI', 'AMGN',  # Healthcare
        'COP', 'SLB', 'PSX',  # Energy
        'CAT', 'HON', 'UPS',  # Industrials
        'LOW', 'TGT', 'SBUX',  # Consumer
        'NEE', 'DUK', 'SO',  # Utilities
    ],
    'high_volume_etfs': [
        'SPY', 'QQQ', 'IWM', 'DIA', 'EEM', 'XLF', 'XLE', 'XLK', 'XLV',
        'GLD', 'TLT', 'HYG', 'VXX'
    ]
}

def main():
    """Download quality data using AlpacaClient"""
    
    # Load configuration
    config = load_config()
    
    # Initialize Alpaca client
    try:
        alpaca_client = AlpacaClient()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return 1
    
    # Combine all symbols
    all_symbols = []
    for category, symbols in QUALITY_SYMBOLS.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    
    logger.info(f"Downloading data for {len(all_symbols)} symbols")
    logger.info(f"Categories: {list(QUALITY_SYMBOLS.keys())}")
    
    # Download historical data for each symbol
    all_data = []
    failed_symbols = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    for i, symbol in enumerate(all_symbols):
        try:
            logger.info(f"[{i+1}/{len(all_symbols)}] Downloading {symbol}")
            
            # Get 2 years of data
            df = alpaca_client.get_bars(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe='Hour'  # Use 'Hour' not '1Hour'
            )
            
            if df is not None and len(df) > 100:
                # Add symbol column
                df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"  ✓ {symbol}: {len(df)} bars downloaded")
            else:
                logger.warning(f"  ✗ {symbol}: Insufficient data")
                failed_symbols.append(symbol)
                
        except Exception as e:
            logger.error(f"  ✗ {symbol}: Failed - {e}")
            failed_symbols.append(symbol)
        
        # Rate limiting to avoid API limits
        time.sleep(0.5)  # Half second between requests
    
    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_data = combined_data.sort_values(['symbol', 'timestamp'])
        
        # Save to file
        output_dir = Path('data')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'quality_historical_data.parquet'
        combined_data.to_parquet(output_file, index=False)
        
        # Create summary
        summary = f"""
Data Download Summary
====================
Total symbols attempted: {len(all_symbols)}
Successful downloads: {len(all_symbols) - len(failed_symbols)}
Failed downloads: {len(failed_symbols)}
Total data points: {len(combined_data):,}
Date range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}
File saved to: {output_file}
File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB

Failed symbols: {', '.join(failed_symbols) if failed_symbols else 'None'}

Symbol distribution:
"""
        # Add symbol counts
        symbol_counts = combined_data['symbol'].value_counts()
        for symbol, count in symbol_counts.head(10).items():
            summary += f"  {symbol}: {count:,} bars\n"
        
        print(summary)
        
        # Save summary
        with open(output_dir / 'download_summary.txt', 'w') as f:
            f.write(summary)
        
        logger.info("\n✅ Data download complete!")
        logger.info(f"Use this data for training: --data {output_file}")
        
    else:
        logger.error("❌ No data downloaded successfully")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())