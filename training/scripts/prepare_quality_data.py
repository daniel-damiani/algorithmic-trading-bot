#!/usr/bin/env python3
"""
Script to prepare high-quality training data for the algorithmic trading bot
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))

from src.configuration import load_config
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

class DataQualityChecker:
    def __init__(self, config):
        self.config = config
        self.alpaca_client = AlpacaClient()
        
    def download_quality_data(self, symbols, days=730):
        """Download high-quality historical data"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        failed_symbols = []
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading data for {symbol}")
                
                # Get data from Alpaca
                bars = self.alpaca_client.get_bars(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe='1Hour'  # Hourly data for better patterns
                )
                
                if bars is not None and len(bars) > 100:
                    # Add symbol column
                    bars['symbol'] = symbol
                    
                    # Quality checks
                    bars = self.apply_quality_filters(bars, symbol)
                    
                    if len(bars) > 100:  # Still has data after filtering
                        all_data.append(bars)
                        logger.info(f"✓ {symbol}: {len(bars)} quality bars")
                    else:
                        logger.warning(f"✗ {symbol}: Insufficient data after quality filtering")
                        failed_symbols.append(symbol)
                else:
                    logger.warning(f"✗ {symbol}: No data retrieved")
                    failed_symbols.append(symbol)
                    
                # Rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"\nData download complete:")
            logger.info(f"  Total records: {len(combined_data):,}")
            logger.info(f"  Successful symbols: {len(symbols) - len(failed_symbols)}")
            logger.info(f"  Failed symbols: {len(failed_symbols)}")
            
            return combined_data, failed_symbols
        else:
            logger.error("No data downloaded successfully")
            return pd.DataFrame(), failed_symbols
    
    def apply_quality_filters(self, data, symbol):
        """Apply quality filters to ensure clean data"""
        
        original_len = len(data)
        
        # Remove any rows with null prices
        data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove zero volume bars (market closed)
        data = data[data['volume'] > 0]
        
        # Remove bars with suspicious price movements (>20% in an hour)
        data['price_change'] = data['close'].pct_change().abs()
        data = data[data['price_change'] < 0.20]
        
        # Remove bars where high < low (data error)
        data = data[data['high'] >= data['low']]
        
        # Remove bars where close is outside high/low range
        data = data[(data['close'] <= data['high']) & (data['close'] >= data['low'])]
        
        # Drop temporary columns
        data = data.drop(columns=['price_change'])
        
        filtered_len = len(data)
        if filtered_len < original_len:
            logger.debug(f"{symbol}: Filtered {original_len - filtered_len} low-quality bars")
        
        return data
    
    def validate_data_quality(self, data):
        """Validate the quality of downloaded data"""
        
        logger.info("\nData Quality Report:")
        logger.info("="*50)
        
        # Check data completeness
        symbols = data['symbol'].unique()
        logger.info(f"Unique symbols: {len(symbols)}")
        
        # Check date range
        date_range = data['timestamp'].agg(['min', 'max'])
        logger.info(f"Date range: {date_range['min']} to {date_range['max']}")
        
        # Check for gaps
        for symbol in symbols[:5]:  # Check first 5 symbols
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            time_diffs = symbol_data['timestamp'].diff()
            
            # Count gaps > 1 day (accounting for weekends)
            large_gaps = time_diffs[time_diffs > timedelta(days=3)]
            if len(large_gaps) > 0:
                logger.warning(f"{symbol}: {len(large_gaps)} gaps > 3 days detected")
        
        # Check price statistics
        price_stats = data.groupby('symbol')['close'].agg(['count', 'mean', 'std'])
        logger.info(f"\nPrice statistics:")
        logger.info(f"  Avg bars per symbol: {price_stats['count'].mean():.0f}")
        logger.info(f"  Min bars per symbol: {price_stats['count'].min()}")
        logger.info(f"  Max bars per symbol: {price_stats['count'].max()}")
        
        # Check for sufficient variance
        low_variance_symbols = price_stats[price_stats['std'] / price_stats['mean'] < 0.001]
        if len(low_variance_symbols) > 0:
            logger.warning(f"Low variance symbols: {low_variance_symbols.index.tolist()}")
        
        return len(symbols), len(data)

def main():
    """Main data preparation function"""
    
    config = load_config()
    checker = DataQualityChecker(config)
    
    # Combine all symbols
    all_symbols = []
    for category, symbols in QUALITY_SYMBOLS.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    
    logger.info(f"Preparing to download data for {len(all_symbols)} symbols")
    logger.info(f"Categories: {list(QUALITY_SYMBOLS.keys())}")
    
    # Download data
    data, failed = checker.download_quality_data(all_symbols, days=730)
    
    if not data.empty:
        # Validate quality
        checker.validate_data_quality(data)
        
        # Save to file
        output_file = Path("data/historical_data_quality.parquet")
        output_file.parent.mkdir(exist_ok=True)
        
        data.to_parquet(output_file, index=False)
        logger.info(f"\nData saved to: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Create a summary
        summary = {
            'download_date': datetime.now().isoformat(),
            'total_symbols': len(all_symbols),
            'successful_symbols': len(all_symbols) - len(failed),
            'failed_symbols': failed,
            'total_records': len(data),
            'date_range': {
                'start': data['timestamp'].min().isoformat(),
                'end': data['timestamp'].max().isoformat()
            }
        }
        
        import json
        with open('data/data_quality_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n✅ Data preparation complete!")
        logger.info("Run training with: python src/train_models.py --data data/historical_data_quality.parquet")
        
    else:
        logger.error("❌ Data preparation failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())