#!/usr/bin/env python3
"""
Download MASSIVE production-quality training data for high-performance model training.

This script downloads extensive historical data for training a production-grade model.
Focuses on liquid stocks with good data quality and long history.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple
import yfinance as yf
import logging
from tqdm import tqdm
import time
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configuration import load_config
from src.data.data_fetcher import DataFetcher
from src.sentiment.reddit_analyzer import RedditAnalyzer
from src.sentiment.news_aggregator import NewsAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Top liquid stocks across different sectors for diversity
PRODUCTION_SYMBOLS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    
    # Large-cap tech
    'AVGO', 'ORCL', 'CRM', 'CSCO', 'ADBE', 'INTC', 'AMD', 'QCOM',
    'IBM', 'NOW', 'UBER', 'NFLX', 'SHOP', 'SNOW', 'PLTR', 'NET',
    
    # Financials
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
    'SCHW', 'BLK', 'SPGI', 'CB', 'PGR', 'TFC', 'USB', 'PNC',
    
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'CVS', 'TMO', 'ABT',
    'DHR', 'BMY', 'AMGN', 'MDT', 'GILD', 'ISRG', 'VRTX', 'REGN',
    
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'DIS', 'NKE',
    'SBUX', 'TGT', 'LOW', 'TJX', 'CMG', 'YUM', 'MAR', 'BKNG',
    
    # Industrials & Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO',
    'BA', 'UNP', 'HON', 'RTX', 'LMT', 'CAT', 'DE', 'GE',
    
    # ETFs for market context
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'GLD', 'TLT',
    
    # High volatility / Meme stocks for edge cases
    'GME', 'AMC', 'BB', 'BBBY', 'SOFI', 'RIOT', 'MARA', 'COIN',
    
    # International ADRs
    'TSM', 'BABA', 'NVO', 'ASML', 'TM', 'SAP', 'SNY', 'BP'
]

async def download_alpaca_data(fetcher: DataFetcher, symbol: str, timeframe: str = '1Hour') -> pd.DataFrame:
    """Download data from Alpaca with long history"""
    try:
        # Try to get 3 years of hourly data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        
        logger.info(f"Downloading {symbol} from Alpaca ({timeframe})...")
        
        bars = await fetcher.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            limit=50000  # Max limit
        )
        
        if bars is not None and not bars.empty:
            logger.info(f"✅ Downloaded {len(bars)} bars for {symbol}")
            return bars
        else:
            logger.warning(f"⚠️ No Alpaca data for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error downloading {symbol} from Alpaca: {e}")
        return None

def download_yfinance_data(symbol: str, period: str = 'max', interval: str = '1h') -> pd.DataFrame:
    """Download data from Yahoo Finance as backup/supplement"""
    try:
        logger.info(f"Downloading {symbol} from Yahoo Finance...")
        
        ticker = yf.Ticker(symbol)
        
        # Get maximum available history
        hist = ticker.history(period=period, interval=interval)
        
        if not hist.empty:
            # Rename columns to match our format
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only OHLCV columns
            hist = hist[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✅ Downloaded {len(hist)} bars for {symbol} from Yahoo")
            return hist
        else:
            logger.warning(f"⚠️ No Yahoo data for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error downloading {symbol} from Yahoo: {e}")
        return None

async def download_sentiment_data(
    symbol: str, 
    reddit_analyzer: RedditAnalyzer,
    news_aggregator: NewsAggregator
) -> Dict:
    """Download sentiment data for training"""
    sentiment_data = {
        'symbol': symbol,
        'reddit_mentions': [],
        'news_mentions': [],
        'sentiment_scores': []
    }
    
    try:
        # Get Reddit sentiment
        logger.info(f"Fetching Reddit sentiment for {symbol}...")
        reddit_sentiment = await reddit_analyzer.analyze_symbol(symbol)
        if reddit_sentiment:
            sentiment_data['reddit_mentions'].append(reddit_sentiment)
        
        # Get news sentiment
        logger.info(f"Fetching news sentiment for {symbol}...")
        news_sentiment = await news_aggregator.get_sentiment(symbol)
        if news_sentiment:
            sentiment_data['news_mentions'].append(news_sentiment)
            
    except Exception as e:
        logger.warning(f"Sentiment fetch error for {symbol}: {e}")
    
    return sentiment_data

async def process_symbol(
    symbol: str,
    fetcher: DataFetcher,
    reddit_analyzer: RedditAnalyzer,
    news_aggregator: NewsAggregator,
    output_dir: Path
) -> Tuple[str, bool]:
    """Process a single symbol - download all data"""
    
    try:
        # Create symbol directory
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        
        # 1. Try Alpaca first (best quality)
        alpaca_data = await download_alpaca_data(fetcher, symbol)
        if alpaca_data is not None:
            all_data.append(('alpaca', alpaca_data))
        
        # 2. Get Yahoo Finance data (backup/supplement)
        yahoo_data = download_yfinance_data(symbol)
        if yahoo_data is not None:
            all_data.append(('yahoo', yahoo_data))
        
        # 3. Get sentiment data
        sentiment = await download_sentiment_data(
            symbol, reddit_analyzer, news_aggregator
        )
        
        # Save all data
        if all_data:
            # Combine data from all sources
            combined_data = None
            for source, data in all_data:
                if combined_data is None:
                    combined_data = data
                else:
                    # Merge, avoiding duplicates
                    combined_data = pd.concat([combined_data, data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            
            # Sort by timestamp
            combined_data = combined_data.sort_index()
            
            # Save market data
            market_file = symbol_dir / f"{symbol}_combined_hourly.csv"
            combined_data.to_csv(market_file)
            logger.info(f"💾 Saved {len(combined_data)} bars to {market_file}")
            
            # Save sentiment data
            if sentiment['reddit_mentions'] or sentiment['news_mentions']:
                sentiment_file = symbol_dir / f"{symbol}_sentiment.json"
                pd.Series(sentiment).to_json(sentiment_file)
                logger.info(f"💾 Saved sentiment data to {sentiment_file}")
            
            return symbol, True
        else:
            logger.warning(f"⚠️ No data available for {symbol}")
            return symbol, False
            
    except Exception as e:
        logger.error(f"❌ Failed to process {symbol}: {e}")
        return symbol, False

async def main():
    """Main download function"""
    
    print("=" * 80)
    print("MASSIVE PRODUCTION DATA DOWNLOAD")
    print(f"Downloading data for {len(PRODUCTION_SYMBOLS)} symbols")
    print("=" * 80)
    
    # Setup
    config = load_config()
    output_dir = Path("data/training/production")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data sources
    fetcher = DataFetcher(config)
    reddit_analyzer = RedditAnalyzer(config)
    news_aggregator = NewsAggregator(config)
    
    # Process in batches to avoid rate limits
    batch_size = 5
    successful = []
    failed = []
    
    for i in range(0, len(PRODUCTION_SYMBOLS), batch_size):
        batch = PRODUCTION_SYMBOLS[i:i+batch_size]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(PRODUCTION_SYMBOLS) + batch_size - 1)//batch_size}")
        print(f"Symbols: {batch}")
        
        # Process batch
        tasks = [
            process_symbol(symbol, fetcher, reddit_analyzer, news_aggregator, output_dir)
            for symbol in batch
        ]
        
        results = await asyncio.gather(*tasks)
        
        for symbol, success in results:
            if success:
                successful.append(symbol)
            else:
                failed.append(symbol)
        
        # Rate limit between batches
        if i + batch_size < len(PRODUCTION_SYMBOLS):
            wait_time = random.uniform(2, 5)
            print(f"Waiting {wait_time:.1f} seconds before next batch...")
            time.sleep(wait_time)
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print(f"✅ Successful: {len(successful)} symbols")
    print(f"❌ Failed: {len(failed)} symbols")
    
    if failed:
        print(f"\nFailed symbols: {failed}")
    
    # Calculate total data size
    total_bars = 0
    for symbol_dir in output_dir.iterdir():
        if symbol_dir.is_dir():
            for csv_file in symbol_dir.glob("*.csv"):
                df = pd.read_csv(csv_file, index_col=0)
                total_bars += len(df)
    
    print(f"\n📊 Total data points: {total_bars:,}")
    print(f"📁 Data saved to: {output_dir.absolute()}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())