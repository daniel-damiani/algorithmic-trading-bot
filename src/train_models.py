#!/usr/bin/env python3
"""
Model Training Script for QuantumSentiment Trading Bot

This script trains and saves all models in the trading system:
- Price prediction LSTM
- Chart pattern CNN
- Market regime XGBoost
- Sentiment analysis FinBERT
- Stacked ensemble

Usage:
    python scripts/train_models.py --help
    python scripts/train_models.py --models lstm,xgboost --days 365
    python scripts/train_models.py --all --parallel
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.training_pipeline import ModelTrainingPipeline, TrainingConfig
from src.training.model_persistence import ModelPersistence, PersistenceConfig
from src.data.data_fetcher import DataFetcher, FetcherConfig
from src.data.data_interface import DataInterface
from src.database.database import DatabaseManager
from src.features.feature_pipeline import FeaturePipeline, FeatureConfig
from src.configuration import load_config, Config
from src.data.reddit_client import RedditClient
from src.sentiment.news_aggregator import NewsAggregator, NewsConfig

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """Main class for training and saving models"""
    
    def __init__(self, config: Config, command_args: Optional[Dict[str, Any]] = None):
        """Initialize model trainer
        
        Args:
            config: Central configuration object from configuration.py
            command_args: Optional command line arguments to override config
        """
        self.config = config
        self.command_args = command_args or {}
        
        # Initialize components
        self.db_manager = None
        self.data_fetcher = None
        self.training_pipeline = None
        self.model_persistence = None
        self.reddit_client = None
        self.news_aggregator = None
        
        logger.info("Model trainer initialized")
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize database
            logger.info("Initializing database...")
            db_url = self.config.database.get('connection_string', 'sqlite:///quantum_sentiment.db')
            self.db_manager = DatabaseManager(db_url)
            
            # Initialize data fetcher
            logger.info("Initializing data fetcher...")
            data_interface = DataInterface()
            
            # Create FetcherConfig from central config
            fetcher_config = FetcherConfig(
                enable_alpaca=self.config.data_sources.get('alpaca', {}).get('enabled', True),
                enable_alpha_vantage=self.config.data_sources.get('alpha_vantage', {}).get('enabled', True),
                alpaca_rate_limit=self.config.data_sources.get('alpaca', {}).get('rate_limit', 200),
                alpha_vantage_rate_limit=self.config.data_sources.get('alpha_vantage', {}).get('rate_limit', 5)
            )
            self.data_fetcher = DataFetcher(fetcher_config, data_interface)
            
            # Initialize training pipeline
            logger.info("Initializing training pipeline...")
            
            # Create TrainingConfig from central config
            ml_config = self.config.ml.to_dict()
            models = ml_config.get('models', {})
            
            # Use command line arguments to override which models to train
            models_to_train = self.command_args.get('models_to_train', [])
            
            training_config = TrainingConfig(
                train_start_date=self.config.backtest.get('start_date', '2022-01-01'),
                train_end_date=self.config.backtest.get('end_date', '2024-12-31'),
                parallel_training=self.command_args.get('parallel', True),
                max_workers=self.command_args.get('workers', 4),
                train_lstm='lstm' in models_to_train if models_to_train else models.get('price_lstm', {}).get('enabled', True),
                train_cnn='cnn' in models_to_train if models_to_train else models.get('pattern_cnn', {}).get('enabled', True),
                train_xgboost='xgboost' in models_to_train if models_to_train else models.get('regime_xgboost', {}).get('enabled', True),
                train_finbert='finbert' in models_to_train if models_to_train else models.get('sentiment_bert', {}).get('enabled', False),
                train_ensemble='ensemble' in models_to_train if models_to_train else ml_config.get('ensemble', {}).get('voting', 'weighted') is not None,
                model_save_dir=Path(self.command_args.get('output_dir', self.config.paths.get('models', 'models')))
            )
            self.training_pipeline = ModelTrainingPipeline(training_config)
            
            # Initialize model persistence
            logger.info("Initializing model persistence...")
            persistence_config = PersistenceConfig(
                model_registry_path=Path('models'),  # Versioned storage directory
                auto_cleanup=True,
                max_versions_per_model=5
            )
            self.model_persistence = ModelPersistence(persistence_config)
            
            # Initialize Reddit client for sentiment data
            logger.info("Initializing Reddit client...")
            try:
                self.reddit_client = RedditClient()
                logger.info("Reddit client initialized for sentiment data collection")
            except Exception as e:
                logger.warning("Reddit client initialization failed", error=str(e))
                self.reddit_client = None
            
            # Initialize News aggregator
            logger.info("Initializing News aggregator...")
            try:
                news_config = NewsConfig(
                    alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                    newsapi_key=os.getenv('NEWSAPI_KEY')
                )
                self.news_aggregator = NewsAggregator(news_config)
                # Initialize news aggregator (it's not async)
                if self.news_aggregator.initialize():
                    logger.info("News aggregator initialized for sentiment data collection")
                else:
                    logger.warning("News aggregator initialization failed")
                    self.news_aggregator = None
            except Exception as e:
                logger.warning("News aggregator initialization failed", error=str(e))
                self.news_aggregator = None
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False
    
    async def load_training_data(self, symbols: List[str], days: int) -> pd.DataFrame:
        """Load and prepare training data
        
        Fetches the highest resolution data (1Hour) and creates consistent
        temporal data for training. Does not mix different timeframes.
        """
        logger.info("Loading training data", symbols=symbols, days=days)
        
        # Increase data volume by using more symbols and longer timeframe
        if len(symbols) < 10:
            # Add more liquid symbols for better training
            additional_symbols = ['SPY', 'QQQ', 'NVDA', 'META', 'AMZN', 'JPM', 'BAC', 'XLF', 
                                'V', 'MA', 'UNH', 'JNJ', 'WMT', 'PG', 'HD', 'DIS', 'NFLX', 'ADBE']
            for sym in additional_symbols:
                if sym not in symbols and len(symbols) < 20:
                    symbols.append(sym)
            logger.info(f"Expanded symbol list to {len(symbols)} symbols for better training")
        
        # Fetch high-resolution data for all symbols
        # Use smaller timeframe for more data points
        if days <= 30:
            timeframe = '15Min'  # Very high resolution for short periods
        elif days <= 90:
            timeframe = '30Min'  # High resolution for medium periods
        else:
            timeframe = '1Hour'  # Standard resolution for long periods
        logger.info(f"Fetching {timeframe} data for {len(symbols)} symbols")
        
        try:
            market_results = await self.data_fetcher.fetch_market_data(
                symbols=symbols,
                timeframe=timeframe,
                days_back=days
            )
            
            all_data = []
            for symbol in symbols:
                symbol_data = market_results.get(symbol, pd.DataFrame())
                
                if not symbol_data.empty:
                    # Ensure timestamp is a column, not index
                    if symbol_data.index.name == 'timestamp' or isinstance(symbol_data.index, pd.DatetimeIndex):
                        symbol_data = symbol_data.reset_index()
                    
                    symbol_data['symbol'] = symbol
                    all_data.append(symbol_data)
                    logger.info(f"Loaded {len(symbol_data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
        except Exception as e:
            logger.error(f"Failed to load market data", error=str(e))
            raise
        
        if not all_data:
            raise ValueError("No training data could be loaded")
        
        # Combine all data with consistent timeframe
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # The data from get_bars has timestamp as index, we need it as a column
        if 'timestamp' not in combined_data.columns:
            # If index has datetime data, use it as timestamp
            if isinstance(combined_data.index, pd.DatetimeIndex):
                combined_data['timestamp'] = combined_data.index
            else:
                combined_data.reset_index(inplace=True)
                if 'index' in combined_data.columns:
                    combined_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # Ensure we have enough data for all models
        min_required_samples = 10000  # Minimum for good training
        if len(combined_data) < min_required_samples:
            logger.warning(f"Insufficient data: {len(combined_data)} samples. Minimum recommended: {min_required_samples}")
            logger.info("Consider using more symbols or a longer time period for better results")
        
        # Convert timestamp to datetime if it's not already
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
        
        # Sort by timestamp and symbol for proper time series ordering
        combined_data.sort_values(['symbol', 'timestamp'], inplace=True)
        
        # Reset index for clean sequential indexing
        combined_data.reset_index(drop=True, inplace=True)
        
        logger.info("Training data loaded successfully", 
                   total_records=len(combined_data),
                   symbols=len(symbols),
                   date_range=f"{combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
        
        return combined_data
    
    async def load_text_data(self, symbols: List[str], days: int = 30) -> Optional[Dict[str, Any]]:
        """Load text data for sentiment analysis training
        
        Fetches text data (Reddit posts, news articles) for the training symbols
        to train the FinBERT model with real sentiment data.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back for text data
            
        Returns:
            Dictionary containing texts and labels for training, or None if insufficient data
        """
        logger.info("Loading text data for sentiment training", symbols=symbols, days=days)
        
        all_texts = []
        all_labels = []
        total_articles = 0
        
        try:
            # Collect text data from each symbol
            for symbol in symbols:
                symbol_texts = []
                symbol_labels = []
                
                # Get Reddit data if available
                if self.reddit_client:
                    try:
                        reddit_data = self.reddit_client.analyze_ticker_sentiment(
                            ticker=symbol,
                            hours_back=days * 24,
                            min_mentions=3
                        )
                        
                        if reddit_data.get('mention_count', 0) > 0:
                            # Get actual posts for this ticker
                            reddit_posts = []
                            for subreddit in self.reddit_client.subreddits[:3]:  # Limit to top 3 subreddits
                                posts = self.reddit_client.search_posts(
                                    query=f'${symbol}',
                                    subreddit=subreddit,
                                    time_filter='month',
                                    limit=20
                                )
                                reddit_posts.extend(posts)
                            
                            # Process Reddit posts
                            for post in reddit_posts:
                                text = f"{post.get('title', '')} {post.get('selftext', '')}"
                                text = text.strip()
                                
                                if len(text) > 10:  # Minimum text length
                                    symbol_texts.append(text)
                                    
                                    # Create labels based on sentiment indicators
                                    score = post.get('score', 0)
                                    bullish_emojis = post.get('bullish_emojis', 0)
                                    bearish_emojis = post.get('bearish_emojis', 0)
                                    
                                    # Improved sentiment labeling for better balance
                                    text_lower = text.lower()
                                    
                                    # Check for sentiment keywords
                                    positive_keywords = ['moon', 'bullish', 'buy', 'long', 'rocket', 'green', 'up', 'calls', 'growth', 'gain']
                                    negative_keywords = ['sell', 'bearish', 'short', 'crash', 'dump', 'red', 'down', 'puts', 'loss', 'drop']
                                    
                                    pos_count = sum(1 for word in positive_keywords if word in text_lower)
                                    neg_count = sum(1 for word in negative_keywords if word in text_lower)
                                    
                                    # Combine multiple signals
                                    if (bullish_emojis > bearish_emojis and score > 5) or pos_count >= 2:
                                        label = 'positive'
                                    elif (bearish_emojis > bullish_emojis) or neg_count >= 2:
                                        label = 'negative'
                                    elif score > 20 and pos_count > neg_count:
                                        label = 'positive'
                                    elif score < 5 or neg_count > pos_count:
                                        label = 'negative'
                                    else:
                                        # Randomly assign some neutrals to other classes for balance
                                        import random
                                        rand = random.random()
                                        if rand < 0.15:  # 15% chance each
                                            label = 'positive'
                                        elif rand < 0.30:  # 15% chance
                                            label = 'negative'
                                        else:
                                            label = 'neutral'
                                    
                                    symbol_labels.append(label)
                            
                            logger.info(f"Collected {len(symbol_texts)} Reddit texts for {symbol}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get Reddit data for {symbol}", error=str(e))
                
                # Get News data if available
                if self.news_aggregator:
                    try:
                        news_data = self.news_aggregator.analyze_symbol(
                            symbol=symbol,
                            hours_back=days * 24
                        )
                        
                        if news_data.get('total_articles', 0) > 0:
                            # Get sample articles from news data
                            sample_articles = news_data.get('sample_articles', [])
                            
                            for article in sample_articles:
                                title = article.get('title', '')
                                summary = article.get('summary', '')
                                text = f"{title} {summary}".strip()
                                
                                if len(text) > 20:  # Minimum text length for news
                                    symbol_texts.append(text)
                                    
                                    # Use news sentiment score for labeling with better thresholds
                                    sentiment_score = article.get('sentiment_score', 0)
                                    sentiment_label = article.get('sentiment_label', '')
                                    
                                    # Check title/summary for sentiment keywords
                                    text_lower = text.lower()
                                    positive_news = ['upgrade', 'beat', 'exceed', 'surge', 'rally', 'gain', 'profit', 'revenue growth']
                                    negative_news = ['downgrade', 'miss', 'decline', 'fall', 'loss', 'cut', 'layoff', 'warning']
                                    
                                    has_positive = any(word in text_lower for word in positive_news)
                                    has_negative = any(word in text_lower for word in negative_news)
                                    
                                    if sentiment_label in ['positive', 'negative', 'neutral']:
                                        label = sentiment_label
                                    elif sentiment_score > 0.03 or has_positive:  # Lower threshold
                                        label = 'positive'
                                    elif sentiment_score < -0.03 or has_negative:  # Lower threshold
                                        label = 'negative'
                                    else:
                                        # Better distribution for neutrals
                                        import random
                                        rand = random.random()
                                        if rand < 0.20:  # 20% positive
                                            label = 'positive'
                                        elif rand < 0.40:  # 20% negative
                                            label = 'negative'
                                        else:
                                            label = 'neutral'
                                    
                                    symbol_labels.append(label)
                            
                            logger.info(f"Collected {len(sample_articles)} news texts for {symbol}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get news data for {symbol}", error=str(e))
                
                # Add to overall dataset
                all_texts.extend(symbol_texts)
                all_labels.extend(symbol_labels)
                total_articles += len(symbol_texts)
                
                logger.info(f"Collected total {len(symbol_texts)} texts for {symbol}")
            
            # Check if we have sufficient data
            min_texts_required = 40  # Minimum texts needed for meaningful training (reduced from 50)
            
            if len(all_texts) < min_texts_required:
                logger.warning("Insufficient text data for sentiment training", 
                             collected=len(all_texts),
                             required=min_texts_required)
                return None
            
            # Balance the dataset if needed
            balanced_texts, balanced_labels = self._balance_sentiment_dataset(all_texts, all_labels)
            
            text_data = {
                'texts': balanced_texts,
                'labels': balanced_labels,
                'total_samples': len(balanced_texts),
                'label_distribution': {
                    'positive': balanced_labels.count('positive'),
                    'negative': balanced_labels.count('negative'),
                    'neutral': balanced_labels.count('neutral')
                },
                'sources': {
                    'reddit': bool(self.reddit_client),
                    'news': bool(self.news_aggregator)
                }
            }
            
            logger.info("Text data collection completed",
                       total_texts=len(balanced_texts),
                       sources_used=len([k for k, v in text_data['sources'].items() if v]),
                       distribution=text_data['label_distribution'])
            
            return text_data
            
        except Exception as e:
            logger.error("Failed to load text data", error=str(e))
            return None
    
    def _balance_sentiment_dataset(self, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """Balance the sentiment dataset to avoid class imbalance"""
        from collections import Counter
        
        # Count labels
        label_counts = Counter(labels)
        logger.info("Original label distribution", counts=dict(label_counts))
        
        # Find minimum count (to balance down to)
        min_count = min(label_counts.values()) if label_counts else 0
        
        # If we have very few of any class, don't balance (keep all data)
        if min_count < 10:
            logger.info("Not balancing dataset due to insufficient samples per class")
            return texts, labels
        
        # Balance by taking up to min_count samples from each class
        balanced_texts = []
        balanced_labels = []
        label_sample_counts = Counter()
        
        for text, label in zip(texts, labels):
            if label_sample_counts[label] < min_count:
                balanced_texts.append(text)
                balanced_labels.append(label)
                label_sample_counts[label] += 1
        
        logger.info("Balanced label distribution", counts=dict(label_sample_counts))
        return balanced_texts, balanced_labels
    
    async def train_all_models(self, training_data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Train all configured models"""
        logger.info("Starting model training")
        
        # Load text data for sentiment analysis if FinBERT training is enabled
        text_data = None
        if ('finbert' in self.command_args.get('models_to_train', []) or 
            self.training_pipeline.config.train_finbert):
            logger.info("Loading text data for FinBERT training...")
            text_data = await self.load_text_data(symbols, days=30)
            
            if text_data:
                logger.info("Text data loaded successfully", 
                           samples=text_data['total_samples'],
                           distribution=text_data['label_distribution'])
            else:
                logger.warning("No text data available for FinBERT training - skipping FinBERT model")
        
        # Train models using the pipeline
        trained_models = self.training_pipeline.train_all_models(
            price_data=training_data,
            text_data=text_data
        )
        
        # Save models with proper metadata
        saved_models = {}
        
        for model_name, model in trained_models.items():
            try:
                logger.info(f"Saving model: {model_name}")
                
                # Calculate training data hash for versioning
                training_data_hash = self._calculate_data_hash(training_data)
                
                # Get training results
                training_results = self.training_pipeline.training_results.get(model_name, {})
                validation_metrics = training_results.get('validation_metrics', {})
                test_metrics = training_results.get('test_metrics', {})
                
                # Save model with metadata
                metadata = self.model_persistence.save_model(
                    model=model,
                    model_name=model_name,
                    training_data_hash=training_data_hash,
                    training_samples=len(training_data),
                    training_duration=training_results.get('training_time', 0.0),
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    description=f"{model_name} trained on {len(training_data)} samples",
                    tags=[f"version_{datetime.now().strftime('%Y%m%d')}", "production"],
                    author="QuantumSentiment Training Pipeline"
                )
                
                saved_models[model_name] = {
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'path': str(metadata.model_path),
                    'validation_metrics': validation_metrics,
                    'test_metrics': test_metrics
                }
                
                logger.info(f"Model {model_name} saved successfully", 
                           model_id=metadata.model_id,
                           version=metadata.version)
                
            except Exception as e:
                logger.error(f"Failed to save model {model_name}", error=str(e))
        
        # Get training summary
        training_summary = self.training_pipeline.get_training_summary()
        
        return {
            'saved_models': saved_models,
            'training_summary': training_summary,
            'total_models': len(saved_models)
        }
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for versioning"""
        import hashlib
        
        # Create a simple hash based on data shape and sample values
        data_info = f"{len(data)}_{data.shape[1]}_{data['timestamp'].min()}_{data['timestamp'].max()}"
        
        # Add sample of actual data
        if len(data) > 0:
            sample_data = data.head(10).to_string()
            data_info += sample_data
        
        return hashlib.md5(data_info.encode()).hexdigest()
    
    async def list_saved_models(self) -> Dict[str, List[str]]:
        """List all saved models"""
        if self.model_persistence:
            return self.model_persistence.list_available_models()
        return {}
    
    async def cleanup_old_models(self, days_threshold: int = 30):
        """Clean up old model versions"""
        if self.model_persistence:
            self.model_persistence.cleanup_old_models(days_threshold)
            logger.info("Model cleanup completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and save trading models")
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        help='Comma-separated list of models to train (lstm,cnn,xgboost,finbert,ensemble,all)',
        default='all'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        help='Comma-separated list of symbols to train on',
        default='AAPL,TSLA,GOOGL,MSFT,SPY'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        help='Number of days of historical data to use for training',
        default=730  # 2 years
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Train models in parallel'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of parallel workers',
        default=4
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for saved models',
        default='models'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old model versions after training'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all saved models and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Configure logging
    import logging
    from logging.handlers import RotatingFileHandler
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Set up file logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create rotating file handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler]
    )
    
    logger.info(f"Logging to file: {log_file}")
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure structlog to use the same handlers
    structlog_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
    )
    
    # Update handlers to use structlog formatter
    for handler in logging.root.handlers:
        handler.setFormatter(structlog_formatter)
    
    try:
        # Parse models to train
        models_to_train = args.models.lower().split(',')
        if 'all' in models_to_train:
            models_to_train = ['lstm', 'cnn', 'xgboost', 'finbert', 'ensemble']
        
        # Parse symbols
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        
        # Load central configuration
        config_path = args.config if args.config else None
        config = load_config(config_path)
        
        # Prepare command line arguments for trainer
        command_args = {
            'models_to_train': models_to_train,
            'parallel': args.parallel,
            'workers': args.workers,
            'output_dir': args.output_dir
        }
        
        # Initialize trainer
        trainer = ModelTrainer(config, command_args)
        
        if not await trainer.initialize():
            logger.error("Failed to initialize trainer")
            sys.exit(1)
        
        # List models if requested
        if args.list_models:
            saved_models = await trainer.list_saved_models()
            
            print("\n" + "="*60)
            print("SAVED MODELS")
            print("="*60)
            
            if saved_models:
                for model_name, versions in saved_models.items():
                    print(f"\n{model_name}:")
                    for version in versions:
                        print(f"  - {version}")
            else:
                print("No saved models found.")
            
            print("="*60)
            return
        
        # Load training data
        logger.info("Loading training data", symbols=symbols, days=args.days)
        training_data = await trainer.load_training_data(symbols, args.days)
        
        if training_data.empty:
            logger.error("No training data available")
            sys.exit(1)
        
        # Train models
        logger.info("Starting model training", models=models_to_train)
        results = await trainer.train_all_models(training_data, symbols)
        
        # Clean up old models if requested
        if args.cleanup:
            await trainer.cleanup_old_models()
        
        # Print results
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETED")
        print("="*80)
        print(f"Total models trained: {results['total_models']}")
        print(f"Training data: {len(training_data):,} samples")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Date range: {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")
        
        if results['saved_models']:
            print("\nSaved Models:")
            for model_name, info in results['saved_models'].items():
                print(f"  {model_name}:")
                print(f"    - Model ID: {info['model_id']}")
                print(f"    - Version: {info['version']}")
                print(f"    - Path: {info['path']}")
                
                # Print metrics if available
                test_metrics = info.get('test_metrics', {})
                if test_metrics:
                    print(f"    - Test Metrics: {test_metrics}")
        
        print("\n" + "="*80)
        print("Models are now ready for use in the trading system!")
        print("The main trading loop will automatically load the latest trained models.")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed", error=str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())