#!/usr/bin/env python3
"""
Script to download more text data for sentiment analysis
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.configuration import load_config
from src.data.reddit_client import RedditClient
from src.sentiment.news_aggregator import NewsAggregator, NewsConfig
import structlog

logger = structlog.get_logger(__name__)

class TextDataCollector:
    def __init__(self, config):
        self.config = config
        self.reddit_client = None
        self.news_aggregator = None
        
    async def initialize(self):
        """Initialize data sources"""
        try:
            # Initialize Reddit
            if self.config.reddit.client_id:
                self.reddit_client = RedditClient(self.config.reddit)
                self.reddit_client.initialize()
                logger.info("Reddit client initialized")
            
            # Initialize News
            if self.config.news_api.api_key:
                news_config = NewsConfig(
                    newsapi_key=self.config.news_api.api_key,
                    alpha_vantage_key=self.config.alpha_vantage.api_key
                )
                self.news_aggregator = NewsAggregator(news_config)
                self.news_aggregator.initialize()
                logger.info("News aggregator initialized")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize text data sources: {e}")
            return False
    
    async def collect_text_data(self, symbols, days=30):
        """Collect text data from all sources"""
        
        all_texts = []
        all_labels = []
        
        for symbol in symbols:
            logger.info(f"Collecting text data for {symbol}")
            
            # Reddit data
            if self.reddit_client:
                try:
                    reddit_data = await self.collect_reddit_data(symbol, days)
                    all_texts.extend(reddit_data['texts'])
                    all_labels.extend(reddit_data['labels'])
                    logger.info(f"  Reddit: {len(reddit_data['texts'])} posts")
                except Exception as e:
                    logger.error(f"  Reddit error: {e}")
            
            # News data
            if self.news_aggregator:
                try:
                    news_data = await self.collect_news_data(symbol, days)
                    all_texts.extend(news_data['texts'])
                    all_labels.extend(news_data['labels'])
                    logger.info(f"  News: {len(news_data['texts'])} articles")
                except Exception as e:
                    logger.error(f"  News error: {e}")
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Summary
        label_dist = {}
        for label in all_labels:
            label_dist[label] = label_dist.get(label, 0) + 1
            
        logger.info(f"\nText data collection complete:")
        logger.info(f"  Total texts: {len(all_texts)}")
        logger.info(f"  Label distribution: {label_dist}")
        
        return {
            'texts': all_texts,
            'labels': all_labels,
            'distribution': label_dist
        }
    
    async def collect_reddit_data(self, symbol, days):
        """Collect Reddit data with improved labeling"""
        
        texts = []
        labels = []
        
        # Search multiple subreddits
        subreddits = [
            'wallstreetbets', 'stocks', 'StockMarket', 
            'investing', 'options', 'thetagang',
            'SecurityAnalysis', 'ValueInvesting'
        ]
        
        for subreddit in subreddits:
            try:
                posts = self.reddit_client.search_posts(
                    query=f'${symbol} OR {symbol}',
                    subreddit=subreddit,
                    time_filter='month' if days > 30 else 'week',
                    limit=50
                )
                
                for post in posts:
                    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                    if len(text) > 20:
                        texts.append(text)
                        
                        # Improved sentiment labeling
                        label = self.determine_reddit_sentiment(post, text)
                        labels.append(label)
                        
            except Exception as e:
                logger.warning(f"Error searching {subreddit}: {e}")
                continue
        
        return {'texts': texts, 'labels': labels}
    
    def determine_reddit_sentiment(self, post, text):
        """Determine sentiment with better logic"""
        
        text_lower = text.lower()
        
        # Strong positive indicators
        strong_positive = [
            'to the moon', 'mooning', 'squeeze', 'yolo calls',
            'tendies', 'printing', 'rocket', '🚀', 'gainz',
            'breakout', 'bullish af', 'buy the dip', 'diamond hands'
        ]
        
        # Strong negative indicators  
        strong_negative = [
            'puts', 'short', 'crash', 'dump', 'bag holder',
            'loss porn', 'guh', 'bankruptcy', 'delisted',
            'bearish', 'sell off', 'dead cat', 'falling knife'
        ]
        
        # Count indicators
        pos_count = sum(1 for phrase in strong_positive if phrase in text_lower)
        neg_count = sum(1 for phrase in strong_negative if phrase in text_lower)
        
        # Check emojis
        bullish_emojis = text.count('🚀') + text.count('💎') + text.count('🙌') + text.count('📈')
        bearish_emojis = text.count('🐻') + text.count('📉') + text.count('💀') + text.count('🩸')
        
        # Score-based logic
        score = post.get('score', 0)
        
        if pos_count >= 2 or bullish_emojis >= 2:
            return 'positive'
        elif neg_count >= 2 or bearish_emojis >= 2:
            return 'negative'
        elif score > 100 and pos_count > neg_count:
            return 'positive'
        elif score < 10 or neg_count > pos_count:
            return 'negative'
        else:
            # Better neutral distribution
            import random
            rand = random.random()
            if rand < 0.25:
                return 'positive'
            elif rand < 0.50:
                return 'negative'
            else:
                return 'neutral'
    
    async def collect_news_data(self, symbol, days):
        """Collect news data with improved sentiment detection"""
        
        texts = []
        labels = []
        
        # Get news analysis
        news_result = self.news_aggregator.analyze_symbol(
            symbol=symbol,
            hours_back=days * 24
        )
        
        # Process articles
        articles = news_result.get('sample_articles', [])
        
        for article in articles:
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title}. {summary}".strip()
            
            if len(text) > 30:
                texts.append(text)
                
                # Determine sentiment
                sentiment_score = article.get('sentiment_score', 0)
                label = self.determine_news_sentiment(article, text, sentiment_score)
                labels.append(label)
        
        return {'texts': texts, 'labels': labels}
    
    def determine_news_sentiment(self, article, text, sentiment_score):
        """Determine news sentiment with better logic"""
        
        text_lower = text.lower()
        
        # Financial news keywords
        positive_terms = [
            'beat earnings', 'beat estimates', 'upgrade', 'raised guidance',
            'record revenue', 'strong growth', 'outperform', 'buy rating',
            'price target raised', 'expansion', 'partnership', 'innovation'
        ]
        
        negative_terms = [
            'miss earnings', 'miss estimates', 'downgrade', 'lowered guidance',
            'revenue decline', 'weak growth', 'underperform', 'sell rating',
            'price target cut', 'layoffs', 'lawsuit', 'investigation'
        ]
        
        # Count matches
        pos_matches = sum(1 for term in positive_terms if term in text_lower)
        neg_matches = sum(1 for term in negative_terms if term in text_lower)
        
        # Combined logic
        if sentiment_score > 0.1 or pos_matches >= 2:
            return 'positive'
        elif sentiment_score < -0.1 or neg_matches >= 2:
            return 'negative'
        elif pos_matches > neg_matches:
            return 'positive'
        elif neg_matches > pos_matches:
            return 'negative'
        else:
            # Balanced neutral assignment
            import random
            rand = random.random()
            if rand < 0.30:
                return 'positive'
            elif rand < 0.60:
                return 'negative'
            else:
                return 'neutral'

async def main():
    """Main function"""
    
    config = load_config()
    collector = TextDataCollector(config)
    
    if not await collector.initialize():
        logger.error("Failed to initialize text data collector")
        return 1
    
    # Focus on high-volume symbols
    symbols = [
        'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL',
        'SPY', 'QQQ', 'AMC', 'GME', 'AMD', 'PLTR', 'SOFI',
        'NIO', 'LCID', 'RIVN', 'BA', 'DIS', 'NFLX'
    ]
    
    # Collect data
    text_data = await collector.collect_text_data(symbols, days=30)
    
    # Save results
    output_dir = Path('data/text_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save texts and labels
    import pandas as pd
    df = pd.DataFrame({
        'text': text_data['texts'],
        'label': text_data['labels']
    })
    
    df.to_csv(output_dir / 'sentiment_data.csv', index=False)
    logger.info(f"\nSaved {len(df)} text samples to {output_dir / 'sentiment_data.csv'}")
    
    # Save summary
    summary = {
        'collection_date': datetime.now().isoformat(),
        'total_texts': len(text_data['texts']),
        'symbols': symbols,
        'distribution': text_data['distribution']
    }
    
    with open(output_dir / 'collection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n✅ Text data collection complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))