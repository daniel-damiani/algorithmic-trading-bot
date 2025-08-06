"""
QuantumSentiment Algorithmic Trading Bot

Main entry point for the complete algorithmic trading system.
Integrates all components for autonomous trading with sentiment analysis.

Features:
- Multi-model ensemble predictions (CNN, LSTM, XGBoost, Transformers)
- Real-time sentiment analysis from Reddit and news
- Advanced portfolio optimization (Black-Litterman, Markowitz, Risk Parity)
- Smart order routing with execution optimization
- Comprehensive risk management
- Paper trading and live trading support
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import structlog
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all components
sys.path.insert(0, str(Path(__file__).parent))
from configuration import Config
from src.config.config_manager import ConfigManager
from src.database import DatabaseManager
from src.data import AlpacaClient, DataFetcher
from src.sentiment import SentimentFusion, RedditSentimentAnalyzer, NewsAggregator
from src.features import FeaturePipeline
from src.models import StackedEnsemble
from src.portfolio import RegimeAwareAllocator
from src.risk import RiskEngine, PositionSizer, PositionSizingConfig, RiskConfig
from src.execution import SmartOrderRouter, RoutingConfig
from src.broker import (
    AlpacaBroker, OrderManager, OrderManagerConfig,
    PositionTracker, PositionTrackerConfig,
    AccountMonitor, AccountMonitorConfig
)
from src.training import ModelPersistence
from src.universe.dynamic_discovery import DynamicSymbolDiscovery

# Configure logging
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


class TradingMode:
    """Trading mode enumeration"""
    FULL_AUTO = "full_auto"
    SEMI_AUTO = "semi_auto"
    PAPER = "paper"
    BACKTEST = "backtest"


class QuantumSentimentBot:
    """Main trading bot orchestrator"""
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        mode: str = TradingMode.PAPER,
        config: Optional[Config] = None,
        broker: Optional[Any] = None
    ):
        """
        Initialize the trading bot
        
        Args:
            config_path: Path to configuration file (ignored if config provided)
            mode: Trading mode (full_auto, semi_auto, paper, backtest)
            config: Pre-loaded Config object (for backtesting)
            broker: Pre-initialized broker (for backtesting with SimulatedBroker)
        """
        
        self.mode = mode
        self.config = config or Config(config_path)
        self.is_running = False
        self._injected_broker = broker  # Store injected broker
        
        # Core components
        self.config_manager = ConfigManager(self.config)
        self.db_manager = None
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.feature_pipeline = None
        self.ensemble_model = None
        self.portfolio_optimizer = None
        self.risk_engine = None
        self.position_sizer = None
        self.execution_router = None
        self.broker = None
        self.dynamic_discovery = None
        
        # Monitoring components
        self.order_manager = None
        self.position_tracker = None
        self.account_monitor = None
        
        # Trading state
        self.current_positions = {}
        self.pending_signals = []
        self.last_prediction_time = None
        
        logger.info("QuantumSentiment Bot initialized",
                   mode=mode,
                   config_path=config_path)
    
    async def initialize(self) -> bool:
        """
        Initialize all components and verify connectivity
        
        Returns:
            True if initialization successful
        """
        
        try:
            logger.info("Starting system initialization...")
            
            # 1. Initialize database
            logger.info("Initializing database...")
            self.db_manager = DatabaseManager(self.config.database.connection_string)
            await self.db_manager.initialize()
            
            # 2. Initialize data fetcher
            logger.info("Initializing data fetcher...")
            self.data_fetcher = DataFetcher(self.config, self.db_manager)
            
            # 3. Initialize sentiment analysis
            logger.info("Initializing sentiment analysis...")
            
            # Create Reddit config from environment variables and main config
            from src.sentiment.reddit_analyzer import RedditConfig
            reddit_config = RedditConfig(
                client_id=os.getenv('REDDIT_CLIENT_ID', ''),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'QuantumSentiment/1.0'),
                subreddits=self.config.data_sources.reddit.subreddits
            )
            reddit_analyzer = RedditSentimentAnalyzer(reddit_config)
            await reddit_analyzer.initialize()  # Initialize Reddit API connection
            
            # Create News config from environment variables
            from src.sentiment.news_aggregator import NewsConfig
            news_config = NewsConfig(
                alpha_vantage_key=os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                newsapi_key=os.getenv('NEWSAPI_KEY', '')
            )
            news_aggregator = NewsAggregator(news_config)
            news_aggregator.initialize()  # Initialize News aggregator
            
            # Store analyzers for sentiment fusion
            self.reddit_analyzer = reddit_analyzer
            self.news_aggregator = news_aggregator
            
            # Initialize SentimentFusion
            from src.sentiment.sentiment_fusion import SentimentFusion, FusionConfig
            fusion_config = FusionConfig()
            self.sentiment_fusion = SentimentFusion(fusion_config)
            
            # Create a wrapper that combines analyzers and fusion
            class SentimentManager:
                def __init__(self, reddit_analyzer, news_aggregator, sentiment_fusion):
                    self.reddit_analyzer = reddit_analyzer
                    self.news_aggregator = news_aggregator  
                    self.sentiment_fusion = sentiment_fusion
                
                async def fuse_sentiment(self, symbol):
                    """Get sentiment from all sources and fuse them"""
                    sentiment_data = {}
                    
                    # Get Reddit sentiment
                    try:
                        reddit_result = await self.reddit_analyzer.analyze_symbol(symbol)
                        if reddit_result:
                            sentiment_data['reddit'] = reddit_result
                    except Exception as e:
                        logger.warning(f"Reddit sentiment failed for {symbol}: {e}")
                    
                    # Get News sentiment  
                    try:
                        news_result = self.news_aggregator.analyze_symbol(symbol)
                        if news_result:
                            sentiment_data['news'] = news_result
                    except Exception as e:
                        logger.warning(f"News sentiment failed for {symbol}: {e}")
                    
                    # Fuse sentiments if we have any data
                    if sentiment_data:
                        fused_result = self.sentiment_fusion.fuse_sentiment(sentiment_data, symbol)
                        
                        # Convert to expected format
                        class FusedSentimentResult:
                            def __init__(self, fused_data):
                                self.score = fused_data.get('sentiment_score', 0.0)
                                self.confidence = fused_data.get('confidence', 0.0)
                                self.sources = list(sentiment_data.keys())
                                self.timestamp = datetime.now()
                        
                        return FusedSentimentResult(fused_result)
                    else:
                        # Return neutral sentiment if no data
                        class NeutralSentimentResult:
                            def __init__(self):
                                self.score = 0.0
                                self.confidence = 0.0
                                self.sources = []
                                self.timestamp = datetime.now()
                        
                        return NeutralSentimentResult()
                
                async def get_aggregated_sentiment(self, symbols):
                    """For backward compatibility with connectivity check"""
                    result = await self.fuse_sentiment(symbols[0] if symbols else 'AAPL')
                    return {
                        'sentiment_score': result.score,
                        'confidence': result.confidence,
                        'sources': result.sources,
                        'timestamp': result.timestamp
                    }
            
            self.sentiment_analyzer = SentimentManager(reddit_analyzer, news_aggregator, self.sentiment_fusion)
            
            # 4. Initialize feature pipeline
            logger.info("Initializing feature pipeline...")
            from src.features.feature_pipeline import FeatureConfig
            feature_config = FeatureConfig(
                enable_technical=True,
                enable_sentiment=True,
                enable_fundamental=True,
                enable_market_structure=True
            )
            self.feature_pipeline = FeaturePipeline(feature_config, self.db_manager)
            
            # 5. Load trained models
            logger.info("Loading trained models...")
            from src.training.model_persistence import PersistenceConfig
            from pathlib import Path
            persistence_config = PersistenceConfig(
                model_registry_path=Path(self.config.paths.models),
                use_model_registry=True
            )
            model_persistence = ModelPersistence(persistence_config)
            self.ensemble_model = await self._load_ensemble_model(model_persistence)
            
            # 6. Initialize portfolio optimizer
            logger.info("Initializing portfolio optimizer...")
            from src.portfolio.regime_allocator import RegimeConfig
            regime_config = RegimeConfig()
            self.portfolio_optimizer = RegimeAwareAllocator(regime_config)
            
            # 7. Initialize risk engine
            logger.info("Initializing risk engine...")
            # Create RiskConfig from main config
            risk_config = RiskConfig(
                max_portfolio_var=getattr(self.config.risk, 'max_portfolio_var', 0.02),
                max_position_weight=getattr(self.config.risk, 'max_position_weight', 0.1),
                max_total_drawdown=getattr(self.config.risk, 'max_drawdown', 0.2),
                max_daily_drawdown=getattr(self.config.risk, 'max_daily_drawdown', 0.02),
                max_leverage=getattr(self.config.risk, 'max_leverage', 1.0),
                enable_circuit_breakers=True
            )
            self.risk_engine = RiskEngine(risk_config)
            
            # 8. Initialize position sizer
            logger.info("Initializing position sizer...")
            position_sizing_config = PositionSizingConfig(
                use_kelly_criterion=True,
                kelly_fraction=0.25,
                max_position_size=0.1,
                min_position_size=0.005,
                enable_confidence_scaling=True,
                enable_volatility_scaling=True,
                enable_momentum_scaling=True
            )
            self.position_sizer = PositionSizer(position_sizing_config)
            
            # 9. Initialize execution router
            logger.info("Initializing execution router...")
            routing_config = RoutingConfig(
                enable_dynamic_strategy_selection=True,
                enable_multi_venue_routing=True,
                enable_order_fragmentation=True
            )
            self.execution_router = SmartOrderRouter(routing_config)
            
            # 10. Initialize broker components
            logger.info("Initializing broker components...")
            await self._initialize_broker()
            
            # 11. Initialize dynamic symbol discovery
            logger.info("Initializing dynamic symbol discovery...")
            await self._initialize_dynamic_discovery()
            
            # 12. Verify connectivity
            logger.info("Verifying system connectivity...")
            if not await self._verify_connectivity():
                raise RuntimeError("System connectivity check failed")
            
            # 13. Log strategy configuration
            self._log_strategy_configuration()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error("System initialization failed", error=str(e))
            import traceback
            traceback.print_exc()
            return False
    
    async def _initialize_broker(self) -> None:
        """Initialize broker and related components"""
        
        # Use injected broker if provided (for backtesting)
        if self._injected_broker:
            self.broker = self._injected_broker
            logger.info("Using injected broker for backtesting")
            
            # Connect to simulated broker
            if not await self.broker.connect():
                raise RuntimeError("Failed to connect to simulated broker")
            return
        
        # Initialize order manager
        order_config = OrderManagerConfig(
            max_orders_per_symbol=10,
            max_total_orders=100,
            enable_order_monitoring=True
        )
        self.order_manager = OrderManager(order_config)
        
        # Initialize position tracker
        position_config = PositionTrackerConfig(
            max_positions=50,
            enable_real_time_pnl=True
        )
        self.position_tracker = PositionTracker(position_config)
        
        # Initialize account monitor
        account_config = AccountMonitorConfig(
            snapshot_interval_seconds=30,
            max_drawdown_threshold=0.1,  # 10%
            enable_email_alerts=False
        )
        self.account_monitor = AccountMonitor(account_config)
        
        # Initialize Alpaca broker
        paper_trading = self.mode in [TradingMode.PAPER, TradingMode.SEMI_AUTO]
        self.broker = AlpacaBroker(
            paper_trading=paper_trading,
            order_manager=self.order_manager,
            position_tracker=self.position_tracker,
            account_monitor=self.account_monitor
        )
        
        # Connect to broker
        if not await self.broker.connect():
            raise RuntimeError("Failed to connect to broker")
        
        # Start monitoring
        await self.broker.start_monitoring()
        
        # Sync current state
        await self.broker.full_sync()
    
    async def _initialize_dynamic_discovery(self) -> None:
        """Initialize dynamic symbol discovery system"""
        try:
            self.dynamic_discovery = DynamicSymbolDiscovery(self.config_manager)
            await self.dynamic_discovery.initialize()  # Initialize async components
            logger.info("Dynamic symbol discovery initialized", 
                       enabled=self.dynamic_discovery.config.enabled)
        except Exception as e:
            logger.warning("Dynamic discovery initialization failed", error=str(e))
            self.dynamic_discovery = None
    
    async def _load_ensemble_model(self, persistence: ModelPersistence) -> Any:
        """Load the trained ensemble model or best individual model"""
        
        try:
            # First try to load the complete StackedEnsemble
            ensemble, metadata = persistence.load_model('StackedEnsemble')
            logger.info("Loaded trained StackedEnsemble successfully", 
                       version=metadata.version,
                       metrics=metadata.metrics)
            return ensemble
        except Exception as e:
            logger.warning(f"Could not load StackedEnsemble: {e}")
            
        # Fallback: Try to load individual models
        model_names = ['MarketRegimeXGBoost', 'XGBoost', 'xgboost_model']
        for model_name in model_names:
            try:
                model, metadata = persistence.load_model(model_name)
                logger.info(f"Loaded {model_name} model as fallback", 
                           version=metadata.version if hasattr(metadata, 'version') else 'unknown',
                           metrics=metadata.metrics if hasattr(metadata, 'metrics') else {})
                
                # Wrap in a simple interface that matches ensemble expectations
                class XGBoostWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.is_trained = True
                        self.model_type = type(model).__name__
                    
                    def predict(self, features):
                        # Handle different model types
                        try:
                            if self.model_type == 'MarketRegimeXGBoost':
                                # This model expects OHLCV data, not features
                                # We need to use a different approach - try to use the model's internal prediction
                                # if it has a trained XGBoost model inside
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
                                    # Try to predict using the internal XGBoost model with features
                                    pred = self.model.model.predict(features.values)
                                else:
                                    # Fallback: return neutral signal
                                    logger.warning(f"Cannot predict with {self.model_type} - returning neutral")
                                    return 0.0
                            else:
                                # Regular XGBoost model
                                pred = self.model.predict(features)
                            
                            # Convert class predictions to signal strength
                            # Classes: 0=Strong Down, 1=Down, 2=Neutral, 3=Up, 4=Strong Up
                            signal_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
                            if hasattr(pred, '__iter__'):
                                return np.array([signal_map.get(int(p), 0.0) for p in pred])
                            else:
                                return signal_map.get(int(pred), 0.0)
                        except Exception as e:
                            logger.warning(f"Prediction failed with {self.model_type}: {e} - returning neutral")
                            return 0.0
                    
                    def predict_proba(self, features):
                        try:
                            if self.model_type == 'MarketRegimeXGBoost':
                                # Try to use internal model for probabilities
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict_proba'):
                                    return self.model.model.predict_proba(features.values)
                                else:
                                    # Fallback: return neutral probabilities
                                    return np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
                            else:
                                if hasattr(self.model, 'predict_proba'):
                                    return self.model.predict_proba(features)
                        except Exception as e:
                            logger.warning(f"Probability prediction failed: {e}")
                            
                        # Final fallback
                        pred = self.predict(features)
                        if hasattr(pred, '__iter__'):
                            # Multi-class case - return uniform probabilities
                            return np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
                        else:
                            # Binary case
                            return np.array([[1-abs(pred), abs(pred)]])
                
                return XGBoostWrapper(model)
            except Exception as e:
                logger.debug(f"Could not load {model_name}: {e}")
                continue
        
        # If we reach here, no models were loaded
        logger.error("CRITICAL: No trained models available - cannot trade!")
        # Return a dummy model that refuses to trade
        class NoModelAvailable:
            def __init__(self):
                self.is_trained = False
            
            def predict(self, features):
                raise RuntimeError("No trained model available")
        
        return NoModelAvailable()
    
    async def _verify_connectivity(self) -> bool:
        """Verify all external connections"""
        
        checks = {
            "database": self.db_manager is not None,
            "broker": self.broker.is_connected,
            "market_data": await self._check_market_data(),
            "sentiment_sources": await self._check_sentiment_sources()
        }
        
        failed_checks = [name for name, status in checks.items() if not status]
        
        if failed_checks:
            logger.error("Connectivity checks failed", failed=failed_checks)
            return False
        
        logger.info("All connectivity checks passed")
        return True
    
    async def _check_market_data(self) -> bool:
        """Check market data connectivity"""
        try:
            # Test with SPY quote
            quote = await self.broker.get_latest_quote("SPY")
            return quote is not None
        except Exception as e:
            logger.error("Market data check failed", error=str(e))
            return False
    
    async def _check_sentiment_sources(self) -> bool:
        """Check sentiment data sources"""
        try:
            # Test sentiment analysis on a sample ticker
            sentiment = await self.sentiment_analyzer.get_aggregated_sentiment(["AAPL"])
            return sentiment is not None
        except Exception as e:
            logger.error("Sentiment source check failed", error=str(e))
            return False
    
    async def run(self) -> None:
        """
        Main trading loop
        """
        
        if not await self.initialize():
            logger.error("Failed to initialize system")
            return
        
        self.is_running = True
        logger.info(f"🚀 STARTING MAIN TRADING LOOP - Mode: {self.mode}")
        logger.info(f"🏪 Market is open: {self.broker.is_market_open()}")
        
        try:
            while self.is_running:
                # Check if market is open (skip for paper trading in testing)
                if not self.broker.is_market_open() and self.mode not in [TradingMode.BACKTEST, TradingMode.PAPER]:
                    logger.info("Market is closed, waiting...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # For paper trading, log market status but continue
                if self.mode == TradingMode.PAPER and not self.broker.is_market_open():
                    logger.info("Market is closed but running in paper mode for testing")
                
                # Run trading cycle
                await self._trading_cycle()
                
                # Sleep based on mode
                if self.mode == TradingMode.BACKTEST:
                    await asyncio.sleep(0.1)  # Fast for backtesting
                else:
                    await asyncio.sleep(60)  # Check every minute
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error("Critical error in main loop", error=str(e))
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self) -> None:
        """Execute one trading cycle"""
        
        cycle_start_time = datetime.now()
        logger.info(f"🔄 TRADING CYCLE START - {cycle_start_time.strftime('%H:%M:%S')}")
        
        try:
            # 1. Update market data
            logger.info("📊 Updating market data...")
            await self._update_market_data()
            
            # 2. Generate predictions (every 5 minutes)
            now = datetime.now()
            if (self.last_prediction_time is None or 
                (now - self.last_prediction_time) > timedelta(minutes=5)):
                
                await self._generate_predictions()
                self.last_prediction_time = now
            else:
                logger.info("⏭️ Skipping predictions (generated within last 5 minutes)")
            
            # 3. Check comprehensive risk limits
            logger.info("⚠️ Performing comprehensive risk assessment...")
            if not await self._check_risk_limits():
                logger.warning("🛑 Risk limits exceeded, trading halted for this cycle")
                return
            
            # 4. Execute pending signals
            await self._execute_signals()
            
            # 5. Monitor positions
            logger.info("👀 Monitoring positions...")
            await self._monitor_positions()
            
            # 6. Update performance metrics
            logger.info("📊 Updating performance metrics...")
            await self._update_metrics()
            
            # Cycle summary
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            positions = self.position_tracker.get_all_positions()
            active_positions = [p for p in positions if not p.is_flat]
            logger.info(f"✅ CYCLE COMPLETE ({cycle_duration:.1f}s) - {len(active_positions)} active positions")
            
        except Exception as e:
            logger.error("Error in trading cycle", error=str(e))
    
    async def _trading_cycle_backtest(self, symbols: List[str], timestamp: datetime) -> None:
        """
        Execute one trading cycle for backtesting - event-driven, no sleep
        
        Args:
            symbols: List of symbols to trade
            timestamp: Current simulation timestamp
        """
        
        logger.debug(f"🔄 BACKTEST CYCLE - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Generate predictions for current timestamp
            if symbols:
                await self._generate_predictions_backtest(symbols, timestamp)
            
            # 2. Check comprehensive risk limits
            if not await self._check_risk_limits():
                logger.debug("🛑 Risk limits exceeded, trading halted for this cycle")
                return
            
            # 3. Execute pending signals
            await self._execute_signals()
            
            # 4. Monitor positions (position tracking handled by simulated broker)
            await self._monitor_positions()
            
        except Exception as e:
            logger.error("Error in backtest trading cycle", 
                        error=str(e), timestamp=timestamp)
    
    async def _generate_predictions_backtest(
        self, 
        symbols: List[str], 
        timestamp: datetime
    ) -> None:
        """
        Generate predictions for backtesting at specific timestamp
        
        Args:
            symbols: List of symbols to analyze
            timestamp: Current simulation timestamp
        """
        
        logger.debug("🧠 Generating backtest predictions...")
        
        # Get market data for all symbols up to current timestamp
        for symbol in symbols:
            try:
                # Get market data up to current timestamp
                bars = await self.broker.get_bars(
                    symbol,
                    timeframe="1Hour",
                    end=timestamp,
                    limit=500
                )
                
                if bars is None or bars.empty:
                    logger.debug(f"⚠️ No market data for {symbol} at {timestamp}")
                    continue
                
                # Ensure we have enough data
                if len(bars) < 50:  # Need minimum bars for technical indicators
                    logger.debug(f"⚠️ Insufficient data for {symbol} ({len(bars)} bars)")
                    continue
                
                logger.debug(f"🔍 Analyzing {symbol} with {len(bars)} bars...")
                
                # Get sentiment data (simplified for backtesting)
                sentiment_dict = {
                    'sentiment_score': 0.0,  # Neutral sentiment for backtesting
                    'confidence': 0.5,
                    'sources': ['backtest'],
                    'timestamp': timestamp
                }
                
                sentiment_df = pd.DataFrame([sentiment_dict])
                sentiment_df.set_index('timestamp', inplace=True)
                
                # Generate features
                feature_result = self.feature_pipeline.generate_features(
                    symbol=symbol,
                    market_data=bars,
                    sentiment_data=sentiment_df
                )
                
                features_dict = feature_result.get('features', {})
                if not features_dict:
                    continue
                
                # Convert to DataFrame for model prediction
                import pandas as pd
                features = pd.DataFrame([features_dict])
                
                # Get prediction from model
                try:
                    if hasattr(self.ensemble_model, 'predict'):
                        raw_prediction = self.ensemble_model.predict(features)
                        
                        if isinstance(raw_prediction, np.ndarray) and len(raw_prediction) > 0:
                            signal_strength = float(raw_prediction[0])
                            confidence = 0.7  # Default confidence
                        else:
                            signal_strength = float(raw_prediction)
                            confidence = 0.7
                    else:
                        logger.warning(f"Model has no predict method for {symbol}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"Prediction failed for {symbol}: {e}")
                    continue
                
                # Create trading signal
                signal = {
                    'symbol': symbol,
                    'signal': 'buy' if signal_strength > 0 else 'sell',
                    'strength': abs(signal_strength),
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'features': features.iloc[-1].to_dict(),
                    'sentiment_data': sentiment_dict,
                    'technical_indicators': self._extract_technical_indicators(features)
                }
                
                # Validate signal
                is_valid = await self._validate_signal_by_strategy(signal)
                if is_valid:
                    self.pending_signals.append(signal)
                    logger.debug(f"✅ BACKTEST SIGNAL: {signal['signal'].upper()} {symbol} (strength: {signal['strength']:.3f})")
                
            except Exception as e:
                logger.debug(f"Failed to generate backtest prediction for {symbol}: {e}")
    
    async def _update_market_data(self) -> None:
        """Update market data for all tracked symbols"""
        
        # Get current positions
        positions = self.position_tracker.get_all_positions()
        symbols = [pos.symbol for pos in positions if not pos.is_flat]
        
        # Add watchlist symbols
        watchlist = self.config.trading.watchlist
        all_symbols = list(set(symbols + watchlist))
        
        if all_symbols:
            # Update prices in position tracker
            prices = await self.broker.get_latest_prices(all_symbols)
            await self.position_tracker.update_all_market_prices(prices)
    
    async def _generate_predictions(self) -> None:
        """Generate trading signals using the ensemble model"""
        
        logger.info("🧠 GENERATING PREDICTIONS...")
        
        # Get universe of stocks to analyze
        universe = await self._get_trading_universe()
        logger.info(f"📋 Analyzing {len(universe)} symbols: {universe}")
        
        predictions_made = 0
        signals_generated = 0
        
        # Batch fetch market data for all symbols
        logger.info("📊 Fetching market data for all symbols...")
        market_data_tasks = []
        for symbol in universe:
            task = self.broker.get_bars(
                symbol,
                timeframe="1Hour",
                limit=500
            )
            market_data_tasks.append((symbol, task))
        
        # Execute all data fetches in parallel
        market_data_results = []
        for symbol, task in market_data_tasks:
            try:
                bars = await task
                if not bars.empty:
                    market_data_results.append((symbol, bars))
                else:
                    logger.warning(f"⚠️ No market data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        logger.info(f"✅ Fetched data for {len(market_data_results)}/{len(universe)} symbols")
        
        # Process each symbol with its data
        for symbol, bars in market_data_results:
            try:
                logger.info(f"🔍 Analyzing {symbol}...")
                logger.info(f"📊 Got {len(bars)} bars for {symbol}, latest price: ${bars.iloc[-1]['close']:.2f}")
                
                # 2. Get sentiment data using real fusion
                try:
                    sentiment_result = await self.sentiment_analyzer.fuse_sentiment(symbol)
                    
                    # Convert FusedSentiment object to DataFrame format expected by feature pipeline
                    if sentiment_result:
                        import pandas as pd
                        sentiment_df = pd.DataFrame([{
                            'timestamp': sentiment_result.timestamp,
                            'sentiment_score': sentiment_result.score,
                            'confidence': sentiment_result.confidence,
                            'source': ','.join(sentiment_result.sources)
                        }])
                        sentiment_df.set_index('timestamp', inplace=True)
                        
                        # Also keep the raw result for signal generation
                        sentiment_dict = {
                            'sentiment_score': sentiment_result.score,
                            'confidence': sentiment_result.confidence,
                            'sources': sentiment_result.sources,
                            'timestamp': sentiment_result.timestamp
                        }
                    else:
                        sentiment_df = None
                        sentiment_dict = None
                except Exception as e:
                    logger.warning(f"Failed to get sentiment for {symbol}: {e}")
                    sentiment_df = None
                    sentiment_dict = None
                
                # 3. Generate features
                feature_result = self.feature_pipeline.generate_features(
                    symbol=symbol,
                    market_data=bars,
                    sentiment_data=sentiment_df
                )
                
                # Extract features dictionary and convert to DataFrame for compatibility
                features_dict = feature_result.get('features', {})
                if not features_dict:
                    continue
                    
                # Convert features dict to DataFrame with single row
                import pandas as pd
                features = pd.DataFrame([features_dict])
                
                # 4. Get ensemble prediction
                try:
                    if hasattr(self.ensemble_model, 'is_trained') and self.ensemble_model.is_trained:
                        # Ensure features are in correct format
                        if isinstance(features, pd.DataFrame) and len(features) > 0:
                            raw_prediction = self.ensemble_model.predict(features)
                            
                            # For 5-class XGBoost, get probability distribution
                            if hasattr(self.ensemble_model, 'predict_proba'):
                                probas = self.ensemble_model.predict_proba(features)
                                # Use probability-weighted signal strength
                                if probas.shape[1] == 5:  # 5-class model
                                    # Classes: 0=Strong Down, 1=Down, 2=Neutral, 3=Up, 4=Strong Up
                                    weights = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                                    signal_strength = float(np.dot(probas[0], weights))
                                    # Confidence based on max probability
                                    confidence = float(np.max(probas[0]))
                                else:
                                    # Fallback for other probability shapes
                                    signal_strength = float(raw_prediction[0]) if isinstance(raw_prediction, np.ndarray) else float(raw_prediction)
                                    confidence = 0.7
                            else:
                                # No probability method, use raw prediction
                                signal_strength = float(raw_prediction[0]) if isinstance(raw_prediction, np.ndarray) else float(raw_prediction)
                                confidence = 0.7
                        else:
                            logger.warning(f"Invalid features format for {symbol}")
                            continue
                    else:
                        # Model not trained yet, log error and refuse to trade
                        logger.error(f"Model not trained - cannot generate predictions for {symbol}")
                        continue
                        
                    prediction = {
                        'signal_strength': signal_strength,
                        'confidence': confidence
                    }
                except Exception as e:
                    logger.warning("Ensemble prediction failed, using fallback", 
                                 symbol=symbol, error=str(e))
                    prediction = {
                        'signal_strength': 0.0,
                        'confidence': 0.0
                    }
                
                signal_strength = prediction.get('signal_strength', 0)
                confidence = prediction.get('confidence', 0)
                predictions_made += 1
                
                logger.info(f"🎯 PREDICTION for {symbol}: strength={signal_strength:.3f}, confidence={confidence:.3f}")
                
                # 5. Create trading signal with enhanced data
                signal = {
                    'symbol': symbol,
                    'signal': 'buy' if signal_strength > 0 else 'sell',
                    'strength': abs(signal_strength),
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'features': features.iloc[-1].to_dict(),
                    'sentiment_data': sentiment_dict,
                    'technical_indicators': self._extract_technical_indicators(features)
                }
                
                # Validate signal using strategy-specific rules
                is_valid = await self._validate_signal_by_strategy(signal)
                if is_valid:
                    self.pending_signals.append(signal)
                    signals_generated += 1
                    logger.info(f"✅ SIGNAL GENERATED: {signal['signal'].upper()} {symbol} (strength: {signal['strength']:.3f})")
                else:
                    logger.info(f"❌ Signal validation failed for {symbol} (strength: {abs(signal_strength):.3f})")
                
            except Exception as e:
                logger.error("Failed to generate prediction",
                           symbol=symbol,
                           error=str(e))
        
        # Summary of prediction generation
        logger.info(f"📊 PREDICTION SUMMARY: {predictions_made} predictions made, {signals_generated} signals generated, {len(self.pending_signals)} total pending")
    
    async def _execute_signals(self) -> None:
        """Execute pending trading signals"""
        
        if not self.pending_signals:
            logger.info("📈 No pending signals to execute")
            return
        
        logger.info(f"💼 EXECUTING SIGNALS: {len(self.pending_signals)} pending signals")
        
        # Sort by strength
        self.pending_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        for signal in self.pending_signals[:]:
            try:
                # Check if we should execute
                should_execute = await self._should_execute_signal(signal)
                
                if not should_execute:
                    continue
                
                # Semi-auto mode: get user approval
                if self.mode == TradingMode.SEMI_AUTO:
                    if not await self._get_user_approval(signal):
                        continue
                
                # Calculate position size
                position_size = await self._calculate_position_size(signal)
                
                if position_size > 0:
                    # Execute order
                    order_id = await self._execute_order(
                        signal['symbol'],
                        signal['signal'],
                        position_size,
                        signal
                    )
                    
                    if order_id:
                        logger.info("Order executed",
                                   order_id=order_id,
                                   symbol=signal['symbol'],
                                   side=signal['signal'],
                                   quantity=position_size)
                        
                        # Remove from pending
                        self.pending_signals.remove(signal)
                
            except Exception as e:
                logger.error("Failed to execute signal",
                           symbol=signal['symbol'],
                           error=str(e))
    
    async def _monitor_positions(self) -> None:
        """Monitor existing positions with comprehensive risk management"""
        
        positions = self.position_tracker.get_all_positions()
        active_positions = [p for p in positions if not p.is_flat]
        
        if not active_positions:
            logger.debug("No active positions to monitor")
            return
        
        logger.info(f"Monitoring {len(active_positions)} active positions")
        
        # Perform position-level risk monitoring
        for position in active_positions:
            try:
                # Enhanced stop loss check using RiskEngine
                if hasattr(self.risk_engine, 'check_stop_loss'):
                    try:
                        if await self.risk_engine.check_stop_loss(position):
                            logger.info(f"Stop loss triggered for {position.symbol}")
                            await self._close_position(position, "stop_loss")
                            continue
                    except Exception as e:
                        logger.debug(f"Risk engine stop loss check failed for {position.symbol}: {e}")
                        # Fallback to basic stop loss check
                        if await self._basic_stop_loss_check(position):
                            await self._close_position(position, "basic_stop_loss")
                            continue
                else:
                    # Basic stop loss if RiskEngine doesn't have the method
                    if await self._basic_stop_loss_check(position):
                        await self._close_position(position, "basic_stop_loss")
                        continue
                
                # Enhanced take profit check using RiskEngine
                if hasattr(self.risk_engine, 'check_take_profit'):
                    try:
                        if await self.risk_engine.check_take_profit(position):
                            logger.info(f"Take profit triggered for {position.symbol}")
                            await self._close_position(position, "take_profit")
                            continue
                    except Exception as e:
                        logger.debug(f"Risk engine take profit check failed for {position.symbol}: {e}")
                        # Fallback to basic take profit check
                        if await self._basic_take_profit_check(position):
                            await self._close_position(position, "basic_take_profit")
                            continue
                else:
                    # Basic take profit if RiskEngine doesn't have the method
                    if await self._basic_take_profit_check(position):
                        await self._close_position(position, "basic_take_profit")
                        continue
                
                # Position-specific risk monitoring
                position_risk = await self._assess_position_risk(position)
                if position_risk.get('critical_risk', False):
                    logger.warning(f"Critical risk detected for {position.symbol}",
                                 risk_factors=position_risk.get('risk_factors', []))
                    await self._close_position(position, "risk_management")
                    continue
                
                # Check for exit signals (strategy-based)
                if await self._should_exit_position(position):
                    await self._close_position(position, "exit_signal")
                    continue
                
                # Log position health
                logger.debug(f"Position {position.symbol} monitored - Status: OK",
                           pnl=position.unrealized_pnl,
                           risk_score=position_risk.get('risk_score', 0))
                
            except Exception as e:
                logger.error("Error monitoring position",
                           symbol=position.symbol,
                           error=str(e))
        
        # Portfolio-level risk monitoring (every 10 cycles to avoid over-computation)
        if hasattr(self, '_monitoring_cycle_count'):
            self._monitoring_cycle_count += 1
        else:
            self._monitoring_cycle_count = 1
        
        if self._monitoring_cycle_count % 10 == 0:
            await self._portfolio_risk_monitoring()
    
    async def _basic_stop_loss_check(self, position) -> bool:
        """Basic stop loss check when RiskEngine method not available
        
        Implements simple percentage-based stop loss logic.
        
        Args:
            position: Position object to check
            
        Returns:
            bool: True if stop loss is triggered
        """
        
        try:
            # Simple percentage-based stop loss
            stop_loss_pct = getattr(self.config.risk, 'stop_loss_pct', 0.05)  # 5% default
            
            if position.is_long:
                loss_pct = (position.average_price - position.current_price) / position.average_price
            else:
                loss_pct = (position.current_price - position.average_price) / position.average_price
            
            return loss_pct > stop_loss_pct
            
        except Exception as e:
            logger.error(f"Error in basic stop loss check: {e}")
            return False
    
    async def _basic_take_profit_check(self, position) -> bool:
        """Basic take profit check when RiskEngine method not available
        
        Implements simple percentage-based take profit logic.
        
        Args:
            position: Position object to check
            
        Returns:
            bool: True if take profit is triggered
        """
        
        try:
            # Simple percentage-based take profit
            take_profit_pct = getattr(self.config.risk, 'take_profit_pct', 0.15)  # 15% default
            
            if position.is_long:
                profit_pct = (position.current_price - position.average_price) / position.average_price
            else:
                profit_pct = (position.average_price - position.current_price) / position.average_price
            
            return profit_pct > take_profit_pct
            
        except Exception as e:
            logger.error(f"Error in basic take profit check: {e}")
            return False
    
    async def _assess_position_risk(self, position) -> Dict[str, Any]:
        """Assess risk for individual position
        
        Evaluates position-specific risk factors including size, P&L, age, and volatility.
        
        Args:
            position: Position object to assess
            
        Returns:
            Dict containing risk_score, risk_factors, and critical_risk flag
        """
        
        try:
            risk_factors = []
            risk_score = 0
            
            # Check position size relative to portfolio
            account = await self.broker.get_account()
            equity = float(account.equity)
            position_value = abs(position.quantity * position.current_price)
            position_weight = position_value / equity if equity > 0 else 0
            
            if position_weight > 0.15:  # 15% position limit
                risk_factors.append("oversized_position")
                risk_score += 30
            
            # Check unrealized P&L
            pnl_pct = position.unrealized_pnl / position_value if position_value > 0 else 0
            if pnl_pct < -0.1:  # 10% loss
                risk_factors.append("large_unrealized_loss")
                risk_score += 25
            
            # Check position age
            if position.opened_at:
                position_age = datetime.now() - position.opened_at
                if position_age > timedelta(days=7):  # Long-held positions
                    risk_factors.append("stale_position")
                    risk_score += 10
            
            # Check volatility (if we can get recent data)
            try:
                bars = await self.broker.get_bars(
                    position.symbol,
                    timeframe="1Hour",
                    limit=24
                )
                
                if not bars.empty:
                    returns = bars['close'].pct_change().dropna()
                    volatility = returns.std()
                    
                    if volatility > 0.03:  # 3% hourly volatility threshold
                        risk_factors.append("high_volatility")
                        risk_score += 15
            except Exception:
                pass  # Skip volatility check if data unavailable
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'critical_risk': risk_score > 50,  # Critical if score > 50
                'position_weight': position_weight,
                'pnl_percent': pnl_pct
            }
            
        except Exception as e:
            logger.error(f"Error assessing position risk for {position.symbol}: {e}")
            return {'risk_score': 0, 'risk_factors': [], 'critical_risk': False}
    
    async def _portfolio_risk_monitoring(self) -> None:
        """Periodic comprehensive portfolio risk monitoring
        
        Performs portfolio-level risk checks including leverage, concentration,
        and position correlations. Called every 10 monitoring cycles.
        """
        
        try:
            positions = self.position_tracker.get_all_positions()
            active_positions = [p for p in positions if not p.is_flat]
            
            if not active_positions:
                return
            
            # Get account info
            account = await self.broker.get_account()
            equity = float(account.equity)
            
            # Calculate portfolio metrics
            total_exposure = sum(abs(p.quantity * p.current_price) for p in active_positions)
            leverage = total_exposure / equity if equity > 0 else 0
            
            # Portfolio concentration check
            position_weights = [(abs(p.quantity * p.current_price) / equity, p.symbol) 
                              for p in active_positions]
            max_weight, max_symbol = max(position_weights) if position_weights else (0, "")
            
            # Log portfolio risk summary
            logger.info("Portfolio risk monitoring summary",
                       active_positions=len(active_positions),
                       total_leverage=f"{leverage:.2f}x",
                       max_position_weight=f"{max_weight:.1%}",
                       max_position_symbol=max_symbol)
            
            # Warnings for concerning metrics
            if leverage > 0.8:  # 80% leverage warning
                logger.warning("High portfolio leverage detected", leverage=f"{leverage:.2f}x")
            
            if max_weight > 0.2:  # 20% concentration warning
                logger.warning("High position concentration", 
                             symbol=max_symbol, weight=f"{max_weight:.1%}")
            
        except Exception as e:
            logger.error(f"Error in portfolio risk monitoring: {e}")
    
    async def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        signal: Dict[str, Any]
    ) -> Optional[str]:
        """Execute order through smart router"""
        
        try:
            # Use smart router for execution
            execution_plan = await self.execution_router.create_execution_plan(
                order_id=f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                urgency="normal"
            )
            
            # Execute through broker
            alpaca_order = await self.broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market",  # Could be enhanced based on execution plan
                client_order_id=execution_plan.order_id
            )
            
            return alpaca_order.id if alpaca_order else None
            
        except Exception as e:
            logger.error("Order execution failed",
                        symbol=symbol,
                        side=side,
                        error=str(e))
            return None
    
    async def _close_position(self, position, reason: str) -> None:
        """Close a position"""
        
        side = "sell" if position.is_long else "buy"
        quantity = abs(position.quantity)
        
        order_id = await self._execute_order(
            position.symbol,
            side,
            quantity,
            {"reason": reason}
        )
        
        if order_id:
            logger.info("Position closed",
                       symbol=position.symbol,
                       reason=reason,
                       quantity=quantity)
    
    async def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal against risk rules"""
        
        # Check position limits
        current_positions = len([p for p in self.position_tracker.get_all_positions() if not p.is_flat])
        if current_positions >= self.config.trading.max_positions:
            return False
        
        # Check concentration limits
        if signal['symbol'] in [p.symbol for p in self.position_tracker.get_all_positions()]:
            return False  # Already have position
        
        return True
    
    async def _should_execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Determine if signal should be executed"""
        
        # Check signal age
        signal_age = datetime.now() - signal['timestamp']
        if signal_age > timedelta(minutes=10):
            return False  # Too old
        
        # Check market conditions
        if not await self._check_market_conditions(signal['symbol']):
            return False
        
        return True
    
    async def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size using advanced PositionSizer with Kelly Criterion"""
        
        try:
            # Get current account value
            account = await self.broker.get_account()
            equity = float(account.equity)
            
            # Prepare signals dictionary for the position sizer
            signals = {signal['symbol']: signal['strength'] if signal['signal'] == 'buy' else -signal['strength']}
            
            # Get confidence scores
            confidence_scores = {signal['symbol']: signal['confidence']}
            
            # Get current positions for the position sizer
            positions = self.position_tracker.get_all_positions()
            current_positions = {}
            for pos in positions:
                if not pos.is_flat:
                    # Convert position to weight (normalized by portfolio value)
                    position_weight = (pos.quantity * pos.current_price) / equity if equity > 0 else 0
                    current_positions[pos.symbol] = position_weight
            
            # Get historical returns data for the symbol
            try:
                # Fetch recent historical data for returns calculation
                bars = await self.broker.get_bars(
                    signal['symbol'],
                    timeframe="1Day",
                    limit=252  # 1 year of daily data for Kelly calculation
                )
                
                if not bars.empty and len(bars) > 30:
                    # Calculate returns
                    returns_data = bars[['close']].pct_change().dropna()
                    returns_data.columns = [signal['symbol']]
                    
                    # Use the position sizer to calculate optimal size
                    sizing_result = self.position_sizer.calculate_position_sizes(
                        signals=signals,
                        returns_data=returns_data,
                        current_positions=current_positions,
                        confidence_scores=confidence_scores,
                        portfolio_value=equity
                    )
                    
                    # Extract the position size for our symbol
                    position_weight = sizing_result['position_sizes'].get(signal['symbol'], 0)
                    
                    # Convert weight to dollar amount
                    position_value = abs(position_weight) * equity
                    
                    # Get current price
                    quote = await self.broker.get_latest_quote(signal['symbol'])
                    if not quote:
                        logger.warning(f"No quote available for {signal['symbol']}, falling back to simple sizing")
                        return await self._calculate_simple_position_size(signal, equity)
                    
                    price = (quote['bid'] + quote['ask']) / 2
                    quantity = int(position_value / price)
                    
                    logger.info(f"Position sizing for {signal['symbol']}", 
                               kelly_weight=position_weight,
                               position_value=position_value,
                               quantity=quantity,
                               sizing_method='kelly_criterion')
                    
                    return max(0, quantity)
                    
                else:
                    logger.warning(f"Insufficient historical data for {signal['symbol']}, using simple sizing")
                    return await self._calculate_simple_position_size(signal, equity)
                    
            except Exception as data_error:
                logger.warning(f"Failed to get historical data for Kelly sizing: {data_error}, falling back to simple sizing")
                return await self._calculate_simple_position_size(signal, equity)
                
        except Exception as e:
            logger.error(f"Error in advanced position sizing: {e}, falling back to simple sizing")
            return await self._calculate_simple_position_size(signal, await self.broker.get_account().equity)
    
    async def _calculate_simple_position_size(self, signal: Dict[str, Any], equity: float) -> float:
        """Fallback simple position sizing method
        
        Uses fixed percentage risk per trade when Kelly Criterion calculation fails.
        
        Args:
            signal: Trading signal dictionary
            equity: Current account equity
            
        Returns:
            float: Number of shares to trade
        """
        
        try:
            # Use fixed percentage risk
            risk_per_trade = self.config.risk.risk_per_trade
            position_value = equity * risk_per_trade
            
            # Get current price
            quote = await self.broker.get_latest_quote(signal['symbol'])
            if not quote:
                return 0
            
            price = (quote['bid'] + quote['ask']) / 2
            quantity = int(position_value / price)
            
            logger.info(f"Simple position sizing for {signal['symbol']}", 
                       position_value=position_value,
                       quantity=quantity,
                       sizing_method='fixed_percentage')
            
            return max(0, quantity)
            
        except Exception as e:
            logger.error(f"Error in simple position sizing: {e}")
            return 0
    
    async def _check_risk_limits(self) -> bool:
        """Comprehensive risk limits check using full RiskEngine capabilities"""
        
        try:
            # Get current positions for risk assessment
            positions = self.position_tracker.get_all_positions()
            account = await self.broker.get_account()
            equity = float(account.equity)
            
            # Convert positions to weights for risk engine
            position_weights = {}
            for pos in positions:
                if not pos.is_flat:
                    position_value = pos.quantity * pos.current_price
                    weight = position_value / equity if equity > 0 else 0
                    position_weights[pos.symbol] = weight
            
            if not position_weights:
                logger.info("No positions to assess risk for")
                return True  # No positions = no risk
            
            # Get historical returns data for risk assessment
            symbols = list(position_weights.keys())
            returns_data_list = []
            
            for symbol in symbols:
                try:
                    bars = await self.broker.get_bars(
                        symbol,
                        timeframe="1Day",
                        limit=252  # 1 year of data for VaR
                    )
                    
                    if not bars.empty:
                        returns = bars[['close']].pct_change().dropna()
                        returns.columns = [symbol]
                        returns_data_list.append(returns)
                except Exception as e:
                    logger.warning(f"Could not fetch returns data for {symbol}: {e}")
            
            if not returns_data_list:
                logger.warning("No returns data available for risk assessment, using basic checks")
                return await self._basic_risk_check()
            
            # Combine returns data
            returns_df = pd.concat(returns_data_list, axis=1).fillna(0)
            
            # Perform comprehensive risk assessment
            risk_assessment = self.risk_engine.assess_portfolio_risk(
                positions=position_weights,
                returns=returns_df
            )
            
            # Log risk assessment summary
            risk_score = risk_assessment.get('overall_risk_score', 0)
            alerts = risk_assessment.get('risk_alerts', [])
            
            logger.info(f"Portfolio risk assessment completed",
                       risk_score=risk_score,
                       alerts_count=len(alerts),
                       critical_alerts=len([a for a in alerts if a['severity'] == 'CRITICAL']))
            
            # Check for critical risk limit breaches
            critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
            if critical_alerts:
                logger.error("CRITICAL RISK ALERTS DETECTED - Trading halted",
                           alerts=[a['message'] for a in critical_alerts])
                return False
            
            # Check overall risk score threshold
            if risk_score > 85:  # High risk threshold
                logger.warning("High portfolio risk score detected", risk_score=risk_score)
                return False
            
            # Check specific limit utilization
            limit_util = risk_assessment.get('limit_utilization', {})
            
            # Fail if any limit is significantly breached
            for limit_name, limit_data in limit_util.items():
                if limit_data.get('breach', False):
                    logger.error(f"Risk limit breached: {limit_name}", 
                               current=limit_data.get('current'),
                               limit=limit_data.get('limit'))
                    return False
                
                # Warning for high utilization
                if limit_data.get('utilization', 0) > 0.9:
                    logger.warning(f"Risk limit high utilization: {limit_name}",
                                 utilization=f"{limit_data['utilization']:.1%}")
            
            logger.info("✅ All risk limits within acceptable ranges")
            return True
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk check: {e}, falling back to basic check")
            return await self._basic_risk_check()
    
    async def _basic_risk_check(self) -> bool:
        """Fallback basic risk checking when comprehensive assessment fails
        
        Performs simplified risk checks based on drawdown and daily loss limits.
        Used when the comprehensive RiskEngine assessment is unavailable.
        
        Returns:
            bool: True if risk limits are within acceptable ranges
        """
        
        try:
            # Get current metrics
            portfolio_metrics = self.position_tracker.get_portfolio_summary()
            
            # Check drawdown
            max_loss = portfolio_metrics.get('max_position_loss', 0)
            if max_loss < -self.config.risk.max_drawdown:
                logger.warning("Basic risk check: Max drawdown exceeded", max_loss=max_loss)
                return False
            
            # Check daily loss
            account_status = self.account_monitor.get_current_status()
            daily_return = account_status.get('performance_metrics', {}).get('daily_return_percent', 0)
            if daily_return < -5:
                logger.warning("Basic risk check: Daily loss limit exceeded", daily_return=daily_return)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in basic risk check: {e}")
            return False
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are favorable"""
        
        # Could check volatility, spread, volume etc.
        return True
    
    async def _get_user_approval(self, signal: Dict[str, Any]) -> bool:
        """Get user approval for trade (semi-auto mode)"""
        
        print("\n" + "="*50)
        print("TRADE SIGNAL REQUIRES APPROVAL")
        print(f"Symbol: {signal['symbol']}")
        print(f"Action: {signal['signal'].upper()}")
        print(f"Strength: {signal['strength']:.2f}")
        print(f"Confidence: {signal['confidence']:.2f}")
        print("="*50)
        
        response = input("Execute trade? (y/n): ").lower()
        return response == 'y'
    
    async def _should_exit_position(self, position) -> bool:
        """Check if position should be exited based on signals"""
        
        # Could re-evaluate the position with current data
        # For now, simple time-based exit
        if position.opened_at:
            hold_time = datetime.now() - position.opened_at
            if hold_time > timedelta(hours=24):  # Exit after 24 hours
                return True
        
        return False
    
    async def _update_metrics(self) -> None:
        """Update performance metrics"""
        
        # Account snapshot
        await self.account_monitor.take_snapshot()
        
        # Log current status
        account_status = self.account_monitor.get_current_status()
        portfolio_summary = self.position_tracker.get_portfolio_summary()
        
        logger.info("Performance update",
                   equity=account_status.get('equity'),
                   positions=portfolio_summary.get('active_positions'),
                   total_pnl=portfolio_summary.get('total_pnl'))
    
    async def _get_trading_universe(self) -> List[str]:
        """Get list of symbols to analyze"""
        
        # Start with watchlist
        universe = self.config.trading.watchlist.copy()
        
        # Add current positions
        positions = self.position_tracker.get_all_positions()
        for position in positions:
            if not position.is_flat and position.symbol not in universe:
                universe.append(position.symbol)
        
        # Add symbols from dynamic discovery if enabled
        if self.dynamic_discovery and self.dynamic_discovery.config.enabled:
            try:
                # Update discovery system with current universe
                self.dynamic_discovery.update_universe(set(universe))
                
                # Run discovery and add new symbols
                new_symbols = await self.dynamic_discovery.run_discovery()
                if new_symbols:
                    logger.info("Adding new symbols from discovery", 
                               symbols=new_symbols, count=len(new_symbols))
                    universe.extend(new_symbols)
                    
                    # Try to update config universe if possible
                    try:
                        current_universe = self.config.get_nested('universe.stocks', [])
                        updated_universe = list(set(current_universe + new_symbols))
                        self.config.update('universe', {'stocks': updated_universe})
                        logger.debug("Updated config universe with new symbols")
                    except Exception as config_error:
                        logger.debug("Could not update config universe", error=str(config_error))
                    
            except Exception as e:
                logger.warning("Dynamic discovery failed", error=str(e))
        
        return list(set(universe))[:50]  # Remove duplicates and limit to 50 symbols
    
    def _log_strategy_configuration(self) -> None:
        """Log current strategy configuration for visibility"""
        
        strategy_mode = self.config.trading.strategy_mode
        logger.info("Trading strategy configuration",
                   strategy_mode=strategy_mode,
                   watchlist_size=len(self.config.trading.watchlist),
                   max_positions=self.config.trading.max_positions,
                   dynamic_discovery_enabled=getattr(self.dynamic_discovery.config, 'enabled', False) if self.dynamic_discovery else False)
        
        # Log strategy-specific settings
        signal_requirements = self.config.trading.signal_requirements
        if hasattr(signal_requirements, strategy_mode):
            strategy_config = getattr(signal_requirements, strategy_mode)
            config_dict = strategy_config.to_dict() if hasattr(strategy_config, 'to_dict') else (strategy_config.__dict__ if hasattr(strategy_config, '__dict__') else {})
            logger.info(f"Strategy '{strategy_mode}' configuration", **config_dict)
    
    def _extract_technical_indicators(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Extract technical indicators from features for strategy validation"""
        if features.empty:
            return {}
        
        latest_features = features.iloc[-1]
        indicators = {}
        
        # Common technical indicators
        for col in latest_features.index:
            if any(indicator in col.lower() for indicator in ['rsi', 'macd', 'sma', 'ema', 'bollinger']):
                indicators[col] = float(latest_features[col]) if pd.notna(latest_features[col]) else 0.0
        
        return indicators
    
    async def _validate_signal_by_strategy(self, signal: Dict[str, Any]) -> bool:
        """Validate signal based on configured strategy mode"""
        
        strategy_mode = self.config.trading.strategy_mode
        signal_requirements = self.config.trading.signal_requirements
        
        # Get strategy config
        strategy_config = signal_requirements.get(strategy_mode, {})
        
        if strategy_mode == "adaptive":
            return await self._validate_adaptive_signal(signal, strategy_config)
        elif strategy_mode == "technical_only":
            return await self._validate_technical_only_signal(signal, strategy_config)
        elif strategy_mode == "sentiment_only":
            return await self._validate_sentiment_only_signal(signal, strategy_config)
        elif strategy_mode == "conservative":
            return await self._validate_conservative_signal(signal, strategy_config)
        else:
            # Default validation
            return await self._validate_signal(signal)
    
    async def _validate_adaptive_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for adaptive strategy mode"""
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        min_signal_strength = config.get('min_signal_strength', 0.6)
        
        # Check minimum signal strength
        if signal['strength'] < min_signal_strength:
            return False
        
        # Boost confidence if both technical and sentiment signals agree
        if config.get('use_available_signals', True):
            has_technical = bool(signal.get('technical_indicators'))
            has_sentiment = bool(signal.get('sentiment_data'))
            
            if has_technical and has_sentiment and config.get('confidence_boost_both', 0) > 0:
                original_confidence = signal['confidence']
                boosted_confidence = min(1.0, original_confidence + config['confidence_boost_both'])
                signal['confidence'] = boosted_confidence
                
                logger.debug("Confidence boosted for multi-signal agreement",
                           symbol=signal['symbol'],
                           original=original_confidence,
                           boosted=boosted_confidence)
        
        return True
    
    async def _validate_technical_only_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for technical-only strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Must have technical indicators
        technical_indicators = signal.get('technical_indicators', {})
        if not technical_indicators:
            logger.debug("Technical-only mode: No technical indicators available",
                        symbol=signal['symbol'])
            return False
        
        # Check required indicators
        required_indicators = config.get('required_indicators', [])
        available_indicators = [name.lower() for name in technical_indicators.keys()]
        
        matching_indicators = []
        for required in required_indicators:
            if any(required in available for available in available_indicators):
                matching_indicators.append(required)
        
        min_confluence = config.get('min_confluence', 2)
        if len(matching_indicators) < min_confluence:
            logger.debug("Technical-only mode: Insufficient indicator confluence",
                        symbol=signal['symbol'],
                        required=min_confluence,
                        available=len(matching_indicators))
            return False
        
        return True
    
    async def _validate_sentiment_only_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for sentiment-only strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Must have sentiment data
        sentiment_data = signal.get('sentiment_data')
        if not sentiment_data:
            logger.debug("Sentiment-only mode: No sentiment data available",
                        symbol=signal['symbol'])
            return False
        
        # Check minimum sentiment score
        min_sentiment_score = config.get('min_sentiment_score', 0.7)
        sentiment_score = abs(sentiment_data.get('sentiment_score', 0))
        
        if sentiment_score < min_sentiment_score:
            logger.debug("Sentiment-only mode: Insufficient sentiment score",
                        symbol=signal['symbol'],
                        score=sentiment_score,
                        required=min_sentiment_score)
            return False
        
        # Check required sources
        required_sources = config.get('required_sources', [])
        available_sources = sentiment_data.get('sources', [])
        
        if required_sources:
            matching_sources = [source for source in required_sources if source in available_sources]
            if not matching_sources:
                logger.debug("Sentiment-only mode: Required sentiment sources not available",
                            symbol=signal['symbol'],
                            required=required_sources,
                            available=available_sources)
                return False
        
        return True
    
    async def _validate_conservative_signal(self, signal: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Validate signal for conservative strategy mode"""
        
        if not config.get('enabled', False):
            return False
        
        # Basic validation first
        if not await self._validate_signal(signal):
            return False
        
        # Higher confidence threshold
        higher_confidence_threshold = config.get('higher_confidence_threshold', 0.8)
        if signal['confidence'] < higher_confidence_threshold:
            logger.debug("Conservative mode: Insufficient confidence",
                        symbol=signal['symbol'],
                        confidence=signal['confidence'],
                        required=higher_confidence_threshold)
            return False
        
        # Require both technical and sentiment confirmation
        require_technical = config.get('require_technical_confirmation', True)
        require_sentiment = config.get('require_sentiment_confirmation', True)
        
        if require_technical and not signal.get('technical_indicators'):
            logger.debug("Conservative mode: Technical confirmation required but not available",
                        symbol=signal['symbol'])
            return False
        
        if require_sentiment and not signal.get('sentiment_data'):
            logger.debug("Conservative mode: Sentiment confirmation required but not available",
                        symbol=signal['symbol'])
            return False
        
        return True
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        
        logger.info("Shutting down trading system...")
        
        self.is_running = False
        
        # Cancel all pending orders
        if self.order_manager:
            active_orders = self.order_manager.get_active_orders()
            for order in active_orders:
                await self.broker.cancel_order(order.order_id)
        
        # Stop monitoring
        if self.broker:
            await self.broker.stop_monitoring()
            await self.broker.disconnect()
        
        # Close database
        if self.db_manager:
            await self.db_manager.close()
        
        logger.info("Trading system shutdown complete")


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="QuantumSentiment Algorithmic Trading Bot")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full_auto", "semi_auto", "paper", "backtest"],
        default="paper",
        help="Trading mode"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = QuantumSentimentBot(args.config, args.mode)
    
    # Override symbols if provided
    if args.symbols:
        bot.config.trading.watchlist = args.symbols
    
    # Run the bot
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())