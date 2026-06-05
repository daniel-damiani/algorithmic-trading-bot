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
import json
import os
import signal
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
        self._universal_feature_gen = None
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

            from src.sentiment.unusual_whales_analyzer import (
                UnusualWhalesAnalyzer,
                UnusualWhalesConfig,
                to_fusion_payload,
            )

            uw_cfg = getattr(self.config.data_sources, "unusual_whales", None)
            scrape_delay = getattr(uw_cfg, "scrape_delay", 900) if uw_cfg else 900
            unusual_whales_analyzer = UnusualWhalesAnalyzer(
                UnusualWhalesConfig(cache_ttl_seconds=int(scrape_delay))
            )
            self.unusual_whales_analyzer = unusual_whales_analyzer
            self._unusual_whales_initialized = False

            # Store analyzers for sentiment fusion
            self.reddit_analyzer = reddit_analyzer
            self.news_aggregator = news_aggregator
            
            # Initialize SentimentFusion
            from src.sentiment.sentiment_fusion import SentimentFusion, FusionConfig
            fusion_config = FusionConfig()
            self.sentiment_fusion = SentimentFusion(fusion_config)
            
            # Create a wrapper that combines analyzers and fusion
            class SentimentManager:
                def __init__(
                    self,
                    reddit_analyzer,
                    news_aggregator,
                    sentiment_fusion,
                    unusual_whales_analyzer,
                    bot,
                ):
                    self.reddit_analyzer = reddit_analyzer
                    self.news_aggregator = news_aggregator
                    self.sentiment_fusion = sentiment_fusion
                    self.unusual_whales_analyzer = unusual_whales_analyzer
                    self._bot = bot

                async def _ensure_unusual_whales(self) -> bool:
                    from src.feature_flags import is_active

                    if not is_active("unusual_whales"):
                        return False
                    if not getattr(self._bot, "_unusual_whales_initialized", False):
                        loop = asyncio.get_running_loop()
                        ok = await loop.run_in_executor(
                            None, self.unusual_whales_analyzer.initialize
                        )
                        self._bot._unusual_whales_initialized = bool(ok)
                    return (
                        self._bot._unusual_whales_initialized
                        and self.unusual_whales_analyzer.scraper_ready
                    )

                async def fuse_sentiment(self, symbol):
                    """Get sentiment from enabled sources and fuse them."""
                    from src.feature_flags import is_active, news_sources_active

                    class NeutralSentimentResult:
                        def __init__(self):
                            self.score = 0.0
                            self.confidence = 0.0
                            self.sources = []
                            self.timestamp = datetime.now()

                    use_reddit = is_active("reddit")
                    news_sources = news_sources_active()
                    use_news = bool(news_sources)
                    use_uw = is_active("unusual_whales")

                    if not use_reddit and not use_news and not use_uw:
                        return NeutralSentimentResult()

                    sentiment_data = {}

                    if use_reddit:
                        try:
                            reddit_result = await self.reddit_analyzer.analyze_symbol(symbol)
                            if reddit_result:
                                sentiment_data["reddit"] = reddit_result
                        except Exception as e:
                            logger.warning(f"Reddit sentiment failed for {symbol}: {e}")

                    if use_news:
                        try:
                            news_result = self.news_aggregator.analyze_symbol(
                                symbol, enabled_sources=news_sources
                            )
                            if news_result:
                                sentiment_data["news"] = news_result
                        except Exception as e:
                            logger.warning(f"News sentiment failed for {symbol}: {e}")

                    if use_uw:
                        try:
                            if await self._ensure_unusual_whales():
                                loop = asyncio.get_running_loop()
                                uw_result = await loop.run_in_executor(
                                    None,
                                    self.unusual_whales_analyzer.analyze_symbol,
                                    symbol,
                                )
                                uw_payload = to_fusion_payload(uw_result)
                                if uw_payload:
                                    sentiment_data["unusual_whales"] = uw_payload
                                else:
                                    logger.debug(
                                        "Unusual Whales: no recent trades for symbol",
                                        symbol=symbol,
                                        congress=uw_result.get("total_congress_trades", 0),
                                    )
                            else:
                                logger.warning(
                                    "Unusual Whales enabled but scraper not ready "
                                    "(install Playwright: playwright install chromium)"
                                )
                        except Exception as e:
                            logger.warning(f"Unusual Whales failed for {symbol}: {e}")

                    if sentiment_data:
                        fused_result = self.sentiment_fusion.fuse_sentiment(sentiment_data, symbol)

                        class FusedSentimentResult:
                            def __init__(self, fused_data):
                                self.score = fused_data.get(
                                    "fused_sentiment",
                                    fused_data.get("sentiment_score", 0.0),
                                )
                                self.confidence = fused_data.get(
                                    "fusion_confidence",
                                    fused_data.get("confidence", 0.0),
                                )
                                self.sources = fused_data.get("sources_used", [])
                                self.timestamp = fused_data.get("timestamp", datetime.now())

                        return FusedSentimentResult(fused_result)
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
            
            self.sentiment_analyzer = SentimentManager(
                reddit_analyzer,
                news_aggregator,
                self.sentiment_fusion,
                unusual_whales_analyzer,
                self,
            )
            
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

            # 14. Apply dashboard risk preset (defaults to low / conservative)
            from src.risk_settings import apply_params_to_config

            applied = apply_params_to_config(self.config)
            logger.info("Runtime risk settings applied", preset_file="cache/dashboard/risk_settings.json", **applied)
            
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
            
            # Initialize position tracker for backtest mode
            position_config = PositionTrackerConfig(
                max_positions=50,
                enable_real_time_pnl=False  # Disabled for backtest
            )
            self.position_tracker = PositionTracker(position_config)
            
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
        """Load the best performing trained model."""
        
        model_load_errors = []
        
        # PRIORITY: sklearn XGBoost saved by train_simple_massive.py
        try:
            from src.models.sklearn_xgboost_predictor import SklearnXGBoostPredictor
            from pathlib import Path

            model = SklearnXGBoostPredictor.load_latest(
                models_root=Path(self.config.paths.models)
            )
            logger.info(
                "Loaded MarketRegimeXGBoost (train_simple_massive)",
                features=len(model.feature_names),
                path=str(model.model_dir),
            )
            return model
        except Exception as e:
            model_load_errors.append(f"MarketRegimeXGBoost/sklearn: {e}")
            logger.warning("Sklearn XGBoost model not loaded", error=str(e))
        
        # FALLBACK: Try other individual models if MarketRegimeXGBoost fails
        fallback_models = ['XGBoost', 'xgboost_model']
        for model_name in fallback_models:
            try:
                model, metadata = persistence.load_model(model_name)
                
                # Fix metadata compatibility if needed
                if not hasattr(metadata, 'metrics'):
                    logger.warning(f"ModelMetadata for {model_name} missing 'metrics' attribute - adding empty metrics")
                    metadata.metrics = getattr(metadata, 'test_metrics', {})
                
                logger.info(f"Loaded {model_name} model as fallback", 
                           version=getattr(metadata, 'version', 'unknown'),
                           metrics=getattr(metadata, 'metrics', {}))
                
                # Verify the model has a predict method
                if not hasattr(model, 'predict'):
                    raise RuntimeError(f"{model_name} model missing predict method. Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
                
                return model
                
            except Exception as e:
                model_load_errors.append(f"{model_name}: {e}")
                logger.error(f"Failed to load {model_name}: {e}")
                continue
        
        # LAST RESORT: Try StackedEnsemble (has config issues but might work)
        try:
            ensemble, metadata = persistence.load_model('StackedEnsemble')
            
            # Fix the metadata compatibility issue
            if not hasattr(metadata, 'metrics'):
                logger.warning("ModelMetadata missing 'metrics' attribute - adding empty metrics")
                metadata.metrics = getattr(metadata, 'test_metrics', {})
            
            logger.warning("Using StackedEnsemble as last resort (may have config issues)")
            
            # Verify the model has a predict method
            if not hasattr(ensemble, 'predict'):
                raise RuntimeError(f"StackedEnsemble model missing predict method. Available methods: {[m for m in dir(ensemble) if not m.startswith('_')]}")
            
            return ensemble
            
        except Exception as e:
            model_load_errors.append(f"StackedEnsemble: {e}")
            logger.error(f"Failed to load StackedEnsemble: {e}")
        
        # If we reach here, NO models were loaded successfully
        error_summary = "\n".join([f"  - {error}" for error in model_load_errors])
        logger.critical("CRITICAL ERROR: No trained models could be loaded!")
        logger.critical(f"Model loading errors:\n{error_summary}")
        
        raise RuntimeError(
            f"TRADING SYSTEM CANNOT START: No trained models available!\n"
            f"Attempted to load models but all failed:\n{error_summary}\n\n"
            f"SOLUTION: Retrain your models or fix the model loading issues.\n"
            f"DO NOT TRADE WITHOUT PROPER MODELS - This prevents accidental trading with hardcoded signals."
        )
    
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
            # Skip market data check in backtest mode - simulated broker doesn't need connectivity
            if self.mode == TradingMode.BACKTEST:
                return True
            
            # Test with SPY quote
            quote = await self.broker.get_latest_quote("SPY")
            return quote is not None
        except Exception as e:
            logger.error("Market data check failed", error=str(e))
            return False
    
    async def _check_sentiment_sources(self) -> bool:
        """Check sentiment data sources"""
        try:
            # Skip sentiment check in backtest mode - use simulated sentiment
            if self.mode == TradingMode.BACKTEST:
                return True
            
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
        self._write_dashboard_heartbeat(
            {"note": "trading loop started"},
            {"active_positions": 0},
        )
        
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

        from src.risk_settings import apply_params_to_config

        apply_params_to_config(self.config)
        
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
            trading_allowed = await self._check_risk_limits()
            if not trading_allowed:
                logger.warning("🛑 Risk limits exceeded, new trades halted this cycle")
            else:
                # 4. Execute pending signals
                await self._execute_signals()

            # Always sync and monitor exits (TP/SL) even when new trades are blocked
            await self.broker.sync_positions()
            logger.info("👀 Monitoring positions...")
            await self._monitor_positions()

            # Cycle summary
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            positions = self.position_tracker.get_all_positions()
            active_positions = [p for p in positions if not p.is_flat]
            logger.info(
                f"✅ CYCLE COMPLETE ({cycle_duration:.1f}s) - "
                f"{len(active_positions)} active positions"
            )

        except Exception as e:
            logger.error("Error in trading cycle", error=str(e))
        finally:
            try:
                await self._update_metrics()
            except Exception as exc:
                logger.error("Failed to update metrics", error=str(exc))
    
    async def _trading_cycle_backtest(self, symbols: List[str], timestamp: datetime) -> None:
        """
        Execute one trading cycle for backtesting - event-driven, no sleep
        
        Args:
            symbols: List of symbols to trade
            timestamp: Current simulation timestamp
        """
        
        logger.info(f"🔄 BACKTEST CYCLE - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Only act on signals from the current bar (hourly backtests accumulate stale signals otherwise)
            self.pending_signals.clear()

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
    
    def _interpret_model_prediction(
        self,
        probas: Optional[np.ndarray] = None,
        raw_prediction: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert model probabilities to a trading signal.

        train_simple_massive uses 5 classes:
        0=strong down, 1=down, 2=hold, 3=up, 4=strong up.
        """
        signal_threshold = getattr(self.config.trading, "signal_threshold", 0.5)
        confidence_threshold = getattr(self.config.trading, "confidence_threshold", 0.4)

        if probas is not None and isinstance(probas, np.ndarray) and probas.size > 0:
            p = probas[0] if probas.ndim > 1 else probas
            n_classes = len(p)

            if n_classes == 2:
                bear_prob = float(p[0])
                bull_prob = float(p[1])
            elif n_classes >= 5:
                bear_prob = float(p[0] + p[1])
                bull_prob = float(p[3] + p[4])
            elif n_classes == 3:
                bear_prob, bull_prob = float(p[0]), float(p[2])
            else:
                bull_prob = float(p[-1])
                bear_prob = float(np.sum(p[:-1]))

            direction_score = bull_prob - bear_prob
            winning_prob = bull_prob if direction_score >= 0 else bear_prob
            strength = max(abs(direction_score), winning_prob)
            min_edge = max(0.05, (1.0 - signal_threshold) * 0.2)
            min_prob = max(0.2, signal_threshold * 0.5)

            if direction_score >= min_edge and bull_prob >= min_prob:
                confidence = bull_prob / max(bull_prob + bear_prob, 1e-9)
                if confidence >= confidence_threshold:
                    return {
                        "signal": "buy",
                        "strength": strength,
                        "confidence": confidence,
                        "bull_prob": bull_prob,
                        "bear_prob": bear_prob,
                    }

            if direction_score <= -min_edge and bear_prob >= min_prob:
                confidence = bear_prob / max(bull_prob + bear_prob, 1e-9)
                if confidence >= confidence_threshold:
                    return {
                        "signal": "sell",
                        "strength": strength,
                        "confidence": confidence,
                        "bull_prob": bull_prob,
                        "bear_prob": bear_prob,
                    }
            return None

        if raw_prediction is not None:
            pred = float(raw_prediction)
            strength = abs(pred - 0.5) * 2
            confidence = min(1.0, strength + 0.2)
            edge = (1.0 - signal_threshold) * 0.1
            if pred > 0.5 + edge and confidence >= confidence_threshold:
                return {"signal": "buy", "strength": strength, "confidence": confidence}
            if pred < 0.5 - edge and confidence >= confidence_threshold:
                return {"signal": "sell", "strength": strength, "confidence": confidence}
        return None

    def _position_market_price(self, position) -> float:
        """Mark price for a position-tracker Position (no current_price field)."""
        if self.position_tracker is not None:
            prices = getattr(self.position_tracker, "market_prices", None) or {}
            mark = prices.get(position.symbol)
            if mark is not None:
                return float(mark)
        return float(getattr(position, "average_price", 0.0) or 0.0)

    def _position_exit_levels(self, position) -> Dict[str, Any]:
        """Stop/take-profit metrics for logging and monitoring."""
        from src.broker.position_exits import compute_exit_levels

        mark = self._position_market_price(position)
        side = "short" if position.is_short else "long"
        stop_loss_pct = getattr(self.config.risk, "stop_loss_pct", 0.05)
        take_profit_pct = getattr(self.config.risk, "take_profit_pct", 0.15)
        return compute_exit_levels(
            side=side,
            avg_entry=float(position.average_price or 0),
            mark=mark,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

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
        
        logger.info(f"🧠 Generating backtest predictions for {len(symbols)} symbols at {timestamp}")
        
        # Get market data for all symbols up to current timestamp
        for symbol in symbols:
            logger.info(f"🔍 Processing symbol: {symbol}")
            try:
                # Get market data up to current timestamp
                bars = await self.broker.get_bars(
                    symbol,
                    timeframe="1Hour",
                    end=timestamp,
                    limit=500
                )
                
                if bars is None or bars.empty:
                    logger.info(f"⚠️ No market data for {symbol} at {timestamp}")
                    continue
                
                logger.info(f"✅ Got {len(bars)} bars for {symbol}")
                
                # Ensure we have enough data (reduced from 50 to 20 for shorter backtests)
                if len(bars) < 20:  # Need minimum bars for technical indicators
                    logger.info(f"⚠️ Insufficient data for {symbol} ({len(bars)} bars)")
                    continue
                
                logger.info(f"✅ Sufficient data for {symbol} ({len(bars)} bars >= 20)")
                
                logger.info(f"🔍 Analyzing {symbol} with {len(bars)} bars...")
                
                # Get sentiment data (simplified for backtesting)
                sentiment_dict = {
                    'sentiment_score': 0.0,  # Neutral sentiment for backtesting
                    'confidence': 0.5,
                    'sources': ['backtest'],
                    'timestamp': timestamp
                }
                
                sentiment_df = pd.DataFrame([sentiment_dict])
                sentiment_df.set_index('timestamp', inplace=True)
                
                # Generate features (match training pipeline when using train_simple_massive model)
                logger.info(f"🔧 Generating features for {symbol}")
                if getattr(self.ensemble_model, "requires_universal_features", False):
                    if self._universal_feature_gen is None:
                        from src.features.universal_features import UniversalFeatureGenerator
                        self._universal_feature_gen = UniversalFeatureGenerator()
                    bars_df = bars.reset_index()
                    if "timestamp" not in bars_df.columns:
                        bars_df = bars_df.rename(columns={bars_df.columns[0]: "timestamp"})
                    feat_df = self._universal_feature_gen.generate_features(bars_df)
                    features_dict = feat_df.iloc[-1].to_dict()
                else:
                    feature_result = self.feature_pipeline.generate_features(
                        symbol=symbol,
                        market_data=bars,
                        sentiment_data=sentiment_df,
                    )
                    features_dict = feature_result.get("features", {})
                logger.info(f"📊 Generated {len(features_dict)} features for {symbol}")
                if not features_dict:
                    logger.info(f"❌ No features generated for {symbol}")
                    continue
                
                # Log some feature keys for debugging
                feature_keys = list(features_dict.keys())[:10]
                logger.debug(f"Feature keys sample: {feature_keys}")
                logger.debug(f"Has 'close'? {'close' in features_dict}, Has 'price'? {'price' in features_dict}")
                
                # Add raw price data that the model might need
                if 'close' not in features_dict and not bars.empty:
                    features_dict['close'] = bars.iloc[-1]['close']
                    features_dict['open'] = bars.iloc[-1]['open']
                    features_dict['high'] = bars.iloc[-1]['high']
                    features_dict['low'] = bars.iloc[-1]['low']
                    features_dict['volume'] = bars.iloc[-1]['volume']
                    logger.debug(f"Added raw price data: close={features_dict['close']}")
                
                # Convert to DataFrame for model prediction
                features = pd.DataFrame([features_dict])
                
                # Get prediction from model
                logger.info(f"🧠 Running model prediction for {symbol}")
                try:
                    # Add symbol_code feature to match training
                    symbol_mapping = {'AAPL': 0, 'MSFT': 1, 'GOOGL': 2, 'AMZN': 3, 'TSLA': 4, 'NVDA': 5, 'META': 6}
                    features_dict['symbol_code'] = symbol_mapping.get(symbol, 99)  # Default to 99 for unknown symbols
                    
                    # Recreate DataFrame with all features
                    features = pd.DataFrame([features_dict])
                    
                    # Log features being passed to model
                    logger.info(f"🔍 Features for {symbol}: {list(features.columns)[:5]}... (total: {features.shape[1]} features, {features.shape[0]} rows)")
                    
                    # Try to get a simple prediction - using predict_proba or predict
                    probas = None
                    raw_prediction = None
                    
                    # For classification models, get probability
                    if hasattr(self.ensemble_model, 'predict_proba'):
                        try:
                            probas = self.ensemble_model.predict_proba(features_dict)
                            logger.info(
                                f"🎯 Predict proba result: shape={probas.shape if hasattr(probas, 'shape') else 'no shape'}, "
                                f"values={probas}"
                            )
                        except Exception as e:
                            logger.warning(f"Predict proba failed: {e}")
                    
                    # Fallback to regular predict
                    if probas is None and hasattr(self.ensemble_model, 'predict'):
                        try:
                            pred_result = self.ensemble_model.predict(features)
                            logger.info(f"🎯 Predict result: type={type(pred_result)}, value={pred_result}")
                            if isinstance(pred_result, np.ndarray) and pred_result.size > 0:
                                if pred_result.dtype in [np.int32, np.int64]:
                                    raw_prediction = float(pred_result[0]) / max(
                                        getattr(self.ensemble_model.model, "n_classes_", 5) - 1, 1
                                    )
                                else:
                                    raw_prediction = float(pred_result[0])
                            elif pred_result is not None:
                                raw_prediction = float(pred_result)
                        except Exception as e:
                            logger.debug(f"Predict failed: {e}")

                    interpreted = self._interpret_model_prediction(probas=probas, raw_prediction=raw_prediction)
                    if interpreted is None:
                        logger.info(f"⏸️ No trade signal for {symbol} (hold/neutral)")
                        continue

                    signal_strength = interpreted["strength"]
                    confidence = interpreted["confidence"]
                    logger.info(
                        f"📈 Signal for {symbol}: {interpreted['signal'].upper()} "
                        f"(strength={signal_strength:.3f}, confidence={confidence:.3f})"
                    )
                        
                except Exception as e:
                    logger.info(f"Prediction failed for {symbol}: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
                
                # Create trading signal
                signal = {
                    'symbol': symbol,
                    'signal': interpreted['signal'],
                    'strength': signal_strength,
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'features': features.iloc[0].to_dict() if not features.empty else {},
                    'sentiment_data': sentiment_dict,
                    'technical_indicators': self._extract_technical_indicators(features)
                }
                
                # Validate signal
                logger.info(f"✅ Validating signal for {symbol} - Signal: {signal['signal']}, Strength: {signal['strength']:.3f}")
                is_valid = await self._validate_signal_by_strategy(signal)
                logger.info(f"🔍 Validation result for {symbol}: {is_valid}")
                if is_valid:
                    self.pending_signals.append(signal)
                    logger.info(f"✅ BACKTEST SIGNAL ADDED: {signal['signal'].upper()} {symbol} (strength: {signal['strength']:.3f})")
                else:
                    logger.info(f"❌ SIGNAL REJECTED: {signal['signal'].upper()} {symbol} (strength: {signal['strength']:.3f})")
                
            except Exception as e:
                logger.info(f"❌ Failed to generate backtest prediction for {symbol}: {e}")
                import traceback
                traceback.print_exc()
    
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
        for idx, (symbol, bars) in enumerate(market_data_results):
            if idx % 5 == 0:
                self._write_dashboard_heartbeat(
                    note=f"analyzing {idx + 1}/{len(market_data_results)}"
                )
            try:
                logger.info(f"🔍 Analyzing {symbol}...")
                logger.info(f"📊 Got {len(bars)} bars for {symbol}, latest price: ${bars.iloc[-1]['close']:.2f}")
                
                # 2. Get sentiment data using real fusion
                try:
                    sentiment_result = await self.sentiment_analyzer.fuse_sentiment(symbol)
                    
                    # Only pass sentiment to features when fusion produced usable data
                    if (
                        sentiment_result
                        and sentiment_result.confidence > 0
                        and sentiment_result.sources
                    ):
                        import pandas as pd
                        sentiment_df = pd.DataFrame([{
                            'timestamp': sentiment_result.timestamp,
                            'sentiment_score': sentiment_result.score,
                            'confidence': sentiment_result.confidence,
                            'source': ','.join(sentiment_result.sources)
                        }])
                        sentiment_df.set_index('timestamp', inplace=True)

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
                        if not (isinstance(features, pd.DataFrame) and len(features) > 0):
                            logger.warning(f"Invalid features format for {symbol}")
                            continue

                        probas = None
                        raw_prediction = None
                        if hasattr(self.ensemble_model, 'predict_proba'):
                            probas = self.ensemble_model.predict_proba(features)
                        else:
                            raw_pred = self.ensemble_model.predict(features)
                            if isinstance(raw_pred, np.ndarray) and raw_pred.size > 0:
                                raw_prediction = float(raw_pred[0])

                        interpreted = self._interpret_model_prediction(
                            probas=probas, raw_prediction=raw_prediction
                        )
                        if interpreted is None:
                            continue

                        prediction = {
                            'signal_strength': interpreted['strength'] if interpreted['signal'] == 'buy'
                            else -interpreted['strength'],
                            'confidence': interpreted['confidence'],
                            'signal': interpreted['signal'],
                        }
                    else:
                        logger.error(f"Model not trained - cannot generate predictions for {symbol}")
                        continue
                        
                    signal_strength = prediction.get('signal_strength', 0)
                    confidence = prediction.get('confidence', 0)
                    signal_side = prediction.get('signal', 'buy' if signal_strength > 0 else 'sell')
                    predictions_made += 1
                
                except Exception as e:
                    logger.warning("Ensemble prediction failed, using fallback", 
                                 symbol=symbol, error=str(e))
                    continue
                
                logger.info(f"🎯 PREDICTION for {symbol}: strength={abs(signal_strength):.3f}, confidence={confidence:.3f}")
                
                # 5. Create trading signal with enhanced data
                signal = {
                    'symbol': symbol,
                    'signal': signal_side,
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
                
                # Log position health (info so TP/SL distance is visible in bot.log)
                levels = self._position_exit_levels(position)
                logger.info(
                    "Position exit check",
                    symbol=position.symbol,
                    side="short" if position.is_short else "long",
                    qty=position.quantity,
                    entry=levels.get("entry_price", position.average_price),
                    mark=self._position_market_price(position),
                    profit_pct=levels.get("profit_pct"),
                    take_profit_at=levels.get("take_profit_price"),
                    stop_loss_at=levels.get("stop_loss_price"),
                    distance_to_tp_pct=levels.get("distance_to_take_profit_pct"),
                    unrealized_pnl=position.unrealized_pnl,
                    risk_score=position_risk.get("risk_score", 0),
                )
                
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
            levels = self._position_exit_levels(position)
            return bool(levels.get("stop_loss_triggered"))
            
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
            levels = self._position_exit_levels(position)
            return bool(levels.get("take_profit_triggered"))
            
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
            mark = self._position_market_price(position)
            position_value = abs(position.quantity * mark)
            position_weight = position_value / equity if equity > 0 else 0
            
            if position_weight > self.config.trading.max_position_size:
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
            total_exposure = sum(
                abs(p.quantity * self._position_market_price(p)) for p in active_positions
            )
            leverage = total_exposure / equity if equity > 0 else 0
            
            # Portfolio concentration check
            position_weights = [
                (abs(p.quantity * self._position_market_price(p)) / equity, p.symbol)
                for p in active_positions
            ]
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
            if self.mode == TradingMode.BACKTEST:
                alpaca_order = await self.broker.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="market",
                )
                return alpaca_order.id if alpaca_order else None

            # Use smart router for live/paper execution
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
        
        # Check position limits for new positions (buy signals)
        if signal['signal'] == 'buy':
            current_positions = len([p for p in self.position_tracker.get_all_positions() if not p.is_flat])
            if current_positions >= self.config.trading.max_positions:
                return False
            
            # Check concentration limits - don't buy if we already have a position
            if signal['symbol'] in [p.symbol for p in self.position_tracker.get_all_positions() if not p.is_flat]:
                return False  # Already have position, don't add more
        
        elif signal['signal'] == 'sell':
            # For sell signals, we must have a position to sell
            current_position = None
            for p in self.position_tracker.get_all_positions():
                if p.symbol == signal['symbol'] and not p.is_flat:
                    current_position = p
                    break
            
            if not current_position:
                logger.debug("Cannot sell - no position in symbol", symbol=signal['symbol'])
                return False  # No position to sell
            
            # Check if we have enough shares to sell
            position_size = await self._calculate_position_size(signal)
            if abs(current_position.quantity) < position_size:
                logger.debug("Cannot sell - insufficient shares", 
                           symbol=signal['symbol'],
                           available=abs(current_position.quantity),
                           requested=position_size)
                return False  # Not enough shares
        
        return True
    
    async def _should_execute_signal(self, signal: Dict[str, Any]) -> bool:
        """Determine if signal should be executed"""
        
        # In backtest mode, use simulation time instead of current system time
        if self.mode == 'backtest' and hasattr(self.broker, 'current_time') and self.broker.current_time:
            current_time = self.broker.current_time
        else:
            # Live/paper trading: use actual current time
            current_time = datetime.now()
        
        signal_timestamp = signal['timestamp']
        
        # Handle timezone compatibility
        if hasattr(signal_timestamp, 'tz') and signal_timestamp.tz is not None:
            # Signal timestamp is timezone-aware
            if current_time.tzinfo is None:
                # Make current_time timezone-aware (UTC)
                import pytz
                current_time = pytz.utc.localize(current_time)
        elif current_time.tzinfo is not None:
            # Current time is timezone-aware but signal is naive
            current_time = current_time.replace(tzinfo=None)
        
        signal_age = current_time - signal_timestamp
        
        # More lenient age check for backtesting (hourly bars can span multiple hours within a session)
        max_signal_age = timedelta(hours=24) if self.mode == 'backtest' else timedelta(minutes=10)
        if signal_age > max_signal_age:
            logger.debug("Signal too old", signal_age=signal_age, max_age=max_signal_age, mode=self.mode)
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
                    mark = self._position_market_price(pos)
                    position_weight = (pos.quantity * mark) / equity if equity > 0 else 0
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
        """PRODUCTION-GRADE comprehensive risk limits check"""
        
        try:
            # Different risk checks for different modes
            if self.mode == TradingMode.BACKTEST:
                return await self._check_backtest_risk_limits()
            else:
                return await self._check_live_risk_limits()
                
        except Exception as e:
            logger.error("CRITICAL: Risk system failure", error=str(e))
            if self.mode in (TradingMode.PAPER, TradingMode.SEMI_AUTO):
                logger.warning("Paper/semi-auto: continuing despite risk system error")
                return True
            raise RuntimeError(f"Risk system failure - trading halted: {e}")
    
    async def _check_backtest_risk_limits(self) -> bool:
        """Production-grade risk checks for backtesting mode"""
        
        if not self.broker or not hasattr(self.broker, 'equity'):
            raise RuntimeError("Simulated broker not properly initialized")
        
        # Get current portfolio state
        initial_capital = getattr(self.broker, 'initial_capital', 10000)
        current_equity = self.broker.equity
        positions = getattr(self.broker, 'positions', {})
        
        # 1. DRAWDOWN CHECK - CRITICAL
        drawdown = (initial_capital - current_equity) / initial_capital
        if drawdown > self.config.risk.max_drawdown:
            logger.warning("RISK BREACH: Maximum drawdown exceeded",
                         drawdown=f"{drawdown:.2%}",
                         limit=f"{self.config.risk.max_drawdown:.2%}",
                         equity=current_equity,
                         initial=initial_capital)
            return False
        
        # 2. DAILY LOSS LIMIT - CRITICAL
        daily_start_equity = getattr(self.broker, '_daily_start_equity', initial_capital)
        daily_pnl = (current_equity - daily_start_equity) / daily_start_equity
        if daily_pnl < -self.config.risk.daily_loss_limit:
            logger.warning("RISK BREACH: Daily loss limit exceeded",
                         daily_pnl=f"{daily_pnl:.2%}",
                         limit=f"{self.config.risk.daily_loss_limit:.2%}")
            return False
        
        # 3. POSITION COUNT LIMIT
        active_positions = len([p for p in positions.values() if abs(p.qty) > 0])
        if active_positions >= self.config.trading.max_positions:
            logger.info("Position limit reached",
                       current=active_positions,
                       max=self.config.trading.max_positions)
            return False
        
        # 4. POSITION SIZE LIMITS
        for symbol, position in positions.items():
            if abs(position.qty) > 0:
                position_value = abs(position.qty * position.current_price)
                position_weight = position_value / current_equity if current_equity > 0 else 0
                
                if position_weight > self.config.trading.max_position_size:
                    logger.warning("RISK BREACH: Position size limit exceeded",
                                 symbol=symbol,
                                 weight=f"{position_weight:.2%}",
                                 limit=f"{self.config.trading.max_position_size:.2%}")
                    return False
        
        # 5. LEVERAGE CHECK
        total_exposure = sum(abs(p.qty * p.current_price) for p in positions.values() if abs(p.qty) > 0)
        leverage = total_exposure / current_equity if current_equity > 0 else 0
        
        if leverage > self.config.risk.max_leverage:
            logger.warning("RISK BREACH: Leverage limit exceeded",
                         leverage=f"{leverage:.2f}x",
                         limit=f"{self.config.risk.max_leverage:.2f}x")
            return False
        
        return True
    
    async def _check_live_risk_limits(self) -> bool:
        """Production-grade risk checks for live trading mode"""
        
        if not self.broker:
            raise RuntimeError("Broker not initialized for live risk checks")
        
        try:
            # Get live account and position data
            account = await self.broker.get_account()
            positions = self.position_tracker.get_all_positions()
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # 1. ACCOUNT STATUS CHECK
            if account.status != 'ACTIVE':
                logger.error("RISK BREACH: Account not active", status=account.status)
                return False
            
            # 2. EQUITY REQUIREMENTS
            min_equity = getattr(self.config.risk, 'min_equity_threshold', 1000)
            if equity < min_equity:
                logger.error("RISK BREACH: Insufficient equity", equity=equity, minimum=min_equity)
                return False
            
            # 3. BUYING POWER CHECK
            if buying_power <= 0:
                logger.warning("RISK WARNING: No buying power available", buying_power=buying_power)
                return False
            
            # 4. POSITION LIMITS (non-flat only)
            active_positions = [p for p in positions if not p.is_flat]
            if len(active_positions) >= self.config.trading.max_positions:
                logger.info(
                    "Position limit reached",
                    current=len(active_positions),
                    max=self.config.trading.max_positions,
                )
                return False
            
            # 5. DRAWDOWN CHECK using historical high
            if not hasattr(self, '_equity_high'):
                self._equity_high = equity
            
            if equity > self._equity_high:
                self._equity_high = equity
            
            drawdown = (self._equity_high - equity) / self._equity_high if self._equity_high > 0 else 0
            if drawdown > self.config.risk.max_drawdown:
                logger.warning("RISK BREACH: Maximum drawdown exceeded",
                             drawdown=f"{drawdown:.2%}",
                             limit=f"{self.config.risk.max_drawdown:.2%}")
                return False
            
            # 6. DAILY LOSS CHECK
            day_change = getattr(account, 'day_change', 0)
            daily_loss_pct = float(day_change) / equity if equity > 0 else 0
            
            if daily_loss_pct < -self.config.risk.daily_loss_limit:
                logger.warning("RISK BREACH: Daily loss limit exceeded",
                             daily_loss=f"{daily_loss_pct:.2%}",
                             limit=f"{self.config.risk.daily_loss_limit:.2%}")
                return False
            
            # 7. POSITION SIZE CHECKS
            for position in positions:
                if not position.is_flat:
                    mark = self._position_market_price(position)
                    position_value = abs(position.quantity * mark)
                    position_weight = position_value / equity if equity > 0 else 0
                    
                    if position_weight > self.config.trading.max_position_size:
                        logger.warning("RISK BREACH: Position size exceeded",
                                     symbol=position.symbol,
                                     weight=f"{position_weight:.2%}",
                                     limit=f"{self.config.trading.max_position_size:.2%}")
                        return False
            
            # 8. VaR CALCULATION (if risk engine available and positions exist)
            if self.risk_engine and len([p for p in positions if not p.is_flat]) > 0:
                try:
                    symbols = [pos.symbol for pos in positions if not pos.is_flat]
                    returns_data = await self._get_returns_data_for_var(symbols)
                    if returns_data is not None:
                        weights = self._calculate_position_weights(positions, equity)
                        portfolio_var = self.risk_engine.calculate_portfolio_var(
                            returns_data=returns_data,
                            weights=weights,
                            confidence_level=0.05,  # 95% VaR
                            time_horizon=1
                        )
                        
                        var_limit = equity * 0.05  # 5% VaR limit
                        if portfolio_var > var_limit:
                            logger.warning("RISK BREACH: Portfolio VaR exceeded",
                                         var=portfolio_var,
                                         limit=var_limit)
                            return False
                
                except Exception as var_error:
                    logger.warning("VaR calculation failed", error=str(var_error))
                    # Continue without VaR check if it fails
            
            return True
            
        except Exception as e:
            logger.error("Live risk check failed", error=str(e))
            if self.mode in (TradingMode.PAPER, TradingMode.SEMI_AUTO):
                logger.warning("Paper/semi-auto: treating risk check error as pass")
                return True
            raise RuntimeError(f"Live risk system failure: {e}")
    
    async def _get_returns_data_for_var(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get historical returns data for VaR calculation"""
        try:
            returns_data_list = []
            
            for symbol in symbols:
                try:
                    bars = await self.broker.get_bars(
                        symbol,
                        timeframe="1Day",
                        limit=252  # 1 year of data
                    )
                    
                    if not bars.empty:
                        returns = bars[['close']].pct_change().dropna()
                        returns.columns = [symbol]
                        returns_data_list.append(returns)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch returns for {symbol}: {e}")
            
            if returns_data_list:
                import pandas as pd
                returns_data = pd.concat(returns_data_list, axis=1)
                return returns_data.fillna(0)
            
            return None
            
        except Exception as e:
            logger.error("Error getting returns data for VaR", error=str(e))
            return None
    
    def _calculate_position_weights(self, positions: List, equity: float) -> np.ndarray:
        """Calculate position weights for VaR calculation"""
        weights = []
        
        for position in positions:
            if not position.is_flat:
                mark = self._position_market_price(position)
                position_value = position.quantity * mark
                weight = position_value / equity if equity > 0 else 0
                weights.append(weight)
            else:
                weights.append(0.0)
        
        return np.array(weights)
    
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

        self._write_dashboard_heartbeat(account_status, portfolio_summary)

    def _write_dashboard_heartbeat(
        self,
        account_status: Optional[Dict[str, Any]] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> None:
        """Write heartbeat JSON for LAN dashboard."""
        try:
            cache_dir = Path(__file__).parent.parent / "cache" / "dashboard"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if account_status is None and self.account_monitor:
                account_status = self.account_monitor.get_current_status()
            if portfolio_summary is None and self.position_tracker:
                portfolio_summary = self.position_tracker.get_portfolio_summary()
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": self.mode,
                "equity": (account_status or {}).get("equity"),
                "positions": (portfolio_summary or {}).get("active_positions", 0),
                "pending_signals": len(self.pending_signals),
                "total_pnl": (portfolio_summary or {}).get("total_pnl"),
                "note": note if note is not None else (account_status or {}).get("note"),
            }
            with open(cache_dir / "bot_heartbeat.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug("Dashboard heartbeat write failed", error=str(e))
    
    async def _get_trading_universe(self) -> List[str]:
        """Get list of symbols to analyze"""
        
        # Start with watchlist
        universe = self.config.trading.watchlist.copy()
        
        # Add current positions
        positions = self.position_tracker.get_all_positions()
        for position in positions:
            if not position.is_flat and position.symbol not in universe:
                universe.append(position.symbol)
        
        # Add symbols from dynamic discovery if enabled (config + dashboard toggle)
        from src.feature_flags import is_active

        discovery_on = is_active("dynamic_discovery")
        if self.dynamic_discovery and self.dynamic_discovery.config.enabled and discovery_on:
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
        
        min_signal_strength = config.get('min_signal_strength', 0.2)
        
        # Check minimum signal strength
        if signal['strength'] < min_signal_strength:
            return False
        
        # Boost confidence if both technical and sentiment signals agree
        if config.get('use_available_signals', True):
            has_technical = bool(signal.get('technical_indicators'))
            has_sentiment = bool(signal.get('sentiment_data'))
            
            if has_technical and has_sentiment and config.get('confidence_boost_both', 0) > 0:
                original_confidence = signal['confidence']
                boosted_confidence = min(1.0, original_confidence + config.get('confidence_boost_both', 0))
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

        if getattr(self, "unusual_whales_analyzer", None):
            try:
                self.unusual_whales_analyzer.cleanup()
            except Exception as e:
                logger.debug("Unusual Whales cleanup failed", error=str(e))
        
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

    def request_stop(signum, frame):
        logger.info("Received stop signal", signal=signum)
        bot.is_running = False

    signal.signal(signal.SIGTERM, request_stop)
    
    # Run the bot
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())