"""
Backtesting Orchestrator

Entry point for running historical simulations of the QuantumSentiment trading strategy.
Provides proper historical simulation without lookahead bias.
"""

import asyncio
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog

from ..configuration import load_config, Config
from ..data.fetcher import DataFetcher
from ..main import QuantumSentimentBot
from .simulated_broker import SimulatedBroker
from .performance_report import PerformanceAnalyzer

logger = structlog.get_logger(__name__)


class BacktestRunner:
    """Orchestrates historical simulation backtests"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the backtest runner
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_fetcher = DataFetcher(self.config)
        self.historical_data = None
        self.simulated_broker = None
        self.bot = None
        self.performance_analyzer = None
        
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Run a complete historical backtest
        
        Args:
            symbols: List of symbols to trade
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            initial_capital: Starting capital for simulation
            
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        logger.info(
            "Starting backtest",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # 1. Load historical data
        await self._load_historical_data(symbols, start_date, end_date)
        
        # 2. Initialize simulated broker
        self.simulated_broker = SimulatedBroker(
            initial_capital=initial_capital,
            commission_per_share=0.005,  # $0.005 per share
            slippage_bps=5  # 5 basis points slippage
        )
        
        # 3. Initialize bot with simulated broker
        self.bot = QuantumSentimentBot(
            config_path=None,
            mode="backtest",
            config=self.config,
            broker=self.simulated_broker
        )
        await self.bot.initialize()
        
        # 4. Run the historical simulation
        await self._run_simulation(symbols)
        
        # 5. Generate performance analysis
        self.performance_analyzer = PerformanceAnalyzer(
            trades=self.simulated_broker.trade_history,
            equity_curve=self.simulated_broker.equity_history,
            initial_capital=initial_capital
        )
        
        results = self.performance_analyzer.generate_report()
        
        logger.info(
            "Backtest completed",
            total_trades=len(self.simulated_broker.trade_history),
            final_equity=self.simulated_broker.equity,
            total_return=results['total_return']
        )
        
        return results
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> None:
        """Load historical market data for the backtest period"""
        logger.info("Loading historical data", symbols=symbols)
        
        # Convert string dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Load data for all symbols
        all_data = {}
        for symbol in symbols:
            try:
                data = await self.data_fetcher.fetch_bars(
                    symbol=symbol,
                    timeframe='1Hour',
                    start=start_dt,
                    end=end_dt,
                    limit=None
                )
                
                if data is not None and not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No historical data available for any symbols")
            
        self.historical_data = all_data
        
        # Set up simulated broker with historical data
        if hasattr(self, 'simulated_broker') and self.simulated_broker:
            self.simulated_broker.set_historical_data(all_data)
    
    async def _run_simulation(self, symbols: List[str]) -> None:
        """Run the bar-by-bar historical simulation"""
        logger.info("Starting historical simulation")
        
        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for symbol_data in self.historical_data.values():
            all_timestamps.update(symbol_data.index)
        
        # Sort timestamps chronologically
        sorted_timestamps = sorted(all_timestamps)
        
        logger.info(f"Simulating {len(sorted_timestamps)} time periods")
        
        # Run simulation bar-by-bar
        for i, timestamp in enumerate(sorted_timestamps):
            try:
                # Update simulated broker's current time
                self.simulated_broker.set_current_time(timestamp)
                
                # Run bot's trading cycle for this timestamp
                await self.bot._trading_cycle_backtest(symbols, timestamp)
                
                # Log progress periodically
                if i % 100 == 0:
                    progress = (i / len(sorted_timestamps)) * 100
                    logger.info(
                        f"Simulation progress: {progress:.1f}%",
                        timestamp=timestamp,
                        equity=self.simulated_broker.equity
                    )
                    
            except Exception as e:
                logger.error(f"Error during simulation at {timestamp}: {e}")
                continue
        
        logger.info("Historical simulation completed")


async def main():
    """Command line entry point for backtesting"""
    parser = argparse.ArgumentParser(description='Run QuantumSentiment Backtest')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], 
                       help='Symbols to trade')
    parser.add_argument('--start-date', required=True, 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD)')  
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    # Run backtest
    runner = BacktestRunner(config_path=args.config)
    results = await runner.run_backtest(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital
    )
    
    # Print summary
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())