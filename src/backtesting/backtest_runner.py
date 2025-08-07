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
import pytz
import os

from ..configuration import load_config, Config
from ..data.data_fetcher import DataFetcher
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
        
        # Initialize database manager for data fetcher
        from ..database import DatabaseManager
        self.db_manager = DatabaseManager()
        self.data_fetcher = DataFetcher(self.config, self.db_manager)
        
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
        
        # 1. Initialize simulated broker first
        self.simulated_broker = SimulatedBroker(
            initial_capital=initial_capital,
            commission_per_share=0.005,  # $0.005 per share
            slippage_bps=5  # 5 basis points slippage
        )
        
        # 2. Load historical data and set it in the broker
        await self._load_historical_data(symbols, start_date, end_date)
        
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
        
        report = self.performance_analyzer.generate_report()
        
        # Extract metrics for easy access
        metrics = report.get('metrics', {})
        results = {
            'total_return': metrics.get('total_return', 0.0),
            'annualized_return': metrics.get('annualized_return', 0.0),
            'volatility': metrics.get('volatility', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'sortino_ratio': metrics.get('sortino_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'calmar_ratio': metrics.get('calmar_ratio', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'total_trades': metrics.get('total_trades', 0),
            'winning_trades': metrics.get('winning_trades', 0),
            'losing_trades': metrics.get('losing_trades', 0),
            'full_report': report  # Include full report for detailed analysis
        }
        
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
        
        # We need extra historical data before the start date for indicators
        # Add 30 days of buffer data for technical indicators
        start_dt = pd.to_datetime(start_date) - timedelta(days=30)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
        
        # Store the actual simulation start date
        self.simulation_start_date = pd.to_datetime(start_date)
        
        # Load data from existing historical files
        all_data = {}
        for symbol in symbols:
            try:
                # Try different file patterns - match our production data format
                import glob
                pattern_paths = [
                    f"data/historical/{symbol}/{symbol}_Hour_*.csv",     # Production format
                    f"data/historical/{symbol}/{symbol}_15Min_*.csv",   # Production format
                    f"data/historical/{symbol}/{symbol}_Day_*.csv",     # Production format
                    f"data/historical/{symbol}/{symbol}_1Hour_*.csv",   # Legacy format
                    f"data/historical/{symbol}/{symbol}_1Day_*.csv"     # Legacy format
                ]
                
                data = None
                for pattern in pattern_paths:
                    files = glob.glob(pattern)
                    if files:
                        # Try each file to find one that covers our date range
                        file_path = None
                        for candidate_file in files:
                            # Quick check if this file might contain our date range
                            try:
                                # Check first and last lines to get date range
                                with open(candidate_file, 'r') as f:
                                    lines = f.readlines()
                                    if len(lines) > 2:  # Header + at least 2 data lines
                                        first_date_str = lines[1].split(',')[0]
                                        last_date_str = lines[-1].split(',')[0]
                                        first_date = pd.to_datetime(first_date_str, utc=True)
                                        last_date = pd.to_datetime(last_date_str, utc=True)
                                        
                                        # Check if our requested range overlaps with file range
                                        start_dt_utc = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
                                        end_dt_utc = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
                                        
                                        if (start_dt_utc <= last_date and end_dt_utc >= first_date):
                                            file_path = candidate_file
                                            logger.info(f"Selected file {file_path} covering {first_date} to {last_date}")
                                            break
                            except Exception:
                                continue
                        
                        if not file_path:
                            # Fallback to the largest file (most comprehensive)
                            file_path = max(files, key=lambda f: os.path.getsize(f))
                            logger.info(f"Using largest file as fallback: {file_path}")
                        logger.info(f"Loading data from {file_path}")
                        
                        try:
                            data = pd.read_csv(file_path)
                            # Parse timestamp as timezone-aware (UTC), then let pandas handle comparisons
                            data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
                            data.set_index('timestamp', inplace=True)
                            
                            # Filter data to requested range - convert filter dates to UTC for comparison
                            logger.info(f"Data shape before filter: {data.shape}")
                            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                            logger.info(f"Filter range: {start_dt} to {end_dt}")
                            
                            # Make filter dates timezone-aware (UTC) to match data
                            start_dt_utc = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
                            end_dt_utc = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
                            
                            # Filter data to requested range
                            data = data[(data.index >= start_dt_utc) & (data.index <= end_dt_utc)]
                            logger.info(f"Data shape after filter: {data.shape}")
                            if not data.empty:
                                logger.info(f"Successfully loaded data for {symbol}: {len(data)} bars from {data.index.min()} to {data.index.max()}")
                                break
                            else:
                                logger.warning(f"Data filtered to empty for {symbol} in range {start_dt_utc} to {end_dt_utc}")
                                data = None  # Reset to None so we try next pattern
                                
                        except Exception as e:
                            logger.error(f"Error loading data from {file_path}: {e}")
                            data = None
                            continue
                
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
        for symbol, symbol_data in self.historical_data.items():
            logger.info(f"Symbol {symbol} has {len(symbol_data)} data points from {symbol_data.index.min()} to {symbol_data.index.max()}")
            all_timestamps.update(symbol_data.index)
        
        # Sort timestamps chronologically
        sorted_timestamps = sorted(all_timestamps)
        
        # Filter to only simulate from the actual start date (not the buffer period)
        if hasattr(self, 'simulation_start_date'):
            simulation_start_utc = self.simulation_start_date.tz_localize('UTC') if self.simulation_start_date.tz is None else self.simulation_start_date.tz_convert('UTC')
            sorted_timestamps = [ts for ts in sorted_timestamps if ts >= simulation_start_utc]
        
        logger.info(f"Simulating {len(sorted_timestamps)} time periods from {sorted_timestamps[0] if sorted_timestamps else 'N/A'} to {sorted_timestamps[-1] if sorted_timestamps else 'N/A'}")
        
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