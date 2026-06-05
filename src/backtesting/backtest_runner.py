"""
Backtesting Orchestrator

Entry point for running historical simulations of the QuantumSentiment trading strategy.
Provides proper historical simulation without lookahead bias.
"""

import asyncio
import argparse
import glob
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog
import pytz
import os

from ..configuration import load_config, Config
from ..data.data_fetcher import DataFetcher
from ..data.alpaca_client import AlpacaClient
from ..main import QuantumSentimentBot
from .simulated_broker import SimulatedBroker
from .performance_report import PerformanceAnalyzer

logger = structlog.get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
        self._parquet_cache: Optional[pd.DataFrame] = None
        self.alpaca_client = AlpacaClient()
        
    async def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        output_dir: Optional[str] = None,
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

        if output_dir:
            self._save_results(output_dir, results, symbols, start_date, end_date, initial_capital)
        
        return results

    def _save_results(
        self,
        output_dir: str,
        results: Dict[str, Any],
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> None:
        """Persist backtest results for dashboard consumption."""
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        equity_series = []
        for point in self.simulated_broker.equity_history:
            ts = point.get("timestamp")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            equity_series.append({"timestamp": str(ts), "equity": float(point.get("equity", 0))})

        payload = {
            k: v for k, v in results.items() if k != "full_report"
        }
        payload["equity_series"] = equity_series
        payload["symbols"] = symbols
        payload["start_date"] = start_date
        payload["end_date"] = end_date
        payload["initial_capital"] = initial_capital

        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        status_path = out / "status.json"
        status = {"state": "done", "updated_at": datetime.utcnow().isoformat() + "Z"}
        if status_path.exists():
            try:
                with open(status_path, encoding="utf-8") as f:
                    status.update(json.load(f))
            except Exception:
                pass
        status["state"] = "done"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

    def _to_utc(self, dt: pd.Timestamp) -> pd.Timestamp:
        if dt.tz is None:
            return dt.tz_localize("UTC")
        return dt.tz_convert("UTC")

    def _filter_date_range(
        self, data: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        start_utc = self._to_utc(start_dt)
        end_utc = self._to_utc(end_dt)
        idx = data.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        filtered = data.copy()
        filtered.index = idx
        return filtered[(filtered.index >= start_utc) & (filtered.index <= end_utc)]

    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df.sort_index()
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return pd.DataFrame()
        return df[required]

    def _load_symbol_from_csv(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        pattern_paths = [
            f"data/historical/{symbol}/{symbol}_Hour_*.csv",
            f"data/historical/{symbol}/{symbol}_15Min_*.csv",
            f"data/historical/{symbol}/{symbol}_Day_*.csv",
            f"data/historical/{symbol}/{symbol}_1Hour_*.csv",
            f"data/historical/{symbol}/{symbol}_1Day_*.csv",
            f"data/training/production/{symbol}/*combined*.csv",
        ]

        for pattern in pattern_paths:
            files = glob.glob(str(PROJECT_ROOT / pattern))
            if not files:
                continue

            file_path = max(files, key=lambda f: os.path.getsize(f))
            try:
                data = pd.read_csv(file_path)
                data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
                data = data.set_index("timestamp")
                filtered = self._filter_date_range(data, start_dt, end_dt)
                normalized = self._normalize_ohlcv(filtered)
                if not normalized.empty:
                    logger.info(
                        "Loaded CSV data",
                        symbol=symbol,
                        file=str(file_path),
                        bars=len(normalized),
                    )
                    return normalized
            except Exception as e:
                logger.error("Error loading CSV", symbol=symbol, file=file_path, error=str(e))
        return pd.DataFrame()

    def _get_parquet_data(self) -> Optional[pd.DataFrame]:
        if self._parquet_cache is not None:
            return self._parquet_cache

        parquet_path = PROJECT_ROOT / "data" / "historical_data_quality.parquet"
        if not parquet_path.exists():
            return None

        df = pd.read_parquet(parquet_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        self._parquet_cache = df
        return df

    def _load_symbol_from_parquet(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        df = self._get_parquet_data()
        if df is None or df.empty or "symbol" not in df.columns:
            return pd.DataFrame()

        symbol_df = df[df["symbol"] == symbol].copy()
        if symbol_df.empty:
            return pd.DataFrame()

        normalized = self._normalize_ohlcv(symbol_df)
        filtered = self._filter_date_range(normalized, start_dt, end_dt)
        if not filtered.empty:
            logger.info("Loaded parquet data", symbol=symbol, bars=len(filtered))
        return filtered

    def _load_symbol_from_database(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        start_naive = start_dt.to_pydatetime().replace(tzinfo=None)
        end_naive = end_dt.to_pydatetime().replace(tzinfo=None)

        for timeframe in ("1Hour", "1h", "1Hour".lower()):
            df = self.db_manager.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_naive,
                end_date=end_naive,
            )
            normalized = self._normalize_ohlcv(df)
            filtered = self._filter_date_range(normalized, start_dt, end_dt)
            if not filtered.empty:
                logger.info("Loaded database data", symbol=symbol, bars=len(filtered))
                return filtered
        return pd.DataFrame()

    def _fetch_symbol_from_alpaca(
        self, symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp
    ) -> pd.DataFrame:
        logger.info("Fetching from Alpaca", symbol=symbol)
        chunks: List[pd.DataFrame] = []
        current_start = start_dt.to_pydatetime().replace(tzinfo=None)
        end_naive = end_dt.to_pydatetime().replace(tzinfo=None)

        while current_start < end_naive:
            chunk = self.alpaca_client.get_bars(
                symbol=symbol,
                timeframe="1Hour",
                start=current_start,
                end=end_naive,
                limit=10000,
            )
            if chunk is None or chunk.empty:
                break

            chunk = self._normalize_ohlcv(chunk.reset_index())
            chunks.append(chunk)
            last_ts = chunk.index.max()
            next_start = (last_ts + pd.Timedelta(hours=1)).to_pydatetime()
            if next_start.tzinfo is not None:
                next_start = next_start.replace(tzinfo=None)
            if next_start <= current_start or len(chunk) < 1000:
                break
            current_start = next_start

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        filtered = self._filter_date_range(combined, start_dt, end_dt)
        if not filtered.empty:
            logger.info("Fetched Alpaca data", symbol=symbol, bars=len(filtered))
        return filtered
    
    async def _load_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> None:
        """Load historical market data for the backtest period"""
        logger.info("Loading historical data", symbols=symbols)
        
        start_dt = pd.to_datetime(start_date) - timedelta(days=30)
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
        self.simulation_start_date = pd.to_datetime(start_date)
        
        all_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                loaders = (
                    self._load_symbol_from_csv,
                    self._load_symbol_from_parquet,
                    self._load_symbol_from_database,
                    self._fetch_symbol_from_alpaca,
                )
                data = pd.DataFrame()
                for loader in loaders:
                    data = loader(symbol, start_dt, end_dt)
                    if data is not None and not data.empty:
                        break

                if data is not None and not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError(
                "No historical data available for any symbols. "
                "Run: python training/scripts/prepare_quality_data.py "
                "or ensure Alpaca credentials can fetch the requested date range."
            )
            
        self.historical_data = all_data
        
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
    parser.add_argument('--output-dir', help='Directory to write results.json for dashboard')
    
    args = parser.parse_args()
    
    out_dir = args.output_dir
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        import json
        status_file = out_path / "status.json"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"state": "running", "updated_at": datetime.utcnow().isoformat() + "Z"}, f)
    
    try:
        # Run backtest
        runner = BacktestRunner(config_path=args.config)
        results = await runner.run_backtest(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            output_dir=out_dir,
        )
    except Exception as e:
        if out_dir:
            import json
            err_path = Path(out_dir) / "status.json"
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": "error",
                        "error": str(e),
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    },
                    f,
                )
        raise
    
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