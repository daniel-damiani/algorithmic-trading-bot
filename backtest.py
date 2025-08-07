#!/usr/bin/env python3
"""
QuantumSentiment Backtest Entry Point

Command-line entry point for running historical backtests of the trading strategy.
This replaces the flawed BACKTEST mode in main.py with proper historical simulation.

Usage:
    python backtest.py --symbols AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31 --capital 10000

Examples:
    # Basic backtest
    python backtest.py --start-date 2024-01-01 --end-date 2024-06-30

    # Custom symbols and capital
    python backtest.py --symbols AAPL GOOGL TSLA --start-date 2024-01-01 --end-date 2024-12-31 --capital 50000

    # Use custom config
    python backtest.py --config config/config_small_data.yaml --start-date 2024-01-01 --end-date 2024-06-30
"""

import asyncio
import sys
import logging
from pathlib import Path
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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

from src.backtesting.backtest_runner import main

if __name__ == "__main__":
    asyncio.run(main())