"""
Backtesting Module for QuantumSentiment Trading System

This module provides historical simulation capabilities for strategy validation
without risking capital or relying on live market conditions.
"""

from .backtest_runner import BacktestRunner
from .simulated_broker import SimulatedBroker
from .performance_report import PerformanceAnalyzer

__all__ = ['BacktestRunner', 'SimulatedBroker', 'PerformanceAnalyzer']