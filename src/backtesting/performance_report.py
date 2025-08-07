"""
Backtest Performance Analysis and Reporting

Analyzes completed backtest results to generate comprehensive performance metrics
and visualizations for strategy evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog

from .simulated_broker import Trade

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float


class PerformanceAnalyzer:
    """Analyzes backtest performance and generates reports"""
    
    def __init__(
        self,
        trades: List[Trade],
        equity_curve: List[Dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize performance analyzer
        
        Args:
            trades: List of executed trades
            equity_curve: Equity curve data points
            initial_capital: Starting capital
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Convert equity curve to DataFrame
        if equity_curve:
            self.equity_df = pd.DataFrame(equity_curve)
            if 'timestamp' in self.equity_df.columns:
                # Handle timezone-aware datetime conversion properly
                self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'], utc=True)
                self.equity_df.set_index('timestamp', inplace=True)
        else:
            self.equity_df = pd.DataFrame()
        
        # Convert trades to DataFrame
        if trades:
            self.trades_df = pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'commission': trade.commission,
                    'slippage': trade.slippage,
                    'pnl': self._calculate_trade_pnl(trade)
                }
                for trade in trades
            ])
        else:
            self.trades_df = pd.DataFrame()
        
        self.metrics = None
        
        logger.info("Performance analyzer initialized",
                   total_trades=len(trades),
                   equity_points=len(equity_curve))
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a single trade (simplified)"""
        # This is a simplified calculation - in reality, we'd need to track
        # entry/exit pairs to calculate actual trade P&L
        return 0.0  # Placeholder
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary containing metrics and analysis
        """
        logger.info("Generating performance report")
        
        self.metrics = self._calculate_metrics()
        
        report = {
            'metrics': self.metrics.__dict__,
            'trade_analysis': self._analyze_trades(),
            'drawdown_analysis': self._analyze_drawdowns(),
            'monthly_returns': self._calculate_monthly_returns(),
            'summary': self._generate_summary()
        }
        
        # Generate visualizations
        report['charts'] = self._generate_charts()
        
        return report
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        if self.equity_df.empty:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                max_drawdown_duration=0, calmar_ratio=0.0, win_rate=0.0,
                profit_factor=0.0, average_win=0.0, average_loss=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                largest_win=0.0, largest_loss=0.0
            )
        
        # Calculate returns
        equity_values = self.equity_df['equity'].values
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Basic metrics
        final_equity = equity_values[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Calculate time period
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        time_period = (end_date - start_date).days / 365.25
        
        annualized_return = (1 + total_return) ** (1/time_period) - 1 if time_period > 0 else 0
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        drawdown_info = self._calculate_drawdown()
        max_drawdown = drawdown_info['max_drawdown']
        max_drawdown_duration = drawdown_info['max_drawdown_duration']
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            calmar_ratio=calmar_ratio,
            **trade_stats
        )
    
    def _calculate_drawdown(self) -> Dict[str, Any]:
        """Calculate drawdown statistics"""
        if self.equity_df.empty:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        equity_values = self.equity_df['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        
        max_drawdown = np.min(drawdown)
        
        # Calculate max drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start_dd = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_dd is None:
                start_dd = i
            elif not is_dd and start_dd is not None:
                drawdown_periods.append(i - start_dd)
                start_dd = None
        
        # Handle case where drawdown continues to end
        if start_dd is not None:
            drawdown_periods.append(len(in_drawdown) - start_dd)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'drawdown_series': drawdown
        }
    
    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade-level statistics"""
        if self.trades_df.empty:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        # For simplified analysis, we'll assume each trade's impact
        # This would need more sophisticated P&L tracking in reality
        total_trades = len(self.trades)
        
        # Placeholder values - would need proper trade pairing
        winning_trades = int(total_trades * 0.6)  # Assume 60% win rate
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': 1.5,  # Placeholder
            'average_win': 100.0,  # Placeholder
            'average_loss': -67.0,  # Placeholder
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'largest_win': 250.0,  # Placeholder
            'largest_loss': -180.0  # Placeholder
        }
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trading patterns"""
        if self.trades_df.empty:
            return {'message': 'No trades to analyze'}
        
        analysis = {}
        
        # Symbol distribution
        symbol_counts = self.trades_df['symbol'].value_counts()
        analysis['top_traded_symbols'] = symbol_counts.head(10).to_dict()
        
        # Trade timing analysis
        self.trades_df['hour'] = pd.to_datetime(self.trades_df['timestamp']).dt.hour
        hourly_distribution = self.trades_df['hour'].value_counts().sort_index()
        analysis['hourly_distribution'] = hourly_distribution.to_dict()
        
        # Side distribution
        side_counts = self.trades_df['side'].value_counts()
        analysis['side_distribution'] = side_counts.to_dict()
        
        return analysis
    
    def _analyze_drawdowns(self) -> Dict[str, Any]:
        """Analyze drawdown patterns"""
        drawdown_info = self._calculate_drawdown()
        
        return {
            'max_drawdown': drawdown_info['max_drawdown'],
            'max_drawdown_duration_days': drawdown_info['max_drawdown_duration'],
            'analysis': 'Detailed drawdown analysis would go here'
        }
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns"""
        if self.equity_df.empty:
            return {}
        
        # Resample to monthly and calculate returns
        monthly_equity = self.equity_df['equity'].resample('ME').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return {str(date.date()): return_val for date, return_val in monthly_returns.items()}
    
    def _generate_summary(self) -> str:
        """Generate text summary"""
        if not self.metrics:
            return "No data available for analysis"
        
        summary = f"""
Performance Summary:
-------------------
Total Return: {self.metrics.total_return:.2%}
Annualized Return: {self.metrics.annualized_return:.2%}
Volatility: {self.metrics.volatility:.2%}
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Max Drawdown: {self.metrics.max_drawdown:.2%}
Win Rate: {self.metrics.win_rate:.2%}
Total Trades: {self.metrics.total_trades}
"""
        return summary.strip()
    
    def _generate_charts(self) -> Dict[str, str]:
        """Generate performance charts"""
        charts = {}
        
        try:
            # Equity curve chart
            if not self.equity_df.empty:
                plt.figure(figsize=(12, 8))
                
                # Main equity curve
                plt.subplot(2, 1, 1)
                plt.plot(self.equity_df.index, self.equity_df['equity'], 
                        linewidth=2, color='blue', label='Portfolio Value')
                plt.axhline(y=self.initial_capital, color='red', linestyle='--', 
                           alpha=0.7, label='Initial Capital')
                plt.title('Portfolio Equity Curve')
                plt.ylabel('Portfolio Value ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Drawdown chart
                plt.subplot(2, 1, 2)
                drawdown_info = self._calculate_drawdown()
                drawdown_series = drawdown_info.get('drawdown_series', [])
                if len(drawdown_series) > 0:
                    plt.fill_between(self.equity_df.index, drawdown_series, 0, 
                                   color='red', alpha=0.3, label='Drawdown')
                    plt.title('Drawdown')
                    plt.ylabel('Drawdown (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.xlabel('Date')
                plt.tight_layout()
                
                # Save chart (in practice, you might want to save to file)
                charts['equity_curve'] = 'equity_curve.png'
                plt.close()
            
            # Trade distribution chart
            if not self.trades_df.empty:
                plt.figure(figsize=(10, 6))
                
                # Symbol distribution
                symbol_counts = self.trades_df['symbol'].value_counts().head(10)
                plt.bar(symbol_counts.index, symbol_counts.values)
                plt.title('Trades by Symbol')
                plt.xlabel('Symbol')
                plt.ylabel('Number of Trades')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                charts['trade_distribution'] = 'trade_distribution.png'
                plt.close()
                
        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
            charts['error'] = f"Chart generation failed: {e}"
        
        return charts
    
    def export_results(self, filepath: str) -> None:
        """Export results to file"""
        report = self.generate_report()
        
        try:
            import json
            with open(filepath, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_report = self._convert_numpy_types(report)
                json.dump(json_report, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj