"""
Backtesting framework for the Binance USDT-M Futures Swing Trading Bot.

Modules:
  data_manager — Historical kline download & SQLite storage
  engine       — Event-driven backtesting engine
  metrics      — Performance metrics (Sharpe, Sortino, drawdown, etc.)
  optimizer    — Walk-forward parameter optimization
  report       — HTML report generation

CLI:
  python -m backtester sync-data --days 90
  python -m backtester run --start 2025-01-01 --end 2025-03-20
  python -m backtester optimize --param confidence_threshold --range 0.4,0.8,0.05
  python -m backtester report --run-id <id>
"""

from backtester.data_manager import DataManager
from backtester.engine import BacktestEngine, BacktestResult
from backtester.metrics import compute_metrics, PerformanceMetrics
from backtester.optimizer import WalkForwardOptimizer
from backtester.report import generate_report

__all__ = [
    "DataManager",
    "BacktestEngine",
    "BacktestResult",
    "compute_metrics",
    "PerformanceMetrics",
    "WalkForwardOptimizer",
    "generate_report",
]
