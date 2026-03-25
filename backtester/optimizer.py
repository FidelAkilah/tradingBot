"""
Walk-Forward Optimizer — Parameter optimization with overfit detection.

Splits data into in-sample (70%) and out-of-sample (30%), optimizes
parameters on in-sample, validates on out-of-sample, and flags
degradation >30% as potential overfitting.
"""

import copy
import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from config import BotConfig, CONFIG
from backtester.data_manager import DataManager
from backtester.engine import BacktestEngine, BacktestResult
from backtester.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

# Default optimizable parameters and their config paths
OPTIMIZABLE_PARAMS = {
    "confidence_threshold": {
        "path": "candle.adx_trending_threshold",
        "description": "Minimum ADX for trend confirmation",
        "note": "Affects adx_trending_threshold; confidence min is gate logic",
    },
    "atr_tp_multiplier": {
        "path": "candle.atr_tp_multiplier",
        "description": "ATR multiplier for take-profit distance",
    },
    "atr_sl_multiplier": {
        "path": "candle.atr_sl_multiplier",
        "description": "ATR multiplier for stop-loss distance",
    },
    "trailing_stop_mult": {
        "path": "exit.chandelier_atr_mult",
        "description": "Chandelier trailing stop ATR multiplier",
    },
    "tp1_atr_mult": {
        "path": "exit.tp1_atr_mult",
        "description": "TP1 (partial) ATR multiplier",
    },
    "tp2_atr_mult": {
        "path": "exit.tp2_atr_mult",
        "description": "TP2 (partial) ATR multiplier",
    },
    "min_post_fee_rr": {
        "path": "trading.min_post_fee_rr",
        "description": "Minimum post-fee risk-reward ratio",
    },
    "leverage": {
        "path": "futures.leverage",
        "description": "Default leverage multiplier",
    },
    "adx_trending_threshold": {
        "path": "candle.adx_trending_threshold",
        "description": "ADX threshold to classify as trending",
    },
    "adx_ranging_threshold": {
        "path": "candle.adx_ranging_threshold",
        "description": "ADX threshold below which is ranging",
    },
}


@dataclass
class OptimizationRun:
    """Result from a single parameter combination."""
    params: Dict[str, Any]
    in_sample: Optional[PerformanceMetrics] = None
    out_of_sample: Optional[PerformanceMetrics] = None
    is_score: float = 0.0
    oos_score: float = 0.0
    degradation_pct: float = 0.0
    is_overfit: bool = False


@dataclass
class OptimizationResult:
    """Complete result from a walk-forward optimization."""
    param_name: str
    param_values: List[Any]
    runs: List[OptimizationRun] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_is_score: float = 0.0
    best_oos_score: float = 0.0
    overfit_warning: bool = False
    in_sample_period: str = ""
    out_of_sample_period: str = ""


def _score_metrics(m: PerformanceMetrics) -> float:
    """
    Score a backtest result for optimization ranking.
    Balances profit with risk: Sharpe-weighted profit factor.
    """
    if m.total_trades < 5:
        return -999.0

    # Penalize very low trade counts
    trade_penalty = min(1.0, m.total_trades / 20.0)

    # Core score: profit factor * sqrt(trade_count) * sharpe
    pf = min(m.profit_factor, 10.0)  # Cap to avoid outlier domination
    sharpe = max(m.sharpe_ratio, -5.0)
    wr = m.win_rate

    score = (
        pf * trade_penalty
        * (1 + sharpe * 0.3)
        * (1 - m.max_drawdown_pct / 100 * 0.5)
    )

    return score


class WalkForwardOptimizer:
    """
    Walk-forward parameter optimizer with overfit detection.

    Splits data into in-sample (70%) and out-of-sample (30%),
    runs backtests across parameter grid, and validates results.
    """

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        data_manager: Optional[DataManager] = None,
        is_ratio: float = 0.70,
        overfit_threshold: float = 0.30,
    ):
        self.base_config = config or copy.deepcopy(CONFIG)
        self.dm = data_manager or DataManager()
        self.is_ratio = is_ratio
        self.overfit_threshold = overfit_threshold

    def optimize(
        self,
        param_name: str,
        param_range: List[Any],
        symbols: List[str],
        start_date: str,
        end_date: str,
        primary_tf: str = "1h",
    ) -> OptimizationResult:
        """
        Run walk-forward optimization for a single parameter.

        Args:
            param_name: Parameter name (key in OPTIMIZABLE_PARAMS or dot-path)
            param_range: List of values to test
            symbols: Trading pairs to test
            start_date: Full data start date "YYYY-MM-DD"
            end_date: Full data end date "YYYY-MM-DD"
            primary_tf: Primary timeframe for stepping

        Returns:
            OptimizationResult with all runs and best params
        """
        # Resolve parameter path
        param_path = param_name
        if param_name in OPTIMIZABLE_PARAMS:
            param_path = OPTIMIZABLE_PARAMS[param_name]["path"]

        # Split dates into in-sample and out-of-sample
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        is_days = int(total_days * self.is_ratio)
        is_end_dt = start_dt + timedelta(days=is_days)
        oos_start_dt = is_end_dt + timedelta(days=1)

        is_start = start_date
        is_end = is_end_dt.strftime("%Y-%m-%d")
        oos_start = oos_start_dt.strftime("%Y-%m-%d")
        oos_end = end_date

        logger.info(
            f"Walk-forward optimization: {param_name}\n"
            f"  In-sample:     {is_start} to {is_end} ({is_days} days)\n"
            f"  Out-of-sample: {oos_start} to {oos_end} ({total_days - is_days} days)\n"
            f"  Values to test: {param_range}"
        )

        result = OptimizationResult(
            param_name=param_name,
            param_values=param_range,
            in_sample_period=f"{is_start} to {is_end}",
            out_of_sample_period=f"{oos_start} to {oos_end}",
        )

        # Run backtests for each parameter value
        for i, value in enumerate(param_range):
            logger.info(f"  [{i + 1}/{len(param_range)}] {param_name} = {value}")

            overrides = {param_path: value}

            # In-sample run
            is_engine = BacktestEngine(
                config=copy.deepcopy(self.base_config),
                data_manager=self.dm,
            )
            is_result = is_engine.run(
                symbols=symbols,
                start_date=is_start,
                end_date=is_end,
                primary_tf=primary_tf,
                param_overrides=overrides,
            )

            # Out-of-sample run
            oos_engine = BacktestEngine(
                config=copy.deepcopy(self.base_config),
                data_manager=self.dm,
            )
            oos_result = oos_engine.run(
                symbols=symbols,
                start_date=oos_start,
                end_date=oos_end,
                primary_tf=primary_tf,
                param_overrides=overrides,
            )

            is_score = _score_metrics(is_result.metrics) if is_result.metrics else -999
            oos_score = _score_metrics(oos_result.metrics) if oos_result.metrics else -999

            # Check for overfitting
            degradation = 0.0
            is_overfit = False
            if is_score > 0:
                degradation = (is_score - oos_score) / is_score
                is_overfit = degradation > self.overfit_threshold

            run = OptimizationRun(
                params={param_name: value},
                in_sample=is_result.metrics,
                out_of_sample=oos_result.metrics,
                is_score=is_score,
                oos_score=oos_score,
                degradation_pct=degradation * 100,
                is_overfit=is_overfit,
            )
            result.runs.append(run)

            logger.info(
                f"    IS score={is_score:.2f}, OOS score={oos_score:.2f}, "
                f"degradation={degradation:.1%}"
                f"{' ⚠️ OVERFIT' if is_overfit else ''}"
            )

        # Find best parameter (by OOS score, excluding overfit)
        valid_runs = [r for r in result.runs if not r.is_overfit and r.oos_score > -999]
        if not valid_runs:
            valid_runs = result.runs  # Fallback to all if everything is overfit

        if valid_runs:
            best = max(valid_runs, key=lambda r: r.oos_score)
            result.best_params = best.params
            result.best_is_score = best.is_score
            result.best_oos_score = best.oos_score
            result.overfit_warning = best.is_overfit

        # Flag if ALL non-trivial runs show overfitting
        non_trivial = [r for r in result.runs if r.is_score > 0]
        if non_trivial and all(r.is_overfit for r in non_trivial):
            result.overfit_warning = True
            logger.warning(
                f"⚠️  ALL parameter values show >30% OOS degradation. "
                f"This parameter may not be robust to optimize."
            )

        return result

    def grid_optimize(
        self,
        param_grid: Dict[str, List[Any]],
        symbols: List[str],
        start_date: str,
        end_date: str,
        primary_tf: str = "1h",
    ) -> List[OptimizationRun]:
        """
        Run a multi-parameter grid search.

        Args:
            param_grid: Dict of {param_path: [values]} for grid search
            symbols: Trading pairs
            start_date: Start date
            end_date: End date

        Returns:
            List of OptimizationRun sorted by OOS score (best first)
        """
        # Build grid
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combos = list(itertools.product(*param_values))

        # Resolve paths
        resolved_names = []
        for name in param_names:
            if name in OPTIMIZABLE_PARAMS:
                resolved_names.append(OPTIMIZABLE_PARAMS[name]["path"])
            else:
                resolved_names.append(name)

        # Split dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        is_days = int(total_days * self.is_ratio)
        is_end_dt = start_dt + timedelta(days=is_days)
        oos_start_dt = is_end_dt + timedelta(days=1)

        is_start = start_date
        is_end = is_end_dt.strftime("%Y-%m-%d")
        oos_start = oos_start_dt.strftime("%Y-%m-%d")
        oos_end = end_date

        logger.info(
            f"Grid optimization: {len(combos)} combinations\n"
            f"  Parameters: {param_names}\n"
            f"  IS: {is_start} to {is_end}, OOS: {oos_start} to {oos_end}"
        )

        runs = []
        for i, combo in enumerate(combos):
            overrides = dict(zip(resolved_names, combo))
            display_params = dict(zip(param_names, combo))

            logger.info(f"  [{i + 1}/{len(combos)}] {display_params}")

            # IS run
            is_engine = BacktestEngine(
                config=copy.deepcopy(self.base_config),
                data_manager=self.dm,
            )
            is_result = is_engine.run(
                symbols=symbols,
                start_date=is_start,
                end_date=is_end,
                primary_tf=primary_tf,
                param_overrides=overrides,
            )

            # OOS run
            oos_engine = BacktestEngine(
                config=copy.deepcopy(self.base_config),
                data_manager=self.dm,
            )
            oos_result = oos_engine.run(
                symbols=symbols,
                start_date=oos_start,
                end_date=oos_end,
                primary_tf=primary_tf,
                param_overrides=overrides,
            )

            is_score = _score_metrics(is_result.metrics) if is_result.metrics else -999
            oos_score = _score_metrics(oos_result.metrics) if oos_result.metrics else -999

            degradation = 0.0
            is_overfit = False
            if is_score > 0:
                degradation = (is_score - oos_score) / is_score
                is_overfit = degradation > self.overfit_threshold

            runs.append(OptimizationRun(
                params=display_params,
                in_sample=is_result.metrics,
                out_of_sample=oos_result.metrics,
                is_score=is_score,
                oos_score=oos_score,
                degradation_pct=degradation * 100,
                is_overfit=is_overfit,
            ))

        # Sort by OOS score
        runs.sort(key=lambda r: r.oos_score, reverse=True)
        return runs


def parse_param_range(range_str: str) -> List[float]:
    """
    Parse a parameter range string like "0.4,0.8,0.05" into a list of values.
    Format: "start,end,step"
    """
    parts = range_str.split(",")
    if len(parts) == 3:
        start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        values = []
        v = start
        while v <= end + step * 0.001:  # Small epsilon for float comparison
            values.append(round(v, 6))
            v += step
        return values
    elif len(parts) > 1:
        return [float(x.strip()) for x in parts]
    else:
        return [float(parts[0])]
