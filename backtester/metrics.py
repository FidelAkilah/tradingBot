"""
Performance Metrics — Comprehensive trade and portfolio analytics.

Computes: win rate, profit factor, Sharpe, Sortino, Calmar, max drawdown,
recovery factor, streaks, fee impact, and per-category breakdowns.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DrawdownInfo:
    """Details about a drawdown period."""
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    drawdown_pct: float = 0.0
    start_ts: float = 0.0
    trough_ts: float = 0.0
    recovery_ts: Optional[float] = None
    duration_candles: int = 0

    @property
    def recovered(self) -> bool:
        return self.recovery_ts is not None


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a backtest run."""

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Rates
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0
    avg_rr: float = 0.0
    expectancy: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_duration_candles: int = 0
    drawdowns: List[DrawdownInfo] = field(default_factory=list)

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Time metrics
    trades_per_day: float = 0.0
    avg_hold_duration_s: float = 0.0
    total_days: float = 0.0

    # Streaks
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    current_streak: int = 0
    current_streak_type: str = ""

    # Recovery
    recovery_factor: float = 0.0

    # Fee impact
    total_fees: float = 0.0
    gross_pnl_before_fees: float = 0.0
    fee_drag_pct: float = 0.0

    # Equity
    starting_equity: float = 0.0
    ending_equity: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    # Per-category breakdowns
    by_symbol: Dict[str, dict] = field(default_factory=dict)
    by_direction: Dict[str, dict] = field(default_factory=dict)
    by_regime: Dict[str, dict] = field(default_factory=dict)
    by_session: Dict[str, dict] = field(default_factory=dict)
    by_exit_reason: Dict[str, dict] = field(default_factory=dict)

    # Monthly returns
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    # Best/worst trades
    best_trade_pnl: float = 0.0
    best_trade_id: str = ""
    worst_trade_pnl: float = 0.0
    worst_trade_id: str = ""


def compute_metrics(
    trades: List[dict],
    equity_curve: List[Tuple[float, float]],
    starting_equity: float,
    risk_free_rate: float = 0.04,
) -> PerformanceMetrics:
    """
    Compute full performance metrics from backtest results.

    Args:
        trades: List of trade dicts with keys:
            trade_id, symbol, side, entry_price, exit_price, pnl_usd,
            gross_pnl_usd, fee_cost_usd, entry_time, exit_time,
            exit_reason, regime, session, post_fee_rr, duration_s
        equity_curve: List of (timestamp, equity_value) tuples
        starting_equity: Initial portfolio value in USD
        risk_free_rate: Annual risk-free rate for Sharpe (default 4%)

    Returns:
        PerformanceMetrics with all fields populated
    """
    m = PerformanceMetrics()
    m.starting_equity = starting_equity

    if not trades:
        m.ending_equity = starting_equity
        return m

    # ── Basic counts ────────────────────────────────────────────
    m.total_trades = len(trades)
    pnls = [t["pnl_usd"] for t in trades]

    for pnl in pnls:
        if pnl > 0.001:
            m.winning_trades += 1
            m.gross_profit += pnl
        elif pnl < -0.001:
            m.losing_trades += 1
            m.gross_loss += abs(pnl)
        else:
            m.breakeven_trades += 1

    m.total_pnl = sum(pnls)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0
    m.loss_rate = m.losing_trades / m.total_trades if m.total_trades > 0 else 0

    # ── Profit factor ───────────────────────────────────────────
    m.profit_factor = (
        m.gross_profit / m.gross_loss if m.gross_loss > 0 else
        (999.0 if m.gross_profit > 0 else 0.0)
    )

    # ── Averages ────────────────────────────────────────────────
    m.avg_pnl = m.total_pnl / m.total_trades if m.total_trades > 0 else 0
    m.avg_win = (
        m.gross_profit / m.winning_trades if m.winning_trades > 0 else 0
    )
    m.avg_loss = (
        m.gross_loss / m.losing_trades if m.losing_trades > 0 else 0
    )
    m.avg_rr = m.avg_win / m.avg_loss if m.avg_loss > 0 else 0
    m.expectancy = (m.win_rate * m.avg_win) - (m.loss_rate * m.avg_loss)

    # ── Fee impact ──────────────────────────────────────────────
    m.total_fees = sum(t.get("fee_cost_usd", 0) or 0 for t in trades)
    m.gross_pnl_before_fees = sum(
        t.get("gross_pnl_usd", t["pnl_usd"] + (t.get("fee_cost_usd", 0) or 0))
        for t in trades
    )
    if m.gross_pnl_before_fees != 0:
        m.fee_drag_pct = (m.total_fees / abs(m.gross_pnl_before_fees)) * 100

    # ── Streaks ─────────────────────────────────────────────────
    win_streak = 0
    loss_streak = 0
    current = 0
    current_type = ""

    for pnl in pnls:
        if pnl > 0.001:
            if current_type == "win":
                current += 1
            else:
                current = 1
                current_type = "win"
            win_streak = max(win_streak, current)
        elif pnl < -0.001:
            if current_type == "loss":
                current += 1
            else:
                current = 1
                current_type = "loss"
            loss_streak = max(loss_streak, current)
        # breakeven doesn't break streaks

    m.longest_win_streak = win_streak
    m.longest_loss_streak = loss_streak
    m.current_streak = current
    m.current_streak_type = current_type

    # ── Time metrics ────────────────────────────────────────────
    first_ts = trades[0].get("entry_time", 0)
    last_ts = trades[-1].get("exit_time", trades[-1].get("entry_time", 0))
    m.total_days = max((last_ts - first_ts) / 86400, 1)
    m.trades_per_day = m.total_trades / m.total_days

    durations = [t.get("duration_s", 0) or 0 for t in trades]
    m.avg_hold_duration_s = sum(durations) / len(durations) if durations else 0

    # ── Equity & returns ────────────────────────────────────────
    m.ending_equity = equity_curve[-1][1] if equity_curve else starting_equity
    m.total_return_pct = (
        (m.ending_equity - starting_equity) / starting_equity * 100
        if starting_equity > 0 else 0
    )
    if m.total_days > 0:
        annual_factor = 365.0 / m.total_days
        total_return = m.ending_equity / starting_equity if starting_equity > 0 else 1
        m.annualized_return_pct = (total_return ** annual_factor - 1) * 100

    # ── Drawdown analysis ───────────────────────────────────────
    if equity_curve:
        _compute_drawdowns(m, equity_curve)

    # ── Risk-adjusted returns ───────────────────────────────────
    if equity_curve and len(equity_curve) >= 2:
        _compute_risk_ratios(m, equity_curve, risk_free_rate)

    # ── Recovery factor ─────────────────────────────────────────
    m.recovery_factor = (
        m.total_pnl / m.max_drawdown_usd
        if m.max_drawdown_usd > 0 else 0
    )

    # ── Best/worst trades ───────────────────────────────────────
    best = max(trades, key=lambda t: t["pnl_usd"])
    worst = min(trades, key=lambda t: t["pnl_usd"])
    m.best_trade_pnl = best["pnl_usd"]
    m.best_trade_id = str(best.get("trade_id", ""))
    m.worst_trade_pnl = worst["pnl_usd"]
    m.worst_trade_id = str(worst.get("trade_id", ""))

    # ── Per-category breakdowns ─────────────────────────────────
    m.by_symbol = _breakdown(trades, "symbol")
    m.by_direction = _breakdown(trades, "side")
    m.by_regime = _breakdown(trades, "regime")
    m.by_session = _breakdown(trades, "session")
    m.by_exit_reason = _breakdown(trades, "exit_reason")

    # ── Monthly returns ─────────────────────────────────────────
    _compute_monthly_returns(m, equity_curve)

    return m


def _compute_drawdowns(m: PerformanceMetrics, equity_curve: List[Tuple[float, float]]):
    """Compute drawdown statistics from equity curve."""
    peak = equity_curve[0][1]
    peak_ts = equity_curve[0][0]
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    current_dd: Optional[DrawdownInfo] = None
    all_drawdowns: List[DrawdownInfo] = []
    dd_candles = 0

    for ts, eq in equity_curve:
        if eq >= peak:
            # Recovered or new high
            if current_dd is not None:
                current_dd.recovery_ts = ts
                all_drawdowns.append(current_dd)
                current_dd = None
            peak = eq
            peak_ts = ts
            dd_candles = 0
        else:
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
            dd_usd = peak - eq

            if current_dd is None:
                current_dd = DrawdownInfo(
                    peak_equity=peak,
                    trough_equity=eq,
                    drawdown_pct=dd_pct,
                    start_ts=peak_ts,
                    trough_ts=ts,
                )
            elif dd_pct > current_dd.drawdown_pct:
                current_dd.trough_equity = eq
                current_dd.drawdown_pct = dd_pct
                current_dd.trough_ts = ts

            dd_candles += 1
            current_dd.duration_candles = dd_candles

            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd

    # Handle ongoing drawdown at end
    if current_dd is not None:
        all_drawdowns.append(current_dd)

    m.max_drawdown_pct = max_dd_pct
    m.max_drawdown_usd = max_dd_usd
    m.max_drawdown_duration_candles = max(
        (dd.duration_candles for dd in all_drawdowns), default=0
    )
    # Keep top 10 drawdowns
    m.drawdowns = sorted(
        all_drawdowns, key=lambda d: d.drawdown_pct, reverse=True
    )[:10]


def _compute_risk_ratios(
    m: PerformanceMetrics,
    equity_curve: List[Tuple[float, float]],
    risk_free_rate: float,
):
    """Compute Sharpe, Sortino, and Calmar ratios."""
    # Compute periodic returns
    equities = [eq for _, eq in equity_curve]
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            r = (equities[i] - equities[i - 1]) / equities[i - 1]
            returns.append(r)

    if not returns:
        return

    # Determine annualization factor based on data frequency
    total_seconds = equity_curve[-1][0] - equity_curve[0][0]
    avg_interval = total_seconds / len(returns) if returns else 3600
    periods_per_year = 365.25 * 86400 / avg_interval if avg_interval > 0 else 8760

    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - (risk_free_rate / periods_per_year)

    # Standard deviation of returns
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    # Sharpe ratio (annualized)
    if std_dev > 0:
        m.sharpe_ratio = excess_return / std_dev * math.sqrt(periods_per_year)

    # Sortino ratio — only downside deviation
    downside_returns = [r for r in returns if r < 0]
    if downside_returns:
        downside_var = sum(r ** 2 for r in downside_returns) / len(returns)
        downside_std = math.sqrt(downside_var)
        if downside_std > 0:
            m.sortino_ratio = excess_return / downside_std * math.sqrt(periods_per_year)

    # Calmar ratio = annualized return / max drawdown
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.annualized_return_pct / m.max_drawdown_pct


def _breakdown(trades: List[dict], key: str) -> Dict[str, dict]:
    """Break down trades by a category key."""
    groups: Dict[str, List[dict]] = {}
    for t in trades:
        val = t.get(key, "unknown") or "unknown"
        groups.setdefault(val, []).append(t)

    result = {}
    for val, group in groups.items():
        wins = sum(1 for t in group if t["pnl_usd"] > 0.001)
        losses = sum(1 for t in group if t["pnl_usd"] < -0.001)
        total_pnl = sum(t["pnl_usd"] for t in group)
        gross_profit = sum(t["pnl_usd"] for t in group if t["pnl_usd"] > 0)
        gross_loss = sum(abs(t["pnl_usd"]) for t in group if t["pnl_usd"] < 0)

        result[val] = {
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(group) if group else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(group) if group else 0,
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0
                else (999.0 if gross_profit > 0 else 0.0)
            ),
        }
    return result


def _compute_monthly_returns(
    m: PerformanceMetrics,
    equity_curve: List[Tuple[float, float]],
):
    """Compute monthly return percentages."""
    if not equity_curve:
        return

    from datetime import datetime, timezone

    # Group equity by month
    monthly_start: Dict[str, float] = {}
    monthly_end: Dict[str, float] = {}

    for ts, eq in equity_curve:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        key = dt.strftime("%Y-%m")
        if key not in monthly_start:
            monthly_start[key] = eq
        monthly_end[key] = eq

    for month in sorted(monthly_start.keys()):
        start_eq = monthly_start[month]
        end_eq = monthly_end[month]
        if start_eq > 0:
            m.monthly_returns[month] = (end_eq - start_eq) / start_eq * 100
