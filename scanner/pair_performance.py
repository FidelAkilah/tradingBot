"""
Pair Performance Tracker — tracks per-pair win rate, avg P&L, and
auto-enables/disables pairs based on historical performance.

Reads trade history from the database (or an in-memory fallback)
and computes rolling statistics per symbol.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PairStats:
    """Rolling performance statistics for a single pair."""
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0               # 0-100
    total_pnl_usd: float = 0.0
    avg_pnl_usd: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    profit_factor: float = 0.0
    contribution_pct: float = 0.0       # Pct of total bot P&L from this pair
    disabled: bool = False
    disable_reason: str = ""
    auto_included: bool = False         # Flagged for auto-include by scanner


class PairPerformanceTracker:
    """
    Tracks per-pair trading performance and auto-disables/includes pairs.

    Auto-disable: pair with <35% win rate over last 20 trades → disabled
    Auto-include: pair with >60% win rate appears in scanner → flagged

    Works with both database trades and in-memory trade records.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.scanner
        self._stats: Dict[str, PairStats] = {}
        self._trade_pnls: Dict[str, List[float]] = {}  # Per-symbol PnL history
        self._last_update: float = 0.0

    @property
    def stats(self) -> Dict[str, PairStats]:
        return dict(self._stats)

    def update_from_trades(self, trades: List[dict]):
        """
        Update pair stats from a list of trade records.

        Args:
            trades: List of trade dicts with at minimum:
                    symbol, pnl_usd, is_open (0/1)
                    Most recent trades should be last.
        """
        # Group trades by symbol (closed only)
        by_symbol: Dict[str, List[dict]] = {}
        for t in trades:
            if t.get("is_open", 0):
                continue
            sym = t.get("symbol", "")
            if not sym:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append(t)

        total_bot_pnl = sum(
            t.get("pnl_usd", 0) for t in trades if not t.get("is_open", 0)
        )

        lookback = self.sc.perf_lookback_trades

        for sym, sym_trades in by_symbol.items():
            # Take last N trades
            recent = sym_trades[-lookback:]
            stats = PairStats(symbol=sym)
            stats.total_trades = len(recent)

            pnl_values = [t.get("pnl_usd", 0) for t in recent]
            stats.wins = sum(1 for p in pnl_values if p > 0)
            stats.losses = sum(1 for p in pnl_values if p <= 0)
            stats.total_pnl_usd = sum(pnl_values)
            stats.win_rate = (stats.wins / stats.total_trades * 100.0
                              if stats.total_trades > 0 else 0.0)
            stats.avg_pnl_usd = (stats.total_pnl_usd / stats.total_trades
                                 if stats.total_trades > 0 else 0.0)

            win_pnls = [p for p in pnl_values if p > 0]
            loss_pnls = [p for p in pnl_values if p <= 0]
            stats.avg_win_usd = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
            stats.avg_loss_usd = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0

            gross_profit = sum(win_pnls)
            gross_loss = abs(sum(loss_pnls))
            stats.profit_factor = (gross_profit / gross_loss
                                   if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0)

            if abs(total_bot_pnl) > 0:
                stats.contribution_pct = (stats.total_pnl_usd / abs(total_bot_pnl)) * 100.0

            # Auto-disable check
            min_trades = self.sc.perf_min_trades_for_disable
            if (stats.total_trades >= min_trades
                    and stats.win_rate < self.sc.perf_disable_wr_below):
                stats.disabled = True
                stats.disable_reason = (
                    f"win rate {stats.win_rate:.1f}% < {self.sc.perf_disable_wr_below}% "
                    f"over {stats.total_trades} trades"
                )

            # Auto-include flag
            if stats.win_rate >= self.sc.perf_auto_include_wr_above and stats.total_trades >= 5:
                stats.auto_included = True

            self._stats[sym] = stats

        self._last_update = time.time()

    def update_from_db(self):
        """Update stats from database trade history."""
        try:
            import database as db
            trades = db.get_trades(limit=500)  # Last 500 trades
            self.update_from_trades(trades)
        except Exception as e:
            logger.debug(f"PairPerformanceTracker: DB update error: {e}")

    def get_disabled_pairs(self) -> List[str]:
        """Return list of pairs that should be disabled."""
        return [sym for sym, s in self._stats.items() if s.disabled]

    def get_auto_include_pairs(self) -> List[str]:
        """Return list of pairs flagged for auto-inclusion."""
        return [sym for sym, s in self._stats.items() if s.auto_included]

    def get_pair_stats(self, symbol: str) -> Optional[PairStats]:
        """Get stats for a specific pair."""
        return self._stats.get(symbol)

    def record_trade(self, symbol: str, pnl_usd: float):
        """Record a single trade result (for real-time updates without DB query)."""
        if symbol not in self._stats:
            self._stats[symbol] = PairStats(symbol=symbol)
        if symbol not in self._trade_pnls:
            self._trade_pnls[symbol] = []

        self._trade_pnls[symbol].append(pnl_usd)

        stats = self._stats[symbol]
        stats.total_trades += 1
        stats.total_pnl_usd += pnl_usd
        if pnl_usd > 0:
            stats.wins += 1
        else:
            stats.losses += 1
        stats.win_rate = (stats.wins / stats.total_trades * 100.0
                          if stats.total_trades > 0 else 0.0)
        stats.avg_pnl_usd = (stats.total_pnl_usd / stats.total_trades
                              if stats.total_trades > 0 else 0.0)

        # Compute profit factor from trade history
        pnls = self._trade_pnls[symbol]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p <= 0))
        stats.profit_factor = (gross_profit / gross_loss
                               if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0)

        # Re-check disable/include
        min_trades = self.sc.perf_min_trades_for_disable
        stats.disabled = (
            stats.total_trades >= min_trades
            and stats.win_rate < self.sc.perf_disable_wr_below
        )
        if stats.disabled:
            stats.disable_reason = (
                f"win rate {stats.win_rate:.1f}% < {self.sc.perf_disable_wr_below}% "
                f"over {stats.total_trades} trades"
            )
        else:
            stats.disable_reason = ""

        stats.auto_included = (
            stats.win_rate >= self.sc.perf_auto_include_wr_above
            and stats.total_trades >= 5
        )

    def get_summary(self) -> dict:
        """Get JSON-serializable summary for dashboard."""
        pairs = []
        for sym, s in sorted(self._stats.items(), key=lambda x: x[1].total_pnl_usd, reverse=True):
            pairs.append({
                "symbol": sym,
                "total_trades": s.total_trades,
                "wins": s.wins,
                "losses": s.losses,
                "win_rate": round(s.win_rate, 1),
                "total_pnl_usd": round(s.total_pnl_usd, 4),
                "avg_pnl_usd": round(s.avg_pnl_usd, 4),
                "profit_factor": round(s.profit_factor, 2) if s.profit_factor != float('inf') else 999.0,
                "contribution_pct": round(s.contribution_pct, 1),
                "disabled": s.disabled,
                "disable_reason": s.disable_reason,
                "auto_included": s.auto_included,
            })
        return {
            "pairs": pairs,
            "disabled_count": sum(1 for s in self._stats.values() if s.disabled),
            "auto_include_count": sum(1 for s in self._stats.values() if s.auto_included),
            "last_update": self._last_update,
        }
