"""
Daily Target Tracker — Core state for daily profit target management.

Tracks realized P&L, unrealized P&L, daily progress toward target,
and coordinates with ModeController for trading mode transitions.
"""

import datetime
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on daily progress."""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    PROTECTING = "protecting"
    HALTED = "halted"


@dataclass
class DailyTargetState:
    """Current daily target tracking state."""
    date: str = ""                         # UTC date YYYY-MM-DD
    day_open_equity: float = 0.0           # Balance at 00:00 UTC
    current_equity: float = 0.0            # Real-time balance
    daily_target_pct: float = 2.0          # Configurable (default 2.0)
    daily_target_amount: float = 0.0       # day_open_equity * daily_target_pct / 100
    target_equity: float = 0.0            # day_open_equity + daily_target_amount
    realized_pnl_today: float = 0.0        # Sum of closed trade P&L today
    unrealized_pnl: float = 0.0            # Current open position P&L
    total_pnl_today: float = 0.0           # realized + unrealized
    pct_achieved: float = 0.0              # total_pnl_today / daily_target_amount * 100
    daily_loss_limit: float = 0.0          # Max acceptable loss today
    daily_loss_consumed_pct: float = 0.0   # How much of loss limit has been used
    mode: TradingMode = TradingMode.NORMAL
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    streak_days: int = 0                   # Consecutive days hitting target
    miss_streak: int = 0                   # Consecutive missed days
    original_target_pct: float = 2.0       # Before auto-reduction
    target_was_reduced: bool = False        # Flag if target was auto-reduced
    day_elapsed_pct: float = 0.0           # How much of the trading day has elapsed

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "day_open_equity": round(self.day_open_equity, 4),
            "current_equity": round(self.current_equity, 4),
            "daily_target_pct": round(self.daily_target_pct, 2),
            "daily_target_amount": round(self.daily_target_amount, 4),
            "target_equity": round(self.target_equity, 4),
            "realized_pnl_today": round(self.realized_pnl_today, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "total_pnl_today": round(self.total_pnl_today, 4),
            "pct_achieved": round(self.pct_achieved, 2),
            "daily_loss_limit": round(self.daily_loss_limit, 4),
            "daily_loss_consumed_pct": round(self.daily_loss_consumed_pct, 2),
            "mode": self.mode.value,
            "trades_today": self.trades_today,
            "wins_today": self.wins_today,
            "losses_today": self.losses_today,
            "streak_days": self.streak_days,
            "miss_streak": self.miss_streak,
            "original_target_pct": round(self.original_target_pct, 2),
            "target_was_reduced": self.target_was_reduced,
            "day_elapsed_pct": round(self.day_elapsed_pct, 2),
        }


@dataclass
class DailyTargetContext:
    """Daily target state bundled for the position sizer."""
    target_hit: bool = False              # pct_achieved >= 100%
    pct_achieved: float = 0.0
    day_elapsed_pct: float = 0.0
    remaining_target_pct: float = 100.0
    daily_target_pct: float = 2.0
    mode: TradingMode = TradingMode.NORMAL
    intraday_dd_pct: float = 0.0          # Drawdown from day-open equity
    behind_schedule: bool = False


class DailyTargetTracker:
    """
    Tracks daily progress toward the profit target.

    Central coordinator that:
    - Tracks realized + unrealized P&L against daily target
    - Computes daily loss limits (asymmetric by design)
    - Provides integration hooks for position sizer and risk manager
    - Delegates mode transitions to ModeController
    - Delegates daily reset to Compounder
    """

    def __init__(self, initial_equity: float, config: BotConfig = CONFIG):
        self.config = config
        self.dtc = config.daily_target

        # Validate and warn about target
        self._validate_target(self.dtc.daily_target_pct)

        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        target_amount = initial_equity * self.dtc.daily_target_pct / 100.0
        loss_limit = target_amount * self.dtc.daily_loss_limit_pct / 100.0

        self.state = DailyTargetState(
            date=today,
            day_open_equity=initial_equity,
            current_equity=initial_equity,
            daily_target_pct=self.dtc.daily_target_pct,
            daily_target_amount=target_amount,
            target_equity=initial_equity + target_amount,
            daily_loss_limit=loss_limit,
            original_target_pct=self.dtc.daily_target_pct,
        )

        logger.info(
            f"Daily target initialized: {self.dtc.daily_target_pct}% "
            f"= ${target_amount:.2f} on ${initial_equity:.2f} equity | "
            f"Loss limit: ${loss_limit:.2f}"
        )

    @staticmethod
    def _validate_target(target_pct: float):
        """Warn or block on dangerous daily targets."""
        if target_pct > 10.0:
            raise ValueError(
                f"ERROR: {target_pct}% daily target is not achievable with "
                f"sustainable risk management. Maximum allowed: 10%. "
                f"Recommended: 1-3%."
            )
        if target_pct > 5.0:
            logger.warning(
                f"WARNING: Daily target of {target_pct}% requires extremely "
                f"favorable conditions and high leverage. At this target, the "
                f"probability of consecutive losing days causing significant "
                f"drawdown is high. Recommended target: 1-3%."
            )

    # ─────────────────────────────────────────
    # REAL-TIME UPDATES
    # ─────────────────────────────────────────

    def record_trade(self, pnl_usd: float):
        """
        Record a closed trade result. Updates realized P&L and counters.
        Called after every trade close.
        """
        self.state.realized_pnl_today += pnl_usd
        self.state.trades_today += 1

        if pnl_usd > 0:
            self.state.wins_today += 1
        else:
            self.state.losses_today += 1

        self._recalculate()

        logger.info(
            f"[DailyTarget] Trade recorded: ${pnl_usd:+.2f} | "
            f"Realized: ${self.state.realized_pnl_today:+.2f} | "
            f"Target: {self.state.pct_achieved:.1f}% achieved | "
            f"Mode: {self.state.mode.value}"
        )

    def update_unrealized(self, unrealized_pnl: float):
        """
        Update unrealized P&L from open positions.
        Called on every price tick (or periodically).
        """
        self.state.unrealized_pnl = unrealized_pnl
        self._recalculate()

    def update_equity(self, current_equity: float):
        """Update current equity from external source."""
        self.state.current_equity = current_equity
        self._recalculate()

    def _recalculate(self):
        """Recalculate derived fields after any state change."""
        self.state.total_pnl_today = (
            self.state.realized_pnl_today + self.state.unrealized_pnl
        )

        if self.state.daily_target_amount > 0:
            self.state.pct_achieved = (
                self.state.total_pnl_today / self.state.daily_target_amount * 100.0
            )
        else:
            self.state.pct_achieved = 0.0

        # Loss limit consumption
        if self.state.daily_loss_limit > 0 and self.state.total_pnl_today < 0:
            self.state.daily_loss_consumed_pct = (
                abs(self.state.total_pnl_today) / self.state.daily_loss_limit * 100.0
            )
        else:
            self.state.daily_loss_consumed_pct = 0.0

        # Day elapsed (0.0 at 00:00 UTC, 1.0 at 23:59 UTC)
        now_utc = datetime.datetime.utcnow()
        seconds_into_day = (
            now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second
        )
        self.state.day_elapsed_pct = seconds_into_day / 86400.0

    # ─────────────────────────────────────────
    # INTEGRATION HOOKS
    # ─────────────────────────────────────────

    def get_position_size_multiplier(self) -> float:
        """
        Multiplier applied to position sizing based on current mode.

        Called by position_sizer.py.
        """
        if self.state.mode == TradingMode.PROTECTING:
            return self.dtc.protecting_size_mult  # 0.60
        if self.state.mode == TradingMode.HALTED:
            return 0.0
        return 1.0  # NORMAL and AGGRESSIVE use standard sizing

    def get_leverage_adjustment(self) -> int:
        """
        Extra leverage tiers allowed in AGGRESSIVE mode.

        Returns additional leverage steps (0 normally, 1 in AGGRESSIVE).
        Applied as: max_leverage_for_trade = normal_max + adjustment
        """
        if self.state.mode == TradingMode.AGGRESSIVE:
            return 1  # Allow one extra leverage tier
        return 0

    def get_confidence_threshold(self) -> float:
        """
        Minimum confidence threshold for current mode.

        Returns the minimum confidence score required to enter a trade.
        """
        if self.state.mode == TradingMode.AGGRESSIVE:
            return self.dtc.aggressive_confidence_min  # 0.50
        if self.state.mode == TradingMode.PROTECTING:
            return self.dtc.protecting_confidence_min  # 0.70
        return 0.55  # NORMAL default

    def get_max_positions(self) -> int:
        """Max concurrent positions for current mode."""
        if self.state.mode == TradingMode.AGGRESSIVE:
            return self.dtc.aggressive_max_positions  # 3
        return self.config.trading.max_open_positions  # default 2

    def get_trailing_stop_multiplier(self) -> float:
        """
        Multiplier for trailing stop distance.

        In PROTECTING mode, tighten stops by 20% (multiply distance by 0.8).
        """
        if self.state.mode == TradingMode.PROTECTING:
            return self.dtc.protecting_trailing_tighten  # 0.80
        return 1.0

    def should_halt(self) -> tuple:
        """
        Check if daily loss limit has been hit.

        Returns: (should_halt: bool, reason: str)
        Called by risk manager.
        """
        if self.state.mode == TradingMode.HALTED:
            return True, f"Daily target HALTED: loss limit consumed"

        if (self.state.daily_loss_limit > 0
                and self.state.total_pnl_today < 0
                and abs(self.state.total_pnl_today) >= self.state.daily_loss_limit):
            return True, (
                f"Daily loss limit hit: ${abs(self.state.total_pnl_today):.2f} "
                f">= ${self.state.daily_loss_limit:.2f} limit"
            )

        return False, "OK"

    def should_move_stops_to_breakeven(self) -> bool:
        """In PROTECTING mode, move stops to breakeven if possible."""
        return self.state.mode == TradingMode.PROTECTING

    def get_sizing_context(self) -> DailyTargetContext:
        """Build context object for the position sizer."""
        s = self.state
        remaining = max(0.0, 100.0 - s.pct_achieved)

        intraday_dd = 0.0
        if s.day_open_equity > 0:
            intraday_dd = max(0.0,
                (s.day_open_equity - s.current_equity) / s.day_open_equity * 100.0)

        behind = (
            s.pct_achieved < self.dtc.aggressive_trigger_pct
            and s.day_elapsed_pct > self.dtc.aggressive_time_trigger
        )

        return DailyTargetContext(
            target_hit=s.pct_achieved >= 100.0,
            pct_achieved=s.pct_achieved,
            day_elapsed_pct=s.day_elapsed_pct,
            remaining_target_pct=remaining,
            daily_target_pct=s.daily_target_pct,
            mode=s.mode,
            intraday_dd_pct=intraday_dd,
            behind_schedule=behind,
        )

    def force_target(self, new_target_pct: float):
        """Force daily target to a specific percentage (drawdown protection)."""
        if new_target_pct < self.state.daily_target_pct:
            old = self.state.daily_target_pct
            self.state.daily_target_pct = new_target_pct
            self.state.daily_target_amount = (
                self.state.day_open_equity * new_target_pct / 100.0
            )
            self.state.target_equity = (
                self.state.day_open_equity + self.state.daily_target_amount
            )
            self._recalculate()
            logger.warning(
                f"[DailyTarget] Target forced: {old:.1f}% → {new_target_pct:.1f}% "
                f"(drawdown protection)"
            )

    def get_daily_progress(self) -> dict:
        """Full progress report for dashboard API."""
        self._recalculate()
        return self.state.to_dict()

    # ─────────────────────────────────────────
    # DAILY RESET
    # ─────────────────────────────────────────

    def reset_day(self, new_equity: float, new_target_pct: Optional[float] = None):
        """
        Reset for a new trading day. Called by Compounder at 00:00 UTC.

        Args:
            new_equity: New day's opening equity (from wallet or tracked)
            new_target_pct: Override target % (for auto-reduction)
        """
        target_pct = new_target_pct if new_target_pct is not None else self.state.daily_target_pct
        target_amount = new_equity * target_pct / 100.0
        loss_limit = target_amount * self.dtc.daily_loss_limit_pct / 100.0

        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        self.state = DailyTargetState(
            date=today,
            day_open_equity=new_equity,
            current_equity=new_equity,
            daily_target_pct=target_pct,
            daily_target_amount=target_amount,
            target_equity=new_equity + target_amount,
            daily_loss_limit=loss_limit,
            streak_days=self.state.streak_days,
            miss_streak=self.state.miss_streak,
            original_target_pct=self.state.original_target_pct,
            target_was_reduced=target_pct < self.state.original_target_pct,
        )

        logger.info(
            f"[DailyTarget] New day: {today} | "
            f"Equity: ${new_equity:.2f} | "
            f"Target: {target_pct}% = ${target_amount:.2f} | "
            f"Loss limit: ${loss_limit:.2f} | "
            f"Streak: {self.state.streak_days}d"
        )
