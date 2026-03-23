"""
Compounder — Daily reset, compound equity tracking, and streak management.

Handles the 00:00 UTC daily rollover:
- Records final equity as previous day's close
- Calculates whether daily target was achieved
- Sets new day_open_equity (compound base)
- Updates streak counters
- Auto-reduces target after consecutive misses
"""

import datetime
import logging
from typing import Optional

from config import BotConfig, CONFIG
from daily_target.tracker import DailyTargetState, DailyTargetTracker

logger = logging.getLogger(__name__)


class Compounder:
    """
    Manages daily equity compounding and streak tracking.

    The daily_equity history (persisted to SQLite) is the source of truth
    for the compound growth chart.
    """

    def __init__(self, tracker: DailyTargetTracker, config: BotConfig = CONFIG):
        self.tracker = tracker
        self.config = config
        self.dtc = config.daily_target
        self._last_reset_date: str = tracker.state.date

    def check_daily_reset(self, current_equity: float) -> Optional[dict]:
        """
        Check if a new UTC day has begun. If so, perform daily rollover.

        Args:
            current_equity: Current account equity (USD)

        Returns:
            Daily summary dict if reset occurred, None otherwise.
        """
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if today == self._last_reset_date:
            return None

        # New day! Record previous day and reset.
        summary = self._perform_reset(current_equity, today)
        self._last_reset_date = today
        return summary

    def _perform_reset(self, current_equity: float, new_date: str) -> dict:
        """
        Perform the daily rollover.

        1. Record previous day's results
        2. Update streak
        3. Check auto-target-reduction
        4. Reset tracker for new day
        """
        state = self.tracker.state

        # Previous day summary
        target_hit = state.pct_achieved >= 100.0
        actual_pct = 0.0
        if state.day_open_equity > 0:
            actual_pct = (
                (current_equity - state.day_open_equity)
                / state.day_open_equity * 100.0
            )

        summary = {
            "date": state.date,
            "open_equity": state.day_open_equity,
            "close_equity": current_equity,
            "target_pct": state.daily_target_pct,
            "actual_pct": round(actual_pct, 4),
            "target_hit": target_hit,
            "realized_pnl": state.realized_pnl_today,
            "trades": state.trades_today,
            "wins": state.wins_today,
            "losses": state.losses_today,
            "mode_at_close": state.mode.value,
            "streak": state.streak_days,
        }

        # Update streaks
        if target_hit:
            state.streak_days += 1
            state.miss_streak = 0
            logger.info(
                f"[Compounder] Target HIT on {state.date}! "
                f"Actual: {actual_pct:+.2f}% vs target {state.daily_target_pct}% | "
                f"Streak: {state.streak_days} days"
            )
            if state.streak_days == 7:
                logger.info(
                    f"[Compounder] 7-day streak! Excellent consistency."
                )
        else:
            state.miss_streak += 1
            state.streak_days = 0
            logger.warning(
                f"[Compounder] Target MISSED on {state.date}. "
                f"Actual: {actual_pct:+.2f}% vs target {state.daily_target_pct}% | "
                f"Miss streak: {state.miss_streak} days"
            )

        summary["streak"] = state.streak_days
        summary["miss_streak"] = state.miss_streak

        # Auto-reduce target after consecutive misses
        new_target_pct = self._check_target_adjustment(state)

        # Determine new day's opening equity (compound base)
        new_equity = current_equity  # Use current equity (compounding)

        # Reset the tracker for the new day
        self.tracker.reset_day(new_equity, new_target_pct)

        logger.info(
            f"[Compounder] Day reset: {new_date} | "
            f"New base: ${new_equity:.2f} | "
            f"Target: {self.tracker.state.daily_target_pct}% = "
            f"${self.tracker.state.daily_target_amount:.2f}"
        )

        return summary

    def _check_target_adjustment(self, state: DailyTargetState) -> Optional[float]:
        """
        Check if target should be auto-reduced or restored.

        Returns new target % or None to keep current.
        """
        if not self.dtc.auto_target_reduction:
            return None

        # Severe miss streak → reduce to 1%
        if state.miss_streak >= self.dtc.miss_severe_days:
            if state.daily_target_pct > 1.0:
                logger.warning(
                    f"[Compounder] {state.miss_streak} consecutive misses! "
                    f"Auto-reducing target to 1.0%. "
                    f"Review strategy and market conditions."
                )
                return 1.0

        # Moderate miss streak → reduce by configured %
        if state.miss_streak >= self.dtc.miss_reduce_days:
            reduction = 1.0 - (self.dtc.miss_reduce_pct / 100.0)
            new_target = state.daily_target_pct * reduction
            new_target = max(new_target, 0.5)  # Never below 0.5%
            if new_target < state.daily_target_pct:
                logger.warning(
                    f"[Compounder] {state.miss_streak} consecutive misses. "
                    f"Auto-reducing target: {state.daily_target_pct}% → "
                    f"{new_target:.1f}%"
                )
                return new_target

        # Restore check: if at reduced target and hitting it consistently
        if (state.target_was_reduced
                and state.streak_days >= self.dtc.restore_streak_days):
            original = state.original_target_pct
            logger.info(
                f"[Compounder] {state.streak_days}-day streak at reduced target! "
                f"Restoring original target: {original}%"
            )
            return original

        return None

    def get_compound_projection(self, days: int = 30) -> list:
        """
        Project compound growth for N days at current target %.

        Returns list of {day, equity} projections.
        """
        equity = self.tracker.state.current_equity
        target = self.tracker.state.daily_target_pct / 100.0
        projection = []
        for day in range(1, days + 1):
            equity *= (1.0 + target)
            projection.append({
                "day": day,
                "projected_equity": round(equity, 4),
            })
        return projection
