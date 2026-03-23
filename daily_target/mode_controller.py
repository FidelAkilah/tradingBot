"""
Mode Controller — Trading mode transitions based on daily progress.

Manages 4 trading modes:
- NORMAL: Standard trading (0-60% of target achieved)
- AGGRESSIVE: Behind schedule, increase risk slightly
- PROTECTING: Near/at target, lock in gains
- HALTED: Daily loss limit hit, no new entries
"""

import logging
from typing import Optional

from config import BotConfig, CONFIG
from daily_target.tracker import DailyTargetState, TradingMode

logger = logging.getLogger(__name__)


class ModeController:
    """
    Determines and manages trading mode transitions.

    Called on every state update to evaluate whether the mode should change.
    All transitions are logged with full context.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.dtc = config.daily_target
        self._last_mode = TradingMode.NORMAL

    def evaluate(self, state: DailyTargetState,
                 regime_is_trending: bool = True) -> TradingMode:
        """
        Evaluate current state and return the appropriate trading mode.

        Args:
            state: Current DailyTargetState
            regime_is_trending: Whether current market regime favors trading

        Returns:
            The new TradingMode
        """
        old_mode = state.mode
        new_mode = self._determine_mode(state, regime_is_trending)

        if new_mode != old_mode:
            self._log_transition(old_mode, new_mode, state, regime_is_trending)

        return new_mode

    def _determine_mode(self, state: DailyTargetState,
                        regime_is_trending: bool) -> TradingMode:
        """Core mode determination logic."""

        # HALTED takes priority — only resets at start of new day
        if state.mode == TradingMode.HALTED:
            return TradingMode.HALTED

        # Check daily loss limit → HALTED
        if (state.daily_loss_limit > 0
                and state.total_pnl_today < 0
                and abs(state.total_pnl_today) >= state.daily_loss_limit):
            return TradingMode.HALTED

        # PROTECTING: enter at >80% achieved, stay until <60%
        if state.pct_achieved >= self.dtc.protecting_trigger_pct:
            return TradingMode.PROTECTING

        # If currently PROTECTING, stay unless dropped below 60%
        if state.mode == TradingMode.PROTECTING:
            if state.pct_achieved >= 60.0:
                return TradingMode.PROTECTING
            return TradingMode.NORMAL

        # AGGRESSIVE: behind schedule AND enough of day elapsed
        if (self.dtc.aggressive_mode_enabled
                and state.pct_achieved < self.dtc.aggressive_trigger_pct
                and state.day_elapsed_pct > self.dtc.aggressive_time_trigger
                and regime_is_trending
                and state.daily_loss_consumed_pct < 30.0):
            return TradingMode.AGGRESSIVE

        # If currently AGGRESSIVE, stay unless conditions no longer met
        if state.mode == TradingMode.AGGRESSIVE:
            # Exit AGGRESSIVE if target progress improved past trigger
            if state.pct_achieved >= self.dtc.aggressive_trigger_pct:
                return TradingMode.NORMAL
            # Exit AGGRESSIVE if loss limit is getting close
            if state.daily_loss_consumed_pct >= 30.0:
                return TradingMode.NORMAL
            # Exit AGGRESSIVE if regime turned unfavorable
            if not regime_is_trending:
                return TradingMode.NORMAL
            # Stay in AGGRESSIVE
            return TradingMode.AGGRESSIVE

        return TradingMode.NORMAL

    def _log_transition(self, old: TradingMode, new: TradingMode,
                        state: DailyTargetState, regime_trending: bool):
        """Log every mode transition with full context."""
        context = (
            f"pct_achieved={state.pct_achieved:.1f}% "
            f"day_elapsed={state.day_elapsed_pct:.1%} "
            f"realized=${state.realized_pnl_today:+.2f} "
            f"unrealized=${state.unrealized_pnl:+.2f} "
            f"loss_consumed={state.daily_loss_consumed_pct:.1f}% "
            f"regime_trending={regime_trending} "
            f"trades={state.trades_today} W:{state.wins_today} L:{state.losses_today}"
        )

        if new == TradingMode.HALTED:
            logger.critical(
                f"[MODE] {old.value} → HALTED | {context}"
            )
        elif new == TradingMode.AGGRESSIVE:
            logger.warning(
                f"[MODE] {old.value} → AGGRESSIVE | Behind schedule. {context}"
            )
        elif new == TradingMode.PROTECTING:
            logger.info(
                f"[MODE] {old.value} → PROTECTING | Locking in gains. {context}"
            )
        else:
            logger.info(
                f"[MODE] {old.value} → NORMAL | {context}"
            )

    def force_halt(self, state: DailyTargetState, reason: str) -> TradingMode:
        """Force transition to HALTED mode (e.g., drawdown circuit breaker)."""
        old = state.mode
        logger.critical(f"[MODE] {old.value} → HALTED (forced) | Reason: {reason}")
        return TradingMode.HALTED
