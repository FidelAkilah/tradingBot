"""
Tests for the daily target system — mode transitions, compound logic,
loss limits, and streak management.

Validates:
1. DailyTargetTracker state management and integration hooks
2. ModeController transitions (NORMAL/AGGRESSIVE/PROTECTING/HALTED)
3. Compounder daily reset, streak tracking, auto-target reduction
4. Loss limit enforcement (asymmetric by design)
5. Compound projection math
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from unittest.mock import patch
from config import BotConfig
from daily_target.tracker import DailyTargetTracker, DailyTargetState, DailyTargetContext, TradingMode
from daily_target.mode_controller import ModeController
from daily_target.compounder import Compounder


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.daily_target.daily_target_pct = 2.0
    cfg.daily_target.daily_loss_limit_pct = 50.0
    cfg.daily_target.aggressive_mode_enabled = True
    cfg.daily_target.aggressive_trigger_pct = 20.0
    cfg.daily_target.aggressive_time_trigger = 0.6
    cfg.daily_target.protecting_trigger_pct = 80.0
    cfg.daily_target.protecting_confidence_min = 0.70
    cfg.daily_target.protecting_size_mult = 0.60
    cfg.daily_target.protecting_trailing_tighten = 0.80
    cfg.daily_target.aggressive_confidence_min = 0.50
    cfg.daily_target.aggressive_max_positions = 3
    cfg.daily_target.auto_target_reduction = True
    cfg.daily_target.miss_reduce_days = 3
    cfg.daily_target.miss_severe_days = 5
    cfg.daily_target.miss_reduce_pct = 20.0
    cfg.daily_target.restore_streak_days = 5
    cfg.trading.max_open_positions = 2
    return cfg


@pytest.fixture
def tracker(config):
    return DailyTargetTracker(initial_equity=100.0, config=config)


@pytest.fixture
def mode_ctrl(config):
    return ModeController(config)


@pytest.fixture
def compounder(tracker, config):
    return Compounder(tracker, config)


# ─────────────────────────────────────────
# TRACKER — STATE MANAGEMENT
# ─────────────────────────────────────────

class TestDailyTargetTracker:
    def test_initialization(self, tracker):
        """Tracker initializes with correct target calculations."""
        s = tracker.state
        assert s.day_open_equity == 100.0
        assert s.daily_target_pct == 2.0
        assert s.daily_target_amount == 2.0  # 100 * 2%
        assert s.target_equity == 102.0      # 100 + 2
        assert s.daily_loss_limit == 1.0     # 2.0 * 50%
        assert s.mode == TradingMode.NORMAL

    def test_record_win(self, tracker):
        """Recording a winning trade updates P&L and counters."""
        tracker.record_trade(0.50)
        s = tracker.state
        assert s.realized_pnl_today == 0.50
        assert s.trades_today == 1
        assert s.wins_today == 1
        assert s.losses_today == 0
        assert s.pct_achieved == 25.0  # 0.50 / 2.00 * 100

    def test_record_loss(self, tracker):
        """Recording a losing trade updates counters."""
        tracker.record_trade(-0.30)
        s = tracker.state
        assert s.realized_pnl_today == -0.30
        assert s.losses_today == 1
        assert s.pct_achieved == -15.0  # -0.30 / 2.00 * 100

    def test_record_multiple_trades(self, tracker):
        """Multiple trades accumulate correctly."""
        tracker.record_trade(0.80)
        tracker.record_trade(-0.20)
        tracker.record_trade(0.60)
        s = tracker.state
        assert abs(s.realized_pnl_today - 1.20) < 0.01
        assert s.trades_today == 3
        assert s.wins_today == 2
        assert s.losses_today == 1
        assert abs(s.pct_achieved - 60.0) < 0.01

    def test_unrealized_pnl_updates(self, tracker):
        """Unrealized P&L affects total P&L and progress."""
        tracker.record_trade(0.50)  # +$0.50 realized
        tracker.update_unrealized(0.80)  # +$0.80 unrealized
        s = tracker.state
        assert s.total_pnl_today == 1.30
        assert abs(s.pct_achieved - 65.0) < 0.01

    def test_loss_limit_consumption(self, tracker):
        """Loss consumption percentage tracks correctly."""
        tracker.record_trade(-0.50)  # Loss of $0.50
        s = tracker.state
        # loss_limit = $1.00, consumed $0.50 = 50%
        assert abs(s.daily_loss_consumed_pct - 50.0) < 0.01

    def test_validate_target_blocks_above_10(self):
        """Targets above 10% are rejected."""
        with pytest.raises(ValueError, match="not achievable"):
            DailyTargetTracker._validate_target(11.0)

    def test_validate_target_warns_above_5(self, config):
        """Targets above 5% produce a warning but don't block."""
        config.daily_target.daily_target_pct = 7.0
        # Should not raise
        tracker = DailyTargetTracker(initial_equity=100.0, config=config)
        assert tracker.state.daily_target_pct == 7.0


# ─────────────────────────────────────────
# TRACKER — INTEGRATION HOOKS
# ─────────────────────────────────────────

class TestTrackerHooks:
    def test_position_size_multiplier_normal(self, tracker):
        assert tracker.get_position_size_multiplier() == 1.0

    def test_position_size_multiplier_protecting(self, tracker):
        tracker.state.mode = TradingMode.PROTECTING
        assert tracker.get_position_size_multiplier() == 0.60

    def test_position_size_multiplier_halted(self, tracker):
        tracker.state.mode = TradingMode.HALTED
        assert tracker.get_position_size_multiplier() == 0.0

    def test_leverage_adjustment_normal(self, tracker):
        assert tracker.get_leverage_adjustment() == 0

    def test_leverage_adjustment_aggressive(self, tracker):
        tracker.state.mode = TradingMode.AGGRESSIVE
        assert tracker.get_leverage_adjustment() == 1

    def test_confidence_threshold_normal(self, tracker):
        assert tracker.get_confidence_threshold() == 0.55

    def test_confidence_threshold_aggressive(self, tracker):
        tracker.state.mode = TradingMode.AGGRESSIVE
        assert tracker.get_confidence_threshold() == 0.50

    def test_confidence_threshold_protecting(self, tracker):
        tracker.state.mode = TradingMode.PROTECTING
        assert tracker.get_confidence_threshold() == 0.70

    def test_max_positions_normal(self, tracker):
        assert tracker.get_max_positions() == 2

    def test_max_positions_aggressive(self, tracker):
        tracker.state.mode = TradingMode.AGGRESSIVE
        assert tracker.get_max_positions() == 3

    def test_trailing_stop_mult_normal(self, tracker):
        assert tracker.get_trailing_stop_multiplier() == 1.0

    def test_trailing_stop_mult_protecting(self, tracker):
        tracker.state.mode = TradingMode.PROTECTING
        assert tracker.get_trailing_stop_multiplier() == 0.80

    def test_should_halt_on_loss_limit(self, tracker):
        """should_halt triggers when loss exceeds daily_loss_limit."""
        tracker.record_trade(-1.00)  # Loss = $1.00 = 100% of limit
        halt, reason = tracker.should_halt()
        assert halt
        assert "loss limit" in reason.lower()

    def test_should_not_halt_below_limit(self, tracker):
        tracker.record_trade(-0.50)  # Loss = 50% of $1.00 limit
        halt, _ = tracker.should_halt()
        assert not halt

    def test_should_halt_when_already_halted(self, tracker):
        tracker.state.mode = TradingMode.HALTED
        halt, _ = tracker.should_halt()
        assert halt

    def test_breakeven_stop_flag(self, tracker):
        tracker.state.mode = TradingMode.NORMAL
        assert not tracker.should_move_stops_to_breakeven()
        tracker.state.mode = TradingMode.PROTECTING
        assert tracker.should_move_stops_to_breakeven()


# ─────────────────────────────────────────
# LOSS LIMIT — ASYMMETRIC DESIGN
# ─────────────────────────────────────────

class TestLossLimit:
    def test_asymmetric_loss_limit(self, tracker):
        """Loss limit is 50% of target: can gain 2% but only lose 1%."""
        s = tracker.state
        # Target: gain $2.00 (2%)
        # Loss limit: $1.00 (1%)
        assert s.daily_target_amount == 2.0
        assert s.daily_loss_limit == 1.0

    def test_loss_limit_scales_with_equity(self, config):
        """Loss limit scales proportionally with equity."""
        tracker = DailyTargetTracker(initial_equity=500.0, config=config)
        s = tracker.state
        assert s.daily_target_amount == 10.0  # 500 * 2%
        assert s.daily_loss_limit == 5.0      # 10 * 50%

    def test_unrealized_counts_toward_loss_limit(self, tracker):
        """Unrealized losses count toward the daily loss limit."""
        tracker.update_unrealized(-0.80)
        s = tracker.state
        assert abs(s.daily_loss_consumed_pct - 80.0) < 0.01

    def test_combined_realized_unrealized_loss(self, tracker):
        """Combined realized + unrealized losses against limit."""
        tracker.record_trade(-0.40)
        tracker.update_unrealized(-0.30)
        s = tracker.state
        assert s.total_pnl_today == -0.70
        assert abs(s.daily_loss_consumed_pct - 70.0) < 0.01


# ─────────────────────────────────────────
# MODE CONTROLLER — TRANSITIONS
# ─────────────────────────────────────────

class TestModeTransitions:
    def test_starts_normal(self, mode_ctrl, tracker):
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.NORMAL

    def test_normal_to_protecting(self, mode_ctrl, tracker):
        """>80% of target achieved → PROTECTING."""
        tracker.record_trade(1.60)  # 80% of $2.00
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.PROTECTING

    def test_normal_to_protecting_exact_threshold(self, mode_ctrl, tracker):
        """Exactly 80% → PROTECTING."""
        tracker.record_trade(1.60)
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.PROTECTING

    def test_protecting_stays_above_60(self, mode_ctrl, tracker):
        """PROTECTING stays if progress is still >60%."""
        tracker.record_trade(1.60)
        tracker.state.mode = TradingMode.PROTECTING
        # Simulate small unrealized loss
        tracker.update_unrealized(-0.20)
        # pct_achieved = (1.60 - 0.20) / 2.0 * 100 = 70%
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.PROTECTING

    def test_protecting_to_normal_below_60(self, mode_ctrl, tracker):
        """PROTECTING → NORMAL if progress drops below 60%."""
        tracker.record_trade(1.60)
        tracker.state.mode = TradingMode.PROTECTING
        tracker.update_unrealized(-0.50)
        # pct_achieved = (1.60 - 0.50) / 2.0 * 100 = 55%
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.NORMAL

    def test_normal_to_aggressive(self, mode_ctrl, tracker):
        """Behind schedule + enough time elapsed + trending → AGGRESSIVE."""
        tracker.state.pct_achieved = 10.0  # <20% of target
        tracker.state.day_elapsed_pct = 0.7  # >60% of day
        tracker.state.daily_loss_consumed_pct = 0.0
        mode = mode_ctrl.evaluate(tracker.state, regime_is_trending=True)
        assert mode == TradingMode.AGGRESSIVE

    def test_aggressive_requires_trending(self, mode_ctrl, tracker):
        """AGGRESSIVE requires trending market regime."""
        tracker.state.pct_achieved = 10.0
        tracker.state.day_elapsed_pct = 0.7
        mode = mode_ctrl.evaluate(tracker.state, regime_is_trending=False)
        assert mode == TradingMode.NORMAL

    def test_aggressive_blocked_by_loss_consumed(self, mode_ctrl, tracker):
        """AGGRESSIVE blocked if >30% of daily loss consumed."""
        tracker.state.pct_achieved = 10.0
        tracker.state.day_elapsed_pct = 0.7
        tracker.state.daily_loss_consumed_pct = 35.0
        mode = mode_ctrl.evaluate(tracker.state, regime_is_trending=True)
        assert mode == TradingMode.NORMAL

    def test_aggressive_exits_when_progress_improves(self, mode_ctrl, tracker):
        """AGGRESSIVE → NORMAL when target progress catches up."""
        tracker.state.mode = TradingMode.AGGRESSIVE
        tracker.state.pct_achieved = 25.0  # >= 20% trigger
        tracker.state.day_elapsed_pct = 0.7
        mode = mode_ctrl.evaluate(tracker.state, regime_is_trending=True)
        assert mode == TradingMode.NORMAL

    def test_aggressive_disabled_by_config(self, config, tracker):
        """AGGRESSIVE doesn't activate when disabled in config."""
        config.daily_target.aggressive_mode_enabled = False
        mc = ModeController(config)
        tracker.state.pct_achieved = 10.0
        tracker.state.day_elapsed_pct = 0.7
        mode = mc.evaluate(tracker.state, regime_is_trending=True)
        assert mode == TradingMode.NORMAL

    def test_loss_limit_triggers_halt(self, mode_ctrl, tracker):
        """Exceeding daily loss limit → HALTED."""
        tracker.record_trade(-1.00)  # Hit the $1.00 loss limit
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.HALTED

    def test_halted_stays_halted(self, mode_ctrl, tracker):
        """HALTED persists until new day reset."""
        tracker.state.mode = TradingMode.HALTED
        tracker.record_trade(0.50)  # Even a win doesn't un-halt
        mode = mode_ctrl.evaluate(tracker.state)
        assert mode == TradingMode.HALTED

    def test_force_halt(self, mode_ctrl, tracker):
        """force_halt transitions to HALTED regardless of state."""
        tracker.state.mode = TradingMode.PROTECTING
        mode = mode_ctrl.force_halt(tracker.state, "drawdown circuit breaker")
        assert mode == TradingMode.HALTED

    def test_protecting_overrides_aggressive(self, mode_ctrl, tracker):
        """If target is >80% achieved, PROTECTING wins over AGGRESSIVE conditions."""
        tracker.state.pct_achieved = 85.0
        tracker.state.day_elapsed_pct = 0.7
        mode = mode_ctrl.evaluate(tracker.state, regime_is_trending=True)
        assert mode == TradingMode.PROTECTING


# ─────────────────────────────────────────
# COMPOUNDER — DAILY RESET
# ─────────────────────────────────────────

class TestCompounder:
    def test_no_reset_same_day(self, compounder, tracker):
        """No reset if still the same UTC day."""
        result = compounder.check_daily_reset(100.0)
        assert result is None

    def test_reset_new_day(self, compounder, tracker):
        """Reset occurs when UTC date changes."""
        # Force a different last reset date
        compounder._last_reset_date = "2020-01-01"
        summary = compounder.check_daily_reset(105.0)
        assert summary is not None
        assert summary["open_equity"] == 100.0
        assert summary["close_equity"] == 105.0

    def test_reset_updates_equity(self, compounder, tracker):
        """After reset, tracker has new equity base."""
        compounder._last_reset_date = "2020-01-01"
        compounder.check_daily_reset(105.0)
        s = tracker.state
        assert s.day_open_equity == 105.0
        assert s.current_equity == 105.0
        # New target: 105 * 2% = 2.10
        assert abs(s.daily_target_amount - 2.10) < 0.01

    def test_target_hit_increments_streak(self, compounder, tracker):
        """Hitting the target increments streak."""
        tracker.record_trade(2.00)  # 100% of target
        compounder._last_reset_date = "2020-01-01"
        summary = compounder.check_daily_reset(102.0)
        assert summary["target_hit"] is True
        assert tracker.state.streak_days == 1
        assert tracker.state.miss_streak == 0

    def test_target_miss_increments_miss_streak(self, compounder, tracker):
        """Missing the target increments miss_streak and resets streak."""
        tracker.record_trade(0.50)  # Only 25% of target
        compounder._last_reset_date = "2020-01-01"
        summary = compounder.check_daily_reset(100.50)
        assert summary["target_hit"] is False
        assert tracker.state.streak_days == 0
        assert tracker.state.miss_streak == 1

    def test_streak_accumulates_over_days(self, compounder, tracker):
        """Multiple winning days accumulate streak."""
        equity = 100.0
        for day_num in range(1, 4):
            # Record enough P&L to hit 100% of target at current equity
            target_amount = tracker.state.daily_target_amount
            tracker.record_trade(target_amount)
            compounder._last_reset_date = f"2020-01-{day_num:02d}"
            equity += target_amount
            compounder.check_daily_reset(equity)
        assert tracker.state.streak_days == 3

    def test_compound_projection(self, compounder):
        """Compound projection math is correct."""
        projection = compounder.get_compound_projection(5)
        assert len(projection) == 5
        # At 2% daily from $100:
        # Day 1: 102.00, Day 2: 104.04, Day 3: 106.12
        assert abs(projection[0]["projected_equity"] - 102.0) < 0.01
        assert abs(projection[1]["projected_equity"] - 104.04) < 0.01


# ─────────────────────────────────────────
# AUTO-TARGET REDUCTION
# ─────────────────────────────────────────

class TestAutoTargetReduction:
    def test_no_reduction_below_threshold(self, compounder, tracker):
        """No reduction if miss streak < 3."""
        tracker.state.miss_streak = 2
        result = compounder._check_target_adjustment(tracker.state)
        assert result is None

    def test_reduce_after_3_misses(self, compounder, tracker):
        """Target reduced by 20% after 3 consecutive misses."""
        tracker.state.miss_streak = 3
        tracker.state.daily_target_pct = 2.0
        new_target = compounder._check_target_adjustment(tracker.state)
        # 2.0 * 0.80 = 1.6
        assert new_target is not None
        assert abs(new_target - 1.6) < 0.01

    def test_severe_reduction_after_5_misses(self, compounder, tracker):
        """Target drops to 1.0% after 5 consecutive misses."""
        tracker.state.miss_streak = 5
        tracker.state.daily_target_pct = 2.0
        new_target = compounder._check_target_adjustment(tracker.state)
        assert new_target == 1.0

    def test_no_reduction_below_min(self, compounder, tracker):
        """Target can't go below 0.5%."""
        tracker.state.miss_streak = 3
        tracker.state.daily_target_pct = 0.6
        new_target = compounder._check_target_adjustment(tracker.state)
        # 0.6 * 0.80 = 0.48 → clamped to 0.5
        assert new_target == 0.5

    def test_restore_after_streak_at_reduced_target(self, compounder, tracker):
        """Restore original target after hitting reduced target 5 days straight."""
        tracker.state.target_was_reduced = True
        tracker.state.original_target_pct = 3.0
        tracker.state.daily_target_pct = 1.6
        tracker.state.streak_days = 5
        new_target = compounder._check_target_adjustment(tracker.state)
        assert new_target == 3.0  # Restored to original

    def test_no_restore_if_not_reduced(self, compounder, tracker):
        """Don't restore if target was never reduced."""
        tracker.state.target_was_reduced = False
        tracker.state.streak_days = 5
        result = compounder._check_target_adjustment(tracker.state)
        assert result is None

    def test_auto_reduction_disabled(self, config, tracker):
        """No auto-reduction when disabled in config."""
        config.daily_target.auto_target_reduction = False
        comp = Compounder(tracker, config)
        tracker.state.miss_streak = 5
        result = comp._check_target_adjustment(tracker.state)
        assert result is None


# ─────────────────────────────────────────
# DAILY RESET
# ─────────────────────────────────────────

class TestDailyReset:
    def test_reset_clears_counters(self, tracker):
        """reset_day clears all daily counters."""
        tracker.record_trade(0.50)
        tracker.record_trade(-0.20)
        tracker.update_unrealized(0.30)
        tracker.reset_day(100.60)
        s = tracker.state
        assert s.realized_pnl_today == 0.0
        assert s.unrealized_pnl == 0.0
        assert s.trades_today == 0
        assert s.wins_today == 0
        assert s.losses_today == 0
        assert s.mode == TradingMode.NORMAL

    def test_reset_with_new_target(self, tracker):
        """reset_day with overridden target %."""
        tracker.reset_day(110.0, new_target_pct=1.5)
        s = tracker.state
        assert s.daily_target_pct == 1.5
        assert abs(s.daily_target_amount - 1.65) < 0.01  # 110 * 1.5%
        assert s.target_was_reduced is True  # 1.5 < original 2.0

    def test_reset_preserves_streaks(self, tracker):
        """Streaks survive reset (set before reset_day)."""
        tracker.state.streak_days = 5
        tracker.state.miss_streak = 0
        tracker.reset_day(105.0)
        assert tracker.state.streak_days == 5

    def test_reset_computes_new_loss_limit(self, tracker):
        """New loss limit = new_target_amount * 50%."""
        tracker.reset_day(200.0)
        s = tracker.state
        # 200 * 2% = 4.0, loss limit = 4.0 * 50% = 2.0
        assert abs(s.daily_loss_limit - 2.0) < 0.01


# ─────────────────────────────────────────
# DAILY PROGRESS DICT
# ─────────────────────────────────────────

class TestDailyProgress:
    def test_to_dict_has_all_fields(self, tracker):
        """get_daily_progress returns all required fields."""
        progress = tracker.get_daily_progress()
        required = [
            "date", "day_open_equity", "current_equity",
            "daily_target_pct", "daily_target_amount", "target_equity",
            "realized_pnl_today", "unrealized_pnl", "total_pnl_today",
            "pct_achieved", "daily_loss_limit", "daily_loss_consumed_pct",
            "mode", "trades_today", "wins_today", "losses_today",
            "streak_days", "miss_streak", "original_target_pct",
            "target_was_reduced", "day_elapsed_pct",
        ]
        for field in required:
            assert field in progress, f"Missing field: {field}"

    def test_mode_is_string_in_dict(self, tracker):
        """Mode is serialized as string, not enum."""
        progress = tracker.get_daily_progress()
        assert progress["mode"] == "normal"
        tracker.state.mode = TradingMode.PROTECTING
        progress = tracker.get_daily_progress()
        assert progress["mode"] == "protecting"


# ─────────────────────────────────────────
# DAILY TARGET CONTEXT
# ─────────────────────────────────────────

class TestDailyTargetContext:
    def test_context_default_state(self, tracker):
        """Fresh tracker produces neutral context."""
        ctx = tracker.get_sizing_context()
        assert isinstance(ctx, DailyTargetContext)
        assert ctx.target_hit is False
        assert ctx.pct_achieved == 0.0
        assert ctx.behind_schedule is False
        assert ctx.mode == TradingMode.NORMAL
        assert ctx.intraday_dd_pct == 0.0

    def test_context_after_gains(self, tracker):
        """After trades, context reflects progress."""
        tracker.record_trade(1.0)  # 50% of $2 target
        ctx = tracker.get_sizing_context()
        assert ctx.pct_achieved == 50.0
        assert ctx.remaining_target_pct == 50.0
        assert ctx.target_hit is False

    def test_context_target_hit(self, tracker):
        """After hitting target, context shows target_hit."""
        tracker.record_trade(2.50)  # 125% of target
        ctx = tracker.get_sizing_context()
        assert ctx.target_hit is True
        assert ctx.pct_achieved == 125.0
        assert ctx.remaining_target_pct == 0.0

    def test_context_intraday_drawdown(self, tracker):
        """Equity drop from day-open shows as intraday DD."""
        tracker.state.current_equity = 96.0  # 4% DD from 100
        ctx = tracker.get_sizing_context()
        assert abs(ctx.intraday_dd_pct - 4.0) < 0.01

    def test_context_behind_schedule(self, tracker, config):
        """Behind schedule when low progress + late in day."""
        tracker.state.pct_achieved = 10.0  # < 20% trigger
        tracker.state.day_elapsed_pct = 0.7  # > 0.6 time trigger
        ctx = tracker.get_sizing_context()
        assert ctx.behind_schedule is True

    def test_context_not_behind_early(self, tracker, config):
        """Not behind schedule if early in day."""
        tracker.state.pct_achieved = 10.0
        tracker.state.day_elapsed_pct = 0.3  # < 0.6 time trigger
        ctx = tracker.get_sizing_context()
        assert ctx.behind_schedule is False

    def test_context_mode_propagates(self, tracker):
        """Context carries current trading mode."""
        tracker.state.mode = TradingMode.PROTECTING
        ctx = tracker.get_sizing_context()
        assert ctx.mode == TradingMode.PROTECTING


# ─────────────────────────────────────────
# FORCE TARGET REDUCTION
# ─────────────────────────────────────────

class TestForceTarget:
    def test_force_reduces_target(self, tracker):
        """Force target reduces target and recalculates."""
        assert tracker.state.daily_target_pct == 2.0
        tracker.force_target(1.0)
        assert tracker.state.daily_target_pct == 1.0
        assert tracker.state.daily_target_amount == 1.0  # 100 * 1%
        assert tracker.state.target_equity == 101.0

    def test_force_ignores_higher_target(self, tracker):
        """Force target only reduces, never increases."""
        tracker.force_target(5.0)  # Higher than current 2%
        assert tracker.state.daily_target_pct == 2.0  # Unchanged

    def test_force_updates_pct_achieved(self, tracker):
        """After force-reducing target, pct_achieved recalculates."""
        tracker.record_trade(1.0)  # 50% of $2 target
        tracker.force_target(1.0)  # Now $1 target
        assert tracker.state.pct_achieved == 100.0  # $1 / $1 = 100%
