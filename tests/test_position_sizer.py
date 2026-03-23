"""
Tests for daily-target-aware position sizing.

Validates:
1. Kelly fraction calculation from rolling trade history
2. Confidence-based position multiplier
3. Dynamic leverage with interpolation and target-hit reduction
4. Overall drawdown-based position scaling
5. Intraday drawdown scaling (tighter thresholds)
6. Consecutive loss exponential reduction
7. Target progress multiplier (behind-schedule, target-achieved, protecting)
8. Per-symbol halt after consecutive losses
9. Full calculate() integration with DailyTargetContext
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from config import BotConfig
from position_sizer import PositionSizer, SizingResult, TradeOutcome
from daily_target.tracker import DailyTargetContext, TradingMode


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.trading.kelly_lookback = 50
    cfg.trading.kelly_min_trades = 20
    cfg.trading.kelly_default_pct = 0.15
    cfg.trading.kelly_fraction = 0.5
    cfg.trading.min_position_pct = 0.05
    cfg.trading.max_position_pct = 0.25
    cfg.futures.max_leverage = 20
    cfg.risk.drawdown_scale_5 = 1.0
    cfg.risk.drawdown_scale_10 = 0.7
    cfg.risk.drawdown_scale_15 = 0.4
    cfg.risk.drawdown_scale_25 = 0.0
    cfg.risk.drawdown_halt = 25.0
    cfg.risk.intraday_dd_scale_3 = 1.0
    cfg.risk.intraday_dd_scale_5 = 0.7
    cfg.risk.intraday_dd_scale_7 = 0.4
    cfg.risk.intraday_dd_halt = 7.0
    cfg.risk.consec_loss_base = 0.7
    cfg.risk.consec_loss_halt_count = 4
    cfg.risk.consec_loss_halt_s = 7200.0
    cfg.daily_target.protecting_size_mult = 0.60
    return cfg


@pytest.fixture
def sizer(config):
    return PositionSizer(config)


def _ctx(**kwargs):
    """Helper to create DailyTargetContext with overrides."""
    defaults = dict(
        target_hit=False,
        pct_achieved=0.0,
        day_elapsed_pct=0.0,
        remaining_target_pct=100.0,
        daily_target_pct=2.0,
        mode=TradingMode.NORMAL,
        intraday_dd_pct=0.0,
        behind_schedule=False,
    )
    defaults.update(kwargs)
    return DailyTargetContext(**defaults)


# ─────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────

class TestKellyFraction:
    def test_default_when_insufficient_trades(self, sizer):
        """Before kelly_min_trades, use conservative default."""
        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert not used_kelly
        assert kelly_pct == 0.15  # kelly_default_pct (was 0.10, now 0.15)

    def test_kelly_with_60pct_win_rate(self, sizer, config):
        """60% WR with 1:1 W/L ratio → Kelly f = 0.2, half-Kelly = 0.10."""
        for _ in range(12):
            sizer._trade_history.append(TradeOutcome(pnl_usd=1.0, is_win=True))
        for _ in range(8):
            sizer._trade_history.append(TradeOutcome(pnl_usd=-1.0, is_win=False))

        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        # f = 0.6 - (0.4 / 1.0) = 0.2, half-Kelly = 0.1
        assert abs(kelly_f - 0.2) < 0.01
        assert abs(kelly_pct - 0.10) < 0.01

    def test_kelly_with_high_win_rate(self, sizer):
        """80% WR with 2:1 W/L → large Kelly → capped at max_position_pct."""
        for _ in range(16):
            sizer._trade_history.append(TradeOutcome(pnl_usd=2.0, is_win=True))
        for _ in range(4):
            sizer._trade_history.append(TradeOutcome(pnl_usd=-1.0, is_win=False))

        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        # f = 0.8 - (0.2 / 2.0) = 0.7, half-Kelly = 0.35 → capped at 0.25
        assert kelly_pct == 0.25

    def test_kelly_all_wins(self, sizer):
        """All wins → max position."""
        for _ in range(20):
            sizer._trade_history.append(TradeOutcome(pnl_usd=1.0, is_win=True))
        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        assert kelly_pct == 0.25  # max_position_pct

    def test_kelly_all_losses(self, sizer):
        """All losses → min position."""
        for _ in range(20):
            sizer._trade_history.append(TradeOutcome(pnl_usd=-1.0, is_win=False))
        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        assert kelly_pct == 0.05  # min_position_pct

    def test_kelly_negative_edge(self, sizer):
        """40% WR with 1:1 → negative Kelly → floored to min."""
        for _ in range(8):
            sizer._trade_history.append(TradeOutcome(pnl_usd=1.0, is_win=True))
        for _ in range(12):
            sizer._trade_history.append(TradeOutcome(pnl_usd=-1.0, is_win=False))

        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        assert kelly_f < 0
        assert kelly_pct == 0.05


# ─────────────────────────────────────────
# CONFIDENCE MULTIPLIER
# ─────────────────────────────────────────

class TestConfidenceMultiplier:
    def test_low_confidence(self):
        assert PositionSizer._confidence_multiplier(0.55) == 0.6
        assert PositionSizer._confidence_multiplier(0.64) == 0.6

    def test_medium_confidence(self):
        assert PositionSizer._confidence_multiplier(0.65) == 0.8
        assert PositionSizer._confidence_multiplier(0.74) == 0.8

    def test_high_confidence(self):
        assert PositionSizer._confidence_multiplier(0.75) == 1.0
        assert PositionSizer._confidence_multiplier(0.84) == 1.0

    def test_very_high_confidence(self):
        assert PositionSizer._confidence_multiplier(0.85) == 1.2
        assert PositionSizer._confidence_multiplier(0.95) == 1.2


# ─────────────────────────────────────────
# DYNAMIC LEVERAGE (target-aware, interpolated)
# ─────────────────────────────────────────

class TestDynamicLeverage:
    def test_low_confidence_leverage(self, sizer):
        """0.55 → 5x (bottom of 5-8 range)."""
        assert sizer._dynamic_leverage(0.55) == 5

    def test_low_confidence_top(self, sizer):
        """0.64 → 7x (near top of 5-8 range, interpolated)."""
        lev = sizer._dynamic_leverage(0.64)
        assert 7 <= lev <= 8

    def test_medium_confidence_leverage(self, sizer):
        """0.65 → 8x (bottom of 8-12 range)."""
        assert sizer._dynamic_leverage(0.65) == 8

    def test_medium_confidence_mid(self, sizer):
        """0.70 → 10x (midpoint of 8-12 range)."""
        assert sizer._dynamic_leverage(0.70) == 10

    def test_high_confidence_leverage(self, sizer):
        """0.75 → 12x (bottom of 12-16 range)."""
        assert sizer._dynamic_leverage(0.75) == 12

    def test_high_confidence_mid(self, sizer):
        """0.80 → 14x (midpoint of 12-16 range)."""
        assert sizer._dynamic_leverage(0.80) == 14

    def test_very_high_confidence_leverage(self, sizer):
        """0.85 → 16x (bottom of 16-20 range)."""
        assert sizer._dynamic_leverage(0.85) == 16

    def test_max_confidence_leverage(self, sizer):
        """1.0 → 20x (top of 16-20 range)."""
        assert sizer._dynamic_leverage(1.0) == 20

    def test_leverage_cap(self, sizer):
        """Never exceeds max_leverage."""
        sizer.fc.max_leverage = 10
        assert sizer._dynamic_leverage(0.95) == 10

    def test_target_hit_reduces_40pct(self, sizer):
        """After target hit, leverage reduces by 40%."""
        # 0.85 → 16x normal, 16 * 0.6 = 9.6 → round to 10
        assert sizer._dynamic_leverage(0.85, target_hit=True) == 10
        # 0.75 → 12x normal, 12 * 0.6 = 7.2 → round to 7
        assert sizer._dynamic_leverage(0.75, target_hit=True) == 7
        # 0.55 → 5x normal, 5 * 0.6 = 3.0 → 3
        assert sizer._dynamic_leverage(0.55, target_hit=True) == 3

    def test_target_hit_high_confidence(self, sizer):
        """1.0 → 20x normal, 20 * 0.6 = 12x after target hit."""
        assert sizer._dynamic_leverage(1.0, target_hit=True) == 12


# ─────────────────────────────────────────
# OVERALL DRAWDOWN SCALING
# ─────────────────────────────────────────

class TestDrawdownScaling:
    def test_no_drawdown(self, sizer):
        assert sizer._drawdown_multiplier(0.0) == 1.0
        assert sizer._drawdown_multiplier(4.9) == 1.0

    def test_moderate_drawdown(self, sizer):
        assert sizer._drawdown_multiplier(5.0) == 0.7
        assert sizer._drawdown_multiplier(9.9) == 0.7

    def test_heavy_drawdown(self, sizer):
        assert sizer._drawdown_multiplier(10.0) == 0.4
        assert sizer._drawdown_multiplier(14.9) == 0.4

    def test_severe_drawdown(self, sizer):
        """15-25% → floor to 0.0 (will use min_position_pct)."""
        assert sizer._drawdown_multiplier(15.0) == 0.0
        assert sizer._drawdown_multiplier(24.9) == 0.0

    def test_drawdown_pct_calculation(self):
        assert PositionSizer._drawdown_pct(95.0, 100.0) == 5.0
        assert PositionSizer._drawdown_pct(100.0, 100.0) == 0.0
        assert abs(PositionSizer._drawdown_pct(80.0, 100.0) - 20.0) < 0.01


# ─────────────────────────────────────────
# INTRADAY DRAWDOWN SCALING
# ─────────────────────────────────────────

class TestIntradayDrawdownScaling:
    def test_no_intraday_drawdown(self, sizer):
        assert sizer._intraday_drawdown_multiplier(0.0) == 1.0
        assert sizer._intraday_drawdown_multiplier(2.9) == 1.0

    def test_mild_intraday_drawdown(self, sizer):
        """3-5% → reduce 30%."""
        assert sizer._intraday_drawdown_multiplier(3.0) == 0.7
        assert sizer._intraday_drawdown_multiplier(4.9) == 0.7

    def test_moderate_intraday_drawdown(self, sizer):
        """5-7% → reduce 60%."""
        assert sizer._intraday_drawdown_multiplier(5.0) == 0.4
        assert sizer._intraday_drawdown_multiplier(6.9) == 0.4

    def test_severe_intraday_drawdown(self, sizer):
        """>=7% → halt (returns 0.0, but halt check catches first)."""
        assert sizer._intraday_drawdown_multiplier(7.0) == 0.0
        assert sizer._intraday_drawdown_multiplier(10.0) == 0.0

    def test_intraday_halt_in_calculate(self, sizer):
        """Calculate should halt when intraday DD >= 7%."""
        ctx = _ctx(intraday_dd_pct=8.0)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.is_halted
        assert "Intraday" in result.halt_reason

    def test_no_intraday_halt_without_context(self, sizer):
        """Without DailyTargetContext, intraday DD is not checked."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            current_time=time.time(),
        )
        assert not result.is_halted
        assert result.intraday_dd_pct == 0.0
        assert result.intraday_dd_mult == 1.0


# ─────────────────────────────────────────
# TARGET PROGRESS MULTIPLIER
# ─────────────────────────────────────────

class TestTargetProgressMultiplier:
    def test_normal_mode(self, sizer):
        """Normal mode, no special conditions → 1.0."""
        ctx = _ctx(mode=TradingMode.NORMAL)
        assert sizer._target_progress_multiplier(ctx, 0.75) == 1.0

    def test_halted_mode(self, sizer):
        """HALTED → 0.0."""
        ctx = _ctx(mode=TradingMode.HALTED)
        assert sizer._target_progress_multiplier(ctx, 0.75) == 0.0

    def test_target_hit(self, sizer):
        """Target achieved → 0.5x."""
        ctx = _ctx(target_hit=True, pct_achieved=120.0)
        assert sizer._target_progress_multiplier(ctx, 0.75) == 0.5

    def test_protecting_mode(self, sizer):
        """PROTECTING mode → 0.6x (protecting_size_mult)."""
        ctx = _ctx(mode=TradingMode.PROTECTING, pct_achieved=85.0)
        assert sizer._target_progress_multiplier(ctx, 0.75) == 0.60

    def test_behind_schedule_high_confidence(self, sizer):
        """Behind schedule + confidence >= 0.70 → 1.3x boost."""
        ctx = _ctx(behind_schedule=True, pct_achieved=10.0, day_elapsed_pct=0.7)
        assert sizer._target_progress_multiplier(ctx, 0.75) == 1.3

    def test_behind_schedule_low_confidence(self, sizer):
        """Behind schedule + confidence < 0.70 → normal 1.0 (no boost)."""
        ctx = _ctx(behind_schedule=True, pct_achieved=10.0, day_elapsed_pct=0.7)
        assert sizer._target_progress_multiplier(ctx, 0.60) == 1.0

    def test_none_context(self, sizer):
        """No context → 1.0."""
        assert sizer._target_progress_multiplier(None, 0.75) == 1.0

    def test_target_hit_overrides_protecting(self, sizer):
        """Target hit takes priority over PROTECTING mode."""
        ctx = _ctx(
            target_hit=True, pct_achieved=110.0,
            mode=TradingMode.PROTECTING,
        )
        assert sizer._target_progress_multiplier(ctx, 0.75) == 0.5


# ─────────────────────────────────────────
# CONSECUTIVE LOSS SCALING
# ─────────────────────────────────────────

class TestConsecutiveLossScaling:
    def test_no_losses(self, sizer):
        assert sizer._consecutive_loss_multiplier(0) == 1.0

    def test_one_loss(self, sizer):
        assert abs(sizer._consecutive_loss_multiplier(1) - 0.7) < 0.01

    def test_two_losses(self, sizer):
        assert abs(sizer._consecutive_loss_multiplier(2) - 0.49) < 0.01

    def test_three_losses(self, sizer):
        assert abs(sizer._consecutive_loss_multiplier(3) - 0.343) < 0.01


# ─────────────────────────────────────────
# RECORD OUTCOME + PER-SYMBOL HALT
# ─────────────────────────────────────────

class TestRecordOutcome:
    def test_win_resets_consecutive_losses(self, sizer):
        sizer.record_outcome("BTC/USDT", -1.0, time.time())
        sizer.record_outcome("BTC/USDT", -1.0, time.time())
        assert sizer.get_consecutive_losses("BTC/USDT") == 2
        sizer.record_outcome("BTC/USDT", 1.0, time.time())
        assert sizer.get_consecutive_losses("BTC/USDT") == 0

    def test_symbol_halt_after_consecutive_losses(self, sizer):
        """4 consecutive losses should halt the symbol for 2 hours."""
        now = time.time()
        for i in range(4):
            sizer.record_outcome("ETH/USDT", -1.0, now + i)

        assert "ETH/USDT" in sizer._symbol_halt_until
        assert sizer._symbol_halt_until["ETH/USDT"] > now

    def test_halted_symbol_blocks_calculate(self, sizer):
        now = time.time()
        for i in range(4):
            sizer.record_outcome("ETH/USDT", -1.0, now + i)

        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            symbol="ETH/USDT", current_time=now + 5,
        )
        assert result.is_halted
        assert "ETH/USDT" in result.halt_reason

    def test_halt_expires(self, sizer):
        now = time.time()
        for i in range(4):
            sizer.record_outcome("ETH/USDT", -1.0, now + i)

        # Halt is set at (now + 3) + 7200 = now + 7203, so check past that
        future = now + 7204
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            symbol="ETH/USDT", current_time=future,
        )
        assert not result.is_halted

    def test_other_symbols_unaffected(self, sizer):
        """Halting ETH should not affect BTC."""
        now = time.time()
        for i in range(4):
            sizer.record_outcome("ETH/USDT", -1.0, now + i)

        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            symbol="BTC/USDT", current_time=now + 5,
        )
        assert not result.is_halted


# ─────────────────────────────────────────
# FULL CALCULATE() INTEGRATION
# ─────────────────────────────────────────

class TestCalculateIntegration:
    def test_basic_calculation(self, sizer):
        """Basic sizing with no history, normal confidence."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert not result.is_halted
        assert result.position_usd > 0
        assert result.leverage == 12  # 0.75 → bottom of 12-16 range
        assert result.notional_usd == result.position_usd * result.leverage
        assert result.confidence_mult == 1.0
        assert result.drawdown_mult == 1.0

    def test_drawdown_reduces_size(self, sizer):
        """7% drawdown should reduce position by 30%."""
        normal = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            symbol="BTC/USDT", current_time=time.time(),
        )
        drawdown = sizer.calculate(
            equity=93.0, peak_equity=100.0, confidence=0.75,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert drawdown.position_usd < normal.position_usd
        assert drawdown.drawdown_mult == 0.7

    def test_drawdown_halt(self, sizer):
        """26% drawdown should halt."""
        result = sizer.calculate(
            equity=74.0, peak_equity=100.0, confidence=0.75,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert result.is_halted
        assert "Drawdown" in result.halt_reason

    def test_low_confidence_reduces_leverage_and_size(self, sizer):
        """Low confidence → 5x leverage and 0.6x size mult."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.55,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert result.leverage == 5
        assert result.confidence_mult == 0.6

    def test_high_confidence_increases_leverage_and_size(self, sizer):
        """High confidence → 16x leverage (bottom of 16-20 range)."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.85,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert result.leverage == 16
        assert result.confidence_mult == 1.2

    def test_regime_and_session_multipliers(self, sizer):
        """Regime and session multipliers reduce position size."""
        full = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            regime_mult=1.0, session_mult=1.0, current_time=time.time(),
        )
        reduced = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            regime_mult=0.5, session_mult=0.8, current_time=time.time(),
        )
        assert reduced.position_usd < full.position_usd

    def test_position_pct_clamped(self, sizer):
        """Position % should always be between min and max."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.55,
            regime_mult=0.1, session_mult=0.1, current_time=time.time(),
        )
        assert result.position_usd >= 100.0 * 0.05  # min_position_pct
        assert result.position_usd <= 100.0 * 0.25  # max_position_pct

    def test_sizing_result_to_dict(self, sizer):
        """SizingResult.to_dict() should return all fields."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            current_time=time.time(),
        )
        d = result.to_dict()
        assert "position_usd" in d
        assert "leverage" in d
        assert "kelly_pct" in d
        assert "confidence_mult" in d
        assert "drawdown_mult" in d
        assert "consec_loss_mult" in d
        assert "intraday_dd_mult" in d
        assert "intraday_dd_pct" in d
        assert "target_progress_mult" in d

    def test_get_state(self, sizer):
        """get_state() should return dashboard-friendly dict."""
        state = sizer.get_state()
        assert "kelly_fraction" in state
        assert "rolling_trades" in state
        assert "consecutive_losses" in state
        assert "halted_pairs" in state


# ─────────────────────────────────────────
# DAILY TARGET CONTEXT INTEGRATION
# ─────────────────────────────────────────

class TestDailyTargetContextIntegration:
    def test_target_hit_reduces_leverage(self, sizer):
        """When target is hit, leverage should be 40% lower."""
        ctx_normal = _ctx(target_hit=False)
        ctx_hit = _ctx(target_hit=True, pct_achieved=110.0)

        r_normal = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.85,
            daily_target_ctx=ctx_normal, current_time=time.time(),
        )
        r_hit = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.85,
            daily_target_ctx=ctx_hit, current_time=time.time(),
        )
        assert r_normal.leverage == 16
        assert r_hit.leverage == 10  # 16 * 0.6 = 9.6 → round to 10
        assert r_hit.target_progress_mult == 0.5

    def test_target_hit_reduces_size(self, sizer):
        """Target achieved → 0.5x position size."""
        ctx = _ctx(target_hit=True, pct_achieved=120.0)
        r_normal = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            current_time=time.time(),
        )
        r_hit = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert r_hit.position_usd < r_normal.position_usd
        assert r_hit.target_progress_mult == 0.5

    def test_behind_schedule_boost(self, sizer):
        """Behind schedule + high confidence → 1.3x."""
        ctx = _ctx(behind_schedule=True, pct_achieved=10.0, day_elapsed_pct=0.7)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.target_progress_mult == 1.3

    def test_behind_schedule_no_boost_low_conf(self, sizer):
        """Behind schedule + low confidence → no boost."""
        ctx = _ctx(behind_schedule=True, pct_achieved=10.0, day_elapsed_pct=0.7)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.60,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.target_progress_mult == 1.0

    def test_protecting_mode_size(self, sizer):
        """PROTECTING mode → 0.6x size."""
        ctx = _ctx(mode=TradingMode.PROTECTING, pct_achieved=85.0)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.target_progress_mult == 0.60

    def test_halted_mode_blocks(self, sizer):
        """HALTED mode → target_progress_mult = 0.0, floored to min size."""
        ctx = _ctx(mode=TradingMode.HALTED)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        # With target_progress_mult=0.0, position_pct is floored to min
        assert result.target_progress_mult == 0.0
        assert result.position_usd == 100.0 * 0.05  # min_position_pct

    def test_intraday_dd_reduces_size(self, sizer):
        """4% intraday DD → 30% reduction."""
        ctx_normal = _ctx(intraday_dd_pct=0.0)
        ctx_dd = _ctx(intraday_dd_pct=4.0)

        r_normal = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx_normal, current_time=time.time(),
        )
        r_dd = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx_dd, current_time=time.time(),
        )
        assert r_dd.intraday_dd_mult == 0.7
        assert r_dd.position_usd < r_normal.position_usd

    def test_intraday_dd_halts(self, sizer):
        """8% intraday DD → halt."""
        ctx = _ctx(intraday_dd_pct=8.0)
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.is_halted
        assert "Intraday" in result.halt_reason

    def test_combined_drawdowns(self, sizer):
        """Both overall and intraday drawdown multiply together."""
        ctx = _ctx(intraday_dd_pct=4.0)
        result = sizer.calculate(
            equity=93.0, peak_equity=100.0, confidence=0.75,
            daily_target_ctx=ctx, current_time=time.time(),
        )
        assert result.drawdown_mult == 0.7       # 7% overall
        assert result.intraday_dd_mult == 0.7     # 4% intraday
        # Combined: 0.7 * 0.7 = 0.49 applied to position

    def test_no_context_backward_compat(self, sizer):
        """Without context, all target-aware features default to neutral."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.75,
            current_time=time.time(),
        )
        assert result.target_progress_mult == 1.0
        assert result.intraday_dd_mult == 1.0
        assert result.intraday_dd_pct == 0.0
        assert not result.is_halted
