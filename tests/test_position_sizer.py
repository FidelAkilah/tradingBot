"""
Tests for Kelly Criterion position sizing and drawdown scaling.

Validates:
1. Kelly fraction calculation from rolling trade history
2. Confidence-based position multiplier
3. Drawdown-based position scaling
4. Consecutive loss exponential reduction
5. Dynamic leverage mapping
6. Per-symbol halt after consecutive losses
7. Full calculate() integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from config import BotConfig
from position_sizer import PositionSizer, SizingResult, TradeOutcome


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.trading.kelly_lookback = 50
    cfg.trading.kelly_min_trades = 20
    cfg.trading.kelly_default_pct = 0.10
    cfg.trading.kelly_fraction = 0.5
    cfg.trading.min_position_pct = 0.05
    cfg.trading.max_position_pct = 0.20
    cfg.futures.max_leverage = 15
    cfg.risk.drawdown_scale_5 = 1.0
    cfg.risk.drawdown_scale_10 = 0.7
    cfg.risk.drawdown_scale_15 = 0.4
    cfg.risk.drawdown_scale_25 = 0.0
    cfg.risk.drawdown_halt = 25.0
    cfg.risk.consec_loss_base = 0.7
    cfg.risk.consec_loss_halt_count = 4
    cfg.risk.consec_loss_halt_s = 7200.0
    return cfg


@pytest.fixture
def sizer(config):
    return PositionSizer(config)


# ─────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────

class TestKellyFraction:
    def test_default_when_insufficient_trades(self, sizer):
        """Before kelly_min_trades, use conservative default."""
        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert not used_kelly
        assert kelly_pct == 0.10  # kelly_default_pct

    def test_kelly_with_60pct_win_rate(self, sizer, config):
        """60% WR with 1:1 W/L ratio → Kelly f = 0.2, half-Kelly = 0.10."""
        # Seed 20 trades: 12 wins (+$1), 8 losses (-$1)
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
        # f = 0.8 - (0.2 / 2.0) = 0.7, half-Kelly = 0.35 → capped at 0.20
        assert kelly_pct == 0.20

    def test_kelly_all_wins(self, sizer):
        """All wins → max position."""
        for _ in range(20):
            sizer._trade_history.append(TradeOutcome(pnl_usd=1.0, is_win=True))
        kelly_f, kelly_pct, used_kelly = sizer._kelly_fraction()
        assert used_kelly
        assert kelly_pct == 0.20  # max_position_pct

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
        # f = 0.4 - (0.6 / 1.0) = -0.2 → half = -0.1 → floored to 0.05
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
# DYNAMIC LEVERAGE
# ─────────────────────────────────────────

class TestDynamicLeverage:
    def test_low_confidence_leverage(self, sizer):
        assert sizer._dynamic_leverage(0.55) == 5
        assert sizer._dynamic_leverage(0.64) == 5

    def test_medium_confidence_leverage(self, sizer):
        assert sizer._dynamic_leverage(0.65) == 8
        assert sizer._dynamic_leverage(0.74) == 8

    def test_high_confidence_leverage(self, sizer):
        assert sizer._dynamic_leverage(0.75) == 10
        assert sizer._dynamic_leverage(0.84) == 10

    def test_very_high_confidence_leverage(self, sizer):
        assert sizer._dynamic_leverage(0.85) == 12

    def test_leverage_cap(self, sizer):
        """Never exceeds max_leverage."""
        sizer.fc.max_leverage = 8
        assert sizer._dynamic_leverage(0.95) == 8


# ─────────────────────────────────────────
# DRAWDOWN SCALING
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

        # After halt period, should be able to trade again
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
        assert result.leverage == 10
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
        """High confidence → 12x leverage and 1.2x size mult."""
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.90,
            symbol="BTC/USDT", current_time=time.time(),
        )
        assert result.leverage == 12
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
        # Very low confidence + drawdown should still be >= min
        result = sizer.calculate(
            equity=100.0, peak_equity=100.0, confidence=0.55,
            regime_mult=0.1, session_mult=0.1, current_time=time.time(),
        )
        assert result.position_usd >= 100.0 * 0.05  # min_position_pct
        assert result.position_usd <= 100.0 * 0.20  # max_position_pct

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

    def test_get_state(self, sizer):
        """get_state() should return dashboard-friendly dict."""
        state = sizer.get_state()
        assert "kelly_fraction" in state
        assert "rolling_trades" in state
        assert "consecutive_losses" in state
        assert "halted_pairs" in state
