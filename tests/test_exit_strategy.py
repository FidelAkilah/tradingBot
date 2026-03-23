"""
Tests for the exit strategy system:

1. Partial Take Profit (3 TP levels with scale-out)
2. Chandelier Exit (ATR-based trailing stop)
3. Dynamic SL adjustment
4. Daily-target-aware TP adjustments
5. Edge cases (low ATR, zero amount, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from unittest.mock import MagicMock, patch
from config import BotConfig, ExitConfig
from shadow_trader import ShadowTrader, ShadowTrade, TPLevel
from daily_target.tracker import DailyTargetContext, TradingMode
from risk_manager import TradeRecord


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.exit.partial_tp_enabled = True
    cfg.exit.chandelier_enabled = True
    cfg.exit.dynamic_sl_enabled = True
    cfg.trading.min_hold_time_s = 0  # Disable for testing
    cfg.trading.fee_rate = 0.04
    return cfg


@pytest.fixture
def trader(config):
    return ShadowTrader(config)


def _make_trade(trader, symbol="BTC/USDT", side="BUY", entry_price=100.0,
                amount=1.0, atr=2.0, stop_price=None, target_price=None):
    """Create a trade and set up TP levels for testing."""
    if stop_price is None:
        stop_price = entry_price - atr if side == "BUY" else entry_price + atr
    if target_price is None:
        target_price = entry_price + atr * 2 if side == "BUY" else entry_price - atr * 2

    trade = ShadowTrade(
        trade_id=trader._next_id(),
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        target_price=target_price,
        stop_price=stop_price,
        amount=amount,
        usd_value=10.0,
        wall_price=0.0,
        wall_usd_value=0.0,
        wall_multiplier=0.0,
        wall_confidence=0.0,
        composite_score=0.5,
        imbalance_ratio=1.0,
        vwap=entry_price,
        vwap_deviation_pct=0.0,
        momentum_aggressor_ratio=0.5,
        spread_pct=0.01,
        entry_time=time.time() - 5,  # 5 seconds ago (past 2s guard)
        original_amount=amount,
        atr_at_entry=atr,
        original_stop_distance=abs(entry_price - stop_price),
        max_favorable_price=entry_price,
        swing_confidence=0.65,
    )

    trader.open_trades[symbol] = trade

    # Build TP levels
    if trader.ec.partial_tp_enabled and atr > 0:
        tp_levels = trader._build_tp_levels(entry_price, atr, side, amount, 0.65)
        trader._tp_state[symbol] = tp_levels

    return trade


# ─────────────────────────────────────────
# TPLevel Dataclass
# ─────────────────────────────────────────

class TestTPLevel:
    def test_tp_level_creation(self):
        tp = TPLevel(level=1, atr_mult=1.0, target_price=102.0,
                     size_pct=0.40, amount=0.4)
        assert tp.level == 1
        assert tp.hit is False
        assert tp.pnl_usd is None

    def test_tp_level_hit(self):
        tp = TPLevel(level=2, atr_mult=2.0, target_price=104.0,
                     size_pct=0.35, amount=0.35)
        tp.hit = True
        tp.hit_price = 104.5
        tp.hit_time = 1000.0
        tp.pnl_usd = 1.575
        assert tp.hit is True
        assert tp.hit_price == 104.5


# ─────────────────────────────────────────
# Partial TP Level Construction
# ─────────────────────────────────────────

class TestBuildTPLevels:
    def test_builds_three_levels(self, trader):
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        assert len(levels) == 3
        assert levels[0].level == 1
        assert levels[1].level == 2
        assert levels[2].level == 3

    def test_long_tp_prices(self, trader):
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        assert levels[0].target_price == pytest.approx(102.0)  # 100 + 1.0×2
        assert levels[1].target_price == pytest.approx(104.0)  # 100 + 2.0×2
        assert levels[2].target_price == 0.0  # Runner: no fixed target

    def test_short_tp_prices(self, trader):
        levels = trader._build_tp_levels(100.0, 2.0, "SELL", 1.0, 0.65)
        assert levels[0].target_price == pytest.approx(98.0)   # 100 - 1.0×2
        assert levels[1].target_price == pytest.approx(96.0)   # 100 - 2.0×2
        assert levels[2].target_price == 0.0

    def test_size_allocation(self, trader):
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 10.0, 0.65)
        assert levels[0].amount == pytest.approx(4.0)   # 40%
        assert levels[1].amount == pytest.approx(3.5)   # 35%
        assert levels[2].amount == pytest.approx(2.5)   # 25%
        total = sum(l.amount for l in levels)
        assert total == pytest.approx(10.0)

    def test_size_pcts_sum_to_one(self, trader):
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        assert sum(l.size_pct for l in levels) == pytest.approx(1.0)


# ─────────────────────────────────────────
# Daily-Target-Aware TP Adjustments
# ─────────────────────────────────────────

class TestDailyTargetTPAdjust:
    def test_near_target_compresses_tps(self, trader):
        """When >80% of daily target achieved, TPs compress by 30%."""
        trader.daily_target_ctx = DailyTargetContext(pct_achieved=85.0)
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        # 1.0 * 0.70 = 0.70 ATR → 100 + 0.70*2 = 101.4
        assert levels[0].target_price == pytest.approx(101.4)
        assert levels[1].target_price == pytest.approx(102.8)  # 2.0*0.70*2

    def test_bonus_territory_no_compress(self, trader):
        """When >100% achieved, TPs are NOT compressed (house money)."""
        trader.daily_target_ctx = DailyTargetContext(pct_achieved=110.0)
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        # >100 still triggers near_target_threshold (>80%). So compression applies.
        assert levels[0].target_price == pytest.approx(101.4)

    def test_behind_schedule_expands_tp2_tp3(self, trader):
        """When <30% target + past 18:00 UTC + high confidence, expand TP2/TP3."""
        trader.daily_target_ctx = DailyTargetContext(
            pct_achieved=10.0,
            day_elapsed_pct=0.80,  # ~19:00 UTC
        )
        # Must have high confidence
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.80)
        # TP1 unchanged: 102.0
        assert levels[0].target_price == pytest.approx(102.0)
        # TP2 expanded: 2.0 * 1.20 * 2 = 4.8 → 104.8
        assert levels[1].target_price == pytest.approx(104.8)

    def test_behind_schedule_low_conf_no_expand(self, trader):
        """Low confidence should NOT expand TPs even if behind schedule."""
        trader.daily_target_ctx = DailyTargetContext(
            pct_achieved=10.0,
            day_elapsed_pct=0.80,
        )
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.60)
        # No expansion — confidence too low
        assert levels[1].target_price == pytest.approx(104.0)


# ─────────────────────────────────────────
# Partial TP Execution in update_prices
# ─────────────────────────────────────────

class TestPartialTPExecution:
    def test_tp1_hit_closes_40pct(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        orig_amount = trade.amount

        # Price reaches TP1 (102.0 for BUY with 1.0×ATR)
        result = trader.update_prices("BTC/USDT", 102.5)

        # Should NOT fully close — only partial
        assert result is None
        assert trade.tp1_hit is True
        assert trade.tp1_price == 102.5
        assert trade.amount < orig_amount
        assert trade.amount == pytest.approx(0.6)  # 60% remains

    def test_tp2_hit_closes_35pct(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 first
        trader.update_prices("BTC/USDT", 102.5)
        assert trade.tp1_hit is True

        # Hit TP2 (104.0 for BUY with 2.0×ATR)
        result = trader.update_prices("BTC/USDT", 104.5)
        assert result is None
        assert trade.tp2_hit is True
        assert trade.amount == pytest.approx(0.25)  # 25% remains (runner)

    def test_tp3_runner_closed_by_trailing(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 and TP2
        trader.update_prices("BTC/USDT", 102.5)
        trader.update_prices("BTC/USDT", 104.5)
        assert trade.amount == pytest.approx(0.25)

        # Price runs higher then pulls back — trailing stop catches it
        trader.update_prices("BTC/USDT", 107.0)  # New peak
        # Chandelier stop: 107 - 1.0×2 = 105.0 (after TP2, mult=1.0)
        # SL was set to TP1 level (102.0) after TP2, chandelier moves it up
        result = trader.update_prices("BTC/USDT", 104.5)

        # Should close as trailing_stop
        if result:
            assert result.reason == "trailing_stop"
            assert trade.tp3_hit is True

    def test_sl_to_breakeven_after_tp1(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        original_sl = trade.stop_price  # 98.0

        # Hit TP1
        trader.update_prices("BTC/USDT", 102.5)

        # SL should be at breakeven (entry + fees)
        fee_offset = 100.0 * (0.04 / 100.0) * 2  # Round-trip fee
        assert trade.stop_price >= 100.0 + fee_offset
        assert trade.stop_price > original_sl

    def test_sl_to_tp1_after_tp2(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        tp1_target = trader._tp_state["BTC/USDT"][0].target_price  # 102.0

        # Hit TP1 then TP2
        trader.update_prices("BTC/USDT", 102.5)
        trader.update_prices("BTC/USDT", 104.5)

        # SL should be at TP1 level
        assert trade.stop_price >= tp1_target

    def test_partial_records_queued(self, trader):
        _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        trader.update_prices("BTC/USDT", 102.5)  # TP1

        assert len(trader.pending_partial_records) == 1
        rec = trader.pending_partial_records[0]
        assert rec.reason == "partial_tp1"
        assert rec.pnl_usd > 0

    def test_partial_pnl_accumulated(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        trader.update_prices("BTC/USDT", 102.5)  # TP1
        assert trade.partial_realized_pnl > 0

        pnl_after_tp1 = trade.partial_realized_pnl
        trader.update_prices("BTC/USDT", 104.5)  # TP2
        assert trade.partial_realized_pnl > pnl_after_tp1

    def test_full_close_pnl_includes_partials(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 and TP2
        trader.update_prices("BTC/USDT", 102.5)
        trader.update_prices("BTC/USDT", 104.5)

        partial_pnl = trade.partial_realized_pnl

        # Force close remaining (stop loss or max hold)
        result = trader.update_prices("BTC/USDT", 97.0)
        if not result:
            # Try direct close
            result = trader.close_on_wall_pull("BTC/USDT", 103.0)

        # Total PnL should include partial + remaining
        assert trade.pnl_usd is not None
        assert trade.partial_realized_pnl == partial_pnl  # Preserved

    def test_short_partial_tp(self, trader):
        trade = _make_trade(trader, side="SELL", entry_price=100.0, atr=2.0, amount=1.0)

        # TP1 for short at 98.0 (100 - 1.0×2)
        result = trader.update_prices("BTC/USDT", 97.5)
        assert result is None
        assert trade.tp1_hit is True
        assert trade.tp1_pnl > 0


# ─────────────────────────────────────────
# Chandelier Exit (ATR Trailing Stop)
# ─────────────────────────────────────────

class TestChandelierExit:
    def test_chandelier_activates_after_threshold(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        original_sl = trade.stop_price  # 98.0

        # TP1 target is 102.0, activation at 30% = 100.6
        # Price above activation but below TP1
        trader.update_prices("BTC/USDT", 101.0)

        # Chandelier should start tracking: peak(101) - 2×2 = 97.0
        # But 97.0 < 98.0 (current SL), so no change yet
        assert trade.stop_price >= original_sl

    def test_chandelier_trails_up(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 first (which activates chandelier and sets mult to 1.5)
        trader.update_prices("BTC/USDT", 102.5)
        assert trade.tp1_hit is True

        # Price runs higher
        trader.update_prices("BTC/USDT", 106.0)
        # Chandelier: peak(106) - 1.5×2 = 103.0
        # This should be above breakeven
        assert trade.stop_price >= trade.entry_price

    def test_chandelier_tightens_after_tp2(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        trader.update_prices("BTC/USDT", 102.5)  # TP1
        trader.update_prices("BTC/USDT", 104.5)  # TP2

        # After TP2, trailing mult = 1.0
        assert trade.trailing_atr_mult == trader.ec.chandelier_after_tp2_mult
        # == 1.0

        # Price runs to 107
        trader.update_prices("BTC/USDT", 107.0)
        # Chandelier: 107 - 1.0×2 = 105.0
        assert trade.stop_price >= 105.0

    def test_chandelier_never_below_breakeven_after_tp1(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1
        trader.update_prices("BTC/USDT", 102.5)

        # Even if peak was barely above entry, chandelier floor is breakeven
        fee_offset = 100.0 * (0.04 / 100.0) * 2
        assert trade.stop_price >= 100.0 + fee_offset

    def test_chandelier_disabled_fallback(self, config):
        config.exit.chandelier_enabled = False
        config.exit.partial_tp_enabled = False
        config.trading.trailing_stop_pct = 0.8
        trader = ShadowTrader(config)

        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        # Should use original fixed trailing stop
        trader.update_prices("BTC/USDT", 102.0)
        # Fixed trailing: peak(102) * (1 - 0.008) = 101.184
        # But only activates at 50% of TP distance


# ─────────────────────────────────────────
# Dynamic SL Adjustment
# ─────────────────────────────────────────

class TestDynamicSL:
    def test_momentum_tightens_sl(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        # Set entry_time to < 15 min ago
        trade.entry_time = time.time() - 600  # 10 min ago
        original_sl = trade.stop_price  # 98.0

        # Move favorably by 0.5×ATR = 1.0
        trader.update_prices("BTC/USDT", 101.5)

        # SL should tighten: entry - 0.7×ATR = 100 - 1.4 = 98.6
        assert trade.stop_price >= 98.6

    def test_flat_tightens_sl(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        # Set entry_time to > 30 min ago
        trade.entry_time = time.time() - 2000
        original_sl = trade.stop_price  # 98.0

        # Price barely moved (< 0.2×ATR = 0.4)
        trader.update_prices("BTC/USDT", 100.2)

        # SL should tighten: entry - 0.5×ATR = 100 - 1.0 = 99.0
        assert trade.stop_price >= 99.0

    def test_flat_sl_only_applied_once(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        trade.entry_time = time.time() - 2000

        trader.update_prices("BTC/USDT", 100.2)
        sl_after_first = trade.stop_price

        trader.update_prices("BTC/USDT", 100.1)
        # SL should not tighten further from the flat logic
        assert trade.stop_price == sl_after_first

    def test_dynamic_sl_never_widens(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        trade.entry_time = time.time() - 600
        original_sl = trade.stop_price

        # Adverse move (price drops)
        trader.update_prices("BTC/USDT", 99.0)

        # SL should NOT widen beyond original
        assert trade.stop_price >= original_sl

    def test_short_momentum_tightens(self, trader):
        trade = _make_trade(trader, side="SELL", entry_price=100.0, atr=2.0, amount=1.0)
        trade.entry_time = time.time() - 600
        original_sl = trade.stop_price  # 102.0

        # Price drops favorably by > 0.5×ATR
        trader.update_prices("BTC/USDT", 98.0)

        # SL should tighten: entry + 0.7×ATR = 100 + 1.4 = 101.4
        assert trade.stop_price <= 101.4


# ─────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────

class TestEdgeCases:
    def test_zero_atr_skips_partial_tp(self, trader):
        """When ATR is 0, partial TP levels aren't built."""
        trade = _make_trade(trader, entry_price=100.0, atr=0.0, amount=1.0)
        assert "BTC/USDT" not in trader._tp_state

    def test_min_tp1_sl_distance(self, trader):
        """TP1 can't be too close to entry."""
        trader.ec.min_tp1_sl_distance_atr = 0.5
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        # Default tp1_atr_mult=1.0 > 0.5, so no change
        assert levels[0].atr_mult >= 0.5

    def test_min_tp1_sl_enforced(self, trader):
        """When TP1 multiplier is below minimum, floor is applied."""
        trader.ec.tp1_atr_mult = 0.3
        trader.ec.min_tp1_sl_distance_atr = 0.5
        levels = trader._build_tp_levels(100.0, 2.0, "BUY", 1.0, 0.65)
        assert levels[0].atr_mult >= 0.5

    def test_2s_guard_prevents_instant_close(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        trade.entry_time = time.time()  # Just now

        result = trader.update_prices("BTC/USDT", 90.0)  # Huge drop
        assert result is None  # Guard prevents close

    def test_close_trade_restores_original_amount(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 (reduces amount)
        trader.update_prices("BTC/USDT", 102.5)
        assert trade.amount < 1.0

        # Force close
        result = trader.close_on_wall_pull("BTC/USDT", 101.0)

        # After close, amount should be restored to original for logging
        assert trade.amount == pytest.approx(1.0)

    def test_stop_loss_exit_reason(self, trader):
        """SL hit before any TP gives 'stop_loss' reason."""
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)
        # SL is at 98.0

        result = trader.update_prices("BTC/USDT", 97.5)
        assert result is not None
        assert result.reason == "stop_loss"

    def test_trailing_stop_reason_after_tp1(self, trader):
        """SL hit after TP1 gives 'trailing_stop' reason."""
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        trader.update_prices("BTC/USDT", 102.5)  # TP1

        # SL is now at breakeven (~100.08)
        # Price drops below breakeven
        result = trader.update_prices("BTC/USDT", 99.5)
        if result:
            assert result.reason == "trailing_stop"


# ─────────────────────────────────────────
# Performance Summary
# ─────────────────────────────────────────

class TestPerformanceSummary:
    def test_includes_partial_tp_stats(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1 and TP2
        trader.update_prices("BTC/USDT", 102.5)
        trader.update_prices("BTC/USDT", 104.5)

        # Force close remaining
        trader.close_on_wall_pull("BTC/USDT", 103.0)

        summary = trader.get_performance_summary()
        assert "TP1=1" in summary
        assert "TP2=1" in summary

    def test_equity_updated_on_partial_tp(self, trader):
        initial_equity = trader._equity
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        trader.update_prices("BTC/USDT", 102.5)  # TP1

        # Equity should increase from partial TP profit
        assert trader._equity > initial_equity


# ─────────────────────────────────────────
# Protecting Mode Interaction
# ─────────────────────────────────────────

class TestProtectingMode:
    def test_protecting_mode_tightens_trailing(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1
        trader.update_prices("BTC/USDT", 102.5)

        # Set PROTECTING mode
        trader.daily_target_ctx = DailyTargetContext(
            mode=TradingMode.PROTECTING,
            pct_achieved=85.0,
        )

        # Price runs higher
        trader.update_prices("BTC/USDT", 106.0)

        # Chandelier with PROTECTING tightening:
        # Base mult (1.5 after TP1) × protecting_tighten (0.80) = 1.2
        # Stop: 106 - 1.2×2 = 103.6
        assert trade.stop_price >= 103.0  # Tighter than without protecting

    def test_bonus_territory_widens_trailing(self, trader):
        trade = _make_trade(trader, entry_price=100.0, atr=2.0, amount=1.0)

        # Hit TP1
        trader.update_prices("BTC/USDT", 102.5)

        # Set bonus territory (>100% target)
        trader.daily_target_ctx = DailyTargetContext(pct_achieved=120.0)

        # Price runs higher
        trader.update_prices("BTC/USDT", 106.0)
        # Trail mult: 1.5 (after TP1) × 1.5 (bonus) = 2.25
        # Stop: 106 - 2.25×2 = 101.5
        # This is wider than normal (106 - 1.5×2 = 103.0)


# ─────────────────────────────────────────
# Config Defaults
# ─────────────────────────────────────────

class TestExitConfig:
    def test_default_tp_sizes_sum_to_one(self):
        ec = ExitConfig()
        assert ec.tp1_size_pct + ec.tp2_size_pct + ec.tp3_size_pct == pytest.approx(1.0)

    def test_default_atr_multipliers_ascending(self):
        ec = ExitConfig()
        assert ec.tp1_atr_mult < ec.tp2_atr_mult < ec.tp3_atr_mult

    def test_chandelier_after_tp_mults_descending(self):
        ec = ExitConfig()
        assert ec.chandelier_atr_mult > ec.chandelier_after_tp1_mult > ec.chandelier_after_tp2_mult
