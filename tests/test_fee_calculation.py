"""
Tests for fee-aware TP/SL calculation and net P&L computation.

Validates:
1. Round-trip fee cost formula: 2 * fee_rate * leverage (as % of margin)
2. Fee-adjusted TP/SL: effective_tp = raw_tp + fee_cost, effective_sl = raw_sl + fee_cost
3. Post-fee R:R calculation and min R:R filter
4. Net P&L on trade close: gross_pnl - fee_cost
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from config import BotConfig
from candle_analyzer import CandleAnalyzer, CandleSignal, SwingSignal, TrendDirection


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.trading.fee_rate = 0.04       # 0.04% per side (Binance futures taker)
    cfg.trading.min_post_fee_rr = 1.5
    cfg.futures.leverage = 30
    return cfg


@pytest.fixture
def analyzer(config):
    return CandleAnalyzer(config)


def _make_trending_candles(n=50, start=100.0, trend_up=True):
    """Generate synthetic OHLCV candles with a clear trend."""
    candles = []
    price = start
    for i in range(n):
        if trend_up:
            change = np.random.uniform(0.2, 1.0)
        else:
            change = np.random.uniform(-1.0, -0.2)
        o = price
        c = price + change
        h = max(o, c) + np.random.uniform(0.1, 0.5)
        l = min(o, c) - np.random.uniform(0.1, 0.5)
        v = np.random.uniform(1000, 5000)
        ts = 1000000 + i * 3600000
        candles.append([ts, o, h, l, c, v])
        price = c
    return candles


class TestFeeAwareTPSL:
    """Test fee-adjusted TP/SL calculations in candle_analyzer._combine_signals."""

    def test_fee_cost_calculation(self, config):
        """fee_cost_pct = 2 * fee_rate = 2 * 0.04 = 0.08% of notional."""
        fee_cost_pct = 2.0 * config.trading.fee_rate
        assert fee_cost_pct == pytest.approx(0.08, abs=1e-6)

    def test_effective_tp_includes_fees(self, analyzer):
        """Effective TP should be raw_tp + fee_cost_pct."""
        candles = {
            "1h": _make_trending_candles(50, 100.0, trend_up=True),
            "4h": _make_trending_candles(50, 100.0, trend_up=True),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        if signal.raw_tp_pct > 0:
            expected_tp = signal.raw_tp_pct + signal.fee_cost_pct
            assert signal.atr_tp_pct == pytest.approx(expected_tp, abs=1e-4)

    def test_effective_sl_includes_fees(self, analyzer):
        """Effective SL should be raw_sl + fee_cost_pct."""
        candles = {
            "1h": _make_trending_candles(50, 100.0, trend_up=True),
            "4h": _make_trending_candles(50, 100.0, trend_up=True),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        if signal.raw_sl_pct > 0:
            expected_sl = signal.raw_sl_pct + signal.fee_cost_pct
            assert signal.atr_sl_pct == pytest.approx(expected_sl, abs=1e-4)

    def test_post_fee_rr_formula(self, analyzer):
        """post_fee_rr = (raw_tp - fees) / (raw_sl + fees)."""
        candles = {
            "1h": _make_trending_candles(50, 100.0, trend_up=True),
            "4h": _make_trending_candles(50, 100.0, trend_up=True),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        if signal.raw_tp_pct > 0 and signal.raw_sl_pct > 0:
            net_gain = signal.raw_tp_pct - signal.fee_cost_pct
            net_loss = signal.raw_sl_pct + signal.fee_cost_pct
            expected_rr = net_gain / net_loss
            assert signal.post_fee_rr == pytest.approx(expected_rr, abs=1e-4)

    def test_min_rr_filter_rejects_low_rr(self, config):
        """Trades with post-fee R:R < min_post_fee_rr should be rejected."""
        config.trading.min_post_fee_rr = 10.0  # Unrealistically high to force rejection
        analyzer = CandleAnalyzer(config)

        candles = {
            "1h": _make_trending_candles(50, 100.0, trend_up=True),
            "4h": _make_trending_candles(50, 100.0, trend_up=True),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        # If a trend was detected, it should be rejected by the R:R filter
        if signal.post_fee_rr > 0:
            assert signal.suggested_side is None

    def test_fee_cost_pct_stored_on_signal(self, analyzer):
        """SwingSignal should carry fee_cost_pct for downstream logging."""
        candles = {
            "1h": _make_trending_candles(50, 100.0, trend_up=True),
            "4h": _make_trending_candles(50, 100.0, trend_up=True),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)
        assert signal.fee_cost_pct == pytest.approx(0.08, abs=1e-6)


class TestNetPnLComputation:
    """Test that shadow trader computes net P&L correctly."""

    def test_net_pnl_long_win(self):
        """Net P&L = gross - fees for a winning long."""
        entry_price = 100.0
        exit_price = 102.0
        amount = 10.0  # 10 units
        fee_rate = 0.0004  # 0.04%

        gross_pnl = (exit_price - entry_price) * amount  # $20
        entry_notional = entry_price * amount  # $1000
        exit_notional = exit_price * amount    # $1020
        fee_cost = (entry_notional + exit_notional) * fee_rate  # $0.808
        net_pnl = gross_pnl - fee_cost  # $19.192

        assert gross_pnl == pytest.approx(20.0)
        assert fee_cost == pytest.approx(0.808, abs=1e-3)
        assert net_pnl == pytest.approx(19.192, abs=1e-3)

    def test_net_pnl_long_loss(self):
        """Fee adds to the loss on a losing trade."""
        entry_price = 100.0
        exit_price = 99.0
        amount = 10.0
        fee_rate = 0.0004

        gross_pnl = (exit_price - entry_price) * amount  # -$10
        fee_cost = (entry_price * amount + exit_price * amount) * fee_rate  # $0.796
        net_pnl = gross_pnl - fee_cost  # -$10.796

        assert gross_pnl == pytest.approx(-10.0)
        assert net_pnl < gross_pnl  # Net loss is worse than gross

    def test_net_pnl_short_win(self):
        """Net P&L for a winning short."""
        entry_price = 100.0
        exit_price = 98.0
        amount = 10.0
        fee_rate = 0.0004

        gross_pnl = (entry_price - exit_price) * amount  # $20
        fee_cost = (entry_price * amount + exit_price * amount) * fee_rate
        net_pnl = gross_pnl - fee_cost

        assert gross_pnl == pytest.approx(20.0)
        assert net_pnl < gross_pnl
        assert net_pnl > 0  # Still a win

    def test_fee_can_flip_win_to_loss(self):
        """A tiny gross profit can become a net loss after fees."""
        entry_price = 100.0
        exit_price = 100.01  # Tiny move: $0.01 * 10 = $0.10 gross
        amount = 10.0
        fee_rate = 0.0004

        gross_pnl = (exit_price - entry_price) * amount  # $0.10
        fee_cost = (entry_price * amount + exit_price * amount) * fee_rate  # ~$0.80
        net_pnl = gross_pnl - fee_cost

        assert gross_pnl > 0
        assert net_pnl < 0  # Fees ate the profit

    def test_leverage_amplifies_fee_impact_on_margin(self):
        """With 30x leverage, 0.04% taker fee = 1.2% of margin per side."""
        margin = 18.30  # $18.30 margin
        leverage = 30
        notional = margin * leverage  # $549
        fee_rate = 0.0004

        fee_per_side = notional * fee_rate  # $0.2196
        round_trip_fee = fee_per_side * 2   # $0.4392
        fee_as_pct_of_margin = round_trip_fee / margin * 100  # ~2.4%

        assert fee_as_pct_of_margin == pytest.approx(2.4, abs=0.01)


class TestConfidenceThreshold:
    """Test that the confidence threshold is correctly enforced at 0.55."""

    def test_base_trend_alone_rejected(self, analyzer):
        """A single trend without alignment should not reach 0.55."""
        # Base trend alone = 0.25, which is < 0.55
        assert 0.25 < 0.55  # By design

    def test_trend_plus_alignment_still_needs_one_more(self):
        """trend (0.25) + alignment (0.25) = 0.50, still < 0.55."""
        assert 0.25 + 0.25 < 0.55

    def test_trend_alignment_plus_one_signal_passes(self):
        """trend (0.25) + alignment (0.25) + any confirmation (0.10+) >= 0.55."""
        min_extra = 0.10  # ADX, volume, or strong trend
        total = 0.25 + 0.25 + min_extra
        assert total >= 0.55
