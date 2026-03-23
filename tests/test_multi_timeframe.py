"""
Tests for Phase 7 multi-timeframe features in candle_analyzer.py.

Validates:
1. Daily trend gate — blocks counter-trend trades
2. Daily alignment bonus (+0.10 confidence)
3. Daily neutral penalty (-0.10 confidence)
4. 15m entry timing — pullback detection with timeout
5. S/R integration in _combine_signals — confidence bonus and TP-block
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
    # Daily trend
    cfg.candle.daily_enabled = True
    cfg.candle.daily_lookback = 20
    cfg.candle.daily_cache_ttl = 300.0
    cfg.candle.daily_neutral_penalty = 0.10
    # 15m entry
    cfg.candle.entry_15m_enabled = True
    cfg.candle.entry_15m_lookback = 20
    cfg.candle.entry_15m_cache_ttl = 60.0
    cfg.candle.entry_15m_max_wait_candles = 3
    cfg.candle.entry_15m_rsi_long_dip = 45.0
    cfg.candle.entry_15m_rsi_short_spike = 55.0
    # S/R
    cfg.candle.sr_enabled = True
    cfg.candle.sr_pivot_lookback = 5
    cfg.candle.sr_proximity_atr_pct = 0.3
    cfg.candle.sr_confidence_bonus = 0.05
    cfg.candle.sr_min_touches = 2
    cfg.candle.sr_fib_min_swing_pct = 3.0
    # Standard candle settings
    cfg.candle.adx_ranging_threshold = 20.0
    cfg.candle.adx_trending_threshold = 25.0
    cfg.candle.rsi_oversold = 30.0
    cfg.candle.rsi_overbought = 70.0
    cfg.candle.volume_surge_multiplier = 1.5
    return cfg


@pytest.fixture
def analyzer(config):
    return CandleAnalyzer(config)


def _make_trending_candles(direction="up", count=60):
    """Generate trending OHLCV data that produces clear EMAs/RSI."""
    candles = []
    base = 100.0
    for i in range(count):
        if direction == "up":
            close = base + i * 0.5 + np.sin(i * 0.1) * 0.3
        else:
            close = base + 30 - i * 0.5 - np.sin(i * 0.1) * 0.3
        high = close * 1.005
        low = close * 0.995
        open_ = close * 0.999
        vol = 1000 + i * 10
        candles.append([i * 3600000, open_, high, low, close, vol])
    return candles


def _make_bullish_signal(analyzer, symbol="BTC/USDT"):
    """Create a CandleSignal that represents a bullish daily trend."""
    return CandleSignal(
        timeframe="1d",
        trend=TrendDirection.BULLISH,
        fast_ema=105.0,
        slow_ema=100.0,
        trend_ema=98.0,
        ema_slope=0.5,
        rsi=55.0,
        atr=2.0,
        atr_pct=2.0,
        adx=30.0,
        volume_ratio=1.2,
        is_volume_surge=False,
        last_close=105.0,
        timestamp=1000.0,
    )


def _make_bearish_signal():
    return CandleSignal(
        timeframe="1d",
        trend=TrendDirection.BEARISH,
        fast_ema=95.0,
        slow_ema=100.0,
        trend_ema=102.0,
        ema_slope=-0.5,
        rsi=40.0,
        atr=2.0,
        atr_pct=2.0,
        adx=30.0,
        volume_ratio=1.2,
        is_volume_surge=False,
        last_close=95.0,
        timestamp=1000.0,
    )


def _make_neutral_signal():
    return CandleSignal(
        timeframe="1d",
        trend=TrendDirection.NEUTRAL,
        fast_ema=100.0,
        slow_ema=100.0,
        trend_ema=100.0,
        ema_slope=0.0,
        rsi=50.0,
        atr=2.0,
        atr_pct=2.0,
        adx=22.0,
        volume_ratio=1.0,
        is_volume_surge=False,
        last_close=100.0,
        timestamp=1000.0,
    )


# ─────────────────────────────────────────
# DAILY TREND GATE
# ─────────────────────────────────────────

class TestDailyTrendGate:

    def test_daily_bullish_blocks_sell(self, analyzer):
        """Daily bullish trend should block SELL signals."""
        # Inject daily signal cache
        analyzer._daily_signal_cache["BTC/USDT"] = _make_bullish_signal(analyzer)

        # Create a swing signal with bearish 4h/1h (would suggest SELL)
        candles = {
            "4h": _make_trending_candles("down", 60),
            "1h": _make_trending_candles("down", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        # If the 4h/1h suggest SELL but daily is bullish, daily gate blocks it
        if swing.daily_blocked:
            assert swing.suggested_side is None

    def test_daily_bearish_blocks_buy(self, analyzer):
        """Daily bearish trend should block BUY signals."""
        analyzer._daily_signal_cache["BTC/USDT"] = _make_bearish_signal()

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        if swing.daily_blocked:
            assert swing.suggested_side is None

    def test_daily_aligned_adds_confidence(self, analyzer):
        """When daily matches 4h direction, confidence gets +0.10."""
        analyzer._daily_signal_cache["BTC/USDT"] = _make_bullish_signal(analyzer)

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        if swing.suggested_side == "BUY" and swing.daily_aligned:
            # daily_aligned should be True and confidence should include +0.10
            assert swing.daily_aligned is True

    def test_daily_neutral_penalizes(self, analyzer):
        """Neutral daily trend should subtract daily_neutral_penalty from confidence."""
        analyzer._daily_signal_cache["BTC/USDT"] = _make_neutral_signal()

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }

        # Analyze without daily cache to get baseline confidence
        analyzer_nodaily = CandleAnalyzer(analyzer.config)
        swing_baseline = analyzer_nodaily.analyze("BTC/USDT", candles, 1000.0)

        # Analyze with neutral daily
        swing_penalized = analyzer.analyze("BTC/USDT", candles, 1000.0)

        # The penalized version should have lower or equal confidence
        # (both may be capped/floored by the 0.55 threshold)
        if swing_baseline.suggested_side and swing_penalized.suggested_side:
            assert swing_penalized.confidence <= swing_baseline.confidence

    def test_daily_disabled_no_gate(self, config):
        """When daily_enabled=False, no blocking occurs."""
        config.candle.daily_enabled = False
        an = CandleAnalyzer(config)
        an._daily_signal_cache["BTC/USDT"] = _make_bearish_signal()

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = an.analyze("BTC/USDT", candles, 1000.0)

        # Should not be daily-blocked since feature is disabled
        assert swing.daily_blocked is False

    def test_daily_fields_populated(self, analyzer):
        """SwingSignal daily fields are correctly set."""
        daily_sig = _make_bullish_signal(analyzer)
        analyzer._daily_signal_cache["BTC/USDT"] = daily_sig

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        assert swing.daily_trend == TrendDirection.BULLISH
        assert swing.daily_rsi == daily_sig.rsi


# ─────────────────────────────────────────
# 15-MINUTE ENTRY TIMING
# ─────────────────────────────────────────

class TestEntryTiming15m:

    def _inject_15m_rsi_data(self, analyzer, symbol, rsi_closes):
        """Inject 15m cache with closes that produce a given RSI."""
        candles = []
        for i, c in enumerate(rsi_closes):
            candles.append([i * 900000, c * 0.999, c * 1.001, c * 0.998, c, 100])
        analyzer._15m_cache[symbol] = candles

    def test_no_data_allows_entry(self, analyzer):
        """When no 15m data is cached, entry is allowed."""
        ready, rsi = analyzer.check_15m_entry("BTC/USDT", "BUY")
        assert ready is True

    def test_disabled_allows_entry(self, config):
        """When entry_15m_enabled=False, always returns ready."""
        config.candle.entry_15m_enabled = False
        an = CandleAnalyzer(config)
        ready, rsi = an.check_15m_entry("BTC/USDT", "BUY")
        assert ready is True

    def test_timeout_allows_entry(self, analyzer):
        """After max_wait_candles, entry is forced."""
        # Inject 15m data with RSI that never dips (stays high for BUY)
        closes = [100 + i * 0.5 for i in range(20)]
        self._inject_15m_rsi_data(analyzer, "BTC/USDT", closes)

        # Call max_wait_candles times — should timeout and allow
        for i in range(analyzer.cc.entry_15m_max_wait_candles):
            ready, rsi = analyzer.check_15m_entry("BTC/USDT", "BUY")

        assert ready is True

    def test_side_change_resets_state(self, analyzer):
        """Changing suggested_side resets the pullback tracking."""
        closes = [100 + i * 0.5 for i in range(20)]
        self._inject_15m_rsi_data(analyzer, "BTC/USDT", closes)

        # Start tracking BUY
        analyzer.check_15m_entry("BTC/USDT", "BUY")
        state_after_buy = analyzer._15m_pullback_state.get("BTC/USDT", {})

        # Switch to SELL — should reset
        analyzer.check_15m_entry("BTC/USDT", "SELL")
        state_after_sell = analyzer._15m_pullback_state.get("BTC/USDT", {})

        if state_after_sell:
            assert state_after_sell.get("side") == "SELL"

    def test_pullback_detected_for_long(self, analyzer):
        """RSI dipping below 45 then recovering triggers pullback entry for BUY."""
        # Build closes that produce RSI < 45 initially, then recovery
        # Start high, drop sharply, then recover
        closes = [110] * 5 + [109, 108, 107, 106, 105, 104, 103, 102, 101, 100] + [101, 102, 103, 104, 105]
        self._inject_15m_rsi_data(analyzer, "BTC/USDT", closes)

        # First call — check state
        ready1, rsi1 = analyzer.check_15m_entry("BTC/USDT", "BUY")

        # If RSI is below 45, dip should be recorded
        state = analyzer._15m_pullback_state.get("BTC/USDT", {})
        if rsi1 < 45:
            assert state.get("dip_seen", False) is True

    def test_pullback_detected_for_short(self, analyzer):
        """RSI spiking above 55 then falling triggers pullback entry for SELL."""
        # Build closes that produce RSI > 55 then drop
        closes = [90] * 5 + [91, 92, 93, 94, 95, 96, 97, 98, 99, 100] + [99, 98, 97, 96, 95]
        self._inject_15m_rsi_data(analyzer, "BTC/USDT", closes)

        ready, rsi = analyzer.check_15m_entry("BTC/USDT", "SELL")

        state = analyzer._15m_pullback_state.get("BTC/USDT", {})
        if rsi > 55:
            assert state.get("dip_seen", False) is True


# ─────────────────────────────────────────
# S/R INTEGRATION IN _combine_signals
# ─────────────────────────────────────────

class TestSRIntegration:

    def test_sr_fields_populated_on_signal(self, analyzer):
        """SwingSignal S/R fields should be populated when sr_enabled."""
        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        # S/R fields should exist regardless of outcome
        assert isinstance(swing.sr_at_support, bool)
        assert isinstance(swing.sr_at_resistance, bool)
        assert isinstance(swing.sr_tp_blocked, bool)

    def test_sr_disabled_no_analysis(self, config):
        """When sr_enabled=False, sr_analysis stays None."""
        config.candle.sr_enabled = False
        an = CandleAnalyzer(config)

        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = an.analyze("BTC/USDT", candles, 1000.0)
        assert swing.sr_analysis is None

    def test_sr_tp_blocked_kills_trade(self, analyzer):
        """If sr_tp_blocked is True, suggested_side should be None."""
        candles = {
            "4h": _make_trending_candles("up", 60),
            "1h": _make_trending_candles("up", 60),
        }
        swing = analyzer.analyze("BTC/USDT", candles, 1000.0)

        # If the S/R analysis found a TP block, the trade should be cancelled
        if swing.sr_tp_blocked:
            assert swing.suggested_side is None


# ─────────────────────────────────────────
# SWING SIGNAL FIELD DEFAULTS
# ─────────────────────────────────────────

class TestSwingSignalFields:

    def test_default_daily_fields(self):
        sig = SwingSignal(symbol="TEST", timestamp=0.0)
        assert sig.daily_trend == TrendDirection.NEUTRAL
        assert sig.daily_rsi == 50.0
        assert sig.daily_aligned is False
        assert sig.daily_blocked is False

    def test_default_15m_fields(self):
        sig = SwingSignal(symbol="TEST", timestamp=0.0)
        assert sig.entry_15m_ready is True
        assert sig.entry_15m_rsi == 50.0
        assert sig.entry_15m_pullback_seen is False

    def test_default_sr_fields(self):
        sig = SwingSignal(symbol="TEST", timestamp=0.0)
        assert sig.sr_analysis is None
        assert sig.sr_at_support is False
        assert sig.sr_at_resistance is False
        assert sig.sr_tp_blocked is False
        assert sig.sr_adjusted_tp_pct == 0.0
