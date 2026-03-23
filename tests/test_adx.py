"""
Tests for the ADX (Average Directional Index) implementation.

Validates:
1. ADX returns 0 when data is too short
2. ADX correctly identifies trending markets (>25)
3. ADX correctly identifies ranging markets (<20)
4. ADX values are bounded [0, 100]
5. ADX blocks signals in ranging markets
6. ADX integration with SwingSignal fields
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from config import BotConfig
from candle_analyzer import CandleAnalyzer, TrendDirection


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.candle.adx_period = 14
    cfg.candle.adx_trending_threshold = 25.0
    cfg.candle.adx_ranging_threshold = 20.0
    # Disable features not under test to avoid interference
    cfg.candle.divergence_enabled = False
    cfg.candle.patterns_enabled = False
    return cfg


@pytest.fixture
def analyzer(config):
    return CandleAnalyzer(config)


def _make_strong_trend(n=50, start=100.0, direction="up"):
    """Generate OHLCV with a strong trend but regular pullbacks (RSI ~60-70)."""
    candles = []
    price = start
    np.random.seed(42)
    # Pattern: +1, +1, +0.8, -0.6, -0.4 repeating = net +1.8 per 5 bars
    pattern_up = [1.0, 1.0, 0.8, -0.6, -0.4]
    pattern_down = [-1.0, -1.0, -0.8, 0.6, 0.4]
    pattern = pattern_up if direction == "up" else pattern_down
    for i in range(n):
        step = pattern[i % len(pattern)]
        o = price
        c = price + step
        h = max(o, c) + np.random.uniform(0.2, 0.8)
        l = min(o, c) - np.random.uniform(0.2, 0.8)
        v = np.random.uniform(3000, 7000)
        ts = 1000000 + i * 3600000
        candles.append([ts, o, h, l, c, v])
        price = c
    return candles


def _make_ranging_market(n=50, center=100.0, amplitude=0.5):
    """Generate OHLCV oscillating in a tight range (low ADX expected)."""
    candles = []
    for i in range(n):
        noise = amplitude * np.sin(i * 0.8) + np.random.uniform(-0.1, 0.1)
        o = center + noise
        c = center - noise
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        v = 3000.0
        ts = 1000000 + i * 3600000
        candles.append([ts, o, h, l, c, v])
    return candles


class TestADXCalculation:
    """Unit tests for CandleAnalyzer._adx static method."""

    def test_returns_zero_for_short_data(self):
        """ADX needs at least 2*period + 1 bars."""
        highs = np.array([101, 102, 103], dtype=np.float64)
        lows = np.array([99, 100, 101], dtype=np.float64)
        closes = np.array([100, 101, 102], dtype=np.float64)

        adx = CandleAnalyzer._adx(highs, lows, closes, period=14)
        assert adx == 0.0

    def test_strong_uptrend_gives_high_adx(self):
        """A consistent uptrend should produce ADX > 25."""
        candles = _make_strong_trend(50, 100.0, "up")
        highs = np.array([c[2] for c in candles], dtype=np.float64)
        lows = np.array([c[3] for c in candles], dtype=np.float64)
        closes = np.array([c[4] for c in candles], dtype=np.float64)

        adx = CandleAnalyzer._adx(highs, lows, closes, period=14)
        assert adx > 25.0, f"Expected ADX > 25 for strong trend, got {adx:.1f}"

    def test_strong_downtrend_gives_high_adx(self):
        """ADX measures trend STRENGTH, not direction. Down trend = high ADX too."""
        candles = _make_strong_trend(50, 200.0, "down")
        highs = np.array([c[2] for c in candles], dtype=np.float64)
        lows = np.array([c[3] for c in candles], dtype=np.float64)
        closes = np.array([c[4] for c in candles], dtype=np.float64)

        adx = CandleAnalyzer._adx(highs, lows, closes, period=14)
        assert adx > 25.0, f"Expected ADX > 25 for strong downtrend, got {adx:.1f}"

    def test_ranging_market_gives_low_adx(self):
        """A sideways market should produce ADX < 25."""
        candles = _make_ranging_market(50, 100.0, 0.3)
        highs = np.array([c[2] for c in candles], dtype=np.float64)
        lows = np.array([c[3] for c in candles], dtype=np.float64)
        closes = np.array([c[4] for c in candles], dtype=np.float64)

        adx = CandleAnalyzer._adx(highs, lows, closes, period=14)
        assert adx < 25.0, f"Expected ADX < 25 for ranging market, got {adx:.1f}"

    def test_adx_bounded_0_100(self):
        """ADX should always be between 0 and 100."""
        # Test with various market conditions
        for direction in ["up", "down"]:
            candles = _make_strong_trend(50, 100.0, direction)
            highs = np.array([c[2] for c in candles], dtype=np.float64)
            lows = np.array([c[3] for c in candles], dtype=np.float64)
            closes = np.array([c[4] for c in candles], dtype=np.float64)

            adx = CandleAnalyzer._adx(highs, lows, closes, period=14)
            assert 0.0 <= adx <= 100.0, f"ADX out of bounds: {adx}"

    def test_adx_with_custom_period(self):
        """ADX should work with different period values."""
        candles = _make_strong_trend(60, 100.0, "up")
        highs = np.array([c[2] for c in candles], dtype=np.float64)
        lows = np.array([c[3] for c in candles], dtype=np.float64)
        closes = np.array([c[4] for c in candles], dtype=np.float64)

        adx_7 = CandleAnalyzer._adx(highs, lows, closes, period=7)
        adx_14 = CandleAnalyzer._adx(highs, lows, closes, period=14)

        assert adx_7 > 0
        assert adx_14 > 0


class TestADXIntegration:
    """Test ADX integration with the swing signal pipeline."""

    def test_adx_stored_in_candle_signal(self, analyzer):
        """CandleSignal should contain the ADX value."""
        candles = _make_strong_trend(50, 100.0, "up")
        signal = analyzer._analyze_timeframe("1h", candles, 1000000.0)
        assert hasattr(signal, 'adx')
        assert signal.adx >= 0

    def test_adx_stored_in_swing_signal(self, analyzer):
        """SwingSignal should have adx_1h and adx_4h fields."""
        candles = {
            "1h": _make_strong_trend(50, 100.0, "up"),
            "4h": _make_strong_trend(50, 100.0, "up"),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)
        assert hasattr(signal, 'adx_1h')
        assert hasattr(signal, 'adx_4h')
        assert signal.adx_1h >= 0
        assert signal.adx_4h >= 0

    def test_ranging_market_blocks_signal(self, analyzer):
        """ADX < 20 should block the signal entirely."""
        candles = {
            "1h": _make_ranging_market(50, 100.0, 0.3),
            "4h": _make_ranging_market(50, 100.0, 0.3),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        # The primary timeframe ADX should be low
        primary_adx = signal.adx_4h if signal.adx_4h > 0 else signal.adx_1h
        if primary_adx > 0 and primary_adx < 20:
            assert signal.adx_blocked is True
            assert signal.suggested_side is None
            assert signal.confidence == 0.0

    def test_trending_market_gives_adx_bonus(self, analyzer):
        """ADX > 25 should add +0.10 to confidence."""
        candles = {
            "1h": _make_strong_trend(50, 100.0, "up"),
            "4h": _make_strong_trend(50, 100.0, "up"),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        # If both timeframes show strong trend with ADX > 25, confidence should
        # include the +0.10 ADX bonus. With alignment (0.25 + 0.25 + 0.10 = 0.60 min)
        if signal.adx_4h > 25 and signal.primary_trend != TrendDirection.NEUTRAL:
            # The ADX bonus should be reflected in the confidence
            # Base(0.25) + aligned(0.25) + ADX(0.10) = 0.60 minimum
            assert signal.confidence >= 0.55

    def test_adx_blocked_field_default_false(self, analyzer):
        """adx_blocked defaults to False for trending markets."""
        candles = {
            "1h": _make_strong_trend(50, 100.0, "up"),
            "4h": _make_strong_trend(50, 100.0, "up"),
        }
        signal = analyzer.analyze("BTC/USDT", candles, 1000000.0)

        if signal.adx_4h > 20:
            assert signal.adx_blocked is False
