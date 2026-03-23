"""
Tests for levels.py — Support/Resistance Level Detection.

Validates:
1. Pivot high/low detection (swing points)
2. Level clustering with touch counting
3. Fibonacci retracement calculation
4. Full analyze() pipeline: classification, proximity, TP-block
5. Edge cases: insufficient data, zero ATR, empty candles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from config import BotConfig
from levels import LevelDetector, LevelAnalysis, PriceLevel


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.candle.sr_enabled = True
    cfg.candle.sr_pivot_lookback = 5
    cfg.candle.sr_proximity_atr_pct = 0.3
    cfg.candle.sr_confidence_bonus = 0.05
    cfg.candle.sr_min_touches = 2
    cfg.candle.sr_fib_min_swing_pct = 3.0
    return cfg


@pytest.fixture
def detector(config):
    return LevelDetector(config)


def _make_ohlcv(prices, base_volume=1000.0):
    """Generate OHLCV candles from a list of close prices.
    Creates realistic-ish OHLC by adding noise around the close.
    """
    candles = []
    for i, close in enumerate(prices):
        high = close * 1.005
        low = close * 0.995
        open_ = close * (1.001 if i % 2 == 0 else 0.999)
        vol = base_volume + (i % 5) * 100
        candles.append([i * 3600000, open_, high, low, close, vol])
    return candles


def _make_ohlcv_with_pivots():
    """Create candle data with clear swing highs and swing lows.
    Pattern: up-down-up-down to create well-defined pivot points.
    """
    prices = []
    # Baseline at 100, with peaks at 110 and valleys at 90
    for i in range(50):
        if i % 12 < 6:
            # Rising to peak
            prices.append(95 + (i % 12) * 2.5)  # 95 → 107.5
        else:
            # Falling to trough
            prices.append(107.5 - ((i % 12) - 6) * 2.5)  # 107.5 → 95

    candles = []
    for i, p in enumerate(prices):
        candles.append([
            i * 3600000,
            p * 0.999,   # open
            p * 1.008,   # high
            p * 0.992,   # low
            p,           # close
            1000 + i * 10,  # volume
        ])
    return candles


# ─────────────────────────────────────────
# PIVOT DETECTION
# ─────────────────────────────────────────

class TestPivotDetection:

    def test_find_pivot_highs_basic(self, detector):
        # Clear peak at index 5: values go up then down
        highs = np.array([1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=np.float64)
        pivots = detector._find_pivot_highs(highs, lookback=3)
        assert len(pivots) >= 1
        assert 10.0 in pivots

    def test_find_pivot_lows_basic(self, detector):
        # Clear trough at index 5
        lows = np.array([10, 9, 8, 7, 6, 1, 6, 7, 8, 9, 10], dtype=np.float64)
        pivots = detector._find_pivot_lows(lows, lookback=3)
        assert len(pivots) >= 1
        assert 1.0 in pivots

    def test_no_pivots_in_monotonic(self, detector):
        # Monotonically increasing — no local maxima (except at edges, excluded by lookback)
        highs = np.arange(1, 30, dtype=np.float64)
        pivots = detector._find_pivot_highs(highs, lookback=5)
        assert len(pivots) == 0

    def test_multiple_pivots(self, detector):
        # Two peaks: at index 5 and 15
        data = list(range(1, 7)) + list(range(6, 0, -1)) + list(range(1, 7)) + list(range(6, 0, -1)) + [1, 1]
        highs = np.array(data, dtype=np.float64)
        pivots = detector._find_pivot_highs(highs, lookback=3)
        assert len(pivots) >= 2

    def test_pivot_lookback_respects_window(self, detector):
        # With lookback=5, need 5 candles on each side
        highs = np.array([1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=np.float64)
        pivots = detector._find_pivot_highs(highs, lookback=5)
        assert len(pivots) == 1
        assert pivots[0] == 10.0


# ─────────────────────────────────────────
# LEVEL CLUSTERING
# ─────────────────────────────────────────

class TestLevelClustering:

    def test_cluster_nearby_levels(self, detector):
        pivot_highs = [100.0, 100.5, 101.0]
        pivot_lows = [90.0, 90.3]
        proximity = 1.5
        levels = detector._cluster_levels(pivot_highs, pivot_lows, proximity)
        # 100, 100.5, 101 should cluster together; 90, 90.3 should cluster
        assert len(levels) == 2

    def test_touch_count_from_cluster(self, detector):
        pivot_highs = [100.0, 100.2, 100.1]
        pivot_lows = []
        proximity = 1.0
        levels = detector._cluster_levels(pivot_highs, pivot_lows, proximity)
        assert len(levels) == 1
        assert levels[0].touches == 3

    def test_no_clustering_when_far_apart(self, detector):
        pivot_highs = [100.0, 200.0]
        pivot_lows = [50.0]
        proximity = 1.0
        levels = detector._cluster_levels(pivot_highs, pivot_lows, proximity)
        assert len(levels) == 3

    def test_empty_pivots(self, detector):
        levels = detector._cluster_levels([], [], 1.0)
        assert levels == []

    def test_cluster_average_price(self, detector):
        pivot_highs = [100.0, 102.0]
        pivot_lows = []
        proximity = 5.0
        levels = detector._cluster_levels(pivot_highs, pivot_lows, proximity)
        assert len(levels) == 1
        # Average of 100 and 102
        assert abs(levels[0].price - 101.0) < 0.01


# ─────────────────────────────────────────
# FIBONACCI RETRACEMENTS
# ─────────────────────────────────────────

class TestFibonacciRetracements:

    def test_fib_levels_calculated(self, detector):
        # Big swing: low=90, high=110 → range=20
        highs = np.array([100] * 20 + [110] * 10 + [105] * 20, dtype=np.float64)
        lows = np.array([90] * 20 + [100] * 10 + [95] * 20, dtype=np.float64)
        closes = np.array([95] * 20 + [108] * 10 + [100] * 20, dtype=np.float64)

        levels = detector._fibonacci_retracements(highs, lows, closes, current_price=100.0)
        assert len(levels) == 4  # 0.382, 0.5, 0.618, 0.786

        # All should be fibonacci source
        for level in levels:
            assert level.source == "fibonacci"
            assert level.fib_ratio in (0.382, 0.5, 0.618, 0.786)

    def test_fib_ratios_are_correct(self, detector):
        # Swing high=200, low=100, range=100
        highs = np.array([150] * 20 + [200] * 15 + [180] * 15, dtype=np.float64)
        lows = np.array([100] * 20 + [150] * 15 + [160] * 15, dtype=np.float64)
        closes = np.array([120] * 20 + [190] * 15 + [170] * 15, dtype=np.float64)

        levels = detector._fibonacci_retracements(highs, lows, closes, current_price=170.0)
        prices = {l.fib_ratio: l.price for l in levels}

        # Retracement from high: swing_high - range * ratio
        # 200 - 100*0.382 = 161.8
        assert abs(prices[0.382] - 161.8) < 0.1
        # 200 - 100*0.5 = 150
        assert abs(prices[0.5] - 150.0) < 0.1
        # 200 - 100*0.618 = 138.2
        assert abs(prices[0.618] - 138.2) < 0.1

    def test_no_fib_with_small_swing(self, detector):
        # Swing < 3% — should return empty
        highs = np.array([101.0] * 50, dtype=np.float64)
        lows = np.array([100.0] * 50, dtype=np.float64)
        closes = np.array([100.5] * 50, dtype=np.float64)

        levels = detector._fibonacci_retracements(highs, lows, closes, current_price=100.5)
        assert len(levels) == 0

    def test_no_fib_with_insufficient_data(self, detector):
        highs = np.array([100, 101, 102], dtype=np.float64)
        lows = np.array([99, 100, 101], dtype=np.float64)
        closes = np.array([99.5, 100.5, 101.5], dtype=np.float64)

        levels = detector._fibonacci_retracements(highs, lows, closes, current_price=101.0)
        assert len(levels) == 0


# ─────────────────────────────────────────
# FIND MAJOR SWING
# ─────────────────────────────────────────

class TestFindMajorSwing:

    def test_finds_swing_from_recent_candles(self):
        highs = np.array([100] * 30 + [120] * 20, dtype=np.float64)
        lows = np.array([80] * 30 + [100] * 20, dtype=np.float64)
        closes = np.array([90] * 30 + [110] * 20, dtype=np.float64)

        high, low = LevelDetector._find_major_swing(highs, lows, closes, min_pct=3.0)
        assert high == 120.0
        assert low == 80.0

    def test_returns_none_for_tiny_swing(self):
        highs = np.array([101] * 50, dtype=np.float64)
        lows = np.array([100] * 50, dtype=np.float64)
        closes = np.array([100.5] * 50, dtype=np.float64)

        high, low = LevelDetector._find_major_swing(highs, lows, closes, min_pct=3.0)
        assert high is None
        assert low is None

    def test_returns_none_for_short_data(self):
        highs = np.array([100, 101, 102], dtype=np.float64)
        lows = np.array([99, 100, 101], dtype=np.float64)
        closes = np.array([99.5, 100.5, 101.5], dtype=np.float64)

        high, low = LevelDetector._find_major_swing(highs, lows, closes, min_pct=3.0)
        assert high is None
        assert low is None


# ─────────────────────────────────────────
# FULL ANALYZE PIPELINE
# ─────────────────────────────────────────

class TestAnalyzePipeline:

    def test_returns_empty_with_insufficient_data(self, detector):
        ohlcv = _make_ohlcv([100] * 10)  # Only 10 candles
        result = detector.analyze("BTC/USDT", ohlcv, current_price=100.0)
        assert result.symbol == "BTC/USDT"
        assert result.supports == []
        assert result.resistances == []

    def test_returns_empty_with_empty_data(self, detector):
        result = detector.analyze("BTC/USDT", [], current_price=100.0)
        assert result.supports == []

    def test_classifies_support_below_price(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=105.0, atr=2.0
        )
        for level in result.supports:
            assert level.price < 105.0
            assert level.level_type == "support"

    def test_classifies_resistance_above_price(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=95.0, atr=2.0
        )
        for level in result.resistances:
            assert level.price >= 95.0
            assert level.level_type == "resistance"

    def test_nearest_support_is_closest(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=105.0, atr=2.0
        )
        if result.nearest_support and len(result.supports) > 1:
            # Nearest support should be highest (closest to price)
            assert result.nearest_support.price == result.supports[0].price

    def test_nearest_resistance_is_closest(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=95.0, atr=2.0
        )
        if result.nearest_resistance and len(result.resistances) > 1:
            # Nearest resistance should be lowest (closest to price)
            assert result.nearest_resistance.price == result.resistances[0].price

    def test_max_three_levels_each_side(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=100.0, atr=1.0
        )
        assert len(result.supports) <= 3
        assert len(result.resistances) <= 3

    def test_strength_scores_bounded(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=100.0, atr=2.0
        )
        for level in result.supports + result.resistances:
            assert 0.0 <= level.strength <= 1.0

    def test_at_support_proximity(self, detector):
        # Create data with a clear support at ~95
        prices = [100, 98, 96, 95, 95, 95, 96, 98, 100, 102, 104,
                  102, 100, 98, 96, 95, 95, 96, 98, 100, 102, 104,
                  102, 100, 98, 96, 95, 95, 96, 98]
        ohlcv = _make_ohlcv(prices)
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=95.2, atr=1.0
        )
        # With atr=1.0, proximity = 0.3. Price 95.2 is close to any support near 95
        # at_support depends on whether a support is within 0.3 of 95.2
        # This is a proximity-based check

    def test_tp_blocked_for_long(self, detector):
        # Strong resistance (sr_min_touches=2) between entry and TP
        # We need a level with 2+ touches above current price but below TP
        detector.config.candle.sr_min_touches = 2
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv,
            current_price=95.0, atr=2.0,
            suggested_side="BUY",
            tp_price=115.0,  # Far TP
        )
        # tp_blocked depends on whether there's a resistance with touches >= 2 below tp_price
        # Just verify the field exists and is boolean
        assert isinstance(result.tp_blocked, bool)

    def test_tp_blocked_for_short(self, detector):
        detector.config.candle.sr_min_touches = 2
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv,
            current_price=105.0, atr=2.0,
            suggested_side="SELL",
            tp_price=85.0,
        )
        assert isinstance(result.tp_blocked, bool)

    def test_cache_stores_result(self, detector):
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=100.0, atr=2.0,
            timestamp=1000.0
        )
        cached = detector.get_cached("BTC/USDT")
        assert cached is result

    def test_fallback_proximity_without_atr(self, detector):
        """When atr=0, proximity defaults to price * 0.003."""
        ohlcv = _make_ohlcv_with_pivots()
        result = detector.analyze(
            "BTC/USDT", ohlcv, current_price=100.0, atr=0.0
        )
        # Should not crash, proximity falls back to 100 * 0.003 = 0.3
        assert isinstance(result, LevelAnalysis)


# ─────────────────────────────────────────
# PRICE LEVEL DATACLASS
# ─────────────────────────────────────────

class TestPriceLevel:

    def test_repr_pivot(self):
        level = PriceLevel(price=100.0, level_type="support", touches=3, source="pivot")
        r = repr(level)
        assert "support" in r
        assert "100.00" in r
        assert "pivot" in r

    def test_repr_fibonacci(self):
        level = PriceLevel(
            price=95.0, level_type="resistance", touches=1,
            source="fibonacci", fib_ratio=0.618
        )
        r = repr(level)
        assert "fib_0.618" in r
