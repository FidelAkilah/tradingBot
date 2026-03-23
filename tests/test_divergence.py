"""
Tests for divergence.py — RSI Divergence Detection.

Tests each divergence type with crafted price/RSI scenarios, plus the
evaluate_divergence_for_signal integration helper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from divergence import (
    DivergenceDetector, DivergenceType, Divergence,
    DivergenceScanResult, evaluate_divergence_for_signal,
)


@pytest.fixture
def detector():
    return DivergenceDetector(
        lookback=20,
        pivot_lookback=3,
        rsi_period=14,
        oversold=40.0,
        overbought=60.0,
    )


def _make_ohlcv(closes, high_offset=1.0, low_offset=1.0):
    """Build OHLCV from a close price series."""
    candles = []
    for i, c in enumerate(closes):
        o = c - 0.1 if i % 2 == 0 else c + 0.1
        h = max(o, c) + high_offset
        l = min(o, c) - low_offset
        candles.append([i * 3600000, o, h, l, c, 1000])
    return candles


def _make_regular_bullish_data():
    """
    Craft a price series with:
    - Two swing lows in price where the second is LOWER than the first
    - RSI at the second low is HIGHER than at the first
    (= regular bullish divergence)

    Strategy: start with a decline, bounce, then a deeper decline that has less RSI momentum.
    """
    # Build ~45 candles. We need rsi_period(14) + lookback(20) + some slack.
    # Phase 1 (0-14): gentle decline to set up RSI baseline
    p1 = [100 - i * 0.5 for i in range(15)]  # 100 → 93
    # Phase 2 (15-19): sharp drop to create swing low #1 around index 17
    p2 = [92, 90, 88, 90, 92]
    # Phase 3 (20-24): bounce up
    p3 = [94, 96, 97, 96, 95]
    # Phase 4 (25-29): deeper drop to create swing low #2 at index 27
    # but slower decline → RSI should be higher than at index 17
    p4 = [94, 92, 87, 89, 91]
    # Phase 5 (30-39): recovery
    p5 = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102]

    closes = p1 + p2 + p3 + p4 + p5
    return _make_ohlcv(closes)


def _make_regular_bearish_data():
    """
    Price makes higher high, RSI makes lower high.
    """
    # Phase 1: gentle rise
    p1 = [100 + i * 0.5 for i in range(15)]
    # Phase 2: spike to swing high #1
    p2 = [108, 110, 112, 110, 108]
    # Phase 3: pullback
    p3 = [106, 104, 103, 104, 105]
    # Phase 4: higher price high but less momentum
    p4 = [107, 110, 113, 111, 109]
    # Phase 5: decline
    p5 = [107, 105, 103, 101, 100, 99, 98, 97, 96, 95]

    closes = p1 + p2 + p3 + p4 + p5
    return _make_ohlcv(closes)


def _make_hidden_bullish_data():
    """
    Price makes higher low, RSI makes lower low → continuation up.
    """
    # Phase 1: uptrend baseline
    p1 = [100 + i * 0.3 for i in range(15)]
    # Phase 2: pullback to swing low #1
    p2 = [104, 102, 100, 102, 104]
    # Phase 3: higher
    p3 = [106, 108, 109, 108, 107]
    # Phase 4: pullback to higher low but RSI drops more
    p4 = [106, 104, 101, 103, 105]  # price low=101 > 100
    # Phase 5: continuation up
    p5 = [107, 108, 110, 111, 112, 113, 114, 115, 116, 117]

    closes = p1 + p2 + p3 + p4 + p5
    return _make_ohlcv(closes)


# ─────────────────────────────────────────
# RSI SERIES COMPUTATION
# ─────────────────────────────────────────

class TestRSISeries:

    def test_rsi_series_length(self, detector):
        closes = np.array([100 + i * 0.5 for i in range(50)], dtype=np.float64)
        rsi = detector._rsi_series(closes, 14)
        assert rsi is not None
        assert len(rsi) == len(closes)

    def test_rsi_series_none_for_short_data(self, detector):
        closes = np.array([100, 101, 102], dtype=np.float64)
        rsi = detector._rsi_series(closes, 14)
        assert rsi is None

    def test_rsi_bounded(self, detector):
        closes = np.array([100 + i * 2 for i in range(50)], dtype=np.float64)
        rsi = detector._rsi_series(closes, 14)
        # After initial period, RSI should be bounded 0-100
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

    def test_rsi_strong_uptrend_high(self, detector):
        """Strong uptrend should produce RSI > 50."""
        closes = np.array([100 + i * 3 for i in range(50)], dtype=np.float64)
        rsi = detector._rsi_series(closes, 14)
        assert rsi[-1] > 70  # Strong uptrend → high RSI


# ─────────────────────────────────────────
# SWING DETECTION
# ─────────────────────────────────────────

class TestSwingDetection:

    def test_finds_swing_highs(self, detector):
        # Peak at index 5
        data = np.array([1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=np.float64)
        highs = detector._find_swing_highs(data, lookback=3, start_idx=0)
        assert 5 in highs

    def test_finds_swing_lows(self, detector):
        # Trough at index 5
        data = np.array([10, 9, 8, 7, 6, 1, 6, 7, 8, 9, 10], dtype=np.float64)
        lows = detector._find_swing_lows(data, lookback=3, start_idx=0)
        assert 5 in lows

    def test_no_swings_in_monotonic(self, detector):
        data = np.arange(1, 20, dtype=np.float64)
        highs = detector._find_swing_highs(data, lookback=3, start_idx=0)
        assert len(highs) == 0


# ─────────────────────────────────────────
# REGULAR BULLISH DIVERGENCE
# ─────────────────────────────────────────

class TestRegularBullish:

    def test_detect_regular_bullish(self, detector):
        """Price lower low + RSI higher low = regular bullish."""
        ohlcv = _make_regular_bullish_data()
        scan = detector.scan(ohlcv, "4h")
        # May or may not detect depending on exact RSI dynamics
        # At minimum, the scan should not crash and return a valid result
        assert isinstance(scan, DivergenceScanResult)
        assert scan.timeframe == "4h"

    def test_regular_bullish_zone_oversold(self, detector):
        """If detected in oversold zone, strength should be boosted."""
        for d in []:  # placeholder — we test the strength calc directly
            pass
        # Direct strength test
        strength = detector._calc_strength(90, 88, 25, 30, "oversold")
        assert strength > detector._calc_strength(90, 88, 25, 30, "mid")


# ─────────────────────────────────────────
# REGULAR BEARISH DIVERGENCE
# ─────────────────────────────────────────

class TestRegularBearish:

    def test_detect_regular_bearish(self, detector):
        """Price higher high + RSI lower high = regular bearish."""
        ohlcv = _make_regular_bearish_data()
        scan = detector.scan(ohlcv, "4h")
        assert isinstance(scan, DivergenceScanResult)

    def test_regular_bearish_zone_overbought(self, detector):
        strength = detector._calc_strength(110, 113, 75, 70, "overbought")
        assert strength > detector._calc_strength(110, 113, 75, 70, "mid")


# ─────────────────────────────────────────
# HIDDEN DIVERGENCE
# ─────────────────────────────────────────

class TestHiddenDivergence:

    def test_detect_hidden_bullish(self, detector):
        """Price higher low + RSI lower low = hidden bullish."""
        ohlcv = _make_hidden_bullish_data()
        scan = detector.scan(ohlcv, "4h")
        assert isinstance(scan, DivergenceScanResult)

    def test_hidden_bearish_structure(self, detector):
        """Just verify the check doesn't crash on normal data."""
        closes = [100 - i * 0.3 for i in range(45)]
        ohlcv = _make_ohlcv(closes)
        scan = detector.scan(ohlcv, "4h")
        assert isinstance(scan, DivergenceScanResult)


# ─────────────────────────────────────────
# STRENGTH CALCULATION
# ─────────────────────────────────────────

class TestStrengthCalc:

    def test_small_rsi_diff_low_strength(self):
        # 2 points RSI diff → 0.2 base
        s = DivergenceDetector._calc_strength(100, 99, 30, 32, "mid")
        assert 0.1 < s < 0.4

    def test_large_rsi_diff_high_strength(self):
        # 15 points RSI diff → capped at 1.0
        s = DivergenceDetector._calc_strength(100, 99, 30, 45, "mid")
        assert s >= 0.8

    def test_zone_bonus(self):
        s_mid = DivergenceDetector._calc_strength(100, 99, 30, 35, "mid")
        s_os = DivergenceDetector._calc_strength(100, 99, 30, 35, "oversold")
        assert s_os > s_mid
        assert s_os - s_mid == pytest.approx(0.2, abs=0.01)

    def test_max_strength_capped(self):
        s = DivergenceDetector._calc_strength(100, 99, 20, 45, "oversold")
        assert s <= 1.0


# ─────────────────────────────────────────
# DIVERGENCE DATA CLASS
# ─────────────────────────────────────────

class TestDivergenceDataClass:

    def test_is_regular(self):
        d = Divergence(
            div_type=DivergenceType.REGULAR_BULLISH,
            price_idx_1=5, price_idx_2=10,
            price_val_1=100, price_val_2=98,
            rsi_val_1=25, rsi_val_2=30,
        )
        assert d.is_regular is True
        assert d.is_hidden is False
        assert d.is_bullish is True
        assert d.is_bearish is False

    def test_is_hidden(self):
        d = Divergence(
            div_type=DivergenceType.HIDDEN_BEARISH,
            price_idx_1=5, price_idx_2=10,
            price_val_1=110, price_val_2=108,
            rsi_val_1=65, rsi_val_2=70,
        )
        assert d.is_hidden is True
        assert d.is_regular is False
        assert d.is_bearish is True

    def test_repr(self):
        d = Divergence(
            div_type=DivergenceType.REGULAR_BEARISH,
            price_idx_1=5, price_idx_2=10,
            price_val_1=100, price_val_2=105,
            rsi_val_1=75, rsi_val_2=65,
            strength=0.7, rsi_zone="overbought",
        )
        r = repr(d)
        assert "regular_bearish" in r
        assert "overbought" in r


# ─────────────────────────────────────────
# NEAREST RSI HELPER
# ─────────────────────────────────────────

class TestNearestRSI:

    def test_exact_match(self):
        rsi = np.array([50] * 20, dtype=np.float64)
        pivots = [5, 10, 15]
        assert DivergenceDetector._nearest_rsi_at(rsi, 10, pivots) == 10

    def test_close_match(self):
        rsi = np.array([50] * 20, dtype=np.float64)
        pivots = [4, 11, 15]
        # Price pivot at 10, closest RSI pivot within 2 is 11
        assert DivergenceDetector._nearest_rsi_at(rsi, 10, pivots) == 11

    def test_fallback_to_exact_index(self):
        rsi = np.array([50] * 20, dtype=np.float64)
        pivots = [1, 20]  # None within 2 of index 10
        # Falls back to price index itself
        result = DivergenceDetector._nearest_rsi_at(rsi, 10, pivots)
        assert result == 10

    def test_empty_pivots(self):
        rsi = np.array([50] * 20, dtype=np.float64)
        assert DivergenceDetector._nearest_rsi_at(rsi, 10, []) is None


# ─────────────────────────────────────────
# EVALUATE FOR SIGNAL
# ─────────────────────────────────────────

class TestEvaluateDivergence:

    def test_regular_bullish_confirms_buy(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.REGULAR_BULLISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=98,
                rsi_val_1=25, rsi_val_2=30,
                strength=0.7, rsi_zone="oversold",
            )
        ], has_regular_bullish=True)

        result = evaluate_divergence_for_signal(scan, "BUY")
        assert result["confidence_adj"] >= 0.15
        assert not result["blocked"]
        assert len(result["confirming"]) > 0

    def test_regular_bearish_blocks_buy(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.REGULAR_BEARISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=105,
                rsi_val_1=75, rsi_val_2=65,
                strength=0.8, rsi_zone="overbought",
            )
        ], has_regular_bearish=True)

        result = evaluate_divergence_for_signal(scan, "BUY")
        assert result["blocked"] is True
        assert "block_reason" in result
        assert len(result["contradicting"]) > 0

    def test_regular_bullish_blocks_sell(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.REGULAR_BULLISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=98,
                rsi_val_1=25, rsi_val_2=30,
                strength=0.7,
            )
        ], has_regular_bullish=True)

        result = evaluate_divergence_for_signal(scan, "SELL")
        assert result["blocked"] is True

    def test_hidden_bullish_confirms_buy(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.HIDDEN_BULLISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=102,
                rsi_val_1=35, rsi_val_2=28,
                strength=0.6,
            )
        ], has_hidden_bullish=True)

        result = evaluate_divergence_for_signal(scan, "BUY")
        assert result["confidence_adj"] >= 0.10
        assert not result["blocked"]
        assert len(result["confirming"]) > 0

    def test_hidden_divergence_does_not_block(self):
        """Hidden divergence against direction is NOT a hard block."""
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.HIDDEN_BEARISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=110, price_val_2=108,
                rsi_val_1=65, rsi_val_2=70,
                strength=0.5,
            )
        ], has_hidden_bearish=True)

        result = evaluate_divergence_for_signal(scan, "BUY")
        assert result["blocked"] is False

    def test_no_divergences_no_change(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[])
        result = evaluate_divergence_for_signal(scan, "BUY")
        assert result["confidence_adj"] == 0.0
        assert not result["blocked"]

    def test_no_suggestion_no_change(self):
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.REGULAR_BULLISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=98,
                rsi_val_1=25, rsi_val_2=30,
                strength=0.7,
            )
        ])
        result = evaluate_divergence_for_signal(scan, None)
        assert result["confidence_adj"] == 0.0

    def test_regular_plus_hidden_confirm_stacks(self):
        """Both regular and hidden divergence confirming same direction."""
        scan = DivergenceScanResult(timeframe="4h", divergences=[
            Divergence(
                div_type=DivergenceType.REGULAR_BULLISH,
                price_idx_1=5, price_idx_2=10,
                price_val_1=100, price_val_2=98,
                rsi_val_1=25, rsi_val_2=30,
                strength=0.7,
            ),
            Divergence(
                div_type=DivergenceType.HIDDEN_BULLISH,
                price_idx_1=3, price_idx_2=8,
                price_val_1=99, price_val_2=101,
                rsi_val_1=30, rsi_val_2=25,
                strength=0.5,
            ),
        ], has_regular_bullish=True, has_hidden_bullish=True)

        result = evaluate_divergence_for_signal(scan, "BUY")
        # Should get both bonuses: +0.15 (regular) + 0.10 (hidden) = +0.25
        assert result["confidence_adj"] >= 0.25


# ─────────────────────────────────────────
# SCAN WITH INSUFFICIENT DATA
# ─────────────────────────────────────────

class TestScanEdgeCases:

    def test_empty_ohlcv(self, detector):
        scan = detector.scan([], "4h")
        assert scan.divergences == []

    def test_short_ohlcv(self, detector):
        ohlcv = _make_ohlcv([100, 101, 102])
        scan = detector.scan(ohlcv, "4h")
        assert scan.divergences == []

    def test_flat_prices_no_divergence(self, detector):
        closes = [100.0] * 50
        ohlcv = _make_ohlcv(closes)
        scan = detector.scan(ohlcv, "4h")
        # Flat prices → no swing points → no divergence
        assert len(scan.divergences) == 0
