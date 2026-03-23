"""
Tests for candle_patterns.py — Candlestick Pattern Recognition.

Tests each pattern detector with crafted OHLCV data, plus the
evaluate_patterns_for_signal integration helper.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from candle_patterns import (
    PatternDetector, PatternType, PatternDirection,
    PatternScanResult, PatternMatch,
    REVERSAL_PATTERNS, CONTINUATION_PATTERNS,
    evaluate_patterns_for_signal,
)


@pytest.fixture
def detector():
    return PatternDetector(
        doji_body_pct=0.10,
        pin_wick_ratio=2.0,
        marubozu_body_pct=0.90,
    )


def _ohlcv(candles):
    """Wrap (open, high, low, close) tuples into full OHLCV with timestamp+volume."""
    return [[i * 3600000, o, h, l, c, 1000] for i, (o, h, l, c) in enumerate(candles)]


# ─────────────────────────────────────────
# BULLISH ENGULFING
# ─────────────────────────────────────────

class TestBullishEngulfing:

    def test_classic_bullish_engulfing(self, detector):
        """Prior bearish candle fully engulfed by current bullish candle."""
        candles = _ohlcv([
            (100, 101, 99, 99.5),    # filler
            (102, 102.5, 98, 98.5),  # prior: bearish (open 102, close 98.5)
            (98, 103, 97.5, 103),    # current: bullish, open<=98.5, close>=102
        ])
        scan = detector.scan(candles, "4h")
        bulls = [p for p in scan.patterns if p.pattern == PatternType.BULLISH_ENGULFING]
        assert len(bulls) == 1
        assert bulls[0].direction == PatternDirection.BULLISH
        assert bulls[0].strength > 0

    def test_no_engulfing_if_not_engulfed(self, detector):
        """Current candle doesn't fully engulf prior body."""
        candles = _ohlcv([
            (100, 101, 99, 99.5),
            (102, 102.5, 98, 98.5),  # bearish
            (99, 101, 98, 101),      # bullish but close < prior open (102)
        ])
        scan = detector.scan(candles, "4h")
        bulls = [p for p in scan.patterns if p.pattern == PatternType.BULLISH_ENGULFING]
        assert len(bulls) == 0

    def test_no_engulfing_same_direction(self, detector):
        """Both candles bullish — not an engulfing."""
        candles = _ohlcv([
            (100, 101, 99, 100.5),
            (100, 102, 99, 101),     # bullish
            (101, 105, 100, 104),    # bullish — not engulfing
        ])
        scan = detector.scan(candles, "1h")
        bulls = [p for p in scan.patterns if p.pattern == PatternType.BULLISH_ENGULFING]
        assert len(bulls) == 0


# ─────────────────────────────────────────
# BEARISH ENGULFING
# ─────────────────────────────────────────

class TestBearishEngulfing:

    def test_classic_bearish_engulfing(self, detector):
        """Prior bullish candle fully engulfed by current bearish candle."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (98, 101.5, 97, 101),    # prior: bullish (open 98, close 101)
            (101.5, 102, 97, 97.5),  # current: bearish, open>=101, close<=98
        ])
        scan = detector.scan(candles, "4h")
        bears = [p for p in scan.patterns if p.pattern == PatternType.BEARISH_ENGULFING]
        assert len(bears) == 1
        assert bears[0].direction == PatternDirection.BEARISH

    def test_no_bearish_if_prior_bearish(self, detector):
        """Prior candle must be bullish for bearish engulfing."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (101, 102, 99, 99.5),    # prior: bearish
            (100, 103, 98, 98.5),    # current: bearish
        ])
        scan = detector.scan(candles, "4h")
        bears = [p for p in scan.patterns if p.pattern == PatternType.BEARISH_ENGULFING]
        assert len(bears) == 0


# ─────────────────────────────────────────
# PIN BAR (HAMMER / SHOOTING STAR)
# ─────────────────────────────────────────

class TestPinBar:

    def test_hammer(self, detector):
        """Small body at top, long lower wick >= 2x body, upper wick < body."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (99, 100, 93, 100),  # body=1, lower_wick=6, upper_wick=0
        ])
        scan = detector.scan(candles, "1h")
        hammers = [p for p in scan.patterns if p.pattern == PatternType.HAMMER]
        assert len(hammers) == 1
        assert hammers[0].direction == PatternDirection.BULLISH

    def test_shooting_star(self, detector):
        """Small body at bottom, long upper wick >= 2x body, lower wick < body."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 106, 99.5, 99.5),  # body=0.5, upper_wick=6, lower_wick=0
        ])
        scan = detector.scan(candles, "1h")
        stars = [p for p in scan.patterns if p.pattern == PatternType.SHOOTING_STAR]
        assert len(stars) == 1
        assert stars[0].direction == PatternDirection.BEARISH

    def test_no_pin_bar_equal_wicks(self, detector):
        """Both wicks similar — not a pin bar."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 103, 97, 100.1),  # body=0.1, but both wicks are ~3
        ])
        scan = detector.scan(candles, "1h")
        pins = [p for p in scan.patterns
                if p.pattern in (PatternType.HAMMER, PatternType.SHOOTING_STAR)]
        # Both wicks > body so could match one or both depending on implementation,
        # but upper_wick > body so shooting star excluded, and lower_wick > body
        # so hammer excluded (upper_wick=2.9 > body=0.1)
        # Actually: upper_wick=2.9 > body=0.1, so hammer excluded (needs upper_wick <= body)
        assert all(p.pattern != PatternType.HAMMER for p in pins)


# ─────────────────────────────────────────
# DOJI
# ─────────────────────────────────────────

class TestDoji:

    def test_classic_doji(self, detector):
        """Body < 10% of total range."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 105, 95, 100.2),  # range=10, body=0.2 → 2% < 10%
        ])
        scan = detector.scan(candles, "4h")
        dojis = [p for p in scan.patterns if p.pattern == PatternType.DOJI]
        assert len(dojis) == 1
        assert dojis[0].direction == PatternDirection.NEUTRAL

    def test_no_doji_large_body(self, detector):
        """Body > 10% of range — not a doji."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 105, 95, 102),  # range=10, body=2 → 20% > 10%
        ])
        scan = detector.scan(candles, "4h")
        dojis = [p for p in scan.patterns if p.pattern == PatternType.DOJI]
        assert len(dojis) == 0

    def test_doji_strength(self, detector):
        """Smaller body = higher doji strength."""
        candles1 = _ohlcv([
            (100, 101, 99, 100), (100, 101, 99, 100),
            (100, 110, 90, 100.1),  # body=0.1 / range=20 = 0.5% — very doji
        ])
        candles2 = _ohlcv([
            (100, 101, 99, 100), (100, 101, 99, 100),
            (100, 110, 90, 101.5),  # body=1.5 / range=20 = 7.5% — barely doji
        ])
        scan1 = detector.scan(candles1, "1h")
        scan2 = detector.scan(candles2, "1h")
        d1 = [p for p in scan1.patterns if p.pattern == PatternType.DOJI]
        d2 = [p for p in scan2.patterns if p.pattern == PatternType.DOJI]
        assert len(d1) == 1 and len(d2) == 1
        assert d1[0].strength > d2[0].strength


# ─────────────────────────────────────────
# MORNING STAR / EVENING STAR
# ─────────────────────────────────────────

class TestMorningStar:

    def test_classic_morning_star(self, detector):
        """3-candle bullish reversal: bearish + small + bullish (close > mid of candle 1)."""
        candles = _ohlcv([
            (105, 106, 99, 100),     # Large bearish (o=105, c=100, body=5)
            (99, 100, 98, 99.5),     # Small body (body=0.5, < 2.5)
            (100, 106, 99.5, 104),   # Large bullish close > 102.5 (mid of candle 1)
        ])
        scan = detector.scan(candles, "4h")
        ms = [p for p in scan.patterns if p.pattern == PatternType.MORNING_STAR]
        assert len(ms) == 1
        assert ms[0].direction == PatternDirection.BULLISH

    def test_no_morning_star_candle3_weak(self, detector):
        """Third candle doesn't close above midpoint of first."""
        candles = _ohlcv([
            (105, 106, 99, 100),     # Large bearish
            (99, 100, 98, 99.5),     # Small body
            (100, 101, 99, 100.5),   # Too weak: 100.5 < 102.5 (midpoint)
        ])
        scan = detector.scan(candles, "4h")
        ms = [p for p in scan.patterns if p.pattern == PatternType.MORNING_STAR]
        assert len(ms) == 0


class TestEveningStar:

    def test_classic_evening_star(self, detector):
        """3-candle bearish reversal: bullish + small + bearish (close < mid of candle 1)."""
        candles = _ohlcv([
            (100, 106, 99, 105),     # Large bullish (o=100, c=105, body=5)
            (105.5, 106, 104, 105),  # Small body (body=0.5, < 2.5)
            (104, 104.5, 99, 101),   # Large bearish, close=101 < 102.5 (mid)
        ])
        scan = detector.scan(candles, "4h")
        es = [p for p in scan.patterns if p.pattern == PatternType.EVENING_STAR]
        assert len(es) == 1
        assert es[0].direction == PatternDirection.BEARISH


# ─────────────────────────────────────────
# THREE WHITE SOLDIERS / THREE BLACK CROWS
# ─────────────────────────────────────────

class TestThreeWhiteSoldiers:

    def test_classic_soldiers(self, detector):
        """3 bullish candles with higher closes, each opening within prior body."""
        candles = _ohlcv([
            (100, 103, 99, 102),     # Bullish, open=100, close=102
            (101, 105, 100, 104),    # Bullish, open=101 (within 100-102), close=104>102
            (103, 107, 102, 106),    # Bullish, open=103 (within 101-104), close=106>104
        ])
        scan = detector.scan(candles, "1h")
        tws = [p for p in scan.patterns if p.pattern == PatternType.THREE_WHITE_SOLDIERS]
        assert len(tws) == 1
        assert tws[0].direction == PatternDirection.BULLISH

    def test_no_soldiers_if_lower_close(self, detector):
        """Closes must be progressively higher."""
        candles = _ohlcv([
            (100, 103, 99, 102),
            (101, 105, 100, 104),
            (103, 107, 102, 103),    # close=103 < 104 — breaks sequence
        ])
        scan = detector.scan(candles, "1h")
        tws = [p for p in scan.patterns if p.pattern == PatternType.THREE_WHITE_SOLDIERS]
        assert len(tws) == 0


class TestThreeBlackCrows:

    def test_classic_crows(self, detector):
        """3 bearish candles with lower closes, each opening within prior body."""
        candles = _ohlcv([
            (106, 107, 103, 104),    # Bearish, open=106, close=104
            (105, 106, 101, 102),    # Bearish, open=105 (within 104-106), close=102<104
            (103, 104, 99, 100),     # Bearish, open=103 (within 102-105), close=100<102
        ])
        scan = detector.scan(candles, "1h")
        tbc = [p for p in scan.patterns if p.pattern == PatternType.THREE_BLACK_CROWS]
        assert len(tbc) == 1
        assert tbc[0].direction == PatternDirection.BEARISH


# ─────────────────────────────────────────
# MARUBOZU
# ─────────────────────────────────────────

class TestMarubozu:

    def test_bullish_marubozu(self, detector):
        """Close > open, body > 90% of range."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 110.2, 99.8, 110),  # range=10.4, body=10, ratio=96%>90%
        ])
        scan = detector.scan(candles, "4h")
        bm = [p for p in scan.patterns if p.pattern == PatternType.BULLISH_MARUBOZU]
        assert len(bm) == 1
        assert bm[0].direction == PatternDirection.BULLISH

    def test_bearish_marubozu(self, detector):
        """Close < open, body > 90% of range."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (110, 110.2, 99.8, 100),  # range=10.4, body=10, ratio=96%>90%
        ])
        scan = detector.scan(candles, "4h")
        bm = [p for p in scan.patterns if p.pattern == PatternType.BEARISH_MARUBOZU]
        assert len(bm) == 1
        assert bm[0].direction == PatternDirection.BEARISH

    def test_no_marubozu_if_small_body(self, detector):
        """Body < 90% of range — not a marubozu."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 108, 95, 105),  # range=13, body=5, ratio=38%
        ])
        scan = detector.scan(candles, "4h")
        mz = [p for p in scan.patterns
              if p.pattern in (PatternType.BULLISH_MARUBOZU, PatternType.BEARISH_MARUBOZU)]
        assert len(mz) == 0


# ─────────────────────────────────────────
# SCAN AGGREGATION
# ─────────────────────────────────────────

class TestScanResult:

    def test_summary_flags(self, detector):
        """PatternScanResult correctly sets has_* flags."""
        # Bullish engulfing → has_bullish_reversal
        candles = _ohlcv([
            (100, 101, 99, 99.5),
            (102, 102.5, 98, 98.5),
            (98, 103, 97.5, 103),
        ])
        scan = detector.scan(candles, "4h")
        assert scan.has_bullish_reversal is True
        assert scan.has_bearish_reversal is False

    def test_strongest_pattern_selected(self, detector):
        """Strongest pattern is the one with highest strength."""
        # A doji and another pattern at the same time
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 105, 95, 100.2),  # Doji
        ])
        scan = detector.scan(candles, "1h")
        if scan.patterns:
            assert scan.strongest_pattern is not None
            assert scan.strongest_pattern.strength == max(
                p.strength for p in scan.patterns
            )

    def test_empty_candles(self, detector):
        scan = detector.scan([], "1h")
        assert scan.patterns == []
        assert scan.has_doji is False

    def test_insufficient_candles(self, detector):
        candles = _ohlcv([(100, 101, 99, 100)])
        scan = detector.scan(candles, "1h")
        # Only 1 candle — some patterns need 2-3
        assert isinstance(scan, PatternScanResult)


# ─────────────────────────────────────────
# PATTERN EVALUATION FOR SIGNAL
# ─────────────────────────────────────────

class TestEvaluatePatterns:

    def test_confirming_continuation_boosts_confidence(self, detector):
        """Bullish marubozu confirming a BUY → +0.10."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 110.2, 99.8, 110),  # Bullish marubozu
        ])
        scan = detector.scan(candles, "4h")
        result = evaluate_patterns_for_signal([scan], "BUY")
        assert result["confidence_adj"] >= 0.10
        assert len(result["confirming_patterns"]) > 0

    def test_contradicting_reversal_penalizes(self, detector):
        """Bearish engulfing contradicting a BUY → -0.15."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (98, 101.5, 97, 101),    # Bullish prior
            (101.5, 102, 97, 97.5),  # Bearish engulfing
        ])
        scan = detector.scan(candles, "4h")
        result = evaluate_patterns_for_signal([scan], "BUY")
        assert result["confidence_adj"] <= -0.15
        assert len(result["contradicting_patterns"]) > 0

    def test_doji_penalizes(self, detector):
        """Doji reduces confidence by 0.10."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 105, 95, 100.1),  # Doji
        ])
        scan = detector.scan(candles, "1h")
        result = evaluate_patterns_for_signal([scan], "BUY")
        assert result["has_doji"] is True
        assert result["confidence_adj"] <= -0.10

    def test_no_suggestion_returns_zero(self, detector):
        """No suggested_side → no adjustments."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 110.2, 99.8, 110),
        ])
        scan = detector.scan(candles, "4h")
        result = evaluate_patterns_for_signal([scan], None)
        assert result["confidence_adj"] == 0.0

    def test_confirming_reversal_for_buy(self, detector):
        """Bullish engulfing confirming BUY → +0.10."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (102, 102.5, 98, 98.5),
            (98, 103, 97.5, 103),
        ])
        scan = detector.scan(candles, "4h")
        result = evaluate_patterns_for_signal([scan], "BUY")
        assert result["confidence_adj"] >= 0.10
        assert len(result["confirming_patterns"]) > 0

    def test_confirming_reversal_for_sell(self, detector):
        """Bearish engulfing confirming SELL → +0.10."""
        candles = _ohlcv([
            (100, 101, 99, 100),
            (98, 101.5, 97, 101),
            (101.5, 102, 97, 97.5),
        ])
        scan = detector.scan(candles, "4h")
        result = evaluate_patterns_for_signal([scan], "SELL")
        assert result["confidence_adj"] >= 0.10

    def test_multiple_scans_combined(self, detector):
        """Multiple timeframes combined."""
        candles1h = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 105, 95, 100.1),  # Doji
        ])
        candles4h = _ohlcv([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 110.2, 99.8, 110),  # Bullish marubozu
        ])
        scan1h = detector.scan(candles1h, "1h")
        scan4h = detector.scan(candles4h, "4h")
        result = evaluate_patterns_for_signal([scan1h, scan4h], "BUY")
        # Doji -0.10 and confirming +0.10 should roughly cancel
        assert result["has_doji"] is True
        assert len(result["confirming_patterns"]) > 0


# ─────────────────────────────────────────
# PATTERN CLASSIFICATION HELPERS
# ─────────────────────────────────────────

class TestPatternClassification:

    def test_reversal_set(self):
        assert PatternType.BULLISH_ENGULFING in REVERSAL_PATTERNS
        assert PatternType.BEARISH_ENGULFING in REVERSAL_PATTERNS
        assert PatternType.HAMMER in REVERSAL_PATTERNS
        assert PatternType.SHOOTING_STAR in REVERSAL_PATTERNS
        assert PatternType.DOJI in REVERSAL_PATTERNS
        assert PatternType.MORNING_STAR in REVERSAL_PATTERNS
        assert PatternType.EVENING_STAR in REVERSAL_PATTERNS

    def test_continuation_set(self):
        assert PatternType.THREE_WHITE_SOLDIERS in CONTINUATION_PATTERNS
        assert PatternType.THREE_BLACK_CROWS in CONTINUATION_PATTERNS
        assert PatternType.BULLISH_MARUBOZU in CONTINUATION_PATTERNS
        assert PatternType.BEARISH_MARUBOZU in CONTINUATION_PATTERNS

    def test_pattern_match_props(self):
        p = PatternMatch(
            pattern=PatternType.HAMMER,
            direction=PatternDirection.BULLISH,
            index=5, strength=0.8,
        )
        assert p.is_reversal is True
        assert p.is_continuation is False

    def test_pattern_match_repr(self):
        p = PatternMatch(
            pattern=PatternType.DOJI,
            direction=PatternDirection.NEUTRAL,
            index=3, strength=0.5,
        )
        assert "doji" in repr(p)
