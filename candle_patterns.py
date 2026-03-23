"""
Candlestick Pattern Recognition — NumPy-based pattern detection on OHLCV data.

Detects reversal and continuation patterns across timeframes:

Reversal patterns (filter bad entries):
  - Bullish/Bearish Engulfing
  - Pin Bar (Hammer / Shooting Star)
  - Doji (indecision)
  - Morning Star / Evening Star (3-candle reversal)

Continuation patterns (confirm entries):
  - Three White Soldiers / Three Black Crows
  - Bullish/Bearish Marubozu (>90% body, strong conviction)

Each detector returns a list of PatternMatch with direction, strength, and index.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"   # doji — indecision


class PatternType(Enum):
    # Reversal
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    # Continuation
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    BULLISH_MARUBOZU = "bullish_marubozu"
    BEARISH_MARUBOZU = "bearish_marubozu"


# Which patterns are reversal vs continuation
REVERSAL_PATTERNS = {
    PatternType.BULLISH_ENGULFING,
    PatternType.BEARISH_ENGULFING,
    PatternType.HAMMER,
    PatternType.SHOOTING_STAR,
    PatternType.DOJI,
    PatternType.MORNING_STAR,
    PatternType.EVENING_STAR,
}

CONTINUATION_PATTERNS = {
    PatternType.THREE_WHITE_SOLDIERS,
    PatternType.THREE_BLACK_CROWS,
    PatternType.BULLISH_MARUBOZU,
    PatternType.BEARISH_MARUBOZU,
}


@dataclass
class PatternMatch:
    """A detected candlestick pattern."""
    pattern: PatternType
    direction: PatternDirection
    index: int           # Candle index where pattern completes
    strength: float      # 0.0-1.0 — how textbook the pattern is
    timeframe: str = ""  # "1h", "4h", etc.

    @property
    def is_reversal(self) -> bool:
        return self.pattern in REVERSAL_PATTERNS

    @property
    def is_continuation(self) -> bool:
        return self.pattern in CONTINUATION_PATTERNS

    def __repr__(self):
        return (f"Pattern({self.pattern.value} {self.direction.value} "
                f"str={self.strength:.2f} idx={self.index})")


@dataclass
class PatternScanResult:
    """Aggregated pattern scan across a timeframe."""
    timeframe: str
    patterns: List[PatternMatch] = field(default_factory=list)

    # Pre-computed summary for the most recent candle
    has_bullish_reversal: bool = False
    has_bearish_reversal: bool = False
    has_bullish_continuation: bool = False
    has_bearish_continuation: bool = False
    has_doji: bool = False
    strongest_pattern: Optional[PatternMatch] = None

    def __repr__(self):
        return f"Scan({self.timeframe} patterns={len(self.patterns)})"


class PatternDetector:
    """
    Scans OHLCV data for candlestick patterns.

    All detection uses the last few candles only (indices -1, -2, -3)
    since we only care about actionable current patterns.
    """

    def __init__(self, doji_body_pct: float = 0.10, pin_wick_ratio: float = 2.0,
                 marubozu_body_pct: float = 0.90):
        self.doji_body_pct = doji_body_pct
        self.pin_wick_ratio = pin_wick_ratio
        self.marubozu_body_pct = marubozu_body_pct

    def scan(self, ohlcv: list, timeframe: str = "") -> PatternScanResult:
        """
        Scan OHLCV data for all patterns on the most recent candles.

        Args:
            ohlcv: List of [timestamp, open, high, low, close, volume]
            timeframe: Label for logging ("1h", "4h")

        Returns:
            PatternScanResult with all detected patterns
        """
        result = PatternScanResult(timeframe=timeframe)

        if not ohlcv or len(ohlcv) < 3:
            return result

        opens = np.array([c[1] for c in ohlcv], dtype=np.float64)
        highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv], dtype=np.float64)
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)

        n = len(opens)
        all_patterns = []

        # Check patterns on the last candle (index n-1)
        all_patterns.extend(self._detect_engulfing(opens, highs, lows, closes, n))
        all_patterns.extend(self._detect_pin_bar(opens, highs, lows, closes, n))
        all_patterns.extend(self._detect_doji(opens, highs, lows, closes, n))
        all_patterns.extend(self._detect_morning_evening_star(opens, highs, lows, closes, n))
        all_patterns.extend(self._detect_three_soldiers_crows(opens, highs, lows, closes, n))
        all_patterns.extend(self._detect_marubozu(opens, highs, lows, closes, n))

        # Tag timeframe
        for p in all_patterns:
            p.timeframe = timeframe

        result.patterns = all_patterns

        # Summarize for the signal logic
        for p in all_patterns:
            if p.pattern == PatternType.DOJI:
                result.has_doji = True
            elif p.is_reversal:
                if p.direction == PatternDirection.BULLISH:
                    result.has_bullish_reversal = True
                elif p.direction == PatternDirection.BEARISH:
                    result.has_bearish_reversal = True
            elif p.is_continuation:
                if p.direction == PatternDirection.BULLISH:
                    result.has_bullish_continuation = True
                elif p.direction == PatternDirection.BEARISH:
                    result.has_bearish_continuation = True

        if all_patterns:
            result.strongest_pattern = max(all_patterns, key=lambda p: p.strength)

        return result

    # ─────────────────────────────────────────
    # REVERSAL PATTERNS
    # ─────────────────────────────────────────

    def _detect_engulfing(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Bullish Engulfing: prior candle is bearish, current candle's body
        fully engulfs (open < prior close AND close > prior open).
        Bearish Engulfing: mirror image.
        """
        if n < 2:
            return []

        patterns = []
        i = n - 1  # Current (last) candle

        curr_open, curr_close = opens[i], closes[i]
        prev_open, prev_close = opens[i - 1], closes[i - 1]

        curr_body = abs(curr_close - curr_open)
        prev_body = abs(prev_close - prev_open)
        curr_range = highs[i] - lows[i]

        # Avoid noise on tiny candles
        if curr_range <= 0 or prev_body <= 0:
            return []

        # Bullish engulfing: prev bearish, current bullish, body engulfs
        if (prev_close < prev_open and curr_close > curr_open and
                curr_open <= prev_close and curr_close >= prev_open):
            # Strength: how much bigger current body is vs previous
            strength = min(curr_body / prev_body, 3.0) / 3.0
            patterns.append(PatternMatch(
                pattern=PatternType.BULLISH_ENGULFING,
                direction=PatternDirection.BULLISH,
                index=i, strength=strength,
            ))

        # Bearish engulfing: prev bullish, current bearish, body engulfs
        if (prev_close > prev_open and curr_close < curr_open and
                curr_open >= prev_close and curr_close <= prev_open):
            strength = min(curr_body / prev_body, 3.0) / 3.0
            patterns.append(PatternMatch(
                pattern=PatternType.BEARISH_ENGULFING,
                direction=PatternDirection.BEARISH,
                index=i, strength=strength,
            ))

        return patterns

    def _detect_pin_bar(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Hammer (bullish pin bar): small body at top, long lower wick (>2x body).
        Shooting Star (bearish pin bar): small body at bottom, long upper wick.
        """
        if n < 1:
            return []

        patterns = []
        i = n - 1

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        total_range = h - l
        if total_range <= 0:
            return []

        body = abs(c - o)
        body_top = max(c, o)
        body_bottom = min(c, o)
        upper_wick = h - body_top
        lower_wick = body_bottom - l

        # Prevent division by zero for doji-like candles
        if body < total_range * 0.01:
            return []

        # Hammer: lower wick > 2x body, upper wick < body
        if (lower_wick >= self.pin_wick_ratio * body and
                upper_wick <= body):
            strength = min(lower_wick / body / self.pin_wick_ratio, 2.0) / 2.0
            patterns.append(PatternMatch(
                pattern=PatternType.HAMMER,
                direction=PatternDirection.BULLISH,
                index=i, strength=strength,
            ))

        # Shooting Star: upper wick > 2x body, lower wick < body
        if (upper_wick >= self.pin_wick_ratio * body and
                lower_wick <= body):
            strength = min(upper_wick / body / self.pin_wick_ratio, 2.0) / 2.0
            patterns.append(PatternMatch(
                pattern=PatternType.SHOOTING_STAR,
                direction=PatternDirection.BEARISH,
                index=i, strength=strength,
            ))

        return patterns

    def _detect_doji(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Doji: body < 10% of total range (indecision).
        """
        if n < 1:
            return []

        i = n - 1
        total_range = highs[i] - lows[i]
        if total_range <= 0:
            return []

        body = abs(closes[i] - opens[i])
        body_ratio = body / total_range

        if body_ratio <= self.doji_body_pct:
            # Strength inversely proportional to body size
            strength = 1.0 - (body_ratio / self.doji_body_pct)
            return [PatternMatch(
                pattern=PatternType.DOJI,
                direction=PatternDirection.NEUTRAL,
                index=i, strength=strength,
            )]

        return []

    def _detect_morning_evening_star(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Morning Star (bullish): 3-candle reversal.
          1. Large bearish candle
          2. Small body (gap down preferred, but not required on futures)
          3. Large bullish candle closing above midpoint of candle 1

        Evening Star (bearish): mirror image.
        """
        if n < 3:
            return []

        patterns = []
        i = n - 1  # Third candle (current)

        o1, c1 = opens[i - 2], closes[i - 2]
        o2, c2 = opens[i - 1], closes[i - 1]
        o3, c3 = opens[i], closes[i]

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        range1 = highs[i - 2] - lows[i - 2]

        if range1 <= 0 or body1 <= 0:
            return []

        mid1 = (o1 + c1) / 2.0

        # Morning Star
        if (c1 < o1 and          # Candle 1: bearish
                body2 < body1 * 0.5 and  # Candle 2: small body (< half of candle 1)
                c3 > o3 and          # Candle 3: bullish
                c3 > mid1):          # Candle 3 closes above midpoint of candle 1
            strength = min(body3 / body1, 1.5) / 1.5
            patterns.append(PatternMatch(
                pattern=PatternType.MORNING_STAR,
                direction=PatternDirection.BULLISH,
                index=i, strength=strength,
            ))

        # Evening Star
        if (c1 > o1 and          # Candle 1: bullish
                body2 < body1 * 0.5 and  # Candle 2: small body
                c3 < o3 and          # Candle 3: bearish
                c3 < mid1):          # Candle 3 closes below midpoint of candle 1
            strength = min(body3 / body1, 1.5) / 1.5
            patterns.append(PatternMatch(
                pattern=PatternType.EVENING_STAR,
                direction=PatternDirection.BEARISH,
                index=i, strength=strength,
            ))

        return patterns

    # ─────────────────────────────────────────
    # CONTINUATION PATTERNS
    # ─────────────────────────────────────────

    def _detect_three_soldiers_crows(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Three White Soldiers: 3 consecutive bullish candles with higher closes,
        each opening within prior body.

        Three Black Crows: 3 consecutive bearish candles with lower closes,
        each opening within prior body.
        """
        if n < 3:
            return []

        patterns = []
        i = n - 1

        o1, c1 = opens[i - 2], closes[i - 2]
        o2, c2 = opens[i - 1], closes[i - 1]
        o3, c3 = opens[i], closes[i]

        # Three White Soldiers
        if (c1 > o1 and c2 > o2 and c3 > o3 and  # All bullish
                c2 > c1 and c3 > c2 and                 # Higher closes
                o2 >= o1 and o2 <= c1 and               # Open within prior body
                o3 >= o2 and o3 <= c2):
            avg_body = (abs(c1 - o1) + abs(c2 - o2) + abs(c3 - o3)) / 3.0
            avg_range = ((highs[i-2] - lows[i-2]) + (highs[i-1] - lows[i-1]) +
                         (highs[i] - lows[i])) / 3.0
            strength = min(avg_body / avg_range, 1.0) if avg_range > 0 else 0.5
            patterns.append(PatternMatch(
                pattern=PatternType.THREE_WHITE_SOLDIERS,
                direction=PatternDirection.BULLISH,
                index=i, strength=strength,
            ))

        # Three Black Crows
        if (c1 < o1 and c2 < o2 and c3 < o3 and  # All bearish
                c2 < c1 and c3 < c2 and                 # Lower closes
                o2 <= o1 and o2 >= c1 and               # Open within prior body
                o3 <= o2 and o3 >= c2):
            avg_body = (abs(c1 - o1) + abs(c2 - o2) + abs(c3 - o3)) / 3.0
            avg_range = ((highs[i-2] - lows[i-2]) + (highs[i-1] - lows[i-1]) +
                         (highs[i] - lows[i])) / 3.0
            strength = min(avg_body / avg_range, 1.0) if avg_range > 0 else 0.5
            patterns.append(PatternMatch(
                pattern=PatternType.THREE_BLACK_CROWS,
                direction=PatternDirection.BEARISH,
                index=i, strength=strength,
            ))

        return patterns

    def _detect_marubozu(
        self, opens: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, closes: np.ndarray, n: int
    ) -> List[PatternMatch]:
        """
        Marubozu: body > 90% of total range (strong conviction candle).
        Bullish: close > open, body fills almost the entire range.
        Bearish: close < open, same.
        """
        if n < 1:
            return []

        patterns = []
        i = n - 1

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        total_range = h - l
        if total_range <= 0:
            return []

        body = abs(c - o)
        body_ratio = body / total_range

        if body_ratio >= self.marubozu_body_pct:
            strength = min((body_ratio - self.marubozu_body_pct) /
                           (1.0 - self.marubozu_body_pct), 1.0)
            if c > o:
                patterns.append(PatternMatch(
                    pattern=PatternType.BULLISH_MARUBOZU,
                    direction=PatternDirection.BULLISH,
                    index=i, strength=strength,
                ))
            elif c < o:
                patterns.append(PatternMatch(
                    pattern=PatternType.BEARISH_MARUBOZU,
                    direction=PatternDirection.BEARISH,
                    index=i, strength=strength,
                ))

        return patterns


def evaluate_patterns_for_signal(
    scan_results: List[PatternScanResult],
    suggested_side: Optional[str],
) -> dict:
    """
    Evaluate scanned patterns against a proposed trade direction.

    Returns dict with:
      - confidence_adj: float to add/subtract from confidence
      - blocked: bool — if a strong reversal contradicts the trade
      - block_reason: str
      - confirming_patterns: list of pattern names
      - contradicting_patterns: list of pattern names
    """
    result = {
        "confidence_adj": 0.0,
        "blocked": False,
        "block_reason": "",
        "confirming_patterns": [],
        "contradicting_patterns": [],
        "has_doji": False,
    }

    if not suggested_side:
        return result

    for scan in scan_results:
        for p in scan.patterns:
            # Doji on signal candle — reduce confidence
            if p.pattern == PatternType.DOJI:
                result["has_doji"] = True
                continue

            # Reversal patterns
            if p.is_reversal:
                if suggested_side == "BUY" and p.direction == PatternDirection.BEARISH:
                    result["contradicting_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "SELL" and p.direction == PatternDirection.BULLISH:
                    result["contradicting_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "BUY" and p.direction == PatternDirection.BULLISH:
                    result["confirming_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "SELL" and p.direction == PatternDirection.BEARISH:
                    result["confirming_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )

            # Continuation patterns
            elif p.is_continuation:
                if suggested_side == "BUY" and p.direction == PatternDirection.BULLISH:
                    result["confirming_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "SELL" and p.direction == PatternDirection.BEARISH:
                    result["confirming_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "BUY" and p.direction == PatternDirection.BEARISH:
                    result["contradicting_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )
                elif suggested_side == "SELL" and p.direction == PatternDirection.BULLISH:
                    result["contradicting_patterns"].append(
                        f"{p.pattern.value}({scan.timeframe})"
                    )

    # Compute adjustments
    if result["confirming_patterns"]:
        result["confidence_adj"] += 0.10

    if result["contradicting_patterns"]:
        result["confidence_adj"] -= 0.15

    if result["has_doji"]:
        result["confidence_adj"] -= 0.10

    return result
