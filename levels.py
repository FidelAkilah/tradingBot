"""
Support/Resistance Level Detection — Price structure analysis.

Detects key price levels from historical candle data:
1. Swing high/low pivot points (5-candle lookback)
2. Level strength scoring based on touch count
3. Fibonacci retracements from the most recent major swing
4. Nearest S/R for entry/exit optimization
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """A detected support or resistance price level."""
    price: float
    level_type: str          # "support" or "resistance"
    touches: int = 1         # How many times price reacted at this level
    source: str = "pivot"    # "pivot" or "fibonacci"
    fib_ratio: float = 0.0   # 0.382, 0.5, 0.618, 0.786 (if fibonacci)
    strength: float = 0.0    # Composite strength score (0-1)

    def __repr__(self):
        src = f"fib_{self.fib_ratio}" if self.source == "fibonacci" else "pivot"
        return f"Level({self.level_type} ${self.price:.2f} touches={self.touches} {src})"


@dataclass
class LevelAnalysis:
    """Complete S/R analysis for a symbol."""
    symbol: str
    current_price: float
    supports: List[PriceLevel] = field(default_factory=list)     # Below price, sorted descending
    resistances: List[PriceLevel] = field(default_factory=list)   # Above price, sorted ascending
    fib_levels: List[PriceLevel] = field(default_factory=list)    # Fibonacci retracements
    nearest_support: Optional[PriceLevel] = None
    nearest_resistance: Optional[PriceLevel] = None
    at_support: bool = False       # Price within proximity of a support
    at_resistance: bool = False    # Price within proximity of a resistance
    tp_blocked: bool = False       # A strong resistance blocks the TP target (longs)


class LevelDetector:
    """
    Detects support/resistance levels from OHLCV candle data.

    Uses pivot point detection (swing highs/lows) and Fibonacci
    retracements from the most recent major swing.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.cc = config.candle
        self._cache: Dict[str, LevelAnalysis] = {}
        self._cache_ts: Dict[str, float] = {}

    def analyze(
        self,
        symbol: str,
        ohlcv_1h: list,
        current_price: float,
        atr: float = 0.0,
        suggested_side: Optional[str] = None,
        tp_price: Optional[float] = None,
        timestamp: float = 0.0,
    ) -> LevelAnalysis:
        """
        Detect S/R levels and return analysis relative to current price.

        Args:
            symbol: Trading pair
            ohlcv_1h: 1h OHLCV candle data (list of [ts, o, h, l, c, v])
            current_price: Current mid price
            atr: ATR value for proximity calculation
            suggested_side: "BUY" or "SELL" for TP-block check
            tp_price: Target price to check against resistance/support
            timestamp: Current timestamp
        """
        if not ohlcv_1h or len(ohlcv_1h) < 20:
            return LevelAnalysis(symbol=symbol, current_price=current_price)

        highs = np.array([c[2] for c in ohlcv_1h], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv_1h], dtype=np.float64)
        closes = np.array([c[4] for c in ohlcv_1h], dtype=np.float64)

        # 1. Detect pivot highs and lows
        pivot_lookback = self.cc.sr_pivot_lookback
        pivot_highs = self._find_pivot_highs(highs, pivot_lookback)
        pivot_lows = self._find_pivot_lows(lows, pivot_lookback)

        # 2. Cluster nearby pivots and count touches
        proximity = atr * self.cc.sr_proximity_atr_pct if atr > 0 else current_price * 0.003
        all_levels = self._cluster_levels(pivot_highs, pivot_lows, proximity)

        # 3. Add Fibonacci retracements
        fib_levels = self._fibonacci_retracements(highs, lows, closes, current_price)

        # 4. Classify as support/resistance relative to current price
        supports = []
        resistances = []

        for level in all_levels:
            if level.price < current_price:
                level.level_type = "support"
                supports.append(level)
            else:
                level.level_type = "resistance"
                resistances.append(level)

        for level in fib_levels:
            if level.price < current_price:
                level.level_type = "support"
                supports.append(level)
            else:
                level.level_type = "resistance"
                resistances.append(level)

        # Sort: supports descending (nearest first), resistances ascending (nearest first)
        supports.sort(key=lambda l: l.price, reverse=True)
        resistances.sort(key=lambda l: l.price)

        # Keep nearest 3 each
        supports = supports[:3]
        resistances = resistances[:3]

        # Compute strength scores
        max_touches = max((l.touches for l in supports + resistances), default=1)
        for level in supports + resistances:
            touch_score = min(level.touches / max(max_touches, 1), 1.0)
            fib_bonus = 0.3 if level.source == "fibonacci" else 0.0
            level.strength = min(touch_score * 0.7 + fib_bonus, 1.0)

        # Nearest levels
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None

        # Proximity checks
        at_support = False
        at_resistance = False
        if nearest_support and proximity > 0:
            at_support = abs(current_price - nearest_support.price) <= proximity
        if nearest_resistance and proximity > 0:
            at_resistance = abs(current_price - nearest_resistance.price) <= proximity

        # TP blocked check: is there a strong resistance between entry and TP?
        tp_blocked = False
        if tp_price is not None and suggested_side == "BUY":
            for r in resistances:
                if r.price < tp_price and r.touches >= self.cc.sr_min_touches:
                    tp_blocked = True
                    break
        elif tp_price is not None and suggested_side == "SELL":
            for s in supports:
                if s.price > tp_price and s.touches >= self.cc.sr_min_touches:
                    tp_blocked = True
                    break

        result = LevelAnalysis(
            symbol=symbol,
            current_price=current_price,
            supports=supports,
            resistances=resistances,
            fib_levels=fib_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            at_support=at_support,
            at_resistance=at_resistance,
            tp_blocked=tp_blocked,
        )

        self._cache[symbol] = result
        self._cache_ts[symbol] = timestamp
        return result

    def get_cached(self, symbol: str) -> Optional[LevelAnalysis]:
        """Return cached analysis if available."""
        return self._cache.get(symbol)

    # ─────────────────────────────────────────
    # PIVOT DETECTION
    # ─────────────────────────────────────────

    @staticmethod
    def _find_pivot_highs(highs: np.ndarray, lookback: int) -> List[float]:
        """Find swing high pivot points (local maxima)."""
        pivots = []
        for i in range(lookback, len(highs) - lookback):
            window = highs[i - lookback: i + lookback + 1]
            if highs[i] == np.max(window):
                pivots.append(float(highs[i]))
        return pivots

    @staticmethod
    def _find_pivot_lows(lows: np.ndarray, lookback: int) -> List[float]:
        """Find swing low pivot points (local minima)."""
        pivots = []
        for i in range(lookback, len(lows) - lookback):
            window = lows[i - lookback: i + lookback + 1]
            if lows[i] == np.min(window):
                pivots.append(float(lows[i]))
        return pivots

    # ─────────────────────────────────────────
    # LEVEL CLUSTERING
    # ─────────────────────────────────────────

    @staticmethod
    def _cluster_levels(
        pivot_highs: List[float],
        pivot_lows: List[float],
        proximity: float,
    ) -> List[PriceLevel]:
        """
        Cluster nearby pivot points into consolidated levels.
        Each touch of a level increases its strength.
        """
        all_pivots = sorted(pivot_highs + pivot_lows)
        if not all_pivots:
            return []

        clusters: List[List[float]] = []
        current_cluster = [all_pivots[0]]

        for price in all_pivots[1:]:
            if abs(price - current_cluster[-1]) <= proximity:
                current_cluster.append(price)
            else:
                clusters.append(current_cluster)
                current_cluster = [price]
        clusters.append(current_cluster)

        levels = []
        for cluster in clusters:
            avg_price = sum(cluster) / len(cluster)
            levels.append(PriceLevel(
                price=avg_price,
                level_type="unknown",  # Classified later
                touches=len(cluster),
                source="pivot",
            ))

        return levels

    # ─────────────────────────────────────────
    # FIBONACCI RETRACEMENTS
    # ─────────────────────────────────────────

    def _fibonacci_retracements(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        current_price: float,
    ) -> List[PriceLevel]:
        """
        Calculate Fibonacci retracement levels from the most recent major swing.

        A "major swing" is defined as a move > sr_fib_min_swing_pct%.
        """
        min_swing_pct = self.cc.sr_fib_min_swing_pct
        if len(closes) < 10:
            return []

        # Find the most recent significant swing
        swing_high, swing_low = self._find_major_swing(highs, lows, closes, min_swing_pct)

        if swing_high is None or swing_low is None:
            return []

        swing_range = swing_high - swing_low
        if swing_range <= 0:
            return []

        fib_ratios = [0.382, 0.5, 0.618, 0.786]
        levels = []

        # Determine swing direction: if current price is closer to high → downswing (retracement up)
        # If current price is closer to low → upswing (retracement down)
        for ratio in fib_ratios:
            # Retracement from high
            level_price = swing_high - swing_range * ratio
            levels.append(PriceLevel(
                price=level_price,
                level_type="unknown",
                touches=1,
                source="fibonacci",
                fib_ratio=ratio,
            ))

        return levels

    @staticmethod
    def _find_major_swing(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        min_pct: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Find the most recent major swing (high-low pair) with > min_pct% range.
        Searches backwards from most recent candles.
        """
        n = len(highs)
        if n < 10:
            return None, None

        # Look at last 50 candles (or all available)
        window = min(n, 50)
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]

        swing_high = float(np.max(recent_highs))
        swing_low = float(np.min(recent_lows))

        if swing_low <= 0:
            return None, None

        swing_pct = (swing_high - swing_low) / swing_low * 100.0
        if swing_pct < min_pct:
            return None, None

        return swing_high, swing_low
