"""
RSI Divergence Detection — Leading indicator for reversals and continuations.

Detects divergences between price action and RSI:

Regular divergence (reversal signals):
  - Bullish: Price makes lower low, RSI makes higher low → reversal UP
  - Bearish: Price makes higher high, RSI makes lower high → reversal DOWN

Hidden divergence (continuation signals):
  - Bullish hidden: Price makes higher low, RSI makes lower low → trend continues UP
  - Bearish hidden: Price makes lower high, RSI makes higher high → trend continues DOWN

Uses swing pivot detection on both price and RSI to find divergence pairs.
Checks 4h timeframe by default (most reliable for swing trading).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DivergenceType(Enum):
    REGULAR_BULLISH = "regular_bullish"     # Price lower low, RSI higher low → reversal up
    REGULAR_BEARISH = "regular_bearish"     # Price higher high, RSI lower high → reversal down
    HIDDEN_BULLISH = "hidden_bullish"       # Price higher low, RSI lower low → continuation up
    HIDDEN_BEARISH = "hidden_bearish"       # Price lower high, RSI higher high → continuation down


@dataclass
class Divergence:
    """A detected RSI divergence."""
    div_type: DivergenceType
    # Price swing points
    price_idx_1: int       # Earlier swing index
    price_idx_2: int       # Later swing index (closer to current)
    price_val_1: float     # Price at earlier swing
    price_val_2: float     # Price at later swing
    # RSI swing points
    rsi_val_1: float       # RSI at earlier swing
    rsi_val_2: float       # RSI at later swing
    # Quality
    strength: float = 0.0  # 0.0-1.0 — how clean the divergence is
    rsi_zone: str = ""     # "oversold", "overbought", or "mid"

    @property
    def is_regular(self) -> bool:
        return self.div_type in (DivergenceType.REGULAR_BULLISH,
                                 DivergenceType.REGULAR_BEARISH)

    @property
    def is_hidden(self) -> bool:
        return self.div_type in (DivergenceType.HIDDEN_BULLISH,
                                 DivergenceType.HIDDEN_BEARISH)

    @property
    def is_bullish(self) -> bool:
        return self.div_type in (DivergenceType.REGULAR_BULLISH,
                                 DivergenceType.HIDDEN_BULLISH)

    @property
    def is_bearish(self) -> bool:
        return self.div_type in (DivergenceType.REGULAR_BEARISH,
                                 DivergenceType.HIDDEN_BEARISH)

    def __repr__(self):
        return (f"Divergence({self.div_type.value} "
                f"price={self.price_val_1:.2f}→{self.price_val_2:.2f} "
                f"rsi={self.rsi_val_1:.1f}→{self.rsi_val_2:.1f} "
                f"str={self.strength:.2f} zone={self.rsi_zone})")


@dataclass
class DivergenceScanResult:
    """Result of scanning for divergences on a timeframe."""
    timeframe: str
    divergences: List[Divergence] = field(default_factory=list)
    # Pre-computed flags
    has_regular_bullish: bool = False
    has_regular_bearish: bool = False
    has_hidden_bullish: bool = False
    has_hidden_bearish: bool = False
    strongest: Optional[Divergence] = None


class DivergenceDetector:
    """
    Detects RSI divergences on OHLCV data using swing pivot analysis.

    Args:
        lookback: How many candles back to search for divergence pairs (10-20)
        pivot_lookback: Window for pivot point detection (3-5 candles each side)
        rsi_period: RSI calculation period
        oversold: RSI threshold for oversold zone (strengthens bullish div)
        overbought: RSI threshold for overbought zone (strengthens bearish div)
    """

    def __init__(
        self,
        lookback: int = 20,
        pivot_lookback: int = 3,
        rsi_period: int = 14,
        oversold: float = 40.0,
        overbought: float = 60.0,
    ):
        self.lookback = lookback
        self.pivot_lookback = pivot_lookback
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def scan(self, ohlcv: list, timeframe: str = "") -> DivergenceScanResult:
        """
        Scan OHLCV data for RSI divergences.

        Returns DivergenceScanResult with all detected divergences.
        """
        result = DivergenceScanResult(timeframe=timeframe)

        if not ohlcv or len(ohlcv) < self.rsi_period + self.lookback:
            return result

        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv], dtype=np.float64)

        # Compute RSI series (need per-bar RSI, not just final value)
        rsi_series = self._rsi_series(closes, self.rsi_period)
        if rsi_series is None:
            return result

        n = len(closes)
        scan_start = max(0, n - self.lookback)

        # Find swing highs/lows in price and RSI within lookback window
        price_swing_highs = self._find_swing_highs(highs, self.pivot_lookback, scan_start)
        price_swing_lows = self._find_swing_lows(lows, self.pivot_lookback, scan_start)
        rsi_swing_highs = self._find_swing_highs(rsi_series, self.pivot_lookback, scan_start)
        rsi_swing_lows = self._find_swing_lows(rsi_series, self.pivot_lookback, scan_start)

        all_divs = []

        # Regular Bullish: price lower low + RSI higher low
        all_divs.extend(self._check_regular_bullish(
            lows, rsi_series, price_swing_lows, rsi_swing_lows
        ))

        # Regular Bearish: price higher high + RSI lower high
        all_divs.extend(self._check_regular_bearish(
            highs, rsi_series, price_swing_highs, rsi_swing_highs
        ))

        # Hidden Bullish: price higher low + RSI lower low
        all_divs.extend(self._check_hidden_bullish(
            lows, rsi_series, price_swing_lows, rsi_swing_lows
        ))

        # Hidden Bearish: price lower high + RSI higher high
        all_divs.extend(self._check_hidden_bearish(
            highs, rsi_series, price_swing_highs, rsi_swing_highs
        ))

        result.divergences = all_divs

        for d in all_divs:
            if d.div_type == DivergenceType.REGULAR_BULLISH:
                result.has_regular_bullish = True
            elif d.div_type == DivergenceType.REGULAR_BEARISH:
                result.has_regular_bearish = True
            elif d.div_type == DivergenceType.HIDDEN_BULLISH:
                result.has_hidden_bullish = True
            elif d.div_type == DivergenceType.HIDDEN_BEARISH:
                result.has_hidden_bearish = True

        if all_divs:
            result.strongest = max(all_divs, key=lambda d: d.strength)

        return result

    # ─────────────────────────────────────────
    # DIVERGENCE CHECKS
    # ─────────────────────────────────────────

    def _check_regular_bullish(
        self, lows: np.ndarray, rsi: np.ndarray,
        price_lows: List[int], rsi_lows: List[int]
    ) -> List[Divergence]:
        """Price makes LOWER low, RSI makes HIGHER low → bullish reversal."""
        divs = []
        for i in range(len(price_lows) - 1):
            idx1, idx2 = price_lows[i], price_lows[i + 1]
            if lows[idx2] >= lows[idx1]:
                continue  # Not a lower low

            # Find RSI lows near these price lows
            rsi1 = self._nearest_rsi_at(rsi, idx1, rsi_lows)
            rsi2 = self._nearest_rsi_at(rsi, idx2, rsi_lows)
            if rsi1 is None or rsi2 is None:
                continue

            if rsi[rsi2] > rsi[rsi1]:  # RSI higher low
                zone = "oversold" if rsi[rsi2] < self.oversold else "mid"
                strength = self._calc_strength(
                    lows[idx1], lows[idx2], rsi[rsi1], rsi[rsi2], zone
                )
                divs.append(Divergence(
                    div_type=DivergenceType.REGULAR_BULLISH,
                    price_idx_1=idx1, price_idx_2=idx2,
                    price_val_1=float(lows[idx1]), price_val_2=float(lows[idx2]),
                    rsi_val_1=float(rsi[rsi1]), rsi_val_2=float(rsi[rsi2]),
                    strength=strength, rsi_zone=zone,
                ))
        return divs

    def _check_regular_bearish(
        self, highs: np.ndarray, rsi: np.ndarray,
        price_highs: List[int], rsi_highs: List[int]
    ) -> List[Divergence]:
        """Price makes HIGHER high, RSI makes LOWER high → bearish reversal."""
        divs = []
        for i in range(len(price_highs) - 1):
            idx1, idx2 = price_highs[i], price_highs[i + 1]
            if highs[idx2] <= highs[idx1]:
                continue  # Not a higher high

            rsi1 = self._nearest_rsi_at(rsi, idx1, rsi_highs)
            rsi2 = self._nearest_rsi_at(rsi, idx2, rsi_highs)
            if rsi1 is None or rsi2 is None:
                continue

            if rsi[rsi2] < rsi[rsi1]:  # RSI lower high
                zone = "overbought" if rsi[rsi2] > self.overbought else "mid"
                strength = self._calc_strength(
                    highs[idx1], highs[idx2], rsi[rsi1], rsi[rsi2], zone
                )
                divs.append(Divergence(
                    div_type=DivergenceType.REGULAR_BEARISH,
                    price_idx_1=idx1, price_idx_2=idx2,
                    price_val_1=float(highs[idx1]), price_val_2=float(highs[idx2]),
                    rsi_val_1=float(rsi[rsi1]), rsi_val_2=float(rsi[rsi2]),
                    strength=strength, rsi_zone=zone,
                ))
        return divs

    def _check_hidden_bullish(
        self, lows: np.ndarray, rsi: np.ndarray,
        price_lows: List[int], rsi_lows: List[int]
    ) -> List[Divergence]:
        """Price makes HIGHER low, RSI makes LOWER low → bullish continuation."""
        divs = []
        for i in range(len(price_lows) - 1):
            idx1, idx2 = price_lows[i], price_lows[i + 1]
            if lows[idx2] <= lows[idx1]:
                continue  # Not a higher low

            rsi1 = self._nearest_rsi_at(rsi, idx1, rsi_lows)
            rsi2 = self._nearest_rsi_at(rsi, idx2, rsi_lows)
            if rsi1 is None or rsi2 is None:
                continue

            if rsi[rsi2] < rsi[rsi1]:  # RSI lower low
                zone = "oversold" if rsi[rsi2] < self.oversold else "mid"
                strength = self._calc_strength(
                    lows[idx1], lows[idx2], rsi[rsi1], rsi[rsi2], zone
                )
                divs.append(Divergence(
                    div_type=DivergenceType.HIDDEN_BULLISH,
                    price_idx_1=idx1, price_idx_2=idx2,
                    price_val_1=float(lows[idx1]), price_val_2=float(lows[idx2]),
                    rsi_val_1=float(rsi[rsi1]), rsi_val_2=float(rsi[rsi2]),
                    strength=strength, rsi_zone=zone,
                ))
        return divs

    def _check_hidden_bearish(
        self, highs: np.ndarray, rsi: np.ndarray,
        price_highs: List[int], rsi_highs: List[int]
    ) -> List[Divergence]:
        """Price makes LOWER high, RSI makes HIGHER high → bearish continuation."""
        divs = []
        for i in range(len(price_highs) - 1):
            idx1, idx2 = price_highs[i], price_highs[i + 1]
            if highs[idx2] >= highs[idx1]:
                continue  # Not a lower high

            rsi1 = self._nearest_rsi_at(rsi, idx1, rsi_highs)
            rsi2 = self._nearest_rsi_at(rsi, idx2, rsi_highs)
            if rsi1 is None or rsi2 is None:
                continue

            if rsi[rsi2] > rsi[rsi1]:  # RSI higher high
                zone = "overbought" if rsi[rsi2] > self.overbought else "mid"
                strength = self._calc_strength(
                    highs[idx1], highs[idx2], rsi[rsi1], rsi[rsi2], zone
                )
                divs.append(Divergence(
                    div_type=DivergenceType.HIDDEN_BEARISH,
                    price_idx_1=idx1, price_idx_2=idx2,
                    price_val_1=float(highs[idx1]), price_val_2=float(highs[idx2]),
                    rsi_val_1=float(rsi[rsi1]), rsi_val_2=float(rsi[rsi2]),
                    strength=strength, rsi_zone=zone,
                ))
        return divs

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    @staticmethod
    def _find_swing_highs(data: np.ndarray, lookback: int,
                          start_idx: int) -> List[int]:
        """Find local maxima (swing highs) within data from start_idx onward."""
        highs = []
        for i in range(max(start_idx, lookback), len(data) - lookback):
            window = data[i - lookback: i + lookback + 1]
            if data[i] == np.max(window):
                highs.append(i)
        return highs

    @staticmethod
    def _find_swing_lows(data: np.ndarray, lookback: int,
                         start_idx: int) -> List[int]:
        """Find local minima (swing lows) within data from start_idx onward."""
        lows = []
        for i in range(max(start_idx, lookback), len(data) - lookback):
            window = data[i - lookback: i + lookback + 1]
            if data[i] == np.min(window):
                lows.append(i)
        return lows

    @staticmethod
    def _nearest_rsi_at(rsi: np.ndarray, price_idx: int,
                        rsi_pivots: List[int]) -> Optional[int]:
        """Find the RSI pivot index closest to a price pivot index."""
        if not rsi_pivots:
            return None
        # Allow up to 2 candles of slack
        best = None
        best_dist = float("inf")
        for ri in rsi_pivots:
            dist = abs(ri - price_idx)
            if dist <= 2 and dist < best_dist:
                best = ri
                best_dist = dist
        # Fallback: use RSI value at exact price index if no nearby pivot
        if best is None and 0 <= price_idx < len(rsi):
            return price_idx
        return best

    @staticmethod
    def _calc_strength(
        price1: float, price2: float,
        rsi1: float, rsi2: float,
        zone: str,
    ) -> float:
        """
        Compute divergence strength (0-1) based on:
        - RSI divergence magnitude (how far RSI moved against price)
        - Zone bonus (oversold/overbought makes it stronger)
        """
        rsi_diff = abs(rsi2 - rsi1)
        # Normalize RSI difference: 5 points = 0.5, 10+ = 1.0
        rsi_score = min(rsi_diff / 10.0, 1.0)

        # Zone bonus
        zone_bonus = 0.2 if zone in ("oversold", "overbought") else 0.0

        return min(rsi_score + zone_bonus, 1.0)

    @staticmethod
    def _rsi_series(closes: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Compute RSI for every bar (not just the last one)."""
        if len(closes) < period + 1:
            return None

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi = np.full(len(closes), 50.0)

        # Seed with SMA
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi


def evaluate_divergence_for_signal(
    scan_result: DivergenceScanResult,
    suggested_side: Optional[str],
) -> dict:
    """
    Evaluate detected divergences against a proposed trade direction.

    Returns dict with:
      - confidence_adj: float
      - blocked: bool — regular divergence AGAINST direction = hard block
      - block_reason: str
      - confirming: list of divergence descriptions
      - contradicting: list of divergence descriptions
    """
    result = {
        "confidence_adj": 0.0,
        "blocked": False,
        "block_reason": "",
        "confirming": [],
        "contradicting": [],
    }

    if not suggested_side or not scan_result.divergences:
        return result

    for d in scan_result.divergences:
        label = f"{d.div_type.value}({scan_result.timeframe})"

        if d.is_regular:
            if suggested_side == "BUY" and d.is_bullish:
                result["confirming"].append(label)
            elif suggested_side == "SELL" and d.is_bearish:
                result["confirming"].append(label)
            elif suggested_side == "BUY" and d.is_bearish:
                result["contradicting"].append(label)
                result["blocked"] = True
                result["block_reason"] = (
                    f"Regular bearish divergence on {scan_result.timeframe} "
                    f"contradicts BUY signal"
                )
            elif suggested_side == "SELL" and d.is_bullish:
                result["contradicting"].append(label)
                result["blocked"] = True
                result["block_reason"] = (
                    f"Regular bullish divergence on {scan_result.timeframe} "
                    f"contradicts SELL signal"
                )

        elif d.is_hidden:
            if suggested_side == "BUY" and d.is_bullish:
                result["confirming"].append(label)
            elif suggested_side == "SELL" and d.is_bearish:
                result["confirming"].append(label)
            # Hidden divergence against direction is not a hard block — just info

    # Confidence adjustments
    has_regular_confirm = any(
        "regular" in c for c in result["confirming"]
    )
    has_hidden_confirm = any(
        "hidden" in c for c in result["confirming"]
    )

    if has_regular_confirm:
        result["confidence_adj"] += 0.15
    if has_hidden_confirm:
        result["confidence_adj"] += 0.10

    return result
