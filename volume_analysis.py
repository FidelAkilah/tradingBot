"""
Volume, Funding Rate & Open Interest Analysis — Market microstructure signals.

Three independent analysis modules:
1. Enhanced Volume Analysis
   - On-Balance Volume (OBV) with EMA trend
   - Buy/Sell Volume estimation (candle-based)
   - Volume Profile (simplified POC/VAH/VAL)

2. Funding Rate Filter
   - Extreme funding as contrarian signal
   - Persistent funding as trend confirmation

3. Open Interest Analysis
   - OI change vs price divergence detection
   - Rising OI + trend = strong conviction
   - Falling OI + trend = weak/exhaustion

All three produce confidence adjustments that integrate into the
CandleAnalyzer's scoring pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class VolumeAnalysisResult:
    """Combined output of all volume-based analyses."""
    # OBV
    obv_trend: str = "neutral"         # "bullish", "bearish", "neutral"
    obv_value: float = 0.0
    obv_ema: float = 0.0
    obv_divergence: bool = False       # OBV vs price divergence

    # Buy/Sell Volume
    buy_volume_ratio: float = 0.5      # 0.0 = all sell, 1.0 = all buy
    buy_volume_total: float = 0.0
    sell_volume_total: float = 0.0
    volume_pressure: str = "neutral"   # "buying", "selling", "neutral"

    # Volume Profile
    poc_price: float = 0.0             # Point of Control (highest volume price)
    vah_price: float = 0.0             # Value Area High
    val_price: float = 0.0             # Value Area Low
    price_vs_poc: str = "at"           # "above", "below", "at"

    # Confidence adjustment
    confidence_adj: float = 0.0        # Sum of volume-based adjustments
    signals: List[str] = field(default_factory=list)  # Reasons for adjustment


@dataclass
class FundingRateResult:
    """Funding rate analysis output."""
    current_rate: float = 0.0          # Current funding rate
    avg_rate_8h: float = 0.0           # Average over recent periods
    is_extreme_positive: bool = False  # Very crowded long
    is_extreme_negative: bool = False  # Very crowded short
    is_persistent: bool = False        # Same sign for N periods

    # Confidence adjustment
    confidence_adj: float = 0.0
    block_long: bool = False           # Extreme +funding → risky to go long
    block_short: bool = False          # Extreme -funding → risky to go short
    signals: List[str] = field(default_factory=list)


@dataclass
class OpenInterestResult:
    """Open interest analysis output."""
    current_oi: float = 0.0
    oi_change_pct: float = 0.0         # % change over lookback
    price_change_pct: float = 0.0
    oi_trend: str = "flat"             # "rising", "falling", "flat"

    # OI-Price relationship
    conviction: str = "neutral"        # "strong", "weak", "exhaustion", "neutral"
    is_divergence: bool = False        # OI falling while price trending

    # Confidence adjustment
    confidence_adj: float = 0.0
    signals: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Volume Analyzer
# ─────────────────────────────────────────────

class VolumeAnalyzer:
    """
    Enhanced volume analysis using OHLCV candle data.

    Does NOT require tick-level data — works entirely from candle arrays,
    making it compatible with ccxt fetch_ohlcv.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.vc = config.volume

    def analyze(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        opens: np.ndarray,
    ) -> VolumeAnalysisResult:
        """
        Run full volume analysis on OHLCV arrays.

        Args:
            highs, lows, closes, volumes, opens: arrays from candle data
            (minimum ~20 candles required)

        Returns:
            VolumeAnalysisResult with OBV, buy/sell, volume profile
        """
        result = VolumeAnalysisResult()

        if len(closes) < 20:
            return result

        # 1. OBV analysis
        self._compute_obv(closes, volumes, result)

        # 2. Buy/Sell volume estimation
        self._compute_buy_sell_volume(opens, highs, lows, closes, volumes, result)

        # 3. Volume Profile
        self._compute_volume_profile(highs, lows, closes, volumes, result)

        # Aggregate confidence adjustments
        result.confidence_adj = sum(self._score_signals(result))

        return result

    def _compute_obv(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        result: VolumeAnalysisResult,
    ):
        """
        On-Balance Volume (OBV) with EMA trend detection.

        OBV accumulates volume on up-closes and subtracts on down-closes.
        Compare OBV to its own EMA to detect trend.
        """
        obv = np.zeros(len(closes))
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

        result.obv_value = obv[-1]

        # EMA of OBV for trend detection
        ema_period = self.vc.obv_ema_period
        obv_ema = self._ema(obv, ema_period)
        result.obv_ema = obv_ema[-1]

        # Determine OBV trend
        if obv[-1] > obv_ema[-1]:
            result.obv_trend = "bullish"
        elif obv[-1] < obv_ema[-1]:
            result.obv_trend = "bearish"
        else:
            result.obv_trend = "neutral"

        # OBV divergence: price making new high but OBV not (or vice versa)
        lookback = min(self.vc.obv_divergence_lookback, len(closes) - 1)
        if lookback >= 5:
            price_high_idx = np.argmax(closes[-lookback:])
            obv_high_idx = np.argmax(obv[-lookback:])
            price_low_idx = np.argmin(closes[-lookback:])
            obv_low_idx = np.argmin(obv[-lookback:])

            # Price at new high but OBV peaked earlier → bearish divergence
            if price_high_idx == lookback - 1 and obv_high_idx < lookback - 3:
                result.obv_divergence = True
                result.signals.append("obv_bearish_divergence")

            # Price at new low but OBV bottomed earlier → bullish divergence
            if price_low_idx == lookback - 1 and obv_low_idx < lookback - 3:
                result.obv_divergence = True
                result.signals.append("obv_bullish_divergence")

    def _compute_buy_sell_volume(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        result: VolumeAnalysisResult,
    ):
        """
        Estimate buy vs sell volume using candle-body position.

        Uses the "close location value" (CLV) method:
        buy_pct = (close - low) / (high - low)   for each candle
        This approximates the fraction of volume that was buying pressure.
        """
        lookback = self.vc.buy_sell_lookback
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        recent_volumes = volumes[-lookback:]

        total_buy = 0.0
        total_sell = 0.0

        for i in range(len(recent_closes)):
            hl_range = recent_highs[i] - recent_lows[i]
            if hl_range <= 0:
                # Doji — split evenly
                buy_pct = 0.5
            else:
                buy_pct = (recent_closes[i] - recent_lows[i]) / hl_range

            buy_vol = recent_volumes[i] * buy_pct
            sell_vol = recent_volumes[i] * (1.0 - buy_pct)
            total_buy += buy_vol
            total_sell += sell_vol

        total = total_buy + total_sell
        result.buy_volume_total = total_buy
        result.sell_volume_total = total_sell

        if total > 0:
            result.buy_volume_ratio = total_buy / total
        else:
            result.buy_volume_ratio = 0.5

        # Classify pressure
        threshold = self.vc.buy_sell_threshold
        if result.buy_volume_ratio >= threshold:
            result.volume_pressure = "buying"
        elif result.buy_volume_ratio <= (1.0 - threshold):
            result.volume_pressure = "selling"
        else:
            result.volume_pressure = "neutral"

    def _compute_volume_profile(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        result: VolumeAnalysisResult,
    ):
        """
        Simplified Volume Profile — find POC, VAH, VAL.

        Divides the price range into N bins, distributes volume proportionally,
        finds the bin with highest volume (POC), and computes the 70% value area.
        """
        lookback = self.vc.profile_lookback
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        recent_volumes = volumes[-lookback:]

        price_high = float(np.max(recent_highs))
        price_low = float(np.min(recent_lows))
        price_range = price_high - price_low

        if price_range <= 0:
            result.poc_price = recent_closes[-1]
            result.vah_price = recent_closes[-1]
            result.val_price = recent_closes[-1]
            return

        n_bins = self.vc.profile_bins
        bin_size = price_range / n_bins
        bins = np.zeros(n_bins)

        # Distribute volume across bins based on candle range
        for i in range(len(recent_closes)):
            candle_low = recent_lows[i]
            candle_high = recent_highs[i]
            candle_vol = recent_volumes[i]

            # Find which bins this candle spans
            low_bin = max(0, int((candle_low - price_low) / bin_size))
            high_bin = min(n_bins - 1, int((candle_high - price_low) / bin_size))

            n_candle_bins = high_bin - low_bin + 1
            vol_per_bin = candle_vol / n_candle_bins if n_candle_bins > 0 else 0

            for b in range(low_bin, high_bin + 1):
                bins[b] += vol_per_bin

        # POC: bin with highest volume
        poc_bin = int(np.argmax(bins))
        result.poc_price = price_low + (poc_bin + 0.5) * bin_size

        # Value Area: 70% of total volume around POC
        total_vol = np.sum(bins)
        target_vol = total_vol * self.vc.profile_value_area_pct

        # Expand outward from POC
        va_low_bin = poc_bin
        va_high_bin = poc_bin
        accumulated = bins[poc_bin]

        while accumulated < target_vol and (va_low_bin > 0 or va_high_bin < n_bins - 1):
            expand_low = bins[va_low_bin - 1] if va_low_bin > 0 else 0
            expand_high = bins[va_high_bin + 1] if va_high_bin < n_bins - 1 else 0

            if expand_low >= expand_high and va_low_bin > 0:
                va_low_bin -= 1
                accumulated += bins[va_low_bin]
            elif va_high_bin < n_bins - 1:
                va_high_bin += 1
                accumulated += bins[va_high_bin]
            else:
                va_low_bin -= 1
                accumulated += bins[va_low_bin]

        result.val_price = price_low + va_low_bin * bin_size
        result.vah_price = price_low + (va_high_bin + 1) * bin_size

        # Current price vs POC
        current_price = recent_closes[-1]
        poc_dist = abs(current_price - result.poc_price) / result.poc_price
        if poc_dist < self.vc.profile_poc_proximity:
            result.price_vs_poc = "at"
        elif current_price > result.poc_price:
            result.price_vs_poc = "above"
        else:
            result.price_vs_poc = "below"

    def _score_signals(self, result: VolumeAnalysisResult) -> List[float]:
        """Convert volume analysis into confidence adjustments."""
        adjustments = []

        # OBV trend confirmation: +0.05 if OBV agrees with direction
        # (Caller checks alignment with trade side)
        if result.obv_trend in ("bullish", "bearish"):
            adjustments.append(self.vc.obv_confirm_bonus)
            result.signals.append(f"obv_{result.obv_trend}")

        # OBV divergence: -0.10 (price vs OBV disagree)
        if result.obv_divergence:
            adjustments.append(self.vc.obv_divergence_penalty)

        # Strong buying/selling pressure: +0.05
        if result.volume_pressure in ("buying", "selling"):
            adjustments.append(self.vc.buy_sell_pressure_bonus)
            result.signals.append(f"volume_pressure_{result.volume_pressure}")

        return adjustments

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(data) < period:
            return data.copy()
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema


# ─────────────────────────────────────────────
# Funding Rate Analyzer
# ─────────────────────────────────────────────

class FundingRateAnalyzer:
    """
    Analyzes funding rate data for contrarian/confirmation signals.

    Funding rate is fetched via ccxt exchange.fetch_funding_rate() and
    exchange.fetch_funding_rate_history(). This class takes the raw data
    and produces signals.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.vc = config.volume

        # Cache: symbol → list of (timestamp, rate) tuples
        self._rate_history: Dict[str, List[Tuple[float, float]]] = {}
        self._last_fetch: Dict[str, float] = {}

    def update_rate(self, symbol: str, rate: float, timestamp: float = None):
        """Record a new funding rate observation."""
        if timestamp is None:
            timestamp = time.time()

        if symbol not in self._rate_history:
            self._rate_history[symbol] = []

        self._rate_history[symbol].append((timestamp, rate))

        # Keep only recent history (last 24h = ~3 periods at 8h funding)
        cutoff = timestamp - 86400.0
        self._rate_history[symbol] = [
            (t, r) for t, r in self._rate_history[symbol] if t >= cutoff
        ]

    def analyze(self, symbol: str, current_rate: float = None) -> FundingRateResult:
        """
        Analyze funding rate for a symbol.

        Args:
            symbol: Trading pair
            current_rate: Current funding rate (optional, uses latest cached)

        Returns:
            FundingRateResult with signals and confidence adjustment
        """
        result = FundingRateResult()
        history = self._rate_history.get(symbol, [])

        if current_rate is not None:
            result.current_rate = current_rate
        elif history:
            result.current_rate = history[-1][1]
        else:
            return result

        # Average rate
        if history:
            rates = [r for _, r in history]
            result.avg_rate_8h = float(np.mean(rates[-3:])) if len(rates) >= 3 else result.current_rate
        else:
            result.avg_rate_8h = result.current_rate

        # Extreme positive funding: crowded longs → contrarian short signal
        extreme_thresh = self.vc.funding_extreme_threshold
        if result.current_rate >= extreme_thresh:
            result.is_extreme_positive = True
            result.block_long = True
            result.confidence_adj -= self.vc.funding_extreme_penalty
            result.signals.append("extreme_positive_funding")

        # Extreme negative funding: crowded shorts → contrarian long signal
        elif result.current_rate <= -extreme_thresh:
            result.is_extreme_negative = True
            result.block_short = True
            result.confidence_adj -= self.vc.funding_extreme_penalty
            result.signals.append("extreme_negative_funding")

        # Persistent same-sign funding = trend confirmation
        if len(history) >= self.vc.funding_persistent_periods:
            recent = [r for _, r in history[-self.vc.funding_persistent_periods:]]
            all_positive = all(r > 0 for r in recent)
            all_negative = all(r < 0 for r in recent)

            if all_positive or all_negative:
                result.is_persistent = True
                result.confidence_adj += self.vc.funding_persistent_bonus
                result.signals.append("persistent_funding")

        return result


# ─────────────────────────────────────────────
# Open Interest Analyzer
# ─────────────────────────────────────────────

class OpenInterestAnalyzer:
    """
    Analyzes open interest changes relative to price action.

    OI + Price relationships:
    - Rising OI + Rising Price = New longs entering → strong bullish
    - Rising OI + Falling Price = New shorts entering → strong bearish
    - Falling OI + Rising Price = Shorts covering → weak bullish (exhaustion?)
    - Falling OI + Falling Price = Longs exiting → weak bearish (exhaustion?)
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.vc = config.volume

        # Cache: symbol → list of (timestamp, oi_value) tuples
        self._oi_history: Dict[str, List[Tuple[float, float]]] = {}
        self._last_fetch: Dict[str, float] = {}

    def update_oi(self, symbol: str, oi_value: float, timestamp: float = None):
        """Record a new open interest observation."""
        if timestamp is None:
            timestamp = time.time()

        if symbol not in self._oi_history:
            self._oi_history[symbol] = []

        self._oi_history[symbol].append((timestamp, oi_value))

        # Keep 24h history
        cutoff = timestamp - 86400.0
        self._oi_history[symbol] = [
            (t, v) for t, v in self._oi_history[symbol] if t >= cutoff
        ]

    def analyze(
        self,
        symbol: str,
        current_price: float,
        price_change_pct: float,
        current_oi: float = None,
    ) -> OpenInterestResult:
        """
        Analyze OI changes relative to price.

        Args:
            symbol: Trading pair
            current_price: Current price
            price_change_pct: Price change % over lookback period
            current_oi: Current open interest (optional, uses latest cached)

        Returns:
            OpenInterestResult with conviction assessment
        """
        result = OpenInterestResult()
        result.price_change_pct = price_change_pct
        history = self._oi_history.get(symbol, [])

        if current_oi is not None:
            result.current_oi = current_oi
        elif history:
            result.current_oi = history[-1][1]
        else:
            return result

        # Calculate OI change
        if len(history) >= 2:
            old_oi = history[0][1]
            if old_oi > 0:
                result.oi_change_pct = (result.current_oi - old_oi) / old_oi * 100.0
            else:
                result.oi_change_pct = 0.0
        else:
            return result

        # Classify OI trend
        rising_thresh = self.vc.oi_rising_threshold
        falling_thresh = self.vc.oi_falling_threshold

        if result.oi_change_pct >= rising_thresh:
            result.oi_trend = "rising"
        elif result.oi_change_pct <= falling_thresh:
            result.oi_trend = "falling"
        else:
            result.oi_trend = "flat"

        # OI-Price conviction matrix
        price_up = price_change_pct > self.vc.oi_price_move_threshold
        price_down = price_change_pct < -self.vc.oi_price_move_threshold
        oi_rising = result.oi_trend == "rising"
        oi_falling = result.oi_trend == "falling"

        if oi_rising and (price_up or price_down):
            # New money entering — strong conviction in the direction
            result.conviction = "strong"
            result.confidence_adj += self.vc.oi_strong_conviction_bonus
            result.signals.append("oi_rising_strong_conviction")

        elif oi_falling and (price_up or price_down):
            # Money leaving — weak/exhaustion move
            result.conviction = "exhaustion"
            result.is_divergence = True
            result.confidence_adj -= self.vc.oi_exhaustion_penalty
            result.signals.append("oi_falling_exhaustion")

        elif oi_rising and not price_up and not price_down:
            # OI building but price flat — anticipation, neutral
            result.conviction = "neutral"
            result.signals.append("oi_building_flat_price")

        else:
            result.conviction = "neutral"

        return result
