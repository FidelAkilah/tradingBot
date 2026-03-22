"""
Market Regime Classifier — determines whether conditions favor trading.

Three independent detection methods vote on the current regime:
1. ADX-based trend strength (extended thresholds)
2. Bollinger Band Width percentile (squeeze vs expansion)
3. Price vs EMAs (trending vs choppy)

A 2-of-3 voting system combines results. Entry is only allowed when
at least 2 methods agree the market is "trending".

Breakout override: if SQUEEZING but BB width expanding >30% in 3 candles
AND volume >2x average, allow the trade with tighter SL (0.7x ATR).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classification."""
    STRONG_TREND = "strong_trend"   # ADX > 30 — full size
    TREND = "trend"                 # ADX 25-30 — full size
    WEAK = "weak"                   # ADX 20-25 — reduce 50%
    RANGING = "ranging"             # ADX < 20 — block
    SQUEEZING = "squeezing"        # BB width low — potential breakout
    EXPANDING = "expanding"         # BB width high — volatility surge
    CHOPPY = "choppy"              # Price whipsawing across EMAs


@dataclass
class RegimeResult:
    """Combined regime detection output."""
    # Individual method results
    adx_regime: RegimeType = RegimeType.RANGING
    bb_regime: RegimeType = RegimeType.RANGING
    ema_regime: RegimeType = RegimeType.RANGING

    # Combined verdict
    combined_regime: RegimeType = RegimeType.RANGING
    is_trending: bool = False       # 2-of-3 agree on trending
    regime_blocked: bool = True     # Should we block trading?
    size_multiplier: float = 0.0    # Position size scaling (0.0 = blocked, 1.0 = full)

    # Breakout override
    is_breakout: bool = False       # Squeeze breakout detected
    breakout_sl_mult: float = 1.0   # SL multiplier (0.7 for breakouts)

    # Raw values for logging
    adx_value: float = 0.0
    bb_width_pctl: float = 0.0
    ema_cross_count: int = 0
    bb_width_change_pct: float = 0.0


class MarketRegimeClassifier:
    """
    Classifies market regime using 3 independent methods with 2-of-3 voting.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.rc = config.regime
        self.cc = config.candle

    def classify(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        adx_value: float = 0.0,
    ) -> RegimeResult:
        """
        Run all 3 regime detection methods and combine via voting.

        Args:
            highs, lows, closes, volumes: OHLCV arrays from candle data
            adx_value: Pre-computed ADX (from CandleAnalyzer._adx)

        Returns:
            RegimeResult with combined verdict
        """
        result = RegimeResult()

        # Method 1: ADX-based regime
        result.adx_regime = self._adx_regime(adx_value)
        result.adx_value = adx_value

        # Method 2: Bollinger Band Width percentile
        bb_regime, bb_pctl, bb_change = self._bb_width_regime(closes)
        result.bb_regime = bb_regime
        result.bb_width_pctl = bb_pctl
        result.bb_width_change_pct = bb_change

        # Method 3: Price vs EMAs
        ema_regime, cross_count = self._ema_regime(closes)
        result.ema_regime = ema_regime
        result.ema_cross_count = cross_count

        # 2-of-3 voting
        trending_votes = sum([
            self._is_trending_vote(result.adx_regime),
            self._is_trending_vote(result.bb_regime),
            self._is_trending_vote(result.ema_regime),
        ])

        result.is_trending = trending_votes >= 2

        # Determine combined regime and position sizing
        if result.is_trending:
            # Use the most specific trending regime from ADX
            if result.adx_regime == RegimeType.STRONG_TREND:
                result.combined_regime = RegimeType.STRONG_TREND
                result.size_multiplier = 1.0
            elif result.adx_regime == RegimeType.TREND:
                result.combined_regime = RegimeType.TREND
                result.size_multiplier = 1.0
            elif result.adx_regime == RegimeType.WEAK:
                result.combined_regime = RegimeType.WEAK
                result.size_multiplier = 0.5
            else:
                # ADX says ranging but other 2 say trending — cautious entry
                result.combined_regime = RegimeType.WEAK
                result.size_multiplier = 0.5
            result.regime_blocked = False
        else:
            # Not trending — determine if squeezing, choppy, or ranging
            if result.bb_regime == RegimeType.SQUEEZING:
                result.combined_regime = RegimeType.SQUEEZING
            elif result.ema_regime == RegimeType.CHOPPY:
                result.combined_regime = RegimeType.CHOPPY
            else:
                result.combined_regime = RegimeType.RANGING
            result.size_multiplier = 0.0
            result.regime_blocked = True

        # Breakout override: squeeze → expansion with volume
        if result.regime_blocked and result.bb_regime == RegimeType.SQUEEZING:
            if self._check_breakout(closes, volumes):
                result.is_breakout = True
                result.regime_blocked = False
                result.combined_regime = RegimeType.EXPANDING
                result.size_multiplier = 1.0
                result.breakout_sl_mult = self.rc.breakout_sl_mult  # 0.7x ATR

        return result

    def _adx_regime(self, adx: float) -> RegimeType:
        """Classify regime based on ADX value."""
        if adx <= 0:
            return RegimeType.RANGING
        if adx >= self.rc.adx_strong_trend:
            return RegimeType.STRONG_TREND
        if adx >= self.rc.adx_trend:
            return RegimeType.TREND
        if adx >= self.rc.adx_weak:
            return RegimeType.WEAK
        return RegimeType.RANGING

    def _bb_width_regime(self, closes: np.ndarray):
        """
        Classify regime based on Bollinger Band Width percentile.

        BB Width = (upper - lower) / middle
        Compare current width to percentile of last N candles.

        Returns: (regime, percentile, width_change_pct)
        """
        period = self.rc.bb_period
        std_mult = self.rc.bb_std_dev
        lookback = self.rc.bb_lookback

        if len(closes) < period + 2:
            # Not enough data for even one BB width — abstain (don't block)
            return RegimeType.TREND, 50.0, 0.0

        # Compute BB width for the last `lookback` candles
        widths = []
        for i in range(period, len(closes)):
            window = closes[i - period:i]
            sma = np.mean(window)
            std = np.std(window, ddof=1)
            if sma > 0:
                upper = sma + std_mult * std
                lower = sma - std_mult * std
                width = (upper - lower) / sma
                widths.append(width)

        if len(widths) < 2:
            return RegimeType.RANGING, 50.0, 0.0

        # Use last `lookback` widths for percentile
        recent_widths = widths[-lookback:] if len(widths) >= lookback else widths
        current_width = widths[-1]

        # Percentile of current width vs historical
        percentile = float(np.sum(np.array(recent_widths) <= current_width) / len(recent_widths) * 100.0)

        # Width change over last 3 candles for breakout detection
        if len(widths) >= 4:
            width_3_ago = widths[-4]
            width_change_pct = ((current_width - width_3_ago) / width_3_ago * 100.0) if width_3_ago > 0 else 0.0
        else:
            width_change_pct = 0.0

        if percentile >= self.rc.bb_expanding_pctl:
            regime = RegimeType.EXPANDING
        elif percentile <= self.rc.bb_squeezing_pctl:
            regime = RegimeType.SQUEEZING
        else:
            regime = RegimeType.TREND  # Normal range = trending-ish

        return regime, percentile, width_change_pct

    def _ema_regime(self, closes: np.ndarray):
        """
        Classify regime based on price relationship to EMAs.

        Trending: price consistently above/below all 3 EMAs for N bars.
        Choppy: multiple EMA crosses in recent window.

        Returns: (regime, cross_count)
        """
        if len(closes) < self.cc.trend_ema_period + self.rc.ema_choppy_window:
            # Not enough data — abstain (don't block)
            return RegimeType.TREND, 0

        # Compute EMAs
        fast_ema = self._ema(closes, self.cc.fast_ema_period)
        slow_ema = self._ema(closes, self.cc.slow_ema_period)
        trend_ema = self._ema(closes, self.cc.trend_ema_period)

        # Count crosses in the last ema_choppy_window candles
        window = self.rc.ema_choppy_window
        cross_count = 0
        for i in range(-window, 0):
            if i - 1 < -len(closes):
                continue
            # A "cross" is when price crosses any EMA
            prev_above_fast = closes[i - 1] > fast_ema[i - 1]
            curr_above_fast = closes[i] > fast_ema[i]
            if prev_above_fast != curr_above_fast:
                cross_count += 1

            prev_above_slow = closes[i - 1] > slow_ema[i - 1]
            curr_above_slow = closes[i] > slow_ema[i]
            if prev_above_slow != curr_above_slow:
                cross_count += 1

        # Check if price is consistently on one side of all EMAs
        trend_bars = self.rc.ema_trend_bars
        if len(closes) >= trend_bars:
            recent = closes[-trend_bars:]
            recent_fast = fast_ema[-trend_bars:]
            recent_slow = slow_ema[-trend_bars:]
            recent_trend = trend_ema[-trend_bars:]

            all_above = np.all(recent > recent_fast) and np.all(recent > recent_slow) and np.all(recent > recent_trend)
            all_below = np.all(recent < recent_fast) and np.all(recent < recent_slow) and np.all(recent < recent_trend)

            if all_above or all_below:
                return RegimeType.TREND, cross_count

        if cross_count >= self.rc.ema_choppy_crosses:
            return RegimeType.CHOPPY, cross_count

        # Not clearly trending or choppy
        return RegimeType.WEAK, cross_count

    def _check_breakout(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """
        Check breakout override conditions:
        1. BB width expanding >30% in last 3 candles
        2. Volume > 2x average
        """
        # BB width expansion check
        period = self.rc.bb_period
        if len(closes) < period + self.rc.breakout_bb_candles + 1:
            return False

        # Compute BB width for current and 3-candles-ago
        def _bb_width_at(idx):
            window = closes[idx - period:idx]
            sma = np.mean(window)
            std = np.std(window, ddof=1)
            if sma > 0:
                return (2 * self.rc.bb_std_dev * std) / sma
            return 0.0

        current_bb = _bb_width_at(len(closes))
        past_bb = _bb_width_at(len(closes) - self.rc.breakout_bb_candles)

        if past_bb <= 0:
            return False

        bb_change_pct = (current_bb - past_bb) / past_bb * 100.0

        if bb_change_pct < self.rc.breakout_bb_expand_pct:
            return False

        # Volume check
        if len(volumes) < 20:
            return False

        avg_vol = np.mean(volumes[-20:])
        current_vol = volumes[-1]

        if avg_vol <= 0:
            return False

        vol_ratio = current_vol / avg_vol

        if vol_ratio < self.rc.breakout_volume_mult:
            return False

        logger.info(
            f"Breakout override: BB width expanded {bb_change_pct:.1f}% "
            f"(>{self.rc.breakout_bb_expand_pct}%), volume {vol_ratio:.1f}x "
            f"(>{self.rc.breakout_volume_mult}x)"
        )
        return True

    @staticmethod
    def _is_trending_vote(regime: RegimeType) -> bool:
        """Does this regime type count as a 'trending' vote?"""
        return regime in (
            RegimeType.STRONG_TREND,
            RegimeType.TREND,
            RegimeType.EXPANDING,
        )

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average (same as CandleAnalyzer._ema)."""
        if len(data) < period:
            return data.copy()
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
