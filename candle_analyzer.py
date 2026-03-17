"""
Multi-Timeframe Candle Analyzer — The swing trading brain.

Unlike the order-book-only approach that failed at scalping,
this uses 1h and 4h candles to identify:
1. Trend direction (EMA crossover + slope)
2. Momentum (RSI)
3. Volatility (ATR for dynamic TP/SL)
4. Volume confirmation (surge detection)

The order book walls are still used, but as ENTRY CONFIRMATION
(not as position anchors). If a strong bid wall aligns with
a bullish candle setup, that's a high-confidence trade.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class CandleSignal:
    """Analysis result from candle data for a single timeframe."""
    timeframe: str
    trend: TrendDirection
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    trend_ema: float = 0.0          # 50-period EMA for long-term direction
    ema_slope: float = 0.0          # Rate of change of slow EMA (trend strength)
    rsi: float = 50.0
    atr: float = 0.0                # Average True Range (volatility)
    atr_pct: float = 0.0            # ATR as % of price
    volume_ratio: float = 1.0       # Current vol / avg vol
    is_volume_surge: bool = False
    last_close: float = 0.0
    timestamp: float = 0.0


@dataclass
class SwingSignal:
    """Combined multi-timeframe signal for swing trade decisions."""
    symbol: str
    timestamp: float
    signals: Dict[str, CandleSignal] = field(default_factory=dict)

    # Derived fields
    primary_trend: TrendDirection = TrendDirection.NEUTRAL
    trend_aligned: bool = False     # Do all timeframes agree?
    suggested_side: Optional[str] = None  # "BUY", "SELL", or None
    confidence: float = 0.0         # 0.0 to 1.0

    # Dynamic TP/SL from ATR
    atr_tp_distance: float = 0.0    # In price units
    atr_sl_distance: float = 0.0
    atr_tp_pct: float = 0.0         # As percentage
    atr_sl_pct: float = 0.0

    # RSI context
    rsi_1h: float = 50.0
    rsi_4h: float = 50.0
    is_oversold: bool = False
    is_overbought: bool = False


class CandleAnalyzer:
    """
    Fetches and analyzes multi-timeframe candles for swing trade entries.

    Key insight vs the old approach:
    - Old: order book snapshot → detect wall → trade in front of wall → wall disappears → loss
    - New: candle trend + RSI + ATR → confirm with order book wall → trade with ATR-based TP/SL
           → hold for hours, not seconds → wall disappearing doesn't matter
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.cc = config.candle

        # Cache candle data to avoid refetching
        self._candle_cache: Dict[str, Dict[str, list]] = defaultdict(dict)
        self._last_fetch: Dict[str, float] = {}
        self._fetch_cooldown = 60.0  # Refetch candles every 60s at most

    async def fetch_candles(self, exchange, symbol: str) -> Dict[str, list]:
        """
        Fetch candles for all configured timeframes.
        Caches results to avoid hammering the API.
        """
        now = time.time()
        cache_key = symbol
        last = self._last_fetch.get(cache_key, 0)

        if now - last < self._fetch_cooldown and cache_key in self._candle_cache:
            return self._candle_cache[cache_key]

        candles = {}
        for tf in self.cc.timeframes:
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, tf,
                    limit=self.cc.lookback_candles
                )
                if ohlcv and len(ohlcv) >= 20:
                    candles[tf] = ohlcv
                    logger.debug(f"[{symbol}] Fetched {len(ohlcv)} {tf} candles")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to fetch {tf} candles: {e}")

        self._candle_cache[cache_key] = candles
        self._last_fetch[cache_key] = now
        return candles

    def analyze(self, symbol: str, candles: Dict[str, list], timestamp: float) -> SwingSignal:
        """
        Run multi-timeframe analysis and produce a SwingSignal.
        """
        signal = SwingSignal(symbol=symbol, timestamp=timestamp)

        for tf, ohlcv in candles.items():
            if not ohlcv or len(ohlcv) < 20:
                continue

            cs = self._analyze_timeframe(tf, ohlcv, timestamp)
            signal.signals[tf] = cs

            if tf == "1h":
                signal.rsi_1h = cs.rsi
            elif tf == "4h":
                signal.rsi_4h = cs.rsi

        # Combine signals across timeframes
        self._combine_signals(signal)

        return signal

    def _analyze_timeframe(self, timeframe: str, ohlcv: list, timestamp: float) -> CandleSignal:
        """Analyze a single timeframe's candle data."""
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes = np.array([c[5] for c in ohlcv], dtype=np.float64)

        last_close = closes[-1]

        # EMAs
        fast_ema = self._ema(closes, self.cc.fast_ema_period)
        slow_ema = self._ema(closes, self.cc.slow_ema_period)
        trend_ema = self._ema(closes, self.cc.trend_ema_period)

        # EMA slope (rate of change over last 5 periods)
        if len(slow_ema) >= 5:
            slope = (slow_ema[-1] - slow_ema[-5]) / slow_ema[-5] * 100 if slow_ema[-5] > 0 else 0
        else:
            slope = 0.0

        # RSI
        rsi = self._rsi(closes, self.cc.rsi_period)

        # ATR
        atr = self._atr(highs, lows, closes, self.cc.atr_period)
        atr_pct = (atr / last_close * 100) if last_close > 0 else 0

        # Volume
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_vol = volumes[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        is_surge = vol_ratio >= self.cc.volume_surge_multiplier

        # Trend determination
        trend = self._determine_trend(
            fast_ema[-1] if len(fast_ema) > 0 else 0,
            slow_ema[-1] if len(slow_ema) > 0 else 0,
            trend_ema[-1] if len(trend_ema) > 0 else 0,
            last_close, slope
        )

        return CandleSignal(
            timeframe=timeframe,
            trend=trend,
            fast_ema=fast_ema[-1] if len(fast_ema) > 0 else 0,
            slow_ema=slow_ema[-1] if len(slow_ema) > 0 else 0,
            trend_ema=trend_ema[-1] if len(trend_ema) > 0 else 0,
            ema_slope=slope,
            rsi=rsi,
            atr=atr,
            atr_pct=atr_pct,
            volume_ratio=vol_ratio,
            is_volume_surge=is_surge,
            last_close=last_close,
            timestamp=timestamp,
        )

    def _determine_trend(
        self, fast: float, slow: float, trend: float,
        price: float, slope: float
    ) -> TrendDirection:
        """
        Determine trend from EMA relationships.

        Strong Bullish: price > fast > slow > trend, positive slope
        Bullish: fast > slow, price above slow
        Neutral: EMAs tangled
        Bearish: fast < slow, price below slow
        Strong Bearish: price < fast < slow < trend, negative slope
        """
        if fast <= 0 or slow <= 0:
            return TrendDirection.NEUTRAL

        if price > fast > slow and (trend <= 0 or slow > trend) and slope > 0.1:
            return TrendDirection.STRONG_BULLISH
        elif fast > slow and price > slow:
            return TrendDirection.BULLISH
        elif price < fast < slow and (trend <= 0 or slow < trend) and slope < -0.1:
            return TrendDirection.STRONG_BEARISH
        elif fast < slow and price < slow:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _combine_signals(self, signal: SwingSignal):
        """
        Combine multi-timeframe signals into a final trade decision.

        Rules:
        - Primary trend comes from 4h (or highest available timeframe)
        - 1h must confirm the direction
        - RSI acts as a filter (don't buy overbought, don't sell oversold)
        - ATR sets dynamic TP/SL
        """
        trends = {}
        for tf, cs in signal.signals.items():
            trends[tf] = cs

        # Get primary trend (prefer 4h)
        primary = trends.get("4h", trends.get("1h"))
        secondary = trends.get("1h", trends.get("4h"))

        if not primary:
            return

        signal.primary_trend = primary.trend

        # Check alignment
        bullish_trends = {TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH}
        bearish_trends = {TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH}

        primary_bull = primary.trend in bullish_trends
        primary_bear = primary.trend in bearish_trends
        secondary_bull = secondary.trend in bullish_trends if secondary else False
        secondary_bear = secondary.trend in bearish_trends if secondary else False

        signal.trend_aligned = (primary_bull and secondary_bull) or (primary_bear and secondary_bear)

        # RSI filter
        signal.is_oversold = signal.rsi_1h < self.cc.rsi_oversold or signal.rsi_4h < self.cc.rsi_oversold
        signal.is_overbought = signal.rsi_1h > self.cc.rsi_overbought or signal.rsi_4h > self.cc.rsi_overbought

        # ATR-based TP/SL (use 1h ATR for swing trades)
        atr_source = trends.get("1h", primary)
        if atr_source.atr > 0:
            signal.atr_tp_distance = atr_source.atr * self.cc.atr_tp_multiplier
            signal.atr_sl_distance = atr_source.atr * self.cc.atr_sl_multiplier
            signal.atr_tp_pct = signal.atr_tp_distance / atr_source.last_close * 100
            signal.atr_sl_pct = signal.atr_sl_distance / atr_source.last_close * 100

        # Trade suggestion
        confidence = 0.0

        if primary_bull:
            if signal.is_overbought:
                signal.suggested_side = None  # Don't buy overbought
            else:
                signal.suggested_side = "BUY"
                confidence += 0.3
                if signal.trend_aligned:
                    confidence += 0.3
                if signal.is_oversold:
                    confidence += 0.2  # Buy the dip in an uptrend
                if primary.trend == TrendDirection.STRONG_BULLISH:
                    confidence += 0.1
                if any(cs.is_volume_surge for cs in signal.signals.values()):
                    confidence += 0.1

        elif primary_bear:
            if signal.is_oversold:
                signal.suggested_side = None  # Don't sell oversold
            else:
                signal.suggested_side = "SELL"
                confidence += 0.3
                if signal.trend_aligned:
                    confidence += 0.3
                if signal.is_overbought:
                    confidence += 0.2  # Sell the rally in a downtrend
                if primary.trend == TrendDirection.STRONG_BEARISH:
                    confidence += 0.1
                if any(cs.is_volume_surge for cs in signal.signals.values()):
                    confidence += 0.1

        signal.confidence = min(confidence, 1.0)

        # Require minimum confidence
        if signal.confidence < 0.3:
            signal.suggested_side = None

    # ─────────────────────────────────────────
    # TECHNICAL INDICATORS
    # ─────────────────────────────────────────

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(data) < period:
            return data.copy()

        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range."""
        if len(highs) < period + 1:
            return 0.0

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        return float(np.mean(tr[-period:]))

    def get_signal_summary(self, signal: SwingSignal) -> str:
        """Human-readable summary."""
        lines = [f"  ── Candle Analysis: {signal.symbol} ──"]

        for tf, cs in sorted(signal.signals.items()):
            vol_flag = " VOL_SURGE" if cs.is_volume_surge else ""
            lines.append(
                f"    {tf}: {cs.trend.value} | EMA {cs.fast_ema:.2f}/{cs.slow_ema:.2f} "
                f"| RSI={cs.rsi:.1f} | ATR={cs.atr_pct:.2f}% "
                f"| slope={cs.ema_slope:+.3f}%{vol_flag}"
            )

        aligned = "ALIGNED" if signal.trend_aligned else "DIVERGENT"
        lines.append(
            f"    Combined: {signal.primary_trend.value} ({aligned}) | "
            f"conf={signal.confidence:.2f} | "
            f"suggestion={signal.suggested_side or 'HOLD'}"
        )

        if signal.atr_tp_pct > 0:
            lines.append(
                f"    ATR targets: TP={signal.atr_tp_pct:.2f}% | SL={signal.atr_sl_pct:.2f}%"
            )

        return "\n".join(lines)
