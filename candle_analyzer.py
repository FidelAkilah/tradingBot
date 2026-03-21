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
    adx: float = 0.0                # Average Directional Index (trend strength)
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

    # Dynamic TP/SL from ATR (fee-adjusted)
    atr_tp_distance: float = 0.0    # In price units
    atr_sl_distance: float = 0.0
    atr_tp_pct: float = 0.0         # As percentage (includes fee compensation)
    atr_sl_pct: float = 0.0         # As percentage (includes fee on losing side)
    raw_tp_pct: float = 0.0         # Pre-fee TP %
    raw_sl_pct: float = 0.0         # Pre-fee SL %
    fee_cost_pct: float = 0.0       # Round-trip fee as % of position
    post_fee_rr: float = 0.0        # Risk-reward ratio after fees

    # RSI context
    rsi_1h: float = 50.0
    rsi_4h: float = 50.0
    is_oversold: bool = False
    is_overbought: bool = False

    # ADX trend strength
    adx_1h: float = 0.0
    adx_4h: float = 0.0
    adx_blocked: bool = False       # True if ADX < ranging threshold


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

        # ADX
        adx = self._adx(highs, lows, closes, self.cc.adx_period)

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
            adx=adx,
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

        Scoring (restructured for 0.55 min threshold):
          Base candle trend (non-neutral):           +0.25
          Both timeframes aligned:                   +0.25
          RSI context confirmation:                  +0.15
          Strong trend (STRONG_BULLISH/BEARISH):     +0.10
          Volume surge (>1.5x average):              +0.10
          ADX filter pass (>25):                     +0.10
          (OB imbalance bonus applied in liquidity_analyzer: +0.05)

        A trade needs at minimum: base trend + timeframe alignment + one more signal.
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

        # Populate ADX fields
        sig_1h = trends.get("1h")
        sig_4h = trends.get("4h")
        signal.adx_1h = sig_1h.adx if sig_1h else 0.0
        signal.adx_4h = sig_4h.adx if sig_4h else 0.0

        # ADX hard block: if primary timeframe ADX < ranging threshold, reject
        primary_adx = sig_4h.adx if sig_4h else (sig_1h.adx if sig_1h else 0.0)
        if primary_adx > 0 and primary_adx < self.cc.adx_ranging_threshold:
            signal.adx_blocked = True
            signal.suggested_side = None
            signal.confidence = 0.0
            logger.debug(
                f"[{signal.symbol}] ADX block: {primary_adx:.1f} < "
                f"{self.cc.adx_ranging_threshold} (ranging/choppy market)"
            )
            return

        # Check alignment
        bullish_trends = {TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH}
        bearish_trends = {TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH}

        primary_bull = primary.trend in bullish_trends
        primary_bear = primary.trend in bearish_trends
        secondary_bull = secondary.trend in bullish_trends if secondary else False
        secondary_bear = secondary.trend in bearish_trends if secondary else False

        signal.trend_aligned = (primary_bull and secondary_bull) or (primary_bear and secondary_bear)

        # RSI context flags
        signal.is_oversold = signal.rsi_1h < self.cc.rsi_oversold or signal.rsi_4h < self.cc.rsi_oversold
        signal.is_overbought = signal.rsi_1h > self.cc.rsi_overbought or signal.rsi_4h > self.cc.rsi_overbought

        # Hard-block at EXTREME RSI when BOTH timeframes agree
        extreme_oversold = signal.rsi_1h < 20 and signal.rsi_4h < 25
        extreme_overbought = signal.rsi_1h > 80 and signal.rsi_4h > 75

        # --- Confidence scoring (new weights) ---
        confidence = 0.0

        if primary_bull:
            if extreme_overbought:
                signal.suggested_side = None
            else:
                signal.suggested_side = "BUY"
                confidence += 0.25  # Base candle trend
                if signal.trend_aligned:
                    confidence += 0.25  # Both timeframes aligned
                # RSI context: buying oversold dip in uptrend
                if signal.is_oversold:
                    confidence += 0.15
                # Strong trend bonus
                if primary.trend == TrendDirection.STRONG_BULLISH:
                    confidence += 0.10
                # Volume surge
                if any(cs.is_volume_surge for cs in signal.signals.values()):
                    confidence += 0.10
                # ADX trending bonus
                if primary_adx >= self.cc.adx_trending_threshold:
                    confidence += 0.10

        elif primary_bear:
            if extreme_oversold:
                signal.suggested_side = None
            else:
                signal.suggested_side = "SELL"
                confidence += 0.25  # Base candle trend
                if signal.trend_aligned:
                    confidence += 0.25  # Both timeframes aligned
                # RSI context: selling overbought rally in downtrend
                if signal.is_overbought:
                    confidence += 0.15
                # Strong trend bonus
                if primary.trend == TrendDirection.STRONG_BEARISH:
                    confidence += 0.10
                # Volume surge
                if any(cs.is_volume_surge for cs in signal.signals.values()):
                    confidence += 0.10
                # ADX trending bonus
                if primary_adx >= self.cc.adx_trending_threshold:
                    confidence += 0.10

        signal.confidence = min(confidence, 1.0)

        # Require minimum confidence of 0.55 (was 0.30)
        # This means: base trend (0.25) + alignment (0.25) + at least one more signal
        if signal.confidence < 0.55:
            signal.suggested_side = None

        # ATR-based TP/SL with fee adjustment
        atr_source = trends.get("1h", primary)
        if atr_source.atr > 0:
            if signal.trend_aligned and signal.confidence >= 0.6:
                tp_mult = self.cc.atr_tp_multiplier * 1.25  # 2.5x ATR
                sl_mult = self.cc.atr_sl_multiplier * 0.8   # 0.8x ATR
            else:
                tp_mult = self.cc.atr_tp_multiplier          # 2x ATR
                sl_mult = self.cc.atr_sl_multiplier           # 1x ATR

            raw_tp_distance = atr_source.atr * tp_mult
            raw_sl_distance = atr_source.atr * sl_mult
            raw_tp_pct = raw_tp_distance / atr_source.last_close * 100
            raw_sl_pct = raw_sl_distance / atr_source.last_close * 100

            # Fee-aware adjustment:
            # Round-trip fee cost = 2 * fee_rate * leverage (as % of margin)
            # In price terms: fee_cost_pct = 2 * fee_rate (as % of notional)
            tc = self.config.trading
            fc = self.config.futures
            fee_cost_pct = 2.0 * tc.fee_rate  # e.g., 2 * 0.04 = 0.08% of notional

            # Adjust TP upward to ensure net profit after fees
            # Adjust SL to account for fee drag on losing side
            effective_tp_pct = raw_tp_pct + fee_cost_pct
            effective_sl_pct = raw_sl_pct + fee_cost_pct

            # Post-fee R:R calculation
            net_tp_gain = raw_tp_pct - fee_cost_pct  # What you actually keep on a win
            net_sl_loss = raw_sl_pct + fee_cost_pct   # What you actually lose (SL + fees)
            post_fee_rr = net_tp_gain / net_sl_loss if net_sl_loss > 0 else 0.0

            signal.raw_tp_pct = raw_tp_pct
            signal.raw_sl_pct = raw_sl_pct
            signal.fee_cost_pct = fee_cost_pct
            signal.post_fee_rr = post_fee_rr

            # Use fee-adjusted distances for actual TP/SL targets
            signal.atr_tp_distance = atr_source.last_close * effective_tp_pct / 100.0
            signal.atr_sl_distance = atr_source.last_close * effective_sl_pct / 100.0
            signal.atr_tp_pct = effective_tp_pct
            signal.atr_sl_pct = effective_sl_pct

            # Reject if post-fee R:R is too low
            if post_fee_rr < tc.min_post_fee_rr and signal.suggested_side:
                logger.debug(
                    f"[{signal.symbol}] Post-fee R:R too low: {post_fee_rr:.2f} < "
                    f"{tc.min_post_fee_rr:.1f} (raw_tp={raw_tp_pct:.2f}%, "
                    f"raw_sl={raw_sl_pct:.2f}%, fees={fee_cost_pct:.2f}%)"
                )
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

    @staticmethod
    def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """
        Average Directional Index — measures trend strength (not direction).

        ADX > 25 = trending market (good for swing trades)
        ADX < 20 = ranging/choppy market (avoid trading)
        ADX 20-25 = weak trend (proceed with caution)

        Computed using Wilder's smoothing (NumPy, no TA-Lib).
        """
        n = len(highs)
        if n < period * 2 + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's smoothing (exponential with alpha = 1/period)
        alpha = 1.0 / period

        atr_smooth = np.zeros(len(tr))
        plus_dm_smooth = np.zeros(len(plus_dm))
        minus_dm_smooth = np.zeros(len(minus_dm))

        # Seed with SMA of first `period` values
        atr_smooth[period - 1] = np.mean(tr[:period])
        plus_dm_smooth[period - 1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period - 1] = np.mean(minus_dm[:period])

        for i in range(period, len(tr)):
            atr_smooth[i] = atr_smooth[i - 1] * (1 - alpha) + tr[i] * alpha
            plus_dm_smooth[i] = plus_dm_smooth[i - 1] * (1 - alpha) + plus_dm[i] * alpha
            minus_dm_smooth[i] = minus_dm_smooth[i - 1] * (1 - alpha) + minus_dm[i] * alpha

        # +DI and -DI (from period-1 onward where smoothed values are valid)
        valid = slice(period - 1, None)
        atr_v = atr_smooth[valid]
        # Avoid division by zero
        atr_safe = np.where(atr_v > 0, atr_v, 1.0)
        plus_di = 100.0 * plus_dm_smooth[valid] / atr_safe
        minus_di = 100.0 * minus_dm_smooth[valid] / atr_safe

        # DX
        di_sum = plus_di + minus_di
        di_sum_safe = np.where(di_sum > 0, di_sum, 1.0)
        dx = 100.0 * np.abs(plus_di - minus_di) / di_sum_safe

        if len(dx) < period:
            return 0.0

        # ADX = Wilder-smoothed DX
        adx = np.zeros(len(dx))
        adx[period - 1] = np.mean(dx[:period])
        for i in range(period, len(dx)):
            adx[i] = adx[i - 1] * (1 - alpha) + dx[i] * alpha

        return float(adx[-1])

    def get_signal_summary(self, signal: SwingSignal) -> str:
        """Human-readable summary."""
        lines = [f"  ── Candle Analysis: {signal.symbol} ──"]

        for tf, cs in sorted(signal.signals.items()):
            vol_flag = " VOL_SURGE" if cs.is_volume_surge else ""
            lines.append(
                f"    {tf}: {cs.trend.value} | EMA {cs.fast_ema:.2f}/{cs.slow_ema:.2f} "
                f"| RSI={cs.rsi:.1f} | ATR={cs.atr_pct:.2f}% | ADX={cs.adx:.1f} "
                f"| slope={cs.ema_slope:+.3f}%{vol_flag}"
            )

        aligned = "ALIGNED" if signal.trend_aligned else "DIVERGENT"
        adx_status = "BLOCKED" if signal.adx_blocked else f"1h={signal.adx_1h:.1f}/4h={signal.adx_4h:.1f}"
        lines.append(
            f"    Combined: {signal.primary_trend.value} ({aligned}) | "
            f"conf={signal.confidence:.2f} | ADX={adx_status} | "
            f"suggestion={signal.suggested_side or 'HOLD'}"
        )

        if signal.atr_tp_pct > 0:
            lines.append(
                f"    ATR targets: TP={signal.atr_tp_pct:.2f}% | SL={signal.atr_sl_pct:.2f}% "
                f"(raw TP={signal.raw_tp_pct:.2f}% SL={signal.raw_sl_pct:.2f}%) "
                f"| fees={signal.fee_cost_pct:.3f}% | post-fee R:R={signal.post_fee_rr:.2f}"
            )

        return "\n".join(lines)
