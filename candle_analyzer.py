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
from candle_patterns import PatternDetector, PatternScanResult, evaluate_patterns_for_signal
from divergence import DivergenceDetector, DivergenceScanResult, evaluate_divergence_for_signal
from levels import LevelDetector, LevelAnalysis
from market_regime import MarketRegimeClassifier, RegimeResult, RegimeType
from session_filter import SessionFilter, SessionResult, SessionType
from volume_analysis import VolumeAnalyzer, VolumeAnalysisResult, FundingRateAnalyzer, FundingRateResult, OpenInterestAnalyzer, OpenInterestResult

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

    # MACD
    macd_line: float = 0.0          # EMA(fast) - EMA(slow)
    macd_signal_line: float = 0.0   # EMA of MACD line
    macd_histogram: float = 0.0     # MACD - Signal
    macd_histogram_prev: float = 0.0  # Previous candle's histogram
    macd_crossover_bars: int = -1   # Bars since last MACD/signal crossover (-1 = none)


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

    # Market regime
    regime: str = "ranging"         # Combined regime label
    regime_blocked: bool = False    # True if regime says don't trade
    regime_size_mult: float = 1.0   # Position size multiplier from regime
    regime_is_breakout: bool = False
    regime_breakout_sl_mult: float = 1.0
    regime_adx_regime: str = "ranging"
    regime_bb_regime: str = "ranging"
    regime_ema_regime: str = "ranging"
    regime_bb_width_pctl: float = 0.0

    # Session filter
    session: str = "dead_zone"      # Current session label
    session_blocked: bool = False   # True if session blocks this pair
    session_size_mult: float = 1.0  # Position size multiplier from session

    # Daily trend filter
    daily_trend: TrendDirection = TrendDirection.NEUTRAL
    daily_rsi: float = 50.0
    daily_aligned: bool = False     # True if 4h trend matches daily trend
    daily_blocked: bool = False     # True if daily trend opposes signal direction

    # 15-minute entry timing
    entry_15m_ready: bool = True    # True if 15m pullback detected or timed out
    entry_15m_rsi: float = 50.0     # Current 15m RSI
    entry_15m_pullback_seen: bool = False  # Did the 15m chart show a pullback?

    # Support/Resistance
    sr_analysis: Optional['LevelAnalysis'] = None
    sr_at_support: bool = False     # Near support level (good for longs)
    sr_at_resistance: bool = False  # Near resistance level (good for shorts)
    sr_tp_blocked: bool = False     # TP target blocked by a strong level
    sr_adjusted_tp_pct: float = 0.0 # TP adjusted to respect S/R (0 = no adjustment)

    # Candlestick patterns
    pattern_scans: List = field(default_factory=list)  # PatternScanResult per TF
    pattern_confirming: List[str] = field(default_factory=list)
    pattern_contradicting: List[str] = field(default_factory=list)
    pattern_has_doji: bool = False
    pattern_blocked: bool = False   # Reversal pattern contradicts trade direction

    # RSI divergence
    divergence_scan: Optional['DivergenceScanResult'] = None
    divergence_confirming: List[str] = field(default_factory=list)
    divergence_contradicting: List[str] = field(default_factory=list)
    divergence_blocked: bool = False  # Regular div AGAINST direction = hard block
    divergence_block_reason: str = ""

    # Volume analysis
    volume_analysis: Optional['VolumeAnalysisResult'] = None
    obv_trend: str = "neutral"
    obv_divergence: bool = False
    volume_pressure: str = "neutral"
    poc_price: float = 0.0

    # Funding rate
    funding_rate: Optional['FundingRateResult'] = None
    funding_rate_value: float = 0.0
    funding_extreme: bool = False
    funding_blocked: bool = False      # Extreme funding blocks direction

    # Open interest
    oi_result: Optional['OpenInterestResult'] = None
    oi_change_pct: float = 0.0
    oi_conviction: str = "neutral"
    oi_divergence: bool = False

    # MACD
    macd_confirms: bool = False         # MACD aligns with trade direction
    macd_crossover_fresh: bool = False  # Fresh MACD crossover within lookback
    macd_diverges: bool = False         # MACD opposes trade direction

    # BB Squeeze + Keltner Channel
    squeeze_active: bool = False        # BB inside Keltner Channel (true squeeze)
    squeeze_releasing: bool = False     # Squeeze just released with confirmation
    squeeze_direction: str = ""         # "BUY" or "SELL" from band break
    squeeze_sl_mult: float = 1.0       # SL override for squeeze release
    squeeze_tp1_mult: float = 1.0      # TP1 override for squeeze release


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

        # Market regime classifier, session filter, S/R detector
        self._regime = MarketRegimeClassifier(config)
        self._session = SessionFilter(config)
        self._levels = LevelDetector(config)

        # Candlestick pattern + RSI divergence detectors
        self._patterns = PatternDetector(
            doji_body_pct=self.cc.pattern_doji_body_pct,
            pin_wick_ratio=self.cc.pattern_pin_wick_ratio,
            marubozu_body_pct=self.cc.pattern_marubozu_body_pct,
        )
        self._divergence = DivergenceDetector(
            lookback=self.cc.divergence_lookback,
            pivot_lookback=self.cc.divergence_pivot_lookback,
            rsi_period=self.cc.rsi_period,
            oversold=self.cc.divergence_oversold,
            overbought=self.cc.divergence_overbought,
        )

        # Volume, Funding Rate, and Open Interest analyzers
        self._volume_analyzer = VolumeAnalyzer(config)
        self._funding_analyzer = FundingRateAnalyzer(config)
        self._oi_analyzer = OpenInterestAnalyzer(config)

        # Cache candle data to avoid refetching
        self._candle_cache: Dict[str, Dict[str, list]] = defaultdict(dict)
        self._last_fetch: Dict[str, float] = {}
        self._fetch_cooldown = 60.0  # Refetch candles every 60s at most

        # Daily candle cache (300s TTL — daily candles change slowly)
        self._daily_cache: Dict[str, list] = {}
        self._daily_fetch_ts: Dict[str, float] = {}
        self._daily_signal_cache: Dict[str, CandleSignal] = {}

        # 15m candle cache (60s TTL)
        self._15m_cache: Dict[str, list] = {}
        self._15m_fetch_ts: Dict[str, float] = {}

        # 15m pullback state tracking per symbol
        self._15m_pullback_state: Dict[str, dict] = {}

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

    async def fetch_daily(self, exchange, symbol: str) -> Optional[list]:
        """
        Fetch daily candles with 300s TTL cache.
        Returns OHLCV list or None.
        """
        if not self.cc.daily_enabled:
            return None

        now = time.time()
        last = self._daily_fetch_ts.get(symbol, 0)
        if now - last < self.cc.daily_cache_ttl and symbol in self._daily_cache:
            return self._daily_cache[symbol]

        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol, "1d", limit=self.cc.daily_lookback
            )
            if ohlcv and len(ohlcv) >= 15:
                self._daily_cache[symbol] = ohlcv
                self._daily_fetch_ts[symbol] = now
                # Pre-compute daily signal
                self._daily_signal_cache[symbol] = self._analyze_timeframe(
                    "1d", ohlcv, now
                )
                logger.debug(f"[{symbol}] Fetched {len(ohlcv)} daily candles")
                return ohlcv
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to fetch daily candles: {e}")
        return self._daily_cache.get(symbol)

    async def fetch_15m(self, exchange, symbol: str) -> Optional[list]:
        """
        Fetch 15m candles with 60s TTL cache.
        Returns OHLCV list or None.
        """
        if not self.cc.entry_15m_enabled:
            return None

        now = time.time()
        last = self._15m_fetch_ts.get(symbol, 0)
        if now - last < self.cc.entry_15m_cache_ttl and symbol in self._15m_cache:
            return self._15m_cache[symbol]

        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol, "15m", limit=self.cc.entry_15m_lookback
            )
            if ohlcv and len(ohlcv) >= 15:
                self._15m_cache[symbol] = ohlcv
                self._15m_fetch_ts[symbol] = now
                logger.debug(f"[{symbol}] Fetched {len(ohlcv)} 15m candles")
                return ohlcv
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to fetch 15m candles: {e}")
        return self._15m_cache.get(symbol)

    def get_daily_signal(self, symbol: str) -> Optional[CandleSignal]:
        """Return cached daily CandleSignal if available."""
        return self._daily_signal_cache.get(symbol)

    def analyze(self, symbol: str, candles: Dict[str, list], timestamp: float) -> SwingSignal:
        """
        Run multi-timeframe analysis and produce a SwingSignal.
        """
        # Cache raw candle data for regime classifier to use
        self._candle_cache[symbol] = candles

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

        # MACD
        macd_line_val = 0.0
        macd_signal_val = 0.0
        macd_hist_val = 0.0
        macd_hist_prev_val = 0.0
        macd_xover_bars = -1

        if self.cc.macd_enabled and len(closes) >= self.cc.macd_slow_period + self.cc.macd_signal_period:
            macd_fast_ema = self._ema(closes, self.cc.macd_fast_period)
            macd_slow_ema = self._ema(closes, self.cc.macd_slow_period)
            macd_raw = macd_fast_ema - macd_slow_ema
            macd_sig = self._ema(macd_raw, self.cc.macd_signal_period)
            macd_hist = macd_raw - macd_sig

            macd_line_val = float(macd_raw[-1])
            macd_signal_val = float(macd_sig[-1])
            macd_hist_val = float(macd_hist[-1])
            macd_hist_prev_val = float(macd_hist[-2]) if len(macd_hist) >= 2 else 0.0

            # Find most recent MACD/signal crossover
            lookback = min(self.cc.macd_crossover_lookback, len(macd_raw) - 1)
            for b in range(lookback):
                idx = len(macd_raw) - 1 - b
                prev_idx = idx - 1
                if prev_idx < 0:
                    break
                curr_above = macd_raw[idx] > macd_sig[idx]
                prev_above = macd_raw[prev_idx] > macd_sig[prev_idx]
                if curr_above != prev_above:
                    macd_xover_bars = b
                    break

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
            macd_line=macd_line_val,
            macd_signal_line=macd_signal_val,
            macd_histogram=macd_hist_val,
            macd_histogram_prev=macd_hist_prev_val,
            macd_crossover_bars=macd_xover_bars,
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

        # --- Market Regime Classification ---
        # Use primary timeframe data for regime detection
        primary_ohlcv = signal.signals.get("4h", signal.signals.get("1h"))
        if primary_ohlcv:
            # Get raw arrays from the candle data stored in the analyze() method
            # We reconstruct from the signal cache — the caller passes ohlcv data
            pass  # Regime is computed below after we gather arrays

        regime_result = self._classify_regime(signal)
        signal.regime = regime_result.combined_regime.value
        signal.regime_blocked = regime_result.regime_blocked
        signal.regime_size_mult = regime_result.size_multiplier
        signal.regime_is_breakout = regime_result.is_breakout
        signal.regime_breakout_sl_mult = regime_result.breakout_sl_mult
        signal.regime_adx_regime = regime_result.adx_regime.value
        signal.regime_bb_regime = regime_result.bb_regime.value
        signal.regime_ema_regime = regime_result.ema_regime.value
        signal.regime_bb_width_pctl = regime_result.bb_width_pctl

        if regime_result.regime_blocked:
            signal.suggested_side = None
            signal.confidence = 0.0
            logger.debug(
                f"[{signal.symbol}] Regime block: {regime_result.combined_regime.value} "
                f"(ADX={regime_result.adx_regime.value}, BB={regime_result.bb_regime.value}, "
                f"EMA={regime_result.ema_regime.value})"
            )
            return

        # --- Session Filter ---
        session_result = self._session.check(signal.symbol)
        signal.session = session_result.session.value
        signal.session_blocked = session_result.is_blocked
        signal.session_size_mult = session_result.size_multiplier

        if session_result.is_blocked:
            signal.suggested_side = None
            signal.confidence = 0.0
            logger.debug(
                f"[{signal.symbol}] Session block: {session_result.block_reason}"
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

        # --- Daily Trend Gate ---
        daily_sig = self._daily_signal_cache.get(signal.symbol)
        if daily_sig and self.cc.daily_enabled:
            signal.daily_trend = daily_sig.trend
            signal.daily_rsi = daily_sig.rsi

            daily_bull = daily_sig.trend in bullish_trends
            daily_bear = daily_sig.trend in bearish_trends
            daily_neutral = daily_sig.trend == TrendDirection.NEUTRAL

            # Check alignment with 4h trend
            if primary_bull and daily_bull:
                signal.daily_aligned = True
                confidence += 0.10  # Daily alignment bonus
            elif primary_bear and daily_bear:
                signal.daily_aligned = True
                confidence += 0.10  # Daily alignment bonus

            # Gate: block counter-trend trades
            if signal.suggested_side == "BUY" and daily_bear:
                signal.daily_blocked = True
                signal.suggested_side = None
                confidence = 0.0
                logger.debug(
                    f"[{signal.symbol}] Daily trend GATE: blocked BUY "
                    f"(daily={daily_sig.trend.value})"
                )
            elif signal.suggested_side == "SELL" and daily_bull:
                signal.daily_blocked = True
                signal.suggested_side = None
                confidence = 0.0
                logger.debug(
                    f"[{signal.symbol}] Daily trend GATE: blocked SELL "
                    f"(daily={daily_sig.trend.value})"
                )
            elif daily_neutral and signal.suggested_side:
                # Neutral daily: allow both but penalize
                confidence -= self.cc.daily_neutral_penalty

        # --- Candlestick Pattern Integration ---
        if self.cc.patterns_enabled and signal.suggested_side:
            pattern_scans = []
            for tf, ohlcv_data in self._candle_cache.get(signal.symbol, {}).items():
                if ohlcv_data and len(ohlcv_data) >= 3:
                    scan = self._patterns.scan(ohlcv_data, timeframe=tf)
                    pattern_scans.append(scan)
            signal.pattern_scans = pattern_scans

            pat_eval = evaluate_patterns_for_signal(pattern_scans, signal.suggested_side)
            signal.pattern_confirming = pat_eval["confirming_patterns"]
            signal.pattern_contradicting = pat_eval["contradicting_patterns"]
            signal.pattern_has_doji = pat_eval["has_doji"]
            confidence += pat_eval["confidence_adj"]

            if pat_eval["contradicting_patterns"]:
                logger.debug(
                    f"[{signal.symbol}] Pattern contradict: "
                    f"{pat_eval['contradicting_patterns']} "
                    f"(conf adj {pat_eval['confidence_adj']:+.2f})"
                )
            if pat_eval["confirming_patterns"]:
                logger.debug(
                    f"[{signal.symbol}] Pattern confirm: "
                    f"{pat_eval['confirming_patterns']}"
                )

        # --- RSI Divergence Integration ---
        if self.cc.divergence_enabled and signal.suggested_side:
            # Use 4h candles for divergence (most reliable for swing)
            div_ohlcv = self._candle_cache.get(signal.symbol, {}).get(
                "4h", self._candle_cache.get(signal.symbol, {}).get("1h")
            )
            if div_ohlcv and len(div_ohlcv) >= self.cc.divergence_lookback + self.cc.rsi_period:
                div_scan = self._divergence.scan(div_ohlcv, timeframe="4h")
                signal.divergence_scan = div_scan

                div_eval = evaluate_divergence_for_signal(div_scan, signal.suggested_side)
                signal.divergence_confirming = div_eval["confirming"]
                signal.divergence_contradicting = div_eval["contradicting"]
                confidence += div_eval["confidence_adj"]

                if div_eval["blocked"]:
                    signal.divergence_blocked = True
                    signal.divergence_block_reason = div_eval["block_reason"]
                    signal.suggested_side = None
                    confidence = 0.0
                    logger.debug(
                        f"[{signal.symbol}] Divergence BLOCK: {div_eval['block_reason']}"
                    )
                elif div_eval["confirming"]:
                    logger.debug(
                        f"[{signal.symbol}] Divergence confirm: "
                        f"{div_eval['confirming']} (conf adj {div_eval['confidence_adj']:+.2f})"
                    )

        # --- Volume Analysis Integration ---
        if signal.suggested_side:
            vol_ohlcv = self._candle_cache.get(signal.symbol, {}).get(
                "4h", self._candle_cache.get(signal.symbol, {}).get("1h")
            )
            if vol_ohlcv and len(vol_ohlcv) >= 20:
                v_opens = np.array([c[1] for c in vol_ohlcv], dtype=np.float64)
                v_highs = np.array([c[2] for c in vol_ohlcv], dtype=np.float64)
                v_lows = np.array([c[3] for c in vol_ohlcv], dtype=np.float64)
                v_closes = np.array([c[4] for c in vol_ohlcv], dtype=np.float64)
                v_volumes = np.array([c[5] for c in vol_ohlcv], dtype=np.float64)

                vol_result = self._volume_analyzer.analyze(
                    v_highs, v_lows, v_closes, v_volumes, v_opens,
                )
                signal.volume_analysis = vol_result
                signal.obv_trend = vol_result.obv_trend
                signal.obv_divergence = vol_result.obv_divergence
                signal.volume_pressure = vol_result.volume_pressure
                signal.poc_price = vol_result.poc_price

                # Only apply OBV confirm bonus when direction aligns with trade
                obv_aligned = (
                    (signal.suggested_side == "BUY" and vol_result.obv_trend == "bullish")
                    or (signal.suggested_side == "SELL" and vol_result.obv_trend == "bearish")
                )
                pressure_aligned = (
                    (signal.suggested_side == "BUY" and vol_result.volume_pressure == "buying")
                    or (signal.suggested_side == "SELL" and vol_result.volume_pressure == "selling")
                )

                if obv_aligned:
                    confidence += self.config.volume.obv_confirm_bonus
                if vol_result.obv_divergence:
                    confidence += self.config.volume.obv_divergence_penalty
                if pressure_aligned:
                    confidence += self.config.volume.buy_sell_pressure_bonus

        # --- Funding Rate Integration ---
        if signal.suggested_side and self.config.volume.funding_enabled:
            fr_result = self._funding_analyzer.analyze(signal.symbol)
            signal.funding_rate = fr_result
            signal.funding_rate_value = fr_result.current_rate
            signal.funding_extreme = fr_result.is_extreme_positive or fr_result.is_extreme_negative

            if signal.suggested_side == "BUY" and fr_result.block_long:
                signal.funding_blocked = True
                confidence -= self.config.volume.funding_extreme_penalty
                logger.debug(
                    f"[{signal.symbol}] Funding rate penalty: extreme positive "
                    f"({fr_result.current_rate:.4f}%) — risky to go long"
                )
            elif signal.suggested_side == "SELL" and fr_result.block_short:
                signal.funding_blocked = True
                confidence -= self.config.volume.funding_extreme_penalty
                logger.debug(
                    f"[{signal.symbol}] Funding rate penalty: extreme negative "
                    f"({fr_result.current_rate:.4f}%) — risky to go short"
                )
            elif fr_result.is_persistent:
                # Persistent funding in same direction confirms trend
                confidence += self.config.volume.funding_persistent_bonus

        # --- Open Interest Integration ---
        if signal.suggested_side and self.config.volume.oi_enabled:
            oi_result = self._oi_analyzer.analyze(
                signal.symbol,
                current_price=primary.last_close if primary else 0.0,
                price_change_pct=primary.ema_slope if primary else 0.0,
            )
            signal.oi_result = oi_result
            signal.oi_change_pct = oi_result.oi_change_pct
            signal.oi_conviction = oi_result.conviction
            signal.oi_divergence = oi_result.is_divergence
            confidence += oi_result.confidence_adj

        # --- MACD Integration ---
        if self.cc.macd_enabled and signal.suggested_side:
            macd_confirmed_count = 0
            macd_crossover_found = False
            macd_diverge_count = 0

            for tf, cs in signal.signals.items():
                if cs.macd_line == 0.0 and cs.macd_signal_line == 0.0:
                    continue

                if signal.suggested_side == "BUY":
                    # LONG: MACD > signal AND histogram positive and growing
                    if (cs.macd_line > cs.macd_signal_line
                            and cs.macd_histogram > 0
                            and cs.macd_histogram > cs.macd_histogram_prev):
                        macd_confirmed_count += 1
                    elif cs.macd_line < cs.macd_signal_line:
                        macd_diverge_count += 1
                elif signal.suggested_side == "SELL":
                    # SHORT: MACD < signal AND histogram negative and shrinking
                    if (cs.macd_line < cs.macd_signal_line
                            and cs.macd_histogram < 0
                            and cs.macd_histogram < cs.macd_histogram_prev):
                        macd_confirmed_count += 1
                    elif cs.macd_line > cs.macd_signal_line:
                        macd_diverge_count += 1

                # Fresh crossover aligned with trade direction
                if cs.macd_crossover_bars >= 0:
                    if ((signal.suggested_side == "BUY" and cs.macd_line > cs.macd_signal_line)
                            or (signal.suggested_side == "SELL" and cs.macd_line < cs.macd_signal_line)):
                        macd_crossover_found = True

            if macd_confirmed_count > 0:
                signal.macd_confirms = True
                confidence += self.cc.macd_confirm_bonus
            if macd_crossover_found:
                signal.macd_crossover_fresh = True
                confidence += self.cc.macd_crossover_bonus
            if macd_diverge_count > 0 and macd_confirmed_count == 0:
                signal.macd_diverges = True
                confidence -= self.cc.macd_diverge_penalty

        # --- BB Squeeze + Keltner Channel Integration ---
        if self.cc.squeeze_enabled and signal.suggested_side:
            sq_ohlcv = self._candle_cache.get(signal.symbol, {}).get(
                "4h", self._candle_cache.get(signal.symbol, {}).get("1h")
            )
            if sq_ohlcv and len(sq_ohlcv) >= 30:
                sq_closes = np.array([c[4] for c in sq_ohlcv], dtype=np.float64)
                sq_highs = np.array([c[2] for c in sq_ohlcv], dtype=np.float64)
                sq_lows = np.array([c[3] for c in sq_ohlcv], dtype=np.float64)
                sq_volumes = np.array([c[5] for c in sq_ohlcv], dtype=np.float64)

                sq_active, sq_releasing, sq_dir = self._detect_squeeze(
                    sq_closes, sq_highs, sq_lows, sq_volumes,
                )
                signal.squeeze_active = sq_active

                if sq_releasing and sq_dir == signal.suggested_side:
                    signal.squeeze_releasing = True
                    signal.squeeze_direction = sq_dir
                    signal.squeeze_sl_mult = self.cc.squeeze_release_sl_mult
                    signal.squeeze_tp1_mult = self.cc.squeeze_release_tp1_mult
                    confidence += self.cc.squeeze_release_bonus
                    logger.debug(
                        f"[{signal.symbol}] Squeeze release {sq_dir}: "
                        f"+{self.cc.squeeze_release_bonus} conf, "
                        f"SL×{self.cc.squeeze_release_sl_mult}, TP1×{self.cc.squeeze_release_tp1_mult}"
                    )

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

            # Breakout override: tighter SL (0.7x ATR)
            if regime_result.is_breakout:
                sl_mult *= regime_result.breakout_sl_mult

            # Squeeze release override: tighter SL + wider TP
            if signal.squeeze_releasing:
                sl_mult *= signal.squeeze_sl_mult
                tp_mult = max(tp_mult, signal.squeeze_tp1_mult)

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

        # --- S/R Level Integration ---
        if self.cc.sr_enabled and signal.suggested_side and atr_source.atr > 0:
            ohlcv_for_levels = self._candle_cache.get(signal.symbol, {}).get("1h")
            if ohlcv_for_levels and len(ohlcv_for_levels) >= 20:
                tp_price = None
                if signal.suggested_side == "BUY":
                    tp_price = atr_source.last_close * (1 + signal.atr_tp_pct / 100.0)
                elif signal.suggested_side == "SELL":
                    tp_price = atr_source.last_close * (1 - signal.atr_tp_pct / 100.0)

                sr = self._levels.analyze(
                    symbol=signal.symbol,
                    ohlcv_1h=ohlcv_for_levels,
                    current_price=atr_source.last_close,
                    atr=atr_source.atr,
                    suggested_side=signal.suggested_side,
                    tp_price=tp_price,
                    timestamp=signal.timestamp,
                )
                signal.sr_analysis = sr
                signal.sr_at_support = sr.at_support
                signal.sr_at_resistance = sr.at_resistance
                signal.sr_tp_blocked = sr.tp_blocked

                # S/R confidence bonus: near support for longs, near resistance for shorts
                if signal.suggested_side == "BUY" and sr.at_support:
                    signal.confidence = min(signal.confidence + self.cc.sr_confidence_bonus, 1.0)
                elif signal.suggested_side == "SELL" and sr.at_resistance:
                    signal.confidence = min(signal.confidence + self.cc.sr_confidence_bonus, 1.0)

                # TP blocked by a strong level: skip trade
                if sr.tp_blocked:
                    logger.debug(
                        f"[{signal.symbol}] S/R block: TP target blocked by strong level"
                    )
                    signal.suggested_side = None

    # ─────────────────────────────────────────
    # 15m ENTRY TIMING
    # ─────────────────────────────────────────

    def check_15m_entry(self, symbol: str, suggested_side: str) -> Tuple[bool, float]:
        """
        Check 15m chart for pullback entry timing.

        For LONG: wait for 15m RSI to dip below 45 then cross back above.
        For SHORT: wait for 15m RSI to spike above 55 then cross back below.

        Returns:
            (ready: bool, rsi_15m: float)
        """
        if not self.cc.entry_15m_enabled:
            return True, 50.0

        ohlcv = self._15m_cache.get(symbol)
        if not ohlcv or len(ohlcv) < 15:
            return True, 50.0  # No data, allow entry

        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        current_rsi = self._rsi(closes, self.cc.rsi_period)

        # Track pullback state
        state = self._15m_pullback_state.get(symbol, {})
        candles_waited = state.get("candles_waited", 0)
        dip_seen = state.get("dip_seen", False)
        active_side = state.get("side")

        # Reset if side changed
        if active_side != suggested_side:
            state = {"side": suggested_side, "candles_waited": 0, "dip_seen": False}
            candles_waited = 0
            dip_seen = False

        candles_waited += 1
        state["candles_waited"] = candles_waited

        if suggested_side == "BUY":
            if current_rsi < self.cc.entry_15m_rsi_long_dip:
                state["dip_seen"] = True
            if state.get("dip_seen") and current_rsi >= self.cc.entry_15m_rsi_long_dip:
                # Dip and recovery — pullback entry found
                self._15m_pullback_state[symbol] = {}
                return True, current_rsi
        elif suggested_side == "SELL":
            if current_rsi > self.cc.entry_15m_rsi_short_spike:
                state["dip_seen"] = True
            if state.get("dip_seen") and current_rsi <= self.cc.entry_15m_rsi_short_spike:
                self._15m_pullback_state[symbol] = {}
                return True, current_rsi

        # Timeout: enter at market after max_wait_candles
        if candles_waited >= self.cc.entry_15m_max_wait_candles:
            self._15m_pullback_state[symbol] = {}
            return True, current_rsi

        self._15m_pullback_state[symbol] = state
        return False, current_rsi

    # ─────────────────────────────────────────
    # REGIME HELPER
    # ─────────────────────────────────────────

    def _classify_regime(self, signal: SwingSignal) -> RegimeResult:
        """Run regime classification using cached OHLCV data from the signal."""
        # We need raw candle arrays — get from _candle_cache
        symbol = signal.symbol
        candle_data = self._candle_cache.get(symbol, {})

        # Prefer 4h for regime, fall back to 1h
        ohlcv = candle_data.get("4h", candle_data.get("1h"))

        if not ohlcv or len(ohlcv) < 20:
            # No cached data — use ADX only for a basic regime call
            primary_adx = signal.adx_4h if signal.adx_4h > 0 else signal.adx_1h
            return self._regime.classify(
                np.array([100.0]), np.array([99.0]), np.array([100.0]),
                np.array([1000.0]), adx_value=primary_adx,
            )

        highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv], dtype=np.float64)
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        volumes = np.array([c[5] for c in ohlcv], dtype=np.float64)

        primary_adx = signal.adx_4h if signal.adx_4h > 0 else signal.adx_1h

        return self._regime.classify(highs, lows, closes, volumes, adx_value=primary_adx)

    # ─────────────────────────────────────────
    # BB SQUEEZE + KELTNER CHANNEL
    # ─────────────────────────────────────────

    def _detect_squeeze(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ):
        """
        Detect Bollinger Band squeeze using Keltner Channel.

        True squeeze: BB inside KC (BB_upper < KC_upper AND BB_lower > KC_lower).
        Squeeze release: was squeezing, now BB outside KC + price breaking bands + volume surge.

        Returns: (squeeze_active, squeeze_releasing, squeeze_direction)
        """
        cc = self.cc
        if not cc.squeeze_enabled:
            return False, False, ""

        bb_period = cc.squeeze_bb_period
        bb_std_mult = cc.squeeze_bb_std
        kc_ema_period = cc.squeeze_kc_ema_period
        kc_atr_period = cc.squeeze_kc_atr_period
        kc_atr_mult = cc.squeeze_kc_atr_mult

        n = len(closes)
        min_len = max(bb_period, kc_ema_period, kc_atr_period + 1) + 2
        if n < min_len:
            return False, False, ""

        # KC center line (EMA)
        kc_ema = self._ema(closes, kc_ema_period)

        def _bb_bands_at(end_idx):
            """BB upper/lower using candles ending at end_idx (exclusive)."""
            window = closes[end_idx - bb_period:end_idx]
            sma = float(np.mean(window))
            std = float(np.std(window, ddof=1))
            return sma + bb_std_mult * std, sma - bb_std_mult * std

        def _kc_atr_at(idx):
            """ATR for Keltner Channel at a specific index."""
            start = max(0, idx - kc_atr_period)
            h = highs[start:idx + 1]
            lo = lows[start:idx + 1]
            c = closes[start:idx + 1]
            if len(h) < 2:
                return 0.0
            tr = np.maximum(
                h[1:] - lo[1:],
                np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1]))
            )
            return float(np.mean(tr[-min(kc_atr_period, len(tr)):]))

        # Current candle: BB and KC bands
        bb_upper, bb_lower = _bb_bands_at(n)
        kc_center = float(kc_ema[n - 1])
        kc_atr = _kc_atr_at(n - 1)
        kc_upper = kc_center + kc_atr_mult * kc_atr
        kc_lower = kc_center - kc_atr_mult * kc_atr

        squeeze_now = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        if squeeze_now:
            return True, False, ""

        # Check if any of the last 3 candles were in a squeeze (for release detection)
        was_squeezing = False
        for offset in range(1, 4):
            check_end = n - offset
            check_idx = n - 1 - offset
            if check_end < bb_period or check_idx < kc_atr_period:
                break
            bb_u, bb_l = _bb_bands_at(check_end)
            kc_c = float(kc_ema[check_idx])
            kc_a = _kc_atr_at(check_idx)
            kc_u = kc_c + kc_atr_mult * kc_a
            kc_l = kc_c - kc_atr_mult * kc_a
            if bb_u < kc_u and bb_l > kc_l:
                was_squeezing = True
                break

        # Squeeze release: was squeezing, now BB outside KC
        if was_squeezing:
            # Volume confirmation
            avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
            vol_ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0
            has_volume = vol_ratio >= cc.squeeze_release_volume_mult

            # Price breaking BB direction
            current_close = float(closes[-1])
            if current_close > bb_upper and has_volume:
                return False, True, "BUY"
            elif current_close < bb_lower and has_volume:
                return False, True, "SELL"

        return False, False, ""

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

        # Patterns
        if signal.pattern_confirming or signal.pattern_contradicting or signal.pattern_has_doji:
            pat_parts = []
            if signal.pattern_confirming:
                pat_parts.append(f"confirm={signal.pattern_confirming}")
            if signal.pattern_contradicting:
                pat_parts.append(f"contradict={signal.pattern_contradicting}")
            if signal.pattern_has_doji:
                pat_parts.append("DOJI")
            lines.append(f"    Patterns: {' | '.join(pat_parts)}")

        # Divergence
        if signal.divergence_blocked:
            lines.append(f"    Divergence: BLOCKED — {signal.divergence_block_reason}")
        elif signal.divergence_confirming:
            lines.append(f"    Divergence: confirm={signal.divergence_confirming}")

        # Volume analysis
        if signal.volume_analysis:
            va = signal.volume_analysis
            lines.append(
                f"    Volume: OBV={va.obv_trend} | pressure={va.volume_pressure} "
                f"| buy_ratio={va.buy_volume_ratio:.2f} | POC={va.poc_price:.2f}"
            )
            if va.obv_divergence:
                lines.append(f"    Volume: OBV DIVERGENCE detected")

        # Funding rate
        if signal.funding_rate and signal.funding_rate_value != 0.0:
            fr = signal.funding_rate
            status = "EXTREME" if signal.funding_extreme else "normal"
            lines.append(
                f"    Funding: {fr.current_rate:+.4f}% ({status})"
                f"{' BLOCKED' if signal.funding_blocked else ''}"
            )

        # Open interest
        if signal.oi_result and signal.oi_change_pct != 0.0:
            oi = signal.oi_result
            lines.append(
                f"    OI: change={oi.oi_change_pct:+.1f}% | "
                f"conviction={oi.conviction}"
                f"{' | DIVERGENCE' if oi.is_divergence else ''}"
            )

        # MACD
        if signal.macd_confirms or signal.macd_diverges or signal.macd_crossover_fresh:
            macd_parts = []
            if signal.macd_confirms:
                macd_parts.append("CONFIRMS")
            if signal.macd_crossover_fresh:
                macd_parts.append("FRESH_CROSSOVER")
            if signal.macd_diverges:
                macd_parts.append("DIVERGES")
            # Show MACD values from primary TF
            cs_4h = signal.signals.get("4h")
            cs_1h = signal.signals.get("1h")
            macd_cs = cs_4h if cs_4h else cs_1h
            if macd_cs:
                macd_parts.append(
                    f"MACD={macd_cs.macd_line:.4f} sig={macd_cs.macd_signal_line:.4f} "
                    f"hist={macd_cs.macd_histogram:.4f}"
                )
            lines.append(f"    MACD: {' | '.join(macd_parts)}")

        # BB Squeeze
        if signal.squeeze_active or signal.squeeze_releasing:
            if signal.squeeze_active:
                lines.append(f"    Squeeze: ACTIVE (BB inside KC)")
            if signal.squeeze_releasing:
                lines.append(
                    f"    Squeeze: RELEASE {signal.squeeze_direction} "
                    f"(SL×{signal.squeeze_sl_mult}, TP1×{signal.squeeze_tp1_mult})"
                )

        return "\n".join(lines)
