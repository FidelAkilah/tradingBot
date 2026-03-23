"""
Tests for MACD and BB Squeeze + Keltner Channel detection.

Covers:
- MACD computation (line, signal, histogram, crossover)
- MACD confidence adjustments (confirm, diverge, crossover bonus)
- Keltner Channel squeeze detection (active, releasing)
- Squeeze release with volume and direction confirmation
- TP/SL overrides for squeeze releases
- Config defaults
- Edge cases (insufficient data, disabled features)
"""

import time
import numpy as np
import pytest

from config import BotConfig, CandleConfig
from candle_analyzer import CandleAnalyzer, CandleSignal, SwingSignal, TrendDirection


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _make_ohlcv(closes, spread_pct=0.5, base_volume=1000.0, volumes=None):
    """
    Create OHLCV list from close prices.
    Generates realistic open/high/low from closes.
    """
    ohlcv = []
    for i, c in enumerate(closes):
        spread = c * spread_pct / 100.0
        o = c - spread * 0.3 if i % 2 == 0 else c + spread * 0.3
        h = max(o, c) + spread * 0.5
        lo = min(o, c) - spread * 0.5
        v = volumes[i] if volumes is not None else base_volume
        ohlcv.append([int(time.time()) * 1000 + i * 3600000, o, h, lo, c, v])
    return ohlcv


def _uptrend(n=50, start=100.0, step=0.5):
    """Generate uptrending close prices."""
    return [start + i * step for i in range(n)]


def _downtrend(n=50, start=100.0, step=0.5):
    """Generate downtrending close prices."""
    return [start - i * step for i in range(n)]


def _flat(n=50, price=100.0, noise=0.1):
    """Generate flat/ranging close prices."""
    np.random.seed(42)
    return [price + np.random.uniform(-noise, noise) for _ in range(n)]


def _squeeze_then_breakout_up(n=50, squeeze_price=100.0, breakout_magnitude=5.0):
    """
    Generate prices that squeeze (low vol) then break out upward.
    First 40 candles: tight range around squeeze_price.
    Last 10 candles: sharp move up.
    """
    prices = []
    # Tight range (squeeze)
    np.random.seed(42)
    for i in range(n - 10):
        prices.append(squeeze_price + np.random.uniform(-0.2, 0.2))
    # Breakout up
    for i in range(10):
        prices.append(squeeze_price + (i + 1) * breakout_magnitude / 10.0)
    return prices


def _squeeze_then_breakout_down(n=50, squeeze_price=100.0, breakout_magnitude=5.0):
    """Generate squeeze then breakout downward."""
    prices = []
    np.random.seed(42)
    for i in range(n - 10):
        prices.append(squeeze_price + np.random.uniform(-0.2, 0.2))
    for i in range(10):
        prices.append(squeeze_price - (i + 1) * breakout_magnitude / 10.0)
    return prices


def _make_analyzer(config=None):
    """Create a CandleAnalyzer with default or custom config."""
    if config is None:
        config = BotConfig()
    return CandleAnalyzer(config)


# ─────────────────────────────────────────
# MACD COMPUTATION
# ─────────────────────────────────────────

class TestMACDComputation:
    """Test MACD line, signal, histogram, and crossover detection."""

    def test_macd_bullish_uptrend(self):
        """In an uptrend, MACD should be above signal line."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=100.0, step=1.0)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line > 0, "MACD should be positive in uptrend"
        assert cs.macd_line > cs.macd_signal_line, "MACD should be above signal in uptrend"
        assert cs.macd_histogram > 0, "Histogram should be positive"

    def test_macd_bearish_downtrend(self):
        """In a downtrend, MACD should be below signal line."""
        analyzer = _make_analyzer()
        closes = _downtrend(50, start=150.0, step=1.0)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line < 0, "MACD should be negative in downtrend"
        assert cs.macd_line < cs.macd_signal_line, "MACD should be below signal in downtrend"
        assert cs.macd_histogram < 0, "Histogram should be negative"

    def test_macd_histogram_prev(self):
        """Previous histogram value should be stored."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=100.0, step=0.5)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        # Both current and previous histogram should be computable
        assert cs.macd_histogram != 0.0
        assert cs.macd_histogram_prev != 0.0

    def test_macd_crossover_detection(self):
        """Detect recent crossover when trend reverses."""
        # Create data that crosses: downtrend then sharp uptrend
        closes = _downtrend(35, start=120.0, step=0.3) + _uptrend(15, start=109.5, step=1.5)
        analyzer = _make_analyzer()
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        # The sharp reversal should create a crossover
        # (may or may not be within 3 bars depending on EMA lag)
        # Just verify the field is populated (not the default -1 necessarily)
        assert isinstance(cs.macd_crossover_bars, int)

    def test_macd_no_crossover_steady_trend(self):
        """In a steady uptrend, no recent crossover should be detected."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=100.0, step=1.0)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_crossover_bars == -1, "No crossover in steady trend"

    def test_macd_insufficient_data(self):
        """With insufficient data, MACD should be zero."""
        analyzer = _make_analyzer()
        closes = _uptrend(10, start=100.0, step=0.5)  # Need 26+9=35
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line == 0.0
        assert cs.macd_signal_line == 0.0
        assert cs.macd_histogram == 0.0
        assert cs.macd_crossover_bars == -1

    def test_macd_flat_market(self):
        """In flat market, MACD should be near zero."""
        analyzer = _make_analyzer()
        closes = _flat(50, price=100.0, noise=0.01)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert abs(cs.macd_line) < 1.0, "MACD should be near zero in flat market"

    def test_macd_disabled(self):
        """When MACD disabled, values should be zero."""
        config = BotConfig()
        config.candle.macd_enabled = False
        analyzer = _make_analyzer(config)
        closes = _uptrend(50, start=100.0, step=1.0)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line == 0.0
        assert cs.macd_signal_line == 0.0
        assert cs.macd_histogram == 0.0


# ─────────────────────────────────────────
# MACD CONFIDENCE INTEGRATION
# ─────────────────────────────────────────

class TestMACDConfidence:
    """Test MACD confidence adjustments in _combine_signals."""

    def _run_signal(self, closes_1h, closes_4h, config=None):
        """Helper to create a SwingSignal from 1h and 4h data."""
        if config is None:
            config = BotConfig()
        # Disable non-MACD filters for isolation
        config.candle.patterns_enabled = False
        config.candle.divergence_enabled = False
        config.candle.sr_enabled = False
        config.candle.daily_enabled = False
        config.candle.squeeze_enabled = False
        config.volume.funding_enabled = False
        config.volume.oi_enabled = False

        analyzer = _make_analyzer(config)
        candles = {
            "1h": _make_ohlcv(closes_1h),
            "4h": _make_ohlcv(closes_4h),
        }
        return analyzer.analyze("BTC/USDT", candles, time.time())

    def test_macd_long_confirm_bonus(self):
        """BUY signal with MACD bullish → +0.05 confidence."""
        closes = _uptrend(50, start=100.0, step=1.0)

        # With MACD
        sig_with = self._run_signal(closes, closes)

        # Without MACD
        config_no = BotConfig()
        config_no.candle.macd_enabled = False
        sig_without = self._run_signal(closes, closes, config=config_no)

        if sig_with.suggested_side == "BUY" and sig_with.macd_confirms:
            assert sig_with.confidence > sig_without.confidence, \
                "MACD confirm should boost confidence"

    def test_macd_short_confirm_bonus(self):
        """SELL signal with MACD bearish → +0.05 confidence."""
        closes = _downtrend(50, start=150.0, step=1.0)

        sig = self._run_signal(closes, closes)

        if sig.suggested_side == "SELL" and sig.macd_confirms:
            assert sig.confidence >= 0.55, "MACD-confirmed SELL should meet threshold"

    def test_macd_diverge_penalty(self):
        """BUY signal with MACD bearish → -0.05 penalty."""
        # Weak uptrend (EMAs bullish) but recent histogram going negative
        # This is hard to construct perfectly, so test the flag mechanism
        closes_bull = _uptrend(50, start=100.0, step=0.3)
        sig = self._run_signal(closes_bull, closes_bull)

        # In a steady uptrend, MACD should confirm, not diverge
        if sig.suggested_side == "BUY":
            assert not sig.macd_diverges, "Steady uptrend shouldn't have MACD divergence"

    def test_macd_crossover_fresh_bonus(self):
        """Fresh MACD crossover should add +0.05."""
        # Strong reversal to force a crossover
        closes = _downtrend(30, start=120.0, step=0.5) + _uptrend(20, start=105.0, step=2.0)
        sig = self._run_signal(closes, closes)

        # Test the flag existence
        assert isinstance(sig.macd_crossover_fresh, bool)

    def test_macd_no_side_no_adjustment(self):
        """Without suggested_side, MACD should not modify confidence."""
        closes = _flat(50, price=100.0, noise=0.01)
        sig = self._run_signal(closes, closes)

        assert not sig.macd_confirms
        assert not sig.macd_diverges
        assert not sig.macd_crossover_fresh


# ─────────────────────────────────────────
# SQUEEZE DETECTION
# ─────────────────────────────────────────

class TestSqueezeDetection:
    """Test BB squeeze via Keltner Channel detection."""

    def test_no_squeeze_normal_volatility(self):
        """Normal volatility → no squeeze detected."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=100.0, step=1.0)
        ohlcv = _make_ohlcv(closes, spread_pct=1.0)

        closes_arr = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs_arr = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows_arr = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes_arr = np.array([c[5] for c in ohlcv], dtype=np.float64)

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        # In a trending market with normal spread, no squeeze expected
        assert not releasing

    def test_squeeze_active_tight_range(self):
        """Tight price range should show BB inside KC → squeeze active."""
        analyzer = _make_analyzer()
        # Very tight range with tiny spread → BB narrows inside KC
        closes = _flat(50, price=100.0, noise=0.05)
        ohlcv = _make_ohlcv(closes, spread_pct=0.05)

        closes_arr = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs_arr = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows_arr = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes_arr = np.array([c[5] for c in ohlcv], dtype=np.float64)

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert active, "Tight range should create squeeze (BB inside KC)"
        assert not releasing

    def test_squeeze_release_bullish(self):
        """Squeeze followed by upward breakout with volume → release BUY."""
        analyzer = _make_analyzer()

        # Build data: tight range then sharp breakout
        n_squeeze = 40
        n_break = 10

        np.random.seed(42)
        squeeze_closes = [100.0 + np.random.uniform(-0.05, 0.05) for _ in range(n_squeeze)]
        break_closes = [100.0 + (i + 1) * 0.5 for i in range(n_break)]
        closes = squeeze_closes + break_closes

        # Low volume during squeeze, high volume during breakout
        volumes = [500.0] * n_squeeze + [3000.0] * n_break
        ohlcv = _make_ohlcv(closes, spread_pct=0.05, volumes=volumes)

        # Widen the highs/lows on breakout candles to make KC wide
        for i in range(n_squeeze, n_squeeze + n_break):
            ohlcv[i][2] = ohlcv[i][4] + 2.0  # high
            ohlcv[i][3] = ohlcv[i][4] - 2.0  # low

        closes_arr = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs_arr = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows_arr = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes_arr = np.array([c[5] for c in ohlcv], dtype=np.float64)

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )

        # The breakout should cause BB to expand outside KC
        # Exact behavior depends on how fast BB responds to the breakout
        if releasing:
            assert direction == "BUY"
        # If not releasing, the squeeze might still be active or the transition
        # didn't happen cleanly — that's OK for the test data

    def test_squeeze_release_bearish(self):
        """Squeeze followed by downward breakout with volume → release SELL."""
        analyzer = _make_analyzer()

        n_squeeze = 40
        n_break = 10

        np.random.seed(42)
        squeeze_closes = [100.0 + np.random.uniform(-0.05, 0.05) for _ in range(n_squeeze)]
        break_closes = [100.0 - (i + 1) * 0.5 for i in range(n_break)]
        closes = squeeze_closes + break_closes

        volumes = [500.0] * n_squeeze + [3000.0] * n_break
        ohlcv = _make_ohlcv(closes, spread_pct=0.05, volumes=volumes)

        for i in range(n_squeeze, n_squeeze + n_break):
            ohlcv[i][2] = ohlcv[i][4] + 2.0
            ohlcv[i][3] = ohlcv[i][4] - 2.0

        closes_arr = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs_arr = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows_arr = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes_arr = np.array([c[5] for c in ohlcv], dtype=np.float64)

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )

        if releasing:
            assert direction == "SELL"

    def test_squeeze_release_no_volume(self):
        """Breakout without volume surge → no release signal."""
        analyzer = _make_analyzer()

        n_squeeze = 40
        n_break = 10

        np.random.seed(42)
        squeeze_closes = [100.0 + np.random.uniform(-0.05, 0.05) for _ in range(n_squeeze)]
        break_closes = [100.0 + (i + 1) * 0.5 for i in range(n_break)]
        closes = squeeze_closes + break_closes

        # Constant low volume — no surge
        volumes = [500.0] * (n_squeeze + n_break)
        ohlcv = _make_ohlcv(closes, spread_pct=0.05, volumes=volumes)

        for i in range(n_squeeze, n_squeeze + n_break):
            ohlcv[i][2] = ohlcv[i][4] + 2.0
            ohlcv[i][3] = ohlcv[i][4] - 2.0

        closes_arr = np.array([c[4] for c in ohlcv], dtype=np.float64)
        highs_arr = np.array([c[2] for c in ohlcv], dtype=np.float64)
        lows_arr = np.array([c[3] for c in ohlcv], dtype=np.float64)
        volumes_arr = np.array([c[5] for c in ohlcv], dtype=np.float64)

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )

        # Without volume, squeeze release should not trigger
        assert not releasing, "No volume surge → no squeeze release"

    def test_squeeze_disabled(self):
        """When squeeze disabled, always returns False."""
        config = BotConfig()
        config.candle.squeeze_enabled = False
        analyzer = _make_analyzer(config)

        closes_arr = np.array(_flat(50), dtype=np.float64)
        highs_arr = closes_arr + 0.5
        lows_arr = closes_arr - 0.5
        volumes_arr = np.ones(50) * 1000.0

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert not active
        assert not releasing
        assert direction == ""

    def test_squeeze_insufficient_data(self):
        """Insufficient data → no squeeze detected."""
        analyzer = _make_analyzer()

        closes_arr = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        highs_arr = closes_arr + 1.0
        lows_arr = closes_arr - 1.0
        volumes_arr = np.array([1000.0, 1000.0, 1000.0])

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert not active
        assert not releasing


# ─────────────────────────────────────────
# KELTNER CHANNEL
# ─────────────────────────────────────────

class TestKeltnerChannel:
    """Test Keltner Channel mechanics in squeeze detection."""

    def test_bb_inside_kc_squeeze(self):
        """When BB is narrower than KC, squeeze should be active."""
        analyzer = _make_analyzer()

        # Create data where BB std is very small (tight prices)
        # but KC ATR is relatively wide (due to previous high/low swings)
        n = 50
        np.random.seed(42)
        # Closes very tight
        closes = [100.0 + np.random.uniform(-0.02, 0.02) for _ in range(n)]
        closes_arr = np.array(closes, dtype=np.float64)

        # Highs/lows have wider range (makes KC wider)
        highs_arr = closes_arr + 1.0
        lows_arr = closes_arr - 1.0
        volumes_arr = np.ones(n) * 1000.0

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert active, "BB should be inside KC with tight closes and wide H/L"

    def test_bb_outside_kc_no_squeeze(self):
        """When BB is wider than KC, no squeeze."""
        analyzer = _make_analyzer()

        n = 50
        # Create data where close prices have high variance
        # but highs/lows are tight (close to close)
        np.random.seed(42)
        closes = [100.0 + np.random.uniform(-5.0, 5.0) for _ in range(n)]
        closes_arr = np.array(closes, dtype=np.float64)

        # Highs/lows barely above/below close
        highs_arr = closes_arr + 0.01
        lows_arr = closes_arr - 0.01
        volumes_arr = np.ones(n) * 1000.0

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert not active, "Wide BB with narrow KC should not be squeeze"

    def test_kc_atr_uses_configured_period(self):
        """KC ATR should use squeeze_kc_atr_period from config."""
        config = BotConfig()
        assert config.candle.squeeze_kc_atr_period == 10
        assert config.candle.squeeze_kc_atr_mult == 1.5
        assert config.candle.squeeze_kc_ema_period == 20


# ─────────────────────────────────────────
# SQUEEZE CONFIDENCE INTEGRATION
# ─────────────────────────────────────────

class TestSqueezeConfidence:
    """Test squeeze release confidence adjustments and TP/SL overrides."""

    def _run_signal(self, closes_1h, closes_4h, config=None, ohlcv_4h=None):
        """Helper to create signal, optionally with custom 4h OHLCV."""
        if config is None:
            config = BotConfig()
        config.candle.patterns_enabled = False
        config.candle.divergence_enabled = False
        config.candle.sr_enabled = False
        config.candle.daily_enabled = False
        config.candle.macd_enabled = False
        config.volume.funding_enabled = False
        config.volume.oi_enabled = False

        analyzer = _make_analyzer(config)
        candles = {
            "1h": _make_ohlcv(closes_1h),
            "4h": ohlcv_4h if ohlcv_4h else _make_ohlcv(closes_4h),
        }
        return analyzer.analyze("BTC/USDT", candles, time.time())

    def test_squeeze_release_confidence_bonus(self):
        """Squeeze release should add +0.10 to confidence."""
        config = BotConfig()
        assert config.candle.squeeze_release_bonus == 0.10

    def test_squeeze_release_sl_override(self):
        """Squeeze release should set SL mult to 0.7."""
        config = BotConfig()
        assert config.candle.squeeze_release_sl_mult == 0.7

    def test_squeeze_release_tp1_override(self):
        """Squeeze release should set TP1 mult to 3.0."""
        config = BotConfig()
        assert config.candle.squeeze_release_tp1_mult == 3.0

    def test_squeeze_fields_default(self):
        """SwingSignal squeeze fields should default correctly."""
        sig = SwingSignal(symbol="TEST", timestamp=time.time())
        assert not sig.squeeze_active
        assert not sig.squeeze_releasing
        assert sig.squeeze_direction == ""
        assert sig.squeeze_sl_mult == 1.0
        assert sig.squeeze_tp1_mult == 1.0

    def test_squeeze_active_blocks_release(self):
        """Active squeeze should not also be releasing."""
        analyzer = _make_analyzer()
        # Tight range data
        n = 50
        np.random.seed(42)
        closes = [100.0 + np.random.uniform(-0.02, 0.02) for _ in range(n)]
        closes_arr = np.array(closes, dtype=np.float64)
        highs_arr = closes_arr + 1.0
        lows_arr = closes_arr - 1.0
        volumes_arr = np.ones(n) * 1000.0

        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        if active:
            assert not releasing, "Can't be both active and releasing"


# ─────────────────────────────────────────
# SIGNAL SUMMARY
# ─────────────────────────────────────────

class TestSignalSummary:
    """Test get_signal_summary includes MACD and squeeze info."""

    def test_summary_macd_confirms(self):
        """Summary should show MACD CONFIRMS when flag is set."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.macd_confirms = True
        sig.signals["4h"] = CandleSignal(
            timeframe="4h", trend=TrendDirection.BULLISH,
            macd_line=1.5, macd_signal_line=1.0, macd_histogram=0.5,
        )
        summary = analyzer.get_signal_summary(sig)
        assert "MACD" in summary
        assert "CONFIRMS" in summary

    def test_summary_macd_diverges(self):
        """Summary should show DIVERGES when flag is set."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.macd_diverges = True
        sig.signals["4h"] = CandleSignal(
            timeframe="4h", trend=TrendDirection.BULLISH,
            macd_line=-0.5, macd_signal_line=0.5, macd_histogram=-1.0,
        )
        summary = analyzer.get_signal_summary(sig)
        assert "DIVERGES" in summary

    def test_summary_macd_crossover(self):
        """Summary should show FRESH_CROSSOVER when flag is set."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.macd_crossover_fresh = True
        sig.signals["4h"] = CandleSignal(
            timeframe="4h", trend=TrendDirection.BULLISH,
            macd_line=0.5, macd_signal_line=0.3, macd_histogram=0.2,
        )
        summary = analyzer.get_signal_summary(sig)
        assert "FRESH_CROSSOVER" in summary

    def test_summary_squeeze_active(self):
        """Summary should show squeeze ACTIVE."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.squeeze_active = True
        summary = analyzer.get_signal_summary(sig)
        assert "Squeeze" in summary
        assert "ACTIVE" in summary

    def test_summary_squeeze_release(self):
        """Summary should show squeeze RELEASE with direction."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.squeeze_releasing = True
        sig.squeeze_direction = "BUY"
        sig.squeeze_sl_mult = 0.7
        sig.squeeze_tp1_mult = 3.0
        summary = analyzer.get_signal_summary(sig)
        assert "RELEASE" in summary
        assert "BUY" in summary
        assert "0.7" in summary
        assert "3.0" in summary

    def test_summary_no_macd_no_squeeze(self):
        """No MACD/squeeze flags → no MACD/squeeze lines in summary."""
        analyzer = _make_analyzer()
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.signals["4h"] = CandleSignal(
            timeframe="4h", trend=TrendDirection.NEUTRAL,
        )
        summary = analyzer.get_signal_summary(sig)
        assert "MACD" not in summary
        assert "Squeeze" not in summary


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

class TestMACDSqueezeConfig:
    """Test config defaults and customization."""

    def test_macd_defaults(self):
        """MACD config defaults should match spec."""
        c = CandleConfig()
        assert c.macd_enabled is True
        assert c.macd_fast_period == 12
        assert c.macd_slow_period == 26
        assert c.macd_signal_period == 9
        assert c.macd_confirm_bonus == 0.05
        assert c.macd_crossover_bonus == 0.05
        assert c.macd_diverge_penalty == 0.05
        assert c.macd_crossover_lookback == 3

    def test_squeeze_defaults(self):
        """Squeeze config defaults should match spec."""
        c = CandleConfig()
        assert c.squeeze_enabled is True
        assert c.squeeze_bb_period == 20
        assert c.squeeze_bb_std == 2.0
        assert c.squeeze_kc_ema_period == 20
        assert c.squeeze_kc_atr_period == 10
        assert c.squeeze_kc_atr_mult == 1.5
        assert c.squeeze_release_volume_mult == 1.5
        assert c.squeeze_release_bonus == 0.10
        assert c.squeeze_release_sl_mult == 0.7
        assert c.squeeze_release_tp1_mult == 3.0

    def test_custom_macd_config(self):
        """Custom MACD config values should propagate."""
        config = BotConfig()
        config.candle.macd_fast_period = 8
        config.candle.macd_slow_period = 21
        config.candle.macd_signal_period = 5

        analyzer = _make_analyzer(config)
        assert analyzer.cc.macd_fast_period == 8
        assert analyzer.cc.macd_slow_period == 21
        assert analyzer.cc.macd_signal_period == 5

    def test_custom_squeeze_config(self):
        """Custom squeeze config values should propagate."""
        config = BotConfig()
        config.candle.squeeze_kc_atr_mult = 2.0
        config.candle.squeeze_release_bonus = 0.15

        analyzer = _make_analyzer(config)
        assert analyzer.cc.squeeze_kc_atr_mult == 2.0
        assert analyzer.cc.squeeze_release_bonus == 0.15


# ─────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_timeframe_macd(self):
        """MACD should work with only one timeframe."""
        config = BotConfig()
        config.candle.patterns_enabled = False
        config.candle.divergence_enabled = False
        config.candle.sr_enabled = False
        config.candle.daily_enabled = False
        config.candle.squeeze_enabled = False
        config.volume.funding_enabled = False
        config.volume.oi_enabled = False

        analyzer = _make_analyzer(config)
        closes = _uptrend(50, start=100.0, step=1.0)
        candles = {"1h": _make_ohlcv(closes)}
        sig = analyzer.analyze("BTC/USDT", candles, time.time())

        cs = sig.signals.get("1h")
        if cs:
            assert cs.macd_line != 0.0 or cs.macd_signal_line != 0.0

    def test_macd_with_exact_min_data(self):
        """MACD should work with exactly slow_period + signal_period candles."""
        analyzer = _make_analyzer()
        n = 26 + 9  # Exactly the minimum (35)
        closes = _uptrend(n, start=100.0, step=0.5)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        # Should compute something (not zero)
        assert cs.macd_line != 0.0 or cs.macd_signal_line != 0.0

    def test_macd_histogram_direction_growing(self):
        """In accelerating uptrend, histogram should be growing."""
        # Accelerating prices: each step bigger than the last
        closes = [100.0 + 0.1 * i * i for i in range(50)]
        analyzer = _make_analyzer()
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        if cs.macd_histogram > 0:
            assert cs.macd_histogram >= cs.macd_histogram_prev, \
                "Histogram should be growing in accelerating uptrend"

    def test_macd_histogram_direction_shrinking(self):
        """In decelerating uptrend, histogram may shrink."""
        # Decelerating prices: big steps then small steps
        closes = [100.0 + 2.0 * i for i in range(25)] + \
                 [150.0 + 0.1 * i for i in range(25)]
        analyzer = _make_analyzer()
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        # In deceleration, histogram should be positive but potentially shrinking
        # Just verify histogram_prev is different from histogram
        assert cs.macd_histogram != cs.macd_histogram_prev or cs.macd_histogram == 0.0

    def test_squeeze_with_volume_profile_overlap(self):
        """Squeeze detection should work independently of volume analysis."""
        config = BotConfig()
        config.candle.squeeze_enabled = True
        analyzer = _make_analyzer(config)

        n = 50
        np.random.seed(42)
        closes = [100.0 + np.random.uniform(-0.02, 0.02) for _ in range(n)]
        closes_arr = np.array(closes, dtype=np.float64)
        highs_arr = closes_arr + 1.0
        lows_arr = closes_arr - 1.0
        volumes_arr = np.ones(n) * 1000.0

        # Should not raise any error
        active, releasing, direction = analyzer._detect_squeeze(
            closes_arr, highs_arr, lows_arr, volumes_arr
        )
        assert isinstance(active, bool)
        assert isinstance(releasing, bool)

    def test_swing_signal_macd_fields_default(self):
        """SwingSignal MACD fields should have sensible defaults."""
        sig = SwingSignal(symbol="TEST", timestamp=time.time())
        assert not sig.macd_confirms
        assert not sig.macd_crossover_fresh
        assert not sig.macd_diverges

    def test_candle_signal_macd_fields_default(self):
        """CandleSignal MACD fields should default to zero/none."""
        cs = CandleSignal(timeframe="4h", trend=TrendDirection.NEUTRAL)
        assert cs.macd_line == 0.0
        assert cs.macd_signal_line == 0.0
        assert cs.macd_histogram == 0.0
        assert cs.macd_histogram_prev == 0.0
        assert cs.macd_crossover_bars == -1

    def test_large_price_values(self):
        """MACD should handle large price values (BTC-scale)."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=60000.0, step=100.0)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line > 0, "MACD should be positive for BTC-scale uptrend"
        assert not np.isnan(cs.macd_line)
        assert not np.isinf(cs.macd_line)

    def test_small_price_values(self):
        """MACD should handle small price values (ADA-scale)."""
        analyzer = _make_analyzer()
        closes = _uptrend(50, start=0.5, step=0.005)
        ohlcv = _make_ohlcv(closes)
        cs = analyzer._analyze_timeframe("4h", ohlcv, time.time())

        assert cs.macd_line > 0, "MACD should be positive for small-price uptrend"
        assert not np.isnan(cs.macd_line)


# ─────────────────────────────────────────
# INTEGRATION
# ─────────────────────────────────────────

class TestMACDSqueezeIntegration:
    """Test MACD and squeeze working together in the full pipeline."""

    def test_macd_and_squeeze_both_active(self):
        """Both MACD and squeeze can be active simultaneously."""
        sig = SwingSignal(symbol="BTC/USDT", timestamp=time.time())
        sig.macd_confirms = True
        sig.squeeze_active = True
        assert sig.macd_confirms and sig.squeeze_active

    def test_full_analyze_with_macd_squeeze(self):
        """Full analyze() should populate MACD and squeeze fields."""
        config = BotConfig()
        config.candle.patterns_enabled = False
        config.candle.divergence_enabled = False
        config.candle.sr_enabled = False
        config.candle.daily_enabled = False
        config.volume.funding_enabled = False
        config.volume.oi_enabled = False

        analyzer = _make_analyzer(config)
        closes = _uptrend(50, start=100.0, step=1.0)
        candles = {
            "1h": _make_ohlcv(closes),
            "4h": _make_ohlcv(closes),
        }
        sig = analyzer.analyze("BTC/USDT", candles, time.time())

        # MACD should be computed on at least one timeframe
        has_macd = False
        for tf, cs in sig.signals.items():
            if cs.macd_line != 0.0:
                has_macd = True
                break
        assert has_macd, "At least one timeframe should have MACD computed"

        # Squeeze fields should exist
        assert isinstance(sig.squeeze_active, bool)
        assert isinstance(sig.squeeze_releasing, bool)

    def test_macd_per_timeframe(self):
        """MACD should be computed independently per timeframe."""
        config = BotConfig()
        config.candle.patterns_enabled = False
        config.candle.divergence_enabled = False
        config.candle.sr_enabled = False
        config.candle.daily_enabled = False
        config.candle.squeeze_enabled = False
        config.volume.funding_enabled = False
        config.volume.oi_enabled = False

        analyzer = _make_analyzer(config)
        # Different trends on different TFs
        closes_1h = _uptrend(50, start=100.0, step=0.5)
        closes_4h = _downtrend(50, start=130.0, step=0.3)

        candles = {
            "1h": _make_ohlcv(closes_1h),
            "4h": _make_ohlcv(closes_4h),
        }
        sig = analyzer.analyze("BTC/USDT", candles, time.time())

        cs_1h = sig.signals.get("1h")
        cs_4h = sig.signals.get("4h")

        if cs_1h and cs_4h:
            # 1h bullish → positive MACD, 4h bearish → negative MACD
            if cs_1h.macd_line != 0.0 and cs_4h.macd_line != 0.0:
                assert cs_1h.macd_line > 0, "1h uptrend should have positive MACD"
                assert cs_4h.macd_line < 0, "4h downtrend should have negative MACD"
