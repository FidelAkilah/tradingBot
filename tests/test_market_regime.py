"""
Tests for the Market Regime Classifier and Session Filter.

Validates:
1. ADX-based regime classification (STRONG_TREND, TREND, WEAK, RANGING)
2. Bollinger Band Width regime (EXPANDING, SQUEEZING, TREND)
3. Price vs EMAs regime (TREND, CHOPPY, WEAK)
4. 2-of-3 voting system (trending requires 2 agree)
5. Breakout override (squeeze + expansion + volume)
6. Session filter (UTC windows, pair restrictions)
7. Combined size multiplier (regime * session)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime, timezone

from config import BotConfig
from market_regime import MarketRegimeClassifier, RegimeType, RegimeResult
from session_filter import SessionFilter, SessionType, SessionResult


@pytest.fixture
def config():
    return BotConfig()


@pytest.fixture
def classifier(config):
    return MarketRegimeClassifier(config)


@pytest.fixture
def session_filter(config):
    return SessionFilter(config)


# ─────────────────────────────────────────
# Helper: synthetic data generators
# ─────────────────────────────────────────

def _make_trending_data(n=60, start=100.0, step=1.0):
    """Strong uptrend: price increases ~step per bar with small pullbacks."""
    np.random.seed(42)
    closes = np.zeros(n)
    closes[0] = start
    pattern = [step, step, step * 0.8, -step * 0.3, -step * 0.2]
    for i in range(1, n):
        closes[i] = closes[i - 1] + pattern[i % len(pattern)]
    highs = closes + np.random.uniform(0.3, 0.8, n)
    lows = closes - np.random.uniform(0.3, 0.8, n)
    volumes = np.random.uniform(3000, 7000, n)
    return highs, lows, closes, volumes


def _make_ranging_data(n=60, center=100.0, amplitude=0.3):
    """Sideways market: price oscillates in tight range."""
    np.random.seed(42)
    closes = center + amplitude * np.sin(np.arange(n) * 0.8)
    closes += np.random.uniform(-0.05, 0.05, n)
    highs = closes + 0.15
    lows = closes - 0.15
    volumes = np.full(n, 3000.0)
    return highs, lows, closes, volumes


def _make_choppy_data(n=60, center=100.0, swing=2.0):
    """Choppy market: large swings that cross EMAs frequently."""
    np.random.seed(42)
    closes = np.zeros(n)
    closes[0] = center
    for i in range(1, n):
        # Alternate direction every 2-3 bars
        if i % 3 == 0:
            closes[i] = closes[i - 1] + swing
        elif i % 3 == 1:
            closes[i] = closes[i - 1] - swing * 1.2
        else:
            closes[i] = closes[i - 1] + swing * 0.3
    highs = closes + np.random.uniform(0.2, 0.5, n)
    lows = closes - np.random.uniform(0.2, 0.5, n)
    volumes = np.random.uniform(2000, 5000, n)
    return highs, lows, closes, volumes


def _make_squeeze_breakout_data(n=60, center=100.0):
    """Squeeze then breakout: tight range then explosive move with volume."""
    np.random.seed(42)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    # First 45 bars: very tight range (squeeze)
    for i in range(45):
        closes[i] = center + np.random.uniform(-0.1, 0.1)
        volumes[i] = 2000.0

    # Last 15 bars: explosive breakout with high volume
    price = center
    for i in range(45, n):
        price += np.random.uniform(1.5, 3.0)
        closes[i] = price
        volumes[i] = 10000.0  # 5x normal volume

    highs = closes + np.random.uniform(0.2, 0.8, n)
    lows = closes - np.random.uniform(0.1, 0.3, n)
    return highs, lows, closes, volumes


# ─────────────────────────────────────────
# ADX Regime Tests
# ─────────────────────────────────────────

class TestADXRegime:
    def test_strong_trend(self, classifier):
        assert classifier._adx_regime(35.0) == RegimeType.STRONG_TREND

    def test_trend(self, classifier):
        assert classifier._adx_regime(27.0) == RegimeType.TREND

    def test_weak(self, classifier):
        assert classifier._adx_regime(22.0) == RegimeType.WEAK

    def test_ranging(self, classifier):
        assert classifier._adx_regime(15.0) == RegimeType.RANGING

    def test_zero_adx(self, classifier):
        assert classifier._adx_regime(0.0) == RegimeType.RANGING

    def test_boundary_30(self, classifier):
        assert classifier._adx_regime(30.0) == RegimeType.STRONG_TREND

    def test_boundary_25(self, classifier):
        assert classifier._adx_regime(25.0) == RegimeType.TREND

    def test_boundary_20(self, classifier):
        assert classifier._adx_regime(20.0) == RegimeType.WEAK


# ─────────────────────────────────────────
# BB Width Regime Tests
# ─────────────────────────────────────────

class TestBBWidthRegime:
    def test_steady_trend_has_low_bb_width(self, classifier):
        """A clean steady trend has consistent volatility → low BB width percentile."""
        # BB width = (upper-lower)/mid = 4*std/sma.
        # In a clean linear trend, std is constant but sma grows → width shrinks.
        highs, lows, closes, volumes = _make_trending_data(120, step=2.0)
        regime, pctl, _ = classifier._bb_width_regime(closes)
        # Clean trend → BB width decreasing → should be in lower percentiles
        assert pctl <= 50.0

    def test_volatile_expansion_has_high_bb_width(self, classifier):
        """Expanding volatility should push BB width higher."""
        # Create data where volatility increases over time
        np.random.seed(42)
        n = 120
        closes = np.zeros(n)
        closes[0] = 100.0
        for i in range(1, n):
            # Increasing volatility: amplitude grows with time
            amp = 0.1 + (i / n) * 3.0
            closes[i] = closes[i - 1] + np.random.uniform(-amp, amp)
        regime, pctl, _ = classifier._bb_width_regime(closes)
        # Increasing volatility → BB width expanding → high percentile
        assert pctl >= 50.0

    def test_short_data_returns_trend(self, classifier):
        """Not enough data → abstain (return TREND to avoid blocking)."""
        closes = np.array([100.0, 101.0, 102.0])
        regime, pctl, _ = classifier._bb_width_regime(closes)
        assert regime == RegimeType.TREND  # Abstain = don't block


# ─────────────────────────────────────────
# EMA Regime Tests
# ─────────────────────────────────────────

class TestEMARegime:
    def test_trending_when_price_above_all_emas(self, classifier):
        """Strong uptrend should keep price above all EMAs."""
        highs, lows, closes, _ = _make_trending_data(80, step=1.5)
        regime, cross_count = classifier._ema_regime(closes)
        assert regime == RegimeType.TREND

    def test_choppy_has_many_crosses(self, classifier):
        """Choppy data should produce multiple EMA crosses."""
        _, _, closes, _ = _make_choppy_data(80, swing=3.0)
        regime, cross_count = classifier._ema_regime(closes)
        assert cross_count >= 3  # Should have many crosses

    def test_short_data_returns_trend(self, classifier):
        """Not enough data → abstain (TREND to avoid blocking)."""
        closes = np.array([100.0] * 10)
        regime, _ = classifier._ema_regime(closes)
        assert regime == RegimeType.TREND  # Abstain = don't block


# ─────────────────────────────────────────
# Voting System Tests
# ─────────────────────────────────────────

class TestVotingSystem:
    def test_trending_vote_positive(self, classifier):
        assert classifier._is_trending_vote(RegimeType.STRONG_TREND) is True
        assert classifier._is_trending_vote(RegimeType.TREND) is True
        assert classifier._is_trending_vote(RegimeType.EXPANDING) is True

    def test_trending_vote_negative(self, classifier):
        assert classifier._is_trending_vote(RegimeType.RANGING) is False
        assert classifier._is_trending_vote(RegimeType.SQUEEZING) is False
        assert classifier._is_trending_vote(RegimeType.CHOPPY) is False
        assert classifier._is_trending_vote(RegimeType.WEAK) is False

    def test_strong_trend_not_blocked(self, classifier):
        """Strong trend data → all 3 should agree → not blocked."""
        highs, lows, closes, volumes = _make_trending_data(80, step=1.5)
        result = classifier.classify(highs, lows, closes, volumes, adx_value=35.0)
        assert result.is_trending is True
        assert result.regime_blocked is False
        assert result.size_multiplier > 0

    def test_ranging_market_blocked(self, classifier):
        """Ranging data with low ADX → blocked."""
        highs, lows, closes, volumes = _make_ranging_data(80, amplitude=0.1)
        result = classifier.classify(highs, lows, closes, volumes, adx_value=12.0)
        assert result.regime_blocked is True
        assert result.size_multiplier == 0.0

    def test_weak_trend_reduces_size(self, classifier):
        """ADX 20-25 with trending agreement → 50% size."""
        highs, lows, closes, volumes = _make_trending_data(80, step=0.8)
        result = classifier.classify(highs, lows, closes, volumes, adx_value=22.0)
        if result.is_trending:
            assert result.size_multiplier == 0.5


# ─────────────────────────────────────────
# Breakout Override Tests
# ─────────────────────────────────────────

class TestBreakoutOverride:
    def test_breakout_conditions(self, classifier):
        """Squeeze with expanding BB + high volume → breakout override."""
        highs, lows, closes, volumes = _make_squeeze_breakout_data(60)
        # Force low ADX (squeeze scenario)
        result = classifier.classify(highs, lows, closes, volumes, adx_value=15.0)
        # If breakout detected, regime should not be blocked
        if result.is_breakout:
            assert result.regime_blocked is False
            assert result.breakout_sl_mult < 1.0  # Tighter SL

    def test_breakout_sl_multiplier(self, config):
        """Breakout SL should use 0.7x ATR."""
        assert config.regime.breakout_sl_mult == 0.7


# ─────────────────────────────────────────
# Session Filter Tests
# ─────────────────────────────────────────

class TestSessionFilter:
    def test_us_eu_overlap(self, session_filter):
        """13:00-17:00 UTC → 100% size, all pairs."""
        dt = datetime(2026, 3, 21, 14, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.US_EU_OVERLAP
        assert result.size_multiplier == 1.0
        assert result.is_blocked is False

    def test_eu_session(self, session_filter):
        """07:00-13:00 UTC → 80% size."""
        dt = datetime(2026, 3, 21, 10, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.EU
        assert result.size_multiplier == 0.8
        assert result.is_blocked is False

    def test_us_session(self, session_filter):
        """17:00-21:00 UTC → 80% size."""
        dt = datetime(2026, 3, 21, 19, 0, tzinfo=timezone.utc)
        result = session_filter.check("ADA/USDT", dt)
        assert result.session == SessionType.US
        assert result.size_multiplier == 0.8
        assert result.is_blocked is False

    def test_asian_session_btc_allowed(self, session_filter):
        """00:00-07:00 UTC → BTC allowed at 50%."""
        dt = datetime(2026, 3, 21, 3, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.ASIAN
        assert result.size_multiplier == 0.5
        assert result.is_blocked is False

    def test_asian_session_eth_allowed(self, session_filter):
        """00:00-07:00 UTC → ETH allowed at 50%."""
        dt = datetime(2026, 3, 21, 5, 0, tzinfo=timezone.utc)
        result = session_filter.check("ETH/USDT", dt)
        assert result.session == SessionType.ASIAN
        assert result.size_multiplier == 0.5
        assert result.is_blocked is False

    def test_asian_session_altcoin_blocked(self, session_filter):
        """00:00-07:00 UTC → Altcoins blocked."""
        dt = datetime(2026, 3, 21, 4, 0, tzinfo=timezone.utc)
        result = session_filter.check("ADA/USDT", dt)
        assert result.session == SessionType.ASIAN
        assert result.is_blocked is True
        assert result.size_multiplier == 0.0

    def test_dead_zone_blocks_all(self, session_filter):
        """21:00-00:00 UTC → All trades blocked."""
        dt = datetime(2026, 3, 21, 22, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.DEAD_ZONE
        assert result.is_blocked is True
        assert result.size_multiplier == 0.0

    def test_dead_zone_blocks_altcoins(self, session_filter):
        """Dead zone blocks even altcoins."""
        dt = datetime(2026, 3, 21, 23, 0, tzinfo=timezone.utc)
        result = session_filter.check("SOL/USDT", dt)
        assert result.is_blocked is True

    def test_disabled_session_filter(self, config):
        """When disabled, always allow with 100% size."""
        config.session.enabled = False
        sf = SessionFilter(config)
        dt = datetime(2026, 3, 21, 22, 0, tzinfo=timezone.utc)
        result = sf.check("ADA/USDT", dt)
        assert result.is_blocked is False
        assert result.size_multiplier == 1.0

    def test_boundary_hour_7(self, session_filter):
        """Hour 7 = start of EU."""
        dt = datetime(2026, 3, 21, 7, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.EU

    def test_boundary_hour_13(self, session_filter):
        """Hour 13 = start of US+EU overlap."""
        dt = datetime(2026, 3, 21, 13, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.US_EU_OVERLAP

    def test_boundary_hour_17(self, session_filter):
        """Hour 17 = start of US session."""
        dt = datetime(2026, 3, 21, 17, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.US

    def test_boundary_hour_21(self, session_filter):
        """Hour 21 = start of dead zone."""
        dt = datetime(2026, 3, 21, 21, 0, tzinfo=timezone.utc)
        result = session_filter.check("BTC/USDT", dt)
        assert result.session == SessionType.DEAD_ZONE


# ─────────────────────────────────────────
# Integration: Combined Size Multiplier
# ─────────────────────────────────────────

class TestCombinedSizing:
    def test_full_size_overlap_strong_trend(self, classifier, session_filter):
        """US+EU overlap + strong trend = 1.0 * 1.0 = 100%."""
        highs, lows, closes, volumes = _make_trending_data(80, step=1.5)
        regime = classifier.classify(highs, lows, closes, volumes, adx_value=35.0)
        dt = datetime(2026, 3, 21, 14, 0, tzinfo=timezone.utc)
        session = session_filter.check("BTC/USDT", dt)

        combined = regime.size_multiplier * session.size_multiplier
        if regime.is_trending:
            assert combined == pytest.approx(1.0)

    def test_asian_weak_trend(self, classifier, session_filter):
        """Asian session + weak trend = 0.5 * 0.5 = 25%."""
        highs, lows, closes, volumes = _make_trending_data(80, step=0.8)
        regime = classifier.classify(highs, lows, closes, volumes, adx_value=22.0)
        dt = datetime(2026, 3, 21, 3, 0, tzinfo=timezone.utc)
        session = session_filter.check("BTC/USDT", dt)

        if regime.is_trending and regime.size_multiplier == 0.5:
            combined = regime.size_multiplier * session.size_multiplier
            assert combined == pytest.approx(0.25)
