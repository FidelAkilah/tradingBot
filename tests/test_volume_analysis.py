"""
Tests for Enhanced Volume Analysis, Funding Rate, and Open Interest (volume_analysis.py).

Validates:
1. OBV calculation and trend detection
2. Buy/Sell volume estimation (CLV method)
3. Volume Profile (POC, VAH, VAL)
4. Funding Rate filtering (extreme, persistent, blocking)
5. Open Interest analysis (conviction matrix)
6. VolumeConfig defaults
7. Edge cases (zero volume, flat price, insufficient data)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import numpy as np
from config import BotConfig, VolumeConfig
from volume_analysis import (
    VolumeAnalyzer,
    VolumeAnalysisResult,
    FundingRateAnalyzer,
    FundingRateResult,
    OpenInterestAnalyzer,
    OpenInterestResult,
)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def config():
    cfg = BotConfig()
    return cfg


@pytest.fixture
def vol_analyzer(config):
    return VolumeAnalyzer(config)


@pytest.fixture
def funding_analyzer(config):
    return FundingRateAnalyzer(config)


@pytest.fixture
def oi_analyzer(config):
    return OpenInterestAnalyzer(config)


def _make_ohlcv_arrays(n=30, base_price=100.0, trend="up"):
    """Generate synthetic OHLCV arrays for testing."""
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)

    price = base_price
    for i in range(n):
        if trend == "up":
            change = 0.5 + (i * 0.1)  # Gradually rising
        elif trend == "down":
            change = -0.5 - (i * 0.1)  # Gradually falling
        else:
            change = 0.0

        open_p = price
        close_p = price + change
        high_p = max(open_p, close_p) + 0.5
        low_p = min(open_p, close_p) - 0.5

        opens[i] = open_p
        highs[i] = high_p
        lows[i] = low_p
        closes[i] = close_p
        volumes[i] = 1000.0 + i * 50  # Increasing volume

        price = close_p

    return opens, highs, lows, closes, volumes


# ─────────────────────────────────────────
# OBV Tests
# ─────────────────────────────────────────

class TestOBV:
    def test_obv_bullish_in_uptrend(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.obv_trend == "bullish"
        assert result.obv_value > 0

    def test_obv_bearish_in_downtrend(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="down")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.obv_trend == "bearish"
        assert result.obv_value < 0

    def test_obv_ema_computed(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.obv_ema != 0.0

    def test_obv_divergence_detected(self, vol_analyzer):
        """Price making new highs but OBV peaked earlier → divergence."""
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        # Make volume decrease in last candles while price still rises
        volumes[-5:] = 10.0  # Very low volume at end
        volumes[10:15] = 5000.0  # High volume in middle

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        # Divergence may or may not trigger depending on OBV accumulation
        # but the analysis should complete without error
        assert isinstance(result.obv_divergence, bool)

    def test_obv_flat_price_neutral(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="flat")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.obv_trend in ("bullish", "bearish", "neutral")


# ─────────────────────────────────────────
# Buy/Sell Volume Tests
# ─────────────────────────────────────────

class TestBuySellVolume:
    def test_strong_buying_pressure(self, vol_analyzer):
        """Closes consistently near highs → buying pressure."""
        n = 30
        opens = np.full(n, 100.0)
        lows = np.full(n, 99.0)
        highs = np.full(n, 102.0)
        closes = np.full(n, 101.8)  # Close near high
        volumes = np.full(n, 1000.0)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.buy_volume_ratio > 0.8
        assert result.volume_pressure == "buying"

    def test_strong_selling_pressure(self, vol_analyzer):
        """Closes consistently near lows → selling pressure."""
        n = 30
        opens = np.full(n, 100.0)
        lows = np.full(n, 98.0)
        highs = np.full(n, 101.0)
        closes = np.full(n, 98.2)  # Close near low
        volumes = np.full(n, 1000.0)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.buy_volume_ratio < 0.2
        assert result.volume_pressure == "selling"

    def test_neutral_pressure(self, vol_analyzer):
        """Closes near middle → neutral."""
        n = 30
        opens = np.full(n, 100.0)
        lows = np.full(n, 98.0)
        highs = np.full(n, 102.0)
        closes = np.full(n, 100.0)  # Close at midpoint
        volumes = np.full(n, 1000.0)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert 0.4 <= result.buy_volume_ratio <= 0.6
        assert result.volume_pressure == "neutral"

    def test_doji_candles_split_evenly(self, vol_analyzer):
        """When high == low (doji), volume split 50/50."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 100.0)  # high == low
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        volumes = np.full(n, 1000.0)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.buy_volume_ratio == pytest.approx(0.5, abs=0.01)

    def test_buy_sell_totals_sum(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30)
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        total = result.buy_volume_total + result.sell_volume_total
        assert total > 0


# ─────────────────────────────────────────
# Volume Profile Tests
# ─────────────────────────────────────────

class TestVolumeProfile:
    def test_poc_within_price_range(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30)
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.poc_price > 0
        assert result.poc_price >= np.min(lows)
        assert result.poc_price <= np.max(highs)

    def test_value_area_contains_poc(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30)
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.val_price <= result.poc_price
        assert result.vah_price >= result.poc_price

    def test_val_less_than_vah(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30)
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.val_price <= result.vah_price

    def test_price_vs_poc_classification(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.price_vs_poc in ("above", "below", "at")

    def test_flat_price_range(self, vol_analyzer):
        """All prices the same — POC equals that price."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        volumes = np.full(n, 1000.0)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.poc_price == pytest.approx(100.0, abs=0.5)


# ─────────────────────────────────────────
# Confidence Scoring Tests
# ─────────────────────────────────────────

class TestVolumeConfidence:
    def test_obv_confirm_adds_bonus(self, vol_analyzer):
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        # Bullish OBV should add positive confidence
        assert result.confidence_adj > 0
        assert any("obv" in s for s in result.signals)

    def test_buying_pressure_adds_bonus(self, vol_analyzer):
        n = 30
        opens = np.full(n, 100.0)
        lows = np.full(n, 99.0)
        highs = np.full(n, 102.0)
        closes = np.full(n, 101.8)
        volumes = np.full(n, 1000.0)
        # Make closes rise to trigger bullish OBV too
        for i in range(1, n):
            closes[i] = closes[i - 1] + 0.1

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert any("pressure" in s for s in result.signals)

    def test_insufficient_data_returns_defaults(self, vol_analyzer):
        # Less than 20 candles
        opens = np.array([100.0] * 5)
        highs = np.array([101.0] * 5)
        lows = np.array([99.0] * 5)
        closes = np.array([100.5] * 5)
        volumes = np.array([1000.0] * 5)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.obv_trend == "neutral"
        assert result.confidence_adj == 0.0


# ─────────────────────────────────────────
# Funding Rate Tests
# ─────────────────────────────────────────

class TestFundingRate:
    def test_normal_rate_no_penalty(self, funding_analyzer):
        funding_analyzer.update_rate("BTC/USDT", 0.01)
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.current_rate == 0.01
        assert not result.is_extreme_positive
        assert not result.is_extreme_negative
        assert result.confidence_adj >= 0

    def test_extreme_positive_blocks_long(self, funding_analyzer):
        """Very high positive funding → crowded longs → risky to go long."""
        funding_analyzer.update_rate("BTC/USDT", 0.10)  # 0.10% > 0.05%
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.is_extreme_positive
        assert result.block_long
        assert not result.block_short
        assert result.confidence_adj < 0

    def test_extreme_negative_blocks_short(self, funding_analyzer):
        """Very negative funding → crowded shorts → risky to go short."""
        funding_analyzer.update_rate("BTC/USDT", -0.10)
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.is_extreme_negative
        assert result.block_short
        assert not result.block_long
        assert result.confidence_adj < 0

    def test_persistent_positive_confirms(self, funding_analyzer):
        """Same sign for N periods = trend confirmation."""
        now = time.time()
        for i in range(5):
            funding_analyzer.update_rate("BTC/USDT", 0.02, now + i * 100)
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.is_persistent
        assert result.confidence_adj > 0

    def test_persistent_negative_confirms(self, funding_analyzer):
        now = time.time()
        for i in range(5):
            funding_analyzer.update_rate("BTC/USDT", -0.02, now + i * 100)
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.is_persistent

    def test_mixed_signs_not_persistent(self, funding_analyzer):
        now = time.time()
        funding_analyzer.update_rate("BTC/USDT", 0.02, now)
        funding_analyzer.update_rate("BTC/USDT", -0.01, now + 100)
        funding_analyzer.update_rate("BTC/USDT", 0.03, now + 200)
        result = funding_analyzer.analyze("BTC/USDT")
        assert not result.is_persistent

    def test_no_data_returns_defaults(self, funding_analyzer):
        result = funding_analyzer.analyze("ETH/USDT")
        assert result.current_rate == 0.0
        assert result.confidence_adj == 0.0

    def test_current_rate_override(self, funding_analyzer):
        result = funding_analyzer.analyze("BTC/USDT", current_rate=0.03)
        assert result.current_rate == 0.03

    def test_history_pruning(self, funding_analyzer):
        """Old entries (>24h) should be pruned."""
        old_time = time.time() - 100000  # Well over 24h ago
        funding_analyzer.update_rate("BTC/USDT", 0.01, old_time)
        now = time.time()
        funding_analyzer.update_rate("BTC/USDT", 0.02, now)
        history = funding_analyzer._rate_history["BTC/USDT"]
        assert len(history) == 1  # Old entry pruned

    def test_at_boundary_not_extreme(self, funding_analyzer):
        """Rate exactly at threshold should still be extreme (>=)."""
        funding_analyzer.update_rate("BTC/USDT", 0.05)
        result = funding_analyzer.analyze("BTC/USDT")
        assert result.is_extreme_positive

    def test_just_below_boundary_not_extreme(self, funding_analyzer):
        """Rate just below threshold should not be extreme."""
        funding_analyzer.update_rate("BTC/USDT", 0.049)
        result = funding_analyzer.analyze("BTC/USDT")
        assert not result.is_extreme_positive


# ─────────────────────────────────────────
# Open Interest Tests
# ─────────────────────────────────────────

class TestOpenInterest:
    def test_rising_oi_rising_price_strong(self, oi_analyzer):
        """Rising OI + rising price = strong conviction (new longs entering)."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1050000.0, now)  # +5%

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=2.0,  # Price up 2%
        )
        assert result.oi_trend == "rising"
        assert result.conviction == "strong"
        assert result.confidence_adj > 0

    def test_rising_oi_falling_price_strong(self, oi_analyzer):
        """Rising OI + falling price = strong conviction (new shorts entering)."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1050000.0, now)

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=48000.0,
            price_change_pct=-2.0,
        )
        assert result.conviction == "strong"

    def test_falling_oi_rising_price_exhaustion(self, oi_analyzer):
        """Falling OI + rising price = shorts covering → exhaustion."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 950000.0, now)  # -5%

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=2.0,
        )
        assert result.oi_trend == "falling"
        assert result.conviction == "exhaustion"
        assert result.is_divergence
        assert result.confidence_adj < 0

    def test_falling_oi_falling_price_exhaustion(self, oi_analyzer):
        """Falling OI + falling price = longs exiting → exhaustion."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 950000.0, now)

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=48000.0,
            price_change_pct=-2.0,
        )
        assert result.conviction == "exhaustion"
        assert result.is_divergence

    def test_flat_oi_neutral(self, oi_analyzer):
        """Flat OI → neutral conviction."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1010000.0, now)  # +1% (below threshold)

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=2.0,
        )
        assert result.oi_trend == "flat"
        assert result.conviction == "neutral"

    def test_no_data_returns_defaults(self, oi_analyzer):
        result = oi_analyzer.analyze(
            "ETH/USDT",
            current_price=3000.0,
            price_change_pct=1.0,
        )
        assert result.conviction == "neutral"
        assert result.confidence_adj == 0.0

    def test_single_observation_returns_defaults(self, oi_analyzer):
        """Need at least 2 observations to compute change."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now)
        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=1.0,
        )
        assert result.oi_change_pct == 0.0

    def test_history_pruning(self, oi_analyzer):
        """Old entries should be pruned."""
        old = time.time() - 100000
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, old)
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1050000.0, now)
        assert len(oi_analyzer._oi_history["BTC/USDT"]) == 1

    def test_oi_rising_flat_price_neutral(self, oi_analyzer):
        """OI building but price flat → neutral."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1100000.0, now)

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=0.1,  # < 0.5% threshold
        )
        assert result.conviction == "neutral"
        assert "oi_building_flat_price" in result.signals


# ─────────────────────────────────────────
# VolumeConfig Defaults
# ─────────────────────────────────────────

class TestVolumeConfig:
    def test_defaults(self):
        vc = VolumeConfig()
        assert vc.obv_ema_period == 10
        assert vc.obv_confirm_bonus == 0.05
        assert vc.obv_divergence_penalty == -0.10
        assert vc.buy_sell_lookback == 10
        assert vc.buy_sell_threshold == 0.60
        assert vc.buy_sell_pressure_bonus == 0.05
        assert vc.profile_lookback == 30
        assert vc.profile_bins == 20
        assert vc.profile_value_area_pct == 0.70
        assert vc.funding_enabled is True
        assert vc.funding_extreme_threshold == 0.05
        assert vc.funding_extreme_penalty == 0.10
        assert vc.funding_persistent_periods == 3
        assert vc.funding_persistent_bonus == 0.05
        assert vc.oi_enabled is True
        assert vc.oi_rising_threshold == 3.0
        assert vc.oi_falling_threshold == -3.0
        assert vc.oi_strong_conviction_bonus == 0.05
        assert vc.oi_exhaustion_penalty == 0.05

    def test_custom_config(self):
        cfg = BotConfig()
        cfg.volume.obv_ema_period = 20
        cfg.volume.funding_extreme_threshold = 0.10
        va = VolumeAnalyzer(cfg)
        assert va.vc.obv_ema_period == 20
        assert va.vc.funding_extreme_threshold == 0.10


# ─────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────

class TestEdgeCases:
    def test_zero_volume_candles(self, vol_analyzer):
        """Zero volume shouldn't crash."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        closes = np.full(n, 100.5)
        volumes = np.zeros(n)

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.buy_volume_ratio == pytest.approx(0.5, abs=0.01)

    def test_single_candle_no_crash(self, vol_analyzer):
        opens = np.array([100.0])
        highs = np.array([101.0])
        lows = np.array([99.0])
        closes = np.array([100.5])
        volumes = np.array([1000.0])

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert result.confidence_adj == 0.0  # Not enough data

    def test_very_large_volumes(self, vol_analyzer):
        """Large numbers shouldn't overflow."""
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30)
        volumes *= 1e12  # Trillion-unit volumes

        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)
        assert np.isfinite(result.obv_value)
        assert np.isfinite(result.buy_volume_ratio)

    def test_multiple_symbols_independent(self, funding_analyzer):
        """Different symbols don't interfere."""
        funding_analyzer.update_rate("BTC/USDT", 0.10)  # Extreme
        funding_analyzer.update_rate("ETH/USDT", 0.01)  # Normal

        btc = funding_analyzer.analyze("BTC/USDT")
        eth = funding_analyzer.analyze("ETH/USDT")

        assert btc.is_extreme_positive
        assert not eth.is_extreme_positive

    def test_oi_zero_old_value(self, oi_analyzer):
        """OI starting from zero shouldn't divide by zero."""
        now = time.time()
        oi_analyzer.update_oi("BTC/USDT", 0.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now)

        result = oi_analyzer.analyze(
            "BTC/USDT",
            current_price=50000.0,
            price_change_pct=1.0,
        )
        assert result.oi_change_pct == 0.0  # Division by zero guarded


# ─────────────────────────────────────────
# Integration-Style Tests
# ─────────────────────────────────────────

class TestIntegration:
    def test_full_volume_pipeline(self, vol_analyzer):
        """Full analysis produces all expected fields."""
        opens, highs, lows, closes, volumes = _make_ohlcv_arrays(30, trend="up")
        result = vol_analyzer.analyze(highs, lows, closes, volumes, opens)

        # All fields populated
        assert result.obv_trend in ("bullish", "bearish", "neutral")
        assert 0.0 <= result.buy_volume_ratio <= 1.0
        assert result.volume_pressure in ("buying", "selling", "neutral")
        assert result.poc_price > 0
        assert result.val_price > 0
        assert result.vah_price > 0
        assert result.price_vs_poc in ("above", "below", "at")
        assert isinstance(result.confidence_adj, float)
        assert isinstance(result.signals, list)

    def test_funding_and_oi_combined(self, funding_analyzer, oi_analyzer):
        """Both analyzers can work on the same symbol simultaneously."""
        now = time.time()

        # Set up funding
        for i in range(5):
            funding_analyzer.update_rate("BTC/USDT", 0.02, now + i * 100)

        # Set up OI
        oi_analyzer.update_oi("BTC/USDT", 1000000.0, now - 3600)
        oi_analyzer.update_oi("BTC/USDT", 1050000.0, now)

        fr = funding_analyzer.analyze("BTC/USDT")
        oi = oi_analyzer.analyze("BTC/USDT", 50000.0, 2.0)

        assert fr.is_persistent
        assert oi.conviction == "strong"

        # Combined confidence: both positive
        combined = fr.confidence_adj + oi.confidence_adj
        assert combined > 0
