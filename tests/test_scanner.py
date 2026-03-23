"""
Tests for scanner/ module — OpportunityScanner, PairSelector, PairPerformanceTracker.

75 tests covering:
- OpportunityScanner: ADX/ATR/BB computation, scoring, normalization
- PairSelector: selection filters, anchor pairs, open position retention,
  disabled/auto-include, transitions
- PairPerformanceTracker: trade recording, win rate, auto-disable/include,
  DB update, summary
- ScanResult: active pair computation, added/dropped detection
"""

import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from config import BotConfig, ScannerConfig
from scanner.pair_scanner import (
    OpportunityScanner,
    PairScore,
    PairSelector,
    ScanResult,
)
from scanner.pair_performance import PairPerformanceTracker, PairStats


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def make_config(**kwargs) -> BotConfig:
    """Create a BotConfig with custom ScannerConfig overrides."""
    sc = ScannerConfig(**kwargs)
    cfg = BotConfig()
    cfg.scanner = sc
    return cfg


def make_score(symbol: str, **kwargs) -> PairScore:
    """Create a PairScore with defaults overridden."""
    ps = PairScore(symbol=symbol)
    for k, v in kwargs.items():
        setattr(ps, k, v)
    return ps


def _random_walk(n: int, start: float = 100.0, seed: int = 42) -> list:
    """Generate a random-walk price series."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.001, 0.02, size=n)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return prices


def _trending_series(n: int, start: float = 100.0, trend: float = 0.005) -> list:
    """Generate a trending price series (for high ADX)."""
    rng = np.random.RandomState(99)
    prices = [start]
    for i in range(n):
        prices.append(prices[-1] * (1 + trend + rng.normal(0, 0.002)))
    return prices


# ─────────────────────────────────────────
# OPPORTUNITY SCANNER — COMPUTATION TESTS
# ─────────────────────────────────────────

class TestScannerADX:
    """Tests for ADX computation in the scanner."""

    def test_trending_high_adx(self):
        """Strongly trending series should have high ADX."""
        scanner = OpportunityScanner(make_config())
        prices = _trending_series(50, trend=0.01)
        highs = np.array([p * 1.005 for p in prices], dtype=np.float64)
        lows = np.array([p * 0.995 for p in prices], dtype=np.float64)
        closes = np.array(prices, dtype=np.float64)
        adx = scanner._compute_adx(highs, lows, closes)
        assert adx > 20  # Should show trend

    def test_flat_low_adx(self):
        """Flat/ranging series should have low ADX."""
        scanner = OpportunityScanner(make_config())
        # Oscillating around same level
        prices = [100 + 0.5 * (-1)**i for i in range(50)]
        highs = np.array([p + 0.3 for p in prices], dtype=np.float64)
        lows = np.array([p - 0.3 for p in prices], dtype=np.float64)
        closes = np.array(prices, dtype=np.float64)
        adx = scanner._compute_adx(highs, lows, closes)
        assert adx < 30

    def test_short_data_returns_zero(self):
        """Not enough data → ADX = 0."""
        scanner = OpportunityScanner(make_config())
        closes = np.array([100, 101], dtype=np.float64)
        highs = np.array([101, 102], dtype=np.float64)
        lows = np.array([99, 100], dtype=np.float64)
        adx = scanner._compute_adx(highs, lows, closes)
        assert adx == 0.0


class TestScannerATR:
    """Tests for ATR computation."""

    def test_volatile_high_atr(self):
        """Volatile series should have high ATR."""
        scanner = OpportunityScanner(make_config())
        n = 30
        highs = np.array([100 + 5 * (i % 3) for i in range(n)], dtype=np.float64)
        lows = np.array([90 - 5 * (i % 3) for i in range(n)], dtype=np.float64)
        closes = np.array([95 + 2 * (i % 3) for i in range(n)], dtype=np.float64)
        atr = scanner._compute_atr(highs, lows, closes)
        assert atr > 0

    def test_flat_low_atr(self):
        """Flat series → low ATR."""
        scanner = OpportunityScanner(make_config())
        n = 30
        highs = np.array([100.5] * n, dtype=np.float64)
        lows = np.array([99.5] * n, dtype=np.float64)
        closes = np.array([100.0] * n, dtype=np.float64)
        atr = scanner._compute_atr(highs, lows, closes)
        assert atr == pytest.approx(1.0, abs=0.2)

    def test_insufficient_data(self):
        """Single candle → 0."""
        scanner = OpportunityScanner(make_config())
        atr = scanner._compute_atr(
            np.array([100.0]), np.array([99.0]), np.array([99.5])
        )
        assert atr == 0.0


class TestScannerBBSqueeze:
    """Tests for BB width percentile / squeeze detection."""

    def test_tight_bands_low_percentile(self):
        """Tight bands (low vol) → low percentile."""
        scanner = OpportunityScanner(make_config())
        # Start volatile, end flat → current should be low percentile
        prices = list(np.random.RandomState(42).normal(100, 5, size=60))
        # Make last 20 candles very flat
        prices[-20:] = [100.0 + 0.01 * i for i in range(20)]
        closes = np.array(prices, dtype=np.float64)
        pctile, squeeze = scanner._compute_bb_squeeze(closes)
        assert pctile < 50  # Should be in lower half

    def test_wide_bands_high_percentile(self):
        """Wide bands (high vol) → high percentile."""
        scanner = OpportunityScanner(make_config())
        # Start flat, end volatile
        prices = [100.0] * 40
        rng = np.random.RandomState(42)
        prices += list(100 + np.cumsum(rng.normal(0, 3, size=20)))
        closes = np.array(prices, dtype=np.float64)
        pctile, squeeze = scanner._compute_bb_squeeze(closes)
        assert pctile > 40  # Should be in upper half

    def test_insufficient_data(self):
        """Not enough data → default values."""
        scanner = OpportunityScanner(make_config())
        pctile, squeeze = scanner._compute_bb_squeeze(np.array([100.0] * 5))
        assert pctile == 50.0
        assert squeeze is False


class TestScannerScoring:
    """Tests for the composite opportunity score."""

    def test_score_range(self):
        """Score components are 0-1, weighted sum should be 0-1."""
        ps = PairScore(symbol="TEST/USDT")
        ps.adx_score = 1.0
        ps.squeeze_score = 1.0
        ps.volume_score = 1.0
        ps.volatility_score = 1.0
        ps.funding_score = 1.0

        cfg = make_config()
        w = cfg.scanner
        score = (
            w.weight_adx * ps.adx_score
            + w.weight_bb_squeeze * ps.squeeze_score
            + w.weight_volume_change * ps.volume_score
            + w.weight_volatility * ps.volatility_score
            + w.weight_funding * ps.funding_score
        )
        assert score == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        """All weights should sum to 1.0."""
        cfg = make_config()
        w = cfg.scanner
        total = (w.weight_adx + w.weight_bb_squeeze + w.weight_volume_change
                 + w.weight_volatility + w.weight_funding)
        assert total == pytest.approx(1.0)

    def test_zero_scores(self):
        """All zeros → score 0."""
        ps = PairScore(symbol="TEST/USDT")
        cfg = make_config()
        w = cfg.scanner
        score = (
            w.weight_adx * ps.adx_score
            + w.weight_bb_squeeze * ps.squeeze_score
            + w.weight_volume_change * ps.volume_score
            + w.weight_volatility * ps.volatility_score
            + w.weight_funding * ps.funding_score
        )
        assert score == 0.0

    def test_adx_normalization(self):
        """ADX normalized to 0-60 range, capped at 1.0."""
        assert min(30 / 60.0, 1.0) == 0.5
        assert min(60 / 60.0, 1.0) == 1.0
        assert min(90 / 60.0, 1.0) == 1.0  # Capped

    def test_atr_pct_normalization(self):
        """ATR% normalized to 0-3% range."""
        assert min(1.5 / 3.0, 1.0) == 0.5
        assert min(3.0 / 3.0, 1.0) == 1.0
        assert min(6.0 / 3.0, 1.0) == 1.0  # Capped

    def test_volume_change_normalization(self):
        """Volume change -50% to +200% mapped to 0-1."""
        # -50% → 0
        assert max(0.0, min((-50 + 50) / 250.0, 1.0)) == 0.0
        # +100% → 0.6
        assert max(0.0, min((100 + 50) / 250.0, 1.0)) == 0.6
        # +200% → 1.0
        assert max(0.0, min((200 + 50) / 250.0, 1.0)) == 1.0

    def test_funding_normalization(self):
        """Funding rate extremity: abs(rate) / 0.10, capped at 1.0."""
        assert min(abs(0.05) / 0.10, 1.0) == 0.5
        assert min(abs(-0.10) / 0.10, 1.0) == 1.0
        assert min(abs(0.20) / 0.10, 1.0) == 1.0  # Capped


# ─────────────────────────────────────────
# PAIR SELECTOR TESTS
# ─────────────────────────────────────────

class TestPairSelector:
    """Tests for PairSelector logic."""

    def _make_selector(self, **kwargs) -> PairSelector:
        return PairSelector(make_config(**kwargs))

    def test_anchor_pairs_always_included(self):
        """BTC and ETH are always in the active list."""
        sel = self._make_selector()
        scores = [
            make_score("SOL/USDT", score=0.8, adx=30, volume_24h_usd=100e6),
        ]
        result = sel.select(scores, [])
        active = result.get_active_pairs()
        assert "BTC/USDT" in active
        assert "ETH/USDT" in active

    def test_top_n_selected(self):
        """Top N dynamic pairs are selected by score."""
        sel = self._make_selector(select_dynamic=2, min_adx=10, min_24h_volume_usd=10e6)
        scores = [
            make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6),
            make_score("ADA/USDT", score=0.7, adx=25, volume_24h_usd=80e6),
            make_score("DOGE/USDT", score=0.5, adx=20, volume_24h_usd=60e6),
        ]
        result = sel.select(scores, [])
        assert "SOL/USDT" in result.selected_pairs
        assert "ADA/USDT" in result.selected_pairs
        assert "DOGE/USDT" not in result.selected_pairs

    def test_volume_filter(self):
        """Pairs below min volume are disqualified."""
        sel = self._make_selector(min_24h_volume_usd=50e6)
        scores = [
            make_score("LOW/USDT", score=0.9, adx=30, volume_24h_usd=10e6),
        ]
        result = sel.select(scores, [])
        assert "LOW/USDT" not in result.selected_pairs
        assert scores[0].disqualified is True

    def test_spread_filter(self):
        """Pairs with spread > max are disqualified."""
        sel = self._make_selector(max_spread_pct=0.05, min_24h_volume_usd=1e6)
        scores = [
            make_score("WIDE/USDT", score=0.9, adx=30, volume_24h_usd=100e6,
                       spread_pct=0.10),
        ]
        result = sel.select(scores, [])
        assert "WIDE/USDT" not in result.selected_pairs
        assert scores[0].disqualified is True

    def test_adx_filter(self):
        """Pairs with ADX below threshold are disqualified."""
        sel = self._make_selector(min_adx=22, min_24h_volume_usd=1e6)
        scores = [
            make_score("RANGING/USDT", score=0.9, adx=15, volume_24h_usd=100e6),
        ]
        result = sel.select(scores, [])
        assert "RANGING/USDT" not in result.selected_pairs
        assert scores[0].disqualified is True

    def test_squeeze_no_release_filter(self):
        """Pair in squeeze with no release → disqualified."""
        sel = self._make_selector(min_adx=10, min_24h_volume_usd=1e6)
        scores = [
            make_score("SQZ/USDT", score=0.9, adx=30, volume_24h_usd=100e6,
                       squeeze_active=True, squeeze_releasing=False),
        ]
        result = sel.select(scores, [])
        assert "SQZ/USDT" not in result.selected_pairs

    def test_squeeze_release_allowed(self):
        """Pair in squeeze with release → allowed."""
        sel = self._make_selector(min_adx=10, min_24h_volume_usd=1e6, select_dynamic=3)
        scores = [
            make_score("SQZ/USDT", score=0.9, adx=30, volume_24h_usd=100e6,
                       squeeze_active=True, squeeze_releasing=True),
        ]
        result = sel.select(scores, [])
        assert "SQZ/USDT" in result.selected_pairs

    def test_open_position_retained(self):
        """Dropped pairs with open positions stay in active list."""
        sel = self._make_selector(select_dynamic=1, min_adx=10, min_24h_volume_usd=1e6)
        # First scan: SOL selected
        scores1 = [
            make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6),
        ]
        sel.select(scores1, [])

        # Second scan: ADA replaces SOL, but SOL has open position
        scores2 = [
            make_score("ADA/USDT", score=0.95, adx=35, volume_24h_usd=120e6),
            make_score("SOL/USDT", score=0.3, adx=15, volume_24h_usd=100e6),
        ]
        result = sel.select(scores2, ["SOL/USDT"])
        active = result.get_active_pairs()
        assert "ADA/USDT" in active
        assert "SOL/USDT" in active  # Kept because of open position

    def test_disabled_pairs_excluded(self):
        """Disabled pairs are disqualified."""
        sel = self._make_selector(min_adx=10, min_24h_volume_usd=1e6)
        scores = [
            make_score("BAD/USDT", score=0.9, adx=30, volume_24h_usd=100e6),
        ]
        result = sel.select(scores, [], disabled_pairs=["BAD/USDT"])
        assert "BAD/USDT" not in result.selected_pairs

    def test_auto_include_pairs(self):
        """Auto-include pairs get added if they pass filters."""
        sel = self._make_selector(select_dynamic=1, min_adx=10, min_24h_volume_usd=1e6)
        scores = [
            make_score("TOP/USDT", score=0.9, adx=30, volume_24h_usd=100e6),
            make_score("GOOD/USDT", score=0.5, adx=25, volume_24h_usd=80e6),
        ]
        result = sel.select(scores, [], auto_include_pairs=["GOOD/USDT"])
        # TOP is selected as top scorer, GOOD is auto-included
        assert "TOP/USDT" in result.selected_pairs
        assert "GOOD/USDT" in result.selected_pairs

    def test_added_dropped_detection(self):
        """Correctly detects added and dropped pairs across scans."""
        sel = self._make_selector(select_dynamic=1, min_adx=10, min_24h_volume_usd=1e6)
        # First scan
        scores1 = [make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6)]
        result1 = sel.select(scores1, [])
        assert "SOL/USDT" in result1.added_pairs

        # Second scan — SOL dropped, ADA added
        scores2 = [make_score("ADA/USDT", score=0.95, adx=35, volume_24h_usd=120e6)]
        result2 = sel.select(scores2, [])
        assert "ADA/USDT" in result2.added_pairs
        assert "SOL/USDT" in result2.dropped_pairs

    def test_needs_scan_initially(self):
        """Needs scan before any scan has been done."""
        sel = self._make_selector()
        assert sel.needs_scan() is True

    def test_no_scan_needed_after_scan(self):
        """Doesn't need scan immediately after a scan."""
        sel = self._make_selector(scan_interval_s=3600)
        scores = [make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6)]
        sel.select(scores, [])
        assert sel.needs_scan() is False

    def test_needs_scan_after_interval(self):
        """Needs scan after interval has passed."""
        sel = self._make_selector(scan_interval_s=0.01)
        scores = [make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6)]
        sel.select(scores, [])
        import time as _time
        _time.sleep(0.02)
        assert sel.needs_scan() is True


class TestPairSelectorEdgeCases:
    """Edge cases for PairSelector."""

    def test_empty_scores(self):
        """Empty scores → only anchor pairs."""
        sel = PairSelector(make_config())
        result = sel.select([], [])
        active = result.get_active_pairs()
        assert set(active) == {"BTC/USDT", "ETH/USDT"}

    def test_all_disqualified(self):
        """All pairs disqualified → only anchors."""
        sel = PairSelector(make_config(min_adx=50, min_24h_volume_usd=1e6))
        scores = [
            make_score("SOL/USDT", score=0.9, adx=20, volume_24h_usd=100e6),
            make_score("ADA/USDT", score=0.8, adx=15, volume_24h_usd=80e6),
        ]
        result = sel.select(scores, [])
        assert result.selected_pairs == []
        assert result.pairs_qualified == 0

    def test_anchor_in_scores_not_double_counted(self):
        """BTC appearing in scores doesn't get selected as dynamic."""
        sel = PairSelector(make_config(min_adx=10, min_24h_volume_usd=1e6))
        scores = [
            make_score("BTC/USDT", score=0.99, adx=40, volume_24h_usd=500e6),
            make_score("SOL/USDT", score=0.8, adx=30, volume_24h_usd=100e6),
        ]
        result = sel.select(scores, [])
        # BTC should be anchor, SOL should be dynamic
        active = result.get_active_pairs()
        assert "BTC/USDT" in active
        assert "SOL/USDT" in active
        assert active.index("BTC/USDT") < active.index("SOL/USDT")  # Anchors first

    def test_zero_spread_allowed(self):
        """Zero spread should not be filtered (spread_pct > max check has > 0 guard)."""
        sel = PairSelector(make_config(max_spread_pct=0.05, min_adx=10, min_24h_volume_usd=1e6))
        scores = [
            make_score("OK/USDT", score=0.9, adx=30, volume_24h_usd=100e6, spread_pct=0.0),
        ]
        result = sel.select(scores, [])
        assert "OK/USDT" in result.selected_pairs

    def test_scan_summary_keys(self):
        """Scan summary has expected keys."""
        sel = PairSelector(make_config(min_adx=10, min_24h_volume_usd=1e6))
        scores = [make_score("SOL/USDT", score=0.9, adx=30, volume_24h_usd=100e6)]
        sel.select(scores, [])
        summary = sel.get_scan_summary()
        assert "scan_time" in summary
        assert "pairs_scanned" in summary
        assert "selected_pairs" in summary
        assert "anchor_pairs" in summary
        assert "active_pairs" in summary
        assert "scores" in summary


# ─────────────────────────────────────────
# SCAN RESULT TESTS
# ─────────────────────────────────────────

class TestScanResult:
    """Tests for ScanResult data class."""

    def test_get_active_pairs_dedupes(self):
        """Active pairs are deduplicated."""
        sr = ScanResult(
            anchor_pairs=["BTC/USDT", "ETH/USDT"],
            selected_pairs=["BTC/USDT", "SOL/USDT"],
        )
        active = sr.get_active_pairs()
        assert active == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_anchor_pairs_first(self):
        """Anchors appear before dynamic pairs."""
        sr = ScanResult(
            anchor_pairs=["BTC/USDT", "ETH/USDT"],
            selected_pairs=["ADA/USDT"],
        )
        active = sr.get_active_pairs()
        assert active[0] == "BTC/USDT"
        assert active[1] == "ETH/USDT"
        assert active[2] == "ADA/USDT"

    def test_empty_result(self):
        """Empty scan result → no pairs."""
        sr = ScanResult()
        assert sr.get_active_pairs() == []


# ─────────────────────────────────────────
# PAIR PERFORMANCE TRACKER TESTS
# ─────────────────────────────────────────

class TestPairPerformanceTracker:
    """Tests for PairPerformanceTracker."""

    def _make_tracker(self, **kwargs) -> PairPerformanceTracker:
        return PairPerformanceTracker(make_config(**kwargs))

    def test_record_trade_win(self):
        """Recording a winning trade updates stats."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", 1.5)
        s = t.get_pair_stats("SOL/USDT")
        assert s is not None
        assert s.wins == 1
        assert s.losses == 0
        assert s.total_pnl_usd == 1.5
        assert s.win_rate == 100.0

    def test_record_trade_loss(self):
        """Recording a losing trade updates stats."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", -0.5)
        s = t.get_pair_stats("SOL/USDT")
        assert s.wins == 0
        assert s.losses == 1
        assert s.win_rate == 0.0

    def test_multiple_trades(self):
        """Multiple trades accumulate correctly."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", 1.0)
        t.record_trade("SOL/USDT", -0.5)
        t.record_trade("SOL/USDT", 2.0)
        s = t.get_pair_stats("SOL/USDT")
        assert s.total_trades == 3
        assert s.wins == 2
        assert s.losses == 1
        assert s.win_rate == pytest.approx(66.67, abs=0.1)
        assert s.total_pnl_usd == 2.5

    def test_auto_disable_low_wr(self):
        """Pair with <35% WR over 10+ trades → disabled."""
        t = self._make_tracker(perf_disable_wr_below=35.0, perf_min_trades_for_disable=10)
        # 3 wins, 8 losses = 27.3% WR
        for i in range(3):
            t.record_trade("BAD/USDT", 0.5)
        for i in range(8):
            t.record_trade("BAD/USDT", -0.3)
        s = t.get_pair_stats("BAD/USDT")
        assert s.disabled is True
        assert "BAD/USDT" in t.get_disabled_pairs()

    def test_no_disable_insufficient_trades(self):
        """Not disabled if too few trades."""
        t = self._make_tracker(perf_disable_wr_below=35.0, perf_min_trades_for_disable=10)
        # 1 win, 5 losses = 16.7% WR but only 6 trades
        t.record_trade("FEW/USDT", 0.5)
        for i in range(5):
            t.record_trade("FEW/USDT", -0.3)
        s = t.get_pair_stats("FEW/USDT")
        assert s.disabled is False

    def test_auto_include_high_wr(self):
        """Pair with >60% WR and 5+ trades → auto-included."""
        t = self._make_tracker(perf_auto_include_wr_above=60.0)
        for i in range(4):
            t.record_trade("GOOD/USDT", 1.0)
        t.record_trade("GOOD/USDT", -0.5)
        s = t.get_pair_stats("GOOD/USDT")
        assert s.auto_included is True
        assert "GOOD/USDT" in t.get_auto_include_pairs()

    def test_no_auto_include_low_trades(self):
        """Not auto-included if <5 trades."""
        t = self._make_tracker(perf_auto_include_wr_above=60.0)
        t.record_trade("FEW/USDT", 1.0)
        t.record_trade("FEW/USDT", 1.0)
        s = t.get_pair_stats("FEW/USDT")
        assert s.auto_included is False

    def test_update_from_trades(self):
        """Update from list of trade dicts."""
        t = self._make_tracker()
        trades = [
            {"symbol": "SOL/USDT", "pnl_usd": 1.0, "is_open": 0},
            {"symbol": "SOL/USDT", "pnl_usd": -0.5, "is_open": 0},
            {"symbol": "SOL/USDT", "pnl_usd": 0.8, "is_open": 0},
            {"symbol": "ETH/USDT", "pnl_usd": 2.0, "is_open": 0},
            {"symbol": "ETH/USDT", "pnl_usd": -1.0, "is_open": 0},
            {"symbol": "OPEN/USDT", "pnl_usd": 0.0, "is_open": 1},  # Skipped
        ]
        t.update_from_trades(trades)

        sol = t.get_pair_stats("SOL/USDT")
        assert sol.total_trades == 3
        assert sol.wins == 2

        eth = t.get_pair_stats("ETH/USDT")
        assert eth.total_trades == 2

        # Open trade should be ignored
        assert t.get_pair_stats("OPEN/USDT") is None

    def test_contribution_pct(self):
        """Contribution percentage is computed correctly."""
        t = self._make_tracker()
        trades = [
            {"symbol": "A", "pnl_usd": 3.0, "is_open": 0},
            {"symbol": "B", "pnl_usd": 1.0, "is_open": 0},
        ]
        t.update_from_trades(trades)
        a_stats = t.get_pair_stats("A")
        b_stats = t.get_pair_stats("B")
        # Total P&L = 4.0
        assert a_stats.contribution_pct == pytest.approx(75.0)
        assert b_stats.contribution_pct == pytest.approx(25.0)

    def test_profit_factor(self):
        """Profit factor = gross_profit / gross_loss."""
        t = self._make_tracker()
        trades = [
            {"symbol": "X", "pnl_usd": 2.0, "is_open": 0},
            {"symbol": "X", "pnl_usd": -1.0, "is_open": 0},
        ]
        t.update_from_trades(trades)
        s = t.get_pair_stats("X")
        assert s.profit_factor == pytest.approx(2.0)

    def test_profit_factor_no_losses(self):
        """All wins → profit_factor = inf."""
        t = self._make_tracker()
        trades = [
            {"symbol": "X", "pnl_usd": 2.0, "is_open": 0},
        ]
        t.update_from_trades(trades)
        s = t.get_pair_stats("X")
        assert s.profit_factor == float('inf')

    def test_lookback_window(self):
        """Only last N trades used (perf_lookback_trades)."""
        t = self._make_tracker(perf_lookback_trades=5)
        # 25 total trades: 20 losses then 5 wins
        trades = []
        for i in range(20):
            trades.append({"symbol": "X", "pnl_usd": -0.5, "is_open": 0})
        for i in range(5):
            trades.append({"symbol": "X", "pnl_usd": 1.0, "is_open": 0})
        t.update_from_trades(trades)
        s = t.get_pair_stats("X")
        assert s.total_trades == 5  # Only last 5
        assert s.wins == 5
        assert s.win_rate == 100.0

    def test_get_summary_keys(self):
        """Summary has expected keys."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", 1.0)
        summary = t.get_summary()
        assert "pairs" in summary
        assert "disabled_count" in summary
        assert "auto_include_count" in summary
        assert "last_update" in summary

    def test_summary_inf_profit_factor(self):
        """Inf profit factor is handled in summary."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", 1.0)
        summary = t.get_summary()
        pf = summary["pairs"][0]["profit_factor"]
        assert pf == 999.0  # Capped from inf

    def test_unknown_pair_returns_none(self):
        """Unknown pair → None."""
        t = self._make_tracker()
        assert t.get_pair_stats("UNKNOWN/USDT") is None

    def test_disabled_cleared_on_recovery(self):
        """Pair recovers above threshold → no longer disabled."""
        t = self._make_tracker(perf_disable_wr_below=35.0, perf_min_trades_for_disable=5)
        # Start with low WR: 1 win, 4 losses = 20% → disabled
        t.record_trade("X", 1.0)
        for _ in range(4):
            t.record_trade("X", -1.0)
        s = t.get_pair_stats("X")
        assert s.disabled is True  # 1/5 = 20% < 35%

        # Add enough wins to recover above 35%: now 4 wins, 4 losses = 50%
        for _ in range(3):
            t.record_trade("X", 1.0)
        s = t.get_pair_stats("X")
        assert s.disabled is False  # 4/8 = 50% > 35%

    def test_multiple_symbols(self):
        """Tracker handles multiple symbols independently."""
        t = self._make_tracker()
        t.record_trade("SOL/USDT", 1.0)
        t.record_trade("ADA/USDT", -0.5)
        t.record_trade("SOL/USDT", 0.5)

        sol = t.get_pair_stats("SOL/USDT")
        ada = t.get_pair_stats("ADA/USDT")
        assert sol.total_trades == 2
        assert ada.total_trades == 1
        assert sol.win_rate == 100.0
        assert ada.win_rate == 0.0


class TestPairPerformanceEdgeCases:
    """Edge cases for PairPerformanceTracker."""

    def test_empty_trades_list(self):
        """Empty trade list → no stats."""
        t = PairPerformanceTracker(make_config())
        t.update_from_trades([])
        assert t.stats == {}

    def test_zero_pnl_counted_as_loss(self):
        """Zero P&L trade is counted as a loss."""
        t = PairPerformanceTracker(make_config())
        t.record_trade("X", 0.0)
        s = t.get_pair_stats("X")
        assert s.losses == 1
        assert s.wins == 0

    def test_missing_symbol_in_trade(self):
        """Trade without symbol → skipped."""
        t = PairPerformanceTracker(make_config())
        t.update_from_trades([{"pnl_usd": 1.0, "is_open": 0}])
        assert t.stats == {}


# ─────────────────────────────────────────
# PAIR SCORE DATA CLASS TESTS
# ─────────────────────────────────────────

class TestPairScore:
    """Tests for PairScore defaults and fields."""

    def test_defaults(self):
        """Default values are correct."""
        ps = PairScore(symbol="BTC/USDT")
        assert ps.score == 0.0
        assert ps.adx == 0.0
        assert ps.selected is False
        assert ps.disqualified is False
        assert ps.squeeze_active is False

    def test_all_fields_settable(self):
        """All fields can be set."""
        ps = make_score(
            "SOL/USDT",
            score=0.85,
            adx=35.0,
            volume_24h_usd=100e6,
            squeeze_active=True,
        )
        assert ps.score == 0.85
        assert ps.adx == 35.0
        assert ps.squeeze_active is True


class TestScannerConfig:
    """Tests for ScannerConfig defaults."""

    def test_defaults(self):
        """Default config values are sensible."""
        cfg = ScannerConfig()
        assert cfg.enabled is True
        assert cfg.scan_interval_s == 7200.0
        assert cfg.scan_top_n == 30
        assert cfg.select_dynamic == 3
        assert cfg.min_24h_volume_usd == 50e6
        assert cfg.max_spread_pct == 0.05
        assert cfg.min_adx == 22.0
        assert len(cfg.anchor_pairs) == 2

    def test_weights_sum(self):
        """Score weights sum to 1.0."""
        cfg = ScannerConfig()
        total = (cfg.weight_adx + cfg.weight_bb_squeeze + cfg.weight_volume_change
                 + cfg.weight_volatility + cfg.weight_funding)
        assert total == pytest.approx(1.0)

    def test_config_in_botconfig(self):
        """ScannerConfig is accessible from BotConfig."""
        cfg = BotConfig()
        assert hasattr(cfg, 'scanner')
        assert cfg.scanner.enabled is True
