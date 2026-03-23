"""
Tests for the smart re-entry system (reentry.py).

Validates:
1. Stop-out registration
2. Re-entry condition checks (cooldown, trend, ADX, confidence)
3. Daily loss limit blocking
4. Expiry and max re-entries
5. Candidate lifecycle (register → check → clear)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from config import BotConfig, ReentryConfig
from reentry import ReentryManager, ReentryCandidate


@pytest.fixture
def config():
    cfg = BotConfig()
    cfg.reentry.enabled = True
    cfg.reentry.cooldown_s = 180.0
    cfg.reentry.max_reentries_per_signal = 1
    cfg.reentry.expiry_s = 3600.0
    cfg.reentry.size_mult = 0.70
    cfg.reentry.sl_atr_mult = 0.8
    cfg.reentry.min_adx = 25.0
    cfg.reentry.min_confidence = 0.60
    cfg.reentry.daily_loss_block_pct = 50.0
    return cfg


@pytest.fixture
def manager(config):
    return ReentryManager(config)


def _register(manager, symbol="BTC/USDT", side="BUY", entry=100.0,
              stop_out=98.0, stop_time=None, conf=0.70, atr=2.0, amount=1.0):
    """Register a stop-out for testing."""
    if stop_time is None:
        stop_time = time.time() - 300  # 5 min ago
    manager.register_stopout(
        symbol=symbol,
        side=side,
        entry_price=entry,
        stop_out_price=stop_out,
        stop_out_time=stop_time,
        confidence=conf,
        atr=atr,
        amount=amount,
    )


# ─────────────────────────────────────────
# Registration
# ─────────────────────────────────────────

class TestRegistration:
    def test_register_creates_candidate(self, manager):
        _register(manager)
        cand = manager.get_candidate("BTC/USDT")
        assert cand is not None
        assert cand.symbol == "BTC/USDT"
        assert cand.original_side == "BUY"
        assert cand.original_entry_price == 100.0
        assert cand.stop_out_price == 98.0

    def test_register_sets_expiry(self, manager):
        stop_time = time.time()
        _register(manager, stop_time=stop_time)
        cand = manager.get_candidate("BTC/USDT")
        assert cand.expires_at == pytest.approx(stop_time + 3600.0)

    def test_register_disabled_does_nothing(self, config):
        config.reentry.enabled = False
        mgr = ReentryManager(config)
        _register(mgr)
        assert mgr.get_candidate("BTC/USDT") is None

    def test_register_overwrites_existing(self, manager):
        _register(manager, stop_out=98.0)
        _register(manager, stop_out=97.0)
        cand = manager.get_candidate("BTC/USDT")
        assert cand.stop_out_price == 97.0

    def test_register_sell_side(self, manager):
        _register(manager, side="SELL", entry=100.0, stop_out=102.0)
        cand = manager.get_candidate("BTC/USDT")
        assert cand.original_side == "SELL"


# ─────────────────────────────────────────
# Re-entry Checks — Happy Path
# ─────────────────────────────────────────

class TestReentryHappyPath:
    def test_reentry_triggers_when_all_conditions_met(self, manager):
        stop_time = time.time() - 300  # 5 min ago (past 3 min cooldown)
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            symbol="BTC/USDT",
            current_price=99.0,
            current_time=time.time(),
            swing_side="BUY",
            swing_confidence=0.70,
            swing_adx=30.0,
            daily_loss_consumed_pct=20.0,
        )
        assert result == "BUY"

    def test_reentry_returns_original_side(self, manager):
        stop_time = time.time() - 300
        _register(manager, side="SELL", stop_time=stop_time)

        result = manager.check_reentry(
            symbol="BTC/USDT",
            current_price=101.0,
            current_time=time.time(),
            swing_side="SELL",
            swing_confidence=0.70,
            swing_adx=30.0,
            daily_loss_consumed_pct=10.0,
        )
        assert result == "SELL"

    def test_reentry_increments_count(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )
        cand = manager.get_candidate("BTC/USDT")
        assert cand.reentry_count == 1


# ─────────────────────────────────────────
# Re-entry Checks — Blocking Conditions
# ─────────────────────────────────────────

class TestReentryBlocking:
    def test_cooldown_blocks_reentry(self, manager):
        stop_time = time.time() - 60  # Only 1 min ago (cooldown = 3 min)
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )
        assert result is None

    def test_expiry_blocks_and_removes(self, manager):
        stop_time = time.time() - 4000  # Way past expiry (1 hour)
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )
        assert result is None
        assert manager.get_candidate("BTC/USDT") is None

    def test_max_reentries_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        # First re-entry succeeds
        manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )

        # Second re-entry blocked (max = 1)
        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )
        assert result is None

    def test_daily_loss_limit_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0,
            daily_loss_consumed_pct=60.0,  # >50% consumed
        )
        assert result is None

    def test_wrong_direction_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, side="BUY", stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "SELL",  # Different direction
            0.70, 30.0, 10.0,
        )
        assert result is None

    def test_no_signal_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            None,  # No signal
            0.0, 0.0, 10.0,
        )
        assert result is None

    def test_low_confidence_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.50, 30.0, 10.0,  # conf 0.50 < min 0.60
        )
        assert result is None

    def test_low_adx_blocks(self, manager):
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 20.0, 10.0,  # ADX 20 < min 25
        )
        assert result is None

    def test_no_candidate_returns_none(self, manager):
        result = manager.check_reentry(
            "ETH/USDT", 3000.0, time.time(),
            "BUY", 0.70, 30.0, 10.0,
        )
        assert result is None


# ─────────────────────────────────────────
# Candidate Lifecycle
# ─────────────────────────────────────────

class TestCandidateLifecycle:
    def test_clear_removes_candidate(self, manager):
        _register(manager)
        manager.clear_candidate("BTC/USDT")
        assert manager.get_candidate("BTC/USDT") is None

    def test_clear_nonexistent_is_safe(self, manager):
        manager.clear_candidate("NONEXISTENT")  # Should not raise

    def test_cleanup_expired(self, manager):
        stop_time = time.time() - 4000
        _register(manager, symbol="BTC/USDT", stop_time=stop_time)
        _register(manager, symbol="ETH/USDT", stop_time=time.time() - 100)

        manager.cleanup_expired(time.time())

        assert manager.get_candidate("BTC/USDT") is None
        assert manager.get_candidate("ETH/USDT") is not None

    def test_multiple_symbols_independent(self, manager):
        stop_time = time.time() - 300
        _register(manager, symbol="BTC/USDT", stop_time=stop_time)
        _register(manager, symbol="ETH/USDT", side="SELL", stop_time=stop_time)

        btc = manager.get_candidate("BTC/USDT")
        eth = manager.get_candidate("ETH/USDT")

        assert btc.original_side == "BUY"
        assert eth.original_side == "SELL"


# ─────────────────────────────────────────
# Configuration Getters
# ─────────────────────────────────────────

class TestConfigGetters:
    def test_size_mult(self, manager):
        assert manager.get_size_mult() == 0.70

    def test_sl_atr_mult(self, manager):
        assert manager.get_sl_atr_mult() == 0.8

    def test_custom_config(self, config):
        config.reentry.size_mult = 0.50
        config.reentry.sl_atr_mult = 0.6
        mgr = ReentryManager(config)
        assert mgr.get_size_mult() == 0.50
        assert mgr.get_sl_atr_mult() == 0.6


# ─────────────────────────────────────────
# ReentryConfig Defaults
# ─────────────────────────────────────────

class TestReentryConfig:
    def test_defaults(self):
        rc = ReentryConfig()
        assert rc.enabled is True
        assert rc.cooldown_s == 180.0
        assert rc.max_reentries_per_signal == 1
        assert rc.expiry_s == 3600.0
        assert rc.size_mult == 0.70
        assert rc.sl_atr_mult == 0.8
        assert rc.min_adx == 25.0
        assert rc.min_confidence == 0.60
        assert rc.daily_loss_block_pct == 50.0

    def test_daily_loss_at_boundary(self, manager):
        """Exactly at 50% should block."""
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0,
            daily_loss_consumed_pct=50.0,  # Exactly at limit
        )
        assert result is None

    def test_daily_loss_just_below(self, manager):
        """Just below 50% should allow."""
        stop_time = time.time() - 300
        _register(manager, stop_time=stop_time)

        result = manager.check_reentry(
            "BTC/USDT", 99.0, time.time(),
            "BUY", 0.70, 30.0,
            daily_loss_consumed_pct=49.9,
        )
        assert result == "BUY"
