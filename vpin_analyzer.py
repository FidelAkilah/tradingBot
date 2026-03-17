"""
VPIN Analyzer — Volume-Synchronized Probability of Informed Trading.

Based on the Easley, López de Prado, and O'Hara (2012) framework.

VPIN measures order flow toxicity by bucketing trades into equal-volume bars
and measuring the imbalance between buy- and sell-initiated volume within
each bar. High VPIN = high probability that informed traders are active,
which means:
  - Walls are more likely to be eaten (not pulled)
  - Price moves can be violent and directional
  - Market makers widen spreads

Integration with the scalping bot:
  - High VPIN → Block new entries (informed flow = unpredictable)
  - High VPIN + wall break → Ride the momentum instead of scalping
  - Low VPIN → Safe for mean-reversion scalps near walls
  - VPIN spike → Tighten/widen stops dynamically
  - VPIN regime detection → Switch between scalping and momentum strategies
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class FlowRegime(Enum):
    """Market microstructure regime based on VPIN dynamics."""
    LOW_TOXICITY = "low_toxicity"        # Safe for scalping / mean-reversion
    MODERATE_TOXICITY = "moderate"        # Proceed with caution
    HIGH_TOXICITY = "high_toxicity"       # Informed flow — avoid new entries
    TOXIC_SPIKE = "toxic_spike"           # Sudden VPIN jump — exit or hedge


@dataclass
class VolumeBar:
    """A single volume-synchronized bar (bucket)."""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    trade_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    vwap: float = 0.0                     # VWAP within this bar
    price_range: float = 0.0             # High - Low within bar

    @property
    def imbalance(self) -> float:
        """Absolute volume imbalance |V_buy - V_sell| / V_total."""
        if self.total_volume <= 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.total_volume

    @property
    def signed_imbalance(self) -> float:
        """Signed imbalance: positive = buy-heavy, negative = sell-heavy."""
        if self.total_volume <= 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / self.total_volume

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0


@dataclass
class VPINState:
    """Complete VPIN measurement for a symbol at a point in time."""
    vpin: float = 0.0                     # Current VPIN value (0.0 to 1.0)
    vpin_ema: float = 0.0                 # EMA-smoothed VPIN
    regime: FlowRegime = FlowRegime.LOW_TOXICITY
    bucket_count: int = 0                 # Number of completed buckets
    avg_bucket_duration_s: float = 0.0    # How fast buckets fill (market activity proxy)
    vpin_percentile: float = 0.0          # Where current VPIN sits in recent history
    vpin_delta: float = 0.0              # Change in VPIN from previous measurement
    directional_bias: float = 0.0        # Avg signed imbalance: +1=all buys, -1=all sells
    volatility_of_toxicity: float = 0.0  # Std dev of VPIN over recent window
    should_block_entry: bool = False
    should_widen_stops: bool = False
    stop_multiplier: float = 1.0
    timestamp: float = 0.0


# ─────────────────────────────────────────────
# Bulk Trade Classifier (BVC)
# ─────────────────────────────────────────────

class BulkVolumeClassifier:
    """
    Classifies trade volume as buy- or sell-initiated using the
    Bulk Volume Classification (BVC) method.

    BVC is preferred over tick-rule or Lee-Ready for HFT because:
    - It doesn't require quote data alignment
    - It handles the "no-trade" problem gracefully
    - It's volume-synchronized (natural for VPIN)

    Method: Uses the CDF of the standard normal distribution applied
    to normalized price changes to probabilistically assign volume.
    """

    @staticmethod
    def classify_trades(trades: list) -> List[Tuple[float, float, float, float]]:
        """
        Classify a list of trades into buy/sell volume.

        Returns list of (timestamp, buy_volume, sell_volume, price) tuples.
        """
        if len(trades) < 2:
            return []

        results = []
        prev_price = trades[0].get("price", 0)

        for trade in trades[1:]:
            price = trade.get("price", 0)
            amount = trade.get("amount", 0)
            ts = trade.get("timestamp", 0)
            side = trade.get("side", "")

            if price <= 0 or amount <= 0:
                continue

            # If exchange provides taker side, use it directly (most accurate)
            if side == "buy":
                buy_vol = amount
                sell_vol = 0.0
            elif side == "sell":
                buy_vol = 0.0
                sell_vol = amount
            else:
                # Fallback: BVC probabilistic classification
                # Normalize price change by recent volatility
                price_change = price - prev_price
                if prev_price > 0:
                    normalized = price_change / prev_price
                    # Apply sigmoid-like classification
                    # Positive change → likely buy-initiated
                    buy_prob = 1.0 / (1.0 + np.exp(-normalized * 1000))
                    buy_vol = amount * buy_prob
                    sell_vol = amount * (1.0 - buy_prob)
                else:
                    buy_vol = amount * 0.5
                    sell_vol = amount * 0.5

            results.append((ts / 1000.0 if ts > 1e12 else ts, buy_vol, sell_vol, price))
            prev_price = price

        return results


# ─────────────────────────────────────────────
# VPIN Engine
# ─────────────────────────────────────────────

class VPINAnalyzer:
    """
    Computes VPIN (Volume-Synchronized Probability of Informed Trading)
    in real-time from streaming trade data.

    The algorithm:
    1. Classify each trade as buy or sell-initiated (BVC)
    2. Accumulate trades into equal-volume buckets
    3. For each bucket, compute |V_buy - V_sell| / V_bucket
    4. VPIN = mean(imbalances) over the last N buckets
    5. Derive regime, signals, and risk adjustments
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.cfg = config.order_book

        self.classifier = BulkVolumeClassifier()

        # State per symbol
        self._buckets: Dict[str, deque] = {}               # Completed volume bars
        self._current_bucket: Dict[str, VolumeBar] = {}    # Bar being filled
        self._bucket_volume_target: Dict[str, float] = {}  # Volume per bucket
        self._vpin_history: Dict[str, deque] = {}           # VPIN time series
        self._ema_state: Dict[str, float] = {}              # EMA of VPIN
        self._last_state: Dict[str, VPINState] = {}
        self._processed_trade_count: Dict[str, int] = {}

    def _ensure_symbol(self, symbol: str):
        """Initialize state for a new symbol."""
        if symbol not in self._buckets:
            self._buckets[symbol] = deque(maxlen=self.cfg.vpin_bucket_count * 3)
            self._current_bucket[symbol] = VolumeBar(start_time=time.time())
            self._bucket_volume_target[symbol] = self.cfg.vpin_bucket_volume
            self._vpin_history[symbol] = deque(maxlen=500)
            self._ema_state[symbol] = 0.5  # Start neutral
            self._last_state[symbol] = VPINState()
            self._processed_trade_count[symbol] = 0

    def update(self, symbol: str, recent_trades: list, timestamp: float) -> VPINState:
        """
        Process new trades and return updated VPIN state.

        Args:
            symbol: Trading pair
            recent_trades: List of trade dicts from ccxt
            timestamp: Current time

        Returns:
            VPINState with current toxicity assessment
        """
        self._ensure_symbol(symbol)

        if not recent_trades or len(recent_trades) < 2:
            state = self._last_state[symbol]
            state.timestamp = timestamp
            return state

        # Auto-calculate bucket volume if not set
        if self._bucket_volume_target[symbol] <= 0:
            self._auto_bucket_volume(symbol, recent_trades)

        # Classify trades
        classified = self.classifier.classify_trades(recent_trades)
        if not classified:
            state = self._last_state[symbol]
            state.timestamp = timestamp
            return state

        # Feed classified trades into volume buckets
        for ts, buy_vol, sell_vol, price in classified:
            self._add_to_bucket(symbol, buy_vol, sell_vol, price, ts)

        # Compute VPIN from completed buckets
        vpin = self._compute_vpin(symbol)
        prev_vpin = self._ema_state.get(symbol, 0.5)

        # EMA smoothing
        alpha = self.cfg.vpin_ema_alpha
        vpin_ema = alpha * vpin + (1.0 - alpha) * prev_vpin
        self._ema_state[symbol] = vpin_ema

        # Record in history
        self._vpin_history[symbol].append(vpin)

        # Build full state
        state = self._build_state(symbol, vpin, vpin_ema, timestamp)
        self._last_state[symbol] = state

        return state

    def _auto_bucket_volume(self, symbol: str, trades: list):
        """
        Auto-calculate the volume per bucket based on recent trade activity.

        Target: total volume / bucket_count so that buckets fill at a
        reasonable rate (not too fast, not too slow).
        """
        total_vol = sum(t.get("amount", 0) for t in trades if t.get("amount", 0) > 0)
        if total_vol > 0:
            # Set bucket volume so that the given trades fill ~bucket_count buckets
            target = total_vol / self.cfg.vpin_bucket_count
            self._bucket_volume_target[symbol] = max(target, 1e-8)
            logger.debug(
                f"[{symbol}] VPIN auto bucket volume: {self._bucket_volume_target[symbol]:.6f} "
                f"(from {total_vol:.4f} total across {len(trades)} trades)"
            )

    def _add_to_bucket(
        self,
        symbol: str,
        buy_vol: float,
        sell_vol: float,
        price: float,
        timestamp: float,
    ):
        """Add classified volume to the current bucket, completing it if full."""
        bucket = self._current_bucket[symbol]
        target = self._bucket_volume_target[symbol]
        remaining_vol = buy_vol + sell_vol

        while remaining_vol > 0:
            space_left = target - bucket.total_volume

            if remaining_vol >= space_left and space_left > 0:
                # This trade completes the bucket
                fraction = space_left / remaining_vol if remaining_vol > 0 else 0
                add_buy = buy_vol * fraction
                add_sell = sell_vol * fraction

                bucket.buy_volume += add_buy
                bucket.sell_volume += add_sell
                bucket.total_volume += space_left
                bucket.trade_count += 1
                bucket.end_time = timestamp

                # Update VWAP for this bar
                if bucket.total_volume > 0:
                    bucket.vwap = (
                        (bucket.vwap * (bucket.total_volume - space_left) + price * space_left)
                        / bucket.total_volume
                    )

                # Complete the bucket
                self._buckets[symbol].append(bucket)

                # Start a new bucket
                self._current_bucket[symbol] = VolumeBar(start_time=timestamp)
                bucket = self._current_bucket[symbol]

                # Reduce remaining volume
                remaining_vol -= space_left
                buy_vol -= add_buy
                sell_vol -= add_sell

            else:
                # Trade fits within the current bucket
                bucket.buy_volume += buy_vol
                bucket.sell_volume += sell_vol
                bucket.total_volume += remaining_vol
                bucket.trade_count += 1

                if bucket.total_volume > 0:
                    bucket.vwap = (
                        (bucket.vwap * (bucket.total_volume - remaining_vol) + price * remaining_vol)
                        / bucket.total_volume
                    )

                remaining_vol = 0

    def _compute_vpin(self, symbol: str) -> float:
        """
        Compute VPIN = mean of absolute imbalances over the last N buckets.

        VPIN ∈ [0, 1]:
          0.0 = perfectly balanced flow (no informed trading)
          1.0 = completely one-sided flow (all informed)
        """
        buckets = self._buckets[symbol]
        n = min(self.cfg.vpin_bucket_count, len(buckets))

        if n == 0:
            return 0.5  # Neutral prior

        recent = list(buckets)[-n:]
        imbalances = [b.imbalance for b in recent]

        return float(np.mean(imbalances))

    def _build_state(
        self,
        symbol: str,
        vpin: float,
        vpin_ema: float,
        timestamp: float,
    ) -> VPINState:
        """Build the complete VPIN state with regime detection and trading signals."""

        # Regime detection
        regime = self._detect_regime(symbol, vpin, vpin_ema)

        # VPIN percentile (where does current value sit in history?)
        history = list(self._vpin_history[symbol])
        if len(history) > 10:
            percentile = float(np.sum(np.array(history) <= vpin) / len(history))
        else:
            percentile = 0.5

        # VPIN delta (rate of change)
        prev = self._last_state.get(symbol, VPINState())
        delta = vpin - prev.vpin

        # Directional bias from recent buckets
        buckets = list(self._buckets[symbol])
        n = min(self.cfg.vpin_regime_window, len(buckets))
        if n > 0:
            recent = buckets[-n:]
            directional_bias = float(np.mean([b.signed_imbalance for b in recent]))
        else:
            directional_bias = 0.0

        # Volatility of toxicity
        if len(history) > 5:
            vol_of_tox = float(np.std(history[-min(50, len(history)):]))
        else:
            vol_of_tox = 0.0

        # Bucket duration stats
        if n > 0:
            durations = [b.duration_s for b in buckets[-n:] if b.duration_s > 0]
            avg_duration = float(np.mean(durations)) if durations else 0.0
        else:
            avg_duration = 0.0

        # Trading signals
        should_block = vpin_ema >= self.cfg.vpin_block_entry_above
        should_widen = vpin_ema >= self.cfg.vpin_widen_stops_above

        stop_mult = 1.0
        if should_widen:
            # Scale stop multiplier linearly from 1.0 to vpin_stop_multiplier
            # as VPIN goes from widen threshold to 1.0
            range_pct = (vpin_ema - self.cfg.vpin_widen_stops_above) / \
                        (1.0 - self.cfg.vpin_widen_stops_above)
            range_pct = min(max(range_pct, 0.0), 1.0)
            stop_mult = 1.0 + range_pct * (self.cfg.vpin_stop_multiplier - 1.0)

        return VPINState(
            vpin=vpin,
            vpin_ema=vpin_ema,
            regime=regime,
            bucket_count=len(self._buckets[symbol]),
            avg_bucket_duration_s=avg_duration,
            vpin_percentile=percentile,
            vpin_delta=delta,
            directional_bias=directional_bias,
            volatility_of_toxicity=vol_of_tox,
            should_block_entry=should_block,
            should_widen_stops=should_widen,
            stop_multiplier=stop_mult,
            timestamp=timestamp,
        )

    def _detect_regime(self, symbol: str, vpin: float, vpin_ema: float) -> FlowRegime:
        """
        Classify the current market microstructure regime.

        Uses both instantaneous VPIN and EMA to avoid whipsawing:
        - TOXIC_SPIKE: Sudden jump in raw VPIN (delta > 0.15 in one step)
        - HIGH_TOXICITY: Sustained high VPIN (EMA above threshold)
        - MODERATE: VPIN in the uncertain middle zone
        - LOW_TOXICITY: Both raw and EMA below safe threshold
        """
        prev = self._last_state.get(symbol, VPINState())
        delta = vpin - prev.vpin

        # Spike detection: large sudden increase
        if delta > 0.15 and vpin > self.cfg.vpin_toxicity_threshold:
            return FlowRegime.TOXIC_SPIKE

        if vpin_ema >= self.cfg.vpin_toxicity_threshold:
            return FlowRegime.HIGH_TOXICITY

        if vpin_ema <= self.cfg.vpin_safe_threshold:
            return FlowRegime.LOW_TOXICITY

        return FlowRegime.MODERATE_TOXICITY

    # ─────────────────────────────────────────
    # PUBLIC UTILITIES
    # ─────────────────────────────────────────

    def get_state(self, symbol: str) -> VPINState:
        """Get the latest VPIN state for a symbol."""
        return self._last_state.get(symbol, VPINState())

    def get_summary(self, state: VPINState, symbol: str = "") -> str:
        """Human-readable VPIN summary."""
        regime_emoji = {
            FlowRegime.LOW_TOXICITY: "🟢",
            FlowRegime.MODERATE_TOXICITY: "🟡",
            FlowRegime.HIGH_TOXICITY: "🔴",
            FlowRegime.TOXIC_SPIKE: "⚡",
        }
        emoji = regime_emoji.get(state.regime, "❓")

        block_str = " | ⛔ ENTRY BLOCKED" if state.should_block_entry else ""
        widen_str = f" | stops x{state.stop_multiplier:.2f}" if state.should_widen_stops else ""

        return (
            f"  VPIN {symbol}: {state.vpin:.3f} (ema={state.vpin_ema:.3f}) "
            f"{emoji} {state.regime.value} | "
            f"pctl={state.vpin_percentile:.0%} | "
            f"delta={state.vpin_delta:+.3f} | "
            f"bias={state.directional_bias:+.3f} | "
            f"vol={state.volatility_of_toxicity:.3f} | "
            f"buckets={state.bucket_count} | "
            f"avg_fill={state.avg_bucket_duration_s:.1f}s"
            f"{block_str}{widen_str}"
        )
