"""
Liquidity Analyzer — the core intelligence module.

Algorithms:
1. Whale Wall Detection    — Identifies price levels with abnormally large resting orders
2. Spoofing Detection      — Tracks wall persistence to flag fake liquidity
3. Imbalance Ratio Scoring — Measures bid/ask volume asymmetry for directional bias
4. VWAP Anchoring          — Anchors signals relative to volume-weighted average price
5. Momentum Confirmation   — Validates signals with recent trade flow analysis
6. VPIN Integration        — Order flow toxicity from vpin_analyzer.py
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG
from candle_analyzer import CandleAnalyzer, SwingSignal, TrendDirection
from vpin_analyzer import VPINAnalyzer, VPINState, FlowRegime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class WallSide(Enum):
    BID = "bid"
    ASK = "ask"


class SignalDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class LiquidityWall:
    """A detected concentration of liquidity at a price level."""
    price: float
    volume: float                      # Base currency volume
    usd_value: float                   # Estimated USD value
    side: WallSide
    multiplier: float                  # How many x the average volume
    cluster_levels: int                # Number of price levels merged into this wall
    first_seen: float = 0.0            # Timestamp when first detected
    last_seen: float = 0.0
    appearance_count: int = 0          # How many snapshots it appeared in
    disappearance_count: int = 0       # How many snapshots it was absent
    is_spoof_suspect: bool = False
    confidence: float = 1.0            # 0.0 to 1.0, reduced if spoof-like behavior

    @property
    def persistence_ratio(self) -> float:
        """Ratio of appearances to total observations."""
        total = self.appearance_count + self.disappearance_count
        return self.appearance_count / total if total > 0 else 0.0


@dataclass
class ImbalanceSignal:
    """Bid/ask volume imbalance measurement."""
    bid_volume: float
    ask_volume: float
    ratio: float                       # bid_vol / ask_vol
    direction: SignalDirection
    depth_levels: int
    timestamp: float = 0.0


@dataclass
class VWAPState:
    """Volume-weighted average price state."""
    vwap: float = 0.0
    deviation_pct: float = 0.0         # Current price deviation from VWAP
    is_above_vwap: bool = True
    cumulative_volume: float = 0.0
    cumulative_vp: float = 0.0         # Cumulative (volume * price)
    timestamp: float = 0.0


@dataclass
class MomentumSignal:
    """Recent trade flow analysis."""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    net_flow: float = 0.0              # Positive = net buying
    aggressor_ratio: float = 0.5       # 0.0 = all sells, 1.0 = all buys
    direction: SignalDirection = SignalDirection.NEUTRAL
    large_trade_count: int = 0         # Trades > 2x average size
    timestamp: float = 0.0


@dataclass
class AnalysisResult:
    """Complete analysis output for a single symbol snapshot."""
    symbol: str
    timestamp: float
    mid_price: float
    spread_pct: float
    bid_walls: List[LiquidityWall]
    ask_walls: List[LiquidityWall]
    imbalance: ImbalanceSignal
    vwap: VWAPState
    momentum: MomentumSignal
    vpin: VPINState
    swing: Optional[SwingSignal] = None    # Multi-timeframe candle signal
    composite_score: float = 0.0           # -1.0 (strong bearish) to +1.0 (strong bullish)
    trade_suggestion: Optional[str] = None # "BUY", "SELL", or None
    atr_tp_pct: float = 0.0               # ATR-based take profit % (fee-adjusted)
    atr_sl_pct: float = 0.0               # ATR-based stop loss % (fee-adjusted)
    raw_tp_pct: float = 0.0               # Pre-fee TP %
    raw_sl_pct: float = 0.0               # Pre-fee SL %
    fee_cost_pct: float = 0.0             # Round-trip fee as % of notional
    post_fee_rr: float = 0.0              # Risk-reward ratio after fees
    adx: float = 0.0                       # Primary ADX value


# ─────────────────────────────────────────────
# Liquidity Analyzer Engine
# ─────────────────────────────────────────────

class LiquidityAnalyzer:
    """
    Stateful analyzer that processes order book snapshots and trade data
    to produce actionable scalping signals.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.ob_cfg = config.order_book

        # Wall tracking state per symbol: {symbol: {price_key: LiquidityWall}}
        self._wall_history: Dict[str, Dict[str, LiquidityWall]] = defaultdict(dict)
        # Snapshot counter per symbol for spoofing detection window
        self._snapshot_count: Dict[str, int] = defaultdict(int)
        # VWAP state per symbol
        self._vwap_state: Dict[str, VWAPState] = {}
        # Recent analysis results (for trend detection)
        self._result_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # VPIN analyzer instance (shared across all symbols)
        self._vpin = VPINAnalyzer(config)
        # Candle analyzer for multi-timeframe swing signals
        self._candle = CandleAnalyzer(config)

    def analyze(
        self,
        symbol: str,
        order_book: dict,
        recent_trades: list,
        timestamp: float,
        swing_signal: Optional[SwingSignal] = None,
    ) -> AnalysisResult:
        """
        Full analysis pipeline on a single order book snapshot.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            order_book: ccxt order book dict with 'bids', 'asks', etc.
            recent_trades: List of recent trade dicts from ccxt
            timestamp: Current timestamp
            swing_signal: Optional multi-timeframe candle analysis

        Returns:
            AnalysisResult with all signals and composite score
        """
        bids = np.array(order_book.get("bids", []), dtype=np.float64)
        asks = np.array(order_book.get("asks", []), dtype=np.float64)

        if len(bids) == 0 or len(asks) == 0:
            return self._empty_result(symbol, timestamp)

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        spread_pct = ((best_ask - best_bid) / mid_price) * 100.0

        # 1. Whale Wall Detection
        bid_walls = self._detect_walls(bids, WallSide.BID, mid_price, symbol, timestamp)
        ask_walls = self._detect_walls(asks, WallSide.ASK, mid_price, symbol, timestamp)

        # 2. Spoofing Detection (updates wall confidence in-place)
        self._update_spoofing_state(symbol, bid_walls + ask_walls, timestamp)
        self._snapshot_count[symbol] += 1

        # 3. Imbalance Ratio
        imbalance = self._calc_imbalance(bids, asks, timestamp)

        # 4. VWAP Anchoring
        vwap = self._calc_vwap(symbol, recent_trades, mid_price, timestamp)

        # 5. Momentum Confirmation
        momentum = self._calc_momentum(recent_trades, timestamp)

        # 6. VPIN — Order Flow Toxicity
        vpin_state = self._vpin.update(symbol, recent_trades, timestamp)

        # 7. Composite Score (candle-driven for swing, OB as confirmation)
        composite = self._composite_score(bid_walls, ask_walls, imbalance, vwap, momentum, vpin_state, swing_signal)

        # 8. Trade Suggestion (candle trend is primary driver)
        suggestion = self._trade_suggestion(composite, spread_pct, bid_walls, ask_walls, vpin_state, swing_signal)

        # ATR-based TP/SL (fee-adjusted values from swing signal)
        atr_tp = swing_signal.atr_tp_pct if swing_signal else self.config.trading.take_profit_pct
        atr_sl = swing_signal.atr_sl_pct if swing_signal else self.config.trading.stop_loss_pct

        # Fee metrics from swing signal
        raw_tp = swing_signal.raw_tp_pct if swing_signal else 0.0
        raw_sl = swing_signal.raw_sl_pct if swing_signal else 0.0
        fee_cost = swing_signal.fee_cost_pct if swing_signal else 0.0
        post_fee_rr = swing_signal.post_fee_rr if swing_signal else 0.0
        adx_val = swing_signal.adx_4h if swing_signal and swing_signal.adx_4h > 0 else (
            swing_signal.adx_1h if swing_signal else 0.0
        )

        result = AnalysisResult(
            symbol=symbol,
            timestamp=timestamp,
            mid_price=mid_price,
            spread_pct=spread_pct,
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            imbalance=imbalance,
            vwap=vwap,
            momentum=momentum,
            vpin=vpin_state,
            swing=swing_signal,
            composite_score=composite,
            trade_suggestion=suggestion,
            atr_tp_pct=atr_tp if atr_tp > 0 else self.config.trading.take_profit_pct,
            atr_sl_pct=atr_sl if atr_sl > 0 else self.config.trading.stop_loss_pct,
            raw_tp_pct=raw_tp,
            raw_sl_pct=raw_sl,
            fee_cost_pct=fee_cost,
            post_fee_rr=post_fee_rr,
            adx=adx_val,
        )

        self._result_history[symbol].append(result)
        return result

    # ─────────────────────────────────────────
    # 1. WHALE WALL DETECTION
    # ─────────────────────────────────────────

    def _detect_walls(
        self,
        levels: np.ndarray,
        side: WallSide,
        mid_price: float,
        symbol: str,
        timestamp: float,
    ) -> List[LiquidityWall]:
        """
        Detect liquidity walls by finding price levels where volume
        significantly exceeds the average.

        Uses clustering to merge adjacent levels within cluster_range_pct.
        """
        if len(levels) < 3:
            return []

        prices = levels[:, 0]
        volumes = levels[:, 1]
        usd_values = prices * volumes
        avg_volume = np.mean(volumes)

        if avg_volume <= 0:
            return []

        multipliers = volumes / avg_volume

        # Phase 1: Find all levels exceeding the whale multiplier
        whale_mask = (multipliers >= self.ob_cfg.whale_multiplier) & \
                     (usd_values >= self.ob_cfg.min_wall_usd)
        whale_indices = np.where(whale_mask)[0]

        if len(whale_indices) == 0:
            return []

        # Phase 2: Cluster adjacent whale levels
        clusters = self._cluster_levels(
            prices[whale_indices],
            volumes[whale_indices],
            mid_price
        )

        # Phase 3: Build LiquidityWall objects, sorted by USD value
        walls = []
        for cluster_price, cluster_vol, n_levels in clusters:
            usd = cluster_price * cluster_vol
            mult = cluster_vol / avg_volume

            wall = LiquidityWall(
                price=cluster_price,
                volume=cluster_vol,
                usd_value=usd,
                side=side,
                multiplier=mult,
                cluster_levels=n_levels,
                first_seen=timestamp,
                last_seen=timestamp,
                appearance_count=1,
            )
            walls.append(wall)

        # Sort by USD value descending, take top N
        walls.sort(key=lambda w: w.usd_value, reverse=True)
        return walls[:self.ob_cfg.whale_top_n]

    def _cluster_levels(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        mid_price: float
    ) -> List[Tuple[float, float, int]]:
        """
        Merge adjacent price levels within cluster_range_pct of each other.
        Returns list of (volume-weighted avg price, total volume, level count).
        """
        if len(prices) == 0:
            return []

        cluster_range = mid_price * (self.ob_cfg.cluster_range_pct / 100.0)
        clusters = []
        used = set()

        sorted_idx = np.argsort(prices)
        prices = prices[sorted_idx]
        volumes = volumes[sorted_idx]

        for i in range(len(prices)):
            if i in used:
                continue

            cluster_prices = [prices[i]]
            cluster_volumes = [volumes[i]]
            used.add(i)

            for j in range(i + 1, len(prices)):
                if j in used:
                    continue
                if abs(prices[j] - prices[i]) <= cluster_range:
                    cluster_prices.append(prices[j])
                    cluster_volumes.append(volumes[j])
                    used.add(j)

            total_vol = sum(cluster_volumes)
            # Volume-weighted average price for the cluster
            vwap = sum(p * v for p, v in zip(cluster_prices, cluster_volumes)) / total_vol
            clusters.append((vwap, total_vol, len(cluster_prices)))

        return clusters

    # ─────────────────────────────────────────
    # 2. SPOOFING DETECTION
    # ─────────────────────────────────────────

    def _update_spoofing_state(
        self,
        symbol: str,
        current_walls: List[LiquidityWall],
        timestamp: float
    ):
        """
        Track wall persistence across snapshots to identify spoofing.

        A wall that appears and disappears rapidly (flickering) is flagged
        as a spoof suspect, reducing its confidence score.
        """
        history = self._wall_history[symbol]
        window = self.ob_cfg.wall_persistence_window_s

        # Build a key for each current wall (round price to avoid float noise)
        current_keys = set()
        for wall in current_walls:
            key = f"{wall.side.value}_{wall.price:.8f}"
            current_keys.add(key)

            if key in history:
                # Wall seen again — update
                existing = history[key]
                existing.last_seen = timestamp
                existing.appearance_count += 1
                existing.volume = wall.volume  # Update to latest volume
                existing.usd_value = wall.usd_value

                # Update the wall object with historical data
                wall.first_seen = existing.first_seen
                wall.appearance_count = existing.appearance_count
                wall.disappearance_count = existing.disappearance_count
            else:
                # New wall
                history[key] = wall

        # Check for walls that disappeared this snapshot
        stale_keys = []
        for key, wall in history.items():
            if key not in current_keys:
                wall.disappearance_count += 1

            # Evaluate spoofing
            age = timestamp - wall.first_seen
            if age >= window:
                persistence = wall.persistence_ratio
                if persistence < (1.0 - self.ob_cfg.spoof_cancel_threshold):
                    wall.is_spoof_suspect = True
                    wall.confidence = max(0.1, persistence)

                flicker_total = wall.appearance_count + wall.disappearance_count
                if wall.disappearance_count >= self.ob_cfg.spoof_flicker_count and \
                   flicker_total >= self.ob_cfg.spoof_flicker_count * 2:
                    wall.is_spoof_suspect = True
                    wall.confidence = min(wall.confidence, 0.3)

            # Prune walls that haven't been seen in 2x the window
            if timestamp - wall.last_seen > window * 2:
                stale_keys.append(key)

        for key in stale_keys:
            del history[key]

        # Apply spoof data back to current walls
        for wall in current_walls:
            key = f"{wall.side.value}_{wall.price:.8f}"
            if key in history:
                hw = history[key]
                wall.is_spoof_suspect = hw.is_spoof_suspect
                wall.confidence = hw.confidence
                wall.appearance_count = hw.appearance_count
                wall.disappearance_count = hw.disappearance_count

    # ─────────────────────────────────────────
    # 3. IMBALANCE RATIO SCORING
    # ─────────────────────────────────────────

    def _calc_imbalance(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        timestamp: float
    ) -> ImbalanceSignal:
        """
        Calculate the bid/ask volume imbalance across the top N depth levels.

        Ratio > imbalance_signal_threshold → Bullish (more buying pressure)
        Ratio < 1/imbalance_signal_threshold → Bearish (more selling pressure)
        """
        n = min(
            self.ob_cfg.imbalance_depth_levels,
            len(bids),
            len(asks)
        )

        bid_vol = float(np.sum(bids[:n, 1]))
        ask_vol = float(np.sum(asks[:n, 1]))

        ratio = bid_vol / ask_vol if ask_vol > 0 else float("inf")

        threshold = self.ob_cfg.imbalance_signal_threshold
        if ratio >= threshold:
            direction = SignalDirection.BULLISH
        elif ratio <= 1.0 / threshold:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        return ImbalanceSignal(
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            ratio=ratio,
            direction=direction,
            depth_levels=n,
            timestamp=timestamp,
        )

    # ─────────────────────────────────────────
    # 4. VWAP ANCHORING
    # ─────────────────────────────────────────

    def _calc_vwap(
        self,
        symbol: str,
        recent_trades: list,
        mid_price: float,
        timestamp: float
    ) -> VWAPState:
        """
        Calculate VWAP from recent trades and measure current price deviation.

        Trades above VWAP with bullish imbalance = stronger buy signal.
        Trades below VWAP with bearish imbalance = stronger sell signal.
        """
        if not recent_trades:
            state = self._vwap_state.get(symbol, VWAPState(vwap=mid_price))
            state.timestamp = timestamp
            return state

        total_vp = 0.0
        total_vol = 0.0

        for trade in recent_trades:
            price = trade.get("price", 0)
            amount = trade.get("amount", 0)
            if price > 0 and amount > 0:
                total_vp += price * amount
                total_vol += amount

        vwap = total_vp / total_vol if total_vol > 0 else mid_price
        deviation_pct = ((mid_price - vwap) / vwap) * 100.0 if vwap > 0 else 0.0

        state = VWAPState(
            vwap=vwap,
            deviation_pct=deviation_pct,
            is_above_vwap=mid_price >= vwap,
            cumulative_volume=total_vol,
            cumulative_vp=total_vp,
            timestamp=timestamp,
        )
        self._vwap_state[symbol] = state
        return state

    # ─────────────────────────────────────────
    # 5. MOMENTUM CONFIRMATION
    # ─────────────────────────────────────────

    def _calc_momentum(
        self,
        recent_trades: list,
        timestamp: float
    ) -> MomentumSignal:
        """
        Analyze recent trade flow to confirm directional momentum.

        Looks at:
        - Net buy vs sell volume (taker side)
        - Aggressor ratio (what % of volume was buy-initiated)
        - Large trade detection (trades > 2x average)
        """
        if not recent_trades:
            return MomentumSignal(timestamp=timestamp)

        buy_vol = 0.0
        sell_vol = 0.0
        trade_sizes = []

        for trade in recent_trades:
            amount = trade.get("amount", 0)
            side = trade.get("side", "")
            if amount <= 0:
                continue

            trade_sizes.append(amount)
            if side == "buy":
                buy_vol += amount
            else:
                sell_vol += amount

        total_vol = buy_vol + sell_vol
        if total_vol <= 0:
            return MomentumSignal(timestamp=timestamp)

        aggressor_ratio = buy_vol / total_vol
        net_flow = buy_vol - sell_vol

        # Large trade detection
        avg_size = np.mean(trade_sizes) if trade_sizes else 0
        large_trades = sum(1 for s in trade_sizes if s > avg_size * 2)

        # Direction
        if aggressor_ratio > 0.6:
            direction = SignalDirection.BULLISH
        elif aggressor_ratio < 0.4:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        return MomentumSignal(
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            net_flow=net_flow,
            aggressor_ratio=aggressor_ratio,
            direction=direction,
            large_trade_count=large_trades,
            timestamp=timestamp,
        )

    # ─────────────────────────────────────────
    # 6. COMPOSITE SCORING
    # ─────────────────────────────────────────

    def _composite_score(
        self,
        bid_walls: List[LiquidityWall],
        ask_walls: List[LiquidityWall],
        imbalance: ImbalanceSignal,
        vwap: VWAPState,
        momentum: MomentumSignal,
        vpin: VPINState = None,
        swing: SwingSignal = None,
    ) -> float:
        """
        Combine all signals into a single composite score from -1.0 to +1.0.

        SWING WEIGHTS (candle-driven):
        - Candle trend + confidence: 40%  (PRIMARY DRIVER)
        - Order book walls: 15%           (confirmation only)
        - Imbalance ratio: 10%
        - VWAP positioning: 10%
        - Momentum flow: 15%
        - VPIN directional: 10%

        The old approach was 30% walls which meant wall-pulls killed trades.
        Now walls are 15% confirmation — a wall pull barely moves the score.
        """
        # --- Candle/Swing Score (PRIMARY) ---
        swing_signal_val = 0.0
        if swing and swing.suggested_side:
            # Map confidence to signed score
            if swing.suggested_side == "BUY":
                swing_signal_val = swing.confidence  # 0 to 1
            elif swing.suggested_side == "SELL":
                swing_signal_val = -swing.confidence  # -1 to 0

            # Bonus for aligned timeframes
            if swing.trend_aligned:
                swing_signal_val *= 1.2
            swing_signal_val = np.clip(swing_signal_val, -1.0, 1.0)

        # --- Wall Score (confirmation only) ---
        bid_wall_score = sum(
            w.usd_value * w.confidence for w in bid_walls if not w.is_spoof_suspect
        )
        ask_wall_score = sum(
            w.usd_value * w.confidence for w in ask_walls if not w.is_spoof_suspect
        )
        total_wall = bid_wall_score + ask_wall_score
        wall_signal = (bid_wall_score - ask_wall_score) / total_wall if total_wall > 0 else 0.0

        # --- Imbalance Score ---
        if imbalance.ratio > 0 and imbalance.ratio != float("inf"):
            imb_log = np.log(imbalance.ratio)
            imb_signal = np.clip(imb_log / 2.0, -1.0, 1.0)
        else:
            imb_signal = 0.0

        # --- VWAP Score ---
        vwap_signal = np.clip(vwap.deviation_pct / self.ob_cfg.vwap_deviation_pct, -1.0, 1.0)

        # --- Momentum Score ---
        mom_signal = np.clip((momentum.aggressor_ratio - 0.5) * 4.0, -1.0, 1.0)

        # --- VPIN Score ---
        vpin_signal = 0.0
        if vpin and vpin.bucket_count > 0:
            vpin_signal = np.clip(vpin.directional_bias * 2.0, -1.0, 1.0)

        # OB imbalance confirmation: if imbalance agrees with swing direction, +0.05 confidence
        if swing and swing.suggested_side:
            imb_confirms = False
            if swing.suggested_side == "BUY" and imbalance.direction == SignalDirection.BULLISH:
                imb_confirms = True
            elif swing.suggested_side == "SELL" and imbalance.direction == SignalDirection.BEARISH:
                imb_confirms = True
            if imb_confirms:
                swing.confidence = min(swing.confidence + 0.05, 1.0)
                # Re-derive swing_signal_val with updated confidence
                if swing.suggested_side == "BUY":
                    swing_signal_val = swing.confidence
                else:
                    swing_signal_val = -swing.confidence
                if swing.trend_aligned:
                    swing_signal_val *= 1.2
                swing_signal_val = float(np.clip(swing_signal_val, -1.0, 1.0))

        # Weighted composite — candle-driven
        if swing and swing.suggested_side:
            composite = (
                swing_signal_val * 0.40 +
                wall_signal * 0.15 +
                imb_signal * 0.10 +
                vwap_signal * 0.10 +
                mom_signal * 0.15 +
                vpin_signal * 0.10
            )
        else:
            # No candle signal available — use OB-only with reduced confidence
            composite = (
                wall_signal * 0.25 +
                imb_signal * 0.20 +
                vwap_signal * 0.15 +
                mom_signal * 0.25 +
                vpin_signal * 0.15
            )
            # Dampen OB-only signals (less reliable without candle confirmation)
            composite *= 0.6

        # VPIN toxicity dampener (only at extreme levels now)
        if vpin and vpin.regime == FlowRegime.TOXIC_SPIKE:
            composite *= 0.5

        return float(np.clip(composite, -1.0, 1.0))

    # ─────────────────────────────────────────
    # 7. TRADE SUGGESTION
    # ─────────────────────────────────────────

    def _trade_suggestion(
        self,
        composite: float,
        spread_pct: float,
        bid_walls: List[LiquidityWall],
        ask_walls: List[LiquidityWall],
        vpin: VPINState = None,
        swing: SwingSignal = None,
    ) -> Optional[str]:
        """
        Generate a trade suggestion. CANDLE TREND is the primary driver.

        For swing trades, walls are optional confirmation, not required.
        The candle setup (trend + RSI + ATR) is what triggers the trade.
        A supporting wall just boosts confidence.

        Min confidence raised to 0.55 (from 0.30). OB imbalance can add +0.05.
        """
        max_spread = self.config.trading.max_spread_pct
        if spread_pct > max_spread:
            return None

        # VPIN gate: only block at extreme toxicity spike
        if vpin and vpin.should_block_entry:
            return None

        # ADX block: candle_analyzer already set suggested_side=None,
        # but double-check here
        if swing and swing.adx_blocked:
            return None

        COMPOSITE_THRESHOLD = 0.25
        MIN_CONFIDENCE = 0.55

        if composite >= COMPOSITE_THRESHOLD:
            if swing and swing.suggested_side == "BUY" and swing.confidence >= MIN_CONFIDENCE:
                return "BUY"
            # OB-only trade needs a wall AND higher threshold (no change)
            genuine_bid_walls = [w for w in bid_walls if not w.is_spoof_suspect and w.confidence > 0.5]
            if genuine_bid_walls and composite >= 0.5:
                return "BUY"

        elif composite <= -COMPOSITE_THRESHOLD:
            if swing and swing.suggested_side == "SELL" and swing.confidence >= MIN_CONFIDENCE:
                return "SELL"
            genuine_ask_walls = [w for w in ask_walls if not w.is_spoof_suspect and w.confidence > 0.5]
            if genuine_ask_walls and composite <= -0.5:
                return "SELL"

        return None

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _empty_result(self, symbol: str, timestamp: float) -> AnalysisResult:
        return AnalysisResult(
            symbol=symbol,
            timestamp=timestamp,
            mid_price=0.0,
            spread_pct=0.0,
            bid_walls=[],
            ask_walls=[],
            imbalance=ImbalanceSignal(0, 0, 0, SignalDirection.NEUTRAL, 0, timestamp),
            vwap=VWAPState(timestamp=timestamp),
            momentum=MomentumSignal(timestamp=timestamp),
            vpin=VPINState(timestamp=timestamp),
        )

    def get_analysis_summary(self, result: AnalysisResult) -> str:
        """Human-readable summary of an analysis result."""
        lines = [
            f"═══ {result.symbol} @ {result.mid_price:.2f} ═══",
            f"  Spread: {result.spread_pct:.4f}%",
            f"  Composite Score: {result.composite_score:+.3f}",
            f"  Suggestion: {result.trade_suggestion or 'HOLD'}",
            "",
            f"  ── Bid Walls ({len(result.bid_walls)}) ──",
        ]
        for w in result.bid_walls:
            spoof = " ⚠️SPOOF?" if w.is_spoof_suspect else ""
            lines.append(
                f"    ${w.price:,.2f} | {w.volume:.4f} | "
                f"${w.usd_value:,.0f} | {w.multiplier:.1f}x | "
                f"conf={w.confidence:.2f}{spoof}"
            )

        lines.append(f"  ── Ask Walls ({len(result.ask_walls)}) ──")
        for w in result.ask_walls:
            spoof = " ⚠️SPOOF?" if w.is_spoof_suspect else ""
            lines.append(
                f"    ${w.price:,.2f} | {w.volume:.4f} | "
                f"${w.usd_value:,.0f} | {w.multiplier:.1f}x | "
                f"conf={w.confidence:.2f}{spoof}"
            )

        lines.extend([
            "",
            f"  Imbalance: {result.imbalance.ratio:.2f} ({result.imbalance.direction.value})",
            f"  VWAP: {result.vwap.vwap:.2f} | Dev: {result.vwap.deviation_pct:+.3f}%",
            f"  Momentum: aggr={result.momentum.aggressor_ratio:.2f} | "
            f"net_flow={result.momentum.net_flow:+.4f} | "
            f"large_trades={result.momentum.large_trade_count} "
            f"({result.momentum.direction.value})",
            self._vpin.get_summary(result.vpin, result.symbol),
        ])

        return "\n".join(lines)
