"""
Cross-Pair Correlation Guard & Portfolio Exposure Management.

Three components:
1. CorrelationMatrix — rolling 30-day return correlation between all traded pairs.
   Updated daily, cached with 24h TTL, persisted to bot_state.
2. CorrelationGuard — before opening a 2nd position, checks correlation with
   existing positions: >0.80 blocks, 0.60-0.80 halves size, <0.60 allows full.
   Opposite-direction on correlated pairs always allowed (natural hedge).
3. PortfolioHeatMap — tracks total directional exposure across all positions.
   Blocks new positions if net long or short exceeds 20× capital.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class CorrelationResult:
    """Result of correlation check for a proposed trade."""
    allowed: bool = True
    size_multiplier: float = 1.0         # 1.0 = full, 0.5 = reduced
    reason: str = ""
    correlated_with: str = ""            # Which existing position triggered the check
    correlation_value: float = 0.0       # The actual correlation coefficient
    is_natural_hedge: bool = False       # Opposite direction on correlated pair
    is_strong_exception: bool = False    # Both signals strong enough to override


@dataclass
class ExposureState:
    """Current portfolio directional exposure."""
    net_long_exposure: float = 0.0       # Sum of (size × leverage) for longs
    net_short_exposure: float = 0.0      # Sum of (size × leverage) for shorts
    net_exposure: float = 0.0            # net_long - net_short
    gross_exposure: float = 0.0          # net_long + net_short
    directional_bias: str = "neutral"    # "long", "short", or "neutral"
    positions: List[dict] = field(default_factory=list)  # Per-position breakdown
    breach_long: bool = False            # Would breach max long
    breach_short: bool = False           # Would breach max short


# ─────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────

class CorrelationMatrix:
    """
    Computes and caches rolling return correlation between all traded pairs.

    Uses daily close prices to compute log returns, then a Pearson correlation
    matrix. Updated daily with a 24h TTL cache.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.cc = config.correlation

        # pairs → list of daily close prices (most recent last)
        self._price_history: Dict[str, List[float]] = {}

        # The computed correlation matrix: (pair_a, pair_b) → correlation
        self._matrix: Dict[Tuple[str, str], float] = {}

        # Cache timestamp
        self._last_computed: float = 0.0

    @property
    def matrix(self) -> Dict[Tuple[str, str], float]:
        """Return the current correlation matrix."""
        return self._matrix

    def update_prices(self, daily_closes: Dict[str, List[float]]):
        """
        Update price history from daily candle closes.

        Args:
            daily_closes: {symbol: [close_prices]} — most recent last.
                          Each list should have at least lookback_days entries.
        """
        for symbol, closes in daily_closes.items():
            self._price_history[symbol] = closes

    def add_daily_close(self, symbol: str, close_price: float):
        """Append a single daily close for incremental updates."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(close_price)
        # Trim to 2× lookback to avoid unbounded growth
        max_len = self.cc.lookback_days * 2
        if len(self._price_history[symbol]) > max_len:
            self._price_history[symbol] = self._price_history[symbol][-max_len:]

    def compute(self) -> Dict[Tuple[str, str], float]:
        """
        Compute the correlation matrix from stored price history.

        Returns:
            Dict mapping (symbol_a, symbol_b) → Pearson correlation coefficient.
            Self-correlations (a, a) = 1.0 are included.
        """
        symbols = sorted(self._price_history.keys())
        n = len(symbols)

        if n < 2:
            self._matrix = {(s, s): 1.0 for s in symbols}
            self._last_computed = time.time()
            return self._matrix

        lookback = self.cc.lookback_days
        min_candles = self.cc.min_candles

        # Build aligned return arrays
        returns = {}
        for sym in symbols:
            prices = self._price_history.get(sym, [])
            if len(prices) < min_candles:
                continue
            # Use last `lookback` prices
            p = np.array(prices[-lookback:], dtype=np.float64)
            if len(p) < 2:
                continue
            # Log returns
            log_ret = np.diff(np.log(p))
            returns[sym] = log_ret

        # Find common length (all return series must align)
        if len(returns) < 2:
            self._matrix = {(s, s): 1.0 for s in symbols}
            self._last_computed = time.time()
            return self._matrix

        min_len = min(len(r) for r in returns.values())
        if min_len < min_candles - 1:
            self._matrix = {(s, s): 1.0 for s in symbols}
            self._last_computed = time.time()
            return self._matrix

        # Trim all to common length (from the end — most recent)
        aligned = {}
        for sym, ret in returns.items():
            aligned[sym] = ret[-min_len:]

        # Build correlation matrix
        sym_list = sorted(aligned.keys())
        n_syms = len(sym_list)
        data = np.array([aligned[s] for s in sym_list])  # shape: (n_syms, min_len)

        # NumPy corrcoef
        corr = np.corrcoef(data)

        self._matrix = {}
        for i in range(n_syms):
            for j in range(n_syms):
                val = float(corr[i, j])
                if np.isnan(val):
                    val = 0.0
                self._matrix[(sym_list[i], sym_list[j])] = val

        self._last_computed = time.time()
        return self._matrix

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get correlation between two symbols. Returns 0.0 if unknown."""
        if symbol_a == symbol_b:
            return 1.0
        return self._matrix.get(
            (symbol_a, symbol_b),
            self._matrix.get((symbol_b, symbol_a), 0.0)
        )

    def is_stale(self) -> bool:
        """Check if the matrix needs recomputation."""
        if not self._matrix:
            return True
        return (time.time() - self._last_computed) > self.cc.cache_ttl

    def get_matrix_dict(self) -> dict:
        """Serialize matrix for JSON/bot_state storage."""
        return {
            "matrix": {f"{a}|{b}": v for (a, b), v in self._matrix.items()},
            "computed_at": self._last_computed,
            "pairs": sorted(set(a for a, _ in self._matrix.keys())),
        }

    def load_matrix_dict(self, data: dict):
        """Load matrix from JSON/bot_state."""
        if not data:
            return
        matrix_raw = data.get("matrix", {})
        self._matrix = {}
        for key, val in matrix_raw.items():
            parts = key.split("|")
            if len(parts) == 2:
                self._matrix[(parts[0], parts[1])] = float(val)
        self._last_computed = data.get("computed_at", 0.0)


# ─────────────────────────────────────────
# CORRELATION GUARD
# ─────────────────────────────────────────

class CorrelationGuard:
    """
    Checks whether a new position should be allowed given existing positions
    and the cross-pair correlation matrix.

    Rules:
    - correlation > 0.80 (same direction): BLOCK
    - correlation 0.60-0.80 (same direction): allow at 50% size
    - correlation < 0.60: allow full size
    - Opposite direction on correlated pairs: ALWAYS allow (natural hedge)
    - Strong signal exception: if both signals > 0.75 confidence,
      allow correlated same-direction but cap total exposure at 1.5× single
    """

    def __init__(self, matrix: CorrelationMatrix, config: BotConfig = CONFIG):
        self.matrix = matrix
        self.config = config
        self.cc = config.correlation

    def check(
        self,
        new_symbol: str,
        new_side: str,
        new_confidence: float,
        open_positions: Dict[str, dict],
    ) -> CorrelationResult:
        """
        Check if a new trade should be allowed.

        Args:
            new_symbol: Symbol to open (e.g., "SOL/USDT")
            new_side: "BUY" or "SELL"
            new_confidence: Confidence of the new signal
            open_positions: {symbol: {"side": str, "confidence": float, ...}}
                           Currently open positions.

        Returns:
            CorrelationResult with allowed/blocked/size_multiplier.
        """
        if not self.cc.enabled:
            return CorrelationResult(allowed=True, reason="correlation guard disabled")

        if not open_positions:
            return CorrelationResult(allowed=True, reason="no existing positions")

        # Check against each open position
        worst_result = CorrelationResult(allowed=True, size_multiplier=1.0)

        for existing_sym, pos_info in open_positions.items():
            if existing_sym == new_symbol:
                continue  # Already checked by shadow_trader (no stacking)

            existing_side = pos_info.get("side", "")
            existing_conf = pos_info.get("confidence", 0.0)

            corr = self.matrix.get_correlation(new_symbol, existing_sym)
            abs_corr = abs(corr)

            same_direction = (new_side == existing_side)
            opposite_direction = (new_side != existing_side)

            # Rule: opposite direction on correlated pairs → always allow (natural hedge)
            if opposite_direction and abs_corr >= self.cc.medium_corr_threshold:
                result = CorrelationResult(
                    allowed=True,
                    size_multiplier=1.0,
                    reason=f"natural hedge: opposite direction on correlated pair ({corr:.2f})",
                    correlated_with=existing_sym,
                    correlation_value=corr,
                    is_natural_hedge=True,
                )
                # Natural hedge doesn't restrict — continue checking other positions
                continue

            # Same direction checks
            if same_direction and abs_corr >= self.cc.high_corr_threshold:
                # Strong signal exception
                both_strong = (
                    new_confidence >= self.cc.strong_signal_min_conf
                    and existing_conf >= self.cc.strong_signal_min_conf
                )
                if both_strong:
                    result = CorrelationResult(
                        allowed=True,
                        size_multiplier=self.cc.strong_signal_max_exposure - 1.0,
                        reason=(
                            f"high corr ({corr:.2f}) with {existing_sym} same direction, "
                            f"but both signals strong — capped at "
                            f"{self.cc.strong_signal_max_exposure}× exposure"
                        ),
                        correlated_with=existing_sym,
                        correlation_value=corr,
                        is_strong_exception=True,
                    )
                else:
                    result = CorrelationResult(
                        allowed=False,
                        size_multiplier=0.0,
                        reason=(
                            f"BLOCKED: high correlation ({corr:.2f}) with {existing_sym} "
                            f"same direction — too similar exposure"
                        ),
                        correlated_with=existing_sym,
                        correlation_value=corr,
                    )
                    return result  # Hard block — no need to check further

            elif same_direction and abs_corr >= self.cc.medium_corr_threshold:
                result = CorrelationResult(
                    allowed=True,
                    size_multiplier=self.cc.medium_corr_size_mult,
                    reason=(
                        f"medium correlation ({corr:.2f}) with {existing_sym} "
                        f"same direction — size reduced to "
                        f"{self.cc.medium_corr_size_mult * 100:.0f}%"
                    ),
                    correlated_with=existing_sym,
                    correlation_value=corr,
                )
            else:
                result = CorrelationResult(
                    allowed=True,
                    size_multiplier=1.0,
                    reason=f"low correlation ({corr:.2f}) with {existing_sym}",
                    correlated_with=existing_sym,
                    correlation_value=corr,
                )

            # Keep the most restrictive result
            if result.size_multiplier < worst_result.size_multiplier:
                worst_result = result

        return worst_result


# ─────────────────────────────────────────
# PORTFOLIO HEAT MAP
# ─────────────────────────────────────────

class PortfolioHeatMap:
    """
    Tracks total directional exposure across all open positions.

    Exposure = sum of (position_size_usd × leverage × direction)
    where direction = +1 for long, -1 for short.

    Caps net exposure at ±20× capital to prevent excessive directional risk.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.cc = config.correlation

    def compute_exposure(
        self,
        open_positions: Dict[str, dict],
        equity: float,
    ) -> ExposureState:
        """
        Compute current portfolio exposure.

        Args:
            open_positions: {symbol: {"side": str, "usd_value": float,
                            "leverage": int, "amount": float, "entry_price": float}}
            equity: Current equity in USD.

        Returns:
            ExposureState with exposure breakdown.
        """
        state = ExposureState()
        if equity <= 0:
            return state

        for symbol, pos in open_positions.items():
            side = pos.get("side", "")
            usd_value = pos.get("usd_value", 0.0)
            leverage = pos.get("leverage", 1)
            notional = usd_value * leverage

            direction = 1.0 if side == "BUY" else -1.0
            exposure = notional / equity  # As multiple of capital

            if side == "BUY":
                state.net_long_exposure += exposure
            else:
                state.net_short_exposure += exposure

            state.positions.append({
                "symbol": symbol,
                "side": side,
                "usd_value": usd_value,
                "leverage": leverage,
                "notional": notional,
                "exposure_x": exposure,
                "direction": direction,
            })

        state.net_exposure = state.net_long_exposure - state.net_short_exposure
        state.gross_exposure = state.net_long_exposure + state.net_short_exposure

        if abs(state.net_exposure) < 0.1:
            state.directional_bias = "neutral"
        elif state.net_exposure > 0:
            state.directional_bias = "long"
        else:
            state.directional_bias = "short"

        return state

    def check_can_add(
        self,
        new_side: str,
        new_usd_value: float,
        new_leverage: int,
        open_positions: Dict[str, dict],
        equity: float,
    ) -> Tuple[bool, float, str]:
        """
        Check if adding a new position would breach exposure limits.

        Returns:
            (allowed, max_size_fraction, reason)
            max_size_fraction: 1.0 if full size OK, <1.0 if must reduce, 0.0 if blocked.
        """
        if not self.cc.enabled or equity <= 0:
            return True, 1.0, ""

        state = self.compute_exposure(open_positions, equity)

        new_notional = new_usd_value * new_leverage
        new_exposure = new_notional / equity

        if new_side == "BUY":
            projected_long = state.net_long_exposure + new_exposure
            if projected_long > self.cc.max_net_long_exposure:
                remaining = self.cc.max_net_long_exposure - state.net_long_exposure
                if remaining <= 0:
                    return False, 0.0, (
                        f"net long exposure would exceed {self.cc.max_net_long_exposure}× "
                        f"(current: {state.net_long_exposure:.1f}×)"
                    )
                fraction = remaining / new_exposure
                return True, min(fraction, 1.0), (
                    f"long exposure capped: reduced to {fraction * 100:.0f}% "
                    f"(current: {state.net_long_exposure:.1f}×, "
                    f"max: {self.cc.max_net_long_exposure}×)"
                )
        else:
            projected_short = state.net_short_exposure + new_exposure
            if projected_short > self.cc.max_net_short_exposure:
                remaining = self.cc.max_net_short_exposure - state.net_short_exposure
                if remaining <= 0:
                    return False, 0.0, (
                        f"net short exposure would exceed {self.cc.max_net_short_exposure}× "
                        f"(current: {state.net_short_exposure:.1f}×)"
                    )
                fraction = remaining / new_exposure
                return True, min(fraction, 1.0), (
                    f"short exposure capped: reduced to {fraction * 100:.0f}% "
                    f"(current: {state.net_short_exposure:.1f}×, "
                    f"max: {self.cc.max_net_short_exposure}×)"
                )

        return True, 1.0, ""

    def get_exposure_summary(
        self,
        open_positions: Dict[str, dict],
        equity: float,
    ) -> dict:
        """Get full exposure state as a JSON-serializable dict for the dashboard."""
        state = self.compute_exposure(open_positions, equity)
        return {
            "net_long_x": round(state.net_long_exposure, 2),
            "net_short_x": round(state.net_short_exposure, 2),
            "net_exposure_x": round(state.net_exposure, 2),
            "gross_exposure_x": round(state.gross_exposure, 2),
            "directional_bias": state.directional_bias,
            "max_long_x": self.cc.max_net_long_exposure,
            "max_short_x": self.cc.max_net_short_exposure,
            "positions": state.positions,
        }
