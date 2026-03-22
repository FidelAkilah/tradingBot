"""
Position Sizer — Dynamic position sizing using Kelly Criterion.

Combines multiple sizing signals into a single position size:
1. Kelly Criterion (rolling win rate + avg W/L ratio, half-Kelly)
2. Confidence-weighted multiplier
3. Drawdown-based scaling
4. Consecutive loss exponential reduction
5. Dynamic leverage based on confidence

Position size = kelly_pct * confidence_mult * drawdown_mult * consec_loss_mult * equity
Leverage = confidence-mapped value (5x to 12x, hard cap 15x)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Output of position sizing calculation — logs the rationale."""
    position_usd: float = 0.0          # Final margin (USD) to allocate
    leverage: int = 10                  # Dynamic leverage for this trade
    notional_usd: float = 0.0          # position_usd * leverage

    # Component breakdown (for logging)
    kelly_fraction: float = 0.0        # Raw Kelly f
    kelly_pct: float = 0.10            # Half-Kelly (or default)
    confidence_mult: float = 1.0
    drawdown_mult: float = 1.0
    drawdown_pct: float = 0.0
    consec_loss_mult: float = 1.0
    consecutive_losses: int = 0
    regime_mult: float = 1.0
    session_mult: float = 1.0

    # Flags
    is_halted: bool = False
    halt_reason: str = ""
    used_kelly: bool = False           # True if enough trades for Kelly

    def to_dict(self) -> dict:
        return {
            "position_usd": round(self.position_usd, 4),
            "leverage": self.leverage,
            "notional_usd": round(self.notional_usd, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "kelly_pct": round(self.kelly_pct, 4),
            "confidence_mult": round(self.confidence_mult, 4),
            "drawdown_mult": round(self.drawdown_mult, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "consec_loss_mult": round(self.consec_loss_mult, 4),
            "consecutive_losses": self.consecutive_losses,
            "regime_mult": round(self.regime_mult, 4),
            "session_mult": round(self.session_mult, 4),
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "used_kelly": self.used_kelly,
        }


@dataclass
class TradeOutcome:
    """Minimal record of a closed trade for Kelly calculation."""
    pnl_usd: float
    is_win: bool


class PositionSizer:
    """
    Computes dynamic position sizes combining Kelly Criterion with
    confidence weighting, drawdown scaling, and consecutive loss adjustment.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.tc = config.trading
        self.fc = config.futures
        self.rc = config.risk

        # Rolling trade history for Kelly
        self._trade_history: deque = deque(maxlen=self.tc.kelly_lookback)

        # Per-symbol consecutive loss tracking
        self._consecutive_losses: dict = {}  # symbol -> count
        self._symbol_halt_until: dict = {}   # symbol -> timestamp

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def calculate(
        self,
        equity: float,
        peak_equity: float,
        confidence: float,
        symbol: str = "",
        regime_mult: float = 1.0,
        session_mult: float = 1.0,
        current_time: float = 0.0,
    ) -> SizingResult:
        """
        Calculate position size for a potential trade.

        Args:
            equity: Current account equity (USD)
            peak_equity: Peak equity for drawdown calculation
            confidence: Signal confidence score (0.55-1.0)
            symbol: Trading pair (for per-symbol consecutive loss tracking)
            regime_mult: Regime-based size multiplier (from market_regime)
            session_mult: Session-based size multiplier (from session_filter)
            current_time: Current timestamp for halt checking

        Returns:
            SizingResult with position size, leverage, and breakdown
        """
        result = SizingResult()
        result.regime_mult = regime_mult
        result.session_mult = session_mult

        # Check per-symbol halt
        if symbol and symbol in self._symbol_halt_until:
            if current_time < self._symbol_halt_until[symbol]:
                result.is_halted = True
                result.halt_reason = f"{symbol} halted after {self.rc.consec_loss_halt_count}+ losses"
                return result
            else:
                # Halt expired, clear it
                del self._symbol_halt_until[symbol]

        # 1. Kelly Criterion base sizing
        kelly_f, kelly_pct, used_kelly = self._kelly_fraction()
        result.kelly_fraction = kelly_f
        result.kelly_pct = kelly_pct
        result.used_kelly = used_kelly

        # 2. Confidence multiplier
        result.confidence_mult = self._confidence_multiplier(confidence)

        # 3. Drawdown scaling
        result.drawdown_pct = self._drawdown_pct(equity, peak_equity)
        result.drawdown_mult = self._drawdown_multiplier(result.drawdown_pct)

        # Check drawdown halt
        if result.drawdown_pct >= self.rc.drawdown_halt:
            result.is_halted = True
            result.halt_reason = f"Drawdown {result.drawdown_pct:.1f}% >= {self.rc.drawdown_halt}% halt"
            return result

        # 4. Consecutive loss scaling
        consec = self._consecutive_losses.get(symbol, 0)
        result.consecutive_losses = consec
        result.consec_loss_mult = self._consecutive_loss_multiplier(consec)

        # 5. Dynamic leverage
        result.leverage = self._dynamic_leverage(confidence)

        # 6. Combine all multipliers
        base_pct = kelly_pct  # Already half-Kelly or default
        combined_mult = (
            result.confidence_mult
            * result.drawdown_mult
            * result.consec_loss_mult
            * regime_mult
            * session_mult
        )

        position_pct = base_pct * combined_mult

        # Floor and cap
        position_pct = max(position_pct, self.tc.min_position_pct)
        position_pct = min(position_pct, self.tc.max_position_pct)

        # If drawdown_mult forced us to floor, use minimum
        if result.drawdown_mult == 0.0:
            position_pct = self.tc.min_position_pct

        result.position_usd = equity * position_pct
        result.notional_usd = result.position_usd * result.leverage

        return result

    def record_outcome(self, symbol: str, pnl_usd: float, current_time: float = 0.0):
        """
        Record a trade outcome to update Kelly stats and consecutive loss tracking.
        Call this after every closed trade.
        """
        is_win = pnl_usd > 0
        self._trade_history.append(TradeOutcome(pnl_usd=pnl_usd, is_win=is_win))

        if is_win:
            # Reset consecutive losses on any win
            self._consecutive_losses[symbol] = 0
        else:
            consec = self._consecutive_losses.get(symbol, 0) + 1
            self._consecutive_losses[symbol] = consec

            # Check for cooldown/halt thresholds
            if consec >= self.rc.consec_loss_halt_count:
                self._symbol_halt_until[symbol] = current_time + self.rc.consec_loss_halt_s
                logger.warning(
                    f"[{symbol}] {consec} consecutive losses — halted for "
                    f"{self.rc.consec_loss_halt_s / 60:.0f} minutes"
                )

    def get_consecutive_losses(self, symbol: str) -> int:
        return self._consecutive_losses.get(symbol, 0)

    # ─────────────────────────────────────────
    # KELLY CRITERION
    # ─────────────────────────────────────────

    def _kelly_fraction(self):
        """
        Calculate Kelly fraction from rolling trade history.

        f = W - (1 - W) / R
        where W = win rate, R = avg_win / avg_loss

        Returns: (raw_kelly_f, half_kelly_pct, used_kelly_bool)
        """
        n = len(self._trade_history)
        if n < self.tc.kelly_min_trades:
            # Not enough history — use conservative default
            return 0.0, self.tc.kelly_default_pct, False

        wins = [t for t in self._trade_history if t.is_win]
        losses = [t for t in self._trade_history if not t.is_win]

        if not wins or not losses:
            # All wins or all losses — can't compute meaningful Kelly
            if not losses:
                # All wins — use max position
                return 1.0, self.tc.max_position_pct, True
            # All losses — use minimum
            return 0.0, self.tc.min_position_pct, True

        win_rate = len(wins) / n
        avg_win = sum(t.pnl_usd for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl_usd for t in losses) / len(losses))

        if avg_loss == 0:
            return 1.0, self.tc.max_position_pct, True

        r = avg_win / avg_loss  # Win/loss ratio
        kelly_f = win_rate - (1.0 - win_rate) / r

        # Half-Kelly for safety
        half_kelly = kelly_f * self.tc.kelly_fraction

        # Clamp to [min_position_pct, max_position_pct]
        half_kelly = max(half_kelly, self.tc.min_position_pct)
        half_kelly = min(half_kelly, self.tc.max_position_pct)

        return kelly_f, half_kelly, True

    # ─────────────────────────────────────────
    # CONFIDENCE MULTIPLIER
    # ─────────────────────────────────────────

    @staticmethod
    def _confidence_multiplier(confidence: float) -> float:
        """
        Scale position size based on signal confidence.
        Higher confidence → more capital.

        0.55-0.65: 0.6x
        0.65-0.75: 0.8x
        0.75-0.85: 1.0x
        0.85+:     1.2x
        """
        if confidence >= 0.85:
            return 1.2
        if confidence >= 0.75:
            return 1.0
        if confidence >= 0.65:
            return 0.8
        return 0.6  # 0.55-0.65

    # ─────────────────────────────────────────
    # DYNAMIC LEVERAGE
    # ─────────────────────────────────────────

    def _dynamic_leverage(self, confidence: float) -> int:
        """
        Map confidence score to leverage level.

        0.55-0.65: 5x
        0.65-0.75: 8x
        0.75-0.85: 10x
        0.85+:     12x

        Never exceeds max_leverage (default 15x).
        """
        if confidence >= 0.85:
            lev = 12
        elif confidence >= 0.75:
            lev = 10
        elif confidence >= 0.65:
            lev = 8
        else:
            lev = 5

        return min(lev, self.fc.max_leverage)

    # ─────────────────────────────────────────
    # DRAWDOWN SCALING
    # ─────────────────────────────────────────

    @staticmethod
    def _drawdown_pct(equity: float, peak_equity: float) -> float:
        if peak_equity <= 0:
            return 0.0
        return (peak_equity - equity) / peak_equity * 100.0

    def _drawdown_multiplier(self, dd_pct: float) -> float:
        """
        Scale position size based on current drawdown from peak.

        0-5%:   1.0 (normal)
        5-10%:  0.7 (reduce 30%)
        10-15%: 0.4 (reduce 60%)
        15-25%: 0.0 (floor to minimum size)
        >25%:   halted (handled in calculate())
        """
        if dd_pct < 5.0:
            return self.rc.drawdown_scale_5
        if dd_pct < 10.0:
            return self.rc.drawdown_scale_10
        if dd_pct < 15.0:
            return self.rc.drawdown_scale_15
        return self.rc.drawdown_scale_25  # 0.0 → will be floored to min

    # ─────────────────────────────────────────
    # CONSECUTIVE LOSS SCALING
    # ─────────────────────────────────────────

    def _consecutive_loss_multiplier(self, consecutive_losses: int) -> float:
        """
        Exponential size reduction: 0.7 ^ consecutive_losses.

        0 losses: 1.0
        1 loss:   0.7
        2 losses: 0.49
        3 losses: 0.343 (+ 30min cooldown)
        4+ losses: 0.24 (+ 2hr halt)
        """
        if consecutive_losses <= 0:
            return 1.0
        return self.rc.consec_loss_base ** consecutive_losses

    # ─────────────────────────────────────────
    # STATE ACCESS (for API/dashboard)
    # ─────────────────────────────────────────

    def get_state(self) -> dict:
        """Current position sizer state for dashboard."""
        kelly_f, kelly_pct, used_kelly = self._kelly_fraction()

        n = len(self._trade_history)
        wins = sum(1 for t in self._trade_history if t.is_win)
        losses = n - wins
        win_rate = wins / n * 100.0 if n > 0 else 0.0

        avg_win = 0.0
        avg_loss = 0.0
        win_trades = [t for t in self._trade_history if t.is_win]
        loss_trades = [t for t in self._trade_history if not t.is_win]
        if win_trades:
            avg_win = sum(t.pnl_usd for t in win_trades) / len(win_trades)
        if loss_trades:
            avg_loss = abs(sum(t.pnl_usd for t in loss_trades) / len(loss_trades))

        return {
            "kelly_fraction": round(kelly_f, 4),
            "kelly_pct": round(kelly_pct, 4),
            "used_kelly": used_kelly,
            "rolling_trades": n,
            "rolling_wins": wins,
            "rolling_losses": losses,
            "rolling_win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "win_loss_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else 0.0,
            "consecutive_losses": dict(self._consecutive_losses),
            "halted_pairs": {k: v for k, v in self._symbol_halt_until.items()},
        }
