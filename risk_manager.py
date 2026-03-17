"""
Risk Manager — Global risk controls and circuit breakers.

Enforces:
- Daily loss limits (USD and %)
- Max drawdown from peak equity
- Trade count circuit breaker
- Cooldown after consecutive losses
- Spread filter validation
- Position sizing constraints
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade for risk tracking."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    pnl_usd: float
    reason: str              # "take_profit", "stop_loss", "trailing_stop", "wall_pulled"
    timestamp: float = 0.0


@dataclass
class RiskState:
    """Current risk state for the session."""
    starting_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    last_loss_time: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""
    session_start: float = 0.0
    trades_today: List[TradeRecord] = field(default_factory=list)


class RiskManager:
    """
    Centralized risk management enforcing hard limits on losses,
    drawdowns, and trade frequency.
    """

    def __init__(self, initial_equity: float, config: BotConfig = CONFIG):
        self.config = config
        self.rc = config.risk
        self.tc = config.trading

        self.state = RiskState(
            starting_equity=initial_equity,
            current_equity=initial_equity,
            peak_equity=initial_equity,
            session_start=time.time(),
        )

    # ─────────────────────────────────────────
    # PRE-TRADE CHECKS
    # ─────────────────────────────────────────

    def can_trade(self) -> tuple:
        """
        Check all risk conditions before allowing a trade.

        Returns:
            (allowed: bool, reason: str)
        """
        if self.state.is_halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        # Daily loss limit (USD)
        if abs(self.state.daily_pnl) >= self.rc.max_daily_loss_usd and self.state.daily_pnl < 0:
            self._halt(f"Daily USD loss limit hit: ${self.state.daily_pnl:.2f}")
            return False, self.state.halt_reason

        # Daily loss limit (%)
        daily_loss_pct = (self.state.daily_pnl / self.state.starting_equity) * 100.0
        if daily_loss_pct <= -self.rc.max_daily_loss_pct:
            self._halt(f"Daily loss % limit hit: {daily_loss_pct:.2f}%")
            return False, self.state.halt_reason

        # Max drawdown from peak
        drawdown_pct = ((self.state.peak_equity - self.state.current_equity)
                        / self.state.peak_equity * 100.0)
        if drawdown_pct >= self.rc.max_drawdown_pct:
            self._halt(f"Max drawdown hit: {drawdown_pct:.2f}%")
            return False, self.state.halt_reason

        # Trade count circuit breaker
        if self.state.daily_trade_count >= self.rc.max_daily_trades:
            return False, f"Daily trade limit reached: {self.state.daily_trade_count}"

        # Cooldown after loss
        if self.state.consecutive_losses > 0:
            time_since_loss = time.time() - self.state.last_loss_time
            if time_since_loss < self.rc.cooldown_after_loss_s:
                remaining = self.rc.cooldown_after_loss_s - time_since_loss
                return False, f"Cooldown active: {remaining:.1f}s remaining"

        return True, "OK"

    def validate_spread(self, spread_pct: float) -> bool:
        """Check if the current spread is acceptable for scalping."""
        if spread_pct > self.tc.max_spread_pct:
            logger.debug(f"Spread too wide: {spread_pct:.4f}% > {self.tc.max_spread_pct}%")
            return False
        return True

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate the maximum position size in base currency.

        Uses the smaller of:
        - max_position_usd
        - position_pct_of_equity * current_equity

        Adjusts down based on consecutive losses (risk scaling).
        """
        max_usd = min(
            self.tc.max_position_usd,
            self.state.current_equity * self.tc.position_pct_of_equity,
        )

        # Scale down after consecutive losses (Kelly-inspired reduction)
        if self.state.consecutive_losses >= 2:
            scale = max(0.25, 1.0 - (self.state.consecutive_losses * 0.2))
            max_usd *= scale
            logger.info(
                f"Position scaled down to {scale:.0%} due to "
                f"{self.state.consecutive_losses} consecutive losses"
            )

        amount = max_usd / price if price > 0 else 0.0
        return amount

    # ─────────────────────────────────────────
    # POST-TRADE UPDATES
    # ─────────────────────────────────────────

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update risk state."""
        self.state.trades_today.append(trade)
        self.state.daily_trade_count += 1
        self.state.daily_pnl += trade.pnl_usd
        self.state.current_equity += trade.pnl_usd

        # Track peak equity
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

        # Track consecutive losses
        if trade.pnl_usd < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = time.time()
            logger.warning(
                f"Loss recorded: ${trade.pnl_usd:.2f} | "
                f"Consecutive losses: {self.state.consecutive_losses} | "
                f"Daily PnL: ${self.state.daily_pnl:.2f}"
            )
        else:
            self.state.consecutive_losses = 0
            logger.info(
                f"Win recorded: ${trade.pnl_usd:+.2f} | "
                f"Daily PnL: ${self.state.daily_pnl:+.2f}"
            )

    # ─────────────────────────────────────────
    # STATE MANAGEMENT
    # ─────────────────────────────────────────

    def _halt(self, reason: str):
        """Halt all trading."""
        self.state.is_halted = True
        self.state.halt_reason = reason
        logger.critical(f"🛑 TRADING HALTED: {reason}")

    def reset_daily(self, new_equity: Optional[float] = None):
        """Reset daily counters (call at the start of each trading day)."""
        if new_equity is not None:
            self.state.starting_equity = new_equity
            self.state.current_equity = new_equity
            self.state.peak_equity = max(self.state.peak_equity, new_equity)

        self.state.daily_pnl = 0.0
        self.state.daily_trade_count = 0
        self.state.trades_today.clear()
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.consecutive_losses = 0
        logger.info("Daily risk counters reset.")

    def get_risk_summary(self) -> str:
        """Human-readable risk status."""
        dd = ((self.state.peak_equity - self.state.current_equity)
              / self.state.peak_equity * 100.0) if self.state.peak_equity > 0 else 0.0

        wins = sum(1 for t in self.state.trades_today if t.pnl_usd > 0)
        losses = sum(1 for t in self.state.trades_today if t.pnl_usd <= 0)
        total = wins + losses
        win_rate = (wins / total * 100.0) if total > 0 else 0.0

        avg_win = 0.0
        avg_loss = 0.0
        if wins > 0:
            avg_win = sum(t.pnl_usd for t in self.state.trades_today if t.pnl_usd > 0) / wins
        if losses > 0:
            avg_loss = sum(t.pnl_usd for t in self.state.trades_today if t.pnl_usd <= 0) / losses

        return (
            f"═══ Risk Status ═══\n"
            f"  Equity: ${self.state.current_equity:,.2f} "
            f"(peak: ${self.state.peak_equity:,.2f})\n"
            f"  Drawdown: {dd:.2f}%\n"
            f"  Daily PnL: ${self.state.daily_pnl:+,.2f}\n"
            f"  Trades: {total} (W:{wins} L:{losses} | WR:{win_rate:.1f}%)\n"
            f"  Avg Win: ${avg_win:+.2f} | Avg Loss: ${avg_loss:.2f}\n"
            f"  Consecutive Losses: {self.state.consecutive_losses}\n"
            f"  Halted: {self.state.is_halted}"
            + (f" — {self.state.halt_reason}" if self.state.is_halted else "")
        )
