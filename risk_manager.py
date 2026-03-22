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

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import BotConfig, CONFIG
from position_sizer import PositionSizer, SizingResult

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

        # Position sizer for Kelly + drawdown + consecutive loss logic
        self.sizer = PositionSizer(config)

    # ─────────────────────────────────────────
    # PRE-TRADE CHECKS
    # ─────────────────────────────────────────

    def can_trade(self) -> tuple:
        """
        Check all risk conditions before allowing a trade.

        Returns:
            (allowed: bool, reason: str)
        """
        # Auto-reset daily counters at UTC midnight
        self._check_daily_reset()

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

        # Cooldown after loss — escalating based on consecutive losses
        if self.state.consecutive_losses > 0:
            # 3+ losses: extended cooldown
            if self.state.consecutive_losses >= self.rc.consec_loss_cooldown_count:
                cooldown = self.rc.consec_loss_cooldown_s  # 30 min
            else:
                cooldown = self.rc.cooldown_after_loss_s   # 5 min default

            time_since_loss = time.time() - self.state.last_loss_time
            if time_since_loss < cooldown:
                remaining = cooldown - time_since_loss
                return False, f"Cooldown active: {remaining:.1f}s remaining ({self.state.consecutive_losses} consec losses)"

        # Drawdown-based halt (separate from max_drawdown hard halt)
        drawdown_pct = ((self.state.peak_equity - self.state.current_equity)
                        / self.state.peak_equity * 100.0) if self.state.peak_equity > 0 else 0.0
        if drawdown_pct >= self.rc.drawdown_halt:
            self._halt(f"Drawdown halt: {drawdown_pct:.2f}% >= {self.rc.drawdown_halt}%")
            return False, self.state.halt_reason

        return True, "OK"

    def validate_spread(self, spread_pct: float) -> bool:
        """Check if the current spread is acceptable for scalping."""
        if spread_pct > self.tc.max_spread_pct:
            logger.debug(f"Spread too wide: {spread_pct:.4f}% > {self.tc.max_spread_pct}%")
            return False
        return True

    def calculate_position_size(self, price: float, confidence: float = 0.55,
                                symbol: str = "", regime_mult: float = 1.0,
                                session_mult: float = 1.0) -> SizingResult:
        """
        Calculate position size using the PositionSizer.

        Returns a SizingResult with position_usd, leverage, notional_usd,
        and full breakdown of all multipliers used.
        """
        result = self.sizer.calculate(
            equity=self.state.current_equity,
            peak_equity=self.state.peak_equity,
            confidence=confidence,
            symbol=symbol,
            regime_mult=regime_mult,
            session_mult=session_mult,
            current_time=time.time(),
        )

        # Cap at max_position_usd (hard config limit)
        if result.position_usd > self.tc.max_position_usd:
            result.position_usd = self.tc.max_position_usd
            result.notional_usd = result.position_usd * result.leverage

        return result

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
        now = time.time()
        if trade.pnl_usd < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = now
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

        # Feed outcome to position sizer for Kelly + consecutive loss tracking
        self.sizer.record_outcome(trade.symbol, trade.pnl_usd, now)

    # ─────────────────────────────────────────
    # STATE MANAGEMENT
    # ─────────────────────────────────────────

    def _check_daily_reset(self):
        """Auto-reset daily counters when a new UTC day begins."""
        now_utc = datetime.datetime.utcnow().date()
        session_utc = datetime.datetime.utcfromtimestamp(self.state.session_start).date()
        if now_utc > session_utc:
            logger.info(
                f"New trading day ({now_utc}). Resetting daily counters. "
                f"Previous day: {self.state.daily_trade_count} trades, "
                f"PnL: ${self.state.daily_pnl:+.2f}"
            )
            self.reset_daily(new_equity=self.state.current_equity)
            self.state.session_start = time.time()

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
