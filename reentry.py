"""
Smart Re-entry Manager — Re-enters after stop-outs when the trend is intact.

Instead of a blanket 5-minute cooldown after every stop-loss:
1. Checks if the original trend is still valid (ADX, direction, confidence)
2. Waits for a price retrace from the stop-out level
3. Re-enters with reduced size (70%) and tighter SL (0.8× ATR)
4. Maximum 1 re-entry per original signal
5. Daily-target-aware: blocks re-entries if loss limit >50% consumed
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ReentryCandidate:
    """Tracks a potential re-entry after a stop-out."""
    symbol: str
    original_side: str              # "BUY" or "SELL"
    original_entry_price: float
    stop_out_price: float
    stop_out_time: float
    original_confidence: float
    original_atr: float
    original_amount: float
    reentry_count: int = 0
    expires_at: float = 0.0


class ReentryManager:
    """
    Manages smart re-entries after stop-outs.

    After a stop-loss exit:
    - Registers a re-entry candidate with the trade details
    - On each tick, checks if conditions are met for re-entry:
      1. Cooldown elapsed (3 min)
      2. Trend still valid (same direction, sufficient ADX + confidence)
      3. Not expired (1 hour window)
      4. Max re-entries not reached
      5. Daily loss limit not consumed >50%
    - Returns the side to re-enter with, or None
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.rc = config.reentry
        self._candidates: Dict[str, ReentryCandidate] = {}

    def register_stopout(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_out_price: float,
        stop_out_time: float,
        confidence: float,
        atr: float,
        amount: float,
    ):
        """Register a potential re-entry after a stop-out."""
        if not self.rc.enabled:
            return

        self._candidates[symbol] = ReentryCandidate(
            symbol=symbol,
            original_side=side,
            original_entry_price=entry_price,
            stop_out_price=stop_out_price,
            stop_out_time=stop_out_time,
            original_confidence=confidence,
            original_atr=atr,
            original_amount=amount,
            expires_at=stop_out_time + self.rc.expiry_s,
        )

        logger.info(
            f"[REENTRY] Registered candidate: {side} {symbol} "
            f"stopped @ {stop_out_price:.2f} | "
            f"window={self.rc.expiry_s:.0f}s"
        )

    def check_reentry(
        self,
        symbol: str,
        current_price: float,
        current_time: float,
        swing_side: Optional[str],
        swing_confidence: float,
        swing_adx: float,
        daily_loss_consumed_pct: float,
    ) -> Optional[str]:
        """
        Check if a re-entry should be triggered.

        Args:
            symbol: The trading pair
            current_price: Current market price
            current_time: Current timestamp
            swing_side: Current swing signal direction ("BUY", "SELL", or None)
            swing_confidence: Current signal confidence
            swing_adx: Current ADX value
            daily_loss_consumed_pct: How much of daily loss limit is used (0-100)

        Returns:
            The side to re-enter ("BUY" or "SELL"), or None.
        """
        if symbol not in self._candidates:
            return None

        cand = self._candidates[symbol]

        # Expiry check
        if current_time > cand.expires_at:
            del self._candidates[symbol]
            return None

        # Cooldown
        if current_time - cand.stop_out_time < self.rc.cooldown_s:
            return None

        # Max re-entries
        if cand.reentry_count >= self.rc.max_reentries_per_signal:
            del self._candidates[symbol]
            return None

        # Daily loss limit block
        if daily_loss_consumed_pct >= self.rc.daily_loss_block_pct:
            logger.debug(
                f"[REENTRY] {symbol} blocked: daily loss "
                f"{daily_loss_consumed_pct:.1f}% >= {self.rc.daily_loss_block_pct}%"
            )
            return None

        # Signal still valid: same direction
        if not swing_side or swing_side != cand.original_side:
            return None

        # Signal quality checks
        if swing_confidence < self.rc.min_confidence:
            return None
        if swing_adx < self.rc.min_adx:
            return None

        # All conditions met — trigger re-entry
        cand.reentry_count += 1

        logger.info(
            f"[REENTRY] Triggered: {cand.original_side} {symbol} "
            f"@ {current_price:.2f} (was stopped @ {cand.stop_out_price:.2f}) "
            f"| conf={swing_confidence:.2f} ADX={swing_adx:.1f} "
            f"| re-entry #{cand.reentry_count}"
        )

        return cand.original_side

    def get_candidate(self, symbol: str) -> Optional[ReentryCandidate]:
        """Get the active re-entry candidate for a symbol."""
        return self._candidates.get(symbol)

    def clear_candidate(self, symbol: str):
        """Remove a re-entry candidate (after successful re-entry)."""
        self._candidates.pop(symbol, None)

    def get_size_mult(self) -> float:
        """Position size multiplier for re-entries (0.70 = 70%)."""
        return self.rc.size_mult

    def get_sl_atr_mult(self) -> float:
        """SL ATR multiplier for re-entries (tighter than normal)."""
        return self.rc.sl_atr_mult

    def cleanup_expired(self, current_time: float):
        """Remove expired candidates."""
        expired = [
            s for s, c in self._candidates.items()
            if current_time > c.expires_at
        ]
        for s in expired:
            del self._candidates[s]
