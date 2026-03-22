"""
Session Filter — UTC trading window gating and position size adjustment.

Restricts trading to favorable market sessions and adjusts position sizes
based on expected liquidity and volatility.

UTC Windows:
    13:00-17:00  US+EU overlap → 100% size (highest liquidity)
    07:00-13:00  EU session    → 80% size
    17:00-21:00  US session    → 80% size
    00:00-07:00  Asian session → 50% size, BTC/ETH only
    21:00-00:00  Dead zone     → block all trades

Pair restrictions:
    BTC/USDT: allowed all sessions
    ETH/USDT: allowed all sessions except dead zone
    Altcoins:  EU + US sessions only (07:00-21:00)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Trading session classification."""
    US_EU_OVERLAP = "us_eu_overlap"   # 13:00-17:00 UTC
    EU = "eu"                         # 07:00-13:00 UTC
    US = "us"                         # 17:00-21:00 UTC
    ASIAN = "asian"                   # 00:00-07:00 UTC
    DEAD_ZONE = "dead_zone"           # 21:00-00:00 UTC


@dataclass
class SessionResult:
    """Session filter output."""
    session: SessionType = SessionType.DEAD_ZONE
    size_multiplier: float = 0.0    # Position size scaling
    is_blocked: bool = True         # Should we block this trade?
    block_reason: Optional[str] = None
    utc_hour: int = 0


class SessionFilter:
    """
    Determines position size multiplier and trade eligibility
    based on current UTC time and trading pair.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.session

    def check(self, symbol: str, utc_now: Optional[datetime] = None) -> SessionResult:
        """
        Check if trading is allowed for this symbol at this time.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            utc_now: Current UTC datetime (defaults to now)

        Returns:
            SessionResult with size multiplier and block status
        """
        if not self.sc.enabled:
            return SessionResult(
                session=SessionType.US_EU_OVERLAP,
                size_multiplier=1.0,
                is_blocked=False,
                utc_hour=0,
            )

        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        hour = utc_now.hour
        session = self._get_session(hour)
        result = SessionResult(session=session, utc_hour=hour)

        # Apply session rules
        if session == SessionType.DEAD_ZONE:
            result.is_blocked = True
            result.size_multiplier = 0.0
            result.block_reason = f"Dead zone (21:00-00:00 UTC), hour={hour}"
            return result

        if session == SessionType.ASIAN:
            # Asian session: only BTC/ETH, 50% size
            if symbol in self.sc.asian_allowed_pairs:
                result.is_blocked = False
                result.size_multiplier = 0.5
            else:
                result.is_blocked = True
                result.size_multiplier = 0.0
                result.block_reason = f"Asian session blocks {symbol} (only BTC/ETH allowed)"
            return result

        if session == SessionType.EU:
            result.is_blocked = False
            result.size_multiplier = 0.8
            return result

        if session == SessionType.US:
            result.is_blocked = False
            result.size_multiplier = 0.8
            return result

        if session == SessionType.US_EU_OVERLAP:
            result.is_blocked = False
            result.size_multiplier = 1.0
            return result

        return result

    @staticmethod
    def _get_session(utc_hour: int) -> SessionType:
        """Map UTC hour to session type."""
        if 0 <= utc_hour < 7:
            return SessionType.ASIAN
        if 7 <= utc_hour < 13:
            return SessionType.EU
        if 13 <= utc_hour < 17:
            return SessionType.US_EU_OVERLAP
        if 17 <= utc_hour < 21:
            return SessionType.US
        # 21 <= utc_hour < 24
        return SessionType.DEAD_ZONE
