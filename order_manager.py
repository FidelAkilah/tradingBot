"""
Order Manager — Adaptive execution engine with wall-pull protection.

Handles:
- Order placement relative to detected walls
- Real-time wall monitoring (pull detection)
- Order lifecycle management (place → monitor → cancel/fill)
- Position tracking
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import BotConfig, CONFIG
from liquidity_analyzer import AnalysisResult, LiquidityWall, WallSide

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class ManagedOrder:
    """Tracks an order through its lifecycle."""
    order_id: str
    symbol: str
    side: str                          # "buy" or "sell"
    price: float
    amount: float
    order_type: str = "limit"
    state: OrderState = OrderState.PENDING
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    wall_price: float = 0.0            # The wall this order is anchored to
    wall_side: Optional[WallSide] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    cancel_reason: str = ""
    exchange_order: Optional[dict] = None  # Raw exchange response


@dataclass
class Position:
    """Active position state."""
    symbol: str
    side: PositionSide
    entry_price: float
    amount: float
    usd_value: float
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: Optional[float] = None
    wall_anchor_price: float = 0.0     # The wall that triggered this entry
    opened_at: float = 0.0
    pnl: float = 0.0                   # Unrealized P&L


class OrderManager:
    """
    Manages the full order lifecycle with wall-pull protection.

    Flow:
    1. Receive AnalysisResult with trade suggestion
    2. Calculate order price (offset from wall)
    3. Place order via exchange
    4. Monitor wall health — cancel if wall pulled
    5. Track fill → create Position
    6. Monitor position with stop-loss / take-profit
    """

    def __init__(self, exchange, config: BotConfig = CONFIG):
        self.exchange = exchange
        self.config = config
        self.tc = config.trading

        self.open_orders: Dict[str, ManagedOrder] = {}
        self.positions: Dict[str, Position] = {}  # key: symbol
        self._order_counter = 0

    # ─────────────────────────────────────────
    # ORDER PLACEMENT
    # ─────────────────────────────────────────

    async def execute_signal(
        self,
        analysis: AnalysisResult,
        equity_usd: float,
        is_shadow: bool = True,
    ) -> Optional[ManagedOrder]:
        """
        Execute a trade based on analysis result.

        Args:
            analysis: The full analysis output
            equity_usd: Current account equity in USD
            is_shadow: If True, simulate only (don't place real orders)

        Returns:
            ManagedOrder if an order was placed/simulated, None otherwise.
        """
        if not analysis.trade_suggestion:
            return None

        symbol = analysis.symbol

        # Don't stack positions on the same symbol
        if symbol in self.positions:
            logger.debug(f"[{symbol}] Already have open position, skipping.")
            return None

        # Max open positions check
        if len(self.positions) >= self.tc.max_open_positions:
            logger.debug(f"Max open positions ({self.tc.max_open_positions}) reached.")
            return None

        # Calculate position size
        position_usd = min(
            self.tc.max_position_usd,
            equity_usd * self.tc.position_pct_of_equity,
        )

        if analysis.trade_suggestion == "BUY":
            return await self._place_buy(analysis, position_usd, is_shadow)
        elif analysis.trade_suggestion == "SELL":
            return await self._place_sell(analysis, position_usd, is_shadow)

        return None

    async def _place_buy(
        self,
        analysis: AnalysisResult,
        position_usd: float,
        is_shadow: bool,
    ) -> Optional[ManagedOrder]:
        """
        Place a buy order just above the strongest genuine bid wall.
        """
        # Find the best non-spoof bid wall
        walls = [w for w in analysis.bid_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            return None

        anchor_wall = walls[0]  # Strongest by USD value
        mid = analysis.mid_price

        # Place order slightly above the wall
        offset = mid * (self.tc.offset_from_wall_pct / 100.0)
        order_price = anchor_wall.price + offset

        # Ensure we don't cross the ask
        best_ask = mid + (analysis.spread_pct / 100.0 * mid / 2.0)
        order_price = min(order_price, best_ask * 0.9999)

        amount = position_usd / order_price

        # Use ATR-based dynamic TP/SL if available, fall back to config
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else self.tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else self.tc.stop_loss_pct

        stop_loss = order_price * (1.0 - sl_pct / 100.0)
        take_profit = order_price * (1.0 + tp_pct / 100.0)

        order = ManagedOrder(
            order_id=self._next_id(),
            symbol=analysis.symbol,
            side="buy",
            price=round(order_price, 8),
            amount=round(amount, 8),
            order_type=self.tc.order_type,
            wall_price=anchor_wall.price,
            wall_side=WallSide.BID,
            created_at=time.time(),
        )

        if is_shadow:
            order.state = OrderState.OPEN
            logger.info(
                f"[SHADOW] BUY {order.amount:.6f} {order.symbol} "
                f"@ {order.price:.2f} | wall={anchor_wall.price:.2f} "
                f"| SL={stop_loss:.2f} ({sl_pct:.2f}%) | TP={take_profit:.2f} ({tp_pct:.2f}%)"
            )
        else:
            try:
                result = await self.exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type,
                    side="buy",
                    amount=order.amount,
                    price=order.price,
                    params={"timeInForce": self.tc.time_in_force},
                )
                order.exchange_order = result
                order.order_id = result.get("id", order.order_id)
                order.state = OrderState.OPEN
                logger.info(f"[LIVE] BUY order placed: {order.order_id}")
            except Exception as e:
                order.state = OrderState.FAILED
                order.cancel_reason = str(e)
                logger.error(f"[LIVE] Buy order failed: {e}")
                return order

        self.open_orders[order.order_id] = order

        # Pre-create position for tracking (will be activated on fill)
        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            side=PositionSide.LONG,
            entry_price=order.price,
            amount=order.amount,
            usd_value=position_usd,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_price=order.price * (1.0 - self.tc.trailing_stop_pct / 100.0),
            wall_anchor_price=anchor_wall.price,
            opened_at=time.time(),
        )

        return order

    async def _place_sell(
        self,
        analysis: AnalysisResult,
        position_usd: float,
        is_shadow: bool,
    ) -> Optional[ManagedOrder]:
        """
        Place a sell order just below the strongest genuine ask wall.
        """
        walls = [w for w in analysis.ask_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            return None

        anchor_wall = walls[0]
        mid = analysis.mid_price

        offset = mid * (self.tc.offset_from_wall_pct / 100.0)
        order_price = anchor_wall.price - offset

        # Ensure we don't cross the bid
        best_bid = mid - (analysis.spread_pct / 100.0 * mid / 2.0)
        order_price = max(order_price, best_bid * 1.0001)

        amount = position_usd / order_price

        # Use ATR-based dynamic TP/SL if available, fall back to config
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else self.tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else self.tc.stop_loss_pct

        stop_loss = order_price * (1.0 + sl_pct / 100.0)
        take_profit = order_price * (1.0 - tp_pct / 100.0)

        order = ManagedOrder(
            order_id=self._next_id(),
            symbol=analysis.symbol,
            side="sell",
            price=round(order_price, 8),
            amount=round(amount, 8),
            order_type=self.tc.order_type,
            wall_price=anchor_wall.price,
            wall_side=WallSide.ASK,
            created_at=time.time(),
        )

        if is_shadow:
            order.state = OrderState.OPEN
            logger.info(
                f"[SHADOW] SELL {order.amount:.6f} {order.symbol} "
                f"@ {order.price:.2f} | wall={anchor_wall.price:.2f} "
                f"| SL={stop_loss:.2f} ({sl_pct:.2f}%) | TP={take_profit:.2f} ({tp_pct:.2f}%)"
            )
        else:
            try:
                result = await self.exchange.create_order(
                    symbol=order.symbol,
                    type=order.order_type,
                    side="sell",
                    amount=order.amount,
                    price=order.price,
                    params={"timeInForce": self.tc.time_in_force},
                )
                order.exchange_order = result
                order.order_id = result.get("id", order.order_id)
                order.state = OrderState.OPEN
                logger.info(f"[LIVE] SELL order placed: {order.order_id}")
            except Exception as e:
                order.state = OrderState.FAILED
                order.cancel_reason = str(e)
                logger.error(f"[LIVE] Sell order failed: {e}")
                return order

        self.open_orders[order.order_id] = order

        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            side=PositionSide.SHORT,
            entry_price=order.price,
            amount=order.amount,
            usd_value=position_usd,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_price=order.price * (1.0 + self.tc.trailing_stop_pct / 100.0),
            wall_anchor_price=anchor_wall.price,
            opened_at=time.time(),
        )

        return order

    # ─────────────────────────────────────────
    # WALL-PULL DETECTION & PROTECTION
    # ─────────────────────────────────────────

    async def check_wall_health(
        self,
        analysis: AnalysisResult,
        is_shadow: bool = True,
    ) -> List[ManagedOrder]:
        """
        Check if the walls anchoring our open orders still exist.
        Cancel orders whose walls have been pulled.

        Returns list of cancelled orders.
        """
        cancelled = []

        for order_id, order in list(self.open_orders.items()):
            if order.state != OrderState.OPEN:
                continue
            if order.symbol != analysis.symbol:
                continue

            # Check if anchor wall still exists
            walls = analysis.bid_walls if order.wall_side == WallSide.BID else analysis.ask_walls
            wall_still_exists = self._wall_exists(order.wall_price, walls)

            if not wall_still_exists and self.tc.exit_on_wall_pull:
                logger.warning(
                    f"[WALL PULLED] {order.side.upper()} {order.symbol} "
                    f"@ {order.price:.2f} — anchor wall at {order.wall_price:.2f} gone!"
                )
                await self._cancel_order(order, "wall_pulled", is_shadow)
                cancelled.append(order)

                # Also close the associated position
                if order.symbol in self.positions:
                    await self._close_position(order.symbol, "wall_pulled", analysis.mid_price, is_shadow)

        return cancelled

    def _wall_exists(
        self,
        target_price: float,
        walls: List[LiquidityWall],
        tolerance_pct: float = 0.1,
    ) -> bool:
        """Check if a wall at approximately target_price still exists."""
        for wall in walls:
            price_diff_pct = abs(wall.price - target_price) / target_price * 100.0
            if price_diff_pct <= tolerance_pct:
                # Also check volume hasn't collapsed
                return not wall.is_spoof_suspect
        return False

    # ─────────────────────────────────────────
    # POSITION MONITORING
    # ─────────────────────────────────────────

    async def update_positions(
        self,
        symbol: str,
        current_price: float,
        is_shadow: bool = True,
    ) -> Optional[str]:
        """
        Update position P&L and check stop-loss / take-profit triggers.

        Returns action taken: "stop_loss", "take_profit", "trailing_stop", or None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        now = time.time()

        # Calculate unrealized P&L
        if pos.side == PositionSide.LONG:
            pos.pnl = (current_price - pos.entry_price) * pos.amount
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100.0

            # Update trailing stop
            if pos.trailing_stop_price and current_price > pos.entry_price:
                new_trailing = current_price * (1.0 - self.tc.trailing_stop_pct / 100.0)
                pos.trailing_stop_price = max(pos.trailing_stop_price, new_trailing)

            # Check triggers
            if current_price <= pos.stop_loss_price:
                await self._close_position(symbol, "stop_loss", current_price, is_shadow)
                return "stop_loss"
            elif current_price >= pos.take_profit_price:
                await self._close_position(symbol, "take_profit", current_price, is_shadow)
                return "take_profit"
            elif pos.trailing_stop_price and current_price <= pos.trailing_stop_price:
                await self._close_position(symbol, "trailing_stop", current_price, is_shadow)
                return "trailing_stop"

        elif pos.side == PositionSide.SHORT:
            pos.pnl = (pos.entry_price - current_price) * pos.amount
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100.0

            if pos.trailing_stop_price and current_price < pos.entry_price:
                new_trailing = current_price * (1.0 + self.tc.trailing_stop_pct / 100.0)
                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trailing)

            if current_price >= pos.stop_loss_price:
                await self._close_position(symbol, "stop_loss", current_price, is_shadow)
                return "stop_loss"
            elif current_price <= pos.take_profit_price:
                await self._close_position(symbol, "take_profit", current_price, is_shadow)
                return "take_profit"
            elif pos.trailing_stop_price and current_price >= pos.trailing_stop_price:
                await self._close_position(symbol, "trailing_stop", current_price, is_shadow)
                return "trailing_stop"

        return None

    # ─────────────────────────────────────────
    # ORDER / POSITION CLOSING
    # ─────────────────────────────────────────

    async def _cancel_order(
        self,
        order: ManagedOrder,
        reason: str,
        is_shadow: bool,
    ):
        """Cancel an open order."""
        if is_shadow:
            order.state = OrderState.CANCELLED
            order.cancel_reason = reason
            logger.info(f"[SHADOW] Cancelled {order.side} {order.symbol} — reason: {reason}")
        else:
            try:
                await self.exchange.cancel_order(order.order_id, order.symbol)
                order.state = OrderState.CANCELLED
                order.cancel_reason = reason
                logger.info(f"[LIVE] Cancelled order {order.order_id} — reason: {reason}")
            except Exception as e:
                logger.error(f"[LIVE] Failed to cancel {order.order_id}: {e}")

        order.updated_at = time.time()

    async def _close_position(
        self,
        symbol: str,
        reason: str,
        exit_price: float,
        is_shadow: bool,
    ):
        """Close a position by placing a market order."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        close_side = "sell" if pos.side == PositionSide.LONG else "buy"

        if is_shadow:
            logger.info(
                f"[SHADOW] CLOSE {close_side.upper()} {pos.amount:.6f} {symbol} "
                f"@ {exit_price:.2f} | reason={reason} | PnL={pos.pnl:+.2f} USD"
            )
        else:
            try:
                await self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=pos.amount,
                )
                logger.info(
                    f"[LIVE] CLOSE {close_side.upper()} {symbol} | "
                    f"reason={reason} | PnL≈{pos.pnl:+.2f} USD"
                )
            except Exception as e:
                logger.error(f"[LIVE] Failed to close {symbol}: {e}")
                return

        # Remove position
        del self.positions[symbol]

        # Clean up associated open orders
        for oid, order in list(self.open_orders.items()):
            if order.symbol == symbol and order.state == OrderState.OPEN:
                await self._cancel_order(order, f"position_closed_{reason}", is_shadow)

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"shadow_{self._order_counter}"

    def get_open_position_summary(self) -> str:
        """Return a human-readable summary of open positions."""
        if not self.positions:
            return "No open positions."

        lines = ["═══ Open Positions ═══"]
        for sym, pos in self.positions.items():
            lines.append(
                f"  {sym} | {pos.side.value} | entry={pos.entry_price:.2f} | "
                f"size={pos.amount:.6f} | PnL={pos.pnl:+.2f} USD | "
                f"SL={pos.stop_loss_price:.2f} | TP={pos.take_profit_price:.2f}"
            )
        return "\n".join(lines)
