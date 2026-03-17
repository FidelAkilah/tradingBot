"""
WebSocket client for real-time order book streaming via ccxt.pro.
Handles connection lifecycle, reconnection, and top-N pair discovery.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import ccxt.pro as ccxtpro

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


class OrderBookStream:
    """
    Manages real-time order book streams for multiple pairs.

    Responsibilities:
    - Discover top-N pairs by 24h volume dynamically
    - Maintain WebSocket connections for each pair
    - Deliver order book snapshots to registered callbacks
    - Handle reconnection and backpressure
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.exchange: Optional[ccxtpro.Exchange] = None
        self.active_pairs: List[str] = []
        self.callbacks: List[Callable] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._last_update: Dict[str, float] = {}

    async def initialize(self):
        """Create exchange instance and load markets."""
        exchange_class = getattr(ccxtpro, self.config.exchange.exchange_id)
        self.exchange = exchange_class({
            "apiKey": self.config.exchange.api_key,
            "secret": self.config.exchange.api_secret,
            "sandbox": self.config.exchange.sandbox,
            "enableRateLimit": self.config.exchange.rate_limit,
            "options": self.config.exchange.options,
        })
        await self.exchange.load_markets()
        logger.info(f"Exchange initialized: {self.exchange.id} | Markets loaded: {len(self.exchange.markets)}")

    async def discover_top_pairs(self) -> List[str]:
        """
        Fetch top N trading pairs by 24h quote volume (USDT pairs only).
        Falls back to manual_pairs if dynamic discovery fails.
        """
        if not self.config.trading.use_dynamic_pairs:
            self.active_pairs = self.config.trading.manual_pairs
            logger.info(f"Using manual pairs: {self.active_pairs}")
            return self.active_pairs

        try:
            tickers = await self.exchange.fetch_tickers()

            # Filter USDT pairs, rank by quote volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith("/USDT") and ticker.get("quoteVolume"):
                    usdt_pairs.append((symbol, ticker["quoteVolume"]))

            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            self.active_pairs = [p[0] for p in usdt_pairs[:self.config.trading.top_n_pairs]]

            logger.info(f"Top {self.config.trading.top_n_pairs} pairs by volume: {self.active_pairs}")
            for sym, vol in usdt_pairs[:self.config.trading.top_n_pairs]:
                logger.info(f"  {sym}: ${vol:,.0f} 24h volume")

        except Exception as e:
            logger.warning(f"Dynamic pair discovery failed: {e}. Falling back to manual pairs.")
            self.active_pairs = self.config.trading.manual_pairs

        return self.active_pairs

    def on_order_book(self, callback: Callable):
        """
        Register a callback for order book updates.
        Callback signature: async def handler(symbol: str, order_book: dict, timestamp: float)
        """
        self.callbacks.append(callback)
        logger.debug(f"Registered order book callback: {callback.__name__}")

    async def _stream_pair(self, symbol: str):
        """
        Stream order book for a single pair with throttling and error recovery.
        """
        throttle_ms = self.config.order_book.update_speed_ms / 1000.0
        retry_delay = 1.0
        max_retry_delay = 30.0

        while self._running:
            try:
                order_book = await self.exchange.watch_order_book(
                    symbol,
                    limit=self.config.order_book.depth_limit
                )

                now = time.time()
                last = self._last_update.get(symbol, 0)

                # Throttle: skip if we updated too recently
                if now - last < throttle_ms:
                    continue

                self._last_update[symbol] = now

                # Dispatch to all registered callbacks concurrently
                tasks = [cb(symbol, order_book, now) for cb in self.callbacks]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Reset retry delay on success
                retry_delay = 1.0

            except ccxtpro.NetworkError as e:
                logger.warning(f"[{symbol}] Network error: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except ccxtpro.ExchangeError as e:
                logger.error(f"[{symbol}] Exchange error: {e}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except asyncio.CancelledError:
                logger.info(f"[{symbol}] Stream cancelled.")
                break

            except Exception as e:
                logger.error(f"[{symbol}] Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    async def _stream_recent_trades(self, symbol: str):
        """
        Stream recent trades for VWAP calculation.
        Stores in a ring buffer accessible via get_recent_trades().
        """
        self._recent_trades: Dict[str, list] = getattr(self, "_recent_trades", {})
        max_trades = self.config.order_book.vwap_lookback_trades

        while self._running:
            try:
                trades = await self.exchange.watch_trades(symbol)
                if symbol not in self._recent_trades:
                    self._recent_trades[symbol] = []

                self._recent_trades[symbol].extend(trades)
                # Keep only the last N trades (ring buffer behavior)
                if len(self._recent_trades[symbol]) > max_trades * 2:
                    self._recent_trades[symbol] = self._recent_trades[symbol][-max_trades:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{symbol}] Trade stream error: {e}")
                await asyncio.sleep(1.0)

    def get_recent_trades(self, symbol: str, n: Optional[int] = None) -> list:
        """Get the last N trades for a symbol."""
        trades = getattr(self, "_recent_trades", {}).get(symbol, [])
        if n:
            return trades[-n:]
        return trades[-self.config.order_book.vwap_lookback_trades:]

    async def start(self):
        """Start streaming all active pairs."""
        if not self.exchange:
            await self.initialize()
        if not self.active_pairs:
            await self.discover_top_pairs()

        self._running = True
        logger.info(f"Starting order book streams for {len(self.active_pairs)} pairs...")

        for symbol in self.active_pairs:
            task = asyncio.create_task(self._stream_pair(symbol))
            task.set_name(f"ob_stream_{symbol}")
            self._tasks.append(task)

            trade_task = asyncio.create_task(self._stream_recent_trades(symbol))
            trade_task.set_name(f"trade_stream_{symbol}")
            self._tasks.append(trade_task)

        logger.info(f"All streams started. Tasks: {len(self._tasks)}")

    async def stop(self):
        """Gracefully stop all streams and close the exchange."""
        self._running = False
        logger.info("Stopping all streams...")

        # Cancel all stream tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to finish with a hard timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some stream tasks did not stop in time.")
            except Exception:
                pass
        self._tasks.clear()

        # Close exchange connection
        if self.exchange:
            try:
                await asyncio.wait_for(self.exchange.close(), timeout=3.0)
                logger.info("Exchange connection closed.")
            except asyncio.TimeoutError:
                logger.warning("Exchange close timed out.")
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.stop()
