"""
Historical Data Manager — Download and store Binance kline data for backtesting.

Features:
- Downloads OHLCV candles from Binance for all pairs and timeframes
- Stores in SQLite: historical_candles table
- Supports date range selection
- Incremental download (only fetches new candles since last sync)
- Rate-limited to respect Binance API limits
"""

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("BACKTEST_DB_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "backtest_data.db"
))

# Binance kline limits
MAX_CANDLES_PER_REQUEST = 1000

# Timeframe → milliseconds per candle
TIMEFRAME_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass
class SyncStatus:
    """Status of data sync for a symbol/timeframe pair."""
    symbol: str
    timeframe: str
    total_candles: int
    first_timestamp: Optional[float]
    last_timestamp: Optional[float]
    last_sync: Optional[float]


class DataManager:
    """Manages historical candle data for backtesting."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_candles (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_ts
            ON historical_candles(symbol, timeframe, timestamp);

        CREATE TABLE IF NOT EXISTS sync_metadata (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            last_timestamp INTEGER,
            last_sync_time REAL,
            PRIMARY KEY (symbol, timeframe)
        );
        """)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Sync from Binance ──────────────────────────────────────────

    async def sync_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        days: int = 90,
        rate_limit_delay: float = 0.35,
    ):
        """
        Download historical candles from Binance for all symbol/timeframe combos.
        Incrementally fetches only new candles since last sync.
        """
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            import ccxt

        exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

        try:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - (days * 86_400_000)

            total_pairs = len(symbols) * len(timeframes)
            completed = 0

            for symbol in symbols:
                for tf in timeframes:
                    completed += 1
                    logger.info(
                        f"[{completed}/{total_pairs}] Syncing {symbol} {tf}..."
                    )

                    try:
                        await self._sync_pair_tf(
                            exchange, symbol, tf,
                            start_ms, end_ms, rate_limit_delay
                        )
                    except Exception as e:
                        logger.error(f"Failed to sync {symbol} {tf}: {e}")

                    await asyncio.sleep(rate_limit_delay)

            logger.info("Data sync complete.")
        finally:
            await exchange.close()

    async def _sync_pair_tf(
        self,
        exchange,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        rate_limit_delay: float,
    ):
        """Incrementally sync a single symbol/timeframe."""
        conn = self._get_conn()

        # Check last synced timestamp
        row = conn.execute(
            "SELECT last_timestamp FROM sync_metadata WHERE symbol=? AND timeframe=?",
            (symbol, timeframe)
        ).fetchone()

        if row and row["last_timestamp"]:
            # Resume from last synced + 1 candle
            tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
            fetch_from = row["last_timestamp"] + tf_ms
            if fetch_from >= end_ms:
                logger.info(f"  {symbol} {timeframe} already up to date")
                return
        else:
            fetch_from = start_ms

        total_fetched = 0
        current_ms = fetch_from

        while current_ms < end_ms:
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=current_ms,
                    limit=MAX_CANDLES_PER_REQUEST,
                )
            except Exception as e:
                logger.warning(f"  Fetch error at {current_ms}: {e}")
                await asyncio.sleep(1.0)
                continue

            if not ohlcv:
                break

            # Insert candles
            rows = [
                (symbol, timeframe, int(c[0]), c[1], c[2], c[3], c[4], c[5])
                for c in ohlcv
                if c[0] < end_ms  # Don't store incomplete current candle
            ]

            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO historical_candles
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows
                )
                conn.commit()
                total_fetched += len(rows)

            # Move to next batch
            last_ts = ohlcv[-1][0]
            tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)
            current_ms = last_ts + tf_ms

            # Update sync metadata
            conn.execute(
                """INSERT OR REPLACE INTO sync_metadata
                (symbol, timeframe, last_timestamp, last_sync_time)
                VALUES (?, ?, ?, ?)""",
                (symbol, timeframe, last_ts, time.time())
            )
            conn.commit()

            if len(ohlcv) < MAX_CANDLES_PER_REQUEST:
                break

            await asyncio.sleep(rate_limit_delay)

        logger.info(f"  {symbol} {timeframe}: synced {total_fetched} candles")

    # ── Data Retrieval ─────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[List]:
        """
        Retrieve candles as OHLCV lists: [[timestamp, O, H, L, C, V], ...].
        Compatible with ccxt format used by CandleAnalyzer.
        """
        conn = self._get_conn()
        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM historical_candles "
            "WHERE symbol=? AND timeframe=?"
        )
        params: list = [symbol, timeframe]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"

        rows = conn.execute(query, params).fetchall()
        return [
            [r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"]]
            for r in rows
        ]

    def get_candles_as_dict(
        self,
        symbol: str,
        timeframes: List[str],
        end_ts: int,
        lookback_candles: int = 50,
    ) -> Dict[str, List]:
        """
        Get multi-timeframe candles ending at end_ts, formatted as
        {timeframe: [[ts, O, H, L, C, V], ...]} for CandleAnalyzer.analyze().
        """
        result = {}
        for tf in timeframes:
            tf_ms = TIMEFRAME_MS.get(tf, 3_600_000)
            start_ts = end_ts - (lookback_candles * tf_ms)
            candles = self.get_candles(symbol, tf, start_ts, end_ts)
            if candles:
                result[tf] = candles
        return result

    def get_available_range(
        self, symbol: str, timeframe: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (first_timestamp, last_timestamp) for a symbol/timeframe."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT MIN(timestamp) as mn, MAX(timestamp) as mx "
            "FROM historical_candles WHERE symbol=? AND timeframe=?",
            (symbol, timeframe)
        ).fetchone()
        if row and row["mn"] is not None:
            return (row["mn"], row["mx"])
        return (None, None)

    def get_sync_status(self) -> List[SyncStatus]:
        """Get sync status for all symbol/timeframe pairs."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT
                h.symbol,
                h.timeframe,
                COUNT(*) as total_candles,
                MIN(h.timestamp) as first_ts,
                MAX(h.timestamp) as last_ts,
                m.last_sync_time
            FROM historical_candles h
            LEFT JOIN sync_metadata m
                ON h.symbol = m.symbol AND h.timeframe = m.timeframe
            GROUP BY h.symbol, h.timeframe
            ORDER BY h.symbol, h.timeframe
        """).fetchall()

        return [
            SyncStatus(
                symbol=r["symbol"],
                timeframe=r["timeframe"],
                total_candles=r["total_candles"],
                first_timestamp=r["first_ts"],
                last_timestamp=r["last_ts"],
                last_sync=r["last_sync_time"],
            )
            for r in rows
        ]

    def get_primary_timeline(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[int]:
        """
        Get sorted list of timestamps for iteration.
        Used by the engine to step through candles chronologically.
        """
        conn = self._get_conn()
        query = (
            "SELECT DISTINCT timestamp FROM historical_candles "
            "WHERE symbol=? AND timeframe=?"
        )
        params: list = [symbol, timeframe]

        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"
        rows = conn.execute(query, params).fetchall()
        return [r["timestamp"] for r in rows]

    def get_all_symbols(self) -> List[str]:
        """Get all symbols that have data."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM historical_candles ORDER BY symbol"
        ).fetchall()
        return [r["symbol"] for r in rows]

    def delete_data(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ):
        """Delete candle data, optionally filtered by symbol and/or timeframe."""
        conn = self._get_conn()
        if symbol and timeframe:
            conn.execute(
                "DELETE FROM historical_candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe)
            )
            conn.execute(
                "DELETE FROM sync_metadata WHERE symbol=? AND timeframe=?",
                (symbol, timeframe)
            )
        elif symbol:
            conn.execute(
                "DELETE FROM historical_candles WHERE symbol=?", (symbol,)
            )
            conn.execute(
                "DELETE FROM sync_metadata WHERE symbol=?", (symbol,)
            )
        else:
            conn.execute("DELETE FROM historical_candles")
            conn.execute("DELETE FROM sync_metadata")
        conn.commit()
