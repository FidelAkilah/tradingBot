"""
Database layer — SQLite persistence for trades, signals, and bot state.
Designed to work both locally and on Render (file-based SQLite).
"""

import json
import os
import sqlite3
import time
import threading
from dataclasses import asdict
from typing import Any, Dict, List, Optional


DB_PATH = os.environ.get("DB_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bot_data.db"
))

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Thread-safe connection getter."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    """Create tables if they don't exist, then run migrations for new columns."""
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        target_price REAL,
        stop_price REAL,
        amount REAL,
        usd_value REAL,
        pnl_usd REAL,
        pnl_pct REAL,
        composite_score REAL,
        swing_trend TEXT,
        swing_confidence REAL,
        atr_tp_pct REAL,
        atr_sl_pct REAL,
        vpin REAL,
        vpin_regime TEXT,
        entry_time REAL,
        exit_time REAL,
        exit_reason TEXT,
        duration_s REAL,
        is_open INTEGER DEFAULT 1,
        leverage INTEGER DEFAULT 30,
        extra_json TEXT,
        created_at REAL DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        symbol TEXT,
        mid_price REAL,
        composite_score REAL,
        suggestion TEXT,
        swing_trend TEXT,
        swing_confidence REAL,
        vpin REAL,
        vpin_regime TEXT,
        atr_tp_pct REAL,
        atr_sl_pct REAL,
        extra_json TEXT
    );

    CREATE TABLE IF NOT EXISTS bot_state (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at REAL
    );

    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
    CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(is_open);
    CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
    CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
    """)
    conn.commit()

    # --- Migrations: add fee-aware and ADX columns (backward compatible) ---
    _migrate_add_columns(conn, "trades", {
        "gross_pnl_usd": "REAL",
        "fee_cost_usd": "REAL",
        "raw_tp_pct": "REAL",
        "raw_sl_pct": "REAL",
        "fee_cost_pct": "REAL",
        "post_fee_rr": "REAL",
        "adx": "REAL",
    })
    _migrate_add_columns(conn, "signals", {
        "adx": "REAL",
        "post_fee_rr": "REAL",
        "adx_blocked": "INTEGER",
    })


def _migrate_add_columns(conn, table: str, columns: dict):
    """Add columns to a table if they don't already exist."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col_name, col_type in columns.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    conn.commit()


# ─────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────

def insert_trade(trade_data: Dict[str, Any]) -> int:
    conn = get_conn()
    known_cols = {
        'trade_id', 'symbol', 'side', 'entry_price', 'exit_price',
        'target_price', 'stop_price', 'amount', 'usd_value',
        'pnl_usd', 'pnl_pct', 'composite_score', 'swing_trend',
        'swing_confidence', 'atr_tp_pct', 'atr_sl_pct', 'vpin',
        'vpin_regime', 'entry_time', 'exit_time', 'exit_reason',
        'duration_s', 'is_open', 'leverage',
        # Fee-aware and ADX columns
        'gross_pnl_usd', 'fee_cost_usd', 'raw_tp_pct', 'raw_sl_pct',
        'fee_cost_pct', 'post_fee_rr', 'adx',
    }
    extra = {k: v for k, v in trade_data.items() if k not in known_cols}

    cur = conn.execute("""
    INSERT INTO trades (
        trade_id, symbol, side, entry_price, exit_price, target_price,
        stop_price, amount, usd_value, pnl_usd, pnl_pct,
        composite_score, swing_trend, swing_confidence,
        atr_tp_pct, atr_sl_pct, vpin, vpin_regime,
        entry_time, exit_time, exit_reason, duration_s, is_open,
        leverage, gross_pnl_usd, fee_cost_usd, raw_tp_pct, raw_sl_pct,
        fee_cost_pct, post_fee_rr, adx, extra_json
    ) VALUES (
        :trade_id, :symbol, :side, :entry_price, :exit_price, :target_price,
        :stop_price, :amount, :usd_value, :pnl_usd, :pnl_pct,
        :composite_score, :swing_trend, :swing_confidence,
        :atr_tp_pct, :atr_sl_pct, :vpin, :vpin_regime,
        :entry_time, :exit_time, :exit_reason, :duration_s, :is_open,
        :leverage, :gross_pnl_usd, :fee_cost_usd, :raw_tp_pct, :raw_sl_pct,
        :fee_cost_pct, :post_fee_rr, :adx, :extra_json
    )
    """, {
        'trade_id': trade_data.get('trade_id'),
        'symbol': trade_data.get('symbol'),
        'side': trade_data.get('side'),
        'entry_price': trade_data.get('entry_price'),
        'exit_price': trade_data.get('exit_price'),
        'target_price': trade_data.get('target_price'),
        'stop_price': trade_data.get('stop_price'),
        'amount': trade_data.get('amount'),
        'usd_value': trade_data.get('usd_value'),
        'pnl_usd': trade_data.get('pnl_usd'),
        'pnl_pct': trade_data.get('pnl_pct'),
        'composite_score': trade_data.get('composite_score'),
        'swing_trend': trade_data.get('swing_trend'),
        'swing_confidence': trade_data.get('swing_confidence'),
        'atr_tp_pct': trade_data.get('atr_tp_pct'),
        'atr_sl_pct': trade_data.get('atr_sl_pct'),
        'vpin': trade_data.get('vpin'),
        'vpin_regime': trade_data.get('vpin_regime'),
        'entry_time': trade_data.get('entry_time'),
        'exit_time': trade_data.get('exit_time'),
        'exit_reason': trade_data.get('exit_reason'),
        'duration_s': trade_data.get('duration_s'),
        'is_open': 1 if trade_data.get('is_open', True) else 0,
        'leverage': trade_data.get('leverage', 30),
        'gross_pnl_usd': trade_data.get('gross_pnl_usd'),
        'fee_cost_usd': trade_data.get('fee_cost_usd'),
        'raw_tp_pct': trade_data.get('raw_tp_pct'),
        'raw_sl_pct': trade_data.get('raw_sl_pct'),
        'fee_cost_pct': trade_data.get('fee_cost_pct'),
        'post_fee_rr': trade_data.get('post_fee_rr'),
        'adx': trade_data.get('adx'),
        'extra_json': json.dumps(extra, default=str) if extra else None,
    })
    conn.commit()
    return cur.lastrowid


def update_trade_close(trade_id: int, exit_price: float, exit_reason: str,
                       pnl_usd: float, pnl_pct: float, duration_s: float,
                       exit_time: float, gross_pnl_usd: float = None,
                       fee_cost_usd: float = None):
    conn = get_conn()
    conn.execute("""
    UPDATE trades SET
        exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?,
        duration_s = ?, exit_time = ?, is_open = 0,
        gross_pnl_usd = ?, fee_cost_usd = ?
    WHERE trade_id = ? AND is_open = 1
    """, (exit_price, exit_reason, pnl_usd, pnl_pct, duration_s, exit_time,
          gross_pnl_usd, fee_cost_usd, trade_id))
    conn.commit()


def get_trades(limit: int = 100, offset: int = 0, symbol: str = None,
               open_only: bool = False) -> List[dict]:
    conn = get_conn()
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if open_only:
        query += " AND is_open = 1"
    query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_open_trades() -> List[dict]:
    return get_trades(limit=100, open_only=True)


def get_max_trade_id() -> int:
    """Return the highest trade_id in the database (0 if empty)."""
    conn = get_conn()
    row = conn.execute("SELECT COALESCE(MAX(trade_id), 0) as m FROM trades").fetchone()
    return row["m"]


def close_stale_trades():
    """Mark orphaned open trades from previous sessions as closed.

    When the bot restarts, in-memory state is lost but the DB still has
    is_open = 1 rows.  These can never be properly closed, and their
    trade_ids will collide with new trades, corrupting exit data.
    """
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) as c FROM trades WHERE is_open = 1").fetchone()["c"]
    if count > 0:
        conn.execute("""
        UPDATE trades SET
            is_open = 0,
            exit_reason = COALESCE(exit_reason, 'bot_restart'),
            exit_time = COALESCE(exit_time, ?),
            pnl_usd = COALESCE(pnl_usd, 0),
            pnl_pct = COALESCE(pnl_pct, 0),
            duration_s = COALESCE(duration_s, 0)
        WHERE is_open = 1
        """, (time.time(),))
        conn.commit()
    return count


# ─────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────

def insert_signal(signal_data: Dict[str, Any]):
    conn = get_conn()
    known_cols = {
        'timestamp', 'symbol', 'mid_price', 'composite_score',
        'suggestion', 'swing_trend', 'swing_confidence',
        'vpin', 'vpin_regime', 'atr_tp_pct', 'atr_sl_pct',
        'adx', 'post_fee_rr', 'adx_blocked',
    }
    extra = {k: v for k, v in signal_data.items() if k not in known_cols}
    conn.execute("""
    INSERT INTO signals (
        timestamp, symbol, mid_price, composite_score, suggestion,
        swing_trend, swing_confidence, vpin, vpin_regime,
        atr_tp_pct, atr_sl_pct, adx, post_fee_rr, adx_blocked,
        extra_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_data.get('timestamp'),
        signal_data.get('symbol'),
        signal_data.get('mid_price'),
        signal_data.get('composite_score'),
        signal_data.get('suggestion'),
        signal_data.get('swing_trend'),
        signal_data.get('swing_confidence'),
        signal_data.get('vpin'),
        signal_data.get('vpin_regime'),
        signal_data.get('atr_tp_pct'),
        signal_data.get('atr_sl_pct'),
        signal_data.get('adx'),
        signal_data.get('post_fee_rr'),
        1 if signal_data.get('adx_blocked') else 0,
        json.dumps(extra, default=str) if extra else None,
    ))
    conn.commit()


def get_signals(limit: int = 200, symbol: str = None) -> List[dict]:
    conn = get_conn()
    query = "SELECT * FROM signals WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────
# BOT STATE (key-value store)
# ─────────────────────────────────────────

def set_state(key: str, value: Any):
    conn = get_conn()
    conn.execute("""
    INSERT OR REPLACE INTO bot_state (key, value, updated_at)
    VALUES (?, ?, ?)
    """, (key, json.dumps(value, default=str), time.time()))
    conn.commit()


def get_state(key: str, default=None) -> Any:
    conn = get_conn()
    row = conn.execute(
        "SELECT value FROM bot_state WHERE key = ?", (key,)
    ).fetchone()
    if row:
        return json.loads(row["value"])
    return default


def get_all_state() -> Dict[str, Any]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM bot_state").fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


# ─────────────────────────────────────────
# PERFORMANCE AGGREGATES
# ─────────────────────────────────────────

def get_performance_summary() -> dict:
    conn = get_conn()

    total = conn.execute("SELECT COUNT(*) as c FROM trades WHERE is_open = 0").fetchone()["c"]
    wins = conn.execute("SELECT COUNT(*) as c FROM trades WHERE is_open = 0 AND pnl_usd > 0").fetchone()["c"]
    losses = conn.execute("SELECT COUNT(*) as c FROM trades WHERE is_open = 0 AND pnl_usd <= 0").fetchone()["c"]
    open_count = conn.execute("SELECT COUNT(*) as c FROM trades WHERE is_open = 1").fetchone()["c"]

    total_pnl = conn.execute(
        "SELECT COALESCE(SUM(pnl_usd), 0) as s FROM trades WHERE is_open = 0"
    ).fetchone()["s"]

    gross_profit = conn.execute(
        "SELECT COALESCE(SUM(pnl_usd), 0) as s FROM trades WHERE is_open = 0 AND pnl_usd > 0"
    ).fetchone()["s"]
    gross_loss = abs(conn.execute(
        "SELECT COALESCE(SUM(pnl_usd), 0) as s FROM trades WHERE is_open = 0 AND pnl_usd < 0"
    ).fetchone()["s"])

    avg_win = conn.execute(
        "SELECT COALESCE(AVG(pnl_usd), 0) as a FROM trades WHERE is_open = 0 AND pnl_usd > 0"
    ).fetchone()["a"]
    avg_loss = conn.execute(
        "SELECT COALESCE(AVG(pnl_usd), 0) as a FROM trades WHERE is_open = 0 AND pnl_usd <= 0"
    ).fetchone()["a"]
    avg_duration = conn.execute(
        "SELECT COALESCE(AVG(duration_s), 0) as a FROM trades WHERE is_open = 0"
    ).fetchone()["a"]

    # Per-symbol breakdown
    symbol_rows = conn.execute("""
        SELECT symbol,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades WHERE is_open = 0
        GROUP BY symbol
    """).fetchall()

    # Exit reason breakdown
    reason_rows = conn.execute("""
        SELECT exit_reason, COUNT(*) as count
        FROM trades WHERE is_open = 0
        GROUP BY exit_reason
    """).fetchall()

    return {
        "total_trades": total,
        "open_trades": open_count,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "total_pnl_usd": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_duration_s": avg_duration,
        "by_symbol": [dict(r) for r in symbol_rows],
        "by_exit_reason": [dict(r) for r in reason_rows],
    }


def get_pnl_timeseries() -> List[dict]:
    """Cumulative P&L over time for charting."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT exit_time as timestamp, pnl_usd, symbol,
               SUM(pnl_usd) OVER (ORDER BY exit_time) as cumulative_pnl
        FROM trades
        WHERE is_open = 0
        ORDER BY exit_time
    """).fetchall()
    return [dict(r) for r in rows]
