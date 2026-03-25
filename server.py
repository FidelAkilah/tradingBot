"""
FastAPI server — REST API wrapping the trading bot.
Runs locally, provides endpoints for the localhost dashboard.

Usage:
    python server.py                     # Starts API on http://localhost:8000
    python server.py --port 9000         # Custom port
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import database as db
from config import CONFIG, BotConfig, IDR_PER_USD
from main import load_env_file

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# BOT WRAPPER (runs in background)
# ─────────────────────────────────────────

class BotRunner:
    """Manages the trading bot lifecycle in the background."""

    def __init__(self):
        self.bot = None
        self.task: Optional[asyncio.Task] = None
        self.status = "stopped"
        self.started_at: Optional[float] = None
        self.error: Optional[str] = None

    async def start(self, config: BotConfig):
        """Launch the bot as a background task."""
        if self.status == "running":
            return

        # Import here to avoid circular imports
        from main import ScalpingBot, setup_logging, load_api_keys

        config = load_api_keys(config)
        setup_logging(config.log_level)

        self.bot = ScalpingBot(config)

        # Mark as server mode so bot doesn't call loop.stop() on force quit
        self.bot._server_mode = True

        # Hook into shadow trader to persist trades to DB
        if self.bot.shadow:
            self._hook_shadow_trader(self.bot.shadow)

        self.status = "running"
        self.started_at = time.time()
        self.error = None
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            await self.bot.start()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            self.error = str(e)
        finally:
            self.status = "stopped"

    async def stop(self):
        if self.bot and self.status in ("running", "stopping"):
            self.status = "stopping"
            self.bot._running = False

            # Cancel the background task
            if self.task and not self.task.done():
                self.task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self.task), timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

            # Explicitly close exchange connection
            try:
                if self.bot.stream and self.bot.stream.exchange:
                    await asyncio.wait_for(self.bot.stream.exchange.close(), timeout=3)
            except (asyncio.TimeoutError, Exception):
                pass

            self.bot = None
            self.task = None
            self.status = "stopped"

    def _hook_shadow_trader(self, shadow):
        """Monkey-patch shadow trader to also write to DB."""
        # Sync trade counter from DB so IDs never collide across restarts
        max_id = db.get_max_trade_id()
        if max_id > 0:
            shadow._trade_counter = max_id
            logger.info(f"Shadow trade counter synced to {max_id}")

        original_log_trade = shadow._log_trade
        original_log_signal = shadow._log_signal

        def log_trade_with_db(trade, event):
            original_log_trade(trade, event)
            try:
                from dataclasses import asdict
                trade_dict = asdict(trade)
                trade_dict['leverage'] = trade_dict.get('leverage', CONFIG.futures.leverage)
                if event == "OPEN":
                    db.insert_trade(trade_dict)
                elif event == "CLOSE":
                    db.update_trade_close(
                        trade_id=trade.trade_id,
                        exit_price=trade.exit_price or 0,
                        exit_reason=trade.exit_reason or "unknown",
                        pnl_usd=trade.pnl_usd or 0,
                        pnl_pct=trade.pnl_pct or 0,
                        duration_s=trade.duration_s or 0,
                        exit_time=trade.exit_time or time.time(),
                        gross_pnl_usd=getattr(trade, 'gross_pnl_usd', None),
                        fee_cost_usd=getattr(trade, 'fee_cost_usd', None),
                    )
            except Exception as e:
                logger.error(f"DB trade log error: {e}")

        def log_signal_with_db(analysis):
            original_log_signal(analysis)
            try:
                swing = getattr(analysis, 'swing', None)
                db.insert_signal({
                    'timestamp': analysis.timestamp,
                    'symbol': analysis.symbol,
                    'mid_price': analysis.mid_price,
                    'composite_score': analysis.composite_score,
                    'suggestion': analysis.trade_suggestion,
                    'swing_trend': swing.primary_trend.value if swing else None,
                    'swing_confidence': swing.confidence if swing else 0,
                    'vpin': analysis.vpin.vpin,
                    'vpin_regime': analysis.vpin.regime.value,
                    'atr_tp_pct': analysis.atr_tp_pct,
                    'atr_sl_pct': analysis.atr_sl_pct,
                    'adx': getattr(analysis, 'adx', 0.0),
                    'post_fee_rr': getattr(analysis, 'post_fee_rr', 0.0),
                    'adx_blocked': swing.adx_blocked if swing else False,
                    'regime': getattr(analysis, 'regime', None),
                    'regime_blocked': getattr(analysis, 'regime_blocked', False),
                    'regime_is_breakout': getattr(analysis, 'regime_is_breakout', False),
                    'session': getattr(analysis, 'session', None),
                    'session_blocked': getattr(analysis, 'session_blocked', False),
                    'session_size_mult': getattr(analysis, 'session_size_mult', 1.0),
                })
            except Exception as e:
                logger.error(f"DB signal log error: {e}")

        shadow._log_trade = log_trade_with_db
        shadow._log_signal = log_signal_with_db

    def get_status(self) -> dict:
        """Current bot status summary."""
        uptime = time.time() - self.started_at if self.started_at and self.status == "running" else 0

        result = {
            "status": self.status,
            "uptime_s": uptime,
            "uptime_human": _format_duration(uptime),
            "started_at": self.started_at,
            "error": self.error,
            "leverage": CONFIG.futures.leverage,
            "margin_type": CONFIG.futures.margin_type,
            "pairs": CONFIG.trading.manual_pairs,
            "capital_idr": CONFIG.trading.starting_capital_idr,
            "capital_usd": CONFIG.trading.starting_capital_idr / IDR_PER_USD,
            "daily_target_pct": CONFIG.trading.daily_target_pct,
            "is_shadow": CONFIG.shadow.enabled,
        }

        if self.bot and self.bot.risk:
            rs = self.bot.risk.state
            wins = sum(1 for t in rs.trades_today if t.pnl_usd > 0)
            losses = sum(1 for t in rs.trades_today if t.pnl_usd <= 0)
            total_trades = wins + losses
            result.update({
                "equity_usd": rs.current_equity,
                "peak_equity_usd": rs.peak_equity,
                "drawdown_pct": (1 - rs.current_equity / rs.peak_equity) * 100 if rs.peak_equity > 0 else 0,
                "daily_pnl_usd": rs.daily_pnl,
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total_trades * 100 if total_trades > 0 else 0,
            })

        if self.bot and self.bot.shadow:
            result["shadow_total_pnl"] = self.bot.shadow.total_pnl
            result["shadow_open_trades"] = len(self.bot.shadow.open_trades)
            result["shadow_equity"] = self.bot.shadow._equity
            result["shadow_peak_equity"] = self.bot.shadow._peak_equity

            # Position sizer state (Kelly stats, consecutive losses)
            result["position_sizer"] = self.bot.shadow.sizer.get_state()

        # Daily target progress
        if self.bot and hasattr(self.bot, 'daily_target'):
            result["daily_target"] = self.bot.daily_target.get_daily_progress()

        return result


bot_runner = BotRunner()


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Load .env file (local mode)
    load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    # Initialize database
    db.init_db()
    stale = db.close_stale_trades()
    if stale:
        logger.info(f"Closed {stale} orphaned trade(s) from previous session")
    logger.info("Database initialized")

    # Auto-start bot if API keys are configured
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if api_key and api_secret:
        CONFIG.exchange.api_key = api_key
        CONFIG.exchange.api_secret = api_secret

        # Check for sandbox override
        sandbox = os.environ.get("BINANCE_TESTNET", "false").lower()
        CONFIG.exchange.sandbox = sandbox in ("true", "1", "yes")

        # Check for shadow mode override
        shadow = os.environ.get("SHADOW_MODE", "true").lower()
        CONFIG.shadow.enabled = shadow in ("true", "1", "yes")

        logger.info(f"Auto-starting bot (shadow={CONFIG.shadow.enabled}, sandbox={CONFIG.exchange.sandbox})...")
        await bot_runner.start(CONFIG)
    else:
        logger.warning("No API keys found — bot not auto-started. Create a .env file (see .env.example).")

    yield

    # Shutdown — gracefully stop bot before uvicorn exits
    try:
        await bot_runner.stop()
    except Exception as e:
        logger.warning(f"Shutdown error (safe to ignore): {e}")


app = FastAPI(
    title="Crypto Swing Trading Bot API",
    description="REST API for monitoring and controlling the trading bot",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all localhost origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Trading Bot API", "status": "online"}


@app.get("/api/status")
async def get_status():
    """Bot status, equity, P&L overview."""
    return _sanitize_floats(bot_runner.get_status())


@app.get("/api/trades")
async def get_trades(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = None,
    open_only: bool = False,
):
    """Trade history."""
    trades = db.get_trades(limit=limit, offset=offset, symbol=symbol, open_only=open_only)
    return {"trades": trades, "count": len(trades)}


@app.get("/api/positions")
async def get_positions():
    """Currently open positions."""
    open_trades = db.get_open_trades()

    # Also get live data from bot if running
    live_positions = []
    if bot_runner.bot and bot_runner.bot.shadow:
        from dataclasses import asdict
        for sym, trade in bot_runner.bot.shadow.open_trades.items():
            td = asdict(trade)
            td["leverage"] = CONFIG.futures.leverage

            # Compute live unrealized P&L from latest known price
            current_price = bot_runner.bot.shadow.last_prices.get(sym, trade.entry_price)
            if trade.side == "BUY":
                td["pnl_usd"] = (current_price - trade.entry_price) * trade.amount
            else:
                td["pnl_usd"] = (trade.entry_price - current_price) * trade.amount
            td["current_price"] = current_price

            live_positions.append(td)

    return _sanitize_floats({
        "positions": live_positions if live_positions else open_trades,
        "count": len(live_positions) if live_positions else len(open_trades),
    })


@app.get("/api/signals")
async def get_signals(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None,
):
    """Recent analysis signals."""
    signals = db.get_signals(limit=limit, symbol=symbol)
    return {"signals": signals, "count": len(signals)}


def _sanitize_floats(obj):
    """Replace inf/nan with JSON-safe values."""
    import math
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


@app.get("/api/performance")
async def get_performance():
    """Aggregated performance metrics."""
    summary = db.get_performance_summary()
    summary["leverage"] = CONFIG.futures.leverage

    # Add IDR values
    idr_rate = IDR_PER_USD
    summary["total_pnl_idr"] = summary["total_pnl_usd"] * idr_rate
    summary["capital_idr"] = CONFIG.trading.starting_capital_idr
    summary["capital_usd"] = CONFIG.trading.starting_capital_idr / idr_rate

    return _sanitize_floats(summary)


@app.get("/api/pnl-chart")
async def get_pnl_chart():
    """Cumulative P&L timeseries for charting."""
    data = db.get_pnl_timeseries()
    return {"data": data}


@app.get("/api/prices")
async def get_current_prices():
    """Current prices for all tracked pairs."""
    prices = {}
    if bot_runner.bot and bot_runner.bot.stream:
        for symbol in CONFIG.trading.manual_pairs:
            swing = bot_runner.bot._swing_cache.get(symbol)
            if swing:
                prices[symbol] = {
                    "trend": swing.primary_trend.value,
                    "confidence": swing.confidence,
                    "rsi_1h": swing.rsi_1h,
                    "rsi_4h": swing.rsi_4h,
                    "atr_tp": swing.atr_tp_distance,
                    "atr_sl": swing.atr_sl_distance,
                }
    return {"prices": prices, "pairs": CONFIG.trading.manual_pairs}


@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot."""
    if bot_runner.status == "running":
        raise HTTPException(400, "Bot is already running")
    await bot_runner.start(CONFIG)
    return {"message": "Bot started", "status": bot_runner.status}


@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot."""
    if bot_runner.status != "running":
        raise HTTPException(400, "Bot is not running")
    await bot_runner.stop()
    return {"message": "Bot stopped", "status": bot_runner.status}


@app.get("/api/daily-target")
async def get_daily_target():
    """Daily target progress, mode, and compound tracking."""
    result = {}

    if bot_runner.bot and hasattr(bot_runner.bot, 'daily_target'):
        result["progress"] = bot_runner.bot.daily_target.get_daily_progress()
        result["projection_30d"] = bot_runner.bot.compounder.get_compound_projection(30)
    else:
        result["progress"] = None
        result["projection_30d"] = []

    return _sanitize_floats(result)


@app.get("/api/daily-equity")
async def get_daily_equity(limit: int = Query(90, ge=1, le=365)):
    """Daily equity history for compound growth chart."""
    history = db.get_daily_equity(limit=limit)
    summary = db.get_daily_equity_summary()
    return _sanitize_floats({
        "history": history,
        "summary": summary,
    })


@app.get("/api/correlation")
async def get_correlation():
    """Correlation matrix heat map + guard status."""
    result = {"matrix": {}, "pairs": CONFIG.trading.manual_pairs, "stale": True}

    if bot_runner.bot and hasattr(bot_runner.bot, 'corr_matrix'):
        cm = bot_runner.bot.corr_matrix
        if not cm.is_stale():
            result["stale"] = False
        result["matrix"] = cm.get_matrix_dict()

        # Flag highly correlated open positions
        correlated_positions = []
        if bot_runner.bot.shadow:
            open_syms = list(bot_runner.bot.shadow.open_trades.keys())
            for i, s1 in enumerate(open_syms):
                for s2 in open_syms[i + 1:]:
                    corr = cm.get_correlation(s1, s2)
                    if corr is not None:
                        t1 = bot_runner.bot.shadow.open_trades[s1]
                        t2 = bot_runner.bot.shadow.open_trades[s2]
                        correlated_positions.append({
                            "pair": [s1, s2],
                            "correlation": round(corr, 4),
                            "same_direction": t1.side == t2.side,
                            "high_corr": abs(corr) >= CONFIG.correlation.high_corr_threshold,
                        })
        result["correlated_positions"] = correlated_positions

    return _sanitize_floats(result)


@app.get("/api/exposure")
async def get_exposure():
    """Portfolio directional exposure and heat map."""
    result = {
        "net_long": 0.0, "net_short": 0.0, "net_exposure": 0.0,
        "gross_exposure": 0.0, "directional_bias": "neutral",
        "max_long": CONFIG.correlation.max_net_long_exposure,
        "max_short": CONFIG.correlation.max_net_short_exposure,
        "positions": [],
        "breach_long": False, "breach_short": False,
    }

    if bot_runner.bot and hasattr(bot_runner.bot, 'portfolio_heat'):
        from main import ScalpingBot
        open_info = bot_runner.bot._get_open_positions_info()
        equity = 0.0
        if bot_runner.bot.shadow:
            equity = bot_runner.bot.shadow._equity
        elif bot_runner.bot.risk:
            equity = bot_runner.bot.risk.state.current_equity

        if open_info and equity > 0:
            exposure = bot_runner.bot.portfolio_heat.compute_exposure(open_info, equity)
            result.update({
                "net_long": exposure.net_long_exposure,
                "net_short": exposure.net_short_exposure,
                "net_exposure": exposure.net_exposure,
                "gross_exposure": exposure.gross_exposure,
                "directional_bias": exposure.directional_bias,
                "breach_long": exposure.breach_long,
                "breach_short": exposure.breach_short,
                "positions": exposure.positions,
            })

    return _sanitize_floats(result)


@app.get("/api/scanner")
async def get_scanner():
    """Scanner results — pair scores, selections, and reasons."""
    result = {"status": "disabled", "pairs": CONFIG.trading.manual_pairs}

    if not CONFIG.scanner.enabled:
        return result

    if bot_runner.bot and hasattr(bot_runner.bot, 'pair_selector'):
        result = bot_runner.bot.pair_selector.get_scan_summary()
        result["status"] = "active"
    else:
        # Try loading from bot_state
        try:
            cached = db.get_state("last_scan")
            if cached:
                result = cached
                result["status"] = "cached"
        except Exception:
            pass

    return _sanitize_floats(result)


@app.get("/api/scanner/performance")
async def get_scanner_performance():
    """Per-pair performance stats — win rates, P&L, auto-disable flags."""
    result = {"pairs": [], "disabled_count": 0, "auto_include_count": 0}

    if bot_runner.bot and hasattr(bot_runner.bot, 'pair_performance'):
        result = bot_runner.bot.pair_performance.get_summary()

    return _sanitize_floats(result)


@app.post("/api/scanner/scan")
async def trigger_scan():
    """Trigger an on-demand pair scan."""
    if not CONFIG.scanner.enabled:
        raise HTTPException(400, "Scanner is disabled")
    if not bot_runner.bot:
        raise HTTPException(400, "Bot is not running")

    await bot_runner.bot._run_pair_scan()
    return _sanitize_floats(bot_runner.bot.pair_selector.get_scan_summary())


@app.get("/api/daily-target/history")
async def get_daily_target_history(days: int = Query(30, ge=1, le=365)):
    """Daily target achievement history for calendar heat map."""
    history = db.get_daily_equity(limit=days)
    result = []
    for row in history:
        actual = row.get("actual_pct", 0) or 0
        target = row.get("target_pct", 2.0) or 2.0
        result.append({
            "date": row.get("date", ""),
            "target_pct": target,
            "actual_pct": actual,
            "target_hit": row.get("target_hit", False),
            "exceeded": actual > target if target > 0 else False,
            "realized_pnl": row.get("realized_pnl", 0),
            "trades": row.get("trades", 0),
            "wins": row.get("wins", 0),
            "losses": row.get("losses", 0),
            "mode_at_close": row.get("mode_at_close", "normal"),
        })
    return _sanitize_floats({"history": result, "days": days})


@app.get("/api/daily-target/projection")
async def get_daily_target_projection():
    """Compound projections at current rate (7d, 30d, 90d)."""
    equity = CONFIG.trading.starting_capital_idr / IDR_PER_USD
    target_pct = CONFIG.daily_target.daily_target_pct
    streak = 0

    if bot_runner.bot and hasattr(bot_runner.bot, 'daily_target'):
        equity = bot_runner.bot.daily_target.state.current_equity
        target_pct = bot_runner.bot.daily_target.state.daily_target_pct
        streak = bot_runner.bot.daily_target.state.streak_days

    # Compute actual avg daily return from history
    history = db.get_daily_equity(limit=30)
    actual_returns = [r.get("actual_pct", 0) or 0 for r in history if r.get("actual_pct") is not None]
    avg_daily_return = sum(actual_returns) / len(actual_returns) if actual_returns else target_pct

    projections = {}
    for days in [7, 30, 90]:
        at_target = equity * ((1 + target_pct / 100) ** days)
        at_actual = equity * ((1 + avg_daily_return / 100) ** days)
        projections[f"{days}d"] = {
            "days": days,
            "at_target_rate": round(at_target, 2),
            "at_actual_rate": round(at_actual, 2),
            "target_pct_daily": target_pct,
            "actual_pct_daily": round(avg_daily_return, 4),
        }

    return _sanitize_floats({
        "current_equity": equity,
        "streak_days": streak,
        "projections": projections,
    })


@app.get("/api/regime")
async def get_regime():
    """Current market regime per pair with ADX and eligibility info."""
    pairs = CONFIG.trading.manual_pairs
    result = {"pairs": {}, "timestamp": time.time()}

    for symbol in pairs:
        pair_data = {
            "regime": "unknown",
            "adx": 0.0,
            "bb_width_pctl": 0.0,
            "session": "unknown",
            "session_blocked": False,
            "eligible": False,
            "block_reason": "",
            "confidence": 0.0,
            "trend": "neutral",
            "rsi_1h": 50.0,
            "rsi_4h": 50.0,
            "squeeze_active": False,
            "squeeze_releasing": False,
        }

        if bot_runner.bot:
            swing = bot_runner.bot._swing_cache.get(symbol)
            if swing:
                pair_data.update({
                    "regime": swing.regime,
                    "adx": swing.adx_4h if swing.adx_4h > 0 else swing.adx_1h,
                    "bb_width_pctl": swing.regime_bb_width_pctl,
                    "session": swing.session,
                    "session_blocked": swing.session_blocked,
                    "confidence": swing.confidence,
                    "trend": swing.primary_trend.value,
                    "rsi_1h": swing.rsi_1h,
                    "rsi_4h": swing.rsi_4h,
                    "squeeze_active": swing.squeeze_active,
                    "squeeze_releasing": swing.squeeze_releasing,
                })

                # Determine eligibility
                blocked = False
                reason = ""
                if swing.adx_blocked:
                    blocked = True
                    reason = f"ADX too low ({swing.adx_1h:.0f})"
                elif swing.regime_blocked:
                    blocked = True
                    reason = f"Regime: {swing.regime}"
                elif swing.session_blocked:
                    blocked = True
                    reason = f"Session: {swing.session}"
                elif swing.daily_blocked:
                    blocked = True
                    reason = "Daily trend opposes"

                pair_data["eligible"] = not blocked
                pair_data["block_reason"] = reason

        result["pairs"][symbol] = pair_data

    return _sanitize_floats(result)


@app.get("/api/signals/stream")
async def signal_stream():
    """SSE endpoint for live signal feed."""
    async def event_generator():
        last_count = 0
        while True:
            try:
                signals = db.get_signals(limit=5)
                current_count = len(signals)
                if signals and current_count != last_count:
                    import json as _json
                    data = _json.dumps(_sanitize_floats({"signals": signals}))
                    yield f"data: {data}\n\n"
                    last_count = current_count
            except Exception:
                pass
            await asyncio.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/analytics/winrate")
async def get_winrate_analytics(
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    regime: Optional[str] = None,
    session: Optional[str] = None,
):
    """Win rate breakdowns with multiple dimensions."""
    conn = db.get_conn()
    base_where = "WHERE is_open = 0"

    if symbol:
        base_where += f" AND symbol = '{symbol}'"
    if side:
        base_where += f" AND side = '{side}'"

    # By confidence bucket
    conf_buckets = conn.execute(f"""
        SELECT
            CASE
                WHEN swing_confidence >= 0.85 THEN '0.85+'
                WHEN swing_confidence >= 0.75 THEN '0.75-0.85'
                WHEN swing_confidence >= 0.65 THEN '0.65-0.75'
                ELSE '0.55-0.65'
            END as bucket,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            COALESCE(AVG(pnl_usd), 0) as avg_pnl,
            COALESCE(AVG(post_fee_rr), 0) as avg_rr
        FROM trades {base_where}
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()

    # By symbol
    by_symbol = conn.execute(f"""
        SELECT symbol,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl,
               COALESCE(AVG(pnl_usd), 0) as avg_pnl
        FROM trades {base_where}
        GROUP BY symbol
    """).fetchall()

    # By direction
    by_direction = conn.execute(f"""
        SELECT side,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades {base_where}
        GROUP BY side
    """).fetchall()

    # By regime
    by_regime = conn.execute(f"""
        SELECT COALESCE(regime, 'unknown') as regime,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades {base_where}
        GROUP BY regime
    """).fetchall()

    # By session
    by_session = conn.execute(f"""
        SELECT COALESCE(
            CASE
                WHEN CAST(strftime('%H', exit_time, 'unixepoch') AS INT) BETWEEN 13 AND 16 THEN 'overlap'
                WHEN CAST(strftime('%H', exit_time, 'unixepoch') AS INT) BETWEEN 7 AND 12 THEN 'eu'
                WHEN CAST(strftime('%H', exit_time, 'unixepoch') AS INT) BETWEEN 17 AND 20 THEN 'us'
                WHEN CAST(strftime('%H', exit_time, 'unixepoch') AS INT) BETWEEN 0 AND 6 THEN 'asian'
                ELSE 'dead_zone'
            END, 'unknown') as session,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
            COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades {base_where}
        GROUP BY session
    """).fetchall()

    # By exit reason
    by_exit = conn.execute(f"""
        SELECT COALESCE(exit_reason, 'unknown') as exit_reason,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades {base_where}
        GROUP BY exit_reason
    """).fetchall()

    # By day of week
    by_dow = conn.execute(f"""
        SELECT CASE CAST(strftime('%w', entry_time, 'unixepoch') AS INT)
                 WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue'
                 WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri'
                 WHEN 6 THEN 'Sat' END as dow,
               COUNT(*) as trades,
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl_usd), 0) as total_pnl
        FROM trades {base_where}
        GROUP BY dow
    """).fetchall()

    return _sanitize_floats({
        "by_confidence": [dict(r) for r in conf_buckets],
        "by_symbol": [dict(r) for r in by_symbol],
        "by_direction": [dict(r) for r in by_direction],
        "by_regime": [dict(r) for r in by_regime],
        "by_session": [dict(r) for r in by_session],
        "by_exit_reason": [dict(r) for r in by_exit],
        "by_day_of_week": [dict(r) for r in by_dow],
    })


@app.get("/api/analytics/risk")
async def get_risk_analytics():
    """Current risk metrics — drawdown, sizing, leverage, cooldowns."""
    result = {
        "drawdown_pct": 0.0,
        "max_drawdown_pct": 25.0,
        "daily_pnl": 0.0,
        "daily_target_pct": CONFIG.daily_target.daily_target_pct,
        "daily_loss_limit": 0.0,
        "daily_loss_consumed_pct": 0.0,
        "mode": "normal",
        "positions": [],
        "cooldowns": {},
        "sizer_state": {},
    }

    if bot_runner.bot and bot_runner.bot.risk:
        rs = bot_runner.bot.risk.state
        result["drawdown_pct"] = (1 - rs.current_equity / rs.peak_equity) * 100 if rs.peak_equity > 0 else 0
        result["daily_pnl"] = rs.daily_pnl
        result["is_halted"] = rs.is_halted
        result["halt_reason"] = rs.halt_reason
        result["consecutive_losses"] = rs.consecutive_losses

    if bot_runner.bot and hasattr(bot_runner.bot, 'daily_target'):
        dt = bot_runner.bot.daily_target.state
        result["mode"] = dt.mode.value
        result["daily_loss_limit"] = dt.daily_loss_limit
        result["daily_loss_consumed_pct"] = dt.daily_loss_consumed_pct
        result["pct_achieved"] = dt.pct_achieved

    if bot_runner.bot and bot_runner.bot.shadow:
        sizer = bot_runner.bot.shadow.sizer
        result["sizer_state"] = sizer.get_state()
        # Per-position info
        for sym, trade in bot_runner.bot.shadow.open_trades.items():
            result["positions"].append({
                "symbol": sym,
                "side": trade.side,
                "leverage": trade.leverage,
                "kelly_pct": trade.kelly_pct,
                "confidence_mult": trade.confidence_mult,
                "drawdown_mult": trade.drawdown_mult,
                "notional_usd": trade.usd_value * trade.leverage,
            })

    return _sanitize_floats(result)


@app.get("/api/trades/{trade_id}")
async def get_trade_detail(trade_id: int):
    """Detailed trade with all indicator snapshots."""
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"Trade {trade_id} not found")

    trade = dict(row)

    # Parse extra_json if present
    if trade.get("extra_json"):
        try:
            trade["extra"] = json.loads(trade["extra_json"])
        except Exception:
            trade["extra"] = {}

    # Get notes
    notes = db.get_state(f"trade_notes_{trade_id}")
    trade["notes"] = notes or ""

    return _sanitize_floats(trade)


@app.post("/api/trades/{trade_id}/notes")
async def save_trade_notes(trade_id: int, request: Request):
    """Add manual notes to a trade."""
    body = await request.json()
    notes = body.get("notes", "")
    db.set_state(f"trade_notes_{trade_id}", notes)
    return {"trade_id": trade_id, "notes": notes, "saved": True}


@app.get("/api/analytics/performance")
async def get_advanced_performance():
    """Advanced performance: Sharpe, Sortino, rolling win rates, cumulative P&L gross vs net."""
    conn = db.get_conn()
    trades = conn.execute("""
        SELECT exit_time, pnl_usd, gross_pnl_usd, fee_cost_usd, symbol, side
        FROM trades WHERE is_open = 0
        ORDER BY exit_time ASC
    """).fetchall()
    trades = [dict(r) for r in trades]

    if not trades:
        return _sanitize_floats({
            "sharpe": 0, "sortino": 0, "max_drawdown_pct": 0,
            "rolling_7d_wr": 0, "rolling_30d_wr": 0,
            "cumulative_gross": [], "cumulative_net": [],
        })

    import math

    # Compute returns
    pnls = [t["pnl_usd"] for t in trades]
    gross_pnls = [t.get("gross_pnl_usd") or t["pnl_usd"] for t in trades]

    equity = CONFIG.trading.starting_capital_idr / IDR_PER_USD
    returns = [p / equity for p in pnls]  # Simple returns
    mean_r = sum(returns) / len(returns) if returns else 0

    # Sharpe (annualized, assuming ~3 trades/day, 365 days)
    trades_per_day = len(trades) / max(
        (trades[-1]["exit_time"] - trades[0]["exit_time"]) / 86400, 1
    )
    periods_per_year = trades_per_day * 365
    rf_per_period = 0.04 / periods_per_year if periods_per_year > 0 else 0

    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns) if returns else 0
    std = math.sqrt(variance) if variance > 0 else 0
    sharpe = ((mean_r - rf_per_period) / std * math.sqrt(periods_per_year)) if std > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    ds_var = sum(r ** 2 for r in downside) / len(returns) if downside else 0
    ds_std = math.sqrt(ds_var) if ds_var > 0 else 0
    sortino = ((mean_r - rf_per_period) / ds_std * math.sqrt(periods_per_year)) if ds_std > 0 else 0

    # Max drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = (peak - cum) / (equity + peak) * 100 if (equity + peak) > 0 else 0
        max_dd = max(max_dd, dd)

    # Rolling win rates (last 7 and 30 trades as proxy)
    recent_7 = pnls[-7:] if len(pnls) >= 7 else pnls
    recent_30 = pnls[-30:] if len(pnls) >= 30 else pnls
    wr_7 = sum(1 for p in recent_7 if p > 0) / len(recent_7) * 100 if recent_7 else 0
    wr_30 = sum(1 for p in recent_30 if p > 0) / len(recent_30) * 100 if recent_30 else 0

    # Cumulative gross vs net
    cum_gross = []
    cum_net = []
    g_sum = 0
    n_sum = 0
    for t in trades:
        g_sum += t.get("gross_pnl_usd") or t["pnl_usd"]
        n_sum += t["pnl_usd"]
        cum_gross.append({"timestamp": t["exit_time"], "value": round(g_sum, 4)})
        cum_net.append({"timestamp": t["exit_time"], "value": round(n_sum, 4)})

    return _sanitize_floats({
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "rolling_7d_wr": round(wr_7, 1),
        "rolling_30d_wr": round(wr_30, 1),
        "cumulative_gross": cum_gross,
        "cumulative_net": cum_net,
        "total_trades": len(trades),
        "avg_hold_s": sum(
            (t.get("exit_time", 0) or 0) - (t.get("entry_time", 0) or 0)
            for t in trades
        ) / len(trades) if trades else 0,
    })


@app.get("/api/learning/recent")
async def get_learning_activity():
    """Recent AI learning activity and knowledge base entries."""
    try:
        from ai_learning.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(db_path=CONFIG.ai_learning.db_path,
                           embedding_dim=CONFIG.ai_learning.embedding_dim)
        history = kb.get_ingestion_history(10)

        # Get recent high-confidence entries
        conn = kb._get_conn()
        recent = conn.execute("""
            SELECT id, source_type, source_title, category, content,
                   confidence, extraction_date, times_applied, success_rate
            FROM knowledge_entries
            ORDER BY extraction_date DESC LIMIT 20
        """).fetchall()
        recent_entries = [dict(r) for r in recent]

        # Parse content JSON for display
        for entry in recent_entries:
            try:
                entry["content"] = json.loads(entry["content"])
            except (json.JSONDecodeError, TypeError):
                pass

        return _sanitize_floats({
            "status": "active",
            "insights": recent_entries[:10],
            "adjustments": [],
            "recent_ingestions": history,
            "total_entries": kb.get_stats().get("total_entries", 0),
        })
    except Exception:
        return {
            "status": "coming_soon",
            "insights": [],
            "adjustments": [],
            "recent_ingestions": [],
            "message": "AI learning module not yet initialized. Run: python -m ai_learning stats",
        }


@app.get("/api/learning/stats")
async def get_learning_stats():
    """Knowledge base statistics."""
    try:
        from ai_learning.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(db_path=CONFIG.ai_learning.db_path,
                           embedding_dim=CONFIG.ai_learning.embedding_dim)
        return _sanitize_floats(kb.get_stats())
    except Exception as e:
        return {"error": str(e), "total_entries": 0}


@app.get("/api/learning/search")
async def search_knowledge(q: str = Query("", min_length=1),
                           top_k: int = Query(5, ge=1, le=50),
                           category: str = Query(None)):
    """Search the knowledge base semantically."""
    try:
        from ai_learning.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(db_path=CONFIG.ai_learning.db_path,
                           embedding_dim=CONFIG.ai_learning.embedding_dim)
        results = kb.search(q, top_k=top_k, category=category)
        return _sanitize_floats({
            "query": q,
            "results": [
                {
                    "score": r.score,
                    "category": r.entry.category,
                    "content": r.entry.to_dict().get("content", {}),
                    "confidence": r.entry.confidence,
                    "source_type": r.entry.source_type,
                    "source_title": r.entry.source_title,
                    "source_url": r.entry.source_url,
                    "times_applied": r.entry.times_applied,
                    "success_rate": r.entry.success_rate,
                }
                for r in results
            ],
        })
    except Exception as e:
        return {"query": q, "results": [], "error": str(e)}


@app.get("/api/advisor/stats")
async def get_advisor_stats():
    """AI advisor statistics: consultations, agreement rate, cache."""
    bot = bot_runner.bot
    if not bot or not getattr(bot, "advisor", None):
        return {"error": "Advisor not initialized", "total_consultations": 0}
    return _sanitize_floats(bot.advisor.get_stats())


@app.get("/api/advisor/consultation/{trade_id}")
async def get_advisor_consultation(trade_id: int):
    """Get advisor consultation record for a specific trade."""
    bot = bot_runner.bot
    if not bot or not getattr(bot, "advisor", None):
        raise HTTPException(404, "Advisor not initialized")
    record = bot.advisor.get_consultation_for_trade(trade_id)
    if record is None:
        raise HTTPException(404, f"No consultation found for trade {trade_id}")
    return _sanitize_floats(record)


@app.get("/api/advisor/kb-performance")
async def get_advisor_kb_performance():
    """KB entries ranked by application count with success/failure status."""
    bot = bot_runner.bot
    if not bot or not getattr(bot, "advisor", None):
        return {"entries": []}
    entries = bot.advisor.get_kb_performance_report()
    return _sanitize_floats({"entries": entries})


@app.get("/health")
async def health_check():
    """Health check for Render."""
    return {"status": "healthy", "bot": bot_runner.status, "timestamp": time.time()}


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Trading Bot API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n  Trading Bot API starting on http://{args.host}:{args.port}")
    print(f"  Dashboard: http://localhost:3000")
    print(f"  API docs:  http://{args.host}:{args.port}/docs\n")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
