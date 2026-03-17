"""
FastAPI server — REST API wrapping the trading bot.
Hosts on Render, provides endpoints for the Vercel frontend dashboard.

Usage:
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import database as db
from config import CONFIG, BotConfig, IDR_PER_USD

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
        if self.bot and self.status == "running":
            self.status = "stopping"
            self.bot._running = False
            if self.task:
                self.task.cancel()
                try:
                    await asyncio.wait_for(self.task, timeout=10)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            try:
                await self.bot.shutdown()
            except Exception:
                pass
            self.status = "stopped"

    def _hook_shadow_trader(self, shadow):
        """Monkey-patch shadow trader to also write to DB."""
        original_log_trade = shadow._log_trade
        original_log_signal = shadow._log_signal

        def log_trade_with_db(trade, event):
            original_log_trade(trade, event)
            try:
                from dataclasses import asdict
                trade_dict = asdict(trade)
                trade_dict['leverage'] = CONFIG.futures.leverage
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
            result.update({
                "equity_usd": rs.current_equity,
                "peak_equity_usd": rs.peak_equity,
                "drawdown_pct": (1 - rs.current_equity / rs.peak_equity) * 100 if rs.peak_equity > 0 else 0,
                "daily_pnl_usd": rs.daily_pnl,
                "total_trades": rs.total_trades,
                "wins": rs.wins,
                "losses": rs.losses,
                "win_rate": rs.wins / rs.total_trades * 100 if rs.total_trades > 0 else 0,
            })

        if self.bot and self.bot.shadow:
            result["shadow_total_pnl"] = self.bot.shadow.total_pnl
            result["shadow_open_trades"] = len(self.bot.shadow.open_trades)

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
    # Initialize database
    db.init_db()
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

        logger.info("Auto-starting bot with environment API keys...")
        await bot_runner.start(CONFIG)
    else:
        logger.warning("No API keys found — bot not auto-started. Set BINANCE_API_KEY and BINANCE_API_SECRET.")

    yield

    # Shutdown
    await bot_runner.stop()


app = FastAPI(
    title="Crypto Swing Trading Bot API",
    description="REST API for monitoring and controlling the trading bot",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow Vercel frontend
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
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
    return bot_runner.get_status()


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
        for sym, trade in bot_runner.bot.shadow.open_trades.items():
            from dataclasses import asdict
            td = asdict(trade)
            td["leverage"] = CONFIG.futures.leverage
            # Calculate live P&L
            mid = td.get("entry_price", 0)
            if bot_runner.bot.stream and hasattr(bot_runner.bot.stream, '_last_update'):
                pass  # Price comes from the trade's current state
            live_positions.append(td)

    return {
        "positions": live_positions if live_positions else open_trades,
        "count": len(live_positions) if live_positions else len(open_trades),
    }


@app.get("/api/signals")
async def get_signals(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None,
):
    """Recent analysis signals."""
    signals = db.get_signals(limit=limit, symbol=symbol)
    return {"signals": signals, "count": len(signals)}


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

    return summary


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


@app.get("/health")
async def health_check():
    """Health check for Render."""
    return {"status": "healthy", "bot": bot_runner.status, "timestamp": time.time()}


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
