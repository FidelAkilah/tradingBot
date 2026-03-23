"""
Main Orchestrator — Swing Trading Bot with Candle + OB Confluence.

Usage:
    python main.py                          # Shadow mode (default)
    python main.py --live                   # Live on testnet
    python main.py --live --no-testnet      # Live on REAL Binance
    python main.py --pairs BTC/USDT ETH/USDT SOL/USDT
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Optional

from config import CONFIG, BotConfig, FuturesConfig, IDR_PER_USD
from candle_analyzer import CandleAnalyzer, SwingSignal
from correlation import CorrelationMatrix, CorrelationGuard, PortfolioHeatMap
from scanner import OpportunityScanner, PairSelector, PairPerformanceTracker
from daily_target import DailyTargetTracker, DailyTargetContext, ModeController, Compounder, TradingMode
from liquidity_analyzer import LiquidityAnalyzer
from order_manager import OrderManager, PositionSide
from reentry import ReentryManager
from risk_manager import RiskManager, TradeRecord
from shadow_trader import ShadowTrader
from vpin_analyzer import FlowRegime
from websocket_client import OrderBookStream

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", mode="a"),
        ],
    )
    logging.getLogger("ccxt").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENV / API KEY LOADING
# ─────────────────────────────────────────────

def load_env_file(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def load_api_keys(config: BotConfig) -> BotConfig:
    load_env_file()
    if not config.exchange.api_key:
        config.exchange.api_key = os.environ.get("BINANCE_API_KEY")
    if not config.exchange.api_secret:
        config.exchange.api_secret = os.environ.get("BINANCE_API_SECRET")
    testnet_env = os.environ.get("BINANCE_TESTNET", "").lower()
    if testnet_env in ("false", "0", "no"):
        config.exchange.sandbox = False
    return config


async def validate_connection(config: BotConfig) -> bool:
    import ccxt.pro as ccxtpro
    logger.info("Validating Binance API connection...")

    exchange = getattr(ccxtpro, config.exchange.exchange_id)({
        "apiKey": config.exchange.api_key,
        "secret": config.exchange.api_secret,
        "sandbox": config.exchange.sandbox,
        "enableRateLimit": True,
    })

    try:
        balance = await exchange.fetch_balance()
        usdt_free = balance.get("USDT", {}).get("free", 0)
        usdt_total = balance.get("USDT", {}).get("total", 0)
        mode = "TESTNET" if config.exchange.sandbox else "MAINNET"
        logger.info(f"  Connected to Binance {mode}")
        logger.info(f"  USDT balance: {usdt_free:.2f} free / {usdt_total:.2f} total")
        logger.info(f"  ≈ IDR {usdt_total * IDR_PER_USD:,.0f}")

        if not config.shadow.enabled:
            try:
                await exchange.fetch_open_orders("BTC/USDT")
                logger.info("  Trade permissions: OK")
            except Exception:
                logger.warning("  Could not verify trade permissions.")

        await exchange.close()
        return True
    except Exception as e:
        logger.error(f"  Connection error: {e}")
        try:
            await exchange.close()
        except Exception:
            pass
        return False


# ─────────────────────────────────────────────
# BOT ORCHESTRATOR
# ─────────────────────────────────────────────

class ScalpingBot:
    """
    Swing trading bot using candle confluence + order book confirmation.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.is_shadow = config.shadow.enabled

        # Compute USD equity from IDR
        idr_capital = config.trading.starting_capital_idr
        usd_equity = idr_capital / IDR_PER_USD
        config.trading.max_position_usd = usd_equity * config.trading.max_position_pct

        # Core modules
        self.stream = OrderBookStream(config)
        self.analyzer = LiquidityAnalyzer(config)
        self.candle_analyzer = CandleAnalyzer(config)
        self.risk = RiskManager(initial_equity=usd_equity, config=config)
        self.shadow = ShadowTrader(config) if self.is_shadow else None
        self.order_manager: Optional[OrderManager] = None

        # Daily target system
        self.daily_target = DailyTargetTracker(usd_equity, config)
        self.mode_controller = ModeController(config)
        self.compounder = Compounder(self.daily_target, config)

        # Smart re-entry after stop-outs
        self.reentry_manager = ReentryManager(config)

        # Correlation guard and portfolio exposure
        self.corr_matrix = CorrelationMatrix(config)
        self.corr_guard = CorrelationGuard(self.corr_matrix, config)
        self.portfolio_heat = PortfolioHeatMap(config)

        # Opportunity scanner for dynamic pair selection
        self.scanner = OpportunityScanner(config)
        self.pair_selector = PairSelector(config)
        self.pair_performance = PairPerformanceTracker(config)

        # Swing signal cache (updated every 60s, not every OB tick)
        self._swing_cache: dict = {}  # symbol -> SwingSignal
        self._last_candle_fetch: dict = {}

        # Shutdown state
        self._running = False
        self._shutdown_requested = False
        self._force_quit_count = 0
        self._server_mode = False          # True when running under uvicorn/server.py
        self._analysis_count = 0
        self._print_interval = 20

        logger.info(f"  Capital: IDR {idr_capital:,.0f} ≈ ${usd_equity:.2f}")
        logger.info(f"  Max position: ${config.trading.max_position_usd:.2f}")
        logger.info(f"  Daily target: {config.trading.daily_target_pct}%")

    async def start(self):
        mode = "SHADOW" if self.is_shadow else "LIVE"
        logger.info(f"{'='*60}")
        logger.info(f"  SWING TRADING BOT — {mode} MODE")
        logger.info(f"  Candle-driven entries, OB confirmation")
        logger.info(f"{'='*60}")
        logger.info(f"  Press Ctrl+C to stop. Press again to force quit.")

        await self.stream.initialize()

        # ── Set up Futures leverage & margin type ──
        if self.config.futures.enabled:
            await self._setup_futures()

        # Filter out stablecoins and low-vol pairs
        await self.stream.discover_top_pairs()
        self.stream.active_pairs = [
            p for p in self.stream.active_pairs
            if p not in self.config.trading.excluded_pairs
        ]
        logger.info(f"  Active pairs (after filtering): {self.stream.active_pairs}")

        self.order_manager = OrderManager(self.stream.exchange, self.config)

        # Fetch initial candle data for all pairs
        logger.info("  Fetching initial candle data...")
        for symbol in self.stream.active_pairs:
            await self._update_swing_signal(symbol)

        # Initialize correlation matrix
        await self._update_correlation_matrix()

        # Run initial pair scan if scanner enabled
        if self.config.scanner.enabled:
            await self._run_pair_scan()

        self.stream.on_order_book(self._on_order_book_update)

        # Only install signal handlers when running standalone (not under uvicorn)
        if not self._server_mode:
            self._install_signal_handlers()

        self._running = True
        await self.stream.start()

        try:
            await self._periodic_reporter()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _setup_futures(self):
        """Set leverage and margin type for all trading pairs on Binance Futures."""
        fc = self.config.futures
        exchange = self.stream.exchange
        pairs = self.config.trading.manual_pairs

        logger.info(f"  Setting up Futures: {fc.leverage}x leverage, {fc.margin_type} margin")

        for symbol in pairs:
            try:
                # Set margin type (ISOLATED or CROSSED)
                try:
                    await exchange.set_margin_mode(fc.margin_type.lower(), symbol)
                    logger.info(f"  [{symbol}] Margin type: {fc.margin_type}")
                except Exception as e:
                    err_str = str(e)
                    if "No need to change" in err_str or "already" in err_str.lower():
                        logger.info(f"  [{symbol}] Margin type already {fc.margin_type}")
                    elif "-4168" in err_str or "Multi-Assets" in err_str:
                        # Multi-Assets mode doesn't allow margin type changes — that's fine
                        logger.info(f"  [{symbol}] Multi-Assets mode active, using CROSSED margin")
                    else:
                        logger.warning(f"  [{symbol}] Margin type error: {e}")

                # Set leverage
                await exchange.set_leverage(fc.leverage, symbol)
                logger.info(f"  [{symbol}] Leverage: {fc.leverage}x")

            except Exception as e:
                logger.warning(f"  [{symbol}] Futures setup error: {e}")

        logger.info(f"  Futures setup complete for {len(pairs)} pairs")

    def _install_signal_handlers(self):
        loop = asyncio.get_running_loop()

        def _handle_signal():
            self._force_quit_count += 1
            if self._force_quit_count == 1:
                logger.info("\nShutdown requested. Stopping gracefully...")
                self._running = False
                self._shutdown_requested = True
                for task in asyncio.all_tasks(loop):
                    if task is not asyncio.current_task():
                        task.cancel()
            else:
                logger.warning("\nForce quit.")
                # Only stop the loop if we're running standalone (not under uvicorn)
                if not self._server_mode:
                    loop.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except NotImplementedError:
                pass

    async def _update_swing_signal(self, symbol: str):
        """Fetch candles and compute swing signal (called every ~60s)."""
        try:
            candles = await self.candle_analyzer.fetch_candles(
                self.stream.exchange, symbol
            )
            # Fetch daily candles (300s TTL — separate from 1h/4h)
            await self.candle_analyzer.fetch_daily(
                self.stream.exchange, symbol
            )
            # Fetch 15m candles (60s TTL — for entry timing)
            await self.candle_analyzer.fetch_15m(
                self.stream.exchange, symbol
            )
            # Fetch funding rate (5 min TTL)
            await self._fetch_funding_rate(symbol)
            # Fetch open interest (5 min TTL)
            await self._fetch_open_interest(symbol)

            if candles:
                swing = self.candle_analyzer.analyze(symbol, candles, time.time())
                self._swing_cache[symbol] = swing
                self._last_candle_fetch[symbol] = time.time()
        except Exception as e:
            logger.warning(f"[{symbol}] Candle fetch error: {e}")

    async def _fetch_funding_rate(self, symbol: str):
        """Fetch funding rate from exchange (cached by FundingRateAnalyzer TTL)."""
        vc = self.config.volume
        if not vc.funding_enabled:
            return

        fa = self.candle_analyzer._funding_analyzer
        now = time.time()
        last = fa._last_fetch.get(symbol, 0)
        if now - last < vc.funding_fetch_ttl:
            return

        try:
            fr = await self.stream.exchange.fetch_funding_rate(symbol)
            if fr and 'fundingRate' in fr:
                rate = float(fr['fundingRate']) * 100.0  # Convert to percentage
                fa.update_rate(symbol, rate, now)
                fa._last_fetch[symbol] = now
        except Exception as e:
            logger.debug(f"[{symbol}] Funding rate fetch error: {e}")

    async def _fetch_open_interest(self, symbol: str):
        """Fetch open interest from exchange (cached by OI analyzer TTL)."""
        vc = self.config.volume
        if not vc.oi_enabled:
            return

        oia = self.candle_analyzer._oi_analyzer
        now = time.time()
        last = oia._last_fetch.get(symbol, 0)
        if now - last < vc.oi_fetch_ttl:
            return

        try:
            # ccxt unified: fetch_open_interest returns {openInterestValue, ...}
            oi_data = await self.stream.exchange.fetch_open_interest(symbol)
            if oi_data and 'openInterestValue' in oi_data:
                oi_val = float(oi_data['openInterestValue'])
                oia.update_oi(symbol, oi_val, now)
                oia._last_fetch[symbol] = now
        except Exception as e:
            logger.debug(f"[{symbol}] OI fetch error: {e}")

    async def _update_correlation_matrix(self):
        """Fetch daily closes for all pairs and compute correlation matrix."""
        if not self.config.correlation.enabled:
            return
        if not self.corr_matrix.is_stale():
            return

        try:
            # Try loading cached matrix from bot_state
            import database as db
            cached = db.get_state("correlation_matrix")
            if cached:
                self.corr_matrix.load_matrix_dict(cached)
                if not self.corr_matrix.is_stale():
                    logger.info("  Correlation matrix loaded from cache")
                    return
        except Exception:
            pass

        # Fetch daily candle closes for each pair
        daily_closes = {}
        lookback = self.config.correlation.lookback_days
        for symbol in self.stream.active_pairs:
            try:
                ohlcv = await self.stream.exchange.fetch_ohlcv(
                    symbol, "1d", limit=lookback + 5
                )
                if ohlcv and len(ohlcv) >= self.config.correlation.min_candles:
                    daily_closes[symbol] = [c[4] for c in ohlcv]  # Close prices
            except Exception as e:
                logger.debug(f"[{symbol}] Daily close fetch for correlation: {e}")

        if len(daily_closes) >= 2:
            self.corr_matrix.update_prices(daily_closes)
            matrix = self.corr_matrix.compute()
            logger.info(
                f"  Correlation matrix computed: {len(daily_closes)} pairs, "
                f"{len(matrix)} entries"
            )
            # Persist to bot_state
            try:
                import database as db
                db.set_state("correlation_matrix", self.corr_matrix.get_matrix_dict())
            except Exception as e:
                logger.debug(f"Failed to persist correlation matrix: {e}")

    async def _run_pair_scan(self):
        """Run the opportunity scanner and update active pairs."""
        if not self.config.scanner.enabled:
            return

        scan_start = time.time()
        try:
            # Update performance stats from DB
            self.pair_performance.update_from_db()

            # Run scanner
            scores = await self.scanner.scan(self.stream.exchange)
            if not scores:
                logger.warning("Scanner: no results, keeping current pairs")
                return

            # Get open position symbols
            open_syms = []
            if self.is_shadow and self.shadow:
                open_syms = list(self.shadow.open_trades.keys())
            elif self.order_manager:
                open_syms = list(self.order_manager.positions.keys())

            # Select pairs
            result = self.pair_selector.select(
                scores,
                open_positions=open_syms,
                disabled_pairs=self.pair_performance.get_disabled_pairs(),
                auto_include_pairs=self.pair_performance.get_auto_include_pairs(),
            )
            result.scan_duration_s = time.time() - scan_start

            # Update the active pair list on the stream
            new_pairs = result.get_active_pairs()
            old_pairs = set(self.stream.active_pairs)
            self.stream.active_pairs = new_pairs

            # Fetch initial candle data for newly added pairs
            for sym in result.added_pairs:
                if sym not in old_pairs:
                    await self._update_swing_signal(sym)

            # Set up futures for new pairs
            if self.config.futures.enabled:
                for sym in result.added_pairs:
                    if sym not in old_pairs:
                        try:
                            fc = self.config.futures
                            try:
                                await self.stream.exchange.set_margin_mode(
                                    fc.margin_type.lower(), sym
                                )
                            except Exception:
                                pass
                            await self.stream.exchange.set_leverage(fc.leverage, sym)
                        except Exception as e:
                            logger.debug(f"[{sym}] Futures setup error: {e}")

            logger.info(
                f"Scanner: scanned {result.pairs_scanned}, "
                f"qualified {result.pairs_qualified}, "
                f"selected {result.selected_pairs} "
                f"({result.scan_duration_s:.1f}s)"
            )
            if result.added_pairs:
                logger.info(f"  Added: {result.added_pairs}")
            if result.dropped_pairs:
                logger.info(f"  Dropped: {result.dropped_pairs}")
            logger.info(f"  Active pairs: {new_pairs}")

            # Persist scan result to bot_state
            try:
                import database as db
                db.set_state("last_scan", self.pair_selector.get_scan_summary())
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Scanner error: {e}", exc_info=True)

    def _get_open_positions_info(self) -> dict:
        """Get open position info dict for correlation guard checks."""
        positions = {}
        if self.is_shadow and self.shadow:
            for sym, trade in self.shadow.open_trades.items():
                positions[sym] = {
                    "side": trade.side,
                    "confidence": trade.swing_confidence,
                    "usd_value": trade.usd_value,
                    "leverage": trade.leverage,
                    "amount": trade.amount,
                    "entry_price": trade.entry_price,
                }
        elif self.order_manager:
            for sym, pos in self.order_manager.positions.items():
                positions[sym] = {
                    "side": "BUY" if pos.side == PositionSide.LONG else "SELL",
                    "confidence": 0.6,
                    "usd_value": pos.margin,
                    "leverage": pos.leverage,
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                }
        return positions

    async def _on_order_book_update(self, symbol: str, order_book: dict, timestamp: float):
        if not self._running:
            return

        # Refresh candle data every 60s
        last_fetch = self._last_candle_fetch.get(symbol, 0)
        if timestamp - last_fetch > 60.0:
            await self._update_swing_signal(symbol)

        recent_trades = self.stream.get_recent_trades(symbol)
        swing = self._swing_cache.get(symbol)

        # Run full analysis with swing signal
        analysis = self.analyzer.analyze(symbol, order_book, recent_trades, timestamp, swing)
        self._analysis_count += 1

        # ── Shadow Mode ──
        if self.is_shadow and self.shadow:
            # ALWAYS monitor open positions — even when risk is halted.
            # Price updates, TP/SL, wall-pull, and max-hold checks must run
            # regardless of whether new entries are allowed.

            # VPIN dynamic stop widening (one-time, capped at 2x original distance)
            if symbol in self.shadow.open_trades and analysis.vpin.should_widen_stops:
                st = self.shadow.open_trades[symbol]
                if not hasattr(st, '_original_stop_dist') or st._original_stop_dist is None:
                    if st.side == "BUY":
                        st._original_stop_dist = st.entry_price - st.stop_price
                    else:
                        st._original_stop_dist = st.stop_price - st.entry_price

                if st._original_stop_dist and st._original_stop_dist > 0:
                    mult = min(analysis.vpin.stop_multiplier, 2.0)
                    if st.side == "BUY":
                        st.stop_price = st.entry_price - (st._original_stop_dist * mult)
                    else:
                        st.stop_price = st.entry_price + (st._original_stop_dist * mult)

            # Drain pending partial TP records (queued by update_prices)
            for partial_rec in self.shadow.pending_partial_records:
                self.daily_target.record_trade(partial_rec.pnl_usd)
                # Update equity for partial PnL
                self._equity_from_partial(partial_rec.pnl_usd)
            self.shadow.pending_partial_records.clear()

            trade_record = self.shadow.update_prices(symbol, analysis.mid_price)
            if trade_record:
                self.risk.record_trade(trade_record)
                self.daily_target.record_trade(trade_record.pnl_usd)
                self.pair_performance.record_trade(symbol, trade_record.pnl_usd)

                # Register stop-outs for potential re-entry
                if trade_record.reason == "stop_loss":
                    st = self.shadow.closed_trades[-1] if self.shadow.closed_trades else None
                    if st:
                        self.reentry_manager.register_stopout(
                            symbol=st.symbol,
                            side=st.side,
                            entry_price=st.entry_price,
                            stop_out_price=trade_record.exit_price,
                            stop_out_time=trade_record.timestamp,
                            confidence=st.swing_confidence,
                            atr=st.atr_at_entry,
                            amount=st.original_amount or trade_record.amount,
                        )

            # Update unrealized P&L for daily target tracking
            unrealized = sum(
                (self.shadow.last_prices.get(s, t.entry_price) - t.entry_price) * t.amount
                if t.side == "BUY" else
                (t.entry_price - self.shadow.last_prices.get(s, t.entry_price)) * t.amount
                for s, t in self.shadow.open_trades.items()
            )
            self.daily_target.update_unrealized(unrealized)
            self.daily_target.update_equity(self.shadow._equity)

            # Check daily reset (compound)
            daily_summary = self.compounder.check_daily_reset(self.shadow._equity)
            if daily_summary:
                try:
                    import database as db
                    db.insert_daily_equity(daily_summary)
                except Exception as e:
                    logger.error(f"Failed to persist daily equity: {e}")
                self.risk.reset_daily(new_equity=self.shadow._equity)

            # Evaluate trading mode
            regime_trending = True
            if swing and hasattr(swing, 'adx_1h'):
                regime_trending = swing.adx_1h >= self.config.regime.adx_weak
            self.daily_target.state.mode = self.mode_controller.evaluate(
                self.daily_target.state, regime_trending
            )

            # Minimum hold time check — don't close before min_hold
            if symbol in self.shadow.open_trades:
                st = self.shadow.open_trades[symbol]
                hold_time = timestamp - st.entry_time
                if hold_time < self.config.trading.min_hold_time_s:
                    pass  # Don't check wall health yet
                elif not self.config.trading.exit_on_wall_pull:
                    pass  # Wall pull doesn't trigger exit for swing
                else:
                    walls = analysis.bid_walls if st.side == "BUY" else analysis.ask_walls
                    wall_exists = any(
                        abs(w.price - st.wall_price) / st.wall_price < 0.002
                        and not w.is_spoof_suspect
                        for w in walls
                    )
                    if not wall_exists:
                        record = self.shadow.close_on_wall_pull(symbol, analysis.mid_price)
                        if record:
                            self.risk.record_trade(record)

                # Max hold time — force close
                if hold_time > self.config.trading.max_hold_time_s:
                    record = self.shadow.close_on_wall_pull(symbol, analysis.mid_price)
                    if record:
                        logger.info(f"[{symbol}] Max hold time reached, closing.")
                        self.risk.record_trade(record)

            # Only open new trades if risk allows AND spread is acceptable
            can_trade, reason = self.risk.can_trade()
            dt_halt, dt_reason = self.daily_target.should_halt()
            if dt_halt:
                can_trade = False
                reason = dt_reason
                if self.daily_target.state.mode != TradingMode.HALTED:
                    self.daily_target.state.mode = self.mode_controller.force_halt(
                        self.daily_target.state, dt_reason
                    )

            # Mode-based position limit
            max_pos = self.daily_target.get_max_positions()
            if len(self.shadow.open_trades) >= max_pos:
                can_trade = False

            # Mode-based confidence filter
            min_conf = self.daily_target.get_confidence_threshold()
            signal_conf = swing.confidence if swing else 0.0
            if signal_conf < min_conf:
                can_trade = False

            # Drawdown-based target reduction
            dd_pct = 0.0
            if self.risk.state.peak_equity > 0:
                dd_pct = ((self.risk.state.peak_equity - self.risk.state.current_equity)
                          / self.risk.state.peak_equity * 100.0)
            if dd_pct >= 15.0:
                self.daily_target.force_target(0.5)
            elif dd_pct >= 10.0:
                self.daily_target.force_target(1.0)

            if can_trade and self.risk.validate_spread(analysis.spread_pct):
                # Check for smart re-entry before normal signal processing
                if symbol not in self.shadow.open_trades:
                    swing_side = swing.suggested_side if swing else None
                    swing_conf = swing.confidence if swing else 0.0
                    swing_adx = max(swing.adx_1h, swing.adx_4h) if swing else 0.0
                    daily_loss_pct = self.daily_target.state.daily_loss_consumed_pct

                    reentry_side = self.reentry_manager.check_reentry(
                        symbol, analysis.mid_price, timestamp,
                        swing_side, swing_conf, swing_adx, daily_loss_pct,
                    )
                    if reentry_side:
                        # Override analysis suggestion for re-entry
                        analysis.trade_suggestion = reentry_side
                        self.shadow.daily_target_ctx = self.daily_target.get_sizing_context()
                        self.shadow.correlation_size_mult = 1.0  # re-entry already at 70% size
                        trade = self.shadow.process_signal(analysis)
                        if trade:
                            trade.is_reentry = True
                            trade.reentry_count = 1
                            self.reentry_manager.clear_candidate(symbol)
                        # Skip normal signal processing after re-entry attempt
                        return

                # 15m entry timing gate: wait for pullback before entering
                entry_ready = True
                if swing and swing.suggested_side:
                    entry_ready, rsi_15m = self.candle_analyzer.check_15m_entry(
                        symbol, swing.suggested_side
                    )
                    swing.entry_15m_ready = entry_ready
                    swing.entry_15m_rsi = rsi_15m

                if entry_ready:
                    # Correlation guard: check before opening 2nd position
                    corr_mult = 1.0
                    if (self.config.correlation.enabled
                            and swing and swing.suggested_side
                            and self.shadow.open_trades):
                        open_info = self._get_open_positions_info()
                        corr_result = self.corr_guard.check(
                            symbol, swing.suggested_side,
                            swing.confidence, open_info,
                        )
                        if not corr_result.allowed:
                            logger.debug(f"[{symbol}] Correlation block: {corr_result.reason}")
                            return
                        corr_mult = corr_result.size_multiplier

                    # Portfolio exposure check
                    if (self.config.correlation.enabled
                            and swing and swing.suggested_side):
                        open_info = self._get_open_positions_info()
                        equity = self.shadow._equity
                        sizing_usd = equity * self.config.trading.position_pct_of_equity
                        exp_ok, exp_frac, exp_reason = self.portfolio_heat.check_can_add(
                            swing.suggested_side, sizing_usd,
                            self.config.futures.leverage,
                            open_info, equity,
                        )
                        if not exp_ok:
                            logger.debug(f"[{symbol}] Exposure block: {exp_reason}")
                            return
                        corr_mult = min(corr_mult, exp_frac)

                    # Build daily target context for position sizer
                    self.shadow.daily_target_ctx = self.daily_target.get_sizing_context()
                    self.shadow.correlation_size_mult = corr_mult
                    self.shadow.process_signal(analysis)

        # ── Live Mode ──
        else:
            if self.order_manager:
                # Always monitor existing positions
                if symbol in self.order_manager.positions and analysis.vpin.should_widen_stops:
                    pos = self.order_manager.positions[symbol]
                    if not hasattr(pos, '_original_stop_dist') or pos._original_stop_dist is None:
                        if pos.side == PositionSide.LONG:
                            pos._original_stop_dist = pos.entry_price - pos.stop_loss_price
                        else:
                            pos._original_stop_dist = pos.stop_loss_price - pos.entry_price

                    if hasattr(pos, '_original_stop_dist') and pos._original_stop_dist and pos._original_stop_dist > 0:
                        mult = min(analysis.vpin.stop_multiplier, 2.0)
                        if pos.side == PositionSide.LONG:
                            pos.stop_loss_price = pos.entry_price - (pos._original_stop_dist * mult)
                        else:
                            pos.stop_loss_price = pos.entry_price + (pos._original_stop_dist * mult)

                cancelled = await self.order_manager.check_wall_health(analysis, is_shadow=False)
                action = await self.order_manager.update_positions(
                    symbol, analysis.mid_price, is_shadow=False
                )
                if action:
                    self.risk.record_trade(TradeRecord(
                        symbol=symbol, side="unknown", entry_price=0,
                        exit_price=analysis.mid_price, amount=0,
                        pnl_usd=0, reason=action, timestamp=timestamp,
                    ))

                # Only open new trades if risk allows + daily target
                can_trade, reason = self.risk.can_trade()
                dt_halt, dt_reason = self.daily_target.should_halt()
                if dt_halt:
                    can_trade = False
                if can_trade and self.risk.validate_spread(analysis.spread_pct):
                    # 15m entry timing gate
                    entry_ready = True
                    if swing and swing.suggested_side:
                        entry_ready, rsi_15m = self.candle_analyzer.check_15m_entry(
                            symbol, swing.suggested_side
                        )
                        swing.entry_15m_ready = entry_ready
                        swing.entry_15m_rsi = rsi_15m

                    if entry_ready:
                        equity = self.risk.state.current_equity
                        await self.order_manager.execute_signal(analysis, equity, is_shadow=False)

        # Periodic logging
        if self._analysis_count % self._print_interval == 0:
            logger.info(self.analyzer.get_analysis_summary(analysis))
            if swing:
                logger.info(self.candle_analyzer.get_signal_summary(swing))

    def _equity_from_partial(self, pnl: float):
        """Track equity changes from partial TP exits."""
        # Shadow trader already updates its own _equity in the partial TP loop.
        # This method just ensures daily_target has the latest equity.
        if self.is_shadow and self.shadow:
            self.daily_target.update_equity(self.shadow._equity)

    async def _periodic_reporter(self):
        while self._running:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break

            # Periodic pair scan (every 2 hours by default)
            if self.config.scanner.enabled and self.pair_selector.needs_scan():
                await self._run_pair_scan()

            if self._analysis_count > 0 and self._analysis_count % (self._print_interval * 6) < self._print_interval:
                if self.is_shadow and self.shadow:
                    logger.info(self.shadow.get_performance_summary())
                logger.info(self.risk.get_risk_summary())

                # Daily target progress
                dt = self.daily_target.state
                logger.info(
                    f"═══ Daily Target ═══\n"
                    f"  Mode: {dt.mode.value} | Target: {dt.daily_target_pct}%\n"
                    f"  Progress: {dt.pct_achieved:.1f}% | "
                    f"PnL: ${dt.total_pnl_today:+.2f} / ${dt.daily_target_amount:.2f}\n"
                    f"  Loss limit: {dt.daily_loss_consumed_pct:.1f}% consumed | "
                    f"Streak: {dt.streak_days}d"
                )

                for symbol in self.stream.active_pairs:
                    swing = self._swing_cache.get(symbol)
                    if swing:
                        logger.info(self.candle_analyzer.get_signal_summary(swing))

    async def shutdown(self):
        if self._shutdown_requested and self._force_quit_count > 1:
            return

        logger.info("Shutting down bot...")
        self._running = False

        try:
            if self.is_shadow and self.shadow:
                logger.info("\n" + self.shadow.get_performance_summary())
            logger.info("\n" + self.risk.get_risk_summary())
        except Exception:
            pass

        # Suppress ccxt CancelledError noise during shutdown
        loop = asyncio.get_event_loop()
        original_handler = loop.get_exception_handler()

        def _suppress_cancelled(loop, context):
            exc = context.get("exception")
            if isinstance(exc, asyncio.CancelledError):
                return  # Swallow — expected during shutdown
            if original_handler:
                original_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_suppress_cancelled)

        try:
            await asyncio.wait_for(self.stream.stop(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            if self.stream.exchange:
                try:
                    await asyncio.wait_for(self.stream.exchange.close(), timeout=2.0)
                except Exception:
                    pass

        # Restore handler after cleanup
        loop.set_exception_handler(original_handler)
        logger.info("Bot shutdown complete.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Swing Trading Bot — Candle + Order Book Confluence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Shadow mode
  python main.py --live                       # Live on testnet
  python main.py --live --no-testnet          # Live on mainnet
  python main.py --capital 2000000            # IDR 2M starting capital
  python main.py --pairs BTC/USDT SOL/USDT   # Specific pairs
        """,
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--no-testnet", action="store_true")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--capital", type=float, default=1_000_000, help="Starting capital in IDR")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--whale-mult", type=float)
    parser.add_argument("--stop-loss", type=float)
    parser.add_argument("--take-profit", type=float)
    parser.add_argument("--max-daily-loss", type=float)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--api-key")
    parser.add_argument("--api-secret")
    return parser.parse_args()


def apply_args(config: BotConfig, args) -> BotConfig:
    if args.live:
        config.shadow.enabled = False
    if args.no_testnet:
        config.exchange.sandbox = False
    else:
        config.exchange.sandbox = True
    if args.pairs:
        config.trading.use_dynamic_pairs = False
        config.trading.manual_pairs = args.pairs
    if args.top_n:
        config.trading.top_n_pairs = args.top_n
    config.trading.starting_capital_idr = args.capital
    if args.depth:
        config.order_book.depth_limit = args.depth
    if args.whale_mult:
        config.order_book.whale_multiplier = args.whale_mult
    if args.stop_loss:
        config.trading.stop_loss_pct = args.stop_loss
    if args.take_profit:
        config.trading.take_profit_pct = args.take_profit
    if args.max_daily_loss:
        config.risk.max_daily_loss_usd = args.max_daily_loss
    if args.api_key:
        config.exchange.api_key = args.api_key
    if args.api_secret:
        config.exchange.api_secret = args.api_secret
    config.log_level = args.log_level
    return config


async def main():
    args = parse_args()
    config = apply_args(CONFIG, args)
    config = load_api_keys(config)
    setup_logging(config.log_level)

    if not config.shadow.enabled:
        if not config.exchange.api_key or not config.exchange.api_secret:
            logger.error("Live trading requires API keys. See .env.example")
            sys.exit(1)
        valid = await validate_connection(config)
        if not valid:
            sys.exit(1)
        if not config.exchange.sandbox:
            logger.warning("=" * 60)
            logger.warning("  MAINNET — REAL MONEY. Type 'YES' to confirm.")
            logger.warning("=" * 60)
            try:
                if input("  > ").strip() != "YES":
                    sys.exit(0)
            except EOFError:
                sys.exit(0)

    bot = ScalpingBot(config)

    try:
        await bot.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        if bot._running:
            await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForce quit.")
        sys.exit(0)
    except SystemExit:
        raise
