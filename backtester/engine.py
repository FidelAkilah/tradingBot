"""
Backtesting Engine — Event-driven replay of historical candle data.

Iterates candle-by-candle through historical data, running the full signal
pipeline (same code as live) and simulating trade execution with realistic
slippage and partial take-profit.
"""

import copy
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path so we can import bot modules
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from config import BotConfig, CONFIG
from candle_analyzer import CandleAnalyzer, SwingSignal, TrendDirection, CandleSignal
from market_regime import MarketRegimeClassifier, RegimeResult
from session_filter import SessionFilter
from position_sizer import PositionSizer, SizingResult
from risk_manager import RiskManager, TradeRecord
from daily_target import DailyTargetTracker, DailyTargetContext, TradingMode
from backtester.data_manager import DataManager, TIMEFRAME_MS
from backtester.metrics import compute_metrics, PerformanceMetrics

logger = logging.getLogger(__name__)

# Slippage defaults (as fraction of price)
DEFAULT_SLIPPAGE = {
    "BTC/USDT": 0.0002,   # 0.02%
    "ETH/USDT": 0.0003,   # 0.03%
}
DEFAULT_ALT_SLIPPAGE = 0.0005  # 0.05% for altcoins


@dataclass
class SimulatedTrade:
    """A trade tracked by the backtesting engine."""
    trade_id: str
    symbol: str
    side: str                     # "BUY" or "SELL"
    entry_price: float
    entry_time: float
    amount: float                 # Position size in base currency
    notional_usd: float          # Position value in USD
    leverage: int

    # TP/SL levels
    target_price: float           # Final TP display price
    stop_price: float             # Current stop loss
    original_stop: float          # Initial stop loss (never widen beyond)
    tp_levels: List[dict] = field(default_factory=list)  # [{level, target, size_pct, amount, hit, hit_time, pnl}]

    # ATR for trailing
    atr_at_entry: float = 0.0
    trailing_atr_mult: float = 2.0

    # Signal metadata
    confidence: float = 0.0
    regime: str = "ranging"
    session: str = "dead_zone"
    post_fee_rr: float = 0.0
    adx: float = 0.0
    swing_trend: str = "neutral"

    # Sizing details
    kelly_pct: float = 0.0
    drawdown_mult: float = 1.0

    # State
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl_usd: float = 0.0
    fee_cost_usd: float = 0.0
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    duration_s: float = 0.0
    max_favorable_price: float = 0.0
    partial_realized_pnl: float = 0.0
    partial_fees: float = 0.0

    # TP hit tracking
    tp1_hit: bool = False
    tp1_price: Optional[float] = None
    tp1_pnl: Optional[float] = None
    tp2_hit: bool = False
    tp2_price: Optional[float] = None
    tp2_pnl: Optional[float] = None
    tp3_hit: bool = False
    tp3_price: Optional[float] = None
    tp3_pnl: Optional[float] = None

    # Re-entry
    is_reentry: bool = False
    reentry_count: int = 0

    # Volume/funding/OI (from signal)
    obv_trend: str = "neutral"
    obv_divergence: bool = False
    volume_pressure: str = "neutral"
    buy_volume_ratio: float = 0.5
    funding_rate: float = 0.0
    funding_extreme: bool = False
    oi_change_pct: float = 0.0
    oi_conviction: str = "neutral"

    original_amount: float = 0.0

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                d[k] = v
            else:
                d[k] = v
        return d


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    run_id: str
    start_time: float
    end_time: float
    config_snapshot: dict
    param_overrides: dict

    trades: List[dict] = field(default_factory=list)
    equity_curve: List[Tuple[float, float]] = field(default_factory=list)
    signals_log: List[dict] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None

    symbols: List[str] = field(default_factory=list)
    starting_equity: float = 0.0


class BacktestEngine:
    """
    Event-driven backtesting engine that replays historical candles
    through the full signal pipeline.
    """

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        data_manager: Optional[DataManager] = None,
        slippage_map: Optional[Dict[str, float]] = None,
    ):
        self.config = config or copy.deepcopy(CONFIG)
        self.dm = data_manager or DataManager()
        self.slippage_map = slippage_map or DEFAULT_SLIPPAGE
        self.default_slippage = DEFAULT_ALT_SLIPPAGE

        # These get initialized in run()
        self._analyzer: Optional[CandleAnalyzer] = None
        self._sizer: Optional[PositionSizer] = None
        self._risk: Optional[RiskManager] = None
        self._daily_target: Optional[DailyTargetTracker] = None

        # Trade state
        self._trades: List[SimulatedTrade] = []
        self._open_trades: Dict[str, SimulatedTrade] = {}
        self._equity: float = 0.0
        self._peak_equity: float = 0.0
        self._equity_curve: List[Tuple[float, float]] = []
        self._signals_log: List[dict] = []
        self._last_close_time: Dict[str, float] = {}
        self._trade_counter: int = 0

        # Peak/trough tracking for Chandelier exit
        self._peak_prices: Dict[str, float] = {}
        self._trough_prices: Dict[str, float] = {}

        # Dynamic SL state
        self._flat_sl_applied: Dict[str, bool] = {}
        self._momentum_sl_applied: Dict[str, bool] = {}

    def apply_param_overrides(self, overrides: Dict[str, Any]):
        """Apply parameter overrides to the config for sweep runs."""
        for key, value in overrides.items():
            parts = key.split(".")
            obj = self.config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        primary_tf: str = "1h",
        param_overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        Run a backtest over the specified date range.

        Args:
            symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
            start_date: Start date string "YYYY-MM-DD"
            end_date: End date string "YYYY-MM-DD"
            primary_tf: Primary timeframe for stepping (default "1h")
            param_overrides: Optional dict of config overrides for param sweeps
                e.g., {"candle.adx_trending_threshold": 30, "trading.min_post_fee_rr": 2.0}

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        if param_overrides:
            self.apply_param_overrides(param_overrides)

        # Initialize components
        tc = self.config.trading
        idr_capital = tc.starting_capital_idr
        usd_equity = idr_capital / 16_300.0

        self._equity = usd_equity
        self._peak_equity = usd_equity
        self._trades = []
        self._open_trades = {}
        self._equity_curve = []
        self._signals_log = []
        self._last_close_time = {}
        self._trade_counter = 0
        self._peak_prices = {}
        self._trough_prices = {}
        self._flat_sl_applied = {}
        self._momentum_sl_applied = {}

        self._analyzer = CandleAnalyzer(self.config)
        self._sizer = PositionSizer(self.config)
        self._risk = RiskManager(initial_equity=usd_equity, config=self.config)
        self._daily_target = DailyTargetTracker(usd_equity, self.config)

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)

        # Get primary timeline (union of all symbols' 1h candle timestamps)
        all_timestamps = set()
        for symbol in symbols:
            ts_list = self.dm.get_primary_timeline(symbol, primary_tf, start_ts, end_ts)
            all_timestamps.update(ts_list)

        timeline = sorted(all_timestamps)

        if not timeline:
            logger.warning("No candle data found for the specified range.")
            return BacktestResult(
                run_id=run_id,
                start_time=start_ts / 1000,
                end_time=end_ts / 1000,
                config_snapshot=self._config_to_dict(),
                param_overrides=param_overrides or {},
                symbols=symbols,
                starting_equity=usd_equity,
            )

        logger.info(
            f"Backtest {run_id}: {len(symbols)} symbols, "
            f"{len(timeline)} candles, "
            f"{start_date} to {end_date}"
        )

        # Record initial equity
        self._equity_curve.append((timeline[0] / 1000, self._equity))

        # Track current day for daily resets
        current_day = ""

        # ── Main loop: iterate candle by candle ─────────────────
        for i, ts in enumerate(timeline):
            ts_s = ts / 1000.0  # Convert ms to seconds

            # Daily reset check
            dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
            day_str = dt.strftime("%Y-%m-%d")
            if day_str != current_day:
                if current_day:
                    self._daily_reset(ts_s)
                current_day = day_str

            # Process each symbol at this timestamp
            for symbol in symbols:
                self._process_candle(symbol, ts, ts_s)

            # Monitor open trades with current prices
            self._monitor_open_trades(ts, ts_s)

            # Record equity
            self._equity_curve.append((ts_s, self._equity))

            # Progress logging
            if (i + 1) % 500 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(timeline)} candles, "
                    f"{len(self._trades)} trades, equity=${self._equity:.2f}"
                )

        # Close any remaining open trades at last price
        self._close_all_open(timeline[-1] / 1000.0)

        # Build result
        trade_dicts = [t.to_dict() for t in self._trades]
        metrics = compute_metrics(
            trade_dicts, self._equity_curve, usd_equity
        )

        result = BacktestResult(
            run_id=run_id,
            start_time=start_ts / 1000,
            end_time=end_ts / 1000,
            config_snapshot=self._config_to_dict(),
            param_overrides=param_overrides or {},
            trades=trade_dicts,
            equity_curve=self._equity_curve,
            signals_log=self._signals_log,
            metrics=metrics,
            symbols=symbols,
            starting_equity=usd_equity,
        )

        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"PnL=${metrics.total_pnl:.2f}, "
            f"WR={metrics.win_rate:.1%}, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, "
            f"MaxDD={metrics.max_drawdown_pct:.1f}%"
        )

        return result

    # ── Core processing ─────────────────────────────────────────

    def _process_candle(self, symbol: str, ts_ms: int, ts_s: float):
        """Process a single candle timestamp for a symbol."""
        # Build multi-timeframe candle dict for the analyzer
        timeframes = self.config.candle.timeframes  # ["1h", "4h"]
        lookback = self.config.candle.lookback_candles

        candles = self.dm.get_candles_as_dict(symbol, timeframes, ts_ms, lookback)

        if not candles:
            return

        # Also fetch daily candles for the daily gate
        daily_candles = self.dm.get_candles(
            symbol, "1d",
            end_ts=ts_ms,
        )
        if daily_candles and len(daily_candles) >= 15:
            # Inject into analyzer's daily cache
            self._analyzer._daily_cache[symbol] = daily_candles
            self._analyzer._daily_fetch_ts[symbol] = ts_s
            self._analyzer._daily_signal_cache[symbol] = (
                self._analyzer._analyze_timeframe("1d", daily_candles, ts_s)
            )

        # Fetch 15m candles for entry timing
        candles_15m = self.dm.get_candles(
            symbol, "15m",
            end_ts=ts_ms,
        )
        if candles_15m and len(candles_15m) >= 15:
            self._analyzer._15m_cache[symbol] = candles_15m
            self._analyzer._15m_fetch_ts[symbol] = ts_s

        # Run analysis
        swing = self._analyzer.analyze(symbol, candles, ts_s)

        if not swing.suggested_side:
            return

        # Apply all filters from the entry gate sequence
        if not self._passes_entry_gates(swing, symbol, ts_s):
            return

        # Check if we already have a position on this symbol
        if symbol in self._open_trades:
            return

        # Cooldown check
        last_close = self._last_close_time.get(symbol, 0)
        if ts_s - last_close < 300.0:  # 5 min cooldown
            return

        # Risk manager check
        can_trade, reason = self._risk.can_trade()
        if not can_trade:
            return

        # Position sizing
        dt_ctx = self._daily_target.get_daily_target_context()
        sizing = self._sizer.calculate(
            equity=self._equity,
            peak_equity=self._peak_equity,
            confidence=swing.confidence,
            symbol=symbol,
            regime_mult=swing.regime_size_mult,
            session_mult=swing.session_size_mult,
            daily_target_ctx=dt_ctx,
            current_time=ts_s,
        )

        if sizing.is_halted:
            return

        # Open the trade
        self._open_trade(symbol, swing, sizing, ts_s)

    def _passes_entry_gates(self, swing: SwingSignal, symbol: str, ts_s: float) -> bool:
        """Check all entry gate conditions. Returns True if trade should proceed."""
        cc = self.config.candle
        tc = self.config.trading

        # 1. ADX filter
        if swing.adx_blocked:
            return False

        # 2. Regime filter
        if swing.regime_blocked:
            return False

        # 3. Session filter
        if swing.session_blocked:
            return False

        # 4. RSI extreme check (both TFs)
        if swing.suggested_side == "BUY" and swing.is_overbought:
            return False
        if swing.suggested_side == "SELL" and swing.is_oversold:
            return False

        # 5. Daily trend gate
        if swing.daily_blocked:
            return False

        # 7. RSI divergence hard block
        if swing.divergence_blocked:
            return False

        # 13. Minimum confidence
        if swing.confidence < 0.55:
            return False

        # 14. Post-fee R:R
        if swing.post_fee_rr < tc.min_post_fee_rr:
            return False

        # 15. S/R TP-block
        if swing.sr_tp_blocked:
            return False

        # Daily target mode check
        mode = self._daily_target.state.mode
        if mode == TradingMode.HALTED:
            return False
        if mode == TradingMode.PROTECTING and swing.confidence < 0.70:
            return False

        # Position limit check
        max_pos = 2 if mode != TradingMode.AGGRESSIVE else 3
        if len(self._open_trades) >= max_pos:
            return False

        return True

    def _open_trade(
        self,
        symbol: str,
        swing: SwingSignal,
        sizing: SizingResult,
        ts_s: float,
    ):
        """Open a simulated trade."""
        self._trade_counter += 1
        side = swing.suggested_side

        # Apply slippage
        slippage = self.slippage_map.get(symbol, self.default_slippage)
        close_price = swing.signals.get("1h", swing.signals.get("4h"))
        if close_price is None:
            return
        entry_price = close_price.last_close

        if side == "BUY":
            entry_price *= (1 + slippage)  # Buy slightly higher
        else:
            entry_price *= (1 - slippage)  # Sell slightly lower

        notional = sizing.notional_usd
        amount = notional / entry_price

        # TP/SL from swing signal
        tp_pct = swing.atr_tp_pct if swing.atr_tp_pct > 0 else 2.0
        sl_pct = swing.atr_sl_pct if swing.atr_sl_pct > 0 else 1.0

        if side == "BUY":
            target_price = entry_price * (1 + tp_pct / 100)
            stop_price = entry_price * (1 - sl_pct / 100)
        else:
            target_price = entry_price * (1 - tp_pct / 100)
            stop_price = entry_price * (1 + sl_pct / 100)

        # Build partial TP levels
        atr = 0.0
        for tf in ("4h", "1h"):
            sig = swing.signals.get(tf)
            if sig and sig.atr > 0:
                atr = sig.atr
                break

        tp_levels = self._build_tp_levels(entry_price, atr, side, amount, swing.confidence)

        trade = SimulatedTrade(
            trade_id=f"BT-{self._trade_counter:05d}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=ts_s,
            amount=amount,
            original_amount=amount,
            notional_usd=notional,
            leverage=sizing.leverage,
            target_price=target_price,
            stop_price=stop_price,
            original_stop=stop_price,
            tp_levels=tp_levels,
            atr_at_entry=atr,
            trailing_atr_mult=self.config.exit.chandelier_atr_mult,
            confidence=swing.confidence,
            regime=swing.regime,
            session=swing.session,
            post_fee_rr=swing.post_fee_rr,
            adx=swing.adx_4h if swing.adx_4h > 0 else swing.adx_1h,
            swing_trend=swing.primary_trend.value,
            kelly_pct=sizing.kelly_pct,
            drawdown_mult=sizing.drawdown_mult,
            max_favorable_price=entry_price,
            obv_trend=swing.obv_trend,
            obv_divergence=swing.obv_divergence,
            volume_pressure=swing.volume_pressure,
            buy_volume_ratio=0.5,
            funding_rate=swing.funding_rate_value,
            funding_extreme=swing.funding_extreme,
            oi_change_pct=swing.oi_change_pct,
            oi_conviction=swing.oi_conviction,
        )

        self._open_trades[symbol] = trade
        self._peak_prices[symbol] = entry_price
        self._trough_prices[symbol] = entry_price
        self._flat_sl_applied[symbol] = False
        self._momentum_sl_applied[symbol] = False

        logger.debug(
            f"[BT] OPEN {side} {symbol} @ {entry_price:.4f}, "
            f"conf={swing.confidence:.2f}, lev={sizing.leverage}x"
        )

    def _build_tp_levels(
        self,
        entry_price: float,
        atr: float,
        side: str,
        amount: float,
        confidence: float,
    ) -> List[dict]:
        """Build partial TP levels matching shadow_trader logic."""
        ec = self.config.exit
        mults = [ec.tp1_atr_mult, ec.tp2_atr_mult, ec.tp3_atr_mult]
        sizes = [ec.tp1_size_pct, ec.tp2_size_pct, ec.tp3_size_pct]

        # Daily-target-aware adjustments
        ctx = self._daily_target.get_daily_target_context() if self._daily_target else None
        if ctx:
            if ctx.pct_achieved >= getattr(ec, 'target_near_threshold', 80.0):
                compress = getattr(ec, 'target_near_compress', 0.70)
                mults = [m * compress for m in mults]
            elif (ctx.pct_achieved < getattr(ec, 'target_behind_threshold', 30.0)
                  and ctx.day_elapsed_pct > getattr(ec, 'target_behind_utc_hour', 18) / 24.0
                  and confidence >= getattr(ec, 'target_behind_min_conf', 0.75)):
                expand = getattr(ec, 'target_behind_expand', 1.2)
                mults[1] *= expand
                mults[2] *= expand

        levels = []
        for i, (mult, size_pct) in enumerate(zip(mults, sizes)):
            level_num = i + 1
            if atr > 0:
                if side == "BUY":
                    tp = entry_price + (atr * mult) if level_num <= 2 else 0.0
                else:
                    tp = entry_price - (atr * mult) if level_num <= 2 else 0.0
            else:
                tp = 0.0

            levels.append({
                "level": level_num,
                "atr_mult": mult,
                "target_price": tp,
                "size_pct": size_pct,
                "amount": amount * size_pct,
                "hit": False,
                "hit_time": None,
                "hit_price": None,
                "pnl": None,
            })

        return levels

    # ── Trade Monitoring ────────────────────────────────────────

    def _monitor_open_trades(self, ts_ms: int, ts_s: float):
        """Monitor all open trades against current candle prices."""
        symbols_to_close = []

        for symbol, trade in self._open_trades.items():
            # Get current price from the latest 1h candle
            candles_1h = self.dm.get_candles(symbol, "1h", end_ts=ts_ms)
            if not candles_1h:
                continue

            last_candle = candles_1h[-1]
            # Candle format: [ts, open, high, low, close, volume]
            candle_high = last_candle[2]
            candle_low = last_candle[3]
            current_price = last_candle[4]

            # Update peak/trough tracking
            if trade.side == "BUY":
                self._peak_prices[symbol] = max(
                    self._peak_prices.get(symbol, trade.entry_price), candle_high
                )
                trade.max_favorable_price = self._peak_prices[symbol]
            else:
                self._trough_prices[symbol] = min(
                    self._trough_prices.get(symbol, trade.entry_price), candle_low
                )
                trade.max_favorable_price = self._trough_prices[symbol]

            # Check partial TPs using candle high/low
            self._check_partial_tp(trade, candle_high, candle_low, ts_s)

            # Dynamic SL adjustment (first 15-30 min simulation via candle count)
            self._apply_dynamic_sl(trade, current_price, ts_s)

            # Check Chandelier trailing stop
            if self._check_chandelier_stop(trade, current_price, candle_high, candle_low):
                exit_price = self._apply_slippage(symbol, current_price, trade.side, exit=True)
                self._close_trade(trade, exit_price, "chandelier_trailing", ts_s)
                symbols_to_close.append(symbol)
                continue

            # Check stop loss
            if trade.side == "BUY" and candle_low <= trade.stop_price:
                exit_price = self._apply_slippage(symbol, trade.stop_price, trade.side, exit=True)
                self._close_trade(trade, exit_price, "stop_loss", ts_s)
                symbols_to_close.append(symbol)
                continue
            elif trade.side == "SELL" and candle_high >= trade.stop_price:
                exit_price = self._apply_slippage(symbol, trade.stop_price, trade.side, exit=True)
                self._close_trade(trade, exit_price, "stop_loss", ts_s)
                symbols_to_close.append(symbol)
                continue

            # Max hold time (8 hours)
            if ts_s - trade.entry_time > 8 * 3600:
                exit_price = self._apply_slippage(symbol, current_price, trade.side, exit=True)
                self._close_trade(trade, exit_price, "max_hold", ts_s)
                symbols_to_close.append(symbol)
                continue

        for s in symbols_to_close:
            del self._open_trades[s]

    def _check_partial_tp(
        self,
        trade: SimulatedTrade,
        candle_high: float,
        candle_low: float,
        ts_s: float,
    ):
        """Check and execute partial take-profit levels."""
        ec = self.config.exit
        fee_rate = self.config.trading.fee_rate / 100.0

        for tp in trade.tp_levels:
            if tp["hit"] or tp["target_price"] == 0.0:
                continue

            hit = False
            if trade.side == "BUY" and candle_high >= tp["target_price"]:
                hit = True
            elif trade.side == "SELL" and candle_low <= tp["target_price"]:
                hit = True

            if not hit:
                continue

            # Execute partial TP
            fill_price = tp["target_price"]
            fill_price = self._apply_slippage(trade.symbol, fill_price, trade.side, exit=True)
            tp_amount = tp["amount"]

            # Calculate PnL for this tranche
            if trade.side == "BUY":
                gross_pnl = (fill_price - trade.entry_price) * tp_amount
            else:
                gross_pnl = (trade.entry_price - fill_price) * tp_amount

            fees = fill_price * tp_amount * fee_rate * 2  # round-trip
            net_pnl = gross_pnl - fees

            tp["hit"] = True
            tp["hit_time"] = ts_s
            tp["hit_price"] = fill_price
            tp["pnl"] = net_pnl

            trade.partial_realized_pnl += net_pnl
            trade.partial_fees += fees
            trade.amount -= tp_amount

            # Update trade TP hit flags
            level = tp["level"]
            if level == 1:
                trade.tp1_hit = True
                trade.tp1_price = fill_price
                trade.tp1_pnl = net_pnl
                # Move SL to breakeven
                if ec.sl_to_breakeven_after_tp1:
                    # Breakeven = entry + round-trip fees
                    fee_offset = trade.entry_price * fee_rate * 2
                    if trade.side == "BUY":
                        trade.stop_price = max(trade.stop_price, trade.entry_price + fee_offset)
                    else:
                        trade.stop_price = min(trade.stop_price, trade.entry_price - fee_offset)
                # Tighten trailing
                trade.trailing_atr_mult = ec.chandelier_after_tp1_mult
            elif level == 2:
                trade.tp2_hit = True
                trade.tp2_price = fill_price
                trade.tp2_pnl = net_pnl
                # Move SL to TP1 price
                if ec.sl_to_tp1_after_tp2 and trade.tp1_price:
                    if trade.side == "BUY":
                        trade.stop_price = max(trade.stop_price, trade.tp1_price)
                    else:
                        trade.stop_price = min(trade.stop_price, trade.tp1_price)
                # Tighten trailing more
                trade.trailing_atr_mult = ec.chandelier_after_tp2_mult
            elif level == 3:
                trade.tp3_hit = True
                trade.tp3_price = fill_price
                trade.tp3_pnl = net_pnl

            # Record partial to risk manager
            self._risk.record_trade(TradeRecord(
                symbol=trade.symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=fill_price,
                amount=tp_amount,
                pnl_usd=net_pnl,
                reason=f"tp{level}",
                timestamp=ts_s,
            ))
            self._daily_target.record_trade(net_pnl)
            self._equity += net_pnl
            if self._equity > self._peak_equity:
                self._peak_equity = self._equity

    def _check_chandelier_stop(
        self,
        trade: SimulatedTrade,
        current_price: float,
        candle_high: float,
        candle_low: float,
    ) -> bool:
        """Check Chandelier Exit (ATR trailing stop). Returns True if triggered."""
        if not self.config.exit.chandelier_enabled:
            return False
        if trade.atr_at_entry <= 0:
            return False

        atr = trade.atr_at_entry
        mult = trade.trailing_atr_mult
        symbol = trade.symbol

        if trade.side == "BUY":
            peak = self._peak_prices.get(symbol, trade.entry_price)
            trail_stop = peak - (mult * atr)

            # After TP1, never trail below breakeven
            if trade.tp1_hit:
                trail_stop = max(trail_stop, trade.entry_price)

            # Only activate after price moves 30% toward TP1, or after TP1 hit
            tp1_target = trade.tp_levels[0]["target_price"] if trade.tp_levels else 0
            if tp1_target > 0 and not trade.tp1_hit:
                move_to_tp1 = tp1_target - trade.entry_price
                actual_move = peak - trade.entry_price
                if move_to_tp1 > 0 and actual_move / move_to_tp1 < 0.3:
                    return False

            if candle_low <= trail_stop:
                return True
        else:
            trough = self._trough_prices.get(symbol, trade.entry_price)
            trail_stop = trough + (mult * atr)

            if trade.tp1_hit:
                trail_stop = min(trail_stop, trade.entry_price)

            tp1_target = trade.tp_levels[0]["target_price"] if trade.tp_levels else 0
            if tp1_target > 0 and not trade.tp1_hit:
                move_to_tp1 = trade.entry_price - tp1_target
                actual_move = trade.entry_price - trough
                if move_to_tp1 > 0 and actual_move / move_to_tp1 < 0.3:
                    return False

            if candle_high >= trail_stop:
                return True

        return False

    def _apply_dynamic_sl(self, trade: SimulatedTrade, current_price: float, ts_s: float):
        """Apply dynamic SL adjustments based on early price action."""
        if not self.config.exit.dynamic_sl_enabled:
            return

        elapsed = ts_s - trade.entry_time
        atr = trade.atr_at_entry
        if atr <= 0:
            return

        symbol = trade.symbol

        # Momentum check at ~15 min (simulate with 1 candle = ~1h, so check at first candle)
        if elapsed >= 900 and not self._momentum_sl_applied.get(symbol, False):
            if trade.side == "BUY":
                move = current_price - trade.entry_price
            else:
                move = trade.entry_price - current_price

            if move >= 0.5 * atr:
                # Momentum confirmed — tighten SL
                new_sl_dist = 0.7 * atr
                if trade.side == "BUY":
                    new_sl = trade.entry_price - new_sl_dist
                    trade.stop_price = max(trade.stop_price, new_sl)
                else:
                    new_sl = trade.entry_price + new_sl_dist
                    trade.stop_price = min(trade.stop_price, new_sl)
                self._momentum_sl_applied[symbol] = True

        # Flat check at ~30 min
        if elapsed >= 1800 and not self._flat_sl_applied.get(symbol, False):
            if trade.side == "BUY":
                move = abs(current_price - trade.entry_price)
            else:
                move = abs(trade.entry_price - current_price)

            if move < 0.2 * atr:
                # Flat — reduce exposure
                new_sl_dist = 0.5 * atr
                if trade.side == "BUY":
                    new_sl = trade.entry_price - new_sl_dist
                    trade.stop_price = max(trade.stop_price, new_sl)
                else:
                    new_sl = trade.entry_price + new_sl_dist
                    trade.stop_price = min(trade.stop_price, new_sl)
                self._flat_sl_applied[symbol] = True

    def _close_trade(
        self,
        trade: SimulatedTrade,
        exit_price: float,
        reason: str,
        ts_s: float,
    ):
        """Close a trade and update accounting."""
        fee_rate = self.config.trading.fee_rate / 100.0

        # Remaining amount after partial TPs
        remaining = trade.amount
        if remaining <= 0:
            remaining = 0.001  # Avoid division by zero

        if trade.side == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * remaining
        else:
            gross_pnl = (trade.entry_price - exit_price) * remaining

        fees = exit_price * remaining * fee_rate * 2
        net_pnl = gross_pnl - fees

        # Total PnL = partial realized + remaining
        total_pnl = trade.partial_realized_pnl + net_pnl
        total_fees = trade.partial_fees + fees
        total_gross = gross_pnl + sum(
            tp["pnl"] + (tp.get("_fees", 0))
            for tp in trade.tp_levels if tp["hit"]
        ) if False else (total_pnl + total_fees)

        trade.exit_price = exit_price
        trade.exit_time = ts_s
        trade.exit_reason = reason
        trade.pnl_usd = total_pnl
        trade.gross_pnl_usd = total_gross
        trade.fee_cost_usd = total_fees
        trade.pnl_pct = (total_pnl / trade.notional_usd * 100) if trade.notional_usd > 0 else 0
        trade.duration_s = ts_s - trade.entry_time
        trade.is_open = False

        # Update equity
        self._equity += net_pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Record to risk manager and daily target
        self._risk.record_trade(TradeRecord(
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            amount=remaining,
            pnl_usd=net_pnl,
            reason=reason,
            timestamp=ts_s,
        ))
        self._daily_target.record_trade(net_pnl)

        self._trades.append(trade)
        self._last_close_time[trade.symbol] = ts_s

        # Cleanup
        self._peak_prices.pop(trade.symbol, None)
        self._trough_prices.pop(trade.symbol, None)
        self._flat_sl_applied.pop(trade.symbol, None)
        self._momentum_sl_applied.pop(trade.symbol, None)

        logger.debug(
            f"[BT] CLOSE {trade.side} {trade.symbol} @ {exit_price:.4f}, "
            f"reason={reason}, pnl=${total_pnl:.4f}"
        )

    def _close_all_open(self, ts_s: float):
        """Close all remaining open trades at current price."""
        for symbol in list(self._open_trades.keys()):
            trade = self._open_trades[symbol]
            # Get last available price
            candles = self.dm.get_candles(symbol, "1h")
            if candles:
                exit_price = candles[-1][4]  # last close
            else:
                exit_price = trade.entry_price

            exit_price = self._apply_slippage(symbol, exit_price, trade.side, exit=True)
            self._close_trade(trade, exit_price, "backtest_end", ts_s)

        self._open_trades.clear()

    # ── Helpers ─────────────────────────────────────────────────

    def _apply_slippage(
        self, symbol: str, price: float, side: str, exit: bool = False
    ) -> float:
        """Apply realistic slippage to a fill price."""
        slip = self.slippage_map.get(symbol, self.default_slippage)
        if exit:
            # Exiting: buy-side exits sell lower, sell-side exits buy higher
            if side == "BUY":
                return price * (1 - slip)
            else:
                return price * (1 + slip)
        else:
            # Entering
            if side == "BUY":
                return price * (1 + slip)
            else:
                return price * (1 - slip)

    def _daily_reset(self, ts_s: float):
        """Handle daily boundary reset."""
        self._risk.reset_daily_counters()
        # Update daily target equity
        self._daily_target.update_equity(self._equity)

    def _config_to_dict(self) -> dict:
        """Serialize config to dict for result storage."""
        try:
            from dataclasses import asdict
            return asdict(self.config)
        except Exception:
            return {"error": "Could not serialize config"}
