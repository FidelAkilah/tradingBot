"""
Shadow Trader — Simulation mode with full trade logging.

Logs every signal, potential trade, and hypothetical P&L to JSONL files
without placing any real orders. Perfect for strategy validation.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from config import BotConfig, CONFIG
from liquidity_analyzer import AnalysisResult, LiquidityWall, WallSide
from position_sizer import PositionSizer, SizingResult
from risk_manager import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    """A simulated trade record."""
    trade_id: int
    symbol: str
    side: str                    # "BUY" or "SELL"
    entry_price: float
    target_price: float          # Take-profit target
    stop_price: float            # Stop-loss level
    amount: float
    usd_value: float
    wall_price: float
    wall_usd_value: float
    wall_multiplier: float
    wall_confidence: float
    composite_score: float
    imbalance_ratio: float
    vwap: float
    vwap_deviation_pct: float
    momentum_aggressor_ratio: float
    spread_pct: float
    vpin: float = 0.0
    vpin_ema: float = 0.0
    vpin_regime: str = "unknown"
    vpin_directional_bias: float = 0.0
    # Swing / candle fields
    swing_trend: str = "neutral"
    swing_confidence: float = 0.0
    swing_trend_aligned: bool = False
    rsi_1h: float = 50.0
    rsi_4h: float = 50.0
    atr_tp_pct: float = 0.0         # ATR-derived TP % (fee-adjusted)
    atr_sl_pct: float = 0.0         # ATR-derived SL % (fee-adjusted)
    raw_tp_pct: float = 0.0         # Pre-fee TP %
    raw_sl_pct: float = 0.0         # Pre-fee SL %
    fee_cost_pct: float = 0.0       # Round-trip fee as % of notional
    post_fee_rr: float = 0.0        # Risk-reward ratio after fees
    adx: float = 0.0                # ADX at entry (primary timeframe)
    # Market regime
    regime: str = "ranging"
    regime_size_mult: float = 1.0
    regime_is_breakout: bool = False
    # Session filter
    session: str = "dead_zone"
    session_size_mult: float = 1.0
    # Position sizing rationale
    leverage: int = 10
    kelly_pct: float = 0.10
    confidence_mult: float = 1.0
    drawdown_mult: float = 1.0
    drawdown_pct: float = 0.0
    consec_loss_mult: float = 1.0
    consecutive_losses: int = 0
    entry_time: float = 0.0
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    gross_pnl_usd: Optional[float] = None   # P&L before fees
    fee_cost_usd: Optional[float] = None     # Actual fee cost in USD
    pnl_usd: Optional[float] = None          # Net P&L after fees
    pnl_pct: Optional[float] = None          # Net P&L % (of margin)
    duration_s: Optional[float] = None
    is_open: bool = True


class ShadowTrader:
    """
    Tracks hypothetical trades based on analysis signals.

    Simulates order fills with configurable latency, tracks positions
    against live price feeds, and logs everything to JSONL for analysis.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.shadow

        self._trade_counter = 0
        self.open_trades: Dict[str, ShadowTrade] = {}   # key: symbol
        self.closed_trades: List[ShadowTrade] = []
        self.signal_log: List[dict] = []

        # Position sizer for dynamic sizing
        self.sizer = PositionSizer(config)

        # Simulated equity tracking for sizer
        self._equity = config.trading.starting_capital_idr / 16_300.0  # ~$61
        self._peak_equity = self._equity

        # Per-symbol cooldown tracking: symbol -> time of last close
        self._last_close_time: Dict[str, float] = {}
        self._symbol_cooldown_s = 300.0  # 5 min cooldown — swing bot, not scalper

        # Latest known price per symbol (for live unrealized P&L)
        self.last_prices: Dict[str, float] = {}

        # Stats
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

        # Trailing stop tracking (peak price for longs, trough for shorts)
        self._peak_prices: Dict[str, float] = {}
        self._trough_prices: Dict[str, float] = {}

        # Log file
        self._log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.sc.log_file
        )
        self._signal_log_path = self._log_path.replace(".jsonl", "_signals.jsonl")

        logger.info(f"Shadow trader initialized. Log: {self._log_path}")

    # ─────────────────────────────────────────
    # SIGNAL PROCESSING
    # ─────────────────────────────────────────

    def process_signal(self, analysis: AnalysisResult) -> Optional[ShadowTrade]:
        """
        Process an analysis result and open a shadow trade if appropriate.

        Returns the ShadowTrade if one was opened, None otherwise.
        """
        # Log every signal regardless
        self._log_signal(analysis)

        if not analysis.trade_suggestion:
            return None

        symbol = analysis.symbol

        # Don't stack trades on same symbol
        if symbol in self.open_trades:
            return None

        # Per-symbol cooldown — prevent spam re-entry after close
        now = time.time()
        last_close = self._last_close_time.get(symbol, 0)
        if now - last_close < self._symbol_cooldown_s:
            return None

        entry_time = now

        if analysis.trade_suggestion == "BUY":
            return self._open_buy(analysis, entry_time)
        elif analysis.trade_suggestion == "SELL":
            return self._open_sell(analysis, entry_time)

        return None

    def _open_buy(self, analysis: AnalysisResult, entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated long position."""
        walls = [w for w in analysis.bid_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            if not (hasattr(analysis, 'swing') and analysis.swing and analysis.swing.confidence >= 0.55):
                return None

        wall = walls[0] if walls else None
        tc = self.config.trading
        mid = analysis.mid_price
        entry_price = mid

        # Extract swing signal info
        swing = getattr(analysis, 'swing', None)
        swing_conf = swing.confidence if swing else 0.55
        regime_size = getattr(analysis, 'regime_size_mult', 1.0)
        session_size = getattr(analysis, 'session_size_mult', 1.0)

        # Dynamic position sizing via PositionSizer
        sizing = self.sizer.calculate(
            equity=self._equity,
            peak_equity=self._peak_equity,
            confidence=swing_conf,
            symbol=analysis.symbol,
            regime_mult=regime_size,
            session_mult=session_size,
            current_time=entry_time,
        )

        if sizing.is_halted:
            logger.info(f"[SHADOW] BUY blocked: {sizing.halt_reason}")
            return None

        amount = sizing.notional_usd / entry_price

        # Fee-adjusted TP/SL from analysis (already includes fee compensation)
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else tc.stop_loss_pct

        swing_trend = swing.primary_trend.value if swing else "neutral"
        swing_aligned = swing.trend_aligned if swing else False
        rsi_1h = swing.rsi_1h if swing else 50.0
        rsi_4h = swing.rsi_4h if swing else 50.0

        trade = ShadowTrade(
            trade_id=self._next_id(),
            symbol=analysis.symbol,
            side="BUY",
            entry_price=entry_price,
            target_price=entry_price * (1.0 + tp_pct / 100.0),
            stop_price=entry_price * (1.0 - sl_pct / 100.0),
            amount=amount,
            usd_value=sizing.position_usd,
            wall_price=wall.price if wall else 0.0,
            wall_usd_value=wall.usd_value if wall else 0.0,
            wall_multiplier=wall.multiplier if wall else 0.0,
            wall_confidence=wall.confidence if wall else 0.0,
            composite_score=analysis.composite_score,
            imbalance_ratio=analysis.imbalance.ratio,
            vwap=analysis.vwap.vwap,
            vwap_deviation_pct=analysis.vwap.deviation_pct,
            momentum_aggressor_ratio=analysis.momentum.aggressor_ratio,
            spread_pct=analysis.spread_pct,
            vpin=analysis.vpin.vpin,
            vpin_ema=analysis.vpin.vpin_ema,
            vpin_regime=analysis.vpin.regime.value,
            vpin_directional_bias=analysis.vpin.directional_bias,
            swing_trend=swing_trend,
            swing_confidence=swing_conf,
            swing_trend_aligned=swing_aligned,
            rsi_1h=rsi_1h,
            rsi_4h=rsi_4h,
            atr_tp_pct=tp_pct,
            atr_sl_pct=sl_pct,
            raw_tp_pct=getattr(analysis, 'raw_tp_pct', 0.0),
            raw_sl_pct=getattr(analysis, 'raw_sl_pct', 0.0),
            fee_cost_pct=getattr(analysis, 'fee_cost_pct', 0.0),
            post_fee_rr=getattr(analysis, 'post_fee_rr', 0.0),
            adx=getattr(analysis, 'adx', 0.0),
            regime=getattr(analysis, 'regime', 'ranging'),
            regime_size_mult=regime_size,
            regime_is_breakout=getattr(analysis, 'regime_is_breakout', False),
            session=getattr(analysis, 'session', 'dead_zone'),
            session_size_mult=session_size,
            leverage=sizing.leverage,
            kelly_pct=sizing.kelly_pct,
            confidence_mult=sizing.confidence_mult,
            drawdown_mult=sizing.drawdown_mult,
            drawdown_pct=sizing.drawdown_pct,
            consec_loss_mult=sizing.consec_loss_mult,
            consecutive_losses=sizing.consecutive_losses,
            entry_time=entry_time,
        )

        self.open_trades[analysis.symbol] = trade
        self._log_trade(trade, "OPEN")

        logger.info(
            f"[SHADOW OPEN] BUY {trade.amount:.6f} {trade.symbol} "
            f"@ {trade.entry_price:.2f} | margin=${sizing.position_usd:.2f} "
            f"lev={sizing.leverage}x notional=${sizing.notional_usd:.2f} "
            f"| TP={trade.target_price:.2f} ({tp_pct:.2f}%) "
            f"| SL={trade.stop_price:.2f} ({sl_pct:.2f}%) "
            f"| kelly={sizing.kelly_pct:.2f} conf_mult={sizing.confidence_mult:.1f} "
            f"dd_mult={sizing.drawdown_mult:.1f} cl_mult={sizing.consec_loss_mult:.2f} "
            f"| trend={swing_trend} conf={swing_conf:.2f} ADX={trade.adx:.1f} "
            f"| post-fee R:R={trade.post_fee_rr:.2f}"
        )

        return trade

    def _open_sell(self, analysis: AnalysisResult, entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated short position."""
        walls = [w for w in analysis.ask_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            if not (hasattr(analysis, 'swing') and analysis.swing and analysis.swing.confidence >= 0.55):
                return None

        wall = walls[0] if walls else None
        tc = self.config.trading
        mid = analysis.mid_price
        entry_price = mid

        # Extract swing signal info
        swing = getattr(analysis, 'swing', None)
        swing_conf = swing.confidence if swing else 0.55
        regime_size = getattr(analysis, 'regime_size_mult', 1.0)
        session_size = getattr(analysis, 'session_size_mult', 1.0)

        # Dynamic position sizing via PositionSizer
        sizing = self.sizer.calculate(
            equity=self._equity,
            peak_equity=self._peak_equity,
            confidence=swing_conf,
            symbol=analysis.symbol,
            regime_mult=regime_size,
            session_mult=session_size,
            current_time=entry_time,
        )

        if sizing.is_halted:
            logger.info(f"[SHADOW] SELL blocked: {sizing.halt_reason}")
            return None

        amount = sizing.notional_usd / entry_price

        # Fee-adjusted TP/SL from analysis (already includes fee compensation)
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else tc.stop_loss_pct

        swing_trend = swing.primary_trend.value if swing else "neutral"
        swing_aligned = swing.trend_aligned if swing else False
        rsi_1h = swing.rsi_1h if swing else 50.0
        rsi_4h = swing.rsi_4h if swing else 50.0

        trade = ShadowTrade(
            trade_id=self._next_id(),
            symbol=analysis.symbol,
            side="SELL",
            entry_price=entry_price,
            target_price=entry_price * (1.0 - tp_pct / 100.0),
            stop_price=entry_price * (1.0 + sl_pct / 100.0),
            amount=amount,
            usd_value=sizing.position_usd,
            wall_price=wall.price if wall else 0.0,
            wall_usd_value=wall.usd_value if wall else 0.0,
            wall_multiplier=wall.multiplier if wall else 0.0,
            wall_confidence=wall.confidence if wall else 0.0,
            composite_score=analysis.composite_score,
            imbalance_ratio=analysis.imbalance.ratio,
            vwap=analysis.vwap.vwap,
            vwap_deviation_pct=analysis.vwap.deviation_pct,
            momentum_aggressor_ratio=analysis.momentum.aggressor_ratio,
            spread_pct=analysis.spread_pct,
            vpin=analysis.vpin.vpin,
            vpin_ema=analysis.vpin.vpin_ema,
            vpin_regime=analysis.vpin.regime.value,
            vpin_directional_bias=analysis.vpin.directional_bias,
            swing_trend=swing_trend,
            swing_confidence=swing_conf,
            swing_trend_aligned=swing_aligned,
            rsi_1h=rsi_1h,
            rsi_4h=rsi_4h,
            atr_tp_pct=tp_pct,
            atr_sl_pct=sl_pct,
            raw_tp_pct=getattr(analysis, 'raw_tp_pct', 0.0),
            raw_sl_pct=getattr(analysis, 'raw_sl_pct', 0.0),
            fee_cost_pct=getattr(analysis, 'fee_cost_pct', 0.0),
            post_fee_rr=getattr(analysis, 'post_fee_rr', 0.0),
            adx=getattr(analysis, 'adx', 0.0),
            regime=getattr(analysis, 'regime', 'ranging'),
            regime_size_mult=regime_size,
            regime_is_breakout=getattr(analysis, 'regime_is_breakout', False),
            session=getattr(analysis, 'session', 'dead_zone'),
            session_size_mult=session_size,
            leverage=sizing.leverage,
            kelly_pct=sizing.kelly_pct,
            confidence_mult=sizing.confidence_mult,
            drawdown_mult=sizing.drawdown_mult,
            drawdown_pct=sizing.drawdown_pct,
            consec_loss_mult=sizing.consec_loss_mult,
            consecutive_losses=sizing.consecutive_losses,
            entry_time=entry_time,
        )

        self.open_trades[analysis.symbol] = trade
        self._log_trade(trade, "OPEN")

        logger.info(
            f"[SHADOW OPEN] SELL {trade.amount:.6f} {trade.symbol} "
            f"@ {trade.entry_price:.2f} | margin=${sizing.position_usd:.2f} "
            f"lev={sizing.leverage}x notional=${sizing.notional_usd:.2f} "
            f"| TP={trade.target_price:.2f} ({tp_pct:.2f}%) "
            f"| SL={trade.stop_price:.2f} ({sl_pct:.2f}%) "
            f"| kelly={sizing.kelly_pct:.2f} conf_mult={sizing.confidence_mult:.1f} "
            f"dd_mult={sizing.drawdown_mult:.1f} cl_mult={sizing.consec_loss_mult:.2f} "
            f"| trend={swing_trend} conf={swing_conf:.2f} ADX={trade.adx:.1f} "
            f"| post-fee R:R={trade.post_fee_rr:.2f}"
        )

        return trade

    # ─────────────────────────────────────────
    # POSITION MONITORING
    # ─────────────────────────────────────────

    def update_prices(self, symbol: str, current_price: float) -> Optional[TradeRecord]:
        """
        Check if any open shadow trades should be closed based on current price.

        Returns a TradeRecord if a trade was closed, None otherwise.
        """
        # Always track latest price for unrealized P&L display
        self.last_prices[symbol] = current_price

        if symbol not in self.open_trades:
            return None

        trade = self.open_trades[symbol]
        now = time.time()

        # Don't check anything within first 2 seconds — prevents same-tick close
        hold_time = now - trade.entry_time
        if hold_time < 2.0:
            return None

        min_hold = self.config.trading.min_hold_time_s  # 30 min default

        # --- Trailing Stop ---
        # Only activates after min_hold AND price reaches 50% of TP distance.
        # This prevents tick-noise from triggering trailing exits on a swing bot.
        trailing_pct = self.config.trading.trailing_stop_pct
        if trailing_pct > 0 and hold_time >= min_hold:
            if trade.side == "BUY":
                tp_dist = trade.target_price - trade.entry_price
                activation_price = trade.entry_price + (tp_dist * 0.5)
                if current_price >= activation_price:
                    peak = self._peak_prices.get(symbol, current_price)
                    if current_price > peak:
                        peak = current_price
                        self._peak_prices[symbol] = peak
                    trailing_stop = peak * (1.0 - trailing_pct / 100.0)
                    if trailing_stop > trade.stop_price:
                        trade.stop_price = trailing_stop
            else:  # SELL
                tp_dist = trade.entry_price - trade.target_price
                activation_price = trade.entry_price - (tp_dist * 0.5)
                if current_price <= activation_price:
                    trough = self._trough_prices.get(symbol, current_price)
                    if current_price < trough:
                        trough = current_price
                        self._trough_prices[symbol] = trough
                    trailing_stop = trough * (1.0 + trailing_pct / 100.0)
                    if trailing_stop < trade.stop_price:
                        trade.stop_price = trailing_stop

        exit_reason = None
        exit_price = current_price

        # TP only triggers after min_hold_time — this is a swing bot, not a scalper.
        # SL always triggers immediately to protect capital.
        if trade.side == "BUY":
            if current_price >= trade.target_price and hold_time >= min_hold:
                exit_reason = "take_profit"
            elif current_price <= trade.stop_price:
                exit_reason = "trailing_stop" if trade.stop_price > trade.entry_price else "stop_loss"
        else:  # SELL
            if current_price <= trade.target_price and hold_time >= min_hold:
                exit_reason = "take_profit"
            elif current_price >= trade.stop_price:
                exit_reason = "trailing_stop" if trade.stop_price < trade.entry_price else "stop_loss"

        if exit_reason:
            return self._close_trade(trade, exit_price, exit_reason, now)

        return None

    def close_on_wall_pull(self, symbol: str, current_price: float) -> Optional[TradeRecord]:
        """Force close a shadow trade due to wall being pulled."""
        if symbol not in self.open_trades:
            return None

        trade = self.open_trades[symbol]
        return self._close_trade(trade, current_price, "wall_pulled", time.time())

    def _close_trade(
        self,
        trade: ShadowTrade,
        exit_price: float,
        reason: str,
        exit_time: float,
    ) -> TradeRecord:
        """Close a shadow trade and record results.

        P&L already reflects leverage because amount = (margin * leverage) / price.
        Fee cost is computed on the notional value (entry + exit legs).
        """
        if trade.side == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * trade.amount
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.amount

        # Fee calculation: fee on entry notional + fee on exit notional
        tc = self.config.trading
        fee_rate = tc.fee_rate / 100.0  # Convert from % to decimal (0.04% -> 0.0004)
        entry_notional = trade.entry_price * trade.amount
        exit_notional = exit_price * trade.amount
        fee_cost = (entry_notional + exit_notional) * fee_rate

        net_pnl = gross_pnl - fee_cost
        net_pnl_pct = (net_pnl / trade.usd_value) * 100.0

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl_usd = gross_pnl
        trade.fee_cost_usd = fee_cost
        trade.pnl_usd = net_pnl
        trade.pnl_pct = net_pnl_pct
        trade.duration_s = exit_time - trade.entry_time
        trade.is_open = False

        # Update stats with NET P&L (after fees)
        self.total_pnl += net_pnl
        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update simulated equity for position sizer
        self._equity += net_pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Feed outcome to sizer for Kelly + consecutive loss tracking
        self.sizer.record_outcome(trade.symbol, net_pnl, exit_time)

        # Move to closed and record cooldown
        del self.open_trades[trade.symbol]
        self.closed_trades.append(trade)
        self._last_close_time[trade.symbol] = exit_time
        self._peak_prices.pop(trade.symbol, None)
        self._trough_prices.pop(trade.symbol, None)

        self._log_trade(trade, "CLOSE")

        logger.info(
            f"[SHADOW CLOSE] {trade.side} {trade.symbol} "
            f"@ {exit_price:.2f} | reason={reason} "
            f"| gross=${gross_pnl:+.2f} fees=${fee_cost:.2f} "
            f"| net=${net_pnl:+.2f} ({net_pnl_pct:+.2f}%) "
            f"| duration={trade.duration_s:.1f}s"
        )

        return TradeRecord(
            symbol=trade.symbol,
            side=trade.side.lower(),
            entry_price=trade.entry_price,
            exit_price=exit_price,
            amount=trade.amount,
            pnl_usd=net_pnl,
            reason=reason,
            timestamp=exit_time,
        )

    # ─────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────

    def _log_trade(self, trade: ShadowTrade, event: str):
        """Append trade event to JSONL log."""
        record = {
            "event": event,
            "timestamp": time.time(),
            "trade": asdict(trade),
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trade log: {e}")

    def _log_signal(self, analysis: AnalysisResult):
        """Log analysis signal for later review."""
        swing = getattr(analysis, 'swing', None)
        record = {
            "timestamp": analysis.timestamp,
            "symbol": analysis.symbol,
            "mid_price": analysis.mid_price,
            "spread_pct": analysis.spread_pct,
            "composite_score": analysis.composite_score,
            "suggestion": analysis.trade_suggestion,
            "bid_walls": len(analysis.bid_walls),
            "ask_walls": len(analysis.ask_walls),
            "imbalance_ratio": analysis.imbalance.ratio,
            "imbalance_direction": analysis.imbalance.direction.value,
            "vwap": analysis.vwap.vwap,
            "vwap_deviation": analysis.vwap.deviation_pct,
            "momentum_aggressor": analysis.momentum.aggressor_ratio,
            "momentum_direction": analysis.momentum.direction.value,
            "vpin": analysis.vpin.vpin,
            "vpin_ema": analysis.vpin.vpin_ema,
            "vpin_regime": analysis.vpin.regime.value,
            "vpin_directional_bias": analysis.vpin.directional_bias,
            "vpin_blocked_entry": analysis.vpin.should_block_entry,
            "vpin_stop_multiplier": analysis.vpin.stop_multiplier,
            # Swing / candle fields
            "swing_trend": swing.primary_trend.value if swing else "none",
            "swing_confidence": swing.confidence if swing else 0.0,
            "swing_trend_aligned": swing.trend_aligned if swing else False,
            "rsi_1h": swing.rsi_1h if swing else 0.0,
            "rsi_4h": swing.rsi_4h if swing else 0.0,
            "atr_tp_pct": analysis.atr_tp_pct,
            "atr_sl_pct": analysis.atr_sl_pct,
            # Fee-aware metrics
            "raw_tp_pct": getattr(analysis, 'raw_tp_pct', 0.0),
            "raw_sl_pct": getattr(analysis, 'raw_sl_pct', 0.0),
            "fee_cost_pct": getattr(analysis, 'fee_cost_pct', 0.0),
            "post_fee_rr": getattr(analysis, 'post_fee_rr', 0.0),
            # ADX
            "adx": getattr(analysis, 'adx', 0.0),
            "adx_1h": swing.adx_1h if swing else 0.0,
            "adx_4h": swing.adx_4h if swing else 0.0,
            "adx_blocked": swing.adx_blocked if swing else False,
            # Regime
            "regime": getattr(analysis, 'regime', 'ranging'),
            "regime_blocked": getattr(analysis, 'regime_blocked', False),
            "regime_size_mult": getattr(analysis, 'regime_size_mult', 1.0),
            "regime_is_breakout": getattr(analysis, 'regime_is_breakout', False),
            # Session
            "session": getattr(analysis, 'session', 'dead_zone'),
            "session_blocked": getattr(analysis, 'session_blocked', False),
            "session_size_mult": getattr(analysis, 'session_size_mult', 1.0),
        }
        try:
            with open(self._signal_log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write signal log: {e}")

    # ─────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────

    def get_performance_summary(self) -> str:
        """Full performance report."""
        total = self.wins + self.losses
        win_rate = (self.wins / total * 100.0) if total > 0 else 0.0

        avg_win = 0.0
        avg_loss = 0.0
        max_win = 0.0
        max_loss = 0.0

        if self.wins > 0:
            win_pnls = [t.pnl_usd for t in self.closed_trades if t.pnl_usd and t.pnl_usd > 0]
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
            max_win = max(win_pnls) if win_pnls else 0

        if self.losses > 0:
            loss_pnls = [t.pnl_usd for t in self.closed_trades if t.pnl_usd and t.pnl_usd <= 0]
            avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
            max_loss = min(loss_pnls) if loss_pnls else 0

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd and t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.closed_trades if t.pnl_usd and t.pnl_usd < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average duration
        durations = [t.duration_s for t in self.closed_trades if t.duration_s]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Exit reason breakdown
        reasons = {}
        for t in self.closed_trades:
            r = t.exit_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1

        lines = [
            "╔══════════════════════════════════════╗",
            "║     SHADOW TRADING PERFORMANCE       ║",
            "╠══════════════════════════════════════╣",
            f"  Total Trades:    {total}",
            f"  Open Trades:     {len(self.open_trades)}",
            f"  Win Rate:        {win_rate:.1f}%",
            f"  Total PnL:       ${self.total_pnl:+,.2f}",
            f"  Profit Factor:   {profit_factor:.2f}",
            f"  Avg Win:         ${avg_win:+,.2f}",
            f"  Avg Loss:        ${avg_loss:,.2f}",
            f"  Max Win:         ${max_win:+,.2f}",
            f"  Max Loss:        ${max_loss:,.2f}",
            f"  Avg Duration:    {avg_duration:.1f}s",
            "",
            "  Exit Reasons:",
        ]
        for reason, count in sorted(reasons.items()):
            lines.append(f"    {reason}: {count}")

        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)

    def _next_id(self) -> int:
        self._trade_counter += 1
        return self._trade_counter
