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
    atr_tp_pct: float = 0.0         # ATR-derived TP %
    atr_sl_pct: float = 0.0         # ATR-derived SL %
    entry_time: float = 0.0
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
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

        # Stats
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

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

        # Simulate execution latency
        entry_time = time.time() + (self.sc.latency_simulation_ms / 1000.0)

        if analysis.trade_suggestion == "BUY":
            return self._open_buy(analysis, entry_time)
        elif analysis.trade_suggestion == "SELL":
            return self._open_sell(analysis, entry_time)

        return None

    def _open_buy(self, analysis: AnalysisResult, entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated long position."""
        walls = [w for w in analysis.bid_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            # For swing trades, walls are optional confirmation — allow entry without walls
            # if candle signal is strong enough
            if not (hasattr(analysis, 'swing') and analysis.swing and analysis.swing.confidence >= 0.6):
                return None

        wall = walls[0] if walls else None
        tc = self.config.trading
        mid = analysis.mid_price

        if wall:
            offset = mid * (tc.offset_from_wall_pct / 100.0)
            entry_price = wall.price + offset
        else:
            entry_price = mid  # Enter at mid if no wall (candle-driven entry)

        amount = tc.max_position_usd / entry_price

        # Use ATR-based dynamic TP/SL if available, fall back to config
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else tc.stop_loss_pct

        # Extract swing signal info
        swing = getattr(analysis, 'swing', None)
        swing_trend = swing.primary_trend.value if swing else "neutral"
        swing_conf = swing.confidence if swing else 0.0
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
            usd_value=tc.max_position_usd,
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
            entry_time=entry_time,
        )

        self.open_trades[analysis.symbol] = trade
        self._log_trade(trade, "OPEN")

        logger.info(
            f"[SHADOW OPEN] BUY {trade.amount:.6f} {trade.symbol} "
            f"@ {trade.entry_price:.2f} | wall=${trade.wall_usd_value:,.0f} "
            f"| TP={trade.target_price:.2f} ({tp_pct:.2f}%) "
            f"| SL={trade.stop_price:.2f} ({sl_pct:.2f}%) "
            f"| trend={swing_trend} conf={swing_conf:.2f} "
            f"| score={trade.composite_score:+.3f} | vpin={trade.vpin:.3f}"
        )

        return trade

    def _open_sell(self, analysis: AnalysisResult, entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated short position."""
        walls = [w for w in analysis.ask_walls if not w.is_spoof_suspect and w.confidence > 0.5]
        if not walls:
            # For swing trades, walls are optional — allow entry on strong candle signal
            if not (hasattr(analysis, 'swing') and analysis.swing and analysis.swing.confidence >= 0.6):
                return None

        wall = walls[0] if walls else None
        tc = self.config.trading
        mid = analysis.mid_price

        if wall:
            offset = mid * (tc.offset_from_wall_pct / 100.0)
            entry_price = wall.price - offset
        else:
            entry_price = mid  # Enter at mid if no wall (candle-driven entry)

        amount = tc.max_position_usd / entry_price

        # Use ATR-based dynamic TP/SL if available, fall back to config
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else tc.stop_loss_pct

        # Extract swing signal info
        swing = getattr(analysis, 'swing', None)
        swing_trend = swing.primary_trend.value if swing else "neutral"
        swing_conf = swing.confidence if swing else 0.0
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
            usd_value=tc.max_position_usd,
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
            entry_time=entry_time,
        )

        self.open_trades[analysis.symbol] = trade
        self._log_trade(trade, "OPEN")

        logger.info(
            f"[SHADOW OPEN] SELL {trade.amount:.6f} {trade.symbol} "
            f"@ {trade.entry_price:.2f} | wall=${trade.wall_usd_value:,.0f} "
            f"| TP={trade.target_price:.2f} ({tp_pct:.2f}%) "
            f"| SL={trade.stop_price:.2f} ({sl_pct:.2f}%) "
            f"| trend={swing_trend} conf={swing_conf:.2f} "
            f"| score={trade.composite_score:+.3f} | vpin={trade.vpin:.3f}"
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
        if symbol not in self.open_trades:
            return None

        trade = self.open_trades[symbol]
        now = time.time()

        exit_reason = None
        exit_price = current_price

        if trade.side == "BUY":
            if current_price >= trade.target_price:
                exit_reason = "take_profit"
            elif current_price <= trade.stop_price:
                exit_reason = "stop_loss"
        else:  # SELL
            if current_price <= trade.target_price:
                exit_reason = "take_profit"
            elif current_price >= trade.stop_price:
                exit_reason = "stop_loss"

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
        """Close a shadow trade and record results."""
        if trade.side == "BUY":
            pnl = (exit_price - trade.entry_price) * trade.amount
        else:
            pnl = (trade.entry_price - exit_price) * trade.amount

        pnl_pct = (pnl / trade.usd_value) * 100.0

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_usd = pnl
        trade.pnl_pct = pnl_pct
        trade.duration_s = exit_time - trade.entry_time
        trade.is_open = False

        # Update stats
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Move to closed
        del self.open_trades[trade.symbol]
        self.closed_trades.append(trade)

        self._log_trade(trade, "CLOSE")

        logger.info(
            f"[SHADOW CLOSE] {trade.side} {trade.symbol} "
            f"@ {exit_price:.2f} | reason={reason} "
            f"| PnL=${pnl:+.2f} ({pnl_pct:+.2f}%) "
            f"| duration={trade.duration_s:.1f}s"
        )

        return TradeRecord(
            symbol=trade.symbol,
            side=trade.side.lower(),
            entry_price=trade.entry_price,
            exit_price=exit_price,
            amount=trade.amount,
            pnl_usd=pnl,
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
