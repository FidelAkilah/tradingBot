"""
Shadow Trader — Simulation mode with full trade logging.

Logs every signal, potential trade, and hypothetical P&L to JSONL files
without placing any real orders. Perfect for strategy validation.

Supports:
- Partial take profit (3 TP levels with scale-out)
- Chandelier Exit (ATR-based trailing stop)
- Dynamic SL adjustment based on early price action
- Daily-target-aware TP compression/expansion
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from config import BotConfig, CONFIG
from daily_target.tracker import DailyTargetContext, TradingMode
from liquidity_analyzer import AnalysisResult, LiquidityWall, WallSide
from position_sizer import PositionSizer, SizingResult
from risk_manager import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class TPLevel:
    """A single take-profit level for partial exit."""
    level: int               # 1, 2, or 3
    atr_mult: float          # ATR multiplier for this level
    target_price: float      # Computed at entry (0 for runner/TP3)
    size_pct: float          # Fraction of total position
    amount: float            # Absolute units for this tranche
    hit: bool = False
    hit_time: Optional[float] = None
    hit_price: Optional[float] = None
    pnl_usd: Optional[float] = None


@dataclass
class ShadowTrade:
    """A simulated trade record."""
    trade_id: int
    symbol: str
    side: str                    # "BUY" or "SELL"
    entry_price: float
    target_price: float          # Take-profit target (final TP for display)
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
    # ── Partial TP fields ──
    tp1_hit: bool = False
    tp1_price: Optional[float] = None
    tp1_pnl: Optional[float] = None
    tp2_hit: bool = False
    tp2_price: Optional[float] = None
    tp2_pnl: Optional[float] = None
    tp3_hit: bool = False
    tp3_price: Optional[float] = None
    tp3_pnl: Optional[float] = None
    original_amount: float = 0.0
    atr_at_entry: float = 0.0
    partial_realized_pnl: float = 0.0
    partial_fees: float = 0.0
    trailing_atr_mult: float = 0.0   # Current Chandelier multiplier
    original_stop_distance: float = 0.0  # Original SL distance
    max_favorable_price: float = 0.0  # Best price seen during trade
    # ── Re-entry tracking ──
    is_reentry: bool = False
    reentry_count: int = 0
    # ── Volume/Funding/OI ──
    obv_trend: str = "neutral"
    obv_divergence: bool = False
    volume_pressure: str = "neutral"
    buy_volume_ratio: float = 0.5
    poc_price: float = 0.0
    funding_rate: float = 0.0
    funding_extreme: bool = False
    oi_change_pct: float = 0.0
    oi_conviction: str = "neutral"


class ShadowTrader:
    """
    Tracks hypothetical trades based on analysis signals.

    Simulates order fills with configurable latency, tracks positions
    against live price feeds, and logs everything to JSONL for analysis.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.shadow
        self.ec = config.exit

        self._trade_counter = 0
        self.open_trades: Dict[str, ShadowTrade] = {}   # key: symbol
        self.closed_trades: List[ShadowTrade] = []
        self.signal_log: List[dict] = []

        # Position sizer for dynamic sizing
        self.sizer = PositionSizer(config)

        # Simulated equity tracking for sizer
        self._equity = config.trading.starting_capital_idr / 16_300.0  # ~$61
        self._peak_equity = self._equity

        # Daily target context (set externally by main.py)
        self.daily_target_ctx: Optional[DailyTargetContext] = None

        # Correlation size multiplier (set externally by main.py before process_signal)
        self.correlation_size_mult: float = 1.0

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

        # Partial TP state (symbol -> list of TPLevel)
        self._tp_state: Dict[str, List[TPLevel]] = {}

        # Pending partial trade records (drained by main.py for daily target)
        self.pending_partial_records: List[TradeRecord] = []

        # Dynamic SL flag tracking (symbol -> applied)
        self._flat_sl_applied: Dict[str, bool] = {}

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

    def _get_atr_from_analysis(self, analysis: AnalysisResult) -> float:
        """Extract the raw ATR value (in price units) from the swing signal."""
        swing = getattr(analysis, 'swing', None)
        if not swing or not swing.signals:
            return 0.0
        # Prefer 4h ATR, fallback to 1h
        for tf in ("4h", "1h"):
            sig = swing.signals.get(tf)
            if sig and sig.atr > 0:
                return sig.atr
        return 0.0

    def _build_tp_levels(self, entry_price: float, atr: float,
                         side: str, amount: float,
                         swing_conf: float) -> List[TPLevel]:
        """Build partial TP levels with daily-target-aware adjustments."""
        ec = self.ec
        mults = [ec.tp1_atr_mult, ec.tp2_atr_mult, ec.tp3_atr_mult]
        sizes = [ec.tp1_size_pct, ec.tp2_size_pct, ec.tp3_size_pct]

        # --- Daily-target-aware compression/expansion ---
        ctx = self.daily_target_ctx
        if ctx:
            if ctx.pct_achieved >= ec.target_near_threshold:
                # >80% target achieved: tighten all TPs by 30%
                compress = ec.target_near_compress
                mults = [m * compress for m in mults]
            elif (ctx.pct_achieved < ec.target_behind_threshold
                  and ctx.day_elapsed_pct > ec.target_behind_utc_hour / 24.0
                  and swing_conf >= ec.target_behind_min_conf):
                # Behind schedule, late in day, high confidence: expand TP2/TP3
                expand = ec.target_behind_expand
                mults[1] *= expand
                mults[2] *= expand

        # --- Edge case: TP1 must be sufficiently far from entry ---
        if mults[0] < ec.min_tp1_sl_distance_atr:
            mults[0] = ec.min_tp1_sl_distance_atr

        levels = []
        for i, (mult, size_pct) in enumerate(zip(mults, sizes)):
            level_num = i + 1
            if side == "BUY":
                tp = entry_price + (atr * mult) if level_num <= 2 else 0.0
            else:
                tp = entry_price - (atr * mult) if level_num <= 2 else 0.0
            # TP3 (runner) has no fixed target — trailing stop only
            levels.append(TPLevel(
                level=level_num,
                atr_mult=mult,
                target_price=tp,
                size_pct=size_pct,
                amount=amount * size_pct,
            ))

        return levels

    def _open_trade(self, analysis: AnalysisResult, side: str,
                    entry_time: float) -> Optional[ShadowTrade]:
        """Shared logic for opening a BUY or SELL trade."""
        if side == "BUY":
            walls = [w for w in analysis.bid_walls
                     if not w.is_spoof_suspect and w.confidence > 0.5]
        else:
            walls = [w for w in analysis.ask_walls
                     if not w.is_spoof_suspect and w.confidence > 0.5]

        if not walls:
            if not (hasattr(analysis, 'swing') and analysis.swing
                    and analysis.swing.confidence >= 0.55):
                return None

        wall = walls[0] if walls else None
        tc = self.config.trading
        entry_price = analysis.mid_price

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
            daily_target_ctx=self.daily_target_ctx,
            current_time=entry_time,
        )

        if sizing.is_halted:
            logger.info(f"[SHADOW] {side} blocked: {sizing.halt_reason}")
            return None

        # Apply correlation guard size reduction
        notional = sizing.notional_usd * self.correlation_size_mult
        amount = notional / entry_price

        # Fee-adjusted TP/SL from analysis
        tp_pct = analysis.atr_tp_pct if analysis.atr_tp_pct > 0 else tc.take_profit_pct
        sl_pct = analysis.atr_sl_pct if analysis.atr_sl_pct > 0 else tc.stop_loss_pct

        if side == "BUY":
            target_price = entry_price * (1.0 + tp_pct / 100.0)
            stop_price = entry_price * (1.0 - sl_pct / 100.0)
        else:
            target_price = entry_price * (1.0 - tp_pct / 100.0)
            stop_price = entry_price * (1.0 + sl_pct / 100.0)

        swing_trend = swing.primary_trend.value if swing else "neutral"
        swing_aligned = swing.trend_aligned if swing else False
        rsi_1h = swing.rsi_1h if swing else 50.0
        rsi_4h = swing.rsi_4h if swing else 50.0
        atr_val = self._get_atr_from_analysis(analysis)

        trade = ShadowTrade(
            trade_id=self._next_id(),
            symbol=analysis.symbol,
            side=side,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
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
            # Partial TP / exit tracking
            original_amount=amount,
            atr_at_entry=atr_val,
            original_stop_distance=abs(entry_price - stop_price),
            max_favorable_price=entry_price,
            # Volume/Funding/OI at entry
            obv_trend=swing.obv_trend if swing else "neutral",
            obv_divergence=swing.obv_divergence if swing else False,
            volume_pressure=swing.volume_pressure if swing else "neutral",
            buy_volume_ratio=swing.volume_analysis.buy_volume_ratio if swing and swing.volume_analysis else 0.5,
            poc_price=swing.poc_price if swing else 0.0,
            funding_rate=swing.funding_rate_value if swing else 0.0,
            funding_extreme=swing.funding_extreme if swing else False,
            oi_change_pct=swing.oi_change_pct if swing else 0.0,
            oi_conviction=swing.oi_conviction if swing else "neutral",
        )

        # Build partial TP levels
        if self.ec.partial_tp_enabled and atr_val > 0:
            tp_levels = self._build_tp_levels(
                entry_price, atr_val, side, amount, swing_conf
            )
            self._tp_state[analysis.symbol] = tp_levels
            # Update display target to TP2 (the "main" target)
            trade.target_price = tp_levels[1].target_price if tp_levels[1].target_price else target_price

        self.open_trades[analysis.symbol] = trade
        self._flat_sl_applied.pop(analysis.symbol, None)
        self._log_trade(trade, "OPEN")

        tp_info = ""
        if analysis.symbol in self._tp_state:
            levels = self._tp_state[analysis.symbol]
            tp_info = (
                f" | TP1={levels[0].target_price:.2f}({levels[0].size_pct*100:.0f}%)"
                f" TP2={levels[1].target_price:.2f}({levels[1].size_pct*100:.0f}%)"
                f" TP3=trail({levels[2].size_pct*100:.0f}%)"
            )

        logger.info(
            f"[SHADOW OPEN] {side} {trade.amount:.6f} {trade.symbol} "
            f"@ {trade.entry_price:.2f} | margin=${sizing.position_usd:.2f} "
            f"lev={sizing.leverage}x notional=${sizing.notional_usd:.2f} "
            f"| SL={trade.stop_price:.2f} ({sl_pct:.2f}%){tp_info} "
            f"| ATR={atr_val:.2f} "
            f"| trend={swing_trend} conf={swing_conf:.2f} ADX={trade.adx:.1f} "
            f"| post-fee R:R={trade.post_fee_rr:.2f}"
        )

        return trade

    def _open_buy(self, analysis: AnalysisResult,
                  entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated long position."""
        return self._open_trade(analysis, "BUY", entry_time)

    def _open_sell(self, analysis: AnalysisResult,
                   entry_time: float) -> Optional[ShadowTrade]:
        """Open a simulated short position."""
        return self._open_trade(analysis, "SELL", entry_time)

    # ─────────────────────────────────────────
    # POSITION MONITORING
    # ─────────────────────────────────────────

    def update_prices(self, symbol: str, current_price: float) -> Optional[TradeRecord]:
        """
        Check if any open shadow trades should be closed based on current price.

        Implements:
        1. Peak/trough price tracking
        2. Dynamic SL adjustment (first 15-30 min)
        3. Partial TP level checks
        4. Chandelier Exit (ATR trailing stop)
        5. SL/trailing stop exit detection

        Returns a TradeRecord if a trade was FULLY closed, None otherwise.
        Partial TP records are queued in self.pending_partial_records.
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
        ec = self.ec
        atr = trade.atr_at_entry
        is_long = trade.side == "BUY"

        # ── 1. Track peak/trough prices ──
        if is_long:
            peak = self._peak_prices.get(symbol, current_price)
            if current_price > peak:
                self._peak_prices[symbol] = current_price
            if current_price > trade.max_favorable_price:
                trade.max_favorable_price = current_price
        else:
            trough = self._trough_prices.get(symbol, current_price)
            if current_price < trough:
                self._trough_prices[symbol] = current_price
            if trade.max_favorable_price == trade.entry_price or current_price < trade.max_favorable_price:
                trade.max_favorable_price = current_price

        # ── 2. Dynamic SL adjustment (early price action) ──
        if ec.dynamic_sl_enabled and atr > 0:
            if is_long:
                move_atr = (current_price - trade.entry_price) / atr
            else:
                move_atr = (trade.entry_price - current_price) / atr

            # Momentum confirmed: favorable move ≥ 0.5×ATR within 15 min
            if (hold_time <= ec.dynamic_sl_momentum_window_s
                    and move_atr >= ec.dynamic_sl_momentum_atr_move):
                new_sl_dist = atr * ec.dynamic_sl_momentum_tighten
                if is_long:
                    new_sl = trade.entry_price - new_sl_dist
                    if new_sl > trade.stop_price:
                        trade.stop_price = new_sl
                else:
                    new_sl = trade.entry_price + new_sl_dist
                    if new_sl < trade.stop_price:
                        trade.stop_price = new_sl

            # Flat after 30 min: move < 0.2×ATR → tighten to reduce drift exposure
            if (hold_time >= ec.dynamic_sl_flat_window_s
                    and abs(move_atr) < ec.dynamic_sl_flat_atr_move
                    and not self._flat_sl_applied.get(symbol, False)):
                self._flat_sl_applied[symbol] = True
                new_sl_dist = atr * ec.dynamic_sl_flat_tighten
                if is_long:
                    new_sl = trade.entry_price - new_sl_dist
                    if new_sl > trade.stop_price:
                        trade.stop_price = new_sl
                else:
                    new_sl = trade.entry_price + new_sl_dist
                    if new_sl < trade.stop_price:
                        trade.stop_price = new_sl

        # ── 3. Partial TP checks (only after min_hold) ──
        if ec.partial_tp_enabled and symbol in self._tp_state and hold_time >= min_hold:
            tp_levels = self._tp_state[symbol]
            fee_rate = self.config.trading.fee_rate / 100.0

            for level in tp_levels:
                if level.hit:
                    continue
                # TP3 has no fixed target — it's a runner
                if level.level == 3:
                    continue

                hit = False
                if is_long and current_price >= level.target_price:
                    hit = True
                elif not is_long and current_price <= level.target_price:
                    hit = True

                if not hit:
                    continue

                # ── Partial close this tranche ──
                level.hit = True
                level.hit_time = now
                level.hit_price = current_price

                if is_long:
                    gross = (current_price - trade.entry_price) * level.amount
                else:
                    gross = (trade.entry_price - current_price) * level.amount
                fee = (trade.entry_price * level.amount
                       + current_price * level.amount) * fee_rate
                level.pnl_usd = gross - fee

                trade.partial_realized_pnl += level.pnl_usd
                trade.partial_fees += fee
                trade.amount -= level.amount

                # Update equity for partial PnL
                self._equity += level.pnl_usd
                if self._equity > self._peak_equity:
                    self._peak_equity = self._equity

                if level.level == 1:
                    trade.tp1_hit = True
                    trade.tp1_price = current_price
                    trade.tp1_pnl = level.pnl_usd
                elif level.level == 2:
                    trade.tp2_hit = True
                    trade.tp2_price = current_price
                    trade.tp2_pnl = level.pnl_usd

                # ── SL adjustment after TP hits ──
                if level.level == 1 and ec.sl_to_breakeven_after_tp1:
                    fee_offset = trade.entry_price * fee_rate * 2
                    if is_long:
                        be = trade.entry_price + fee_offset
                        trade.stop_price = max(trade.stop_price, be)
                    else:
                        be = trade.entry_price - fee_offset
                        trade.stop_price = min(trade.stop_price, be)
                    trade.trailing_atr_mult = ec.chandelier_after_tp1_mult

                elif level.level == 2 and ec.sl_to_tp1_after_tp2:
                    tp1 = next((l for l in tp_levels if l.level == 1), None)
                    if tp1 and tp1.target_price:
                        if is_long:
                            trade.stop_price = max(trade.stop_price, tp1.target_price)
                        else:
                            trade.stop_price = min(trade.stop_price, tp1.target_price)
                    trade.trailing_atr_mult = ec.chandelier_after_tp2_mult

                self._log_partial_tp(trade, level)

                # Queue partial record for daily target + risk tracking
                self.pending_partial_records.append(TradeRecord(
                    symbol=trade.symbol,
                    side=trade.side.lower(),
                    entry_price=trade.entry_price,
                    exit_price=current_price,
                    amount=level.amount,
                    pnl_usd=level.pnl_usd,
                    reason=f"partial_tp{level.level}",
                    timestamp=now,
                ))

                logger.info(
                    f"[SHADOW TP{level.level}] {trade.symbol} "
                    f"@ {current_price:.2f} | closed {level.size_pct*100:.0f}% "
                    f"({level.amount:.6f}) | PnL: ${level.pnl_usd:+.2f} "
                    f"| remaining: {trade.amount:.6f} "
                    f"| SL→{trade.stop_price:.2f}"
                )

                # If ALL fixed TP levels hit and remaining is only TP3 runner,
                # don't close — let the trailing stop handle TP3
                if trade.amount <= 0:
                    # Edge case: all tranches closed (shouldn't happen normally)
                    return self._close_trade(trade, current_price,
                                             "take_profit_full", now)

        # ── 4. Chandelier Exit (ATR-based trailing stop) ──
        if ec.chandelier_enabled and atr > 0 and hold_time >= min_hold:
            trail_mult = (trade.trailing_atr_mult
                          if trade.trailing_atr_mult > 0
                          else ec.chandelier_atr_mult)

            # Daily target mode: wider trailing when in bonus territory
            ctx = self.daily_target_ctx
            if ctx:
                if ctx.pct_achieved > 100.0:
                    trail_mult *= ec.target_bonus_trail_mult
                elif ctx.mode == TradingMode.PROTECTING:
                    trail_mult *= self.config.daily_target.protecting_trailing_tighten

            trail_distance = atr * trail_mult

            if is_long:
                # Activate after price reaches activation threshold or TP1 hit
                tp1_target = self._get_tp1_target(symbol, trade)
                activation = (trade.entry_price
                              + (tp1_target - trade.entry_price)
                              * ec.chandelier_activation_pct)
                peak = self._peak_prices.get(symbol, current_price)

                if current_price >= activation or trade.tp1_hit:
                    chandelier_stop = peak - trail_distance
                    # Never trail below breakeven after TP1
                    if trade.tp1_hit:
                        fee_offset = (trade.entry_price
                                      * (self.config.trading.fee_rate / 100.0) * 2)
                        chandelier_stop = max(chandelier_stop,
                                              trade.entry_price + fee_offset)
                    if chandelier_stop > trade.stop_price:
                        trade.stop_price = chandelier_stop
            else:
                tp1_target = self._get_tp1_target(symbol, trade)
                activation = (trade.entry_price
                              - (trade.entry_price - tp1_target)
                              * ec.chandelier_activation_pct)
                trough = self._trough_prices.get(symbol, current_price)

                if current_price <= activation or trade.tp1_hit:
                    chandelier_stop = trough + trail_distance
                    if trade.tp1_hit:
                        fee_offset = (trade.entry_price
                                      * (self.config.trading.fee_rate / 100.0) * 2)
                        chandelier_stop = min(chandelier_stop,
                                              trade.entry_price - fee_offset)
                    if chandelier_stop < trade.stop_price:
                        trade.stop_price = chandelier_stop

        elif not ec.chandelier_enabled and hold_time >= min_hold:
            # Fallback: original fixed trailing stop
            trailing_pct = self.config.trading.trailing_stop_pct
            if trailing_pct > 0:
                if is_long:
                    tp_dist = trade.target_price - trade.entry_price
                    activation_price = trade.entry_price + (tp_dist * 0.5)
                    if current_price >= activation_price:
                        peak = self._peak_prices.get(symbol, current_price)
                        if current_price > peak:
                            self._peak_prices[symbol] = current_price
                            peak = current_price
                        trailing_stop = peak * (1.0 - trailing_pct / 100.0)
                        if trailing_stop > trade.stop_price:
                            trade.stop_price = trailing_stop
                else:
                    tp_dist = trade.entry_price - trade.target_price
                    activation_price = trade.entry_price - (tp_dist * 0.5)
                    if current_price <= activation_price:
                        trough = self._trough_prices.get(symbol, current_price)
                        if current_price < trough:
                            self._trough_prices[symbol] = current_price
                            trough = current_price
                        trailing_stop = trough * (1.0 + trailing_pct / 100.0)
                        if trailing_stop < trade.stop_price:
                            trade.stop_price = trailing_stop

        # ── 5. Check exit conditions ──
        exit_reason = None
        exit_price = current_price

        if is_long:
            # Single TP check (when partial TP is disabled)
            if not ec.partial_tp_enabled:
                if current_price >= trade.target_price and hold_time >= min_hold:
                    exit_reason = "take_profit"
            # SL / trailing stop
            if current_price <= trade.stop_price:
                if trade.tp1_hit or trade.stop_price > trade.entry_price:
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = "stop_loss"
        else:
            if not ec.partial_tp_enabled:
                if current_price <= trade.target_price and hold_time >= min_hold:
                    exit_reason = "take_profit"
            if current_price >= trade.stop_price:
                if trade.tp1_hit or trade.stop_price < trade.entry_price:
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = "stop_loss"

        if exit_reason:
            return self._close_trade(trade, exit_price, exit_reason, now)

        return None

    def _get_tp1_target(self, symbol: str, trade: ShadowTrade) -> float:
        """Get TP1 target price for Chandelier activation."""
        if symbol in self._tp_state:
            tp1 = self._tp_state[symbol][0]
            if tp1.target_price:
                return tp1.target_price
        return trade.target_price

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

        P&L accounts for partial exits:
        - remaining_pnl: PnL on whatever position size is left
        - total_pnl: partial_realized_pnl + remaining_pnl
        """
        remaining_amount = trade.amount  # May be reduced by partial TPs
        is_long = trade.side == "BUY"

        if is_long:
            gross_remaining = (exit_price - trade.entry_price) * remaining_amount
        else:
            gross_remaining = (trade.entry_price - exit_price) * remaining_amount

        # Fee only on remaining amount (partial fees already charged)
        tc = self.config.trading
        fee_rate = tc.fee_rate / 100.0
        entry_notional = trade.entry_price * remaining_amount
        exit_notional = exit_price * remaining_amount
        remaining_fee = (entry_notional + exit_notional) * fee_rate

        remaining_pnl = gross_remaining - remaining_fee

        # If TP3 (runner) exits here, mark it
        if trade.tp1_hit and trade.tp2_hit and not trade.tp3_hit:
            trade.tp3_hit = True
            trade.tp3_price = exit_price
            trade.tp3_pnl = remaining_pnl

        # Total PnL: partial exits + final remaining exit
        total_gross = gross_remaining
        total_fees = remaining_fee + trade.partial_fees
        total_net = trade.partial_realized_pnl + remaining_pnl

        # Compute gross from all exits for the record
        if trade.partial_realized_pnl != 0:
            total_gross = total_net + total_fees  # back-derive

        # Restore original amount for reporting
        report_amount = trade.original_amount if trade.original_amount > 0 else remaining_amount
        net_pnl_pct = (total_net / trade.usd_value) * 100.0 if trade.usd_value > 0 else 0.0

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.gross_pnl_usd = total_gross
        trade.fee_cost_usd = total_fees
        trade.pnl_usd = total_net
        trade.pnl_pct = net_pnl_pct
        trade.duration_s = exit_time - trade.entry_time
        trade.is_open = False
        trade.amount = report_amount  # Restore for logging

        # Update stats with NET P&L (after fees)
        self.total_pnl += total_net
        if total_net > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update simulated equity for position sizer
        # Only add remaining_pnl to equity — partial PnL already added
        self._equity += remaining_pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Feed outcome to sizer for Kelly + consecutive loss tracking
        self.sizer.record_outcome(trade.symbol, total_net, exit_time)

        # Move to closed and record cooldown
        del self.open_trades[trade.symbol]
        self.closed_trades.append(trade)
        self._last_close_time[trade.symbol] = exit_time
        self._peak_prices.pop(trade.symbol, None)
        self._trough_prices.pop(trade.symbol, None)
        self._tp_state.pop(trade.symbol, None)
        self._flat_sl_applied.pop(trade.symbol, None)

        self._log_trade(trade, "CLOSE")

        partial_info = ""
        if trade.tp1_hit or trade.tp2_hit:
            partial_info = (
                f" | partials=${trade.partial_realized_pnl:+.2f}"
                f" TP1={'Y' if trade.tp1_hit else 'N'}"
                f" TP2={'Y' if trade.tp2_hit else 'N'}"
                f" TP3={'Y' if trade.tp3_hit else 'N'}"
            )

        logger.info(
            f"[SHADOW CLOSE] {trade.side} {trade.symbol} "
            f"@ {exit_price:.2f} | reason={reason} "
            f"| gross=${total_gross:+.2f} fees=${total_fees:.2f} "
            f"| net=${total_net:+.2f} ({net_pnl_pct:+.2f}%) "
            f"| duration={trade.duration_s:.1f}s{partial_info}"
        )

        return TradeRecord(
            symbol=trade.symbol,
            side=trade.side.lower(),
            entry_price=trade.entry_price,
            exit_price=exit_price,
            amount=report_amount,
            pnl_usd=remaining_pnl,  # Only the remaining portion (partials already recorded)
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

    def _log_partial_tp(self, trade: ShadowTrade, level: TPLevel):
        """Log a partial take-profit event."""
        record = {
            "event": f"PARTIAL_TP{level.level}",
            "timestamp": time.time(),
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "level": level.level,
            "hit_price": level.hit_price,
            "amount_closed": level.amount,
            "pnl_usd": level.pnl_usd,
            "remaining_amount": trade.amount,
            "remaining_pct": trade.amount / trade.original_amount if trade.original_amount else 0,
            "new_stop_price": trade.stop_price,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write partial TP log: {e}")

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

        # Partial TP stats
        tp1_count = sum(1 for t in self.closed_trades if t.tp1_hit)
        tp2_count = sum(1 for t in self.closed_trades if t.tp2_hit)
        tp3_count = sum(1 for t in self.closed_trades if t.tp3_hit)

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
            f"  Partial TP: TP1={tp1_count} TP2={tp2_count} TP3={tp3_count}",
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
