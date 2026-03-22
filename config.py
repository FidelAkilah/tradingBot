"""
Configuration module for the Liquidity Heatmap Swing Trading Bot.

Redesigned for:
- Swing trades (hold for hours, not seconds)
- IDR-denominated capital with daily compounding
- Multi-timeframe analysis (1h/4h candles + order book)
- Wider targets that overcome Binance fee drag
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# CURRENCY
# ─────────────────────────────────────────────

# IDR to USD approximate rate (updated at startup from exchange)
IDR_PER_USD = 16_300.0  # Fallback rate


@dataclass
class FuturesConfig:
    """Binance Futures settings."""
    enabled: bool = True                        # Use USDT-M Futures
    leverage: int = 10                          # 10x default (was 30x — catastrophic at low WR)
    max_leverage: int = 15                      # Hard cap — never exceed 15x
    margin_type: str = "CROSSED"                # CROSSED (required for Multi-Assets mode)
    position_mode: str = "one-way"              # one-way or hedge


@dataclass
class ExchangeConfig:
    """Binance connection settings."""
    exchange_id: str = "binanceusdm"            # USDT-M Futures (was "binance" spot)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = False                       # Mainnet for Render deployment
    rate_limit: bool = True
    options: dict = field(default_factory=lambda: {
        "defaultType": "future",
        "adjustForTimeDifference": True,
    })


@dataclass
class OrderBookConfig:
    """Order book analysis parameters."""
    depth_limit: int = 500
    update_speed_ms: int = 500              # Slower updates OK for swing (was 100)
    snapshot_interval_s: float = 5.0        # Analyze every 5s, not 0.5s

    # --- Whale Detection ---
    whale_multiplier: float = 5.0
    whale_top_n: int = 3
    cluster_range_pct: float = 0.1          # Wider clusters for swing (was 0.05)
    min_wall_usd: float = 100_000           # Higher bar — only real walls (was 50k)

    # --- Spoofing Detection ---
    wall_persistence_window_s: float = 60.0 # Track walls over 60s, not 10s
    spoof_cancel_threshold: float = 0.6
    spoof_flicker_count: int = 5            # More flickers needed (was 3)

    # --- Imbalance Ratio ---
    imbalance_depth_levels: int = 50        # Deeper book (was 20)
    imbalance_signal_threshold: float = 1.5

    # --- VWAP Anchoring ---
    vwap_lookback_trades: int = 1000        # More trades (was 500)
    vwap_deviation_pct: float = 0.5         # Wider deviation band (was 0.3)

    # --- VPIN ---
    vpin_bucket_count: int = 50
    vpin_bucket_volume: float = 0.0
    vpin_lookback_trades: int = 1000
    vpin_toxicity_threshold: float = 0.85   # Much more permissive (was 0.7)
    vpin_safe_threshold: float = 0.5        # Wider safe zone (was 0.4)
    vpin_ema_alpha: float = 0.05            # Slower EMA (was 0.1)
    vpin_regime_window: int = 30
    vpin_block_entry_above: float = 0.99    # Only block at extreme (was 0.95, which blocked too often)
    vpin_widen_stops_above: float = 0.85    # Only widen at extreme (was 0.65)
    vpin_stop_multiplier: float = 1.3       # Less aggressive widening (was 1.5)


@dataclass
class CandleConfig:
    """Multi-timeframe candle analysis (NEW)."""
    enabled: bool = True
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h"])
    lookback_candles: int = 50              # How many candles to fetch

    # --- Trend Detection (EMA crossover) ---
    fast_ema_period: int = 9
    slow_ema_period: int = 21
    trend_ema_period: int = 50              # Long-term trend filter

    # --- RSI ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # --- ATR (for dynamic TP/SL) ---
    atr_period: int = 14
    atr_tp_multiplier: float = 2.0         # TP = entry ± 2x ATR
    atr_sl_multiplier: float = 1.0         # SL = entry ∓ 1x ATR

    # --- Volume Confirmation ---
    volume_surge_multiplier: float = 1.5   # Volume > 1.5x avg = confirmation

    # --- ADX Trend Strength Filter ---
    adx_period: int = 14
    adx_trending_threshold: float = 25.0   # ADX > 25 = trending → +0.10 confidence
    adx_ranging_threshold: float = 20.0    # ADX < 20 = ranging → hard block (reject signal)


@dataclass
class RegimeConfig:
    """Market regime detection parameters."""
    # ADX-based regime thresholds
    adx_strong_trend: float = 30.0      # ADX > 30 = STRONG_TREND (full size)
    adx_trend: float = 25.0             # ADX 25-30 = TREND (full size)
    adx_weak: float = 20.0              # ADX 20-25 = WEAK (reduce 50%)
    # Below 20 = RANGING (block)

    # Bollinger Band Width regime
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_lookback: int = 100              # Percentile window for BB width
    bb_expanding_pctl: float = 70.0     # >70th percentile = EXPANDING
    bb_squeezing_pctl: float = 30.0     # <30th percentile = SQUEEZING

    # Price vs EMAs choppy detection
    ema_trend_bars: int = 5             # Price must stay on one side of all EMAs for N bars
    ema_choppy_crosses: int = 3         # Multiple crosses in N bars = CHOPPY
    ema_choppy_window: int = 10         # Window to count crosses

    # Breakout override
    breakout_bb_expand_pct: float = 30.0   # BB width must expand >30% in 3 candles
    breakout_bb_candles: int = 3
    breakout_volume_mult: float = 2.0      # Volume must be >2x average
    breakout_sl_mult: float = 0.7          # Tighter SL: 0.7x ATR


@dataclass
class SessionConfig:
    """UTC trading session windows and position sizing."""
    enabled: bool = True

    # Session windows (UTC hours): [start, end, size_mult, label]
    # US+EU overlap: 13:00-17:00 → 100%
    # EU session: 07:00-13:00 → 80%
    # US session: 17:00-21:00 → 80%
    # Asian session: 00:00-07:00 → 50% BTC/ETH only
    # Dead zone: 21:00-00:00 → block all

    # Pairs allowed in Asian session only
    asian_allowed_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT"
    ])

    # BTC/USDT allowed in all sessions
    all_session_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USDT"
    ])


@dataclass
class TradingConfig:
    """Execution and position management."""
    # --- Pairs (fixed for futures) ---
    top_n_pairs: int = 5
    manual_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "ADA/USDT", "HYPE/USDT", "SOL/USDT"
    ])
    use_dynamic_pairs: bool = False             # Fixed pairs, no discovery

    # Exclude stablecoins and low-vol trash
    excluded_pairs: List[str] = field(default_factory=lambda: [
        "FDUSD/USDT", "USDC/USDT", "TUSD/USDT", "DAI/USDT",
        "BUSD/USDT", "USDP/USDT", "EUR/USDT", "GBP/USDT",
    ])
    min_24h_volume_usd: float = 100_000_000  # Skip pairs under $100M daily vol

    # --- Order Placement ---
    offset_from_wall_pct: float = 0.05      # Wider offset for swing (was 0.02)
    order_type: str = "limit"
    time_in_force: str = "GTC"

    # --- Position Sizing (IDR) ---
    starting_capital_idr: float = 1_000_000  # Rp 1,000,000
    max_position_pct: float = 0.20           # Max 20% of equity per trade (was 30%)
    max_position_usd: float = 100.0          # Will be computed from IDR
    position_pct_of_equity: float = 0.20     # Default 20% (was 30%)
    min_position_pct: float = 0.05           # Floor: 5% of equity minimum
    max_open_positions: int = 2              # Max 2 concurrent (was 3)

    # --- Kelly Criterion Position Sizing ---
    kelly_lookback: int = 50                 # Rolling window for win rate / avg W/L ratio
    kelly_min_trades: int = 20               # Need >= 20 trades before Kelly kicks in
    kelly_default_pct: float = 0.10          # Conservative 10% until enough history
    kelly_fraction: float = 0.5              # Half-Kelly for safety

    # --- Fee-Aware Trading ---
    fee_rate: float = 0.04                  # Binance futures taker fee per side (0.04%)
    min_post_fee_rr: float = 1.5            # Reject trades where post-fee R:R < 1.5:1

    # --- Swing Targets (ATR-based, these are fallbacks) ---
    take_profit_pct: float = 2.0            # 2% TP (was 0.15% — that was the problem)
    stop_loss_pct: float = 1.0              # 1% SL (was 0.5%)
    trailing_stop_pct: float = 0.8          # Trailing stop at 0.8% from peak

    # --- Spread Filter ---
    max_spread_pct: float = 0.1             # More permissive (was 0.05)

    # --- Wall-Pull Protection ---
    wall_pull_check_interval_s: float = 30.0  # Check every 30s, not 0.5s
    wall_pull_volume_drop_pct: float = 0.7    # 70% volume drop (was 50%)
    exit_on_wall_pull: bool = False            # DON'T exit on wall pull for swing (was True)
    # Instead, walls are used as entry confirmation, not position anchor

    # --- Compounding ---
    daily_target_pct: float = 10.0           # 10% daily target
    compound_profits: bool = True            # Re-invest profits into next day's capital

    # --- Session Timing ---
    min_hold_time_s: float = 1800.0          # Minimum 30 minutes hold (prevent noise exits)
    max_hold_time_s: float = 28800.0         # Maximum 8 hours hold
    preferred_session: str = "any"           # "asian", "european", "us", or "any"


@dataclass
class RiskConfig:
    """Global risk management."""
    max_daily_loss_usd: float = 10.0        # ~Rp 163,000 max daily loss
    max_daily_loss_pct: float = 10.0        # Or 10% of equity
    max_daily_trades: int = 25              # 5 pairs x ~5 trades each
    cooldown_after_loss_s: float = 300.0    # 5 min cooldown (was 30s)
    max_drawdown_pct: float = 25.0          # More room for swing (was 10%)

    # --- Drawdown-Based Position Scaling ---
    # Drawdown from peak equity → position size reduction
    drawdown_scale_5: float = 1.0           # 0-5%: normal
    drawdown_scale_10: float = 0.7          # 5-10%: reduce 30%
    drawdown_scale_15: float = 0.4          # 10-15%: reduce 60%
    drawdown_scale_25: float = 0.0          # 15-25%: minimum size only (floor)
    drawdown_halt: float = 25.0             # >25%: halt all trading

    # --- Consecutive Loss Adjustment ---
    consec_loss_base: float = 0.7           # size_mult = 0.7 ^ consecutive_losses
    consec_loss_cooldown_count: int = 3     # 3 consecutive losses → 30min cooldown
    consec_loss_cooldown_s: float = 1800.0  # 30 minutes
    consec_loss_halt_count: int = 4         # 4+ losses → halt pair for 2 hours
    consec_loss_halt_s: float = 7200.0      # 2 hours


@dataclass
class ShadowConfig:
    """Shadow (simulation) trading settings."""
    enabled: bool = True
    log_file: str = "shadow_trades.jsonl"
    log_order_book_snapshots: bool = False
    latency_simulation_ms: float = 100.0


@dataclass
class BotConfig:
    """Master configuration aggregator."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    order_book: OrderBookConfig = field(default_factory=OrderBookConfig)
    candle: CandleConfig = field(default_factory=CandleConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    log_level: str = "INFO"


CONFIG = BotConfig()
