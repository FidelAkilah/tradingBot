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
    max_leverage: int = 20                      # Hard cap — never exceed 20x
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

    # --- Daily Timeframe Trend Filter ---
    daily_enabled: bool = True             # Use daily candles as trend gate
    daily_lookback: int = 20               # Candles to fetch for daily TF
    daily_cache_ttl: float = 300.0         # 5 min TTL (daily candles change slowly)
    daily_neutral_penalty: float = 0.10    # Confidence reduction when daily is neutral

    # --- 15-Minute Entry Timing ---
    entry_15m_enabled: bool = True         # Check 15m chart for pullback entry
    entry_15m_lookback: int = 20           # Candles to fetch for 15m TF
    entry_15m_cache_ttl: float = 60.0      # 1 min TTL for 15m candles
    entry_15m_max_wait_candles: int = 3    # Wait up to 3 candles (45 min) for pullback
    entry_15m_rsi_long_dip: float = 45.0   # Long: RSI dips below this
    entry_15m_rsi_short_spike: float = 55.0 # Short: RSI spikes above this

    # --- Support/Resistance Levels ---
    sr_enabled: bool = True                # Use S/R level detection
    sr_pivot_lookback: int = 5             # Candle window for pivot detection
    sr_proximity_atr_pct: float = 0.3      # "Near" a level = within 0.3 × ATR
    sr_confidence_bonus: float = 0.05      # Confidence boost when near S/R
    sr_min_touches: int = 2                # Min touches to consider a level "strong"
    sr_fib_min_swing_pct: float = 3.0      # Min swing size (%) for Fibonacci calc

    # --- Candlestick Pattern Recognition ---
    patterns_enabled: bool = True          # Use candlestick pattern detection
    pattern_doji_body_pct: float = 0.10    # Doji: body < 10% of range
    pattern_pin_wick_ratio: float = 2.0    # Pin bar: wick > 2x body
    pattern_marubozu_body_pct: float = 0.90  # Marubozu: body > 90% of range
    pattern_confirm_bonus: float = 0.10    # Confidence boost for confirming pattern
    pattern_contradict_penalty: float = 0.15  # Confidence penalty for contradicting pattern
    pattern_doji_penalty: float = 0.10     # Confidence penalty for doji indecision

    # --- RSI Divergence Detection ---
    divergence_enabled: bool = True        # Use RSI divergence detection
    divergence_lookback: int = 20          # Candles to search for divergence pairs
    divergence_pivot_lookback: int = 3     # Swing pivot detection window
    divergence_oversold: float = 40.0      # RSI zone threshold for bullish div strength
    divergence_overbought: float = 60.0    # RSI zone threshold for bearish div strength
    divergence_regular_bonus: float = 0.15 # Confidence boost for regular div WITH direction
    divergence_hidden_bonus: float = 0.10  # Confidence boost for hidden div WITH direction

    # --- MACD (Moving Average Convergence Divergence) ---
    macd_enabled: bool = True
    macd_fast_period: int = 12             # Fast EMA period
    macd_slow_period: int = 26             # Slow EMA period
    macd_signal_period: int = 9            # Signal line EMA period
    macd_confirm_bonus: float = 0.05       # MACD confirms trade direction
    macd_crossover_bonus: float = 0.05     # Fresh crossover within lookback
    macd_diverge_penalty: float = 0.05     # MACD opposes trade direction
    macd_crossover_lookback: int = 3       # Candles to check for fresh crossover

    # --- BB Squeeze + Keltner Channel ---
    squeeze_enabled: bool = True
    squeeze_bb_period: int = 20            # BB SMA period
    squeeze_bb_std: float = 2.0            # BB standard deviation multiplier
    squeeze_kc_ema_period: int = 20        # Keltner Channel EMA period
    squeeze_kc_atr_period: int = 10        # Keltner Channel ATR period
    squeeze_kc_atr_mult: float = 1.5       # Keltner Channel ATR multiplier
    squeeze_release_volume_mult: float = 1.5  # Volume surge for release confirmation
    squeeze_release_bonus: float = 0.10    # Confidence boost for squeeze release
    squeeze_release_sl_mult: float = 0.7   # Tighter SL for squeeze release (0.7×ATR)
    squeeze_release_tp1_mult: float = 3.0  # Wider TP1 for squeeze release (3×ATR)


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
    max_position_pct: float = 0.25           # Max 25% of equity per trade (Kelly cap)
    max_position_usd: float = 100.0          # Will be computed from IDR
    position_pct_of_equity: float = 0.20     # Default 20% (was 30%)
    min_position_pct: float = 0.05           # Floor: 5% of equity minimum
    max_open_positions: int = 2              # Max 2 concurrent (was 3)

    # --- Kelly Criterion Position Sizing ---
    kelly_lookback: int = 50                 # Rolling window for win rate / avg W/L ratio
    kelly_min_trades: int = 20               # Need >= 20 trades before Kelly kicks in
    kelly_default_pct: float = 0.15          # Conservative 15% until enough history
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

    # --- Intraday Drawdown (from day-open equity, tighter for compounding) ---
    intraday_dd_scale_3: float = 1.0       # 0-3%: normal
    intraday_dd_scale_5: float = 0.7       # 3-5%: reduce 30%
    intraday_dd_scale_7: float = 0.4       # 5-7%: reduce 60%
    intraday_dd_halt: float = 7.0          # >=7%: halt trading for the day

    # --- Consecutive Loss Adjustment ---
    consec_loss_base: float = 0.7           # size_mult = 0.7 ^ consecutive_losses
    consec_loss_cooldown_count: int = 3     # 3 consecutive losses → 30min cooldown
    consec_loss_cooldown_s: float = 1800.0  # 30 minutes
    consec_loss_halt_count: int = 4         # 4+ losses → halt pair for 2 hours
    consec_loss_halt_s: float = 7200.0      # 2 hours


@dataclass
class VolumeConfig:
    """Enhanced volume, funding rate & open interest analysis."""
    # --- OBV ---
    obv_ema_period: int = 10                   # EMA period for OBV trend
    obv_divergence_lookback: int = 10          # Candles to check for OBV-price divergence
    obv_confirm_bonus: float = 0.05            # Confidence boost when OBV confirms direction
    obv_divergence_penalty: float = -0.10      # Penalty when OBV diverges from price

    # --- Buy/Sell Volume Estimation ---
    buy_sell_lookback: int = 10                # Candles for buy/sell volume calc
    buy_sell_threshold: float = 0.60           # Ratio above this = "buying" pressure
    buy_sell_pressure_bonus: float = 0.05      # Confidence boost for aligned pressure

    # --- Volume Profile ---
    profile_lookback: int = 30                 # Candles for volume profile
    profile_bins: int = 20                     # Price bins for profile
    profile_value_area_pct: float = 0.70       # 70% of volume = value area
    profile_poc_proximity: float = 0.005       # Within 0.5% of POC = "at"

    # --- Funding Rate ---
    funding_enabled: bool = True
    funding_extreme_threshold: float = 0.05    # |rate| > 0.05% = extreme
    funding_extreme_penalty: float = 0.10      # Confidence penalty for extreme crowding
    funding_persistent_periods: int = 3        # N periods same sign = persistent
    funding_persistent_bonus: float = 0.05     # Confidence bonus for trend-confirming funding
    funding_fetch_ttl: float = 300.0           # Cache funding data for 5 min

    # --- Open Interest ---
    oi_enabled: bool = True
    oi_rising_threshold: float = 3.0           # OI change > 3% = rising
    oi_falling_threshold: float = -3.0         # OI change < -3% = falling
    oi_price_move_threshold: float = 0.5       # Price move > 0.5% = meaningful
    oi_strong_conviction_bonus: float = 0.05   # Confidence boost for strong conviction
    oi_exhaustion_penalty: float = 0.05        # Penalty for exhaustion signal
    oi_fetch_ttl: float = 300.0                # Cache OI data for 5 min


@dataclass
class ExitConfig:
    """Exit strategy — partial TP, Chandelier trailing, dynamic SL."""
    # --- Partial Take Profit ---
    partial_tp_enabled: bool = True
    tp1_atr_mult: float = 1.0          # TP1: 1.0x ATR — quick profit, cover fees
    tp2_atr_mult: float = 2.0          # TP2: 2.0x ATR — main profit target
    tp3_atr_mult: float = 3.0          # TP3: 3.0x ATR — display only, runner uses trailing
    tp1_size_pct: float = 0.40         # 40% of position
    tp2_size_pct: float = 0.35         # 35% of position
    tp3_size_pct: float = 0.25         # 25% of position (runner, no fixed TP)

    # --- SL moves after TP hits ---
    sl_to_breakeven_after_tp1: bool = True   # Move SL to entry + fees after TP1
    sl_to_tp1_after_tp2: bool = True         # Move SL to TP1 price after TP2

    # --- Daily-target-aware TP adjustments ---
    target_near_compress: float = 0.70       # Compress TPs by 30% when near target
    target_near_threshold: float = 80.0      # >80% of daily target achieved
    target_bonus_trail_mult: float = 1.5     # Wider trailing when >100% (house money)
    target_behind_expand: float = 1.20       # Expand TP2/TP3 by 20% when behind
    target_behind_threshold: float = 30.0    # <30% of daily target achieved
    target_behind_min_conf: float = 0.75     # Only expand for high-confidence
    target_behind_utc_hour: int = 18         # Only after 18:00 UTC

    # --- Chandelier Exit (ATR-based trailing stop) ---
    chandelier_enabled: bool = True
    chandelier_atr_mult: float = 2.0         # Longs: highest_high - 2×ATR
    chandelier_after_tp1_mult: float = 1.5   # Tighten after TP1
    chandelier_after_tp2_mult: float = 1.0   # Tighten further after TP2
    chandelier_activation_pct: float = 0.3   # Activate at 30% of TP1 distance

    # --- Dynamic SL Adjustment (early price action) ---
    dynamic_sl_enabled: bool = True
    dynamic_sl_momentum_window_s: float = 900.0    # 15 min
    dynamic_sl_momentum_atr_move: float = 0.5      # Favorable move ≥ 0.5×ATR
    dynamic_sl_momentum_tighten: float = 0.7       # Tighten SL to 0.7×ATR
    dynamic_sl_flat_window_s: float = 1800.0       # 30 min
    dynamic_sl_flat_atr_move: float = 0.2          # Move < 0.2×ATR = flat
    dynamic_sl_flat_tighten: float = 0.5           # Tighten SL to 0.5×ATR

    # --- Edge cases ---
    min_tp1_sl_distance_atr: float = 0.5     # TP1 must be ≥ 0.5×ATR from entry


@dataclass
class ReentryConfig:
    """Smart re-entry after stop-outs."""
    enabled: bool = True
    cooldown_s: float = 180.0               # 3 min wait after stop-out
    max_reentries_per_signal: int = 1       # Only 1 re-entry per original signal
    expiry_s: float = 3600.0               # Re-entry window: 1 hour
    size_mult: float = 0.70                # 70% of original position size
    sl_atr_mult: float = 0.8              # Tighter SL: 0.8×ATR
    min_adx: float = 25.0                 # ADX must be >25 for re-entry
    min_confidence: float = 0.60           # Signal confidence ≥0.60
    daily_loss_block_pct: float = 50.0     # Block if >50% daily loss consumed


@dataclass
class CorrelationConfig:
    """Cross-pair correlation guard and portfolio exposure management."""
    enabled: bool = True
    # Correlation matrix
    lookback_days: int = 30                  # Rolling window for return correlation
    cache_ttl: float = 86400.0               # 24h cache for correlation matrix
    min_candles: int = 15                    # Min daily candles to compute correlation

    # Correlation guard thresholds
    high_corr_threshold: float = 0.80        # corr > 0.80 → block 2nd position
    medium_corr_threshold: float = 0.60      # corr 0.60-0.80 → reduce size 50%
    medium_corr_size_mult: float = 0.50      # Size multiplier for medium correlation

    # Same-direction exception
    strong_signal_min_conf: float = 0.75     # Both signals must exceed this
    strong_signal_max_exposure: float = 1.5  # Max combined exposure (× single position)

    # Portfolio heat map
    max_net_long_exposure: float = 20.0      # Max net long as multiple of capital
    max_net_short_exposure: float = 20.0     # Max net short as multiple of capital


@dataclass
class ScannerConfig:
    """Opportunity scanner for dynamic pair selection."""
    enabled: bool = True
    scan_interval_s: float = 7200.0              # Scan every 2 hours
    scan_top_n: int = 30                         # Scan top 30 pairs by volume
    select_dynamic: int = 3                      # Select top 3 dynamic pairs
    rate_limit_delay_s: float = 0.35             # Delay between API calls (rate limit)

    # Always-included pairs (most liquid)
    anchor_pairs: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT"
    ])

    # Selection filters
    min_24h_volume_usd: float = 50_000_000.0     # $50M minimum 24h volume
    max_spread_pct: float = 0.05                 # Max 0.05% spread
    min_adx: float = 22.0                        # Some trend present
    min_atr_pct: float = 0.5                     # Min volatility (ATR/price %)

    # Opportunity score weights
    weight_adx: float = 0.30                     # ADX trending strength
    weight_bb_squeeze: float = 0.20              # BB width percentile (squeeze potential)
    weight_volume_change: float = 0.20           # Volume vs 7-day average
    weight_volatility: float = 0.20              # ATR as % of price
    weight_funding: float = 0.10                 # Funding rate extremity

    # Pair performance tracking
    perf_lookback_trades: int = 20               # Last N trades for win rate
    perf_disable_wr_below: float = 35.0          # Auto-disable if WR < 35%
    perf_auto_include_wr_above: float = 60.0     # Auto-include if WR > 60% and in scan
    perf_min_trades_for_disable: int = 10        # Need at least 10 trades before disabling


@dataclass
class ShadowConfig:
    """Shadow (simulation) trading settings."""
    enabled: bool = True
    log_file: str = "shadow_trades.jsonl"
    log_order_book_snapshots: bool = False
    latency_simulation_ms: float = 100.0


@dataclass
class DailyTargetConfig:
    """Daily target and compound profit management."""
    daily_target_pct: float = 2.0            # Default 2% per day (max 10%)
    daily_loss_limit_pct: float = 50.0       # Max loss as % of daily target amount
    aggressive_mode_enabled: bool = True     # Allow AGGRESSIVE mode
    aggressive_trigger_pct: float = 20.0     # Trigger if <20% of target achieved
    aggressive_time_trigger: float = 0.6     # AND >60% of trading day elapsed
    protecting_trigger_pct: float = 80.0     # Trigger PROTECTING at 80% of target
    protecting_confidence_min: float = 0.70  # Min confidence in PROTECTING mode
    protecting_size_mult: float = 0.60       # Reduce position size by 40% in PROTECTING
    protecting_trailing_tighten: float = 0.80  # Tighten trailing stops by 20%
    aggressive_confidence_min: float = 0.50  # Lower threshold in AGGRESSIVE mode
    aggressive_max_positions: int = 3        # Allow 3 concurrent in AGGRESSIVE
    auto_target_reduction: bool = True       # Auto-reduce target after consecutive misses
    miss_reduce_days: int = 3               # Reduce target after 3 missed days
    miss_severe_days: int = 5               # Reduce to 1% after 5 missed days
    miss_reduce_pct: float = 20.0           # Reduce target by 20% on miss streak
    restore_streak_days: int = 5            # Offer restore after 5 hits at reduced target
    compound_from: str = "tracked_equity"    # "wallet_balance" or "tracked_equity"


@dataclass
class AILearningConfig:
    """Configuration for the AI knowledge ingestion pipeline."""
    enabled: bool = True
    db_path: str = ""  # Empty = use default (ai_knowledge.db alongside bot_data.db)

    # YouTube ingestion
    youtube_channels: list = field(default_factory=lambda: [
        # Curated crypto swing trading / TA education channels (IDs)
    ])
    youtube_max_age_days: int = 180          # Ignore videos older than 6 months
    youtube_clickbait_patterns: list = field(default_factory=lambda: [
        r"(?i)\b(100|1000)x\b", r"(?i)guaranteed", r"(?i)get rich",
        r"(?i)millionaire overnight", r"(?i)free money",
    ])
    youtube_chunk_words: int = 500           # Words per transcript chunk
    youtube_chunk_overlap: int = 50          # Overlap between chunks
    youtube_weekly_check: bool = True        # Auto-check channels weekly

    # Paper / blog ingestion
    monitored_urls: list = field(default_factory=list)   # Blog/substack URLs to monitor
    arxiv_query: str = "cryptocurrency trading"          # arXiv search query
    arxiv_max_results: int = 10                          # Papers per sync
    paper_chunk_words: int = 500
    paper_chunk_overlap: int = 50

    # Knowledge extraction (LLM)
    llm_provider: str = "anthropic"          # "anthropic" or "openai"
    llm_model: str = "claude-sonnet-4-5-20250514"  # Model for extraction
    min_confidence: float = 0.5              # Discard extractions below this
    dedup_similarity_threshold: float = 0.85 # Cosine sim threshold for dedup

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    embedding_dim: int = 384                     # Dimension for all-MiniLM-L6-v2

    # Rate limits
    llm_requests_per_minute: int = 30
    llm_retry_max: int = 3
    llm_retry_delay: float = 2.0               # Base delay for exponential backoff


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
    daily_target: DailyTargetConfig = field(default_factory=DailyTargetConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    reentry: ReentryConfig = field(default_factory=ReentryConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    ai_learning: AILearningConfig = field(default_factory=AILearningConfig)
    log_level: str = "INFO"


CONFIG = BotConfig()
