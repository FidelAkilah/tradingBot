# Product Requirements Document (PRD)
# Binance Futures Swing Trading Bot

**Goal**: Improve win rate (WR) of an automated swing trading bot on Binance USDT-M Futures. Current WR is low. This document describes the full system so an AI can analyze weaknesses and suggest improvements.

---

## 1. System Overview

| Attribute | Value |
|-----------|-------|
| **Exchange** | Binance USDT-M Futures (via ccxt / ccxt.pro) |
| **Strategy Type** | Multi-timeframe candle-driven swing trading with order book confirmation |
| **Hold Duration** | 30 minutes to 8 hours |
| **Leverage** | 30x (CROSSED margin, ONE-WAY position mode) |
| **Pairs** | BTC/USDT, ETH/USDT, ADA/USDT, HYPE/USDT, SOL/USDT |
| **Capital** | ~$61 USD (IDR 1,000,000) |
| **Max Positions** | 2 concurrent |
| **Mode** | Shadow (paper) trading by default; live trading available |
| **Stack** | Python (FastAPI + asyncio), Next.js dashboard, SQLite |

---

## 2. Entry Signal Logic

### 2.1 Primary Signal: Multi-Timeframe Candle Analysis

**Timeframes used**: 1-hour and 4-hour candles (50 candle lookback each).

**Indicators computed per timeframe**:

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| Fast EMA | Period 9 | Short-term trend |
| Slow EMA | Period 21 | Medium-term trend |
| Trend EMA | Period 50 | Long-term direction filter |
| RSI | Period 14, OB=70, OS=30 | Momentum / exhaustion |
| ATR | Period 14 | Volatility for dynamic TP/SL |
| Volume Ratio | Current / 20-candle avg, surge=1.5x | Volume confirmation |

**Trend Classification Logic**:

```
STRONG_BULLISH: price > fast_ema > slow_ema > trend_ema AND ema_slope > 0.1%
BULLISH:        fast_ema > slow_ema AND price > slow_ema AND ema_slope > 0.1%
NEUTRAL:        no clear crossover or slope < 0.1%
BEARISH:        fast_ema < slow_ema AND price < slow_ema AND ema_slope < -0.1%
STRONG_BEARISH: price < fast_ema < slow_ema < trend_ema AND ema_slope < -0.1%
```

**Multi-Timeframe Alignment**:
- Primary trend comes from the 4h candle.
- 1h candle provides secondary validation.
- If both timeframes agree on direction: `trend_aligned = True` (+0.3 confidence).
- If 4h has a trend but 1h is neutral: +0.1 confidence.
- If they contradict each other: confidence is reduced (no bonus).

### 2.2 Composite Confidence Scoring

The bot computes a confidence score (0.0 to 1.0) by summing:

| Component | Score Contribution |
|-----------|-------------------|
| Base candle trend (non-neutral) | +0.30 |
| Both timeframes aligned | +0.30 |
| 1h neutral but not contradicting 4h | +0.10 |
| RSI context (buying oversold dip in uptrend, selling overbought rally in downtrend) | +0.20 |
| Strong trend (STRONG_BULLISH or STRONG_BEARISH) | +0.10 |
| Volume surge (>1.5x average) | +0.10 |

**Maximum possible score**: 1.0
**Minimum confidence to open a trade**: 0.30 (30%)

### 2.3 RSI Hard Blocks

Trades are blocked at extreme RSI levels only:
- Block SELL if: 1h RSI < 20 AND 4h RSI < 25 (already deeply oversold)
- Block BUY if: 1h RSI > 80 AND 4h RSI > 75 (already deeply overbought)

### 2.4 Order Book Confirmation (Secondary)

Order book data is used as **confirmation only**, not as the primary signal:

| Feature | Parameter | Purpose |
|---------|-----------|---------|
| Whale wall detection | Min $100k, 5x avg size | Find large support/resistance |
| Spoofing detection | 60s persistence window, 60% cancel rate, 5+ flickers | Filter fake walls |
| Imbalance ratio | Top 50 levels, threshold 1.5x | Bid/ask pressure |
| VWAP | 1000-trade lookback, 0.5% deviation | Fair value reference |
| VPIN | 50 buckets, toxicity > 0.85, block > 0.99 | Market toxicity detection |

**Entry price**: Mid-price (not wall price). Walls confirm, not dictate entry.

### 2.5 VPIN Regime Filtering

| VPIN Level | Regime | Action |
|------------|--------|--------|
| < 0.50 | SAFE | Normal trading |
| 0.50 - 0.85 | CAUTION | Normal trading |
| 0.85 - 0.99 | TOXIC | Widen stops by 1.3x |
| > 0.99 | EXTREME | Block all new entries |

---

## 3. Exit Signal Logic

### 3.1 Take Profit (TP)

- **Calculation**: ATR-based. TP distance = ATR * 2.0 multiplier (scales to 2.5x when trend is aligned).
- **Fallback**: 2.0% if ATR unavailable.
- **Constraint**: Only triggers after min_hold_time (30 minutes).
- **Typical TP %**: 1.5% - 3.0% depending on volatility.

### 3.2 Stop Loss (SL)

- **Calculation**: ATR-based. SL distance = ATR * 1.0 multiplier (scales to 0.8x when trend is aligned).
- **Fallback**: 1.0% if ATR unavailable.
- **No hold time constraint**: Triggers immediately.
- **Typical SL %**: 0.8% - 1.2%.

### 3.3 Risk-Reward Ratio

- **Designed ratio**: ~2:1 to 3.1:1 (TP is 2x-2.5x ATR, SL is 0.8x-1.0x ATR).
- **With 30x leverage**: A 1% adverse move = ~30% margin loss.

### 3.4 Trailing Stop

- **Activation**: After min_hold_time (30 min) AND price reaches 50% of TP distance.
- **Trail amount**: 0.8% from peak (for longs) or trough (for shorts).
- **Effect**: Locks in partial gains if price reverses after moving favorably.

### 3.5 Max Hold Time

- **8 hours maximum**: Force-closes any position held longer.
- **Purpose**: Prevent overnight/weekend exposure.

### 3.6 Wall Pull Exit

- **Disabled** (`exit_on_wall_pull = False`): Since walls are only confirmation, their disappearance does not trigger exit.

---

## 4. Position Sizing & Risk Management

### 4.1 Position Sizing

| Parameter | Value |
|-----------|-------|
| Starting capital | ~$61 (IDR 1,000,000) |
| Max position % of equity | 30% |
| Max USD per trade | ~$18.30 (30% of $61) |
| With 30x leverage | ~$549 notional per trade |
| Max open positions | 2 |
| Max total exposure | ~$1,098 notional |

### 4.2 Daily Risk Limits

| Limit | Value |
|-------|-------|
| Max daily loss (USD) | $10 |
| Max daily loss (%) | 10% of equity |
| Max daily trades | 25 |
| Max drawdown from peak | 25% |

### 4.3 Cooldown System

| Trigger | Cooldown |
|---------|----------|
| After any loss | 300 seconds (5 min) before next trade |
| Per symbol after close | 300 seconds before re-entry on same pair |
| Consecutive losses | Position size reduced by 20% per consecutive loss |

### 4.4 Daily Reset & Compounding

- Resets daily loss counter at UTC midnight.
- Compounding enabled: profits increase next day's capital base.
- Daily target: 10%.

---

## 5. Order Execution Flow

### 5.1 Shadow Mode (Current)

1. Candle analyzer fetches 1h and 4h candles for each pair (60s cache cooldown).
2. Computes EMA crossover, RSI, ATR, volume ratio per timeframe.
3. Generates `SwingSignal` with trend, confidence, ATR-based TP/SL.
4. Order book analyzer runs in parallel: whale walls, imbalance, VPIN.
5. Composite score calculated. If confidence >= 0.30 and spread < 0.1%: generate trade suggestion.
6. Shadow trader checks: no existing position on symbol, no cooldown active.
7. Opens simulated trade at mid-price with 100ms latency simulation.
8. Monitors price updates: checks SL immediately, checks TP after 30 min hold.
9. Trailing stop activates at 50% of TP distance.
10. Closes trade when TP/SL/trailing/max-hold triggers. Logs to JSONL + SQLite.

### 5.2 Live Mode

Same flow but sends actual LIMIT orders to Binance via ccxt. Monitors order fill status. Places reduce-only orders for TP/SL.

---

## 6. Data Persistence

### 6.1 SQLite Database (bot_data.db)

**Tables**:
- `trades`: Full trade lifecycle (entry, exit, P&L, indicators at entry, exit reason)
- `signals`: Every analysis signal logged (timestamp, symbol, score, trend, VPIN)
- `bot_state`: Key-value store for runtime state

### 6.2 JSONL Logs

- `shadow_trades.jsonl`: All simulated trades with full metadata.
- `shadow_trades_signals.jsonl`: Every signal analyzed (~240MB+), including those that didn't result in trades.

---

## 7. Dashboard (Next.js Frontend)

**Components**:
- StatusBar: Bot uptime, equity, mode
- EquityCard: Current balance and drawdown
- PerformanceStats: Win rate, profit factor, trade counts
- PnlChart: Cumulative P&L timeseries
- PositionsPanel: Open positions with unrealized P&L
- TradeHistory: Closed trades table
- SignalGauges: Real-time trend/confidence visualization

**Polling intervals**: 2-5s for status, 10s for performance stats.

---

## 8. Configuration Summary

### Key Tunable Parameters

```
# Candle Analysis
timeframes = ["1h", "4h"]
fast_ema = 9, slow_ema = 21, trend_ema = 50
rsi_period = 14, overbought = 70, oversold = 30
atr_period = 14
atr_tp_multiplier = 2.0, atr_sl_multiplier = 1.0
volume_surge_multiplier = 1.5
min_confidence_threshold = 0.30

# Risk
leverage = 30x
max_position_pct = 30%
max_open_positions = 2
stop_loss_pct = 1.0% (fallback)
take_profit_pct = 2.0% (fallback)
trailing_stop_pct = 0.8%
min_hold_time = 30 minutes
max_hold_time = 8 hours
cooldown_after_loss = 5 minutes
max_daily_loss = $10 / 10%
max_drawdown = 25%

# Order Book
min_wall_usd = $100,000
whale_multiplier = 5x
spoof_persistence = 60s
vpin_block_above = 0.99
vpin_widen_stops_above = 0.85

# Execution
order_type = LIMIT
spread_filter = 0.1%
latency_simulation = 100ms
```

---

## 9. Known Weaknesses / Areas to Investigate for Low Win Rate

### 9.1 Strategy Concerns

1. **Very low confidence threshold (0.30)**: The bot takes trades with only 30% confidence. A base trend signal alone (0.30) can trigger a trade without any confirmation from timeframe alignment, RSI context, or volume.

2. **EMA crossover as sole trend indicator**: No ADX filter to confirm trend strength. The `min_adx_equivalent = 20.0` is defined but there is no actual ADX calculation in the code — only EMAs and slope.

3. **No price action or candlestick pattern analysis**: The bot doesn't look at candle shapes (engulfing, pin bars, doji, etc.) — only trend direction from EMAs.

4. **RSI used only as a hard block at extremes**: RSI 20-80 range provides no signal filtering. RSI divergences are not detected. The RSI "context" bonus (+0.20) rewards buying when RSI is low in an uptrend, but doesn't distinguish between a healthy pullback and trend exhaustion.

5. **No support/resistance levels**: The bot doesn't identify horizontal S/R, Fibonacci levels, or previous swing highs/lows for entry/exit optimization.

6. **1h and 4h only**: No higher timeframe context (daily/weekly) for overall market structure. No lower timeframe (15m/5m) for precise entries.

7. **Volume confirmation is weak**: Only checks if current volume > 1.5x 20-candle average. Doesn't analyze volume profile, buying/selling volume breakdown, or OBV.

### 9.2 Risk/Sizing Concerns

8. **30x leverage with swing trading**: Extremely high leverage for positions held 30 min to 8 hours. A 1% stop loss = 30% capital loss. Even with 30% position sizing, each losing trade costs ~9% of equity.

9. **Small capital ($61) + high leverage = wide effective spread**: The actual fill quality and fees eat significantly into profits at this size.

10. **2:1 RR requires >33% WR to break even, but fees aren't factored**: With Binance futures taker fees (0.04%) * 30x leverage = 1.2% per round trip. This alone consumes 60% of a 2% TP target.

11. **Trailing stop (0.8%) may be too tight for swing trading at 30x**: Small pullbacks within a trend will frequently stop out winners.

### 9.3 Execution Concerns

12. **LIMIT orders with GTC in a swing strategy**: If the market moves, the order may never fill or fill at a suboptimal price.

13. **60-second candle cache cooldown**: In fast markets, the bot may trade on stale candle data.

14. **No re-entry logic**: After a stop loss, the bot cools down 5 minutes then may not re-enter even if the trend resumes.

15. **No partial profit taking**: It's all-or-nothing — no scaling out at partial targets.

### 9.4 Market Regime Concerns

16. **No market regime detection beyond VPIN**: The bot doesn't identify ranging vs trending markets. EMA crossovers generate many false signals in sideways/choppy markets.

17. **No correlation/macro filters**: Doesn't account for BTC dominance, funding rates, open interest trends, or broader market sentiment.

18. **No session-based filtering**: `preferred_session = "any"` means the bot trades during low-liquidity periods where signals are less reliable.

---

## 10. Performance Data Available

The bot logs extensive data that can be analyzed:

- **shadow_trades.jsonl**: Every simulated trade with entry/exit prices, P&L, indicators, confidence scores, exit reasons, durations.
- **shadow_trades_signals.jsonl**: Every signal analyzed (including non-trades) with all indicator values.
- **SQLite trades table**: Queryable trade history with all metrics.
- **SQLite signals table**: Queryable signal history.

**Key metrics to review**:
- Win rate by exit reason (TP vs SL vs trailing vs max_hold)
- Win rate by confidence score range
- Win rate by pair
- Win rate by trend alignment (aligned vs not)
- Average winner vs average loser (actual RR achieved)
- Trade duration distribution
- P&L vs time of day
- Consecutive loss streaks

---

## 11. What I Need From You

**Primary goal**: Improve the win rate of this trading bot while maintaining or improving the risk-reward ratio.

**Specific requests**:

1. **Analyze the strategy weaknesses** listed in Section 9 and prioritize which ones are most likely causing the low WR.

2. **Suggest concrete parameter changes** (e.g., raise confidence threshold, adjust EMA periods, modify ATR multipliers).

3. **Suggest additional indicators or filters** that could improve signal quality without over-fitting (e.g., ADX, Bollinger Bands, MACD, market structure, candlestick patterns).

4. **Recommend leverage and position sizing changes** appropriate for $61 capital and swing trading.

5. **Suggest market regime filtering** to avoid trading during choppy/ranging conditions.

6. **Propose an improved exit strategy** (partial TP, better trailing stop logic, dynamic SL adjustment).

7. **Recommend a backtesting approach** to validate changes before live trading.

8. **Provide code-level changes** where possible (Python, using NumPy for indicators).

**Constraints**:
- Must work on Binance USDT-M Futures.
- Capital is limited (~$61 USD).
- Bot is Python/asyncio-based.
- Technical indicators are computed with NumPy (no TA-Lib dependency).
- Must remain a swing trading strategy (not scalping or HFT).
