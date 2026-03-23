# CLAUDE.md — Binance USDT-M Futures Swing Trading Bot

## What This Is
IDR-denominated swing trading bot for Binance USDT-M Futures. Multi-timeframe candle analysis (1d/4h/1h/15m) drives primary signals; candlestick patterns, RSI divergences, S/R levels, order book liquidity, enhanced volume analysis (OBV/buy-sell/volume profile), funding rate filtering, open interest analysis, MACD momentum confirmation, BB squeeze + Keltner Channel breakout detection, cross-pair correlation guard with portfolio exposure management, and dynamic pair scanning with opportunity scoring filter and confirm entries. Shadow (paper) mode by default, live mode with `--live`. Daily compound profit targeting with 4 trading modes. Partial take-profit scale-out with Chandelier trailing stop and smart re-entry after stop-outs. Opportunity scanner rotates to highest-scoring pairs every 2 hours with per-pair performance tracking.

## Quick Commands
```bash
python -m pytest tests/ -v          # Run all tests (608 total)
python main.py                      # Shadow mode
python main.py --live               # Live on testnet
python server.py                    # FastAPI + bot (dashboard at :3000)
bash start.sh                       # Full stack (server + Next.js dashboard)
```

## Architecture Overview
```
main.py (ScalpingBot orchestrator)
├── websocket_client.py      → OrderBookStream (ccxt.pro WebSocket)
├── candle_analyzer.py       → SwingSignal (EMA/RSI/ATR/ADX/MACD on 1h+4h, daily gate, 15m entry, vol/funding/OI, BB squeeze+KC)
├── candle_patterns.py       → PatternDetector (engulfing/pin bar/doji/star/soldiers/marubozu)
├── divergence.py            → DivergenceDetector (RSI regular+hidden divergence)
├── levels.py                → LevelDetector (S/R pivot points + Fibonacci retracements)
├── liquidity_analyzer.py    → AnalysisResult (walls/imbalance/VWAP/VPIN)
├── volume_analysis.py       → VolumeAnalyzer + FundingRateAnalyzer + OpenInterestAnalyzer
├── market_regime.py         → RegimeResult (ADX/BB/EMA 2-of-3 vote)
├── session_filter.py        → SessionResult (UTC window gating)
├── shadow_trader.py         → ShadowTrader (partial TP, Chandelier exit, dynamic SL, JSONL logs)
├── correlation.py           → CorrelationMatrix + CorrelationGuard + PortfolioHeatMap
├── reentry.py               → ReentryManager (smart re-entry after stop-outs)
├── order_manager.py         → OrderManager (live Binance execution)
├── position_sizer.py        → PositionSizer (Kelly + drawdown + target-aware)
├── risk_manager.py          → RiskManager (daily loss/drawdown/cooldown)
├── daily_target/            → DailyTargetTracker + ModeController + Compounder
├── scanner/                 → OpportunityScanner + PairSelector + PairPerformanceTracker
├── database.py              → SQLite (trades, signals, daily_equity, bot_state)
└── config.py                → BotConfig (all dataclass configs incl. VolumeConfig, ExitConfig, ReentryConfig, CorrelationConfig, ScannerConfig)

server.py                   → FastAPI REST API wrapping ScalpingBot
dashboard/                   → Next.js frontend (proxies to :8000)
```

## File Index

### Core Trading (root)
| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~820 | Bot orchestrator, CLI entry, trading loop (`ScalpingBot`), re-entry integration, funding rate/OI fetching, correlation matrix updates, periodic pair scanning |
| `config.py` | ~520 | All config dataclasses: `BotConfig`, `FuturesConfig`, `TradingConfig`, `RiskConfig`, `DailyTargetConfig`, `CandleConfig` (incl. MACD + squeeze), `VolumeConfig`, `ExitConfig`, `ReentryConfig`, `CorrelationConfig`, `ScannerConfig` |
| `candle_analyzer.py` | ~1400 | Multi-TF swing signals. `CandleAnalyzer` → `SwingSignal` (EMA crossover, RSI, ATR TP/SL, ADX filter, MACD momentum, BB squeeze + Keltner Channel, daily trend gate, 15m entry timing, pattern + divergence + volume/funding/OI integration, S/R levels) |
| `candle_patterns.py` | ~350 | Candlestick pattern recognition. `PatternDetector` → `PatternScanResult`. Detects: engulfing, pin bar (hammer/shooting star), doji, morning/evening star, three soldiers/crows, marubozu. `evaluate_patterns_for_signal()` for confidence adjustments |
| `divergence.py` | ~340 | RSI divergence detection. `DivergenceDetector` → `DivergenceScanResult`. Regular divergence (reversal), hidden divergence (continuation). Swing pivot analysis on price + RSI series. `evaluate_divergence_for_signal()` for confidence/block |
| `levels.py` | ~330 | Support/Resistance detection. `LevelDetector` → `LevelAnalysis`. Pivot points (swing highs/lows), Fibonacci retracements (0.382/0.5/0.618/0.786), level clustering with touch counting, TP-block check |
| `liquidity_analyzer.py` | ~850 | Order book intelligence. `LiquidityAnalyzer` → `AnalysisResult` (whale walls, imbalance, VWAP, VPIN, composite score) |
| `volume_analysis.py` | ~430 | Enhanced volume (OBV + buy/sell estimation + volume profile), funding rate filter, open interest analysis. `VolumeAnalyzer`, `FundingRateAnalyzer`, `OpenInterestAnalyzer` |
| `market_regime.py` | ~350 | Regime classification (STRONG_TREND/TREND/WEAK/RANGING). 2-of-3 vote: ADX + BB Width + EMA alignment |
| `session_filter.py` | ~130 | UTC trading windows. Overlap 13-17→100%, EU 7-13→80%, US 17-21→80%, Asian 0-7→50% BTC/ETH only, Dead 21-0→block |
| `vpin_analyzer.py` | ~490 | Flow toxicity (VPIN). Blocks entries >0.85, widens stops >0.65. `VPINAnalyzer` → `VPINState` |
| `position_sizer.py` | ~310 | Kelly Criterion + confidence + drawdown + intraday DD + target-aware leverage (5x-20x interpolated). Key class: `PositionSizer`, `SizingResult` |
| `risk_manager.py` | ~270 | Global risk controls: daily loss limit, drawdown halt (25%), cooldown. Key class: `RiskManager`, `TradeRecord` |
| `correlation.py` | ~500 | Cross-pair correlation guard + portfolio exposure. `CorrelationMatrix` (rolling 30-day Pearson on log returns), `CorrelationGuard` (block/reduce correlated same-direction), `PortfolioHeatMap` (directional exposure limits) |
| `shadow_trader.py` | ~900 | Paper trading sim with partial TP scale-out, Chandelier Exit, dynamic SL, correlation size mult. `ShadowTrader`, `ShadowTrade` (80+ fields incl. volume/funding/OI), `TPLevel` |
| `reentry.py` | ~170 | Smart re-entry after stop-outs. `ReentryManager`, `ReentryCandidate`. Checks trend validity, ADX, confidence, daily loss limit |
| `order_manager.py` | ~530 | Live order execution. `OrderManager`, `ManagedOrder`, `Position` |
| `websocket_client.py` | ~230 | ccxt.pro WebSocket streaming. `OrderBookStream` |
| `database.py` | ~650 | SQLite: `trades` (with partial TP + re-entry + volume/funding/OI columns), `signals` (with volume/funding/OI), `daily_equity`, `bot_state` tables. WAL mode |
| `server.py` | ~600 | FastAPI REST API. `BotRunner` lifecycle. Endpoints below (incl. correlation + exposure + scanner) |
| `analyze_performance.py` | ~370 | Post-mortem JSONL analysis tool |

### Daily Target Module (`daily_target/`)
| File | Purpose |
|------|---------|
| `tracker.py` | `DailyTargetTracker` — state tracking, loss limits, integration hooks. `DailyTargetState` (20 fields), `DailyTargetContext` (sizer input), `TradingMode` enum |
| `mode_controller.py` | `ModeController` — mode transitions: NORMAL→AGGRESSIVE→PROTECTING→HALTED with hysteresis (enter PROTECTING at 80%, exit at 60%) |
| `compounder.py` | `Compounder` — UTC 00:00 daily reset, streak tracking, auto-target-reduction after 3 missed days |
| `__init__.py` | Re-exports: `DailyTargetTracker`, `DailyTargetState`, `DailyTargetContext`, `TradingMode`, `ModeController`, `Compounder` |

### Scanner Module (`scanner/`)
| File | Lines | Purpose |
|------|-------|---------|
| `pair_scanner.py` | ~575 | `OpportunityScanner` (scan top 30 pairs, ADX/ATR/BB squeeze/volume/funding scoring), `PairScore` (20+ fields), `ScanResult` (anchors + selected + retained), `PairSelector` (filters, top-N selection, auto-include, open-position retention, transition tracking) |
| `pair_performance.py` | ~215 | `PairPerformanceTracker` (per-pair win rate, P&L, profit factor, contribution %), `PairStats` (auto-disable <35% WR, auto-include >60% WR). Real-time `record_trade()` + batch `update_from_trades()` |
| `__init__.py` | Re-exports: `OpportunityScanner`, `PairScore`, `PairSelector`, `ScanResult`, `PairPerformanceTracker` |

### Tests (`tests/`)
| File | Tests | Coverage |
|------|-------|---------|
| `test_position_sizer.py` | 68 | Kelly, confidence mult, leverage interpolation, overall+intraday drawdown, consec loss, target progress mult, per-symbol halt, full calculate() with DailyTargetContext |
| `test_daily_target.py` | 72 | Tracker state, hooks (14), loss limits (4), mode transitions (13), compounder (7), auto-reduction (7), daily reset (4), context (7), force_target (3) |
| `test_exit_strategy.py` | 44 | Partial TP (TP1/TP2/TP3 execution, SL moves, PnL accumulation), Chandelier Exit (activation, trailing, tightening), dynamic SL (momentum, flat, never-widen), daily-target TP adjustments, edge cases, protecting mode |
| `test_reentry.py` | 27 | Registration, re-entry triggers (happy path), blocking conditions (cooldown, expiry, max re-entries, loss limit, wrong direction, low confidence, low ADX), candidate lifecycle, config getters |
| `test_candle_patterns.py` | 36 | All pattern detectors (engulfing, pin bar, doji, morning/evening star, soldiers/crows, marubozu), scan aggregation, evaluate_patterns_for_signal integration |
| `test_divergence.py` | 30 | RSI series, swing detection, regular/hidden divergence, strength calculation, evaluate_divergence_for_signal (confirm/block/stack), edge cases |
| `test_levels.py` | 24 | Pivot detection, level clustering, Fibonacci retracements, full analyze pipeline (classification, proximity, TP-block, caching) |
| `test_multi_timeframe.py` | 15 | Daily trend gate (block/align/penalty), 15m entry timing (pullback/timeout), S/R integration, SwingSignal field defaults |
| `test_volume_analysis.py` | 47 | OBV (trend, EMA, divergence), buy/sell volume (pressure, CLV, doji), volume profile (POC, VAH/VAL, flat range), funding rate (extreme, persistent, blocking, boundary), open interest (conviction matrix, exhaustion, divergence, pruning), edge cases (zero volume, large numbers, multi-symbol) |
| `test_market_regime.py` | — | ADX/BB/EMA regime voting, session filter |
| `test_fee_calculation.py` | — | Post-fee R:R, fee-adjusted TP/SL |
| `test_macd_squeeze.py` | 50 | MACD computation (bullish/bearish/crossover/histogram/disabled), MACD confidence (confirm/diverge/crossover bonus), squeeze detection (active/release BUY/SELL/no volume), Keltner Channel (BB inside/outside KC), squeeze confidence (SL/TP overrides), signal summary, config, edge cases (large/small prices, single TF, min data), integration |
| `test_correlation.py` | 56 | CorrelationMatrix (price updates, log returns, Pearson, serialization, staleness, edge cases), CorrelationGuard (high/medium/low corr, natural hedge, strong signal exception, disabled, multiple positions, boundary values), PortfolioHeatMap (exposure computation, directional bias, breach detection, size capping, summary), integration (full pipeline, 5-pair matrix) |
| `test_scanner.py` | 63 | ADX/ATR/BB squeeze computation (9), scoring normalization (7), PairSelector (filters, top-N, anchor dedup, open-position retention, disabled/auto-include, added/dropped, needs_scan) (18), ScanResult (dedup, ordering) (3), PairPerformanceTracker (record_trade, win rate, auto-disable/include, update_from_trades, profit_factor, lookback, summary, recovery) (15), edge cases (8), config (3) |
| `test_adx.py` | 11 | ADX calculation, integration with SwingSignal |

## Confidence Scoring Pipeline
```
Base confidence:
  Base candle trend (non-neutral):           +0.25
  Both timeframes (1h+4h) aligned:           +0.25
  RSI context confirmation:                  +0.15
  Strong trend (STRONG_BULLISH/BEARISH):     +0.10
  Volume surge (>1.5x average):              +0.10
  ADX filter pass (>25):                     +0.10

Multi-timeframe modifiers:
  Daily trend aligned with 4h:               +0.10
  Daily trend neutral:                       -0.10
  Daily trend opposes signal:                HARD BLOCK

Candlestick pattern modifiers:
  Continuation pattern confirms signal:      +0.10
  Reversal pattern contradicts signal:       -0.15
  Doji on signal candle:                     -0.10

RSI divergence modifiers:
  Regular divergence WITH direction:         +0.15
  Hidden divergence WITH direction:          +0.10
  Regular divergence AGAINST direction:      HARD BLOCK

S/R level modifiers:
  Near support (long) or resistance (short): +0.05
  TP target blocked by strong S/R level:     HARD BLOCK

Volume analysis modifiers:
  OBV trend confirms direction:              +0.05
  OBV divergence (price vs volume):          -0.10
  Buy/sell pressure confirms direction:      +0.05

Funding rate modifiers:
  Extreme positive (>0.05%) + going long:    -0.10 (penalty)
  Extreme negative (<-0.05%) + going short:  -0.10 (penalty)
  Persistent same-sign funding:              +0.05

Open interest modifiers:
  Rising OI + price trending:               +0.05 (strong conviction)
  Falling OI + price trending:              -0.05 (exhaustion)

MACD modifiers:
  MACD confirms direction (histogram growing): +0.05
  Fresh MACD crossover within 3 candles:     +0.05
  MACD diverges from trade direction:        -0.05

BB Squeeze + Keltner Channel modifiers:
  Squeeze release aligned with direction:    +0.10
  Squeeze release SL override:              0.7× ATR (tighter)
  Squeeze release TP override:              3.0× ATR (wider)

Order book (applied in liquidity_analyzer):
  OB imbalance confirms direction:           +0.05

Minimum confidence threshold:                0.55
```

## Entry Gate Sequence
Before a trade is executed, it must pass through these gates in order:
1. ADX ≥ 20 (ranging market block)
2. Market regime ≠ RANGING (2-of-3 vote)
3. Session window allows pair
4. RSI not at extreme (both TFs)
5. Daily trend not counter-signal
6. Candlestick patterns checked (reversal contradiction = penalty, not block)
7. RSI divergence checked (regular against = **HARD BLOCK**)
8. Volume analysis (OBV trend, buy/sell pressure, OBV divergence penalty)
9. Funding rate filter (extreme funding penalizes, persistent confirms)
10. Open interest analysis (conviction/exhaustion adjustments)
11. MACD momentum check (confirm/crossover bonus, diverge penalty)
12. BB Squeeze + Keltner Channel (squeeze release → +0.10 conf, SL/TP overrides)
13. Confidence ≥ 0.55
14. Post-fee R:R ≥ 1.5 (squeeze release widens TP, tightens SL)
15. S/R TP-block check
16. 15m pullback entry timing (or timeout after 3 candles)
17. Smart re-entry check (if stopped out recently, trend still valid)
18. Correlation guard (>0.80 block, 0.60-0.80 reduce 50%, natural hedge allow, strong signal exception)
19. Portfolio exposure check (net long/short ≤ 20× capital, reduce or block if breached)
20. Risk manager `can_trade()` + daily target mode check
21. Position limit + confidence threshold per mode

## Partial Take Profit (Scale-Out)
```
Position is divided into 3 tranches at entry:
  TP1: 40% of position at 1.0× ATR  → secure quick profit, cover fees
  TP2: 35% of position at 2.0× ATR  → main profit target
  TP3: 25% of position (runner)      → no fixed target, trailing stop only

After TP1 hit:
  - SL moves to breakeven (entry price + round-trip fees)
  - Chandelier trailing activates at 1.5× ATR distance

After TP2 hit:
  - SL moves to TP1 price level (guaranteed profit lock)
  - Chandelier trailing tightens to 1.0× ATR distance

TP3 (runner):
  - No fixed target — rides Chandelier trailing stop
  - Captures extended moves beyond TP2
  - Closed by trailing stop, max hold, or wall pull
```

## Daily-Target-Aware Exit Adjustments
```
>80% daily target achieved (PROTECTING):
  - All TP levels compress by 30% (secure the day)
  - Chandelier trailing tightened by protecting_trailing_tighten (0.80)

>100% daily target (bonus territory):
  - Chandelier trailing widened by 1.5× (let runners run with house money)

<30% target + past 18:00 UTC + confidence ≥0.75:
  - TP2/TP3 expanded by 20% (need to catch up on high-quality signals)
```

## Chandelier Exit (ATR Trailing Stop)
```
Replaces fixed 0.8% trailing stop with ATR-based Chandelier Exit:

For longs:  trail = highest_high_since_entry − N × ATR
For shorts: trail = lowest_low_since_entry  + N × ATR

ATR multiplier tightens after each TP hit:
  Before TP1:  2.0× ATR (initial)
  After TP1:   1.5× ATR (tighter)
  After TP2:   1.0× ATR (tightest)

Activation: after price moves 30% toward TP1 (or immediately after TP1 hit)
Floor: never trails below breakeven after TP1 hit
Fallback: uses fixed trailing_stop_pct when chandelier_enabled=False or ATR=0
```

## Dynamic Stop Loss Adjustment
```
First 15 minutes (momentum check):
  If price moves ≥0.5× ATR favorably → tighten SL to 0.7× ATR
  (Momentum confirmed — reduce risk exposure)

After 30 minutes (flat check):
  If price moves <0.2× ATR → tighten SL to 0.5× ATR
  (Reduce drift exposure — trade isn't going anywhere)

Rules:
  - Never widen SL beyond initial level
  - Flat tightening applied only once per trade
  - Works for both longs and shorts
```

## Smart Re-entry (reentry.py)
```
After a stop-loss exit, instead of blanket cooldown:

1. Register the stop-out with trade details
2. Check conditions on each tick:
   - Cooldown elapsed (3 min, not 5 min standard)
   - Original trend still valid (same swing direction)
   - ADX ≥ 25 (still trending)
   - Signal confidence ≥ 0.60
   - Daily loss limit < 50% consumed
3. Re-enter with:
   - 70% of original position size
   - Tighter SL (0.8× ATR)
   - Maximum 1 re-entry per original signal
   - 1-hour expiry window

Blocking:
  - Daily loss limit > 50% consumed → no re-entries for the day
  - Trend direction changed → normal cooldown, no re-entry
  - ADX dropped below 25 → no re-entry (trend weakened)
```

## Key Config Values (config.py defaults)
```
Leverage:       default 10x, max 20x (hard cap)
Position size:  5%-25% of equity (Kelly cap), default 15% pre-Kelly
Kelly:          half-Kelly, 20-trade minimum, 50-trade lookback
Daily target:   2% (max 10%), loss limit = 50% of target amount
Drawdown:       0-5% normal, 5-10% -30%, 10-15% -60%, 15-25% min, >25% halt
Intraday DD:    0-3% normal, 3-5% -30%, 5-7% -60%, ≥7% halt
Fees:           0.04% per side (Binance futures taker)
Min R:R:        1.5:1 post-fee
Hold time:      min 30min, max 8hr
Pairs:          BTC/USDT, ETH/USDT, ADA/USDT, HYPE/USDT, SOL/USDT
Max positions:  2 (NORMAL), 3 (AGGRESSIVE)
Capital:        IDR 1,000,000 ≈ $61 (at 16,300 IDR/USD)
Patterns:       doji body <10%, pin wick >2x body, marubozu body >90%
Divergence:     20-candle lookback, 3-candle pivot, RSI period 14
S/R:            5-candle pivot lookback, 0.3×ATR proximity, 2+ touch strength
Partial TP:     TP1=1.0×ATR(40%), TP2=2.0×ATR(35%), TP3=runner(25%)
Chandelier:     2.0×ATR initial, 1.5× after TP1, 1.0× after TP2
Dynamic SL:     momentum 0.7×ATR@15min, flat 0.5×ATR@30min
Re-entry:       70% size, 0.8×ATR SL, 3min cooldown, 1hr window, max 1
Volume:         OBV EMA 10, buy/sell lookback 10, profile 30 candles/20 bins
Funding:        extreme ±0.05%, persistent 3 periods, 5min TTL
Open Interest:  rising >3%, falling <-3%, price move >0.5%, 5min TTL
MACD:           EMA(12)-EMA(26), Signal EMA(9), crossover lookback 3
Squeeze:        BB(20,2σ) inside KC(EMA20 ± 1.5×ATR10), release vol 1.5x, SL 0.7×ATR, TP1 3×ATR
Correlation:    30-day lookback, 24h cache TTL, 15 min candles, high >0.80, medium 0.60-0.80
Corr Guard:     >0.80 block, 0.60-0.80 → 50% size, strong exception >0.75 conf → 1.5× cap
Exposure:       max net long/short 20×, reduce or block if breached
Scanner:        scan top 30 pairs every 2hr, select top 3 dynamic, 0.35s rate limit delay
Scan filters:   24h vol >$50M, spread <0.05%, ADX >22, no unresolved squeeze
Scan weights:   ADX 30%, BB squeeze 20%, volume change 20%, volatility 20%, funding 10%
Pair perf:      20-trade lookback, disable <35% WR (10+ trades), auto-include >60% WR (5+ trades)
Anchor pairs:   BTC/USDT, ETH/USDT (always active)
```

## Candlestick Patterns (candle_patterns.py)
| Pattern | Type | Direction | Detection Logic |
|---------|------|-----------|-----------------|
| Bullish Engulfing | Reversal | Bullish | Current bullish body fully engulfs prior bearish body |
| Bearish Engulfing | Reversal | Bearish | Current bearish body fully engulfs prior bullish body |
| Hammer | Reversal | Bullish | Lower wick ≥2x body, upper wick ≤ body |
| Shooting Star | Reversal | Bearish | Upper wick ≥2x body, lower wick ≤ body |
| Doji | Reversal | Neutral | Body < 10% of range (indecision) |
| Morning Star | Reversal | Bullish | 3-candle: bearish + small + bullish closing above midpoint |
| Evening Star | Reversal | Bearish | 3-candle: bullish + small + bearish closing below midpoint |
| Three White Soldiers | Continuation | Bullish | 3 bullish candles, higher closes, opens within prior body |
| Three Black Crows | Continuation | Bearish | 3 bearish candles, lower closes, opens within prior body |
| Bullish Marubozu | Continuation | Bullish | Body > 90% of range, close > open |
| Bearish Marubozu | Continuation | Bearish | Body > 90% of range, close < open |

## RSI Divergence (divergence.py)
| Type | Price Action | RSI Action | Signal | Impact |
|------|-------------|------------|--------|--------|
| Regular Bullish | Lower low | Higher low | Reversal UP | +0.15 conf if WITH direction, BLOCK if AGAINST |
| Regular Bearish | Higher high | Lower high | Reversal DOWN | +0.15 conf if WITH direction, BLOCK if AGAINST |
| Hidden Bullish | Higher low | Lower low | Continuation UP | +0.10 conf if WITH direction |
| Hidden Bearish | Lower high | Higher high | Continuation DOWN | +0.10 conf if WITH direction |

## Enhanced Volume Analysis (volume_analysis.py)
| Component | Method | Signal | Impact |
|-----------|--------|--------|--------|
| OBV Trend | OBV vs 10-period EMA | Bullish/Bearish | +0.05 when aligned with trade direction |
| OBV Divergence | Price new high but OBV peaked earlier | Bearish divergence | -0.10 penalty |
| Buy/Sell Volume | Close Location Value (CLV) | Buying/Selling pressure | +0.05 when aligned |
| Volume Profile | POC (Point of Control) | Price vs high-volume zone | Informational (logged) |
| Volume Profile | VAH/VAL (Value Area) | 70% of volume range | Informational (logged) |

## Funding Rate Filter
| Condition | Signal | Impact |
|-----------|--------|--------|
| Rate ≥ +0.05% | Extreme positive (crowded longs) | -0.10 penalty on BUY, `block_long` |
| Rate ≤ -0.05% | Extreme negative (crowded shorts) | -0.10 penalty on SELL, `block_short` |
| Same sign for 3+ periods | Persistent funding | +0.05 trend confirmation |

## Open Interest Analysis
| OI Change | Price Move | Conviction | Impact |
|-----------|-----------|------------|--------|
| Rising (>+3%) | Up or Down (>0.5%) | Strong — new money entering | +0.05 confidence |
| Falling (<-3%) | Up or Down | Exhaustion — money leaving | -0.05 penalty, divergence flag |
| Rising | Flat (<0.5%) | Neutral — anticipation | No adjustment |
| Flat | Any | Neutral | No adjustment |

## MACD Momentum Confirmation (candle_analyzer.py)
| Condition | Signal | Impact |
|-----------|--------|--------|
| MACD > Signal AND histogram positive+growing | LONG confirmation | +0.05 confidence |
| MACD < Signal AND histogram negative+shrinking | SHORT confirmation | +0.05 confidence |
| MACD/Signal crossover within 3 candles | Fresh crossover | +0.05 additional |
| MACD diverges from trade direction | Opposing momentum | -0.05 penalty |

MACD computed on both 1h and 4h timeframes. Confirmation requires at least one TF to agree. Divergence penalty only if no TF confirms.

## BB Squeeze + Keltner Channel (candle_analyzer.py)
| Condition | Detection | Impact |
|-----------|-----------|--------|
| BB inside KC | BB_upper < KC_upper AND BB_lower > KC_lower | squeeze_active=True (informational) |
| Squeeze release | Was squeezing → BB outside KC + price breaks BB + volume ≥1.5x avg | +0.10 confidence |
| Release direction | Price > BB_upper → BUY, Price < BB_lower → SELL | Must match suggested_side |
| Release TP/SL | Tighter SL (0.7× ATR), wider TP (3× ATR) | Overrides standard ATR TP/SL |

Keltner Channel: EMA(20) ± 1.5 × ATR(10). True squeeze = BB(20, 2σ) inside KC. Checks last 3 candles for prior squeeze state to detect releases.

## Correlation Guard (correlation.py)
| Condition | Action | Impact |
|-----------|--------|--------|
| Correlation > 0.80, same direction | BLOCK trade | Prevents double exposure to same move |
| Correlation 0.60-0.80, same direction | Allow at 50% size | Reduces correlated exposure |
| Correlation < 0.60 | Allow full size | Pairs are sufficiently independent |
| Opposite direction, correlated pair | Always allow | Natural hedge — reduces portfolio risk |
| Both signals > 0.75 conf, high corr | Allow, cap at 1.5× single exposure | Strong signal exception |

Rolling 30-day Pearson correlation on log returns. Updated daily with 24h TTL cache. Persisted to bot_state (pipe-delimited key format). Applied in main.py before `process_signal()`, sets `correlation_size_mult` on ShadowTrader.

## Portfolio Exposure (correlation.py)
| Metric | Computation | Limit |
|--------|-------------|-------|
| Net long exposure | Σ(position_usd × leverage) / equity for longs | ≤ 20× capital |
| Net short exposure | Σ(position_usd × leverage) / equity for shorts | ≤ 20× capital |
| Net exposure | net_long - net_short | Informational |
| Gross exposure | net_long + net_short | Informational |
| Directional bias | "long" / "short" / "neutral" (±0.1× threshold) | Informational |

If adding a position would breach limit: remaining capacity returned as fraction, size reduced proportionally. If no remaining capacity: trade blocked entirely.

## Opportunity Scanner (scanner/pair_scanner.py)
| Component | Method | Score Weight |
|-----------|--------|-------------|
| ADX (trending strength) | Wilder-smoothed DX, normalized 0-60 → 0-1 | 30% |
| BB Squeeze Percentile | BB width percentile over rolling lookback, inverted (tight=high) | 20% |
| Volume Change | 24h volume vs 7-day daily average, -50% to +200% → 0-1 | 20% |
| ATR Volatility | ATR/price as %, normalized 0-3% → 0-1 | 20% |
| Funding Rate Extremity | abs(funding)/0.10%, capped at 1.0 | 10% |

Composite score = weighted sum of all sub-scores (0-1 range). Scans top 30 USDT-M futures pairs by 24h quote volume. Rate-limited with configurable delay between ccxt API calls.

## Dynamic Pair Selection (scanner/pair_scanner.py)
| Rule | Behavior |
|------|----------|
| Anchor pairs (BTC, ETH) | Always in active list, never dropped |
| Dynamic selection | Top 3 qualified pairs by opportunity score |
| Volume filter | 24h volume ≥ $50M required |
| Spread filter | Spread ≤ 0.05% required |
| ADX filter | ADX ≥ 22 required (some trend present) |
| Squeeze filter | Squeeze active + no release → disqualified |
| Disabled pairs | Performance-disabled pairs → disqualified |
| Auto-include | >60% WR pairs added if they pass filters (up to 2 extra) |
| Open position retention | Dropped pairs with open positions stay in active list |
| Scan interval | Every 2 hours (configurable), on-demand via API |

## Pair Performance Tracking (scanner/pair_performance.py)
| Metric | Computation | Action |
|--------|-------------|--------|
| Win rate | wins / total_trades × 100 (rolling) | Auto-disable if <35% over 10+ trades |
| Profit factor | gross_profit / gross_loss | Dashboard display, inf → 999.0 |
| Avg P&L | total_pnl / total_trades | Dashboard display |
| Contribution % | pair_pnl / total_bot_pnl × 100 | Dashboard display |
| Auto-disable | WR < 35%, ≥ 10 trades | Pair excluded from selection |
| Auto-include | WR ≥ 60%, ≥ 5 trades | Flagged for auto-include in selector |
| Recovery | WR rises above disable threshold | Disabled flag cleared automatically |

Real-time updates via `record_trade()` after each trade closes. Batch updates via `update_from_db()` on startup. Lookback window: last 20 trades per pair (configurable).

## Trading Modes (4 states)
| Mode | Trigger | Sizing | Confidence Min | Max Positions |
|------|---------|--------|----------------|---------------|
| NORMAL | Default (0-60% target) | 1.0x | 0.55 | 2 |
| AGGRESSIVE | <20% target + >60% day elapsed + trending | 1.3x (conf≥0.70) | 0.50 | 3 |
| PROTECTING | ≥80% target achieved (exits <60%) | 0.6x | 0.70 | 2 |
| HALTED | Loss limit hit or forced | 0.0x (blocked) | — | 0 |

## Position Sizing Pipeline
```
position_pct = kelly_pct
             × confidence_mult     (0.6/0.8/1.0/1.2 by confidence tier)
             × drawdown_mult       (overall DD from peak: 1.0/0.7/0.4/0.0)
             × intraday_dd_mult    (DD from day-open: 1.0/0.7/0.4/halt)
             × consec_loss_mult    (0.7^consecutive_losses)
             × regime_mult         (from market regime)
             × session_mult        (from session filter)
             × target_progress_mult (0.5 target-hit, 0.6 protecting, 1.3 behind+high-conf)

position_usd = clamp(position_pct, 5%, 25%) × equity
leverage     = interpolated 5x-20x by confidence, ×0.6 after target hit
notional     = position_usd × leverage
```

## Dynamic Leverage (interpolated, target-aware)
| Confidence | Before Target Hit | After Target Hit (×0.6) |
|-----------|-------------------|------------------------|
| 0.55-0.65 | 5x → 8x | 3x → 5x |
| 0.65-0.75 | 8x → 12x | 5x → 7x |
| 0.75-0.85 | 12x → 16x | 7x → 10x |
| 0.85+ | 16x → 20x | 10x → 12x |

## API Endpoints (server.py, port 8000)
| Route | Method | Returns |
|-------|--------|---------|
| `/api/status` | GET | Bot status, equity, P&L, open positions, daily target progress, sizer state |
| `/api/trades` | GET | Trade history (paginated, query: limit, offset, symbol) |
| `/api/positions` | GET | Open positions with live unrealized P&L |
| `/api/signals` | GET | Recent analysis signals |
| `/api/performance` | GET | Aggregated stats (win rate, profit factor, by symbol/exit reason) |
| `/api/pnl-chart` | GET | Cumulative P&L timeseries |
| `/api/daily-target` | GET | Daily target progress + 30-day compound projection |
| `/api/daily-equity` | GET | Daily equity history |
| `/api/prices` | GET | Current prices with trend/confidence/RSI/ATR |
| `/api/correlation` | GET | Correlation matrix heat map, stale flag, correlated open positions |
| `/api/exposure` | GET | Portfolio directional exposure, net long/short, breach flags |
| `/api/scanner` | GET | Scanner results: pair scores, selections, active/dropped pairs |
| `/api/scanner/performance` | GET | Per-pair win rate, P&L, profit factor, disabled/auto-include flags |
| `/api/scanner/scan` | POST | Trigger on-demand pair scan |
| `/api/bot/start` | POST | Start the bot |
| `/api/bot/stop` | POST | Stop the bot |
| `/health` | GET | Health check |

## Database Schema (SQLite, WAL mode)
- **trades**: trade_id, symbol, side, entry/exit price, pnl_usd, pnl_pct, exit_reason, leverage, kelly_pct, drawdown_mult, regime, session, post_fee_rr, adx, is_open, tp1_hit/price/pnl, tp2_hit/price/pnl, tp3_hit/price/pnl, original_amount, atr_at_entry, partial_realized_pnl, partial_fees, max_favorable_price, is_reentry, reentry_count, obv_trend, obv_divergence, volume_pressure, buy_volume_ratio, poc_price, funding_rate, funding_extreme, oi_change_pct, oi_conviction
- **signals**: timestamp, symbol, mid_price, composite_score, swing_trend/confidence, vpin, regime, session, adx, obv_trend, obv_divergence, volume_pressure, buy_volume_ratio, funding_rate, funding_extreme, oi_change_pct, oi_conviction
- **daily_equity**: date, open/close_equity, target_pct, actual_pct, target_hit, realized_pnl, wins, losses, streak, miss_streak, mode_at_close
- **bot_state**: key-value store (JSON values)

## Signal Flow
1. **Pair scanning** (2hr) → scan top 30 pairs, score opportunities, select dynamic pairs → update active pair list
2. **WebSocket** (500ms) → order book + trades stream
3. **Candle analysis** (60s) → EMA/RSI/ATR/ADX/MACD on 1h+4h → `SwingSignal` with trend + confidence
4. **Daily candles** (300s TTL) → trend gate blocks counter-trend trades
5. **Candlestick patterns** → scan 1h+4h for reversal/continuation patterns → adjust confidence
6. **RSI divergence** → scan 4h for regular/hidden divergence → adjust confidence or BLOCK
7. **Volume analysis** → OBV trend + buy/sell pressure + volume profile → adjust confidence (±0.05/−0.10)
8. **Funding rate** (300s TTL) → extreme crowding penalty, persistent trend confirmation → adjust confidence
9. **Open interest** (300s TTL) → conviction/exhaustion assessment → adjust confidence (±0.05)
10. **MACD momentum** → confirm/crossover bonus on 1h+4h, diverge penalty → adjust confidence (±0.05)
11. **BB Squeeze + Keltner** → detect squeeze (BB inside KC), release → +0.10 conf, SL/TP overrides
12. **S/R levels** → pivot points + Fibonacci → confidence bonus or TP-block
13. **Liquidity analysis** (per tick) → walls/imbalance/VWAP/VPIN → `AnalysisResult` with composite score
14. **Regime filter** → 2-of-3 vote (ADX/BB/EMA) → block RANGING, allow TREND/STRONG_TREND
15. **Session filter** → UTC window gating → block dead zone, reduce Asian
16. **15m entry timing** → wait for RSI pullback (up to 3 candles) → optimize entry
17. **Smart re-entry** → check if recently stopped-out signal is still valid → re-enter with 70% size
18. **Correlation guard** → check cross-pair correlation with open positions → block/reduce/allow
19. **Portfolio exposure** → check net directional exposure vs 20× capital limit → reduce/block
20. **Risk check** → `can_trade()` → daily loss, drawdown, cooldown, trade count
21. **Daily target check** → mode gating (confidence threshold, position limit, halt)
22. **Position sizing** → Kelly × multiplier chain × correlation_size_mult → `SizingResult` with margin + leverage
23. **Entry** → shadow: `ShadowTrade` with partial TP levels | live: limit order via ccxt
24. **Monitoring** → partial TP scale-out, Chandelier trailing, dynamic SL, VPIN widening, max hold
25. **Exit** → close trade (partial PnL + remaining), record to risk manager + daily target + database + pair performance
26. **Daily reset** (UTC 00:00) → compound equity, update streaks, auto-reduce target if needed

## Key Design Decisions
- **Asymmetric loss limit**: Can gain 2% but max loss is only 1% (50% of target)
- **Intraday DD tighter than overall**: Protects the compound base (3/5/7% vs 5/10/15%)
- **Behind-schedule mode**: Only boosts high-confidence signals (≥0.70) at 1.3x — concentrates capital
- **Target-hit protection**: Both leverage (×0.6) AND size (×0.5) reduce after target achieved
- **Hysteresis on PROTECTING**: Enters at 80% achieved, only exits below 60% — prevents oscillation
- **Per-symbol halt**: 4 consecutive losses on a symbol → 2-hour cooldown (not global halt)
- **Fee-adjusted TP/SL**: ATR-based targets compensate for 0.04% round-trip fees
- **VPIN entry blocking**: Won't enter when informed trading detected (VPIN > 0.99)
- **Daily trend as hard gate**: Blocks counter-trend trades (not just a penalty — a full block)
- **RSI divergence as hard block**: Regular divergence against trade direction = strongest counter-signal, no override
- **Candlestick patterns as soft filter**: Contradicting patterns penalize confidence (-0.15) but don't hard-block, allowing strong signals to still pass
- **15m pullback entry**: Optimizes entry price by waiting for RSI dip/spike with 3-candle timeout
- **Partial TP scale-out**: Secures profits early (TP1 covers fees) while letting runners capture extended moves — prevents "gave back all profits" scenarios
- **SL-to-breakeven after TP1**: Once 40% is banked, the remaining position has zero downside risk on the entry
- **Chandelier Exit vs fixed trailing**: ATR-based trailing adapts to volatility — tight in calm markets, wide in volatile markets
- **Dynamic SL in first 30 min**: Early momentum confirmed → tighter SL; flat price → reduce exposure. Never widens SL beyond initial
- **Smart re-entry vs blanket cooldown**: Only re-enters when trend, ADX, and confidence all confirm — prevents revenge trading while capturing "stopped out then it went to TP" trades
- **Daily-target-aware exits**: PROTECTING mode compresses TPs to secure the day; bonus territory widens trailing (house money); behind-schedule expands targets only for high-confidence signals
- **TP3 runner has no fixed target**: The 25% runner rides trailing only — captures 3×, 5×, even 10× ATR moves on strong trends
- **OBV as volume quality check**: OBV divergence from price is a strong warning signal — price making new highs on declining volume often precedes reversals
- **Buy/sell volume via CLV**: Close Location Value estimates order flow from candles without tick data — compatible with standard OHLCV from ccxt
- **Funding rate as contrarian signal**: Extreme funding rates indicate crowded positioning — penalizes entering the crowded side rather than hard-blocking
- **Persistent funding as confirmation**: Same-sign funding for 3+ periods indicates genuine directional bias, not just short-term noise
- **OI conviction matrix**: Rising OI = new money entering (strong move); Falling OI = money leaving (exhaustion). Treats OI-price divergence as a warning, not a block
- **Volume/funding/OI as soft filters**: All three are confidence adjustments (±0.05/0.10) rather than hard blocks — allows strong signals to still pass while adding nuance
- **MACD as momentum confirmation**: MACD direction + histogram momentum provides additional confirmation layer. Fresh crossovers (within 3 candles) get extra bonus — catches early momentum shifts
- **MACD divergence as soft penalty**: MACD opposing the trade direction is -0.05 penalty, not a hard block — allows strong signals with many other confirmations to still pass
- **MACD per-timeframe**: Both 1h and 4h MACD are checked independently. At least one TF confirming is sufficient for the bonus — acknowledges timeframes can disagree
- **Keltner Channel squeeze over BB percentile**: BB inside KC is a more precise squeeze definition than BB width percentile alone — KC adapts to actual volatility (ATR-based) while BB only measures price standard deviation
- **Squeeze release requires volume confirmation**: BB expanding outside KC without volume is not a valid breakout — prevents false signals from low-liquidity band expansions
- **Squeeze release TP/SL overrides**: Tighter SL (0.7×ATR) reduces risk on explosive moves; wider TP (3×ATR) captures the full squeeze release momentum — these moves tend to be fast and extended
- **Correlation guard as pre-entry filter**: Checked after 15m entry timing but before position sizing — blocking correlated same-direction trades prevents effectively doubling exposure to correlated assets (e.g., long ETH + long SOL when both track BTC)
- **Natural hedge always allowed**: Opposite-direction trades on correlated pairs are inherently risk-reducing — long BTC + short ETH at 0.90 correlation is a spread trade, not double exposure
- **Strong signal exception at 1.5× cap**: When both signals are genuinely strong (>0.75 confidence), the correlation guard relaxes from blocking to capping at 1.5× single position — lets the bot capitalize on rare high-conviction setups while still limiting total correlated exposure
- **Medium correlation reduces, not blocks**: 0.60-0.80 correlation is meaningful but not extreme — reducing to 50% size is more profitable long-term than blocking potentially good trades
- **Correlation size mult applied in shadow_trader**: `correlation_size_mult` multiplies `notional_usd` in `_open_trade()`, reducing the actual position size. Reset to 1.0 before each signal processing to prevent stale state
- **Portfolio exposure as separate guard**: Correlation checks pair-level risk; portfolio exposure checks aggregate directional risk — a portfolio could pass all pair-level checks but still be dangerously concentrated in one direction
- **20× capital net exposure limit**: With 10× leverage and 2 positions at 15% equity each, max normal exposure ≈ 3×. The 20× limit is a safety net for edge cases (aggressive mode, re-entries) not a normal operating constraint
- **Pearson on log returns, not prices**: Log returns normalize magnitude differences between BTC ($50k) and ADA ($0.50), giving meaningful correlation coefficients. Raw price correlation would be dominated by scale
- **Scanner as periodic rotation, not real-time**: 2-hour scan interval balances opportunity detection with API rate limits — market structure doesn't change fast enough to justify more frequent scans of 30 pairs
- **Anchor pairs always active**: BTC and ETH provide consistent liquidity and are always tradeable — removing them risks missing the most liquid opportunities
- **Top 3 dynamic selection**: Limits the active universe to 5 pairs (2 anchors + 3 dynamic) — enough diversity for opportunity capture without spreading thin on position management
- **ADX in scanner vs entry gate**: Scanner ADX threshold (22) is lower than entry gate ADX (25) — scanner identifies "some trend present" for opportunity, entry gate confirms "strong enough to trade"
- **BB width percentile inverted for scoring**: Low percentile (tight bands) = high squeeze potential = high opportunity score — pairs about to break out of compression are the best opportunities
- **Rate limit delay configurable**: 0.35s between API calls × ~4 calls per pair × 30 pairs = ~42s total scan time — safe margin below Binance rate limits while keeping scans under 1 minute
- **Open position retention on drop**: When a pair is dropped from selection but has an open position, it stays active until exit — prevents orphaned positions that can't be managed
- **Performance tracker as rolling window**: Last 20 trades (not all-time) — recent performance is more predictive than historical, and pairs go through trending/ranging cycles
- **Auto-disable threshold at 35% WR**: With partial TP (TP1 at 1×ATR), even a 35% WR pair likely covers fees — below that, the pair is consistently losing money
- **Auto-include requires scanner appearance**: A high-WR pair must also appear in the scanner scan to be auto-included — prevents including illiquid or flat pairs just because they had a lucky streak
- **Profit factor tracked in real-time**: `record_trade()` maintains running gross profit/loss sums — avoids expensive DB queries on every trade close while keeping dashboard current

## Environment
- Python 3.10+, ccxt.pro, numpy, fastapi, uvicorn
- `.env` for API keys: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`
- SQLite database: `bot_data.db` (WAL mode)
- Shadow trade logs: `shadow_trades.jsonl`, `shadow_trades_signals.jsonl`
