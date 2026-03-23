"""
Opportunity Scanner — Scans top Binance USDT-M Futures pairs by volume,
computes opportunity scores, and selects the best pairs for trading.

Runs every 2 hours (configurable). Rate-limit-safe with configurable delays.

Score components:
  - ADX value (trending strength)         — 30%
  - BB width percentile (squeeze potential) — 20%
  - Volume change vs 7-day average        — 20%
  - Recent volatility (ATR as % of price) — 20%
  - Funding rate extremity                — 10%
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BotConfig, CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────

@dataclass
class PairScore:
    """Opportunity score for a single pair."""
    symbol: str
    score: float = 0.0

    # Raw metrics
    adx: float = 0.0
    bb_width_pctile: float = 50.0       # Percentile of current BB width (0-100)
    volume_24h_usd: float = 0.0
    volume_change_pct: float = 0.0      # vs 7-day average
    atr_pct: float = 0.0                # ATR / price as percentage
    funding_rate: float = 0.0           # Signed, in percent
    spread_pct: float = 0.0
    last_price: float = 0.0

    # Sub-scores (0-1 normalized)
    adx_score: float = 0.0
    squeeze_score: float = 0.0
    volume_score: float = 0.0
    volatility_score: float = 0.0
    funding_score: float = 0.0

    # Selection result
    selected: bool = False
    selection_reason: str = ""
    disqualified: bool = False
    disqualify_reason: str = ""

    # Squeeze state
    squeeze_active: bool = False
    squeeze_releasing: bool = False


@dataclass
class ScanResult:
    """Result of a full scanner run."""
    timestamp: float = 0.0
    pairs_scanned: int = 0
    pairs_qualified: int = 0
    selected_pairs: List[str] = field(default_factory=list)
    anchor_pairs: List[str] = field(default_factory=list)
    retained_pairs: List[str] = field(default_factory=list)  # Kept for open positions
    dropped_pairs: List[str] = field(default_factory=list)
    added_pairs: List[str] = field(default_factory=list)
    all_scores: List[PairScore] = field(default_factory=list)
    scan_duration_s: float = 0.0

    def get_active_pairs(self) -> List[str]:
        """Return the full active pair list (anchors + selected + retained)."""
        seen = set()
        result = []
        for p in self.anchor_pairs:
            if p not in seen:
                result.append(p)
                seen.add(p)
        for p in self.selected_pairs:
            if p not in seen:
                result.append(p)
                seen.add(p)
        for p in self.retained_pairs:
            if p not in seen:
                result.append(p)
                seen.add(p)
        return result


# ─────────────────────────────────────────
# OPPORTUNITY SCANNER
# ─────────────────────────────────────────

class OpportunityScanner:
    """
    Scans top Binance USDT-M Futures pairs and scores them for trading opportunity.

    Uses only ccxt exchange methods — no custom Binance API calls.
    Rate-limit safe: configurable delay between sequential API calls.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.scanner

    async def scan(self, exchange) -> List[PairScore]:
        """
        Scan top pairs by volume and compute opportunity scores.

        Args:
            exchange: ccxt.pro exchange instance (already initialized).

        Returns:
            List of PairScore sorted by score descending.
        """
        # Step 1: Get all USDT-M futures tickers, rank by volume
        candidates = await self._get_top_volume_pairs(exchange)
        if not candidates:
            logger.warning("Scanner: no candidates found")
            return []

        logger.info(f"Scanner: scoring {len(candidates)} candidates...")

        # Step 2: For each candidate, fetch OHLCV + funding + ticker for scoring
        scores = []
        for symbol in candidates:
            try:
                ps = await self._score_pair(exchange, symbol)
                if ps:
                    scores.append(ps)
            except Exception as e:
                logger.debug(f"Scanner: error scoring {symbol}: {e}")
            # Rate limit between API calls
            await asyncio.sleep(self.sc.rate_limit_delay_s)

        # Step 3: Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

    async def _get_top_volume_pairs(self, exchange) -> List[str]:
        """Fetch tickers and return top N USDT-M futures pairs by 24h volume."""
        try:
            tickers = await exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"Scanner: failed to fetch tickers: {e}")
            return []

        excluded = set(self.config.trading.excluded_pairs)
        usdt_pairs = []

        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT"):
                continue
            if symbol in excluded:
                continue
            if ":USDT" in symbol:
                # ccxt futures format: "BTC/USDT:USDT" — normalize
                symbol = symbol.split(":")[0]
            quote_vol = ticker.get("quoteVolume", 0) or 0
            usdt_pairs.append((symbol, float(quote_vol)))

        # Sort by volume descending, take top N
        usdt_pairs.sort(key=lambda x: x[1], reverse=True)
        top_n = self.sc.scan_top_n
        return [s for s, _ in usdt_pairs[:top_n]]

    async def _score_pair(self, exchange, symbol: str) -> Optional[PairScore]:
        """
        Fetch market data and compute opportunity score for a single pair.

        Fetches: 1h OHLCV (50 candles), 1d OHLCV (8 candles for 7-day avg),
        funding rate, current ticker.
        """
        ps = PairScore(symbol=symbol)

        # Fetch 1h candles (50 candles for ADX/ATR/BB)
        try:
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, "1h", limit=50)
            await asyncio.sleep(self.sc.rate_limit_delay_s)
        except Exception:
            return None

        if not ohlcv_1h or len(ohlcv_1h) < 20:
            return None

        closes = np.array([c[4] for c in ohlcv_1h], dtype=np.float64)
        highs = np.array([c[2] for c in ohlcv_1h], dtype=np.float64)
        lows = np.array([c[3] for c in ohlcv_1h], dtype=np.float64)
        volumes = np.array([c[5] for c in ohlcv_1h], dtype=np.float64)

        ps.last_price = float(closes[-1])
        if ps.last_price <= 0:
            return None

        # ── ADX ──
        ps.adx = self._compute_adx(highs, lows, closes)
        # Normalize: ADX 0-60 mapped to 0-1, capped
        ps.adx_score = min(ps.adx / 60.0, 1.0)

        # ── ATR as % of price (volatility) ──
        atr = self._compute_atr(highs, lows, closes)
        ps.atr_pct = (atr / ps.last_price) * 100.0 if ps.last_price > 0 else 0.0
        # Normalize: 0-3% ATR mapped to 0-1
        ps.volatility_score = min(ps.atr_pct / 3.0, 1.0)

        # ── BB width percentile (squeeze potential) ──
        ps.bb_width_pctile, ps.squeeze_active = self._compute_bb_squeeze(closes)
        # Low percentile = squeeze = high opportunity score
        # Invert: percentile 0 (tight squeeze) → score 1.0
        ps.squeeze_score = max(0.0, 1.0 - ps.bb_width_pctile / 100.0)

        # ── 24h volume & volume change vs 7-day avg ──
        try:
            ticker = await exchange.fetch_ticker(symbol)
            await asyncio.sleep(self.sc.rate_limit_delay_s)
            ps.volume_24h_usd = float(ticker.get("quoteVolume", 0) or 0)
            ps.spread_pct = 0.0
            bid = ticker.get("bid", 0) or 0
            ask = ticker.get("ask", 0) or 0
            if bid > 0 and ask > 0:
                ps.spread_pct = ((ask - bid) / ((ask + bid) / 2)) * 100.0
        except Exception:
            pass

        # Volume change: current 24h vs rolling 7-day average from daily candles
        try:
            ohlcv_1d = await exchange.fetch_ohlcv(symbol, "1d", limit=8)
            await asyncio.sleep(self.sc.rate_limit_delay_s)
            if ohlcv_1d and len(ohlcv_1d) >= 2:
                daily_vols = [float(c[5]) * float(c[4]) for c in ohlcv_1d[:-1]]  # vol * close ≈ USD
                avg_7d = np.mean(daily_vols) if daily_vols else 0
                if avg_7d > 0:
                    current_vol = float(ohlcv_1d[-1][5]) * float(ohlcv_1d[-1][4])
                    ps.volume_change_pct = ((current_vol - avg_7d) / avg_7d) * 100.0
        except Exception:
            pass

        # Normalize volume change: -50% to +200% mapped to 0-1
        ps.volume_score = max(0.0, min((ps.volume_change_pct + 50) / 250.0, 1.0))

        # ── Funding rate ──
        try:
            fr = await exchange.fetch_funding_rate(symbol)
            await asyncio.sleep(self.sc.rate_limit_delay_s)
            if fr and "fundingRate" in fr:
                ps.funding_rate = float(fr["fundingRate"]) * 100.0
        except Exception:
            pass

        # Funding extremity: abs value, extreme = opportunity
        abs_funding = abs(ps.funding_rate)
        # Normalize: 0-0.1% → 0-1
        ps.funding_score = min(abs_funding / 0.10, 1.0)

        # ── Composite score ──
        w = self.sc
        ps.score = (
            w.weight_adx * ps.adx_score
            + w.weight_bb_squeeze * ps.squeeze_score
            + w.weight_volume_change * ps.volume_score
            + w.weight_volatility * ps.volatility_score
            + w.weight_funding * ps.funding_score
        )

        return ps

    def _compute_adx(self, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        """Compute ADX from OHLC arrays."""
        n = len(closes)
        if n < period + 1:
            return 0.0

        # True Range
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        # +DM / -DM
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder smoothing (EMA with alpha=1/period)
        alpha = 1.0 / period
        atr_val = tr[0]
        pdm_smooth = plus_dm[0]
        ndm_smooth = minus_dm[0]

        for i in range(1, len(tr)):
            atr_val = atr_val * (1 - alpha) + tr[i] * alpha
            pdm_smooth = pdm_smooth * (1 - alpha) + plus_dm[i] * alpha
            ndm_smooth = ndm_smooth * (1 - alpha) + minus_dm[i] * alpha

        if atr_val <= 0:
            return 0.0

        plus_di = (pdm_smooth / atr_val) * 100.0
        minus_di = (ndm_smooth / atr_val) * 100.0
        di_sum = plus_di + minus_di

        if di_sum <= 0:
            return 0.0

        dx = abs(plus_di - minus_di) / di_sum * 100.0

        # For a single ADX value we just return the latest DX
        # (full ADX would smooth DX over period — DX is sufficient for screening)
        return float(dx)

    def _compute_atr(self, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        """Compute ATR (Average True Range)."""
        n = len(closes)
        if n < 2:
            return 0.0

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0

        # Wilder smoothing
        alpha = 1.0 / period
        atr = tr[0]
        for i in range(1, len(tr)):
            atr = atr * (1 - alpha) + tr[i] * alpha
        return float(atr)

    def _compute_bb_squeeze(self, closes: np.ndarray,
                            period: int = 20, std_mult: float = 2.0,
                            lookback: int = 100) -> Tuple[float, bool]:
        """
        Compute BB width percentile and squeeze state.

        Returns:
            (percentile, squeeze_active)
            percentile: 0-100, lower = tighter bands (more squeeze potential)
            squeeze_active: True if current width is below 20th percentile
        """
        n = len(closes)
        if n < period:
            return 50.0, False

        # Compute BB width for each position in the lookback window
        widths = []
        start = max(0, n - lookback)
        for i in range(start + period, n + 1):
            window = closes[i - period:i]
            sma = np.mean(window)
            std = np.std(window, ddof=1) if len(window) > 1 else 0.0
            if sma > 0:
                upper = sma + std_mult * std
                lower = sma - std_mult * std
                width = (upper - lower) / sma
                widths.append(width)

        if not widths:
            return 50.0, False

        current_width = widths[-1]
        # Percentile rank of current width
        below = sum(1 for w in widths if w < current_width)
        percentile = (below / len(widths)) * 100.0

        squeeze_active = percentile < 20.0
        return percentile, squeeze_active


# ─────────────────────────────────────────
# PAIR SELECTOR
# ─────────────────────────────────────────

class PairSelector:
    """
    Selects the best trading pairs from scanner results.

    Always includes anchor pairs (BTC/USDT, ETH/USDT).
    Selects top N dynamic pairs that pass all filters.
    Manages transitions: if a dropped pair has an open position, it stays
    in the active list until that position is closed.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.sc = config.scanner
        self._current_pairs: List[str] = list(self.sc.anchor_pairs)
        self._last_scan: Optional[ScanResult] = None
        self._last_scan_time: float = 0.0

    @property
    def current_pairs(self) -> List[str]:
        return list(self._current_pairs)

    @property
    def last_scan(self) -> Optional[ScanResult]:
        return self._last_scan

    def needs_scan(self) -> bool:
        """Check if a new scan is due."""
        if not self._last_scan:
            return True
        return (time.time() - self._last_scan_time) >= self.sc.scan_interval_s

    def select(
        self,
        scores: List[PairScore],
        open_positions: List[str],
        disabled_pairs: List[str] = None,
        auto_include_pairs: List[str] = None,
    ) -> ScanResult:
        """
        Select pairs from scan results.

        Args:
            scores: Scored pairs from OpportunityScanner.scan()
            open_positions: Symbols with open trades (keep in active list)
            disabled_pairs: Pairs disabled by performance tracker
            auto_include_pairs: Pairs auto-included by performance tracker

        Returns:
            ScanResult with selected pairs and metadata.
        """
        disabled = set(disabled_pairs or [])
        auto_include = set(auto_include_pairs or [])
        previous_pairs = set(self._current_pairs)

        result = ScanResult(
            timestamp=time.time(),
            pairs_scanned=len(scores),
            anchor_pairs=list(self.sc.anchor_pairs),
        )

        # Apply selection filters and disqualify
        qualified = []
        for ps in scores:
            ps.disqualified = False
            ps.disqualify_reason = ""

            if ps.symbol in self.sc.anchor_pairs:
                # Anchor pairs always selected, don't count as dynamic
                ps.selected = True
                ps.selection_reason = "anchor pair"
                continue

            if ps.symbol in disabled:
                ps.disqualified = True
                ps.disqualify_reason = "disabled by performance tracker"
                continue

            if ps.volume_24h_usd < self.sc.min_24h_volume_usd:
                ps.disqualified = True
                ps.disqualify_reason = f"volume ${ps.volume_24h_usd/1e6:.0f}M < ${self.sc.min_24h_volume_usd/1e6:.0f}M min"
                continue

            if ps.spread_pct > self.sc.max_spread_pct and ps.spread_pct > 0:
                ps.disqualified = True
                ps.disqualify_reason = f"spread {ps.spread_pct:.4f}% > {self.sc.max_spread_pct}% max"
                continue

            if ps.adx < self.sc.min_adx:
                ps.disqualified = True
                ps.disqualify_reason = f"ADX {ps.adx:.1f} < {self.sc.min_adx} min"
                continue

            if ps.squeeze_active and not ps.squeeze_releasing:
                ps.disqualified = True
                ps.disqualify_reason = "squeeze active with no release signal"
                continue

            qualified.append(ps)

        result.pairs_qualified = len(qualified)

        # Sort qualified by score descending
        qualified.sort(key=lambda x: x.score, reverse=True)

        # Select top N dynamic pairs
        selected_dynamic = []
        for ps in qualified:
            if len(selected_dynamic) >= self.sc.select_dynamic:
                break
            ps.selected = True
            ps.selection_reason = f"top scorer ({ps.score:.3f})"
            selected_dynamic.append(ps.symbol)

        # Auto-include high-WR pairs that also appeared in scan
        for ps in qualified:
            if ps.symbol in auto_include and ps.symbol not in selected_dynamic:
                if len(selected_dynamic) < self.sc.select_dynamic + 2:  # Allow 2 extra for auto-include
                    ps.selected = True
                    ps.selection_reason = f"auto-included (high win rate)"
                    selected_dynamic.append(ps.symbol)

        result.selected_pairs = selected_dynamic

        # Compute full active list
        new_pairs = set(self.sc.anchor_pairs) | set(selected_dynamic)

        # Keep pairs that have open positions even if not selected
        for sym in open_positions:
            if sym not in new_pairs:
                new_pairs.add(sym)
                if sym not in set(self.sc.anchor_pairs) and sym not in set(selected_dynamic):
                    result.retained_pairs.append(sym)

        # Determine added/dropped
        result.added_pairs = [p for p in new_pairs if p not in previous_pairs]
        result.dropped_pairs = [p for p in previous_pairs if p not in new_pairs]
        result.all_scores = scores

        self._current_pairs = sorted(new_pairs)
        self._last_scan = result
        self._last_scan_time = time.time()

        return result

    def get_scan_summary(self) -> dict:
        """Get a JSON-serializable summary for dashboard."""
        if not self._last_scan:
            return {"status": "no scan yet", "pairs": self._current_pairs}

        sr = self._last_scan
        scores_summary = []
        for ps in sr.all_scores[:15]:  # Top 15 for dashboard
            scores_summary.append({
                "symbol": ps.symbol,
                "score": round(ps.score, 4),
                "adx": round(ps.adx, 1),
                "atr_pct": round(ps.atr_pct, 3),
                "volume_24h_usd": round(ps.volume_24h_usd, 0),
                "volume_change_pct": round(ps.volume_change_pct, 1),
                "bb_width_pctile": round(ps.bb_width_pctile, 1),
                "funding_rate": round(ps.funding_rate, 4),
                "spread_pct": round(ps.spread_pct, 4),
                "squeeze_active": ps.squeeze_active,
                "selected": ps.selected,
                "selection_reason": ps.selection_reason,
                "disqualified": ps.disqualified,
                "disqualify_reason": ps.disqualify_reason,
            })

        return {
            "scan_time": sr.timestamp,
            "pairs_scanned": sr.pairs_scanned,
            "pairs_qualified": sr.pairs_qualified,
            "selected_pairs": sr.selected_pairs,
            "anchor_pairs": sr.anchor_pairs,
            "active_pairs": sr.get_active_pairs(),
            "added_pairs": sr.added_pairs,
            "dropped_pairs": sr.dropped_pairs,
            "scan_duration_s": round(sr.scan_duration_s, 1),
            "scores": scores_summary,
        }
