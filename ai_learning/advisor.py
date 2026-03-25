"""
AI Trading Advisor — consults the knowledge base before trade entry
and learns from outcomes via a post-trade feedback loop.

Pre-trade: queries KB for relevant strategies, risks, and mistakes,
then synthesizes advice via LLM (with caching and rate limiting).

Post-trade: updates KB entry application stats to reinforce good
knowledge and deprecate bad knowledge over time.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────

@dataclass
class AdvisorResult:
    """Result from a pre-trade knowledge consultation."""
    recommendation: str = "PROCEED"       # PROCEED / SKIP / CAUTION
    confidence_adjustment: float = 0.0    # -0.10 to +0.10
    reasoning: str = ""
    suggested_adjustments: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "tp_multiplier": None,
        "sl_multiplier": None,
        "position_size_factor": None,
    })
    source_ids: List[int] = field(default_factory=list)  # KB entry IDs used
    cached: bool = False                  # Was this a cache hit?
    skipped: bool = False                 # Was advisor skipped (unavailable/budget)?
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsultationRecord:
    """Stored record of an advisor consultation."""
    id: Optional[int] = None
    trade_id: Optional[int] = None
    symbol: str = ""
    timestamp: float = 0.0
    recommendation: str = ""
    confidence_adjustment: float = 0.0
    reasoning: str = ""
    suggested_adjustments: str = ""     # JSON string
    kb_entry_ids: str = ""              # JSON list of entry IDs
    was_followed: bool = True
    trade_outcome: str = ""             # win / loss / open
    trade_pnl: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        try:
            d["suggested_adjustments"] = json.loads(d["suggested_adjustments"]) if d["suggested_adjustments"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            d["kb_entry_ids"] = json.loads(d["kb_entry_ids"]) if d["kb_entry_ids"] else []
        except (json.JSONDecodeError, TypeError):
            pass
        return d


# ── Advisory Prompt ───────────────────────────────────────────

_ADVISORY_PROMPT_TEMPLATE = (
    "You are a crypto swing trading advisor for USDT-M futures. "
    "Given the current market setup and relevant knowledge, provide a brief recommendation.\n\n"
    "Current Setup:\n{context}\n\n"
    "Relevant Strategy Knowledge:\n{strategies}\n\n"
    "Relevant Risk Warnings:\n{risks}\n\n"
    "Known Mistakes to Avoid:\n{mistakes}\n\n"
    "Respond with ONLY valid JSON (no markdown, no explanation outside JSON):\n"
    '{{\n'
    '  "recommendation": "PROCEED" or "SKIP" or "CAUTION",\n'
    '  "confidence_adjustment": float between -0.10 and +0.10,\n'
    '  "reasoning": "Brief 1-2 sentence explanation",\n'
    '  "suggested_adjustments": {{\n'
    '    "tp_multiplier": float or null,\n'
    '    "sl_multiplier": float or null,\n'
    '    "position_size_factor": float or null\n'
    '  }}\n'
    '}}'
)


# ── Cache Key Builder ─────────────────────────────────────────

def _cache_key(symbol: str, side: str, regime: str,
               confidence: float, adx: float, rsi_1h: float) -> str:
    """Build a cache key from quantized market conditions."""
    # Quantize continuous values to reduce key space
    conf_q = round(confidence, 1)
    adx_q = round(adx / 5) * 5       # 5-unit buckets
    rsi_q = round(rsi_1h / 10) * 10   # 10-unit buckets
    raw = f"{symbol}|{side}|{regime}|{conf_q}|{adx_q}|{rsi_q}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main Advisor Class ────────────────────────────────────────

class TradingAdvisor:
    """
    Pre-trade knowledge consultation and post-trade feedback loop.

    Queries the knowledge base for relevant trading knowledge, uses an
    LLM to synthesize actionable advice, and tracks which knowledge
    entries lead to winning vs losing trades.
    """

    def __init__(self, config=None, knowledge_base=None):
        self.config = config
        self.kb = knowledge_base

        # Rate limiting
        self._daily_call_budget = 100
        self._calls_today = 0
        self._budget_reset_date = ""

        # Cache: key → (AdvisorResult, timestamp)
        self._cache: Dict[str, Tuple[AdvisorResult, float]] = {}
        self._cache_ttl = 900  # 15 minutes

        # LLM config
        self._provider = "anthropic"
        self._model = "claude-sonnet-4-5-20250514"
        self._retry_max = 2
        self._retry_delay = 1.5
        self._max_tokens = 400  # Keep advisory calls lightweight

        # Feedback thresholds
        self._deprecate_threshold = 0.30   # Success rate below → flag
        self._boost_threshold = 0.65       # Success rate above → boost
        self._min_applications = 10        # Minimum uses before acting

        # Consultation history (in-memory ring buffer for dashboard)
        self._recent_consultations: List[ConsultationRecord] = []
        self._max_history = 100

        # Stats
        self._total_consults = 0
        self._agreements = 0   # Advisor agreed with signal
        self._overrides = 0    # Signal overrode advisor SKIP

        if config:
            self._provider = getattr(config, "llm_provider", self._provider)
            self._model = getattr(config, "llm_model", self._model)
            self._retry_max = getattr(config, "llm_retry_max", self._retry_max)
            self._retry_delay = getattr(config, "llm_retry_delay", self._retry_delay)
            self._daily_call_budget = getattr(config, "advisor_daily_budget", 100)

    # ── Pre-Trade Consultation ────────────────────────────────

    async def consult(
        self,
        symbol: str,
        side: str,
        confidence: float,
        regime: str = "ranging",
        adx: float = 0.0,
        rsi_1h: float = 50.0,
        rsi_4h: float = 50.0,
        patterns_confirming: Optional[List[str]] = None,
        patterns_contradicting: Optional[List[str]] = None,
        squeeze_releasing: bool = False,
        funding_extreme: bool = False,
        oi_conviction: str = "neutral",
    ) -> AdvisorResult:
        """
        Consult the knowledge base and LLM for pre-trade advice.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            side: "BUY" or "SELL"
            confidence: Signal confidence (0.0-1.0)
            regime: Market regime string
            adx: ADX value
            rsi_1h: 1-hour RSI
            rsi_4h: 4-hour RSI
            patterns_confirming: Confirming candlestick patterns
            patterns_contradicting: Contradicting patterns
            squeeze_releasing: BB squeeze release active
            funding_extreme: Extreme funding rate
            oi_conviction: OI conviction level

        Returns:
            AdvisorResult with recommendation and adjustments
        """
        # Check if KB is available
        if not self.kb:
            return self._skip("No knowledge base available")

        # Check daily budget
        self._check_budget_reset()
        if self._calls_today >= self._daily_call_budget:
            return self._skip("Daily advisory call budget exhausted")

        # Check cache
        key = _cache_key(symbol, side, regime, confidence, adx, rsi_1h)
        cached = self._get_cached(key)
        if cached is not None:
            logger.debug(f"[{symbol}] Advisor cache hit")
            return cached

        # Build context string
        context = self._build_context(
            symbol, side, confidence, regime, adx,
            rsi_1h, rsi_4h, patterns_confirming or [],
            patterns_contradicting or [], squeeze_releasing,
            funding_extreme, oi_conviction,
        )

        # Query knowledge base
        strategies, risks, mistakes, source_ids = self._query_kb(symbol, side, regime)

        # If KB returned nothing useful, skip LLM call
        if not source_ids:
            return self._skip("No relevant knowledge found in KB")

        # Call LLM for synthesis
        try:
            result = await self._get_llm_advice(
                context, strategies, risks, mistakes, source_ids
            )
            self._calls_today += 1
            self._total_consults += 1
        except Exception as e:
            logger.warning(f"[{symbol}] Advisor LLM call failed: {e}")
            return self._skip(f"LLM unavailable: {e}")

        # Cache the result
        self._cache[key] = (result, time.time())

        # Record consultation
        record = ConsultationRecord(
            symbol=symbol,
            timestamp=time.time(),
            recommendation=result.recommendation,
            confidence_adjustment=result.confidence_adjustment,
            reasoning=result.reasoning,
            suggested_adjustments=json.dumps(result.suggested_adjustments),
            kb_entry_ids=json.dumps(result.source_ids),
        )
        self._recent_consultations.append(record)
        if len(self._recent_consultations) > self._max_history:
            self._recent_consultations = self._recent_consultations[-self._max_history:]

        logger.info(
            f"[{symbol}] Advisor: {result.recommendation} "
            f"(adj={result.confidence_adjustment:+.2f}) — {result.reasoning}"
        )
        return result

    def apply_advice(
        self,
        result: AdvisorResult,
        current_confidence: float,
    ) -> Tuple[bool, float, str]:
        """
        Apply advisor result to the trading decision.

        Returns:
            (should_proceed, adjusted_confidence, reason)

        Integration rules:
        - SKIP + confidence < 0.70 → honor the skip
        - SKIP + confidence >= 0.70 → log disagreement, proceed anyway
        - CAUTION → apply confidence adjustment, proceed
        - PROCEED → apply confidence adjustment, proceed
        """
        if result.skipped:
            return True, current_confidence, "advisor_skipped"

        adj = max(-0.10, min(0.10, result.confidence_adjustment))
        adjusted = current_confidence + adj

        if result.recommendation == "SKIP":
            if current_confidence < 0.70:
                self._agreements += 1
                return False, adjusted, f"advisor_skip: {result.reasoning}"
            else:
                self._overrides += 1
                logger.info(
                    f"Signal overrides advisor SKIP (conf={current_confidence:.2f}≥0.70)"
                )
                return True, adjusted, f"advisor_overridden: {result.reasoning}"

        if result.recommendation == "CAUTION":
            self._agreements += 1
            return True, adjusted, f"advisor_caution: {result.reasoning}"

        # PROCEED
        self._agreements += 1
        return True, adjusted, f"advisor_proceed: {result.reasoning}"

    # ── Post-Trade Feedback Loop ──────────────────────────────

    def record_trade_outcome(
        self,
        trade_id: int,
        source_ids: List[int],
        pnl_usd: float,
        was_followed: bool = True,
    ):
        """
        After a trade closes, update KB entry stats and advisor records.

        Args:
            trade_id: The closed trade ID
            source_ids: KB entry IDs that were consulted
            pnl_usd: Trade P&L in USD
            was_followed: Whether the advice was actually followed
        """
        success = pnl_usd > 0
        outcome = "win" if success else "loss"

        if not self.kb:
            return

        for entry_id in source_ids:
            try:
                self.kb.update_application_stats(entry_id, success)
            except Exception as e:
                logger.warning(f"Failed to update KB entry {entry_id}: {e}")

        # Update consultation records
        for record in reversed(self._recent_consultations):
            if record.trade_id == trade_id:
                record.trade_outcome = outcome
                record.trade_pnl = pnl_usd
                record.was_followed = was_followed
                break

        # Check for entries needing deprecation or boosting
        self._review_kb_entries(source_ids)

        logger.debug(
            f"Trade {trade_id} outcome={outcome} pnl={pnl_usd:.4f} "
            f"updated {len(source_ids)} KB entries"
        )

    def _review_kb_entries(self, entry_ids: List[int]):
        """Flag/boost KB entries based on accumulated success rates."""
        if not self.kb:
            return

        for entry_id in entry_ids:
            entry = self.kb.get_entry(entry_id)
            if not entry or entry.times_applied < self._min_applications:
                continue

            if entry.success_rate < self._deprecate_threshold:
                logger.warning(
                    f"KB entry {entry_id} flagged for review: "
                    f"success_rate={entry.success_rate:.1%} "
                    f"after {entry.times_applied} applications "
                    f"({entry.category}: {entry.source_title})"
                )
            elif entry.success_rate > self._boost_threshold:
                logger.info(
                    f"KB entry {entry_id} performing well: "
                    f"success_rate={entry.success_rate:.1%} "
                    f"after {entry.times_applied} applications "
                    f"({entry.category}: {entry.source_title})"
                )

    # ── KB Queries ────────────────────────────────────────────

    def _query_kb(
        self, symbol: str, side: str, regime: str
    ) -> Tuple[str, str, str, List[int]]:
        """
        Query the KB for relevant knowledge entries.

        Returns:
            (strategies_text, risks_text, mistakes_text, source_ids)
        """
        pair_short = symbol.replace("/USDT", "")
        direction = "long" if side == "BUY" else "short"

        # Search for relevant strategies
        strat_results = self.kb.search(
            f"{direction} swing entry {pair_short} crypto",
            top_k=3, category="strategy", min_confidence=0.5,
        )
        # Search for relevant risk rules
        risk_results = self.kb.search(
            f"risk {regime} market {direction}",
            top_k=3, category="risk_rule", min_confidence=0.5,
        )
        # Search for mistakes
        mistake_results = self.kb.search(
            f"mistake {direction} {regime} trading",
            top_k=3, category="mistake", min_confidence=0.5,
        )
        # Also search for relevant patterns/indicators
        pattern_results = self.kb.search(
            f"{direction} entry pattern {regime}",
            top_k=2, category="pattern", min_confidence=0.5,
        )

        all_results = strat_results + risk_results + mistake_results + pattern_results
        source_ids = [r.entry.id for r in all_results if r.entry.id]

        def _format_results(results) -> str:
            if not results:
                return "None found."
            lines = []
            for r in results:
                try:
                    content = json.loads(r.entry.content)
                    summary = " | ".join(
                        f"{k}: {v}" for k, v in content.items() if v
                    )
                except (json.JSONDecodeError, TypeError):
                    summary = str(r.entry.content)[:200]

                applied = ""
                if r.entry.times_applied > 0:
                    applied = f" [applied {r.entry.times_applied}x, {r.entry.success_rate:.0%} success]"
                lines.append(f"- {summary}{applied} (conf={r.entry.confidence:.2f})")
            return "\n".join(lines)

        return (
            _format_results(strat_results),
            _format_results(risk_results + pattern_results),
            _format_results(mistake_results),
            source_ids,
        )

    # ── LLM Call ──────────────────────────────────────────────

    async def _get_llm_advice(
        self,
        context: str,
        strategies: str,
        risks: str,
        mistakes: str,
        source_ids: List[int],
    ) -> AdvisorResult:
        """Call LLM to synthesize advisory from KB results + context."""
        prompt = _ADVISORY_PROMPT_TEMPLATE.format(
            context=context,
            strategies=strategies,
            risks=risks,
            mistakes=mistakes,
        )

        raw = await self._call_llm(prompt)
        if raw is None:
            return AdvisorResult(
                recommendation="PROCEED",
                reasoning="LLM call returned no response, defaulting to PROCEED",
                source_ids=source_ids,
                skipped=True,
                skip_reason="LLM returned None",
            )

        parsed = self._parse_response(raw)
        if parsed is None:
            return AdvisorResult(
                recommendation="PROCEED",
                reasoning="Could not parse LLM response, defaulting to PROCEED",
                source_ids=source_ids,
                skipped=True,
                skip_reason="Parse failure",
            )

        # Clamp confidence adjustment
        conf_adj = parsed.get("confidence_adjustment", 0.0)
        if not isinstance(conf_adj, (int, float)):
            conf_adj = 0.0
        conf_adj = max(-0.10, min(0.10, float(conf_adj)))

        rec = parsed.get("recommendation", "PROCEED")
        if rec not in ("PROCEED", "SKIP", "CAUTION"):
            rec = "PROCEED"

        adjustments = parsed.get("suggested_adjustments", {})
        if not isinstance(adjustments, dict):
            adjustments = {}

        # Validate adjustment values
        for key in ("tp_multiplier", "sl_multiplier", "position_size_factor"):
            val = adjustments.get(key)
            if val is not None:
                try:
                    val = float(val)
                    # Sanity bounds
                    if key == "tp_multiplier":
                        val = max(0.5, min(3.0, val))
                    elif key == "sl_multiplier":
                        val = max(0.5, min(2.0, val))
                    elif key == "position_size_factor":
                        val = max(0.3, min(1.5, val))
                    adjustments[key] = val
                except (ValueError, TypeError):
                    adjustments[key] = None

        return AdvisorResult(
            recommendation=rec,
            confidence_adjustment=conf_adj,
            reasoning=str(parsed.get("reasoning", ""))[:500],
            suggested_adjustments=adjustments,
            source_ids=source_ids,
        )

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with retries."""
        for attempt in range(self._retry_max):
            try:
                if self._provider == "anthropic":
                    return await self._call_anthropic(prompt)
                elif self._provider == "openai":
                    return await self._call_openai(prompt)
                else:
                    logger.error(f"Unknown LLM provider: {self._provider}")
                    return None
            except Exception as e:
                delay = self._retry_delay * (2 ** attempt)
                logger.warning(
                    f"Advisor LLM attempt {attempt + 1}/{self._retry_max} failed: {e}"
                )
                if attempt < self._retry_max - 1:
                    await asyncio.sleep(delay)
        return None

    async def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai SDK not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        symbol: str, side: str, confidence: float,
        regime: str, adx: float, rsi_1h: float, rsi_4h: float,
        patterns_confirming: List[str], patterns_contradicting: List[str],
        squeeze_releasing: bool, funding_extreme: bool, oi_conviction: str,
    ) -> str:
        parts = [
            f"Pair: {symbol}, Direction: {side}",
            f"Regime: {regime}, ADX: {adx:.0f}",
            f"RSI 1h/4h: {rsi_1h:.0f}/{rsi_4h:.0f}",
            f"Confidence: {confidence:.2f}",
        ]
        if patterns_confirming:
            parts.append(f"Confirming patterns: {', '.join(patterns_confirming)}")
        if patterns_contradicting:
            parts.append(f"Contradicting patterns: {', '.join(patterns_contradicting)}")
        if squeeze_releasing:
            parts.append("BB Squeeze releasing")
        if funding_extreme:
            parts.append("WARNING: Extreme funding rate")
        if oi_conviction != "neutral":
            parts.append(f"OI conviction: {oi_conviction}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass
        return None

    def _skip(self, reason: str) -> AdvisorResult:
        """Return a no-op advisor result when consultation is skipped."""
        return AdvisorResult(
            recommendation="PROCEED",
            skipped=True,
            skip_reason=reason,
        )

    def _get_cached(self, key: str) -> Optional[AdvisorResult]:
        """Check cache for a recent result."""
        if key not in self._cache:
            return None
        result, ts = self._cache[key]
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        cached_result = AdvisorResult(
            recommendation=result.recommendation,
            confidence_adjustment=result.confidence_adjustment,
            reasoning=result.reasoning,
            suggested_adjustments=result.suggested_adjustments.copy(),
            source_ids=result.source_ids.copy(),
            cached=True,
        )
        return cached_result

    def _check_budget_reset(self):
        """Reset daily call counter at UTC midnight."""
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._budget_reset_date:
            self._budget_reset_date = today
            self._calls_today = 0

    def clear_cache(self):
        """Clear the advisory cache."""
        self._cache.clear()

    # ── Stats / Dashboard ─────────────────────────────────────

    def get_stats(self) -> dict:
        """Return advisor statistics for the dashboard."""
        total = self._total_consults
        return {
            "total_consultations": total,
            "calls_today": self._calls_today,
            "daily_budget": self._daily_call_budget,
            "cache_size": len(self._cache),
            "agreements": self._agreements,
            "overrides": self._overrides,
            "agreement_rate": (
                self._agreements / max(self._agreements + self._overrides, 1)
            ),
            "recent_consultations": [
                r.to_dict() for r in self._recent_consultations[-10:]
            ],
        }

    def get_consultation_for_trade(self, trade_id: int) -> Optional[dict]:
        """Get the advisor consultation record for a specific trade."""
        for record in reversed(self._recent_consultations):
            if record.trade_id == trade_id:
                return record.to_dict()
        return None

    def get_kb_performance_report(self) -> List[dict]:
        """Get KB entries sorted by application frequency with outcomes."""
        if not self.kb:
            return []

        conn = self.kb._get_conn()
        rows = conn.execute("""
            SELECT id, source_type, source_title, category, content,
                   confidence, times_applied, success_rate
            FROM knowledge_entries
            WHERE times_applied > 0
            ORDER BY times_applied DESC
            LIMIT 50
        """).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            status = "neutral"
            if r["times_applied"] >= self._min_applications:
                if r["success_rate"] < self._deprecate_threshold:
                    status = "flagged"
                elif r["success_rate"] > self._boost_threshold:
                    status = "boosted"
            r["status"] = status
            try:
                r["content"] = json.loads(r["content"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(r)
        return results
