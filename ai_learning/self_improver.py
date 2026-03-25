"""
Self-Improvement Engine — automated performance analysis, hypothesis
generation, backtesting validation, and parameter optimization.

Optimizes for DAILY TARGET ACHIEVEMENT RATE as the primary KPI.

Components:
  PerformanceAnalyzer — daily/weekly performance reports from trade history
  HypothesisGenerator — LLM-driven improvement hypotheses from analysis
  HypothesisValidator — backtests hypotheses and checks acceptance criteria
  ImprovementManager — lifecycle management with safety rails
"""

import asyncio
import copy
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ensure parent is on path for imports
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from ai_learning.improvement_db import ImprovementDB, Hypothesis, AnalysisReport


# ── Safety Constants ───────────────────────────────────────────

# Parameters that must NEVER be auto-modified (risk params are human-only)
PROTECTED_PARAMS = frozenset({
    "risk.max_daily_loss_pct",
    "risk.max_drawdown_halt_pct",
    "risk.intraday_dd_halt_pct",
    "futures.max_leverage",
    "trading.max_position_pct",
    "daily_target.loss_limit_pct",
})

MAX_CHANGES_PER_WEEK = 2
MIN_BACKTEST_DAYS = 30
AUTO_REVERT_WR_THRESHOLD = 0.35
REVIEW_PERIOD_DAYS = 7


# ═══════════════════════════════════════════════════════════════
# PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    """
    Analyzes trade history and daily equity to produce structured
    performance reports. Pure computation — no LLM calls.
    """

    def __init__(self, db_path: str = ""):
        self.db_path = db_path

    def _get_conn(self):
        import database as db
        return db.get_conn()

    # ── Daily Analysis (lightweight, 00:05 UTC) ────────────────

    def daily_analysis(self, date: str = None) -> dict:
        """
        Produce a daily performance report.

        Args:
            date: "YYYY-MM-DD" or None for yesterday (UTC)
        """
        if date is None:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            date = yesterday.strftime("%Y-%m-%d")

        conn = self._get_conn()

        # Get daily equity record
        eq_row = conn.execute(
            "SELECT * FROM daily_equity WHERE date=?", (date,)
        ).fetchone()

        equity_data = dict(eq_row) if eq_row else {}

        # Get all trades closed on this date
        day_start = datetime.strptime(date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
        day_end = day_start + 86400

        trades = conn.execute("""
            SELECT * FROM trades
            WHERE is_open=0 AND exit_time >= ? AND exit_time < ?
            ORDER BY exit_time
        """, (day_start, day_end)).fetchall()
        trades = [dict(t) for t in trades]

        if not trades and not equity_data:
            return {"date": date, "has_data": False}

        # Core metrics
        total = len(trades)
        wins = sum(1 for t in trades if (t.get("pnl_usd") or 0) > 0)
        losses = total - wins
        win_rate = wins / max(total, 1)
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)

        # Target achievement
        target_pct = equity_data.get("target_pct", 2.0)
        actual_pct = equity_data.get("actual_pct", 0.0)
        target_hit = equity_data.get("target_hit", 0)

        # Best/worst trade
        sorted_by_pnl = sorted(trades, key=lambda t: t.get("pnl_usd", 0))
        best_trade = sorted_by_pnl[-1] if sorted_by_pnl else None
        worst_trade = sorted_by_pnl[0] if sorted_by_pnl else None

        # P&L by pair
        pnl_by_pair: Dict[str, float] = defaultdict(float)
        count_by_pair: Dict[str, int] = defaultdict(int)
        for t in trades:
            sym = t.get("symbol", "unknown")
            pnl_by_pair[sym] += t.get("pnl_usd", 0)
            count_by_pair[sym] += 1

        # P&L by hour
        pnl_by_hour: Dict[int, float] = defaultdict(float)
        for t in trades:
            et = t.get("exit_time")
            if et:
                hour = datetime.fromtimestamp(et, tz=timezone.utc).hour
                pnl_by_hour[hour] += t.get("pnl_usd", 0)

        # Mode analysis: was PROTECTING entered too early/late?
        mode_at_close = equity_data.get("mode_at_close", "unknown")

        return {
            "date": date,
            "has_data": True,
            "target_hit": bool(target_hit),
            "target_pct": target_pct,
            "actual_pct": actual_pct,
            "over_under_target": actual_pct - target_pct,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl_usd": total_pnl,
            "best_trade": _trade_summary(best_trade) if best_trade else None,
            "worst_trade": _trade_summary(worst_trade) if worst_trade else None,
            "pnl_by_pair": dict(pnl_by_pair),
            "trades_by_pair": dict(count_by_pair),
            "pnl_by_hour": {str(k): v for k, v in sorted(pnl_by_hour.items())},
            "mode_at_close": mode_at_close,
            "open_equity": equity_data.get("open_equity", 0),
            "close_equity": equity_data.get("close_equity", 0),
        }

    # ── Weekly Analysis (comprehensive) ────────────────────────

    def weekly_analysis(self, days: int = 7, end_date: str = None) -> dict:
        """
        Produce a comprehensive weekly performance report with
        segmented analytics across every dimension.

        Args:
            days: Number of days to analyze (default 7)
            end_date: End date "YYYY-MM-DD" or None for today
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime("%Y-%m-%d")

        conn = self._get_conn()

        start_ts = start_dt.timestamp()
        end_ts = (end_dt + timedelta(days=1)).timestamp()

        # All closed trades in the period
        trades = conn.execute("""
            SELECT * FROM trades
            WHERE is_open=0 AND exit_time >= ? AND exit_time < ?
            ORDER BY exit_time
        """, (start_ts, end_ts)).fetchall()
        trades = [dict(t) for t in trades]

        # Daily equity records
        daily_rows = conn.execute("""
            SELECT * FROM daily_equity
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, (start_date, end_date)).fetchall()
        daily_records = [dict(r) for r in daily_rows]

        total = len(trades)
        if total == 0:
            return {
                "period": f"{start_date} to {end_date}",
                "has_data": False,
                "total_trades": 0,
            }

        wins = sum(1 for t in trades if (t.get("pnl_usd") or 0) > 0)
        overall_wr = wins / max(total, 1)
        net_pnl = sum(t.get("pnl_usd", 0) for t in trades)

        # Daily target achievement
        days_total = len(daily_records)
        days_target_hit = sum(1 for d in daily_records if d.get("target_hit"))
        target_pct = daily_records[0].get("target_pct", 2.0) if daily_records else 2.0
        tar = days_target_hit / max(days_total, 1)

        # Compute actual compound growth vs target
        start_eq = daily_records[0].get("open_equity", 0) if daily_records else 0
        end_eq = daily_records[-1].get("close_equity", 0) if daily_records else 0
        compound_actual = ((end_eq / max(start_eq, 0.01)) - 1) * 100 if start_eq > 0 else 0
        compound_target = ((1 + target_pct / 100) ** days_total - 1) * 100

        # Daily P&L
        daily_pnl = [d.get("actual_pct", 0) for d in daily_records]
        avg_daily_pnl = sum(daily_pnl) / max(len(daily_pnl), 1)

        # Best/worst day
        best_day = max(daily_records, key=lambda d: d.get("actual_pct", 0)) if daily_records else None
        worst_day = min(daily_records, key=lambda d: d.get("actual_pct", 0)) if daily_records else None

        # Optimal trade count analysis
        optimal_trades = self._optimal_trade_count(daily_records)

        # ── Segmented analytics ──
        segments = self._compute_segments(trades)
        worst_segments = [s for s in segments if s["win_rate"] < 0.40 and s["trades"] >= 3]
        best_segments = [s for s in segments if s["win_rate"] > 0.60 and s["trades"] >= 3]
        worst_segments.sort(key=lambda s: s["win_rate"])
        best_segments.sort(key=lambda s: -s["win_rate"])

        # Common patterns
        loss_patterns = self._find_loss_patterns(trades)
        win_patterns = self._find_win_patterns(trades)

        return {
            "period": f"{start_date} to {end_date}",
            "has_data": True,
            "overall_win_rate": round(overall_wr, 4),
            "total_trades": total,
            "net_pnl_usd": round(net_pnl, 4),
            "net_pnl_pct": round(compound_actual, 2),
            "daily_target_pct": target_pct,
            "days_target_hit": days_target_hit,
            "days_total": days_total,
            "target_achievement_rate": round(tar, 4),
            "avg_daily_pnl_pct": round(avg_daily_pnl, 2),
            "best_day": _day_summary(best_day) if best_day else None,
            "worst_day": _day_summary(worst_day) if worst_day else None,
            "optimal_trades_per_day": optimal_trades,
            "compound_growth_actual_pct": round(compound_actual, 2),
            "compound_growth_target_pct": round(compound_target, 2),
            "worst_segments": worst_segments[:10],
            "best_segments": best_segments[:10],
            "common_loss_patterns": loss_patterns[:5],
            "common_win_patterns": win_patterns[:5],
            "all_segments": segments,
            "daily_records": [
                _day_summary(d) for d in daily_records
            ],
        }

    # ── Segmentation ───────────────────────────────────────────

    def _compute_segments(self, trades: List[dict]) -> List[dict]:
        """Compute win rate by every dimension."""
        segments = []

        def _add_segment(dimension: str, grouper):
            groups = defaultdict(list)
            for t in trades:
                val = grouper(t)
                if val is not None:
                    groups[val].append(t)
            for value, group_trades in groups.items():
                w = sum(1 for t in group_trades if (t.get("pnl_usd") or 0) > 0)
                n = len(group_trades)
                pnl = sum(t.get("pnl_usd", 0) for t in group_trades)
                segments.append({
                    "dimension": dimension,
                    "value": str(value),
                    "win_rate": round(w / max(n, 1), 4),
                    "trades": n,
                    "pnl_usd": round(pnl, 4),
                })

        # By pair
        _add_segment("pair", lambda t: t.get("symbol"))

        # By direction
        _add_segment("direction", lambda t: t.get("side"))

        # By regime
        _add_segment("regime", lambda t: t.get("regime"))

        # By session
        _add_segment("session", lambda t: t.get("session"))

        # By confidence bucket
        def conf_bucket(t):
            c = t.get("swing_confidence") or t.get("confidence") or 0
            if c >= 0.75:
                return "0.75+"
            elif c >= 0.65:
                return "0.65-0.74"
            elif c >= 0.55:
                return "0.55-0.64"
            return "<0.55"
        _add_segment("confidence", conf_bucket)

        # By day of week
        def dow(t):
            et = t.get("exit_time")
            if et:
                return datetime.fromtimestamp(et, tz=timezone.utc).strftime("%A")
            return None
        _add_segment("day_of_week", dow)

        # By hour of entry
        def entry_hour(t):
            et = t.get("entry_time") or t.get("created_at")
            if et:
                return f"{datetime.fromtimestamp(et, tz=timezone.utc).hour:02d}:00"
            return None
        _add_segment("hour", entry_hour)

        # By exit reason
        _add_segment("exit_reason", lambda t: t.get("exit_reason"))

        return segments

    def _optimal_trade_count(self, daily_records: List[dict]) -> int:
        """Find the trade count that maximizes target achievement."""
        if not daily_records:
            return 0
        count_to_hits = defaultdict(list)
        for d in daily_records:
            n = d.get("trades", 0) or d.get("wins", 0) + d.get("losses", 0)
            hit = bool(d.get("target_hit"))
            count_to_hits[n].append(hit)

        best_count = 0
        best_rate = 0
        for count, hits in count_to_hits.items():
            rate = sum(hits) / max(len(hits), 1)
            if rate > best_rate or (rate == best_rate and count < best_count):
                best_rate = rate
                best_count = count
        return best_count

    def _find_loss_patterns(self, trades: List[dict]) -> List[str]:
        """Identify common patterns among losing trades."""
        losses = [t for t in trades if (t.get("pnl_usd") or 0) < 0]
        if not losses:
            return []

        patterns = []
        # Check regime distribution
        regime_counts = defaultdict(int)
        for t in losses:
            regime_counts[t.get("regime", "unknown")] += 1
        for regime, count in regime_counts.items():
            if count >= 3 and count / max(len(losses), 1) > 0.3:
                patterns.append(f"Frequent losses in {regime} regime ({count}/{len(losses)})")

        # Check session distribution
        session_counts = defaultdict(int)
        for t in losses:
            session_counts[t.get("session", "unknown")] += 1
        for session, count in session_counts.items():
            if count >= 3 and count / max(len(losses), 1) > 0.3:
                patterns.append(f"Frequent losses during {session} session ({count}/{len(losses)})")

        # Check low confidence trades
        low_conf = sum(1 for t in losses if (t.get("swing_confidence") or 0) < 0.60)
        if low_conf >= 3 and low_conf / max(len(losses), 1) > 0.4:
            patterns.append(f"Many losses on low-confidence signals (<0.60): {low_conf}/{len(losses)}")

        # Check low ADX
        low_adx = sum(1 for t in losses if (t.get("adx") or 0) < 25)
        if low_adx >= 3 and low_adx / max(len(losses), 1) > 0.3:
            patterns.append(f"Losses with weak trend (ADX<25): {low_adx}/{len(losses)}")

        # Check funding extreme
        fund_losses = sum(1 for t in losses if t.get("funding_extreme"))
        if fund_losses >= 2:
            patterns.append(f"Losses during extreme funding: {fund_losses}/{len(losses)}")

        return patterns

    def _find_win_patterns(self, trades: List[dict]) -> List[str]:
        """Identify common patterns among winning trades."""
        wins = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
        if not wins:
            return []

        patterns = []

        # High confidence wins
        high_conf = sum(1 for t in wins if (t.get("swing_confidence") or 0) >= 0.70)
        if high_conf >= 3 and high_conf / max(len(wins), 1) > 0.4:
            patterns.append(f"High-confidence signals (≥0.70) win often: {high_conf}/{len(wins)}")

        # Regime alignment
        trend_wins = sum(1 for t in wins if t.get("regime") in ("STRONG_TREND", "TREND"))
        if trend_wins >= 3 and trend_wins / max(len(wins), 1) > 0.4:
            patterns.append(f"Trending regime wins: {trend_wins}/{len(wins)}")

        # TP hits
        tp2_hits = sum(1 for t in wins if t.get("tp2_hit"))
        if tp2_hits >= 2:
            patterns.append(f"TP2+ reached on {tp2_hits}/{len(wins)} winning trades")

        # Strong ADX
        strong_adx = sum(1 for t in wins if (t.get("adx") or 0) >= 30)
        if strong_adx >= 3 and strong_adx / max(len(wins), 1) > 0.4:
            patterns.append(f"Strong trend (ADX≥30) on wins: {strong_adx}/{len(wins)}")

        return patterns


# ═══════════════════════════════════════════════════════════════
# HYPOTHESIS GENERATOR
# ═══════════════════════════════════════════════════════════════

_HYPOTHESIS_PROMPT = (
    "You are a quantitative trading system optimizer. "
    "Analyze this performance report and generate specific, testable improvement hypotheses.\n\n"
    "Performance Report:\n{report}\n\n"
    "Meta-Learning (historical hypothesis success rates):\n{meta}\n\n"
    "Currently tunable parameters and their paths:\n{params}\n\n"
    "Generate 2-5 hypotheses as a JSON array. Each hypothesis must have:\n"
    "- observation: what you noticed in the data\n"
    "- hypothesis: your proposed change and why\n"
    "- parameter_changes: dict of config path -> new value (MUST use valid paths from the list above)\n"
    "- expected_impact: predicted effect on daily target achievement rate\n\n"
    "Rules:\n"
    "- Focus on improving DAILY TARGET ACHIEVEMENT RATE (the #1 KPI)\n"
    "- Only use parameter paths from the provided list\n"
    "- NEVER modify risk parameters (max_daily_loss, max_drawdown, etc)\n"
    "- Be specific with numbers, not vague directions\n"
    "- Prefer small, incremental changes over drastic ones\n"
    "- If a segment has <5 trades, don't base hypotheses on it\n"
    "- Parameter values must be realistic (e.g., ADX 18-30, ATR mult 0.5-4.0)\n\n"
    "Respond with ONLY valid JSON array (no markdown, no explanation outside JSON):\n"
    '[{{"observation": "...", "hypothesis": "...", "parameter_changes": {{"path.to.param": value}}, '
    '"expected_impact": "..."}}]'
)

# Tunable parameters for hypothesis generation
TUNABLE_PARAMS = {
    "candle.adx_trending_threshold": "ADX threshold for trend confirmation (default 25, range 18-35)",
    "candle.adx_ranging_threshold": "ADX below which market is ranging (default 20, range 15-25)",
    "candle.atr_tp_multiplier": "ATR multiplier for TP distance (default 2.0, range 1.0-4.0)",
    "candle.atr_sl_multiplier": "ATR multiplier for SL distance (default 1.0, range 0.5-2.0)",
    "trading.min_post_fee_rr": "Minimum post-fee risk:reward ratio (default 1.5, range 1.0-3.0)",
    "exit.tp1_atr_mult": "TP1 partial take-profit ATR mult (default 1.0, range 0.5-2.0)",
    "exit.tp2_atr_mult": "TP2 partial take-profit ATR mult (default 2.0, range 1.0-3.5)",
    "exit.chandelier_atr_mult": "Chandelier trailing stop ATR mult (default 2.0, range 1.0-3.0)",
    "exit.tp1_pct": "TP1 position percentage (default 0.40, range 0.20-0.60)",
    "exit.tp2_pct": "TP2 position percentage (default 0.35, range 0.20-0.50)",
    "candle.min_confidence": "Minimum confidence to trade (default 0.55, range 0.45-0.70)",
    "candle.macd_enabled": "Enable/disable MACD confirmation (True/False)",
    "candle.squeeze_enabled": "Enable/disable BB squeeze detection (True/False)",
}


class HypothesisGenerator:
    """
    Generates improvement hypotheses from performance analysis
    using LLM synthesis and meta-learning feedback.
    """

    def __init__(self, config=None, improvement_db: Optional[ImprovementDB] = None):
        self.config = config
        self.db = improvement_db
        self._provider = "anthropic"
        self._model = "claude-sonnet-4-5-20250514"
        self._retry_max = 2
        self._retry_delay = 2.0

        if config:
            self._provider = getattr(config, "llm_provider", self._provider)
            self._model = getattr(config, "llm_model", self._model)

    async def generate(self, report: dict) -> List[Hypothesis]:
        """Generate improvement hypotheses from a performance report."""
        if not report.get("has_data"):
            return []

        # Get meta-learning stats
        meta = {}
        if self.db:
            meta = self.db.get_meta_learning_stats()

        # Build prompt
        # Strip large fields from report for prompt
        compact_report = {k: v for k, v in report.items()
                         if k not in ("all_segments", "daily_records")}
        prompt = _HYPOTHESIS_PROMPT.format(
            report=json.dumps(compact_report, indent=2, default=str),
            meta=json.dumps(meta, indent=2, default=str),
            params=json.dumps(TUNABLE_PARAMS, indent=2),
        )

        raw = await self._call_llm(prompt)
        if raw is None:
            logger.warning("Hypothesis generation: LLM call failed")
            return []

        hypotheses = self._parse_hypotheses(raw, report)

        if self.db:
            for h in hypotheses:
                self.db.save_hypothesis(h)

        logger.info(f"Generated {len(hypotheses)} hypotheses from performance report")
        return hypotheses

    def generate_from_rules(self, report: dict) -> List[Hypothesis]:
        """
        Generate hypotheses from simple rules without LLM.
        Fallback when LLM is unavailable.
        """
        hypotheses = []
        now = time.time()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        counter = 0

        worst = report.get("worst_segments", [])
        for seg in worst[:3]:
            counter += 1
            dim = seg["dimension"]
            val = seg["value"]
            wr = seg["win_rate"]
            trades = seg["trades"]

            if trades < 5:
                continue

            h = Hypothesis(
                hypothesis_id=f"H-{date_str}-{counter:03d}",
                created_at=now,
                source="analysis",
            )

            if dim == "pair":
                h.observation = f"{val} win rate is {wr:.0%} ({trades} trades)"
                h.hypothesis = f"Require higher confidence for {val} trades"
                h.parameter_changes = json.dumps({"candle.min_confidence": 0.70})
                h.expected_impact = f"Reduce low-quality {val} trades, improve overall WR"
            elif dim == "regime":
                if val in ("WEAK", "RANGING", "ranging"):
                    h.observation = f"Trades in {val} regime win only {wr:.0%}"
                    h.hypothesis = "Raise ADX ranging threshold to block weak-trend entries"
                    h.parameter_changes = json.dumps({"candle.adx_ranging_threshold": 23})
                    h.expected_impact = "Block ~20% of trades, improve WR by ~5%"
                else:
                    continue
            elif dim == "session":
                h.observation = f"Trades during {val} session win only {wr:.0%}"
                h.hypothesis = f"Poor performance in {val} — consider tighter confidence"
                h.parameter_changes = json.dumps({"candle.min_confidence": 0.65})
                h.expected_impact = "Filter out weak signals in poor sessions"
            elif dim == "confidence" and "0.55" in val:
                h.observation = f"Low-confidence bucket ({val}) wins only {wr:.0%}"
                h.hypothesis = "Raise minimum confidence threshold"
                h.parameter_changes = json.dumps({"candle.min_confidence": 0.60})
                h.expected_impact = "Fewer trades but higher quality"
            else:
                continue

            hypotheses.append(h)

        if self.db:
            for h in hypotheses:
                self.db.save_hypothesis(h)

        return hypotheses

    def _parse_hypotheses(self, raw: str, report: dict) -> List[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            bracket_start = text.find("[")
            bracket_end = text.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                try:
                    parsed = json.loads(text[bracket_start:bracket_end + 1])
                except json.JSONDecodeError:
                    pass

        if not isinstance(parsed, list):
            logger.warning("Hypothesis LLM response is not a JSON array")
            return []

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hypotheses = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            changes = item.get("parameter_changes", {})
            if not isinstance(changes, dict):
                continue

            # Validate parameter paths
            valid_changes = {}
            for path, value in changes.items():
                if path in TUNABLE_PARAMS and path not in PROTECTED_PARAMS:
                    valid_changes[path] = value
            if not valid_changes:
                continue

            h = Hypothesis(
                hypothesis_id=f"H-{date_str}-{i + 1:03d}",
                created_at=time.time(),
                observation=str(item.get("observation", ""))[:500],
                hypothesis=str(item.get("hypothesis", ""))[:500],
                parameter_changes=json.dumps(valid_changes),
                expected_impact=str(item.get("expected_impact", ""))[:500],
                source="llm",
            )
            hypotheses.append(h)

        return hypotheses

    async def _call_llm(self, prompt: str) -> Optional[str]:
        for attempt in range(self._retry_max):
            try:
                if self._provider == "anthropic":
                    return await self._call_anthropic(prompt)
                elif self._provider == "openai":
                    return await self._call_openai(prompt)
                else:
                    return None
            except Exception as e:
                delay = self._retry_delay * (2 ** attempt)
                logger.warning(f"Hypothesis LLM attempt {attempt + 1} failed: {e}")
                if attempt < self._retry_max - 1:
                    await asyncio.sleep(delay)
        return None

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=self._model, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, temperature=0.2,
        )
        return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════
# HYPOTHESIS VALIDATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result from backtesting a hypothesis."""
    passed: bool = False
    hypothesis_id: str = ""
    baseline_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "hypothesis_id": self.hypothesis_id,
            "baseline": self.baseline_metrics,
            "test": self.test_metrics,
            "criteria": self.criteria,
            "error": self.error,
        }


class HypothesisValidator:
    """
    Backtests hypothesis parameter changes and validates against
    acceptance criteria. Primary metric: daily target achievement rate.
    """

    def __init__(self, config=None):
        self.base_config = config

    def validate(
        self,
        hypothesis: Hypothesis,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        min_days: int = MIN_BACKTEST_DAYS,
    ) -> ValidationResult:
        """
        Run backtest with hypothesis parameters and compare to baseline.

        Returns ValidationResult with pass/fail and detailed criteria.
        """
        result = ValidationResult(hypothesis_id=hypothesis.hypothesis_id)

        try:
            changes = json.loads(hypothesis.parameter_changes)
        except (json.JSONDecodeError, TypeError):
            result.error = "Invalid parameter_changes JSON"
            return result

        # Check for protected params
        for param in changes:
            if param in PROTECTED_PARAMS:
                result.error = f"Cannot modify protected parameter: {param}"
                return result

        try:
            from backtester.engine import BacktestEngine
            from backtester.data_manager import DataManager
        except ImportError:
            result.error = "Backtester module not available"
            return result

        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT"]

        # Date range: default last 60 days
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if start_date is None:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=max(min_days, 60))
            start_date = start_dt.strftime("%Y-%m-%d")

        try:
            dm = DataManager()

            # ── Baseline run (current params) ──
            baseline_config = copy.deepcopy(self.base_config or __import__("config").CONFIG)
            baseline_engine = BacktestEngine(config=baseline_config, data_manager=dm)
            baseline_result = baseline_engine.run(
                symbols=symbols, start_date=start_date, end_date=end_date,
            )

            # ── Test run (with hypothesis changes) ──
            test_config = copy.deepcopy(self.base_config or __import__("config").CONFIG)
            test_engine = BacktestEngine(config=test_config, data_manager=dm)
            test_result = test_engine.run(
                symbols=symbols, start_date=start_date, end_date=end_date,
                param_overrides=changes,
            )

        except Exception as e:
            result.error = f"Backtest failed: {e}"
            logger.error(f"Backtest validation failed for {hypothesis.hypothesis_id}: {e}")
            return result

        if not baseline_result.metrics or not test_result.metrics:
            result.error = "Backtest produced no metrics"
            return result

        bm = baseline_result.metrics
        tm = test_result.metrics

        result.baseline_metrics = _metrics_summary(bm)
        result.test_metrics = _metrics_summary(tm)

        # ── Acceptance criteria ──
        # 1. Daily target achievement improvement > 5%
        # Approximate from equity curve: count days where daily return >= target
        b_tar = self._estimate_target_rate(baseline_result)
        t_tar = self._estimate_target_rate(test_result)
        tar_improvement = t_tar - b_tar

        # 2. Win rate improvement > 3%
        wr_improvement = tm.win_rate - bm.win_rate

        # 3. Net PnL improvement > 5%
        b_pnl = bm.total_pnl
        t_pnl = tm.total_pnl
        pnl_improvement = ((t_pnl - b_pnl) / max(abs(b_pnl), 0.01)) if b_pnl != 0 else 0

        # 4. Max drawdown doesn't increase by more than 2%
        dd_increase = tm.max_drawdown_pct - bm.max_drawdown_pct

        criteria = {
            "target_achievement_improvement": {
                "baseline": round(b_tar, 4),
                "test": round(t_tar, 4),
                "improvement": round(tar_improvement, 4),
                "threshold": 0.05,
                "passed": tar_improvement >= 0.05,
            },
            "win_rate_improvement": {
                "baseline": round(bm.win_rate, 4),
                "test": round(tm.win_rate, 4),
                "improvement": round(wr_improvement, 4),
                "threshold": 0.03,
                "passed": wr_improvement >= 0.03,
            },
            "pnl_improvement": {
                "baseline_pnl": round(b_pnl, 4),
                "test_pnl": round(t_pnl, 4),
                "improvement_pct": round(pnl_improvement * 100, 2),
                "threshold_pct": 5.0,
                "passed": pnl_improvement >= 0.05,
            },
            "max_drawdown_check": {
                "baseline_dd": round(bm.max_drawdown_pct, 2),
                "test_dd": round(tm.max_drawdown_pct, 2),
                "increase": round(dd_increase, 2),
                "max_increase": 2.0,
                "passed": dd_increase <= 2.0,
            },
            "min_trades": {
                "test_trades": tm.total_trades,
                "min_required": 10,
                "passed": tm.total_trades >= 10,
            },
        }

        all_passed = all(c["passed"] for c in criteria.values())
        result.criteria = criteria
        result.passed = all_passed

        logger.info(
            f"Hypothesis {hypothesis.hypothesis_id}: "
            f"{'PASSED' if all_passed else 'FAILED'} "
            f"(TAR: {b_tar:.0%}→{t_tar:.0%}, "
            f"WR: {bm.win_rate:.0%}→{tm.win_rate:.0%}, "
            f"DD: {bm.max_drawdown_pct:.1f}%→{tm.max_drawdown_pct:.1f}%)"
        )

        return result

    def _estimate_target_rate(self, backtest_result) -> float:
        """
        Estimate daily target achievement rate from a backtest equity curve.
        Groups equity changes by day, checks if daily return >= target.
        """
        curve = backtest_result.equity_curve
        if len(curve) < 2:
            return 0.0

        config = backtest_result.config_snapshot or {}
        target_pct = 2.0  # Default
        try:
            if isinstance(config, dict):
                target_pct = config.get("daily_target", {}).get("daily_target_pct", 2.0)
        except (AttributeError, TypeError):
            pass

        # Group equity by day
        daily_returns = {}
        for ts, eq in curve:
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if day not in daily_returns:
                daily_returns[day] = {"open": eq, "close": eq}
            daily_returns[day]["close"] = eq

        if not daily_returns:
            return 0.0

        hits = 0
        total = 0
        for day, vals in daily_returns.items():
            open_eq = vals["open"]
            close_eq = vals["close"]
            if open_eq > 0:
                daily_ret = (close_eq - open_eq) / open_eq * 100
                if daily_ret >= target_pct:
                    hits += 1
                total += 1

        return hits / max(total, 1)


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT MANAGER
# ═══════════════════════════════════════════════════════════════

class ImprovementManager:
    """
    Manages the full lifecycle of improvements with safety rails:
    generate → test → queue → approve/auto-apply → review → confirm/revert
    """

    def __init__(
        self,
        config=None,
        improvement_db: Optional[ImprovementDB] = None,
        db_path: str = "",
    ):
        self.config = config
        self.db = improvement_db or ImprovementDB(db_path)
        self.analyzer = PerformanceAnalyzer()
        self.generator = HypothesisGenerator(config, self.db)
        self.validator = HypothesisValidator(config)

    # ── Full Pipeline ──────────────────────────────────────────

    async def run_daily(self, date: str = None):
        """Run daily analysis at 00:05 UTC."""
        report = self.analyzer.daily_analysis(date)

        if report.get("has_data"):
            ar = AnalysisReport(
                report_type="daily",
                period_start=report["date"],
                period_end=report["date"],
                report_data=json.dumps(report, default=str),
            )
            self.db.save_report(ar)
            logger.info(
                f"Daily analysis for {report['date']}: "
                f"{'HIT' if report['target_hit'] else 'MISSED'} target, "
                f"WR={report['win_rate']:.0%}, trades={report['total_trades']}"
            )

        # Check for applied improvements needing review
        self._check_pending_reviews()

        return report

    async def run_weekly(self, days: int = 7) -> dict:
        """
        Run full weekly pipeline:
        analyze → hypothesize → test → queue for approval
        """
        # 1. Analyze
        report = self.analyzer.weekly_analysis(days=days)
        if not report.get("has_data"):
            logger.info("No data for weekly analysis")
            return {"report": report, "hypotheses": [], "validated": []}

        ar = AnalysisReport(
            report_type="weekly",
            period_start=report["period"].split(" to ")[0],
            period_end=report["period"].split(" to ")[1],
            report_data=json.dumps(report, default=str),
        )

        # 2. Generate hypotheses
        try:
            hypotheses = await self.generator.generate(report)
        except Exception as e:
            logger.warning(f"LLM hypothesis generation failed, falling back to rules: {e}")
            hypotheses = self.generator.generate_from_rules(report)

        ar.hypotheses_generated = len(hypotheses)
        self.db.save_report(ar)

        # 3. Validate each hypothesis via backtest
        validated = []
        for h in hypotheses:
            if self.db.count_applied_this_week() >= MAX_CHANGES_PER_WEEK:
                logger.info("Max weekly changes reached, skipping remaining hypotheses")
                break

            h.status = "testing"
            self.db.save_hypothesis(h)

            vr = self.validator.validate(h)
            h.backtest_result = json.dumps(vr.test_metrics, default=str)
            h.baseline_result = json.dumps(vr.baseline_metrics, default=str)
            h.acceptance_details = json.dumps(vr.criteria, default=str)

            if vr.passed:
                h.status = "passed"
                logger.info(f"Hypothesis {h.hypothesis_id} PASSED validation")
            elif vr.error:
                h.status = "failed"
                logger.warning(f"Hypothesis {h.hypothesis_id} ERROR: {vr.error}")
            else:
                h.status = "failed"
                logger.info(f"Hypothesis {h.hypothesis_id} FAILED validation")

            self.db.save_hypothesis(h)
            validated.append({"hypothesis": h.to_dict(), "validation": vr.to_dict()})

        logger.info(
            f"Weekly pipeline: {len(hypotheses)} hypotheses generated, "
            f"{sum(1 for v in validated if v['validation']['passed'])} passed validation"
        )

        return {
            "report": report,
            "hypotheses": [h.to_dict() for h in hypotheses],
            "validated": validated,
        }

    # ── Apply / Revert ─────────────────────────────────────────

    def apply_improvement(self, hypothesis_id: str, auto: bool = False) -> Tuple[bool, str]:
        """
        Apply a passed hypothesis to the live config.

        Args:
            hypothesis_id: The hypothesis to apply
            auto: If True, came from auto-apply (requires walk-forward validation)

        Returns:
            (success, message)
        """
        h = self.db.get_hypothesis(hypothesis_id)
        if not h:
            return False, f"Hypothesis {hypothesis_id} not found"

        if h.status not in ("passed", "generated"):
            return False, f"Hypothesis status is '{h.status}', expected 'passed'"

        # Safety: max changes per week
        if self.db.count_applied_this_week() >= MAX_CHANGES_PER_WEEK:
            return False, f"Maximum {MAX_CHANGES_PER_WEEK} changes per week already reached"

        try:
            changes = json.loads(h.parameter_changes)
        except (json.JSONDecodeError, TypeError):
            return False, "Invalid parameter_changes"

        # Safety: check for protected params
        for param in changes:
            if param in PROTECTED_PARAMS:
                return False, f"Cannot modify protected parameter: {param}"

        # Save current config as baseline snapshot
        from config import CONFIG
        baseline = self._config_to_dict(CONFIG)
        self.db.save_snapshot(hypothesis_id, "baseline", baseline)

        # Apply changes to the running config
        for path, value in changes.items():
            self._set_config_value(CONFIG, path, value)

        # Save applied snapshot
        self.db.save_snapshot(hypothesis_id, "applied", changes)

        # Update hypothesis status
        h.status = "applied"
        h.applied_at = time.time()
        h.review_date = time.time() + REVIEW_PERIOD_DAYS * 86400
        self.db.save_hypothesis(h)

        logger.info(
            f"Applied improvement {hypothesis_id}: {changes} "
            f"(review in {REVIEW_PERIOD_DAYS} days)"
        )

        return True, f"Applied {hypothesis_id}. Review date: {datetime.fromtimestamp(h.review_date, tz=timezone.utc).strftime('%Y-%m-%d')}"

    def revert_improvement(self, hypothesis_id: str, reason: str = "manual") -> Tuple[bool, str]:
        """Revert a previously applied improvement."""
        h = self.db.get_hypothesis(hypothesis_id)
        if not h:
            return False, f"Hypothesis {hypothesis_id} not found"

        if h.status != "applied":
            return False, f"Hypothesis status is '{h.status}', expected 'applied'"

        # Restore baseline config
        snapshots = self.db.get_snapshots(hypothesis_id)
        baseline = None
        for snap in snapshots:
            if snap.get("action") == "baseline":
                baseline = snap.get("parameters")
                break

        if baseline:
            from config import CONFIG
            for path, value in baseline.items():
                try:
                    self._set_config_value(CONFIG, path, value)
                except Exception:
                    pass  # Best effort
            self.db.save_snapshot(hypothesis_id, "reverted", baseline)

        h.status = "reverted"
        h.reverted_at = time.time()
        h.revert_reason = reason
        self.db.save_hypothesis(h)

        logger.info(f"Reverted improvement {hypothesis_id}: {reason}")
        return True, f"Reverted {hypothesis_id}: {reason}"

    def reject_improvement(self, hypothesis_id: str) -> Tuple[bool, str]:
        """Reject a pending improvement."""
        h = self.db.get_hypothesis(hypothesis_id)
        if not h:
            return False, f"Hypothesis {hypothesis_id} not found"
        h.status = "rejected"
        self.db.save_hypothesis(h)
        return True, f"Rejected {hypothesis_id}"

    # ── Pending Reviews ────────────────────────────────────────

    def _check_pending_reviews(self):
        """Check applied improvements past their review date."""
        pending = self.db.get_pending_review()
        for h in pending:
            # Compare live performance since application
            applied_ts = h.applied_at or 0
            days_live = (time.time() - applied_ts) / 86400

            if days_live < REVIEW_PERIOD_DAYS:
                continue

            # Get recent performance
            report = self.analyzer.weekly_analysis(days=REVIEW_PERIOD_DAYS)
            if not report.get("has_data"):
                continue

            live_wr = report.get("overall_win_rate", 0)
            live_tar = report.get("target_achievement_rate", 0)

            h.live_result = json.dumps({
                "win_rate": live_wr,
                "target_achievement_rate": live_tar,
                "days_analyzed": report.get("days_total", 0),
                "total_trades": report.get("total_trades", 0),
            }, default=str)

            # Auto-revert if WR dropped below safety threshold
            if live_wr < AUTO_REVERT_WR_THRESHOLD and report.get("total_trades", 0) >= 10:
                self.revert_improvement(
                    h.hypothesis_id,
                    reason=f"Auto-revert: live WR={live_wr:.0%} < {AUTO_REVERT_WR_THRESHOLD:.0%}"
                )
                logger.warning(
                    f"Auto-reverted {h.hypothesis_id}: "
                    f"live WR={live_wr:.0%} below safety threshold"
                )
            else:
                # Mark as confirmed (keep the change)
                self.db.save_hypothesis(h)
                logger.info(
                    f"Improvement {h.hypothesis_id} review: "
                    f"live WR={live_wr:.0%}, TAR={live_tar:.0%} — keeping"
                )

    # ── Dashboard Data ─────────────────────────────────────────

    def get_dashboard_data(self) -> dict:
        """Get all data needed for the dashboard improvements section."""
        recent_reports = self.db.get_recent_reports(limit=5)
        recent_hypotheses = self.db.get_recent_hypotheses(limit=20)
        pending = self.db.get_hypotheses_by_status("passed", limit=10)
        applied = self.db.get_applied_hypotheses()
        meta = self.db.get_meta_learning_stats()

        return {
            "recent_reports": [r.to_dict() for r in recent_reports],
            "recent_hypotheses": [h.to_dict() for h in recent_hypotheses],
            "pending_approval": [h.to_dict() for h in pending],
            "applied": [h.to_dict() for h in applied],
            "meta_learning": meta,
            "changes_this_week": self.db.count_applied_this_week(),
            "max_changes_per_week": MAX_CHANGES_PER_WEEK,
        }

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _set_config_value(config, path: str, value):
        """Set a value on the config object using dot-notation path."""
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    @staticmethod
    def _config_to_dict(config) -> dict:
        """Extract tunable params from config to a flat dict."""
        result = {}
        for path in TUNABLE_PARAMS:
            parts = path.split(".")
            obj = config
            try:
                for part in parts:
                    obj = getattr(obj, part)
                result[path] = obj
            except AttributeError:
                pass
        return result


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _trade_summary(t: dict) -> dict:
    """Compact trade summary for reports."""
    return {
        "symbol": t.get("symbol"),
        "side": t.get("side"),
        "pnl_usd": round(t.get("pnl_usd", 0), 4),
        "pnl_pct": round(t.get("pnl_pct", 0), 2),
        "regime": t.get("regime"),
        "session": t.get("session"),
        "confidence": round(t.get("swing_confidence") or t.get("confidence") or 0, 3),
        "adx": round(t.get("adx") or 0, 1),
        "exit_reason": t.get("exit_reason"),
    }


def _day_summary(d: dict) -> dict:
    """Compact daily equity summary."""
    return {
        "date": d.get("date"),
        "pnl_pct": round(d.get("actual_pct", 0), 2),
        "target_hit": bool(d.get("target_hit")),
        "trades": d.get("trades", 0) or (d.get("wins", 0) + d.get("losses", 0)),
        "wins": d.get("wins", 0),
        "losses": d.get("losses", 0),
        "wr": round(d.get("wins", 0) / max(d.get("wins", 0) + d.get("losses", 0), 1), 2),
        "mode_at_close": d.get("mode_at_close"),
    }


def _metrics_summary(m) -> dict:
    """Extract key fields from PerformanceMetrics for storage."""
    return {
        "total_trades": m.total_trades,
        "win_rate": round(m.win_rate, 4),
        "total_pnl": round(m.total_pnl, 4),
        "profit_factor": round(m.profit_factor, 2),
        "max_drawdown_pct": round(m.max_drawdown_pct, 2),
        "sharpe_ratio": round(getattr(m, "sharpe_ratio", 0), 2),
        "avg_win": round(m.avg_win, 4),
        "avg_loss": round(m.avg_loss, 4),
    }
