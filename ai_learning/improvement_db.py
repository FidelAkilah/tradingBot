"""
Improvement History Database — tracks hypotheses, parameter changes,
and their outcomes over time for meta-learning.

Uses the same ai_knowledge.db (WAL mode) as the knowledge base.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_local = threading.local()

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ai_knowledge.db",
)


def _get_conn(db_path: str = "") -> sqlite3.Connection:
    path = db_path or _DEFAULT_DB
    key = f"imp_conn_{path}"
    conn = getattr(_local, key, None)
    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        setattr(_local, key, conn)
    return conn


# ── Dataclasses ────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """A generated improvement hypothesis."""
    id: Optional[int] = None
    hypothesis_id: str = ""           # H-YYYY-MM-DD-NNN
    created_at: float = 0.0
    observation: str = ""
    hypothesis: str = ""
    parameter_changes: str = ""       # JSON dict
    expected_impact: str = ""
    source: str = "analysis"          # analysis / kb / manual
    status: str = "generated"         # generated / testing / passed / failed / applied / reverted / rejected
    backtest_result: str = ""         # JSON summary
    baseline_result: str = ""         # JSON summary
    acceptance_details: str = ""      # JSON: which criteria passed/failed
    applied_at: Optional[float] = None
    review_date: Optional[float] = None  # 7 days after apply
    live_result: str = ""             # JSON: live performance after apply
    reverted_at: Optional[float] = None
    revert_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("parameter_changes", "backtest_result", "baseline_result",
                     "acceptance_details", "live_result"):
            try:
                d[key] = json.loads(d[key]) if d[key] else {}
            except (json.JSONDecodeError, TypeError):
                pass
        return d


@dataclass
class AnalysisReport:
    """Stored performance analysis report."""
    id: Optional[int] = None
    created_at: float = 0.0
    report_type: str = "daily"       # daily / weekly
    period_start: str = ""
    period_end: str = ""
    report_data: str = ""            # Full JSON report
    hypotheses_generated: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        try:
            d["report_data"] = json.loads(d["report_data"]) if d["report_data"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        return d


@dataclass
class ParameterSnapshot:
    """Version-controlled parameter state."""
    id: Optional[int] = None
    created_at: float = 0.0
    hypothesis_id: str = ""
    action: str = ""                  # applied / reverted / baseline
    parameters: str = ""              # JSON: full config snapshot or diff


# ── Database Manager ───────────────────────────────────────────

class ImprovementDB:
    """SQLite storage for the self-improvement meta-learning system."""

    def __init__(self, db_path: str = ""):
        self.db_path = db_path
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        return _get_conn(self.db_path)

    def _init_tables(self):
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS improvement_hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL,
            observation TEXT NOT NULL DEFAULT '',
            hypothesis TEXT NOT NULL DEFAULT '',
            parameter_changes TEXT NOT NULL DEFAULT '{}',
            expected_impact TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'analysis',
            status TEXT NOT NULL DEFAULT 'generated',
            backtest_result TEXT NOT NULL DEFAULT '{}',
            baseline_result TEXT NOT NULL DEFAULT '{}',
            acceptance_details TEXT NOT NULL DEFAULT '{}',
            applied_at REAL,
            review_date REAL,
            live_result TEXT NOT NULL DEFAULT '{}',
            reverted_at REAL,
            revert_reason TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS analysis_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            report_type TEXT NOT NULL DEFAULT 'daily',
            period_start TEXT NOT NULL DEFAULT '',
            period_end TEXT NOT NULL DEFAULT '',
            report_data TEXT NOT NULL DEFAULT '{}',
            hypotheses_generated INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS parameter_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            hypothesis_id TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL DEFAULT '',
            parameters TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_hyp_status ON improvement_hypotheses(status);
        CREATE INDEX IF NOT EXISTS idx_hyp_created ON improvement_hypotheses(created_at);
        CREATE INDEX IF NOT EXISTS idx_report_type ON analysis_reports(report_type);
        """)
        conn.commit()

    # ── Hypotheses ──────────────────────────────────────────────

    def save_hypothesis(self, h: Hypothesis) -> int:
        conn = self._get_conn()
        if h.id:
            conn.execute("""
                UPDATE improvement_hypotheses SET
                    observation=?, hypothesis=?, parameter_changes=?,
                    expected_impact=?, source=?, status=?,
                    backtest_result=?, baseline_result=?, acceptance_details=?,
                    applied_at=?, review_date=?, live_result=?,
                    reverted_at=?, revert_reason=?
                WHERE id=?
            """, (h.observation, h.hypothesis, h.parameter_changes,
                  h.expected_impact, h.source, h.status,
                  h.backtest_result, h.baseline_result, h.acceptance_details,
                  h.applied_at, h.review_date, h.live_result,
                  h.reverted_at, h.revert_reason, h.id))
        else:
            h.created_at = h.created_at or time.time()
            cur = conn.execute("""
                INSERT INTO improvement_hypotheses
                (hypothesis_id, created_at, observation, hypothesis,
                 parameter_changes, expected_impact, source, status,
                 backtest_result, baseline_result, acceptance_details,
                 applied_at, review_date, live_result, reverted_at, revert_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (h.hypothesis_id, h.created_at, h.observation, h.hypothesis,
                  h.parameter_changes, h.expected_impact, h.source, h.status,
                  h.backtest_result, h.baseline_result, h.acceptance_details,
                  h.applied_at, h.review_date, h.live_result,
                  h.reverted_at, h.revert_reason))
            h.id = cur.lastrowid
        conn.commit()
        return h.id

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM improvement_hypotheses WHERE hypothesis_id=?",
            (hypothesis_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_hypothesis(row)

    def get_hypotheses_by_status(self, status: str, limit: int = 50) -> List[Hypothesis]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM improvement_hypotheses WHERE status=? ORDER BY created_at DESC LIMIT ?",
            (status, limit)
        ).fetchall()
        return [self._row_to_hypothesis(r) for r in rows]

    def get_recent_hypotheses(self, limit: int = 20) -> List[Hypothesis]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM improvement_hypotheses ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_hypothesis(r) for r in rows]

    def get_applied_hypotheses(self) -> List[Hypothesis]:
        return self.get_hypotheses_by_status("applied")

    def get_pending_review(self) -> List[Hypothesis]:
        """Get applied hypotheses past their review date."""
        conn = self._get_conn()
        now = time.time()
        rows = conn.execute(
            "SELECT * FROM improvement_hypotheses WHERE status='applied' AND review_date <= ?",
            (now,)
        ).fetchall()
        return [self._row_to_hypothesis(r) for r in rows]

    def count_applied_this_week(self) -> int:
        """Count improvements applied in the last 7 days."""
        conn = self._get_conn()
        week_ago = time.time() - 7 * 86400
        row = conn.execute(
            "SELECT COUNT(*) as c FROM improvement_hypotheses WHERE applied_at > ? AND status IN ('applied', 'reverted')",
            (week_ago,)
        ).fetchone()
        return row["c"]

    def get_meta_learning_stats(self) -> dict:
        """Get historical success rates by hypothesis type for meta-learning."""
        conn = self._get_conn()

        total = conn.execute(
            "SELECT COUNT(*) as c FROM improvement_hypotheses WHERE status IN ('applied','reverted')"
        ).fetchone()["c"]
        successful = conn.execute(
            "SELECT COUNT(*) as c FROM improvement_hypotheses WHERE status='applied' AND reverted_at IS NULL AND review_date < ?",
            (time.time(),)
        ).fetchone()["c"]

        # Success by parameter type
        rows = conn.execute("""
            SELECT parameter_changes, status, reverted_at, review_date
            FROM improvement_hypotheses
            WHERE status IN ('applied', 'reverted')
        """).fetchall()

        param_stats: Dict[str, Dict[str, int]] = {}
        for row in rows:
            try:
                changes = json.loads(row["parameter_changes"])
            except (json.JSONDecodeError, TypeError):
                continue
            for param in changes:
                key = param.split(".")[0] if "." in param else param
                if key not in param_stats:
                    param_stats[key] = {"total": 0, "success": 0}
                param_stats[key]["total"] += 1
                if row["status"] == "applied" and row["reverted_at"] is None:
                    if row["review_date"] and row["review_date"] < time.time():
                        param_stats[key]["success"] += 1

        return {
            "total_hypotheses_tested": total,
            "successful": successful,
            "success_rate": successful / max(total, 1),
            "by_parameter_type": {
                k: {**v, "rate": v["success"] / max(v["total"], 1)}
                for k, v in param_stats.items()
            },
        }

    @staticmethod
    def _row_to_hypothesis(row) -> Hypothesis:
        return Hypothesis(
            id=row["id"],
            hypothesis_id=row["hypothesis_id"],
            created_at=row["created_at"],
            observation=row["observation"],
            hypothesis=row["hypothesis"],
            parameter_changes=row["parameter_changes"],
            expected_impact=row["expected_impact"],
            source=row["source"],
            status=row["status"],
            backtest_result=row["backtest_result"],
            baseline_result=row["baseline_result"],
            acceptance_details=row["acceptance_details"],
            applied_at=row["applied_at"],
            review_date=row["review_date"],
            live_result=row["live_result"],
            reverted_at=row["reverted_at"],
            revert_reason=row["revert_reason"],
        )

    # ── Analysis Reports ────────────────────────────────────────

    def save_report(self, report: AnalysisReport) -> int:
        conn = self._get_conn()
        report.created_at = report.created_at or time.time()
        cur = conn.execute("""
            INSERT INTO analysis_reports
            (created_at, report_type, period_start, period_end,
             report_data, hypotheses_generated)
            VALUES (?,?,?,?,?,?)
        """, (report.created_at, report.report_type, report.period_start,
              report.period_end, report.report_data, report.hypotheses_generated))
        report.id = cur.lastrowid
        conn.commit()
        return report.id

    def get_recent_reports(self, report_type: str = None, limit: int = 10) -> List[AnalysisReport]:
        conn = self._get_conn()
        if report_type:
            rows = conn.execute(
                "SELECT * FROM analysis_reports WHERE report_type=? ORDER BY created_at DESC LIMIT ?",
                (report_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM analysis_reports ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [AnalysisReport(
            id=r["id"], created_at=r["created_at"], report_type=r["report_type"],
            period_start=r["period_start"], period_end=r["period_end"],
            report_data=r["report_data"], hypotheses_generated=r["hypotheses_generated"],
        ) for r in rows]

    # ── Parameter Snapshots ─────────────────────────────────────

    def save_snapshot(self, hypothesis_id: str, action: str, parameters: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO parameter_snapshots (created_at, hypothesis_id, action, parameters)
            VALUES (?,?,?,?)
        """, (time.time(), hypothesis_id, action, json.dumps(parameters, default=str)))
        conn.commit()

    def get_snapshots(self, hypothesis_id: str) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM parameter_snapshots WHERE hypothesis_id=? ORDER BY created_at",
            (hypothesis_id,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["parameters"] = json.loads(d["parameters"])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(d)
        return results

    def get_latest_baseline(self) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT parameters FROM parameter_snapshots WHERE action='baseline' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                return json.loads(row["parameters"])
            except (json.JSONDecodeError, TypeError):
                pass
        return None
