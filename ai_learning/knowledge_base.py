"""
Knowledge Base — SQLite storage with vector embeddings for semantic search.

Stores structured knowledge extracted from YouTube videos, research papers,
and trading blogs. Supports cosine similarity search via sentence-transformers
embeddings serialized as numpy arrays in SQLite BLOB columns.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_local = threading.local()


def _default_db_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ai_knowledge.db",
    )


@dataclass
class KnowledgeEntry:
    """Single knowledge item extracted from a source."""
    id: Optional[int] = None
    source_type: str = ""           # youtube / paper / blog
    source_url: str = ""
    source_title: str = ""
    category: str = ""              # strategy / indicator / risk_rule / insight / pattern / exit / mistake
    content: str = ""               # JSON string of the extracted item
    confidence: float = 0.0
    extraction_date: float = 0.0    # Unix timestamp
    times_applied: int = 0
    success_rate: float = 0.0       # Win rate when this knowledge was applied
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    chunk_hash: str = ""            # Hash of source chunk for dedup

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "embedding"}
        d.pop("embedding", None)
        try:
            d["content"] = json.loads(d["content"]) if isinstance(d["content"], str) else d["content"]
        except (json.JSONDecodeError, TypeError):
            pass
        return d


@dataclass
class SearchResult:
    """Result from semantic search."""
    entry: KnowledgeEntry
    score: float  # Cosine similarity


class KnowledgeBase:
    """SQLite-backed knowledge store with vector similarity search."""

    def __init__(self, db_path: str = "", embedding_dim: int = 384):
        self.db_path = db_path or _default_db_path()
        self.embedding_dim = embedding_dim
        self._embedder = None  # Lazy-loaded
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(_local, "kb_conn") or _local.kb_conn is None:
            _local.kb_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _local.kb_conn.row_factory = sqlite3.Row
            _local.kb_conn.execute("PRAGMA journal_mode=WAL")
            _local.kb_conn.execute("PRAGMA synchronous=NORMAL")
        return _local.kb_conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_title TEXT DEFAULT '',
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            extraction_date REAL DEFAULT 0.0,
            times_applied INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            embedding BLOB,
            chunk_hash TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_ke_category ON knowledge_entries(category);
        CREATE INDEX IF NOT EXISTS idx_ke_source_type ON knowledge_entries(source_type);
        CREATE INDEX IF NOT EXISTS idx_ke_source_url ON knowledge_entries(source_url);
        CREATE INDEX IF NOT EXISTS idx_ke_chunk_hash ON knowledge_entries(chunk_hash);
        CREATE INDEX IF NOT EXISTS idx_ke_confidence ON knowledge_entries(confidence);

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_title TEXT DEFAULT '',
            ingested_at REAL NOT NULL,
            chunks_processed INTEGER DEFAULT 0,
            entries_created INTEGER DEFAULT 0,
            status TEXT DEFAULT 'success',
            error_message TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_il_source_url ON ingestion_log(source_url);
        """)
        conn.commit()
        logger.info(f"Knowledge base initialized at {self.db_path}")

    # ── Embedding ─────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence-transformers model loaded: all-MiniLM-L6-v2")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                return None
        return self._embedder

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        vec = embedder.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def _serialize_embedding(self, vec: Optional[np.ndarray]) -> Optional[bytes]:
        if vec is None:
            return None
        return vec.tobytes()

    def _deserialize_embedding(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    # ── CRUD ──────────────────────────────────────────────────

    def add_entry(self, entry: KnowledgeEntry) -> int:
        """Insert a knowledge entry. Returns the new row ID."""
        conn = self._get_conn()

        # Generate embedding from content if not provided
        if entry.embedding is None:
            content_text = entry.content
            if isinstance(content_text, str):
                try:
                    parsed = json.loads(content_text)
                    content_text = " ".join(str(v) for v in parsed.values() if v)
                except (json.JSONDecodeError, AttributeError):
                    pass
            entry.embedding = self.embed_text(str(content_text))

        if not entry.extraction_date:
            entry.extraction_date = time.time()

        cur = conn.execute("""
            INSERT INTO knowledge_entries
                (source_type, source_url, source_title, category, content,
                 confidence, extraction_date, times_applied, success_rate,
                 embedding, chunk_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.source_type, entry.source_url, entry.source_title,
            entry.category, entry.content, entry.confidence,
            entry.extraction_date, entry.times_applied, entry.success_rate,
            self._serialize_embedding(entry.embedding), entry.chunk_hash,
        ))
        conn.commit()
        entry.id = cur.lastrowid
        return cur.lastrowid

    def add_entries(self, entries: List[KnowledgeEntry]) -> int:
        """Batch insert. Returns count of entries added."""
        added = 0
        for entry in entries:
            self.add_entry(entry)
            added += 1
        return added

    def get_entry(self, entry_id: int) -> Optional[KnowledgeEntry]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_entry(dict(row))

    def get_entries_by_category(self, category: str, limit: int = 50) -> List[KnowledgeEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM knowledge_entries WHERE category = ? ORDER BY confidence DESC LIMIT ?",
            (category, limit),
        ).fetchall()
        return [self._row_to_entry(dict(r)) for r in rows]

    def get_entries_by_source(self, source_url: str) -> List[KnowledgeEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM knowledge_entries WHERE source_url = ? ORDER BY id",
            (source_url,),
        ).fetchall()
        return [self._row_to_entry(dict(r)) for r in rows]

    def update_application_stats(self, entry_id: int, success: bool):
        """Record that this knowledge was applied, and whether the trade succeeded."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT times_applied, success_rate FROM knowledge_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return
        times = row["times_applied"] + 1
        old_rate = row["success_rate"]
        # Running average
        new_rate = ((old_rate * (times - 1)) + (1.0 if success else 0.0)) / times
        conn.execute(
            "UPDATE knowledge_entries SET times_applied = ?, success_rate = ? WHERE id = ?",
            (times, new_rate, entry_id),
        )
        conn.commit()

    def delete_entry(self, entry_id: int):
        conn = self._get_conn()
        conn.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
        conn.commit()

    def _row_to_entry(self, row: dict) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            source_type=row["source_type"],
            source_url=row["source_url"],
            source_title=row.get("source_title", ""),
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            extraction_date=row["extraction_date"],
            times_applied=row["times_applied"],
            success_rate=row["success_rate"],
            embedding=self._deserialize_embedding(row.get("embedding")),
            chunk_hash=row.get("chunk_hash", ""),
        )

    # ── Semantic Search ───────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               min_confidence: float = 0.0) -> List[SearchResult]:
        """
        Semantic search over the knowledge base using cosine similarity.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            category: Optional category filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of SearchResult ordered by similarity score descending
        """
        query_vec = self.embed_text(query)
        if query_vec is None:
            # Fall back to keyword search if embeddings not available
            return self._keyword_search(query, top_k, category, min_confidence)

        conn = self._get_conn()

        sql = "SELECT * FROM knowledge_entries WHERE confidence >= ?"
        params: list = [min_confidence]
        if category:
            sql += " AND category = ?"
            params.append(category)

        rows = conn.execute(sql, params).fetchall()

        results: List[SearchResult] = []
        for row in rows:
            row_dict = dict(row)
            entry_vec = self._deserialize_embedding(row_dict.get("embedding"))
            if entry_vec is None:
                continue
            score = float(np.dot(query_vec, entry_vec))
            entry = self._row_to_entry(row_dict)
            results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _keyword_search(self, query: str, top_k: int,
                        category: Optional[str], min_confidence: float) -> List[SearchResult]:
        """Fallback keyword search when embeddings are unavailable."""
        conn = self._get_conn()
        sql = "SELECT * FROM knowledge_entries WHERE confidence >= ?"
        params: list = [min_confidence]
        if category:
            sql += " AND category = ?"
            params.append(category)
        rows = conn.execute(sql, params).fetchall()

        keywords = query.lower().split()
        results = []
        for row in rows:
            row_dict = dict(row)
            text = (row_dict["content"] + " " + row_dict.get("source_title", "")).lower()
            score = sum(1 for kw in keywords if kw in text) / max(len(keywords), 1)
            if score > 0:
                entry = self._row_to_entry(row_dict)
                results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ── Deduplication ─────────────────────────────────────────

    def is_duplicate(self, content_text: str, threshold: float = 0.85) -> bool:
        """
        Check if content is semantically duplicate of existing entries.

        Args:
            content_text: Text to check
            threshold: Cosine similarity threshold (default 0.85)

        Returns:
            True if a near-duplicate exists
        """
        query_vec = self.embed_text(content_text)
        if query_vec is None:
            return False

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT embedding FROM knowledge_entries WHERE embedding IS NOT NULL"
        ).fetchall()

        for row in rows:
            existing_vec = self._deserialize_embedding(row["embedding"])
            if existing_vec is not None:
                sim = float(np.dot(query_vec, existing_vec))
                if sim >= threshold:
                    return True
        return False

    def find_similar(self, content_text: str, threshold: float = 0.85,
                     limit: int = 5) -> List[SearchResult]:
        """Find entries similar to given text above threshold."""
        query_vec = self.embed_text(content_text)
        if query_vec is None:
            return []

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM knowledge_entries WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            existing_vec = self._deserialize_embedding(row_dict.get("embedding"))
            if existing_vec is not None:
                sim = float(np.dot(query_vec, existing_vec))
                if sim >= threshold:
                    entry = self._row_to_entry(row_dict)
                    results.append(SearchResult(entry=entry, score=sim))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ── Ingestion Log ─────────────────────────────────────────

    def log_ingestion(self, source_type: str, source_url: str,
                      source_title: str = "", chunks: int = 0,
                      entries: int = 0, status: str = "success",
                      error: str = ""):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO ingestion_log
                (source_type, source_url, source_title, ingested_at,
                 chunks_processed, entries_created, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (source_type, source_url, source_title, time.time(),
              chunks, entries, status, error))
        conn.commit()

    def was_ingested(self, source_url: str) -> bool:
        """Check if a URL has already been successfully ingested."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM ingestion_log WHERE source_url = ? AND status = 'success' LIMIT 1",
            (source_url,),
        ).fetchone()
        return row is not None

    def get_ingestion_history(self, limit: int = 50) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM ingestion_log ORDER BY ingested_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]

        categories = conn.execute("""
            SELECT category, COUNT(*) as cnt
            FROM knowledge_entries GROUP BY category ORDER BY cnt DESC
        """).fetchall()

        sources = conn.execute("""
            SELECT source_type, COUNT(*) as cnt
            FROM knowledge_entries GROUP BY source_type ORDER BY cnt DESC
        """).fetchall()

        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM knowledge_entries"
        ).fetchone()[0] or 0.0

        top_applied = conn.execute("""
            SELECT id, source_title, category, times_applied, success_rate
            FROM knowledge_entries WHERE times_applied > 0
            ORDER BY times_applied DESC LIMIT 10
        """).fetchall()

        ingestions = conn.execute(
            "SELECT COUNT(*) FROM ingestion_log WHERE status = 'success'"
        ).fetchone()[0]

        return {
            "total_entries": total,
            "categories": {r["category"]: r["cnt"] for r in categories},
            "source_types": {r["source_type"]: r["cnt"] for r in sources},
            "avg_confidence": round(avg_confidence, 3),
            "total_ingestions": ingestions,
            "top_applied": [dict(r) for r in top_applied],
        }
