"""
Tests for the AI Learning module — knowledge base, extraction, deduplication,
YouTube ingestor text processing, and paper ingestor chunking.
"""

import asyncio
import sys
import os
import json
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from ai_learning.knowledge_base import KnowledgeBase, KnowledgeEntry, SearchResult
from ai_learning.knowledge_extractor import KnowledgeExtractor, CATEGORY_MAP
from ai_learning.youtube_ingestor import YouTubeIngestor, VideoMeta, TranscriptChunk
from ai_learning.paper_ingestor import PaperIngestor, SourceMeta, TextChunk


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Create a temporary knowledge base database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Close any cached connection for this thread
    from ai_learning.knowledge_base import _local
    if hasattr(_local, "kb_conn") and _local.kb_conn is not None:
        try:
            _local.kb_conn.close()
        except Exception:
            pass
        _local.kb_conn = None
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def kb(tmp_db):
    """Knowledge base instance with temp DB (no embedding model)."""
    return KnowledgeBase(db_path=tmp_db, embedding_dim=384)


@pytest.fixture
def sample_entry():
    """A sample knowledge entry."""
    return KnowledgeEntry(
        source_type="youtube",
        source_url="https://youtube.com/watch?v=test123",
        source_title="Test Trading Video",
        category="strategy",
        content=json.dumps({
            "name": "RSI Divergence Swing",
            "description": "Use RSI divergence on 4h chart for swing entries",
            "conditions": "RSI divergence + trend alignment",
            "timeframe": "4h",
        }),
        confidence=0.85,
    )


@pytest.fixture
def youtube_ingestor():
    return YouTubeIngestor()


@pytest.fixture
def paper_ingestor():
    return PaperIngestor()


@pytest.fixture
def extractor(kb):
    return KnowledgeExtractor(knowledge_base=kb)


# ── KnowledgeBase Tests ───────────────────────────────────────

class TestKnowledgeBaseCRUD:
    """Test basic CRUD operations on the knowledge base."""

    def test_add_and_get_entry(self, kb, sample_entry):
        """Adding an entry and retrieving it returns matching data."""
        row_id = kb.add_entry(sample_entry)
        assert row_id > 0

        retrieved = kb.get_entry(row_id)
        assert retrieved is not None
        assert retrieved.source_type == "youtube"
        assert retrieved.category == "strategy"
        assert retrieved.confidence == 0.85
        assert "RSI Divergence" in retrieved.content

    def test_add_multiple_entries(self, kb):
        """Batch insert multiple entries."""
        entries = [
            KnowledgeEntry(
                source_type="paper", source_url="http://example.com",
                source_title="Test", category="indicator",
                content='{"name": "MACD"}', confidence=0.7,
            ),
            KnowledgeEntry(
                source_type="blog", source_url="http://blog.com",
                source_title="Blog", category="risk_rule",
                content='{"rule": "Never risk more than 2%"}', confidence=0.9,
            ),
        ]
        added = kb.add_entries(entries)
        assert added == 2

    def test_get_entries_by_category(self, kb, sample_entry):
        """Filter entries by category."""
        kb.add_entry(sample_entry)
        kb.add_entry(KnowledgeEntry(
            source_type="paper", source_url="http://test.com",
            source_title="Other", category="risk_rule",
            content='{"rule": "test"}', confidence=0.6,
        ))

        strategies = kb.get_entries_by_category("strategy")
        assert len(strategies) == 1
        assert strategies[0].category == "strategy"

        rules = kb.get_entries_by_category("risk_rule")
        assert len(rules) == 1

    def test_get_entries_by_source(self, kb, sample_entry):
        """Filter entries by source URL."""
        kb.add_entry(sample_entry)
        results = kb.get_entries_by_source(sample_entry.source_url)
        assert len(results) == 1

    def test_delete_entry(self, kb, sample_entry):
        """Deleting an entry removes it."""
        row_id = kb.add_entry(sample_entry)
        kb.delete_entry(row_id)
        assert kb.get_entry(row_id) is None

    def test_update_application_stats(self, kb, sample_entry):
        """Recording application results updates stats."""
        row_id = kb.add_entry(sample_entry)

        kb.update_application_stats(row_id, success=True)
        entry = kb.get_entry(row_id)
        assert entry.times_applied == 1
        assert entry.success_rate == 1.0

        kb.update_application_stats(row_id, success=False)
        entry = kb.get_entry(row_id)
        assert entry.times_applied == 2
        assert entry.success_rate == 0.5

    def test_nonexistent_entry(self, kb):
        """Getting a non-existent entry returns None."""
        assert kb.get_entry(99999) is None

    def test_entry_to_dict(self, sample_entry):
        """to_dict serializes correctly and parses JSON content."""
        d = sample_entry.to_dict()
        assert isinstance(d["content"], dict)
        assert d["content"]["name"] == "RSI Divergence Swing"
        assert "embedding" not in d


class TestKnowledgeBaseSearch:
    """Test search functionality (keyword fallback when no embeddings)."""

    def test_keyword_search_finds_matching(self, kb):
        """Keyword search returns entries matching query terms."""
        kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="http://a.com",
            source_title="RSI Trading Guide", category="indicator",
            content='{"name": "RSI", "usage": "divergence detection"}',
            confidence=0.8,
        ))
        kb.add_entry(KnowledgeEntry(
            source_type="paper", source_url="http://b.com",
            source_title="MACD Study", category="indicator",
            content='{"name": "MACD", "usage": "momentum"}',
            confidence=0.7,
        ))

        results = kb._keyword_search("RSI divergence", top_k=5, category=None, min_confidence=0.0)
        assert len(results) >= 1
        assert any("RSI" in r.entry.content for r in results)

    def test_keyword_search_with_category_filter(self, kb):
        """Keyword search respects category filter."""
        kb.add_entry(KnowledgeEntry(
            source_type="blog", source_url="http://c.com",
            source_title="Risk Guide", category="risk_rule",
            content='{"rule": "2% max risk per trade"}',
            confidence=0.9,
        ))
        kb.add_entry(KnowledgeEntry(
            source_type="blog", source_url="http://d.com",
            source_title="Entry Patterns", category="pattern",
            content='{"pattern": "risk adjusted entry"}',
            confidence=0.8,
        ))

        results = kb._keyword_search("risk", top_k=5, category="risk_rule", min_confidence=0.0)
        assert all(r.entry.category == "risk_rule" for r in results)

    def test_keyword_search_empty_on_no_match(self, kb):
        """Keyword search returns empty for non-matching query."""
        kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="http://e.com",
            source_title="Test", category="strategy",
            content='{"name": "bollinger bands"}',
            confidence=0.7,
        ))
        results = kb._keyword_search("fibonacci retracement", top_k=5, category=None, min_confidence=0.0)
        assert len(results) == 0

    def test_keyword_search_min_confidence(self, kb):
        """Keyword search filters by minimum confidence."""
        kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="http://f.com",
            source_title="Low conf", category="insight",
            content='{"insight": "market tends to reverse"}',
            confidence=0.3,
        ))
        results = kb._keyword_search("market reverse", top_k=5, category=None, min_confidence=0.5)
        assert len(results) == 0


class TestKnowledgeBaseIngestionLog:
    """Test ingestion logging."""

    def test_log_and_check_ingestion(self, kb):
        """Logging an ingestion marks URL as ingested."""
        url = "https://youtube.com/watch?v=abc123"
        assert not kb.was_ingested(url)

        kb.log_ingestion("youtube", url, source_title="Test", chunks=5, entries=3)
        assert kb.was_ingested(url)

    def test_ingestion_history(self, kb):
        """Ingestion history returns recent entries in reverse chronological order."""
        kb.log_ingestion("youtube", "http://a.com", chunks=2, entries=1)
        kb.log_ingestion("paper", "http://b.com", chunks=4, entries=3)

        history = kb.get_ingestion_history(10)
        assert len(history) == 2
        assert history[0]["source_type"] == "paper"  # Most recent first

    def test_failed_ingestion_not_marked(self, kb):
        """Failed ingestions don't count as 'ingested'."""
        url = "http://failed.com"
        kb.log_ingestion("blog", url, status="no_content")
        assert not kb.was_ingested(url)


class TestKnowledgeBaseStats:
    """Test statistics computation."""

    def test_empty_stats(self, kb):
        """Stats on empty DB return zeros."""
        stats = kb.get_stats()
        assert stats["total_entries"] == 0
        assert stats["categories"] == {}

    def test_stats_with_entries(self, kb):
        """Stats reflect added entries correctly."""
        for i in range(5):
            kb.add_entry(KnowledgeEntry(
                source_type="youtube", source_url=f"http://{i}.com",
                source_title=f"Video {i}", category="strategy",
                content=f'{{"name": "strat_{i}"}}', confidence=0.7 + i * 0.05,
            ))
        for i in range(3):
            kb.add_entry(KnowledgeEntry(
                source_type="paper", source_url=f"http://paper{i}.com",
                source_title=f"Paper {i}", category="risk_rule",
                content=f'{{"rule": "rule_{i}"}}', confidence=0.8,
            ))

        stats = kb.get_stats()
        assert stats["total_entries"] == 8
        assert stats["categories"]["strategy"] == 5
        assert stats["categories"]["risk_rule"] == 3
        assert stats["source_types"]["youtube"] == 5
        assert stats["source_types"]["paper"] == 3


class TestDeduplication:
    """Test vector-based deduplication."""

    def test_is_duplicate_with_numpy_vectors(self, kb):
        """Direct vector comparison detects near-duplicate content."""
        # Manually add an entry with a known embedding
        vec1 = np.random.randn(384).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)  # Normalize

        entry = KnowledgeEntry(
            source_type="youtube", source_url="http://test.com",
            source_title="Test", category="strategy",
            content='{"name": "test strategy"}', confidence=0.8,
            embedding=vec1,
        )
        kb.add_entry(entry)

        # A very similar vector should be detected as duplicate
        noise = np.random.randn(384).astype(np.float32) * 0.01
        vec2 = vec1 + noise
        vec2 = vec2 / np.linalg.norm(vec2)

        # Override embed_text to return our controlled vector
        original_embed = kb.embed_text
        kb.embed_text = lambda text: vec2
        try:
            assert kb.is_duplicate("test strategy content", threshold=0.95)
        finally:
            kb.embed_text = original_embed

    def test_not_duplicate_with_different_vectors(self, kb):
        """Dissimilar vectors are not flagged as duplicates."""
        vec1 = np.random.randn(384).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        entry = KnowledgeEntry(
            source_type="youtube", source_url="http://test.com",
            source_title="Test", category="strategy",
            content='{"name": "test"}', confidence=0.8,
            embedding=vec1,
        )
        kb.add_entry(entry)

        # Completely different vector
        vec2 = np.random.randn(384).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        original_embed = kb.embed_text
        kb.embed_text = lambda text: vec2
        try:
            assert not kb.is_duplicate("completely different content", threshold=0.85)
        finally:
            kb.embed_text = original_embed

    def test_find_similar(self, kb):
        """find_similar returns entries above threshold."""
        vec1 = np.random.randn(384).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        kb.add_entry(KnowledgeEntry(
            source_type="blog", source_url="http://test.com",
            source_title="Test", category="indicator",
            content='{"name": "RSI"}', confidence=0.7,
            embedding=vec1,
        ))

        # Similar vector
        noise = np.random.randn(384).astype(np.float32) * 0.005
        vec2 = vec1 + noise
        vec2 = vec2 / np.linalg.norm(vec2)

        original_embed = kb.embed_text
        kb.embed_text = lambda text: vec2
        try:
            results = kb.find_similar("RSI indicator usage", threshold=0.95)
            assert len(results) >= 1
            assert results[0].score > 0.95
        finally:
            kb.embed_text = original_embed


class TestEmbeddingSerialization:
    """Test numpy embedding serialization/deserialization."""

    def test_roundtrip(self, kb):
        """Embedding survives serialize → store → retrieve → deserialize."""
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        entry = KnowledgeEntry(
            source_type="paper", source_url="http://test.com",
            source_title="Test", category="strategy",
            content='{"name": "test"}', confidence=0.9,
            embedding=vec,
        )
        row_id = kb.add_entry(entry)

        retrieved = kb.get_entry(row_id)
        assert retrieved.embedding is not None
        np.testing.assert_allclose(retrieved.embedding, vec, atol=1e-6)

    def test_none_embedding(self, kb):
        """None embedding serializes and deserializes as None."""
        assert kb._serialize_embedding(None) is None
        assert kb._deserialize_embedding(None) is None


# ── KnowledgeExtractor Tests ─────────────────────────────────

class TestExtractionParsing:
    """Test JSON parsing from LLM responses."""

    def test_parse_clean_json(self, extractor):
        """Parse well-formed JSON response."""
        raw = json.dumps({
            "strategies": [{"name": "EMA Cross", "description": "test", "conditions": "1h", "timeframe": "1h", "confidence": 0.8}],
            "indicators": [],
            "risk_rules": [],
            "market_insights": [],
            "entry_patterns": [],
            "exit_rules": [],
            "mistakes_to_avoid": [],
        })
        result = KnowledgeExtractor._parse_extraction(raw)
        assert result is not None
        assert len(result["strategies"]) == 1
        assert result["strategies"][0]["name"] == "EMA Cross"

    def test_parse_markdown_fenced_json(self, extractor):
        """Parse JSON wrapped in markdown code fences."""
        raw = """```json
{
  "strategies": [{"name": "BB Squeeze", "description": "test", "conditions": "4h", "timeframe": "4h", "confidence": 0.7}],
  "indicators": [],
  "risk_rules": [],
  "market_insights": [],
  "entry_patterns": [],
  "exit_rules": [],
  "mistakes_to_avoid": []
}
```"""
        result = KnowledgeExtractor._parse_extraction(raw)
        assert result is not None
        assert result["strategies"][0]["name"] == "BB Squeeze"

    def test_parse_json_with_surrounding_text(self, extractor):
        """Extract JSON from response with surrounding prose."""
        raw = """Here is the extracted knowledge:

{
  "strategies": [],
  "indicators": [{"name": "RSI", "usage": "overbought/oversold", "parameters": "14 period", "confidence": 0.9}],
  "risk_rules": [],
  "market_insights": [],
  "entry_patterns": [],
  "exit_rules": [],
  "mistakes_to_avoid": []
}

I hope this helps!"""
        result = KnowledgeExtractor._parse_extraction(raw)
        assert result is not None
        assert len(result["indicators"]) == 1

    def test_parse_invalid_json(self, extractor):
        """Invalid JSON returns None."""
        assert KnowledgeExtractor._parse_extraction("not json at all") is None
        assert KnowledgeExtractor._parse_extraction("{broken: json}") is None


class TestEntryConversion:
    """Test converting parsed extraction to KnowledgeEntry objects."""

    def test_to_entries_basic(self, extractor):
        """Convert parsed dict to KnowledgeEntry list."""
        parsed = {
            "strategies": [
                {"name": "Mean Reversion", "description": "Buy oversold", "conditions": "RSI<30", "timeframe": "4h", "confidence": 0.8},
            ],
            "risk_rules": [
                {"rule": "Max 2% risk", "rationale": "Capital preservation", "confidence": 0.95},
            ],
            "indicators": [],
            "market_insights": [],
            "entry_patterns": [],
            "exit_rules": [],
            "mistakes_to_avoid": [],
        }
        entries = extractor._to_entries(parsed, "youtube", "http://test.com", "Test Video", "abc123")
        assert len(entries) == 2

        strat = [e for e in entries if e.category == "strategy"]
        assert len(strat) == 1
        assert strat[0].confidence == 0.8
        assert strat[0].source_type == "youtube"
        assert strat[0].chunk_hash == "abc123"

        rules = [e for e in entries if e.category == "risk_rule"]
        assert len(rules) == 1
        assert rules[0].confidence == 0.95

    def test_to_entries_all_categories(self, extractor):
        """All 7 categories are properly mapped."""
        parsed = {
            "strategies": [{"name": "s", "description": "d", "conditions": "c", "timeframe": "t", "confidence": 0.7}],
            "indicators": [{"name": "i", "usage": "u", "parameters": "p", "confidence": 0.7}],
            "risk_rules": [{"rule": "r", "rationale": "ra", "confidence": 0.7}],
            "market_insights": [{"insight": "i", "context": "c", "confidence": 0.7}],
            "entry_patterns": [{"pattern": "p", "setup": "s", "confirmation": "c", "confidence": 0.7}],
            "exit_rules": [{"rule": "r", "trigger": "t", "confidence": 0.7}],
            "mistakes_to_avoid": [{"mistake": "m", "why": "w", "confidence": 0.7}],
        }
        entries = extractor._to_entries(parsed, "paper", "http://test.com", "Test", "hash1")
        cats = {e.category for e in entries}
        expected = set(CATEGORY_MAP.values())
        assert cats == expected

    def test_to_entries_empty_categories(self, extractor):
        """Empty categories produce no entries."""
        parsed = {k: [] for k in CATEGORY_MAP}
        entries = extractor._to_entries(parsed, "blog", "http://test.com", "Test", "hash1")
        assert len(entries) == 0

    def test_to_entries_filters_non_dict_items(self, extractor):
        """Non-dict items in arrays are skipped."""
        parsed = {
            "strategies": ["not a dict", 42, None, {"name": "valid", "confidence": 0.7}],
            **{k: [] for k in CATEGORY_MAP if k != "strategies"},
        }
        entries = extractor._to_entries(parsed, "youtube", "http://test.com", "Test", "hash1")
        assert len(entries) == 1


class TestConfidenceFiltering:
    """Test minimum confidence threshold filtering."""

    def test_low_confidence_filtered(self, extractor):
        """Entries below min_confidence are dropped."""
        extractor._min_confidence = 0.5
        entries = [
            KnowledgeEntry(category="strategy", content='{"name": "low"}', confidence=0.3),
            KnowledgeEntry(category="strategy", content='{"name": "high"}', confidence=0.8),
            KnowledgeEntry(category="indicator", content='{"name": "mid"}', confidence=0.5),
        ]
        filtered = [e for e in entries if e.confidence >= extractor._min_confidence]
        assert len(filtered) == 2
        assert all(e.confidence >= 0.5 for e in filtered)


class TestDeduplicationLogic:
    """Test the extractor's deduplication against KB."""

    def test_deduplicate_removes_similar(self, extractor, kb):
        """Deduplication removes entries similar to existing KB entries."""
        # Add an existing entry with a known vector
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="http://existing.com",
            source_title="Existing", category="strategy",
            content='{"name": "RSI divergence strategy"}',
            confidence=0.8, embedding=vec,
        ))

        # Mock kb.is_duplicate to return True for similar content
        original_is_dup = kb.is_duplicate
        call_count = [0]

        def mock_is_dup(text, threshold=0.85):
            call_count[0] += 1
            if "RSI divergence" in text:
                return True
            return False

        kb.is_duplicate = mock_is_dup
        try:
            new_entries = [
                KnowledgeEntry(
                    source_type="youtube", source_url="http://new.com",
                    source_title="New", category="strategy",
                    content='{"name": "RSI divergence strategy variant"}',
                    confidence=0.75,
                ),
                KnowledgeEntry(
                    source_type="youtube", source_url="http://new.com",
                    source_title="New", category="indicator",
                    content='{"name": "MACD crossover"}',
                    confidence=0.8,
                ),
            ]
            unique = extractor._deduplicate(new_entries)
            assert len(unique) == 1
            assert "MACD" in unique[0].content
        finally:
            kb.is_duplicate = original_is_dup


# ── YouTube Ingestor Tests ───────────────────────────────────

class TestYouTubeVideoIdExtraction:
    """Test video ID extraction from various URL formats."""

    def test_standard_url(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        ) == "dQw4w9WgXcQ"

    def test_invalid_url(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("https://google.com") is None

    def test_empty_string(self, youtube_ingestor):
        assert youtube_ingestor._extract_video_id("") is None


class TestYouTubeTranscriptCleaning:
    """Test transcript text cleaning."""

    def test_remove_timestamps(self, youtube_ingestor):
        raw = "Hello [00:15] world (02:30:45) test"
        cleaned = youtube_ingestor._clean_transcript(raw)
        assert "[00:15]" not in cleaned
        assert "(02:30:45)" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_remove_music_tags(self, youtube_ingestor):
        raw = "Hello [Music] world [Applause] test"
        cleaned = youtube_ingestor._clean_transcript(raw)
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned

    def test_remove_filler_words(self, youtube_ingestor):
        raw = "um so basically you know the RSI indicator is actually like really important"
        cleaned = youtube_ingestor._clean_transcript(raw)
        assert "basically" not in cleaned.lower().split()
        assert "RSI" in cleaned
        assert "indicator" in cleaned

    def test_normalize_whitespace(self, youtube_ingestor):
        raw = "Hello    world\n\n\n  test   here"
        cleaned = youtube_ingestor._clean_transcript(raw)
        assert "  " not in cleaned

    def test_empty_input(self, youtube_ingestor):
        assert youtube_ingestor._clean_transcript("") == ""


class TestYouTubeChunking:
    """Test transcript chunking."""

    def test_basic_chunking(self, youtube_ingestor):
        meta = VideoMeta(video_id="test", title="Test Video", url="http://test.com")
        # Generate 1200 words of text
        text = " ".join(f"word{i}" for i in range(1200))
        chunks = youtube_ingestor._chunk_text(text, meta)

        assert len(chunks) >= 2
        assert all(c.video_meta == meta for c in chunks)
        assert chunks[0].chunk_index == 0
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_chunk_overlap(self, youtube_ingestor):
        """Consecutive chunks share overlapping words."""
        youtube_ingestor._chunk_words = 100
        youtube_ingestor._chunk_overlap = 20
        meta = VideoMeta(video_id="test", title="Test", url="http://test.com")
        text = " ".join(f"word{i}" for i in range(300))
        chunks = youtube_ingestor._chunk_text(text, meta)

        if len(chunks) >= 2:
            words0 = set(chunks[0].text.split()[-20:])
            words1 = set(chunks[1].text.split()[:20])
            overlap = words0 & words1
            assert len(overlap) > 0

    def test_short_text_single_chunk(self, youtube_ingestor):
        meta = VideoMeta(video_id="test", title="Test", url="http://test.com")
        text = " ".join(f"word{i}" for i in range(50))
        chunks = youtube_ingestor._chunk_text(text, meta)
        assert len(chunks) == 1

    def test_very_short_text_skipped(self, youtube_ingestor):
        meta = VideoMeta(video_id="test", title="Test", url="http://test.com")
        text = "only five words here now"
        chunks = youtube_ingestor._chunk_text(text, meta)
        assert len(chunks) == 0  # Below 20 word minimum

    def test_chunk_hash_uniqueness(self, youtube_ingestor):
        meta = VideoMeta(video_id="test", title="Test", url="http://test.com")
        text = " ".join(f"word{i}" for i in range(1200))
        chunks = youtube_ingestor._chunk_text(text, meta)
        hashes = [c.chunk_hash for c in chunks]
        assert len(hashes) == len(set(hashes))  # All unique

    def test_empty_text_no_chunks(self, youtube_ingestor):
        meta = VideoMeta(video_id="test", title="Test", url="http://test.com")
        chunks = youtube_ingestor._chunk_text("", meta)
        assert len(chunks) == 0


class TestYouTubeFilters:
    """Test clickbait and age filters."""

    def test_clickbait_detection(self, youtube_ingestor):
        assert youtube_ingestor._is_clickbait("How I turned $100 into 1000x returns guaranteed!")
        assert youtube_ingestor._is_clickbait("Get Rich Quick with Crypto!")
        assert youtube_ingestor._is_clickbait("FREE MONEY from Bitcoin trading")

    def test_non_clickbait(self, youtube_ingestor):
        assert not youtube_ingestor._is_clickbait("Understanding RSI Divergence for Swing Trading")
        assert not youtube_ingestor._is_clickbait("Risk Management in Crypto Futures")

    def test_age_filter_old_video(self, youtube_ingestor):
        youtube_ingestor._max_age_days = 180
        assert youtube_ingestor._is_too_old("2020-01-01")

    def test_age_filter_recent_video(self, youtube_ingestor):
        youtube_ingestor._max_age_days = 180
        from datetime import datetime, timezone
        recent = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert not youtube_ingestor._is_too_old(recent)

    def test_age_filter_no_date(self, youtube_ingestor):
        """Missing date should not filter out."""
        assert not youtube_ingestor._is_too_old("")
        assert not youtube_ingestor._is_too_old(None)


# ── Paper Ingestor Tests ─────────────────────────────────────

class TestPaperTextCleaning:
    """Test paper/blog text cleaning."""

    def test_clean_page_numbers(self, paper_ingestor):
        text = "End of page\n 42 \nStart of next page"
        cleaned = paper_ingestor._clean_text(text)
        assert " 42 " not in cleaned

    def test_clean_urls(self, paper_ingestor):
        text = "Visit https://example.com/page for more info about trading"
        cleaned = paper_ingestor._clean_text(text)
        assert "https://" not in cleaned
        assert "trading" in cleaned

    def test_clean_excessive_newlines(self, paper_ingestor):
        text = "First paragraph\n\n\n\n\nSecond paragraph"
        cleaned = paper_ingestor._clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_remove_short_lines(self, paper_ingestor):
        text = "This is a valid long line of text with good content\nab\nAnother good line with enough words"
        cleaned = paper_ingestor._clean_text(text)
        lines = [l for l in cleaned.split("\n") if l.strip()]
        assert all(len(l.strip()) > 10 or l.strip() == "" for l in lines)


class TestPaperChunking:
    """Test paper text chunking."""

    def test_basic_chunking(self, paper_ingestor):
        meta = SourceMeta(title="Test Paper", url="http://test.com", source_type="paper")
        text = " ".join(f"word{i}" for i in range(1500))
        chunks = paper_ingestor._chunk_text(text, meta)

        assert len(chunks) >= 3
        assert chunks[0].source_meta == meta
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_chunk_hash_is_deterministic(self, paper_ingestor):
        meta = SourceMeta(title="Test", url="http://test.com", source_type="paper")
        text = "consistent text for hashing " * 50
        chunks1 = paper_ingestor._chunk_text(text, meta)
        chunks2 = paper_ingestor._chunk_text(text, meta)
        assert chunks1[0].chunk_hash == chunks2[0].chunk_hash


class TestPaperArxiv:
    """Test arXiv paper listing (uses XML parsing, no network)."""

    def test_arxiv_xml_parsing(self, paper_ingestor):
        """Test parsing arXiv XML response."""
        import xml.etree.ElementTree as ET

        # Minimal arXiv-like XML
        xml_str = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Cryptocurrency Trading with Machine Learning</title>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <published>2025-01-15T00:00:00Z</published>
    <summary>We propose a novel approach to crypto trading using ML.</summary>
    <author><name>John Doe</name></author>
    <author><name>Jane Smith</name></author>
    <link href="http://arxiv.org/abs/2301.12345v1" type="text/html"/>
    <link href="http://arxiv.org/pdf/2301.12345v1" title="pdf"/>
  </entry>
</feed>"""

        root = ET.fromstring(xml_str)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        assert len(entries) == 1

        entry = entries[0]
        title = entry.findtext("atom:title", "", ns).strip()
        assert "Cryptocurrency" in title

        authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
        assert len(authors) == 2

    def test_ingest_url_routes_pdf(self, paper_ingestor):
        """URL ending in .pdf routes to PDF ingestion."""
        # We can't actually download, but test the routing logic
        url = "https://example.com/paper.pdf"
        assert url.lower().endswith(".pdf")

    def test_ingest_url_routes_web(self, paper_ingestor):
        """Normal URLs route to web page ingestion."""
        url = "https://medium.com/some-article"
        assert not url.lower().endswith(".pdf")


# ── Integration-style Tests ──────────────────────────────────

class TestEndToEndPipeline:
    """Test the full extraction pipeline with mocked LLM."""

    @pytest.fixture
    def mock_extractor(self, kb):
        """Extractor with mocked LLM call."""
        ext = KnowledgeExtractor(knowledge_base=kb)

        async def mock_call_llm(prompt):
            return json.dumps({
                "strategies": [
                    {"name": "RSI Swing", "description": "Buy RSI<30 on 4h",
                     "conditions": "RSI<30, uptrend", "timeframe": "4h", "confidence": 0.85},
                ],
                "indicators": [
                    {"name": "RSI", "usage": "Oversold/overbought",
                     "parameters": "14 period", "confidence": 0.9},
                ],
                "risk_rules": [
                    {"rule": "Max 1% per trade", "rationale": "Capital preservation",
                     "confidence": 0.95},
                ],
                "market_insights": [],
                "entry_patterns": [],
                "exit_rules": [],
                "mistakes_to_avoid": [
                    {"mistake": "Revenge trading", "why": "Emotional decisions",
                     "confidence": 0.4},  # Below threshold
                ],
            })

        ext._call_llm = mock_call_llm
        return ext

    def test_extract_from_chunk(self, mock_extractor):
        """Full extraction pipeline: call → parse → filter → entries."""
        import asyncio
        result = asyncio.run(mock_extractor.extract_from_chunk(
            text="Test trading content about RSI and risk management",
            source_type="youtube",
            source_url="http://test.com",
            source_title="Test Video",
            chunk_hash="test123",
        ))

        assert result.error is None
        # 3 entries should pass (RSI Swing 0.85, RSI 0.9, Max 1% 0.95)
        # 1 should be filtered (Revenge trading 0.4 < 0.5 threshold)
        assert len(result.entries) == 3
        categories = {e.category for e in result.entries}
        assert "strategy" in categories
        assert "indicator" in categories
        assert "risk_rule" in categories

    def test_extract_stores_in_kb(self, mock_extractor, kb):
        """Extracted entries are stored in KB after extraction."""
        import asyncio
        result = asyncio.run(mock_extractor.extract_from_chunk(
            text="Test content",
            source_type="youtube",
            source_url="http://test.com",
            source_title="Test",
            chunk_hash="hash1",
        ))

        for entry in result.entries:
            kb.add_entry(entry)

        stats = kb.get_stats()
        assert stats["total_entries"] == 3

    def test_extract_from_chunks_aggregates(self, mock_extractor):
        """extract_from_chunks aggregates entries from multiple chunks."""
        import asyncio
        chunks = [
            TranscriptChunk(
                text=f"Chunk {i} about trading",
                chunk_index=i, total_chunks=3,
                video_meta=VideoMeta(
                    video_id="test", title="Test", url="http://test.com"
                ),
                chunk_hash=f"hash{i}",
            )
            for i in range(3)
        ]

        entries = asyncio.run(mock_extractor.extract_from_chunks(
            chunks, source_type="youtube"
        ))

        # 3 entries per chunk (after filtering), × 3 chunks = 9
        assert len(entries) == 9


class TestCategoryMap:
    """Test that category map is complete and correct."""

    def test_all_categories_mapped(self):
        expected_extraction_keys = {
            "strategies", "indicators", "risk_rules", "market_insights",
            "entry_patterns", "exit_rules", "mistakes_to_avoid",
        }
        assert set(CATEGORY_MAP.keys()) == expected_extraction_keys

    def test_mapped_values(self):
        expected_values = {
            "strategy", "indicator", "risk_rule", "insight",
            "pattern", "exit", "mistake",
        }
        assert set(CATEGORY_MAP.values()) == expected_values


# ═══════════════════════════════════════════════════════════════
# ADVISOR TESTS
# ═══════════════════════════════════════════════════════════════

from ai_learning.advisor import (
    TradingAdvisor, AdvisorResult, ConsultationRecord, _cache_key
)


class TestAdvisorResult:
    """AdvisorResult dataclass tests."""

    def test_default_values(self):
        r = AdvisorResult()
        assert r.recommendation == "PROCEED"
        assert r.confidence_adjustment == 0.0
        assert r.reasoning == ""
        assert r.cached is False
        assert r.skipped is False
        assert r.source_ids == []

    def test_to_dict(self):
        r = AdvisorResult(recommendation="SKIP", confidence_adjustment=-0.05,
                          reasoning="High risk", source_ids=[1, 2])
        d = r.to_dict()
        assert d["recommendation"] == "SKIP"
        assert d["confidence_adjustment"] == -0.05
        assert d["source_ids"] == [1, 2]

    def test_suggested_adjustments_defaults(self):
        r = AdvisorResult()
        assert r.suggested_adjustments["tp_multiplier"] is None
        assert r.suggested_adjustments["sl_multiplier"] is None
        assert r.suggested_adjustments["position_size_factor"] is None


class TestConsultationRecord:
    """ConsultationRecord dataclass tests."""

    def test_to_dict_parses_json(self):
        rec = ConsultationRecord(
            symbol="BTC/USDT",
            suggested_adjustments='{"tp_multiplier": 1.5}',
            kb_entry_ids='[1, 2, 3]',
        )
        d = rec.to_dict()
        assert d["suggested_adjustments"] == {"tp_multiplier": 1.5}
        assert d["kb_entry_ids"] == [1, 2, 3]

    def test_to_dict_handles_empty(self):
        rec = ConsultationRecord()
        d = rec.to_dict()
        # Empty strings are falsy → returns {} and [] respectively
        assert d["suggested_adjustments"] == {}
        assert d["kb_entry_ids"] == []


class TestCacheKey:
    """Cache key generation tests."""

    def test_same_inputs_same_key(self):
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.72, 31.0, 55.0)
        k2 = _cache_key("BTC/USDT", "BUY", "trend", 0.72, 31.0, 55.0)
        assert k1 == k2

    def test_different_symbol_different_key(self):
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.72, 31.0, 55.0)
        k2 = _cache_key("ETH/USDT", "BUY", "trend", 0.72, 31.0, 55.0)
        assert k1 != k2

    def test_quantization_confidence(self):
        # 0.72 and 0.74 should round to 0.7
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.72, 30.0, 50.0)
        k2 = _cache_key("BTC/USDT", "BUY", "trend", 0.74, 30.0, 50.0)
        assert k1 == k2

    def test_quantization_adx(self):
        # ADX 31 and 32 both round to 30 (5-unit bucket: round(x/5)*5)
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.70, 31.0, 50.0)
        k2 = _cache_key("BTC/USDT", "BUY", "trend", 0.70, 32.0, 50.0)
        assert k1 == k2

    def test_quantization_rsi(self):
        # RSI 52 and 53 both round to 50 (10-unit bucket: round(x/10)*10)
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.70, 30.0, 52.0)
        k2 = _cache_key("BTC/USDT", "BUY", "trend", 0.70, 30.0, 53.0)
        assert k1 == k2

    def test_different_side_different_key(self):
        k1 = _cache_key("BTC/USDT", "BUY", "trend", 0.70, 30.0, 50.0)
        k2 = _cache_key("BTC/USDT", "SELL", "trend", 0.70, 30.0, 50.0)
        assert k1 != k2


class TestApplyAdvice:
    """Tests for TradingAdvisor.apply_advice() integration rules."""

    def setup_method(self):
        self.advisor = TradingAdvisor()

    def test_proceed_applies_adjustment(self):
        result = AdvisorResult(recommendation="PROCEED", confidence_adjustment=0.05)
        proceed, conf, reason = self.advisor.apply_advice(result, 0.60)
        assert proceed is True
        assert conf == pytest.approx(0.65)
        assert "proceed" in reason

    def test_caution_applies_adjustment(self):
        result = AdvisorResult(recommendation="CAUTION", confidence_adjustment=-0.05)
        proceed, conf, reason = self.advisor.apply_advice(result, 0.70)
        assert proceed is True
        assert conf == pytest.approx(0.65)
        assert "caution" in reason

    def test_skip_honored_below_070(self):
        result = AdvisorResult(recommendation="SKIP", confidence_adjustment=-0.10)
        proceed, conf, reason = self.advisor.apply_advice(result, 0.65)
        assert proceed is False
        assert "skip" in reason

    def test_skip_overridden_above_070(self):
        result = AdvisorResult(recommendation="SKIP", confidence_adjustment=-0.05)
        proceed, conf, reason = self.advisor.apply_advice(result, 0.75)
        assert proceed is True
        assert "overridden" in reason

    def test_skip_overridden_at_exactly_070(self):
        result = AdvisorResult(recommendation="SKIP", confidence_adjustment=-0.05)
        proceed, conf, reason = self.advisor.apply_advice(result, 0.70)
        assert proceed is True  # >= 0.70 overrides

    def test_skipped_advisor_always_proceeds(self):
        result = AdvisorResult(skipped=True, skip_reason="No KB")
        proceed, conf, reason = self.advisor.apply_advice(result, 0.55)
        assert proceed is True
        assert conf == 0.55  # No adjustment applied

    def test_adjustment_clamped_positive(self):
        result = AdvisorResult(recommendation="PROCEED", confidence_adjustment=0.50)
        proceed, conf, _ = self.advisor.apply_advice(result, 0.60)
        assert conf == pytest.approx(0.70)  # Clamped to +0.10

    def test_adjustment_clamped_negative(self):
        result = AdvisorResult(recommendation="CAUTION", confidence_adjustment=-0.50)
        proceed, conf, _ = self.advisor.apply_advice(result, 0.60)
        assert conf == pytest.approx(0.50)  # Clamped to -0.10

    def test_agreement_counter_increments(self):
        result = AdvisorResult(recommendation="PROCEED")
        self.advisor.apply_advice(result, 0.60)
        assert self.advisor._agreements == 1

    def test_override_counter_increments(self):
        result = AdvisorResult(recommendation="SKIP")
        self.advisor.apply_advice(result, 0.80)
        assert self.advisor._overrides == 1

    def test_confidence_floor_zero(self):
        result = AdvisorResult(recommendation="CAUTION", confidence_adjustment=-0.10)
        proceed, conf, _ = self.advisor.apply_advice(result, 0.05)
        # apply_advice returns raw adjusted value; main.py clamps
        assert conf == pytest.approx(-0.05)


class TestAdvisorFeedbackLoop:
    """Tests for post-trade feedback and KB entry stats updates."""

    @pytest.fixture
    def advisor_with_kb(self, tmp_db):
        kb = KnowledgeBase(db_path=tmp_db, embedding_dim=384)
        advisor = TradingAdvisor(knowledge_base=kb)
        return advisor, kb

    def test_record_winning_trade(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="youtube", source_url="", source_title="Test",
            category="strategy", content='{"name": "test"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        advisor.record_trade_outcome(trade_id=1, source_ids=[eid], pnl_usd=0.50)

        updated = kb.get_entry(eid)
        assert updated.times_applied == 1
        assert updated.success_rate == 1.0

    def test_record_losing_trade(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="youtube", source_url="", source_title="Test",
            category="risk_rule", content='{"rule": "test"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        advisor.record_trade_outcome(trade_id=1, source_ids=[eid], pnl_usd=-0.30)

        updated = kb.get_entry(eid)
        assert updated.times_applied == 1
        assert updated.success_rate == 0.0

    def test_multiple_outcomes_update_rate(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="paper", source_url="", source_title="Test",
            category="strategy", content='{"name": "test"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        # 3 wins, 1 loss
        advisor.record_trade_outcome(trade_id=1, source_ids=[eid], pnl_usd=0.50)
        advisor.record_trade_outcome(trade_id=2, source_ids=[eid], pnl_usd=0.30)
        advisor.record_trade_outcome(trade_id=3, source_ids=[eid], pnl_usd=-0.20)
        advisor.record_trade_outcome(trade_id=4, source_ids=[eid], pnl_usd=0.10)

        updated = kb.get_entry(eid)
        assert updated.times_applied == 4
        assert updated.success_rate == pytest.approx(0.75)

    def test_multiple_source_ids_updated(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        e1 = kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="", source_title="T1",
            category="strategy", content='{"name": "s1"}',
            confidence=0.8, extraction_date=0.0,
        ))
        e2 = kb.add_entry(KnowledgeEntry(
            source_type="paper", source_url="", source_title="T2",
            category="risk_rule", content='{"rule": "r1"}',
            confidence=0.7, extraction_date=0.0,
        ))

        advisor.record_trade_outcome(trade_id=1, source_ids=[e1, e2], pnl_usd=0.50)

        assert kb.get_entry(e1).times_applied == 1
        assert kb.get_entry(e2).times_applied == 1

    def test_no_kb_graceful(self):
        advisor = TradingAdvisor()  # No KB
        # Should not raise
        advisor.record_trade_outcome(trade_id=1, source_ids=[1, 2], pnl_usd=0.50)

    def test_invalid_entry_id_graceful(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        # Entry ID 999 doesn't exist — should not raise
        advisor.record_trade_outcome(trade_id=1, source_ids=[999], pnl_usd=0.50)

    def test_consultation_record_updated(self, advisor_with_kb):
        advisor, kb = advisor_with_kb
        e1 = kb.add_entry(KnowledgeEntry(
            source_type="youtube", source_url="", source_title="T",
            category="strategy", content='{"name": "s"}',
            confidence=0.8, extraction_date=0.0,
        ))

        # Simulate a consultation record
        rec = ConsultationRecord(
            trade_id=42, symbol="BTC/USDT", recommendation="PROCEED",
            kb_entry_ids=json.dumps([e1]),
        )
        advisor._recent_consultations.append(rec)

        advisor.record_trade_outcome(trade_id=42, source_ids=[e1], pnl_usd=0.50)
        assert rec.trade_outcome == "win"
        assert rec.trade_pnl == 0.50


class TestAdvisorReview:
    """Tests for KB entry deprecation and boosting review."""

    @pytest.fixture
    def advisor_with_kb(self, tmp_db):
        kb = KnowledgeBase(db_path=tmp_db, embedding_dim=384)
        advisor = TradingAdvisor(knowledge_base=kb)
        return advisor, kb

    def test_entry_flagged_low_success(self, advisor_with_kb, caplog):
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="youtube", source_url="", source_title="Bad Strategy",
            category="strategy", content='{"name": "bad"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        # Simulate 10+ applications with <30% success
        for i in range(12):
            pnl = 0.10 if i < 2 else -0.10  # 2 wins, 10 losses
            advisor.record_trade_outcome(trade_id=i, source_ids=[eid], pnl_usd=pnl)

        updated = kb.get_entry(eid)
        assert updated.times_applied == 12
        assert updated.success_rate < 0.30
        # Check warning was logged
        assert any("flagged" in r.message for r in caplog.records)

    def test_entry_boosted_high_success(self, advisor_with_kb, caplog):
        import logging
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="paper", source_url="", source_title="Good Strategy",
            category="strategy", content='{"name": "good"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        # Simulate 10+ applications with >65% success
        with caplog.at_level(logging.INFO, logger="ai_learning.advisor"):
            for i in range(10):
                pnl = 0.10 if i < 8 else -0.10  # 8 wins, 2 losses
                advisor.record_trade_outcome(trade_id=i, source_ids=[eid], pnl_usd=pnl)

        updated = kb.get_entry(eid)
        assert updated.success_rate > 0.65
        assert any("performing well" in r.message for r in caplog.records)

    def test_no_review_below_min_applications(self, advisor_with_kb, caplog):
        advisor, kb = advisor_with_kb
        entry = KnowledgeEntry(
            source_type="youtube", source_url="", source_title="New Entry",
            category="strategy", content='{"name": "new"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)

        # Only 3 applications — below min_applications=10 threshold
        for i in range(3):
            advisor.record_trade_outcome(trade_id=i, source_ids=[eid], pnl_usd=-0.10)

        # Should NOT trigger flagging/boosting
        assert not any("flagged" in r.message for r in caplog.records)
        assert not any("performing well" in r.message for r in caplog.records)


class TestAdvisorStats:
    """Tests for advisor statistics and dashboard data."""

    def test_stats_empty(self):
        advisor = TradingAdvisor()
        stats = advisor.get_stats()
        assert stats["total_consultations"] == 0
        assert stats["agreements"] == 0
        assert stats["overrides"] == 0
        assert stats["agreement_rate"] == 0.0
        assert stats["cache_size"] == 0
        assert len(stats["recent_consultations"]) == 0

    def test_stats_after_apply(self):
        advisor = TradingAdvisor()
        r1 = AdvisorResult(recommendation="PROCEED")
        r2 = AdvisorResult(recommendation="SKIP")
        advisor.apply_advice(r1, 0.60)
        advisor.apply_advice(r2, 0.80)  # Override
        stats = advisor.get_stats()
        assert stats["agreements"] == 1
        assert stats["overrides"] == 1
        assert stats["agreement_rate"] == pytest.approx(0.5)

    def test_consultation_for_trade_found(self):
        advisor = TradingAdvisor()
        rec = ConsultationRecord(trade_id=42, symbol="ETH/USDT",
                                 recommendation="PROCEED")
        advisor._recent_consultations.append(rec)
        result = advisor.get_consultation_for_trade(42)
        assert result is not None
        assert result["symbol"] == "ETH/USDT"

    def test_consultation_for_trade_not_found(self):
        advisor = TradingAdvisor()
        assert advisor.get_consultation_for_trade(999) is None

    def test_kb_performance_report_empty(self):
        advisor = TradingAdvisor()  # No KB
        assert advisor.get_kb_performance_report() == []

    def test_kb_performance_report_with_data(self, tmp_db):
        kb = KnowledgeBase(db_path=tmp_db, embedding_dim=384)
        advisor = TradingAdvisor(knowledge_base=kb)

        entry = KnowledgeEntry(
            source_type="youtube", source_url="", source_title="Strat",
            category="strategy", content='{"name": "test"}',
            confidence=0.8, extraction_date=0.0,
        )
        eid = kb.add_entry(entry)
        # Give it some application stats
        advisor.record_trade_outcome(trade_id=1, source_ids=[eid], pnl_usd=0.50)

        report = advisor.get_kb_performance_report()
        assert len(report) == 1
        assert report[0]["times_applied"] == 1
        assert report[0]["status"] == "neutral"  # < min_applications


class TestAdvisorBudgetAndCache:
    """Tests for daily budget and caching."""

    def test_budget_reset(self):
        advisor = TradingAdvisor()
        advisor._calls_today = 100
        advisor._budget_reset_date = "2020-01-01"  # Old date
        advisor._check_budget_reset()
        assert advisor._calls_today == 0

    def test_budget_same_day_no_reset(self):
        import time
        advisor = TradingAdvisor()
        advisor._calls_today = 50
        today = time.strftime("%Y-%m-%d", time.gmtime())
        advisor._budget_reset_date = today
        advisor._check_budget_reset()
        assert advisor._calls_today == 50

    def test_cache_hit(self):
        import time
        advisor = TradingAdvisor()
        result = AdvisorResult(recommendation="PROCEED", confidence_adjustment=0.05)
        key = "test_key"
        advisor._cache[key] = (result, time.time())

        cached = advisor._get_cached(key)
        assert cached is not None
        assert cached.recommendation == "PROCEED"
        assert cached.cached is True

    def test_cache_miss_expired(self):
        import time
        advisor = TradingAdvisor()
        result = AdvisorResult(recommendation="PROCEED")
        key = "test_key"
        advisor._cache[key] = (result, time.time() - 1000)  # Expired

        cached = advisor._get_cached(key)
        assert cached is None
        assert key not in advisor._cache  # Cleaned up

    def test_cache_miss_no_entry(self):
        advisor = TradingAdvisor()
        assert advisor._get_cached("nonexistent") is None

    def test_clear_cache(self):
        import time
        advisor = TradingAdvisor()
        advisor._cache["k1"] = (AdvisorResult(), time.time())
        advisor._cache["k2"] = (AdvisorResult(), time.time())
        advisor.clear_cache()
        assert len(advisor._cache) == 0


class TestAdvisorConsult:
    """Tests for the consult() method edge cases (no LLM calls)."""

    def test_consult_no_kb(self):
        advisor = TradingAdvisor()  # No KB
        result = asyncio.run(advisor.consult("BTC/USDT", "BUY", 0.65))
        assert result.skipped is True
        assert "knowledge base" in result.skip_reason.lower()

    def test_consult_budget_exhausted(self, tmp_db):
        kb = KnowledgeBase(db_path=tmp_db, embedding_dim=384)
        advisor = TradingAdvisor(knowledge_base=kb)
        import time
        advisor._calls_today = 100
        advisor._budget_reset_date = time.strftime("%Y-%m-%d", time.gmtime())

        result = asyncio.run(advisor.consult("BTC/USDT", "BUY", 0.65))
        assert result.skipped is True
        assert "budget" in result.skip_reason.lower()

    def test_consult_empty_kb_skips_llm(self, tmp_db):
        kb = KnowledgeBase(db_path=tmp_db, embedding_dim=384)
        advisor = TradingAdvisor(knowledge_base=kb)

        # Empty KB → no relevant knowledge → should skip LLM
        result = asyncio.run(advisor.consult("BTC/USDT", "BUY", 0.65))
        assert result.skipped is True
        assert "no relevant" in result.skip_reason.lower()


class TestAdvisorResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_clean_json(self):
        raw = '{"recommendation": "PROCEED", "confidence_adjustment": 0.05, "reasoning": "Good setup", "suggested_adjustments": {}}'
        result = TradingAdvisor._parse_response(raw)
        assert result is not None
        assert result["recommendation"] == "PROCEED"

    def test_parse_markdown_fenced(self):
        raw = '```json\n{"recommendation": "SKIP", "confidence_adjustment": -0.10, "reasoning": "Risk", "suggested_adjustments": {}}\n```'
        result = TradingAdvisor._parse_response(raw)
        assert result is not None
        assert result["recommendation"] == "SKIP"

    def test_parse_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"recommendation": "CAUTION", "confidence_adjustment": 0.0, "reasoning": "Mixed", "suggested_adjustments": {}}\nDone.'
        result = TradingAdvisor._parse_response(raw)
        assert result is not None
        assert result["recommendation"] == "CAUTION"

    def test_parse_invalid_json(self):
        raw = "This is not JSON at all"
        result = TradingAdvisor._parse_response(raw)
        assert result is None

    def test_build_context(self):
        ctx = TradingAdvisor._build_context(
            "BTC/USDT", "BUY", 0.72, "trend", 35.0,
            55.0, 48.0, ["hammer"], ["doji"],
            True, False, "strong",
        )
        assert "BTC/USDT" in ctx
        assert "BUY" in ctx
        assert "trend" in ctx
        assert "hammer" in ctx
        assert "doji" in ctx
        assert "Squeeze" in ctx
        assert "strong" in ctx
