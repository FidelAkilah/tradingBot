"""
Knowledge Extraction with LLM — sends text chunks to Claude or OpenAI
to extract structured trading knowledge in JSON format.

Handles rate limiting, retries, and deduplication against existing KB.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ai_learning.knowledge_base import KnowledgeBase, KnowledgeEntry

logger = logging.getLogger(__name__)


_EXTRACTION_TEMPLATE = (
    "Analyze this trading content and extract actionable knowledge.\n"
    "Return ONLY valid JSON with this exact structure:\n"
    '{\n'
    '  "strategies": [{"name": "", "description": "", "conditions": "", "timeframe": "", "confidence": 0.0}],\n'
    '  "indicators": [{"name": "", "usage": "", "parameters": "", "confidence": 0.0}],\n'
    '  "risk_rules": [{"rule": "", "rationale": "", "confidence": 0.0}],\n'
    '  "market_insights": [{"insight": "", "context": "", "confidence": 0.0}],\n'
    '  "entry_patterns": [{"pattern": "", "setup": "", "confirmation": "", "confidence": 0.0}],\n'
    '  "exit_rules": [{"rule": "", "trigger": "", "confidence": 0.0}],\n'
    '  "mistakes_to_avoid": [{"mistake": "", "why": "", "confidence": 0.0}]\n'
    '}\n\n'
    "Rules:\n"
    "- Only include items specifically about crypto futures swing trading\n"
    "- Rate your confidence in each item from 0.0 to 1.0\n"
    "- Omit categories with no relevant items (return empty array)\n"
    "- Be specific and actionable, not generic platitudes\n"
    "- If the text is not about trading at all, return all empty arrays\n\n"
    "Content to analyze:\n---\n"
)


def _build_extraction_prompt(text: str) -> str:
    return _EXTRACTION_TEMPLATE + text + "\n---"

# Maps extraction categories to knowledge_entries category values
CATEGORY_MAP = {
    "strategies": "strategy",
    "indicators": "indicator",
    "risk_rules": "risk_rule",
    "market_insights": "insight",
    "entry_patterns": "pattern",
    "exit_rules": "exit",
    "mistakes_to_avoid": "mistake",
}


@dataclass
class ExtractionResult:
    """Result from processing a single chunk."""
    entries: List[KnowledgeEntry]
    raw_response: str = ""
    error: Optional[str] = None
    chunk_hash: str = ""


class KnowledgeExtractor:
    """Extracts structured knowledge from text using LLM APIs."""

    def __init__(self, config=None, knowledge_base: Optional[KnowledgeBase] = None):
        self.config = config
        self.kb = knowledge_base
        self._provider = "anthropic"
        self._model = "claude-sonnet-4-5-20250514"
        self._min_confidence = 0.5
        self._dedup_threshold = 0.85
        self._rpm_limit = 30
        self._retry_max = 3
        self._retry_delay = 2.0
        self._request_count = 0
        self._window_start = time.time()

        if config:
            self._provider = getattr(config, "llm_provider", "anthropic")
            self._model = getattr(config, "llm_model", self._model)
            self._min_confidence = getattr(config, "min_confidence", 0.5)
            self._dedup_threshold = getattr(config, "dedup_similarity_threshold", 0.85)
            self._rpm_limit = getattr(config, "llm_requests_per_minute", 30)
            self._retry_max = getattr(config, "llm_retry_max", 3)
            self._retry_delay = getattr(config, "llm_retry_delay", 2.0)

    # ── Public API ────────────────────────────────────────────

    async def extract_from_chunk(
        self,
        text: str,
        source_type: str = "",
        source_url: str = "",
        source_title: str = "",
        chunk_hash: str = "",
    ) -> ExtractionResult:
        """
        Extract structured knowledge from a single text chunk.

        Args:
            text: The text to analyze
            source_type: youtube / paper / blog
            source_url: URL of the source
            source_title: Title of the source
            chunk_hash: Hash for dedup tracking

        Returns:
            ExtractionResult with list of KnowledgeEntry objects
        """
        await self._rate_limit()

        prompt = _build_extraction_prompt(text[:4000])

        raw = await self._call_llm(prompt)
        if raw is None:
            return ExtractionResult(
                entries=[], error="LLM call failed", chunk_hash=chunk_hash
            )

        # Parse JSON from response
        parsed = self._parse_extraction(raw)
        if parsed is None:
            return ExtractionResult(
                entries=[], raw_response=raw,
                error="Failed to parse LLM response as JSON",
                chunk_hash=chunk_hash,
            )

        # Convert to KnowledgeEntry objects
        entries = self._to_entries(
            parsed, source_type, source_url, source_title, chunk_hash
        )

        # Filter low confidence
        entries = [e for e in entries if e.confidence >= self._min_confidence]

        # Deduplicate against existing KB
        if self.kb:
            entries = self._deduplicate(entries)

        return ExtractionResult(
            entries=entries, raw_response=raw, chunk_hash=chunk_hash
        )

    async def extract_from_chunks(
        self,
        chunks: list,
        source_type: str = "",
        source_url: str = "",
        source_title: str = "",
    ) -> List[KnowledgeEntry]:
        """
        Extract knowledge from multiple chunks and aggregate.

        Works with TranscriptChunk or TextChunk objects (duck typing on .text,
        .source_url, .source_title, .chunk_hash).

        Returns:
            Combined list of deduplicated KnowledgeEntry objects
        """
        all_entries: List[KnowledgeEntry] = []

        for chunk in chunks:
            text = getattr(chunk, "text", str(chunk))
            url = getattr(chunk, "source_url", source_url)
            title = getattr(chunk, "source_title", source_title)
            c_hash = getattr(chunk, "chunk_hash", "")
            s_type = source_type

            result = await self.extract_from_chunk(
                text=text,
                source_type=s_type,
                source_url=url,
                source_title=title,
                chunk_hash=c_hash,
            )

            if result.error:
                logger.warning(
                    f"Extraction error on chunk {c_hash}: {result.error}"
                )
            else:
                all_entries.extend(result.entries)
                logger.debug(
                    f"Chunk {c_hash}: extracted {len(result.entries)} entries"
                )

        logger.info(
            f"Total extracted from {len(chunks)} chunks: {len(all_entries)} entries"
        )
        return all_entries

    # ── LLM Calls ─────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the configured LLM provider with retries."""
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
                    f"LLM call attempt {attempt + 1}/{self._retry_max} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self._retry_max - 1:
                    await asyncio.sleep(delay)

        logger.error(f"LLM call failed after {self._retry_max} attempts")
        return None

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Claude API via the Anthropic SDK."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or add to .env"
            )

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=self._model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Run: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")

        client = openai.AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        return response.choices[0].message.content

    # ── Rate Limiting ─────────────────────────────────────────

    async def _rate_limit(self):
        """Enforce requests-per-minute limit."""
        now = time.time()
        elapsed = now - self._window_start

        if elapsed >= 60:
            self._request_count = 0
            self._window_start = now
        elif self._request_count >= self._rpm_limit:
            wait = 60 - elapsed
            logger.debug(f"Rate limit reached, waiting {wait:.1f}s")
            await asyncio.sleep(wait)
            self._request_count = 0
            self._window_start = time.time()

        self._request_count += 1

    # ── Parsing ───────────────────────────────────────────────

    @staticmethod
    def _parse_extraction(raw: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines if they are fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _to_entries(
        self,
        parsed: dict,
        source_type: str,
        source_url: str,
        source_title: str,
        chunk_hash: str,
    ) -> List[KnowledgeEntry]:
        """Convert parsed extraction dict to KnowledgeEntry list."""
        entries: List[KnowledgeEntry] = []
        now = time.time()

        for extraction_key, category in CATEGORY_MAP.items():
            items = parsed.get(extraction_key, [])
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                confidence = float(item.pop("confidence", 0.5))
                content_json = json.dumps(item, ensure_ascii=False)

                entries.append(KnowledgeEntry(
                    source_type=source_type,
                    source_url=source_url,
                    source_title=source_title,
                    category=category,
                    content=content_json,
                    confidence=confidence,
                    extraction_date=now,
                    chunk_hash=chunk_hash,
                ))

        return entries

    def _deduplicate(self, entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
        """Remove entries that are semantically duplicate of existing KB entries."""
        if not self.kb:
            return entries

        unique: List[KnowledgeEntry] = []
        for entry in entries:
            content_text = entry.content
            try:
                parsed = json.loads(content_text)
                content_text = " ".join(str(v) for v in parsed.values() if v)
            except (json.JSONDecodeError, AttributeError):
                pass

            if not self.kb.is_duplicate(content_text, self._dedup_threshold):
                unique.append(entry)
            else:
                logger.debug(
                    f"Skipping duplicate entry: {content_text[:80]}..."
                )

        if len(entries) != len(unique):
            logger.info(
                f"Dedup: {len(entries)} → {len(unique)} entries "
                f"({len(entries) - len(unique)} duplicates removed)"
            )
        return unique
