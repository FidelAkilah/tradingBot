"""
Research Paper / Blog Ingestion — extracts and chunks text from web pages,
PDFs, and arXiv papers for knowledge extraction.

Uses trafilatura for web text extraction and pdfplumber for PDF parsing.
"""

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SourceMeta:
    """Metadata for a paper or blog source."""
    title: str = ""
    url: str = ""
    source_type: str = ""       # paper / blog
    author: str = ""
    publish_date: str = ""
    abstract: str = ""


@dataclass
class TextChunk:
    """A chunked segment of extracted text."""
    text: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    source_meta: Optional[SourceMeta] = None
    chunk_hash: str = ""

    @property
    def source_url(self) -> str:
        return self.source_meta.url if self.source_meta else ""

    @property
    def source_title(self) -> str:
        return self.source_meta.title if self.source_meta else ""


class PaperIngestor:
    """Ingests research papers, blog posts, and web articles."""

    def __init__(self, config=None):
        self.config = config
        self._chunk_words = 500
        self._chunk_overlap = 50
        self._arxiv_query = "cryptocurrency trading"
        self._arxiv_max_results = 10

        if config:
            self._chunk_words = getattr(config, "paper_chunk_words", 500)
            self._chunk_overlap = getattr(config, "paper_chunk_overlap", 50)
            self._arxiv_query = getattr(config, "arxiv_query", "cryptocurrency trading")
            self._arxiv_max_results = getattr(config, "arxiv_max_results", 10)

    # ── Public API ────────────────────────────────────────────

    def ingest_url(self, url: str) -> List[TextChunk]:
        """
        Ingest a web page or PDF URL.

        Detects content type and routes to appropriate extractor.

        Args:
            url: URL to a blog post, article, or PDF

        Returns:
            List of TextChunk objects
        """
        if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
            return self.ingest_pdf(url)
        return self.ingest_web_page(url)

    def ingest_web_page(self, url: str) -> List[TextChunk]:
        """Extract and chunk text from a web page using trafilatura."""
        try:
            import trafilatura
        except ImportError:
            logger.error("trafilatura not installed. Run: pip install trafilatura")
            return []

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.warning(f"Could not fetch URL: {url}")
                return []

            result = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if not result:
                logger.warning(f"No text extracted from: {url}")
                return []

            # Try to get metadata
            metadata = trafilatura.extract(
                downloaded,
                output_format="json",
                include_comments=False,
            )
            meta_dict = {}
            if metadata:
                import json
                try:
                    meta_dict = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    pass

            meta = SourceMeta(
                title=meta_dict.get("title", urlparse(url).netloc),
                url=url,
                source_type="blog",
                author=meta_dict.get("author", ""),
                publish_date=meta_dict.get("date", ""),
            )

            cleaned = self._clean_text(result)
            chunks = self._chunk_text(cleaned, meta)

            logger.info(
                f"Ingested web page '{meta.title}' → {len(chunks)} chunks "
                f"({len(cleaned.split())} words)"
            )
            return chunks

        except Exception as e:
            logger.error(f"Web page ingestion failed for {url}: {e}")
            return []

    def ingest_pdf(self, path_or_url: str) -> List[TextChunk]:
        """Extract and chunk text from a PDF file or URL."""
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber not installed. Run: pip install pdfplumber")
            return []

        try:
            # If URL, download first
            pdf_path = path_or_url
            is_url = path_or_url.startswith("http")
            if is_url:
                pdf_path = self._download_pdf(path_or_url)
                if not pdf_path:
                    return []

            pages_text = []
            with pdfplumber.open(pdf_path) as pdf:
                title = ""
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                        if i == 0 and not title:
                            # Use first line as title guess
                            lines = text.strip().split("\n")
                            if lines:
                                title = lines[0][:200]

            full_text = "\n\n".join(pages_text)
            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {path_or_url}")
                return []

            meta = SourceMeta(
                title=title or path_or_url.split("/")[-1],
                url=path_or_url,
                source_type="paper",
            )

            cleaned = self._clean_text(full_text)
            chunks = self._chunk_text(cleaned, meta)

            # Clean up downloaded file
            if is_url and pdf_path != path_or_url:
                import os
                try:
                    os.unlink(pdf_path)
                except OSError:
                    pass

            logger.info(
                f"Ingested PDF '{meta.title}' → {len(chunks)} chunks "
                f"({len(cleaned.split())} words)"
            )
            return chunks

        except Exception as e:
            logger.error(f"PDF ingestion failed for {path_or_url}: {e}")
            return []

    def fetch_arxiv_papers(self, query: Optional[str] = None,
                           max_results: Optional[int] = None) -> List[dict]:
        """
        Search arXiv for papers matching the query.

        Returns list of dicts with: title, url, pdf_url, authors, published, summary
        """
        import urllib.request
        import urllib.parse

        q = query or self._arxiv_query
        n = max_results or self._arxiv_max_results

        search_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{urllib.parse.quote(q)}"
            f"&start=0&max_results={n}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        try:
            with urllib.request.urlopen(search_url, timeout=30) as resp:
                xml_data = resp.read().decode("utf-8")
        except Exception as e:
            logger.error(f"arXiv API request failed: {e}")
            return []

        papers = []
        try:
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                published = entry.findtext("atom:published", "", ns)[:10]

                # Get URLs
                links = entry.findall("atom:link", ns)
                abs_url = ""
                pdf_url = ""
                for link in links:
                    if link.get("type") == "text/html":
                        abs_url = link.get("href", "")
                    elif link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                if not abs_url:
                    id_text = entry.findtext("atom:id", "", ns)
                    abs_url = id_text

                authors = [
                    a.findtext("atom:name", "", ns)
                    for a in entry.findall("atom:author", ns)
                ]

                papers.append({
                    "title": title,
                    "url": abs_url,
                    "pdf_url": pdf_url,
                    "authors": authors,
                    "published": published,
                    "summary": summary[:500],
                })

        except ET.ParseError as e:
            logger.error(f"arXiv XML parse error: {e}")

        logger.info(f"Found {len(papers)} arXiv papers for query '{q}'")
        return papers

    def ingest_arxiv_papers(self, query: Optional[str] = None,
                            max_results: Optional[int] = None) -> List[TextChunk]:
        """
        Fetch and ingest arXiv papers matching query.

        Uses PDF extraction for full text, falling back to abstract.
        """
        papers = self.fetch_arxiv_papers(query, max_results)
        all_chunks: List[TextChunk] = []

        for paper in papers:
            try:
                # Try PDF first
                if paper.get("pdf_url"):
                    chunks = self.ingest_pdf(paper["pdf_url"])
                    if chunks:
                        # Override metadata with arXiv info
                        meta = SourceMeta(
                            title=paper["title"],
                            url=paper["url"],
                            source_type="paper",
                            author=", ".join(paper.get("authors", [])),
                            publish_date=paper.get("published", ""),
                            abstract=paper.get("summary", ""),
                        )
                        for c in chunks:
                            c.source_meta = meta
                        all_chunks.extend(chunks)
                        continue

                # Fall back to abstract only
                if paper.get("summary"):
                    meta = SourceMeta(
                        title=paper["title"],
                        url=paper["url"],
                        source_type="paper",
                        author=", ".join(paper.get("authors", [])),
                        publish_date=paper.get("published", ""),
                        abstract=paper.get("summary", ""),
                    )
                    chunks = self._chunk_text(paper["summary"], meta)
                    all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Error ingesting paper '{paper.get('title', '')}': {e}")

        return all_chunks

    # ── Text Processing ───────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove artifacts."""
        # Remove page numbers / headers / footers common in PDFs
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove URLs (they don't add value in chunks)
        text = re.sub(r"https?://\S+", "", text)

        # Normalize whitespace within lines
        text = re.sub(r"[ \t]+", " ", text)

        # Remove very short lines (artifacts)
        lines = text.split("\n")
        lines = [l.strip() for l in lines if len(l.strip()) > 10 or l.strip() == ""]
        text = "\n".join(lines).strip()

        return text

    def _chunk_text(self, text: str, meta: SourceMeta) -> List[TextChunk]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []

        chunks: List[TextChunk] = []
        step = self._chunk_words - self._chunk_overlap

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self._chunk_words]
            if len(chunk_words) < 20:
                continue

            chunk_text = " ".join(chunk_words)
            chunk_hash = hashlib.sha256(chunk_text.lower().encode()).hexdigest()[:16]

            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                total_chunks=0,
                source_meta=meta,
                chunk_hash=chunk_hash,
            ))

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    # ── Helpers ────────────────────────────────────────────────

    def _download_pdf(self, url: str) -> Optional[str]:
        """Download a PDF to a temp file and return the path."""
        import tempfile
        import urllib.request

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            urllib.request.urlretrieve(url, tmp.name)
            return tmp.name
        except Exception as e:
            logger.error(f"PDF download failed for {url}: {e}")
            return None
