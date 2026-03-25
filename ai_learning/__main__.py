"""
CLI entry point for the AI Learning ingestion pipeline.

Usage:
  python -m ai_learning ingest-youtube <url>       — Ingest a single YouTube video
  python -m ai_learning ingest-paper <url>         — Ingest a paper/blog/PDF
  python -m ai_learning sync-sources               — Check all monitored sources for new content
  python -m ai_learning search <query>             — Search the knowledge base
  python -m ai_learning stats                      — Show KB statistics
"""

import argparse
import asyncio
import logging
import os
import sys

# Add parent directory to path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from config import CONFIG
from ai_learning.knowledge_base import KnowledgeBase
from ai_learning.knowledge_extractor import KnowledgeExtractor
from ai_learning.youtube_ingestor import YouTubeIngestor
from ai_learning.paper_ingestor import PaperIngestor

logger = logging.getLogger("ai_learning")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Commands ────────────────────────────────────────────────────

def cmd_ingest_youtube(args):
    """Ingest a single YouTube video."""
    setup_logging(args.verbose)
    cfg = CONFIG.ai_learning
    kb = KnowledgeBase(db_path=cfg.db_path, embedding_dim=cfg.embedding_dim)

    # Check if already ingested
    if kb.was_ingested(args.url):
        print(f"Already ingested: {args.url}")
        if not args.force:
            return
        print("Re-ingesting (--force)")

    ingestor = YouTubeIngestor(cfg)
    chunks = ingestor.ingest_video(args.url)

    if not chunks:
        print("No chunks produced. Video may have no transcript or was filtered out.")
        kb.log_ingestion("youtube", args.url, status="no_content")
        return

    print(f"Extracted {len(chunks)} chunks from video")

    # Extract knowledge
    extractor = KnowledgeExtractor(cfg, kb)
    entries = asyncio.run(
        extractor.extract_from_chunks(chunks, source_type="youtube")
    )

    if not entries:
        print("No knowledge entries extracted.")
        kb.log_ingestion(
            "youtube", args.url,
            source_title=chunks[0].source_title if chunks else "",
            chunks=len(chunks), entries=0,
        )
        return

    # Store in KB
    added = kb.add_entries(entries)
    kb.log_ingestion(
        "youtube", args.url,
        source_title=chunks[0].source_title if chunks else "",
        chunks=len(chunks), entries=added,
    )

    print(f"Added {added} knowledge entries to the knowledge base")
    _print_entry_summary(entries)


def cmd_ingest_paper(args):
    """Ingest a paper, blog, or PDF."""
    setup_logging(args.verbose)
    cfg = CONFIG.ai_learning
    kb = KnowledgeBase(db_path=cfg.db_path, embedding_dim=cfg.embedding_dim)

    if kb.was_ingested(args.url) and not args.force:
        print(f"Already ingested: {args.url}")
        return

    ingestor = PaperIngestor(cfg)
    chunks = ingestor.ingest_url(args.url)

    if not chunks:
        print("No chunks produced. Could not extract text from source.")
        kb.log_ingestion("paper", args.url, status="no_content")
        return

    print(f"Extracted {len(chunks)} chunks")

    extractor = KnowledgeExtractor(cfg, kb)
    entries = asyncio.run(
        extractor.extract_from_chunks(chunks, source_type="paper")
    )

    if not entries:
        print("No knowledge entries extracted.")
        kb.log_ingestion(
            "paper", args.url,
            source_title=chunks[0].source_title if chunks else "",
            chunks=len(chunks), entries=0,
        )
        return

    added = kb.add_entries(entries)
    kb.log_ingestion(
        "paper", args.url,
        source_title=chunks[0].source_title if chunks else "",
        chunks=len(chunks), entries=added,
    )

    print(f"Added {added} knowledge entries to the knowledge base")
    _print_entry_summary(entries)


def cmd_sync_sources(args):
    """Check all monitored sources for new content."""
    setup_logging(args.verbose)
    cfg = CONFIG.ai_learning
    kb = KnowledgeBase(db_path=cfg.db_path, embedding_dim=cfg.embedding_dim)
    extractor = KnowledgeExtractor(cfg, kb)

    total_added = 0

    # 1. YouTube channels
    channels = cfg.youtube_channels
    if channels:
        print(f"\n--- YouTube Channels ({len(channels)}) ---")
        yt = YouTubeIngestor(cfg)
        for channel_id in channels:
            print(f"\nChecking channel: {channel_id}")
            try:
                chunks = yt.ingest_channel(channel_id, max_videos=10)
                # Filter out already-ingested chunks
                new_chunks = [
                    c for c in chunks if not kb.was_ingested(c.source_url)
                ]
                if not new_chunks:
                    print("  No new content")
                    continue

                print(f"  {len(new_chunks)} new chunks from {len(set(c.source_url for c in new_chunks))} videos")
                entries = asyncio.run(
                    extractor.extract_from_chunks(new_chunks, source_type="youtube")
                )
                if entries:
                    added = kb.add_entries(entries)
                    total_added += added
                    # Log each unique video
                    for url in set(c.source_url for c in new_chunks):
                        url_chunks = [c for c in new_chunks if c.source_url == url]
                        url_entries = [e for e in entries if e.source_url == url]
                        kb.log_ingestion(
                            "youtube", url,
                            source_title=url_chunks[0].source_title if url_chunks else "",
                            chunks=len(url_chunks), entries=len(url_entries),
                        )
                    print(f"  Added {added} entries")
            except Exception as e:
                logger.error(f"Error syncing channel {channel_id}: {e}")

    # 2. Monitored URLs (blogs/substacks)
    urls = cfg.monitored_urls
    if urls:
        print(f"\n--- Monitored URLs ({len(urls)}) ---")
        paper = PaperIngestor(cfg)
        for url in urls:
            if kb.was_ingested(url):
                continue
            print(f"\nIngesting: {url}")
            try:
                chunks = paper.ingest_url(url)
                if not chunks:
                    print("  No content extracted")
                    continue

                entries = asyncio.run(
                    extractor.extract_from_chunks(chunks, source_type="blog")
                )
                if entries:
                    added = kb.add_entries(entries)
                    total_added += added
                    kb.log_ingestion(
                        "blog", url,
                        source_title=chunks[0].source_title if chunks else "",
                        chunks=len(chunks), entries=added,
                    )
                    print(f"  Added {added} entries")
            except Exception as e:
                logger.error(f"Error ingesting {url}: {e}")

    # 3. arXiv papers
    print(f"\n--- arXiv Papers (query: '{cfg.arxiv_query}') ---")
    paper = PaperIngestor(cfg)
    try:
        arxiv_papers = paper.fetch_arxiv_papers()
        new_papers = [p for p in arxiv_papers if not kb.was_ingested(p["url"])]
        if not new_papers:
            print("  No new papers")
        else:
            print(f"  Found {len(new_papers)} new papers")
            for p in new_papers:
                print(f"  Processing: {p['title'][:80]}")
                try:
                    # Use abstract for quick extraction (full PDF is slow)
                    if p.get("summary"):
                        chunks = paper._chunk_text(
                            p["summary"],
                            PaperIngestor.__class__.__mro__[0]  # Will use SourceMeta
                        )
                        # Build proper chunks with metadata
                        from ai_learning.paper_ingestor import SourceMeta, TextChunk
                        import hashlib
                        meta = SourceMeta(
                            title=p["title"], url=p["url"],
                            source_type="paper",
                            author=", ".join(p.get("authors", [])),
                            publish_date=p.get("published", ""),
                            abstract=p.get("summary", ""),
                        )
                        chunk_text = p["summary"]
                        chunk_hash = hashlib.sha256(chunk_text.lower().encode()).hexdigest()[:16]
                        chunks = [TextChunk(
                            text=chunk_text, chunk_index=0, total_chunks=1,
                            source_meta=meta, chunk_hash=chunk_hash,
                        )]

                        entries = asyncio.run(
                            extractor.extract_from_chunks(chunks, source_type="paper")
                        )
                        if entries:
                            added = kb.add_entries(entries)
                            total_added += added
                            kb.log_ingestion(
                                "paper", p["url"],
                                source_title=p["title"],
                                chunks=1, entries=added,
                            )
                except Exception as e:
                    logger.error(f"Error processing paper '{p['title'][:60]}': {e}")
    except Exception as e:
        logger.error(f"arXiv sync failed: {e}")

    print(f"\n=== Sync complete: {total_added} total entries added ===")


def cmd_search(args):
    """Search the knowledge base."""
    setup_logging(args.verbose)
    cfg = CONFIG.ai_learning
    kb = KnowledgeBase(db_path=cfg.db_path, embedding_dim=cfg.embedding_dim)

    query = " ".join(args.query)
    results = kb.search(
        query,
        top_k=args.top_k,
        category=args.category,
        min_confidence=args.min_confidence,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nSearch results for: '{query}' (top {len(results)})\n")
    for i, r in enumerate(results, 1):
        e = r.entry
        print(f"  {i}. [{e.category}] (score: {r.score:.3f}, conf: {e.confidence:.2f})")
        try:
            content = json.loads(e.content)
            for k, v in content.items():
                if v:
                    print(f"     {k}: {v}")
        except (json.JSONDecodeError, TypeError):
            print(f"     {e.content[:120]}")
        print(f"     Source: {e.source_title} ({e.source_type})")
        if e.times_applied > 0:
            print(f"     Applied: {e.times_applied}x, success: {e.success_rate:.0%}")
        print()


def cmd_stats(args):
    """Show knowledge base statistics."""
    setup_logging(args.verbose)
    cfg = CONFIG.ai_learning
    kb = KnowledgeBase(db_path=cfg.db_path, embedding_dim=cfg.embedding_dim)

    stats = kb.get_stats()

    print("\n╔══════════════════════════════════════╗")
    print("║      Knowledge Base Statistics       ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  Total entries:     {stats['total_entries']:>14,}  ║")
    print(f"║  Avg confidence:    {stats['avg_confidence']:>14.3f}  ║")
    print(f"║  Total ingestions:  {stats['total_ingestions']:>14,}  ║")
    print("╠══════════════════════════════════════╣")

    if stats["categories"]:
        print("║  By Category:                        ║")
        for cat, cnt in stats["categories"].items():
            print(f"║    {cat:<18s} {cnt:>13,}  ║")

    if stats["source_types"]:
        print("╠══════════════════════════════════════╣")
        print("║  By Source Type:                     ║")
        for src, cnt in stats["source_types"].items():
            print(f"║    {src:<18s} {cnt:>13,}  ║")

    print("╚══════════════════════════════════════╝")

    if stats.get("top_applied"):
        print("\nMost Applied Knowledge:")
        for row in stats["top_applied"]:
            print(
                f"  #{row['id']} [{row['category']}] "
                f"applied {row['times_applied']}x "
                f"(success: {row['success_rate']:.0%}) — {row['source_title'][:50]}"
            )
    print()

    # Recent ingestions
    history = kb.get_ingestion_history(10)
    if history:
        print("Recent Ingestions:")
        import time as _time
        for h in history:
            ts = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(h["ingested_at"]))
            status_icon = "+" if h["status"] == "success" else "x"
            print(
                f"  [{status_icon}] {ts} {h['source_type']:8s} "
                f"{h.get('source_title', '')[:40]:40s} "
                f"chunks={h['chunks_processed']} entries={h['entries_created']}"
            )
    print()


# ── Helpers ─────────────────────────────────────────────────────

def _print_entry_summary(entries):
    """Print a summary of extracted entries by category."""
    from collections import Counter
    cats = Counter(e.category for e in entries)
    print("\nBy category:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")


# Need json for search command output
import json


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ai_learning",
        description="AI Knowledge Ingestion Pipeline for Crypto Trading Bot",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest-youtube
    sp = subparsers.add_parser("ingest-youtube", help="Ingest a YouTube video")
    sp.add_argument("url", type=str, help="YouTube video URL")
    sp.add_argument("--force", action="store_true", help="Re-ingest even if already done")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_ingest_youtube)

    # ingest-paper
    sp = subparsers.add_parser("ingest-paper", help="Ingest a paper/blog/PDF")
    sp.add_argument("url", type=str, help="URL to paper, blog post, or PDF")
    sp.add_argument("--force", action="store_true", help="Re-ingest even if already done")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_ingest_paper)

    # sync-sources
    sp = subparsers.add_parser("sync-sources", help="Sync all monitored sources")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_sync_sources)

    # search
    sp = subparsers.add_parser("search", help="Search the knowledge base")
    sp.add_argument("query", nargs="+", help="Search query")
    sp.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    sp.add_argument("--category", type=str, default=None,
                    help="Filter by category (strategy/indicator/risk_rule/insight/pattern/exit/mistake)")
    sp.add_argument("--min-confidence", type=float, default=0.0,
                    help="Minimum confidence threshold (default: 0.0)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_search)

    # stats
    sp = subparsers.add_parser("stats", help="Show knowledge base statistics")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
