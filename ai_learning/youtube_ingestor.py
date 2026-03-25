"""
YouTube Transcript Ingestion — fetches video metadata and transcripts,
cleans and chunks text for downstream knowledge extraction.

Uses yt-dlp for metadata and youtube-transcript-api for transcripts.
Filters out clickbait and stale content.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Filler words to strip from transcripts ────────────────────
FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "basically",
    "actually", "literally", "right", "so yeah", "okay so",
}

FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b",
    re.IGNORECASE,
)


@dataclass
class VideoMeta:
    """Metadata for a YouTube video."""
    video_id: str = ""
    title: str = ""
    description: str = ""
    channel: str = ""
    channel_id: str = ""
    publish_date: str = ""          # ISO date string
    duration_s: int = 0
    view_count: int = 0
    url: str = ""


@dataclass
class TranscriptChunk:
    """A cleaned, chunked segment of a transcript."""
    text: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    video_meta: Optional[VideoMeta] = None
    chunk_hash: str = ""            # SHA256 of normalized text

    @property
    def source_url(self) -> str:
        return self.video_meta.url if self.video_meta else ""

    @property
    def source_title(self) -> str:
        return self.video_meta.title if self.video_meta else ""


class YouTubeIngestor:
    """Ingests YouTube video transcripts for knowledge extraction."""

    def __init__(self, config=None):
        self.config = config
        self._max_age_days = 180
        self._clickbait_patterns = [
            re.compile(r"(?i)\b(100|1000)x\b"),
            re.compile(r"(?i)guaranteed"),
            re.compile(r"(?i)get rich"),
            re.compile(r"(?i)millionaire overnight"),
            re.compile(r"(?i)free money"),
        ]
        self._chunk_words = 500
        self._chunk_overlap = 50

        if config:
            self._max_age_days = getattr(config, "youtube_max_age_days", 180)
            self._chunk_words = getattr(config, "youtube_chunk_words", 500)
            self._chunk_overlap = getattr(config, "youtube_chunk_overlap", 50)
            patterns = getattr(config, "youtube_clickbait_patterns", [])
            if patterns:
                self._clickbait_patterns = [re.compile(p) for p in patterns]

    # ── Public API ────────────────────────────────────────────

    def ingest_video(self, url: str) -> List[TranscriptChunk]:
        """
        Ingest a single YouTube video: fetch metadata, transcript,
        clean, chunk, and return ready-for-extraction chunks.

        Args:
            url: YouTube video URL

        Returns:
            List of TranscriptChunk objects
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return []

        # Fetch metadata
        meta = self._fetch_metadata(video_id)
        if not meta:
            logger.warning(f"Could not fetch metadata for {video_id}")
            meta = VideoMeta(video_id=video_id, url=url)

        # Check filters
        if self._is_clickbait(meta.title):
            logger.info(f"Skipping clickbait video: {meta.title}")
            return []

        if self._is_too_old(meta.publish_date):
            logger.info(f"Skipping old video ({meta.publish_date}): {meta.title}")
            return []

        # Fetch transcript
        raw_transcript = self._fetch_transcript(video_id)
        if not raw_transcript:
            logger.warning(f"No transcript available for: {meta.title}")
            return []

        # Clean and chunk
        cleaned = self._clean_transcript(raw_transcript)
        chunks = self._chunk_text(cleaned, meta)

        logger.info(
            f"Ingested video '{meta.title}' → {len(chunks)} chunks "
            f"({len(cleaned.split())} words)"
        )
        return chunks

    def ingest_channel(self, channel_id: str,
                       max_videos: int = 20) -> List[TranscriptChunk]:
        """
        Ingest recent videos from a YouTube channel.

        Args:
            channel_id: YouTube channel ID
            max_videos: Maximum number of recent videos to process

        Returns:
            Combined list of TranscriptChunk objects from all videos
        """
        video_urls = self._fetch_channel_videos(channel_id, max_videos)
        all_chunks: List[TranscriptChunk] = []
        for url in video_urls:
            try:
                chunks = self.ingest_video(url)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error ingesting {url}: {e}")
                continue
        return all_chunks

    # ── Metadata Fetching ─────────────────────────────────────

    def _fetch_metadata(self, video_id: str) -> Optional[VideoMeta]:
        """Fetch video metadata using yt-dlp."""
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

            publish = info.get("upload_date", "")
            if publish and len(publish) == 8:
                publish = f"{publish[:4]}-{publish[4:6]}-{publish[6:8]}"

            return VideoMeta(
                video_id=video_id,
                title=info.get("title", ""),
                description=info.get("description", "")[:500],
                channel=info.get("channel", "") or info.get("uploader", ""),
                channel_id=info.get("channel_id", ""),
                publish_date=publish,
                duration_s=info.get("duration", 0) or 0,
                view_count=info.get("view_count", 0) or 0,
                url=url,
            )
        except Exception as e:
            logger.error(f"yt-dlp metadata fetch failed for {video_id}: {e}")
            return None

    def _fetch_channel_videos(self, channel_id: str,
                              max_videos: int = 20) -> List[str]:
        """Fetch recent video URLs from a channel."""
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install yt-dlp")
            return []

        channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": True,
            "playlist_items": f"1:{max_videos}",
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
            entries = info.get("entries", [])
            urls = []
            for entry in entries:
                vid_id = entry.get("id") or entry.get("url", "")
                if vid_id:
                    urls.append(f"https://www.youtube.com/watch?v={vid_id}")
            return urls[:max_videos]
        except Exception as e:
            logger.error(f"Channel video fetch failed for {channel_id}: {e}")
            return []

    # ── Transcript Fetching ───────────────────────────────────

    def _fetch_transcript(self, video_id: str) -> str:
        """
        Fetch transcript text. Prefers manual captions, falls back to
        auto-generated.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            logger.error(
                "youtube-transcript-api not installed. "
                "Run: pip install youtube-transcript-api"
            )
            return ""

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Prefer manually created captions
            transcript = None
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
            except Exception:
                pass

            # Fall back to auto-generated
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except Exception:
                    pass

            if transcript is None:
                logger.warning(f"No English transcript found for {video_id}")
                return ""

            entries = transcript.fetch()
            # Combine all text segments
            return " ".join(entry.get("text", "") for entry in entries)

        except Exception as e:
            logger.warning(f"Transcript fetch failed for {video_id}: {e}")
            return ""

    # ── Text Processing ───────────────────────────────────────

    def _clean_transcript(self, raw: str) -> str:
        """Clean a raw transcript: remove timestamps, filler words, normalize."""
        text = raw

        # Remove timestamp-like patterns [00:00] or (00:00:00)
        text = re.sub(r"[\[\(]\d{1,2}:\d{2}(?::\d{2})?[\]\)]", "", text)

        # Remove [Music], [Applause], etc.
        text = re.sub(r"\[.*?\]", "", text)

        # Remove filler words
        text = FILLER_PATTERN.sub("", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove very short residual fragments
        text = re.sub(r"\b\w\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _chunk_text(self, text: str, meta: VideoMeta) -> List[TranscriptChunk]:
        """Split text into overlapping chunks of ~chunk_words words."""
        words = text.split()
        if not words:
            return []

        chunks: List[TranscriptChunk] = []
        step = self._chunk_words - self._chunk_overlap
        total = max(1, (len(words) - self._chunk_overlap + step - 1) // step)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self._chunk_words]
            if len(chunk_words) < 20:
                continue  # Skip very short tail chunks

            chunk_text = " ".join(chunk_words)
            chunk_hash = hashlib.sha256(chunk_text.lower().encode()).hexdigest()[:16]

            chunks.append(TranscriptChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                total_chunks=0,  # Will update below
                video_meta=meta,
                chunk_hash=chunk_hash,
            ))

        # Update total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    # ── Filters ───────────────────────────────────────────────

    def _is_clickbait(self, title: str) -> bool:
        """Check if a video title matches clickbait patterns."""
        for pattern in self._clickbait_patterns:
            if pattern.search(title):
                return True
        return False

    def _is_too_old(self, publish_date: str) -> bool:
        """Check if a video is older than max_age_days."""
        if not publish_date:
            return False  # Allow if no date available
        try:
            pub = datetime.strptime(publish_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            cutoff = datetime.now(timezone.utc) - timedelta(
                days=self._max_age_days
            )
            return pub < cutoff
        except ValueError:
            return False

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r"(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",
        ]
        for pat in patterns:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return None
