"""
AI Learning Module — knowledge ingestion pipeline for crypto trading.

Ingests YouTube transcripts, research papers, and trading blogs.
Extracts structured knowledge using LLM and stores in a searchable
knowledge base with vector embeddings.
"""

from ai_learning.knowledge_base import KnowledgeBase, KnowledgeEntry, SearchResult
from ai_learning.knowledge_extractor import KnowledgeExtractor, ExtractionResult
from ai_learning.youtube_ingestor import YouTubeIngestor, VideoMeta, TranscriptChunk
from ai_learning.paper_ingestor import PaperIngestor, SourceMeta, TextChunk
from ai_learning.advisor import TradingAdvisor, AdvisorResult, ConsultationRecord
from ai_learning.improvement_db import ImprovementDB, Hypothesis, AnalysisReport, ParameterSnapshot
from ai_learning.self_improver import (
    PerformanceAnalyzer,
    HypothesisGenerator,
    HypothesisValidator,
    ValidationResult,
    ImprovementManager,
    PROTECTED_PARAMS,
)

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "SearchResult",
    "KnowledgeExtractor",
    "ExtractionResult",
    "YouTubeIngestor",
    "VideoMeta",
    "TranscriptChunk",
    "PaperIngestor",
    "SourceMeta",
    "TextChunk",
    "TradingAdvisor",
    "AdvisorResult",
    "ConsultationRecord",
    "ImprovementDB",
    "Hypothesis",
    "AnalysisReport",
    "ParameterSnapshot",
    "PerformanceAnalyzer",
    "HypothesisGenerator",
    "HypothesisValidator",
    "ValidationResult",
    "ImprovementManager",
    "PROTECTED_PARAMS",
]
