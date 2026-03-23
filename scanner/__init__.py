"""
Scanner module — Dynamic pair discovery, scoring, and selection.

Components:
- OpportunityScanner: Scans top 30 Binance USDT-M pairs, computes opportunity scores
- PairSelector: Selects best pairs from scan results with filtering
- PairPerformanceTracker: Tracks per-pair win rate, auto-enable/disable
"""

from scanner.pair_scanner import (
    OpportunityScanner,
    PairScore,
    PairSelector,
    ScanResult,
)
from scanner.pair_performance import PairPerformanceTracker

__all__ = [
    "OpportunityScanner",
    "PairScore",
    "PairSelector",
    "ScanResult",
    "PairPerformanceTracker",
]
