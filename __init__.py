
"""Savant RRF – RRFSAVANTMADE-based semantic pipeline.

This package exposes the main pipeline and components used to build
a conceptual quality layer on top of large language models, using
the RRF semantic space (antonypamo/RRFSAVANTMADE).
"""

from .pipeline import (
    RRFEmbedder,
    RRFMetricExtractor,
    RRFQualityMetaModel,
    RRFSemanticSearch,
    RRFReRanker,
    RRFSavantPipeline,
)

__all__ = [
    "RRFEmbedder",
    "RRFMetricExtractor",
    "RRFQualityMetaModel",
    "RRFSemanticSearch",
    "RRFReRanker",
    "RRFSavantPipeline",
]
