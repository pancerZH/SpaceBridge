"""SpaceBridge: zero-reindexing cross-model query alignment."""

from .anchors import extract_query_proxy, make_query_proxies, make_split, sample_distractors, select_anchors
from .bridge import DOC_PAIR_TOKENS, SpaceBridge, retrieval_stats
from .bundle import CalibrationBundle
from .refine import RefineConfig

__all__ = [
    "CalibrationBundle",
    "DOC_PAIR_TOKENS",
    "RefineConfig",
    "SpaceBridge",
    "extract_query_proxy",
    "make_query_proxies",
    "make_split",
    "retrieval_stats",
    "sample_distractors",
    "select_anchors",
]
