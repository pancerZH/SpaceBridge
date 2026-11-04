"""SpaceBridge: zero-reindexing cross-model query alignment."""

from .anchors import (
    LARGE_INDEX_DOCS, LARGE_PROFILE_DOCS, default_anchor_count, extract_query_proxy, make_query_proxies, make_split, random_anchors,
    resolve_anchor_method, sample_distractors, select_anchors,
)
from .bridge import DOC_PAIR_TOKENS, SpaceBridge, retrieval_stats
from .bundle import CalibrationBundle
from .encoders import CrossEncoderReranker, SentenceTransformerEncoder
from .lexical import LexicalIndex, merge_pool
from .mount import Index, IndexReader, QueryEncoder, freeze, mount
from .refine import RefineConfig
from .serving import ExactBackend, HnswBackend, SearchResult, Searcher, build_hnsw

__all__ = [
    "CalibrationBundle",
    "CrossEncoderReranker",
    "DOC_PAIR_TOKENS",
    "ExactBackend",
    "HnswBackend",
    "Index",
    "IndexReader",
    "LARGE_INDEX_DOCS",
    "LARGE_PROFILE_DOCS",
    "LexicalIndex",
    "QueryEncoder",
    "RefineConfig",
    "SearchResult",
    "Searcher",
    "SentenceTransformerEncoder",
    "SpaceBridge",
    "build_hnsw",
    "default_anchor_count",
    "extract_query_proxy",
    "freeze",
    "make_query_proxies",
    "make_split",
    "merge_pool",
    "mount",
    "random_anchors",
    "resolve_anchor_method",
    "retrieval_stats",
    "sample_distractors",
    "select_anchors",
]
