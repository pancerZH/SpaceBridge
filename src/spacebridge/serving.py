"""The serving path (design §3): encode, project, search, fetch text, rerank, with a timing breakdown per query."""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from . import durable
from .anchors import LARGE_INDEX_DOCS
from .bridge import SpaceBridge
from .lexical import LEXICAL_MIN_DOCS, LEXICAL_SHARE, LexicalIndex, merge_pool, timed_build
from .mount import IndexReader, QueryEncoder
from .quality import lexical_rerank_pool, size_rerank_pool

HNSW_SAVE_SECONDS = 600  # a crash loses at most this much building
HNSW_BATCH_ROWS = 100_000


class SearchBackend(Protocol):
    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Row ids and scores of the top-k of one (1, d) query, best first."""
        ...


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Relevance of each document to the query; higher is more relevant."""
        ...


class ExactBackend:
    """Exact inner-product search over an array in memory. ``device="cuda"`` keeps the vectors on the GPU."""

    def __init__(self, vectors: np.ndarray, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            n, d = vectors.shape
            need, free = n * (d + 1) * 4, torch.cuda.mem_get_info(self.device)[0]  # the vectors, and one score per row
            if need > free:
                raise ValueError(
                    f"{n:,} x {d} vectors need {need / 2**30:.1f} GB on the GPU and {free / 2**30:.1f} GB is free; "
                    'use HnswBackend, or ExactBackend(vectors, device="cpu")'
                )
        self.vectors = torch.as_tensor(np.asarray(vectors, dtype=np.float32), device=self.device)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = torch.as_tensor(q, dtype=torch.float32, device=self.device) @ self.vectors.T
        top = scores.topk(min(k, self.vectors.shape[0]), dim=1)
        return top.indices[0].cpu().numpy(), top.values[0].cpu().numpy()


class HnswBackend:
    """A FAISS HNSW index (optional dependency ``spacebridge[hnsw]``), built with :func:`build_hnsw`."""

    def __init__(self, path: str | Path, ef_search: int = 500):
        import faiss

        self.index, self.ef_search = faiss.read_index(str(path)), ef_search

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.index.hnsw.efSearch = max(self.ef_search, k)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), k)
        found = ids[0] >= 0  # a graph can return fewer than k, padded with -1
        return ids[0][found], scores[0][found]


def save_faiss_index(index, path: Path) -> None:
    """Saves atomically: a crash during the save leaves the previous index in place."""
    import faiss

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        faiss.write_index(index, str(tmp))
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        durable.fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def build_hnsw(
    vectors, path: str | Path, M: int = 32, ef_construction: int = 200, batch_rows: int = HNSW_BATCH_ROWS,
    save_seconds: float = HNSW_SAVE_SECONDS, on_batch: Optional[Callable[[int, int], None]] = None,
):
    """Builds an inner-product HNSW index over an array-like of normalized vectors, in batches, saving it atomically as
    it grows. A rerun continues from the saved index. Returns the finished index."""
    import faiss

    path = Path(path)
    for stale in path.parent.glob(f"{path.name}.tmp*"):  # a crash during a save leaves its copy behind, as large as the index
        stale.unlink()
    if path.exists():
        index = faiss.read_index(str(path))
    else:
        index = faiss.IndexHNSWFlat(vectors.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
    n, last_save = vectors.shape[0], time.perf_counter()
    for start in range(index.ntotal, n, batch_rows):
        index.add(np.asarray(vectors[start : start + batch_rows], dtype=np.float32))
        if on_batch is not None:
            on_batch(index.ntotal, n)
        if time.perf_counter() - last_save >= save_seconds and index.ntotal < n:
            save_faiss_index(index, path)
            last_save = time.perf_counter()
    if not path.exists() or faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY).ntotal < n:
        save_faiss_index(index, path)
    return index


def real_query_overlap(
    bridge: SpaceBridge, queries: list[str], src_encoder: QueryEncoder, tgt_encoder: QueryEncoder, backend: SearchBackend,
    k: int = 30, query_lang: Optional[str] = None, src_queries: Optional[list[str]] = None,
) -> np.ndarray:
    """Per query, the share of the native top-k that the bridged query also returns. Needs unlabeled real queries and the
    source model. ``src_queries`` are the parallel queries in the index language when ``queries`` are not."""
    q_tgt = bridge.transform_query(tgt_encoder.encode_queries(queries), query_lang=query_lang)
    q_src = src_encoder.encode_queries(src_queries if src_queries is not None else queries)
    out = np.empty(len(queries))
    for i in range(len(queries)):
        native, _ = backend.search(q_src[i : i + 1], k)
        bridged, _ = backend.search(q_tgt[i : i + 1], k)
        out[i] = len(set(native.tolist()) & set(bridged.tolist())) / k
    return out


@dataclass
class SearchResult:
    ids: np.ndarray                 # row ids, best first
    scores: np.ndarray              # reranker scores, or first-stage scores without a reranker
    texts: list[str]
    confidence: float               # predicted top-10 agreement with native retrieval (design §5.1); NaN for a
                                    # source-model-free bridge, which has no teacher to calibrate it on
    pool: int                       # candidates actually sent to the reranker
    timing_ms: dict[str, float] = field(default_factory=dict)  # encode, project, search, lexical, fetch, rerank, total, and translate if one ran


class Searcher:
    """One call per query. ``pool`` defaults to the report's advisory pool, or the index size's if the bridge has no report.

    Above ``LARGE_INDEX_DOCS`` documents a reranker is required: first-stage results lose far more there (MS MARCO 1M, nDCG@5
    0.33-0.44 instead of 0.66-0.69). ``allow_no_reranker=True`` opts out, to time search alone.

    ``lexical`` puts BM25 in the rerank pool, a quarter of it (design §4.6). ``"auto"`` builds a ``LexicalIndex`` from the index
    above ``LEXICAL_MIN_DOCS`` documents, if the index is in English and there's a reranker to merge the two lists. A
    ``LexicalIndex`` is used whatever the size and language. ``None`` turns it off. With it on, the pool defaults to 60, or 100
    for a weak or source-free fit, and a query in a language other than the index's skips BM25 and takes the pool it would take
    with BM25 off.

    ``translate(query, query_lang)``, off by default, returns such a query in the index's language, so that BM25 can read it. Only
    BM25 reads the translation: the bridge and the reranker keep the original query. It needs a lexical index, so below
    ``LEXICAL_MIN_DOCS`` pass one. Measured Chinese to English only (design §4.6).
    """

    def __init__(
        self, bridge: SpaceBridge, encoder: QueryEncoder, backend: SearchBackend, index: IndexReader,
        reranker: Optional[Reranker] = None, pool: Optional[int] = None, device: str = "cpu", allow_no_reranker: bool = False,
        lexical: LexicalIndex | str | None = "auto", translate: Optional[Callable[[str, str], str]] = None,
    ):
        if reranker is None and len(index) > LARGE_INDEX_DOCS and not allow_no_reranker:
            raise ValueError(
                f"an index of {len(index):,} documents needs a reranker (required above {LARGE_INDEX_DOCS:,} documents, "
                "design §2.1); pass one, or allow_no_reranker=True to serve first-stage results anyway"
            )
        self.bridge, self.encoder, self.backend, self.index, self.reranker = bridge, encoder, backend, index, reranker
        self.lexical, self.translate = self._resolve_lexical(lexical), translate
        if translate is not None and self.lexical is None:
            raise ValueError("translate feeds BM25, so it needs a reranker and a lexical index: pass lexical=LexicalIndex.build(index)")
        self.plain_pool = pool or (bridge.quality.rerank_pool if bridge.quality is not None else size_rerank_pool(len(index)))
        self.pool = self.plain_pool
        if self.lexical is not None and pool is None:
            self.pool = lexical_rerank_pool(bridge.quality, source_free=bridge.status == "NO_TEACHER")
        self.device = device

    def _resolve_lexical(self, lexical: LexicalIndex | str | None) -> Optional[LexicalIndex]:
        if isinstance(lexical, str) and lexical != "auto":
            raise ValueError(f"lexical must be a LexicalIndex, 'auto' or None, not {lexical!r}")
        if lexical is None or self.reranker is None:  # only a reranker can merge the two lists
            return None
        if lexical != "auto":
            if len(lexical) != len(self.index):
                raise ValueError(f"the lexical index has {len(lexical):,} documents and the index {len(self.index):,}")
            return lexical
        if len(self.index) <= LEXICAL_MIN_DOCS or self.bridge.src_lang != "en":  # the tokenizer keeps only [a-z0-9]+
            return None
        return timed_build(self.index)

    def _now(self) -> float:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def search(self, query: str, k: int = 5, query_lang: Optional[str] = None, pool: Optional[int] = None) -> SearchResult:
        foreign = query_lang not in (None, self.bridge.src_lang)
        lexical = self.lexical if not foreign or self.translate is not None else None
        pool = pool or (self.pool if lexical is not None else self.plain_pool)
        timing = {}
        t0 = t = self._now()

        def lap(name: str) -> None:
            nonlocal t
            now = self._now()
            timing[name] = 1e3 * (now - t)
            t = now

        q_raw = self.encoder.encode_queries([query])
        lap("encode")
        q_hat, confidence = self.bridge.transform_query(q_raw, query_lang=query_lang, return_confidence=True)
        lap("project")
        ids, scores = self.backend.search(q_hat, pool)
        lap("search")
        text = query
        if foreign and lexical is not None:
            text = self.translate(query, query_lang)
            lap("translate")
        if lexical is not None:
            ids = merge_pool(ids, lexical.top(text, pool // LEXICAL_SHARE), pool)
        lap("lexical")
        titles, texts = self.index.titles(ids), self.index.texts(ids)
        docs = [f"{title}\n{text}" if title else text for title, text in zip(titles, texts)]
        lap("fetch")
        if self.reranker is not None:
            scores = self.reranker.score(query, docs)
            order = np.argsort(-scores, kind="stable")
            ids, scores, docs = ids[order], scores[order], [docs[i] for i in order]
        lap("rerank")
        timing["total"] = 1e3 * (t - t0)
        return SearchResult(ids[:k], scores[:k], docs[:k], float(confidence[0]), len(ids), timing)
