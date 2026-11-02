"""The serving path (design §3): encode, project, search, fetch text, rerank, with a timing breakdown per query."""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from . import durable
from .bridge import SpaceBridge
from .mount import IndexReader, QueryEncoder
from .quality import size_rerank_pool

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
        return ids[0], scores[0]


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
    confidence: float               # predicted top-10 agreement with native retrieval (design §5.1)
    pool: int                       # candidates sent to the reranker
    timing_ms: dict[str, float] = field(default_factory=dict)  # encode, project, search, fetch, rerank, total


class Searcher:
    """One call per query. ``pool`` defaults to the report's advisory pool, or the index size's if the bridge has no report
    (above 200K documents a reranker is required)."""

    def __init__(
        self, bridge: SpaceBridge, encoder: QueryEncoder, backend: SearchBackend, index: IndexReader,
        reranker: Optional[Reranker] = None, pool: Optional[int] = None, device: str = "cpu",
    ):
        self.bridge, self.encoder, self.backend, self.index, self.reranker = bridge, encoder, backend, index, reranker
        if pool is None:
            pool = bridge.quality.rerank_pool if bridge.quality is not None else size_rerank_pool(len(index))
        self.pool = pool
        self.device = device

    def _now(self) -> float:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def search(self, query: str, k: int = 5, query_lang: Optional[str] = None, pool: Optional[int] = None) -> SearchResult:
        pool = pool or self.pool
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
        titles, texts = self.index.titles(ids), self.index.texts(ids)
        docs = [f"{title}\n{text}" if title else text for title, text in zip(titles, texts)]
        lap("fetch")
        if self.reranker is not None:
            scores = self.reranker.score(query, docs)
            order = np.argsort(-scores, kind="stable")
            ids, scores, docs = ids[order], scores[order], [docs[i] for i in order]
        lap("rerank")
        timing["total"] = 1e3 * (t - t0)
        return SearchResult(ids[:k], scores[:k], docs[:k], float(confidence[0]), pool, timing)
