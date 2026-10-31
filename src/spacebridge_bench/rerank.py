"""Cross-encoder reranking for end-to-end rows (design §7.3 Row 9), with an on-disk score cache."""

import gc
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_CACHE = Path("data/cache/rerank")


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    hf_id: str
    instructed: bool          # takes the dataset's task instruction as its prompt
    batch_size: int = 64
    max_length: int = 512


RERANKERS: dict[str, RerankerSpec] = {
    spec.key: spec
    for spec in [
        RerankerSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", instructed=False),
        RerankerSpec("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", instructed=True, batch_size=32),
    ]
}


class Reranker:
    """Scores (query, doc) pairs; scores are cached per (instruction, query, doc) text."""

    def __init__(self, spec: RerankerSpec, device: str = "cuda", cache_dir: Path = DEFAULT_CACHE):
        self.spec = spec
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_path = cache_dir / f"{spec.key}_L{spec.max_length}.npz"
        self._model = None
        self._cache: dict[int, float] | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.spec.hf_id,
                device=self.device,
                model_kwargs={"torch_dtype": torch.bfloat16},
                max_length=self.spec.max_length,
            )
        return self._model

    @property
    def cache(self) -> dict[int, float]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path.exists():
                with np.load(self.cache_path) as z:
                    self._cache = dict(zip(z["keys"].tolist(), z["scores"].tolist()))
        return self._cache

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.fromiter(self.cache.keys(), dtype=np.uint64, count=len(self.cache))
        scores = np.fromiter(self.cache.values(), dtype=np.float32, count=len(self.cache))
        np.savez(self.cache_path, keys=keys, scores=scores)

    def score(self, queries: list[str], docs: list[str], task: str, chunk: int = 16_384) -> np.ndarray:
        """Relevance scores of the pairs ``(queries[i], docs[i])``; higher is more relevant."""
        prompt = task if self.spec.instructed else ""
        keys = [_pair_key(prompt, q, d) for q, d in zip(queries, docs)]
        todo = list({k: i for i, k in enumerate(keys) if k not in self.cache}.values())
        for s in range(0, len(todo), chunk):
            part = todo[s : s + chunk]
            new = self._predict([queries[i] for i in part], [docs[i] for i in part], prompt)
            self.cache.update(zip((keys[i] for i in part), new.tolist()))
            self.save_cache()  # per chunk, so an interrupted run keeps its progress
        return np.array([self.cache[k] for k in keys], dtype=np.float32)

    def timed_ms_per_query(self, query: str, docs: list[str], task: str, repeats: int = 3) -> float:
        """Uncached wall-clock to rerank one query's candidate list (excludes model load and warmup)."""
        prompt = task if self.spec.instructed else ""
        self._predict([query] * len(docs), docs, prompt)
        times = []
        for _ in range(repeats):
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._predict([query] * len(docs), docs, prompt)
            if self.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return 1e3 * min(times)

    def _predict(self, queries: list[str], docs: list[str], prompt: str) -> np.ndarray:
        kwargs = {"prompt": prompt} if self.spec.instructed else {}
        scores = self.model.predict(
            list(zip(queries, docs)),
            batch_size=self.spec.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(queries) > 20 * self.spec.batch_size,
            **kwargs,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def _pair_key(prompt: str, query: str, doc: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in (prompt, query, doc):
        h.update(part.encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")


def rerank(
    reranker: Reranker, ranked: np.ndarray, depth: int, queries: list[str], doc_texts: list[str], task: str
) -> np.ndarray:
    """Reorders each query's first-stage top-``depth`` by reranker score; returns (n_queries, depth) row ids."""
    cand = ranked[:, :depth]
    q_idx = np.repeat(np.arange(len(cand)), cand.shape[1])
    scores = reranker.score([queries[i] for i in q_idx], [doc_texts[j] for j in cand.ravel()], task)
    order = np.argsort(-scores.reshape(cand.shape), axis=1, kind="stable")
    return np.take_along_axis(cand, order, axis=1)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-query Jaccard overlap of the top-k sets of two rankings."""
    return np.array([len(set(x[:k]) & set(y[:k])) / len(set(x[:k]) | set(y[:k])) for x, y in zip(a, b)])
