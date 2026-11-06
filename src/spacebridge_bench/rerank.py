"""Cross-encoder reranking for end-to-end rows (design §7.3 Row 9), with a crash-safe on-disk score cache."""

import gc
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from spacebridge.encoders import CPU_BATCH_TOKENS, default_dtype, predict_pairs

from . import cache_meta, durable
from .provenance import _revision
from .store import release_gpu

DEFAULT_CACHE = Path("data/cache/rerank")
_LOG_RECORD = np.dtype([("key", "<u8"), ("score", "<f4")])
SAVE_EVERY_SECONDS = 30  # new scores reach the disk this often, whatever the speed (858 pairs/s on GPU, 5 on CPU)
_CACHES: dict[Path, dict[int, float]] = {}  # one copy per process: loading takes seconds, and every pair needs it
_NEW: set[Path] = set()  # caches with nothing on disk when this process first loaded them: they get a sidecar before their first score


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    hf_id: str
    instructed: bool          # takes the dataset's task instruction as its prompt
    batch_tokens: int = 10_000
    max_length: int = 512
    onnx: Optional[str] = None  # scores on the CPU through this file (scripts/export_reranker_onnx.py), whatever the device


ONNX_DIR = "data/cache/onnx"
RERANKERS: dict[str, RerankerSpec] = {
    spec.key: spec
    for spec in [
        RerankerSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", instructed=False),
        RerankerSpec("qwen3-reranker-0.6b", "Qwen/Qwen3-Reranker-0.6B", instructed=True, batch_tokens=5_000),
        # Lighter ones, for the CPU line (Phase 1 M2, Part E); not run by default. MIT and Apache-2.0; the second is English only.
        RerankerSpec("bge-reranker-base", "BAAI/bge-reranker-base", instructed=False),
        RerankerSpec("ms-marco-minilm", "cross-encoder/ms-marco-MiniLM-L6-v2", instructed=False),
        RerankerSpec("bge-reranker-v2-m3-int8", "BAAI/bge-reranker-v2-m3", instructed=False, onnx=f"{ONNX_DIR}/bge-reranker-v2-m3/model_int8.onnx"),
    ]
}


class OnnxCrossEncoder:
    """What scoring uses of a ``CrossEncoder``, over an ONNX file, on the CPU. Needs ``--extra onnx``. Scores are raw logits."""

    def __init__(self, path: str | Path, hf_id: str, max_length: int = 512, threads: Optional[int] = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.tokenizer, self.max_length = AutoTokenizer.from_pretrained(hf_id), max_length
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def predict(self, pairs, batch_size=None, convert_to_numpy=True, show_progress_bar=False) -> np.ndarray:
        enc = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
        return self.session.run(["logits"], {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")})[0].reshape(-1)


class Reranker:
    """Scores (query, doc) pairs; scores are cached per (instruction, query, doc) text.

    The cache is a read-only ``.npz`` from earlier runs plus one append-only log per process, so concurrent jobs
    cannot overwrite each other and a crash loses at most the last ``SAVE_EVERY_SECONDS`` of scoring.
    """

    def __init__(self, spec: RerankerSpec, device: str = "cuda", cache_dir: Path = DEFAULT_CACHE):
        self.spec = spec
        self.device = device if torch.cuda.is_available() else "cpu"
        # An ONNX export runs on the CPU whatever the device, and its key already names it. Otherwise CPU scores get their own namespace.
        name = spec.key if spec.onnx else cache_meta.namespace(spec.key, self.device)
        self.cache_path = cache_dir / f"{name}_L{spec.max_length}.npz"
        self._model = None

    precision: Optional[str] = None  # set by subclasses that run at another precision than the device's default

    def _meta(self) -> dict:
        cls = cache_meta.device_class("cpu", "int8") if self.spec.onnx else cache_meta.device_class(self.device, self.precision)
        return cache_meta.describe(self.spec.hf_id, _revision("models", self.spec.hf_id), cls, max_length=self.spec.max_length)

    @property
    def model(self):
        if self._model is None:
            if self.spec.onnx:
                self._model = OnnxCrossEncoder(self.spec.onnx, self.spec.hf_id, self.spec.max_length)
                return self._model
            from sentence_transformers import CrossEncoder

            release_gpu()
            self._model = CrossEncoder(
                self.spec.hf_id,
                device=self.device,
                model_kwargs={"torch_dtype": default_dtype(self.device)},
                max_length=self.spec.max_length,
            )
        return self._model

    @property
    def cache(self) -> dict[int, float]:
        if self.cache_path not in _CACHES:
            cache: dict[int, float] = {}
            if self.cache_path.exists():
                with np.load(self.cache_path) as z:
                    cache.update(zip(z["keys"].tolist(), z["scores"].tolist()))
            logs = sorted(self.cache_path.parent.glob(f"{self.cache_path.stem}.*.scores"))
            for log in logs:
                n = log.stat().st_size // _LOG_RECORD.itemsize  # a power loss can tear the last record
                records = np.fromfile(log, dtype=_LOG_RECORD, count=n)
                cache.update(zip(records["key"].tolist(), records["score"].tolist()))
            # Decided from the disk, before anything is scored: a sidecar is always checked; data with no sidecar is a legacy cache,
            # used and counted; nothing at all is a new cache, which gets its sidecar before its first score is written.
            if cache_meta.sidecar(self.cache_path).exists() or self.cache_path.exists() or logs:
                cache_meta.check(self.cache_path, self._meta())
            else:
                _NEW.add(self.cache_path)
            _CACHES[self.cache_path] = cache
        return _CACHES[self.cache_path]

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _describe_new_cache(self) -> None:
        """A new cache says how it was made, before it holds anything."""
        if self.cache_path in _NEW:
            cache_meta.write(self.cache_path, self._meta())
            _NEW.discard(self.cache_path)

    def _save_scores(self, keys: list[int], scores: np.ndarray) -> None:
        self._describe_new_cache()
        records = np.empty(len(keys), dtype=_LOG_RECORD)
        records["key"], records["score"] = np.array(keys, dtype=np.uint64), scores
        durable.append_bytes(self.cache_path.parent / f"{self.cache_path.stem}.{os.getpid()}.scores", records.tobytes())

    def score(self, queries: list[str], docs: list[str], task: str, chunk: int = 1_024) -> np.ndarray:
        """Relevance scores of the pairs ``(queries[i], docs[i])``; higher is more relevant."""
        prompt = task if self.spec.instructed else ""
        keys = [_pair_key(prompt, q, d) for q, d in zip(queries, docs)]
        cache = self.cache
        todo = list({k: i for i, k in enumerate(keys) if k not in cache}.values())
        if todo:
            self._describe_new_cache()
        pos, t_begin = 0, time.perf_counter()
        while pos < len(todo):
            t0 = time.perf_counter()
            part = todo[pos : pos + chunk]
            new = self._predict([queries[i] for i in part], [docs[i] for i in part], prompt)
            part_keys = [keys[i] for i in part]
            cache.update(zip(part_keys, new.tolist()))
            self._save_scores(part_keys, new)
            pos += len(part)
            # Sized to the speed, so a slow device still saves often.
            chunk = int(np.clip(len(part) * SAVE_EVERY_SECONDS / max(time.perf_counter() - t0, 1e-3), 64, 65_536))
            rate = pos / max(time.perf_counter() - t_begin, 1e-9)
            durable.progress(f"rerank {self.spec.key}: {pos:,} / {len(todo):,} pairs, {rate:.0f} pairs/s, {(len(todo) - pos) / rate / 60:.0f} min left")
        return np.array([cache[k] for k in keys], dtype=np.float32)

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
        budget = self.spec.batch_tokens if self.device.startswith("cuda") and not self.spec.onnx else CPU_BATCH_TOKENS
        return predict_pairs(self.model, queries, docs, budget, self.spec.max_length, **kwargs)


def _pair_key(prompt: str, query: str, doc: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in (prompt, query, doc):
        h.update(part.encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")


def rerank(
    reranker: Reranker, ranked: np.ndarray, depth: int, queries: list[str], doc_texts, task: str
) -> np.ndarray:
    """Reorders each query's first-stage top-``depth`` by reranker score; returns (n_queries, depth) row ids."""
    cand = ranked[:, :depth]
    q_idx = np.repeat(np.arange(len(cand)), cand.shape[1])
    scores = reranker.score([queries[i] for i in q_idx], [doc_texts[j] for j in cand.ravel()], task)
    order = np.argsort(-scores.reshape(cand.shape), axis=1, kind="stable")
    return np.take_along_axis(cand, order, axis=1)


def rerank_pools(reranker: Reranker, pools: list[np.ndarray], queries: list[str], doc_texts, task: str) -> np.ndarray:
    """Like ``rerank``, for candidate lists that may differ in length: (n_queries, longest) row ids, padded with -1. Ties keep
    each list's order."""
    flat = np.concatenate(pools)
    q_idx = np.repeat(np.arange(len(pools)), [len(x) for x in pools])
    scores = reranker.score([queries[i] for i in q_idx], [doc_texts[j] for j in flat.tolist()], task)
    out = np.full((len(pools), max(len(x) for x in pools)), -1, dtype=np.int64)
    at = 0
    for i, x in enumerate(pools):
        out[i, : len(x)] = x[np.argsort(-scores[at : at + len(x)], kind="stable")]
        at += len(x)
    return out


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 5) -> np.ndarray:
    """Per-query Jaccard overlap of the top-k sets of two rankings."""
    return np.array([len(set(x[:k]) & set(y[:k])) / len(set(x[:k]) | set(y[:k])) for x, y in zip(a, b)])
