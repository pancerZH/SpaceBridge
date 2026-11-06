"""Encoding with a crash-safe on-disk embedding cache under data/ (design §7.1): bf16 on the GPU, fp32 on the CPU."""

import gc
import hashlib
import json
import shutil
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from spacebridge.encoders import CPU_BATCH_TOKENS, default_dtype, longest_first as _longest_first, run_token_batches, token_lengths

from . import cache_meta, durable
from .models import ModelSpec
from .provenance import _revision
from .store import EmbeddingStore, release_gpu

DEFAULT_CACHE = Path("data/cache/embeddings")
# In-process memo of recently used cached embeddings, so loops over pairs, K and seeds read each file once.
MEMO_BYTES = 6 * 2**30
_MEMO: "OrderedDict[Path, np.ndarray]" = OrderedDict()
# Large encodes are saved in parts, at most this far apart, so a crash or power loss costs little.
PART_ROWS = 65_536
PART_SECONDS = 120
_STORES: dict[tuple, tuple] = {}  # (titles, texts, model) -> (titles, texts, store): one hash and one file per run


class Encoder:
    def __init__(self, spec: ModelSpec, device: str = "cuda", cache_dir: Path = DEFAULT_CACHE):
        self.spec = spec
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir / cache_meta.namespace(spec.cache_key or spec.key, self.device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            release_gpu()
            self._model = SentenceTransformer(
                self.spec.hf_id,
                device=self.device,
                model_kwargs={"torch_dtype": default_dtype(self.device)},
                processor_kwargs=self.spec.tokenizer_kwargs,
                config_kwargs=self.spec.config_kwargs or None,
            )
            self._model.max_seq_length = self.spec.max_seq_length
            self._model.eval()
        return self._model

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def queries(self, texts: list[str], task: str, cache: bool = True) -> np.ndarray:
        return self.encode([self.spec.format_query(t, task) for t in texts], cache=cache)

    def docs(self, titles: list[str], texts: list[str], cache: bool = True) -> np.ndarray:
        return self.encode([self.spec.format_doc(a, b) for a, b in zip(titles, texts)], cache=cache)

    def docs_store(self, titles: list[str], texts: list[str]) -> EmbeddingStore:
        """The documents' embeddings without loading them, encoding them first if they are not cached."""
        memo = (id(titles), id(texts), self.spec.hf_id, self.spec.max_seq_length, self.spec.truncate_dim, self.cache_dir)
        if memo not in _STORES:
            inputs = [self.spec.format_doc(a, b) for a, b in zip(titles, texts)]
            path = self.cache_dir / f"{self._key(inputs)}.npy"
            self._ensure_file(inputs, path)
            _STORES[memo] = (titles, texts, EmbeddingStore(path, self.spec.truncate_dim))
        return _STORES[memo][2]

    def encode(self, inputs: list[str], cache: bool = True) -> np.ndarray:
        """L2-normalized float32 embeddings of fully formatted inputs, cached as float16."""
        if not inputs:
            return np.empty((0, self.spec.dim), dtype=np.float32)
        if not cache:
            return self._slice(self._encode_full(inputs))
        path = self.cache_dir / f"{self._key(inputs)}.npy"
        if path in _MEMO:
            _MEMO.move_to_end(path)
            return self._slice(_MEMO[path])
        self._ensure_file(inputs, path)
        emb = _l2(np.load(path).astype(np.float32))
        _MEMO[path] = emb
        while sum(a.nbytes for a in _MEMO.values()) > MEMO_BYTES and len(_MEMO) > 1:
            _MEMO.popitem(last=False)
        return self._slice(emb)

    def _slice(self, emb: np.ndarray) -> np.ndarray:
        d = self.spec.truncate_dim
        return emb if d is None else _l2(emb[:, :d])

    def timed_encode(self, inputs: list[str]) -> tuple[np.ndarray, dict]:
        """Uncached encode with wall-clock and token throughput, excluding model load and warmup."""
        self._encode_full(inputs[:32])
        n_tokens = int(token_lengths(self.model.tokenizer, inputs, max_length=self.spec.max_seq_length).sum())
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = self._slice(self._encode_full(inputs))
        if self.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        return emb, {
            "encode_ms": 1e3 * seconds,
            "n_inputs": len(inputs),
            "n_tokens": n_tokens,
            "tokens_per_s": n_tokens / seconds,
        }

    @property
    def _budget(self) -> int:
        return self.spec.batch_tokens if self.device.startswith("cuda") else CPU_BATCH_TOKENS

    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        """One batch through the model directly: per-row bookkeeping in ``encode()`` costs fast models up to 30%."""
        # prompt="": inputs are already formatted; suppress any default prompt in the model config
        features = self.model.preprocess(texts, prompt="")
        features = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in features.items()}
        with torch.inference_mode():
            emb = self.model(features)["sentence_embedding"]
        return F.normalize(emb.float(), dim=-1)  # in fp32: bf16 normalization is up to 0.4% off unit length

    def _encode_full(self, inputs: list[str]) -> np.ndarray:
        lengths = token_lengths(self.model.tokenizer, inputs, max_length=self.spec.max_seq_length)
        out: Optional[np.ndarray] = None

        def handle(batch: np.ndarray) -> None:
            nonlocal out
            emb = self._embed_batch([inputs[i] for i in batch]).cpu().numpy()
            if out is None:
                out = np.empty((len(inputs), emb.shape[1]), dtype=np.float32)
            out[batch] = emb

        run_token_batches(lengths, _longest_first(lengths), self._budget, handle)
        return out

    def _meta(self) -> dict:
        return cache_meta.describe(self.spec.hf_id, _revision("models", self.spec.hf_id), cache_meta.device_class(self.device), max_seq_length=self.spec.max_seq_length)

    def _ensure_file(self, inputs: list[str], path: Path) -> None:
        parts = path.with_suffix(".parts")
        if path.exists():
            if parts.exists():  # a crash between finishing the file and removing its parts
                shutil.rmtree(parts)
            cache_meta.check(path, self._meta())
            return
        # The sidecar comes first: a run continuing a crashed encode is checked against how that encode began, so parts made by another
        # model revision or device are never completed by this one.
        if cache_meta.sidecar(path).exists():
            cache_meta.check(path, self._meta())
        else:
            cache_meta.write(path, self._meta())
        self._encode_to_file(inputs, path, parts)

    def _encode_to_file(self, inputs: list[str], path: Path, parts: Path) -> None:
        """Encodes longest first, saving a part every ``PART_ROWS`` rows or ``PART_SECONDS`` seconds. A rerun after a
        crash continues from the last part on disk; the cache file appears only when it is complete."""
        lengths_file = parts / "lengths.npy"
        if lengths_file.exists():
            lengths = np.load(lengths_file)
        else:
            lengths = token_lengths(self.model.tokenizer, inputs, max_length=self.spec.max_seq_length)
        order = _longest_first(lengths)
        tokens_before = np.concatenate([[0], np.cumsum(lengths[order], dtype=np.int64)])
        done = _finished_parts(parts)
        start = done[-1][2] if done else 0
        pending: list[np.ndarray] = []
        first, n_pending, t_last, t_begin = start, 0, time.perf_counter(), time.perf_counter()

        def flush() -> None:
            nonlocal first, n_pending, t_last
            if not lengths_file.exists():
                durable.save_npy(lengths_file, lengths)
            end = first + n_pending
            durable.save_npy(parts / f"{first:010d}_{end:010d}.npy", np.concatenate(pending))
            pending.clear()
            first, n_pending, t_last = end, 0, time.perf_counter()
            rate = (tokens_before[end] - tokens_before[start]) / max(t_last - t_begin, 1e-9)
            eta = (tokens_before[-1] - tokens_before[end]) / max(rate, 1e-9)
            durable.progress(f"encode {self.spec.key}: {end:,} / {len(order):,} rows, {rate / 1e3:.1f}K tok/s, {eta / 60:.0f} min left")

        def handle(batch: np.ndarray) -> None:
            nonlocal n_pending
            pending.append(self._embed_batch([inputs[i] for i in batch]).half().cpu().numpy())
            n_pending += len(batch)
            if n_pending >= PART_ROWS or time.perf_counter() - t_last >= PART_SECONDS:
                flush()

        run_token_batches(lengths, order, self._budget, handle, start=start)
        if not _finished_parts(parts):  # short enough to never need a part
            out = np.empty((len(order), pending[0].shape[1]), dtype=np.float16)
            out[order] = np.concatenate(pending)
            durable.save_npy(path, out)
            return
        if pending:
            flush()
        _assemble(parts, order, path)
        shutil.rmtree(parts)

    def _key(self, inputs: list[str]) -> str:
        h = hashlib.sha256(
            json.dumps({"hf_id": self.spec.hf_id, "max_seq_length": self.spec.max_seq_length}).encode()
        )
        for text in inputs:
            h.update(text.encode())
            h.update(b"\x00")
        return h.hexdigest()[:24]


def concat_embeddings(parts: list[str | Path], target: str | Path, chunk_rows: int = 65_536) -> None:
    """One embedding file from several with the same dtype and width, one after the other, written whole or not at all."""
    arrays = [np.load(p, mmap_mode="r") for p in parts]
    if len({(a.dtype, a.shape[1]) for a in arrays}) != 1:
        raise ValueError(f"these don't line up: {[(a.dtype, a.shape) for a in arrays]}")
    header = {"descr": np.lib.format.dtype_to_descr(arrays[0].dtype), "fortran_order": False, "shape": (sum(len(a) for a in arrays), arrays[0].shape[1])}
    with durable.atomic_file(target) as f:
        np.lib.format.write_array_header_2_0(f, header)
        for a in arrays:
            for s in range(0, len(a), chunk_rows):
                f.write(np.asarray(a[s : s + chunk_rows]).tobytes())


class MountEncoder:
    """``Encoder`` as the engine's ``QueryEncoder``: cached, with the model's prompts and the dataset's task."""

    def __init__(self, spec: ModelSpec, task: str, device: str = "cuda"):
        self.spec, self.task, self.device = spec, task, device
        self.size_class = "large" if spec.params_b >= 2 else "small"
        self.identity = {
            "model": spec.hf_id, "query_template": spec.query_template, "doc_template": spec.doc_template, "task": task,
            "max_seq_length": spec.max_seq_length, "truncate_dim": spec.truncate_dim, "normalized": True,
        }
        self._encoders: dict[int, Encoder] = {}

    def _encoder(self, max_length: int) -> Encoder:
        if max_length not in self._encoders:
            self.unload()  # one model at a time on the GPU
            self._encoders = {max_length: Encoder(replace(self.spec, max_seq_length=max_length), self.device)}
        return self._encoders[max_length]

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encoder(self.spec.max_seq_length).queries(texts, self.task)

    def encode_docs(self, titles: list[str], texts: list[str], max_tokens: Optional[int] = None) -> np.ndarray:
        limit = self.spec.max_seq_length if max_tokens is None else min(max_tokens, self.spec.max_seq_length)
        return self._encoder(limit).docs(titles, texts)

    def unload(self) -> None:
        for enc in self._encoders.values():
            enc.unload()


def _finished_parts(parts: Path) -> list[tuple[Path, int, int]]:
    """Saved parts as (file, first, end) positions in encoding order, up to the first gap."""
    if not parts.exists():
        return []
    found = [(p, *map(int, p.stem.split("_"))) for p in parts.glob("*_*.npy")]
    out, end = [], 0
    for item in sorted(found, key=lambda x: x[1]):
        if item[1] != end:
            break
        out.append(item)
        end = item[2]
    return out


def _assemble(parts: Path, order: np.ndarray, path: Path) -> None:
    """Writes the finished parts to the cache file in input order, through a memory map so nothing has to fit in RAM."""
    found = _finished_parts(parts)
    if not found or found[-1][2] != len(order):
        raise RuntimeError(f"incomplete parts in {parts}")
    dim = np.load(found[0][0], mmap_mode="r").shape[1]
    with durable.atomic_file(path) as f:
        header = {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)), "fortran_order": False, "shape": (len(order), dim)}
        np.lib.format.write_array_header_1_0(f, header)
        f.flush()
        offset = f.tell()
        f.truncate(offset + len(order) * dim * 2)
        out = np.memmap(f, dtype=np.float16, mode="r+", offset=offset, shape=(len(order), dim))
        for part, a, b in found:
            out[order[a:b]] = np.load(part)
        out.flush()
        del out


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
