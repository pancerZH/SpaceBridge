"""bf16 encoding with an on-disk embedding cache under data/ (design §7.1)."""

import gc
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .models import ModelSpec

DEFAULT_CACHE = Path("data/cache/embeddings")
# In-process memo of recently used cached embeddings, so loops over pairs, K and seeds read each file once.
MEMO_BYTES = 6 * 2**30
_MEMO: "OrderedDict[Path, np.ndarray]" = OrderedDict()


class Encoder:
    def __init__(self, spec: ModelSpec, device: str = "cuda", cache_dir: Path = DEFAULT_CACHE):
        self.spec = spec
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir / (spec.cache_key or spec.key)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.spec.hf_id,
                device=self.device,
                model_kwargs={"torch_dtype": torch.bfloat16},
                processor_kwargs=self.spec.tokenizer_kwargs,
                config_kwargs=self.spec.config_kwargs or None,
            )
            self._model.max_seq_length = self.spec.max_seq_length
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

    def encode(self, inputs: list[str], cache: bool = True) -> np.ndarray:
        """L2-normalized float32 embeddings of fully formatted inputs, cached as float16."""
        path = self.cache_dir / f"{self._key(inputs)}.npy"
        if cache and path in _MEMO:
            _MEMO.move_to_end(path)
            return self._slice(_MEMO[path])
        if cache and path.exists():
            emb = _l2(np.load(path).astype(np.float32))
        else:
            emb = self._encode_full(inputs)
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, emb.astype(np.float16))
                emb = _l2(emb.astype(np.float16).astype(np.float32))  # same values as a later cache hit
        if cache:
            _MEMO[path] = emb
            while sum(a.nbytes for a in _MEMO.values()) > MEMO_BYTES and len(_MEMO) > 1:
                _MEMO.popitem(last=False)
        return self._slice(emb)

    def _slice(self, emb: np.ndarray) -> np.ndarray:
        d = self.spec.truncate_dim
        return emb if d is None else _l2(emb[:, :d])

    def timed_encode(self, inputs: list[str]) -> tuple[np.ndarray, dict]:
        """Uncached encode with wall-clock and token throughput, excluding model load and warmup."""
        self._encode_full(inputs[: self.spec.batch_size])
        n_tokens = sum(len(ids) for ids in self.model.tokenizer(inputs, truncation=True, max_length=self.spec.max_seq_length)["input_ids"])
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

    def _encode_full(self, inputs: list[str]) -> np.ndarray:
        emb = self.model.encode(
            inputs,
            prompt="",  # inputs are already formatted; suppress any default prompt in the model config
            batch_size=self.spec.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(inputs) > 5 * self.spec.batch_size,
        )
        return emb.astype(np.float32)

    def _key(self, inputs: list[str]) -> str:
        h = hashlib.sha256(
            json.dumps({"hf_id": self.spec.hf_id, "max_seq_length": self.spec.max_seq_length}).encode()
        )
        for text in inputs:
            h.update(text.encode())
            h.update(b"\x00")
        return h.hexdigest()[:24]


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
