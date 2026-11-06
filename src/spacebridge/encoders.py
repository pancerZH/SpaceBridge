"""Token-budgeted batching and the ``sentence-transformers`` encoder adapter (optional dependency)."""

import threading
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

TOKENIZE_CHUNK = 100_000  # bulk tokenizing keeps only lengths; 8.8M token lists would not fit in RAM
CPU_BATCH_TOKENS = 1_024  # smaller is faster on this CPU: 10.0 s vs. 17.4 s to rerank 60 pairs at 16,384
GPU_BATCH_TOKENS = 20_000
GPU_RERANK_TOKENS = 10_000
SIZE_CLASS_LARGE_PARAMS = 2e9  # at least 2B parameters. Measured at 0.6B and 4B only: the doc-pair pass cost far more than it gained at 4B


def token_lengths(tokenizer, *texts: list[str], max_length: int) -> np.ndarray:
    """Tokens per input, truncated as encoding will; ``texts`` is one list, or two parallel lists for pairs."""
    out = np.empty(len(texts[0]), dtype=np.int32)
    for s in range(0, len(out), TOKENIZE_CHUNK):
        ids = tokenizer(*(t[s : s + TOKENIZE_CHUNK] for t in texts), truncation=True, max_length=max_length)["input_ids"]
        out[s : s + TOKENIZE_CHUNK] = [len(x) for x in ids]
    return out


def run_token_batches(
    lengths: np.ndarray, order: np.ndarray, budget: int, handle: Callable[[np.ndarray], None], start: int = 0,
) -> None:
    """Calls ``handle`` on batches of ``order[start:]`` holding at most ``budget`` tokens once padded.

    ``order`` is longest first, so batches have under 3% padding. Sorting by characters with a fixed batch size wastes
    31-37% of the GPU on padding, and larger fixed batches are slower. Out of memory halves the budget and retries.
    """
    pos = start
    while pos < len(order):
        batch = order[pos : pos + max(1, budget // max(int(lengths[order[pos]]), 1))]
        try:
            handle(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            budget //= 2
            torch.cuda.empty_cache()
            continue
        pos += len(batch)


def longest_first(lengths: np.ndarray) -> np.ndarray:
    return np.argsort(-lengths, kind="stable")


def default_dtype(device: str) -> torch.dtype:
    """bf16 is as fast as fp16 on Blackwell but 2-4x slower than fp32 on a CPU without bf16 instructions."""
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


class SentenceTransformerEncoder:
    """A ``spacebridge.mount.QueryEncoder`` over a ``sentence-transformers`` model. Set the length limit with ``max_seq_length`` here:
    changing ``model.max_seq_length`` afterwards has no effect, so that concurrent calls can't change each other's length.

    Templates hold the model's native prompts, for example ``query_template="query: {text}"``; ``doc_template`` may use
    ``{body}`` (``title\\ntext``), ``{title_or_none}`` and ``{text}``.
    """

    def __init__(
        self, model: str, query_template: str = "{text}", doc_template: str = "{body}", device: str = "cuda",
        batch_tokens: Optional[int] = None, max_seq_length: Optional[int] = None, **model_kwargs,
    ):
        from sentence_transformers import SentenceTransformer

        self.device = device if torch.cuda.is_available() else "cpu"
        self.query_template, self.doc_template = query_template, doc_template
        self.batch_tokens = batch_tokens or (GPU_BATCH_TOKENS if self.device.startswith("cuda") else CPU_BATCH_TOKENS)
        self.model = SentenceTransformer(
            model, device=self.device, model_kwargs={"torch_dtype": default_dtype(self.device)}, **model_kwargs,
        )
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.model.eval()
        params = sum(p.numel() for p in self.model.parameters())
        self.size_class = "large" if params >= SIZE_CLASS_LARGE_PARAMS else "small"
        self.model_name, self._lock = model, threading.RLock()
        self._max_seq_length = self.model.max_seq_length

    @property
    def identity(self) -> dict:
        """What an artifact fitted with this encoder depends on. The commit is read from the loaded model, when it says."""
        config = getattr(getattr(self.model[0], "auto_model", None), "config", None)
        return {
            "model": self.model_name, "revision": getattr(config, "_commit_hash", None), "query_template": self.query_template,
            "doc_template": self.doc_template, "max_seq_length": self._max_seq_length, "normalized": True,
        }

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        # The configured limit, never self.model.max_seq_length: an overlapping call may have set that to its own length.
        return self._encode([self.query_template.format(text=t) for t in texts], self._max_seq_length)

    def encode_docs(self, titles: list[str], texts: list[str], max_tokens: Optional[int] = None) -> np.ndarray:
        if len(titles) != len(texts):
            raise ValueError(f"{len(titles)} titles and {len(texts)} texts: they must be row-aligned")
        inputs = [
            self.doc_template.format(body=f"{title}\n{text}" if title else text, title_or_none=title or "none", text=text)
            for title, text in zip(titles, texts)
        ]
        limit = self._max_seq_length
        return self._encode(inputs, limit if max_tokens is None else min(max_tokens, limit))

    def _encode(self, inputs: list[str], max_length: int) -> np.ndarray:
        if not inputs:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        # The tokenizer's limit is one setting on a shared model, so calls run one at a time: a query must not inherit a
        # mount's 128-token limit, and overlapping calls must not restore the wrong one.
        with self._lock:
            return self._encode_locked(inputs, max_length)

    def _encode_locked(self, inputs: list[str], max_length: int) -> np.ndarray:
        saved, self.model.max_seq_length = self.model.max_seq_length, max_length
        try:
            lengths = token_lengths(self.model.tokenizer, inputs, max_length=max_length)
            out: Optional[np.ndarray] = None

            def handle(batch: np.ndarray) -> None:
                nonlocal out
                features = self.model.preprocess([inputs[i] for i in batch], prompt="")
                features = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in features.items()}
                with torch.inference_mode():
                    emb = self.model(features)["sentence_embedding"]
                # In fp32: bf16 normalization is up to 0.4% off unit length.
                emb = F.normalize(emb.float(), dim=-1).cpu().numpy()
                if out is None:
                    out = np.empty((len(inputs), emb.shape[1]), dtype=np.float32)
                out[batch] = emb

            run_token_batches(lengths, longest_first(lengths), self.batch_tokens, handle)
            return out
        finally:
            self.model.max_seq_length = saved


def predict_pairs(model, queries: list[str], docs: list[str], budget: int, max_length: int, **kwargs) -> np.ndarray:
    """Cross-encoder scores of ``(queries[i], docs[i])``, in token-budgeted batches."""
    lengths = token_lengths(model.tokenizer, queries, docs, max_length=max_length)
    out = np.empty(len(queries), dtype=np.float32)

    def handle(batch: np.ndarray) -> None:
        scores = model.predict(
            [(queries[i], docs[i]) for i in batch], batch_size=len(batch), convert_to_numpy=True, show_progress_bar=False, **kwargs,
        )
        out[batch] = np.asarray(scores, dtype=np.float32).reshape(-1)

    run_token_batches(lengths, longest_first(lengths), budget, handle)
    return out


class CrossEncoderReranker:
    """A ``spacebridge.serving.Reranker`` over a ``sentence-transformers`` cross-encoder. ``bge-reranker-v2-m3`` is the
    reference: multilingual, Apache-2.0, and stable under bf16."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda", batch_tokens: Optional[int] = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.batch_tokens = batch_tokens or (GPU_RERANK_TOKENS if self.device.startswith("cuda") else CPU_BATCH_TOKENS)
        self.model = CrossEncoder(model, device=self.device, model_kwargs={"torch_dtype": default_dtype(self.device)}, max_length=max_length)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return predict_pairs(self.model, [query] * len(docs), docs, self.batch_tokens, self.max_length)
