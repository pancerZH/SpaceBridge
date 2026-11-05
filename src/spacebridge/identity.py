"""What ties saved artifacts to the corpus and models they were made for (design §6).

A checksum says a file is the file that was saved. It doesn't say the file belongs with this index: a bundle, bridge, BM25 index or
HNSW graph used with another corpus of the same size returns plausible, wrong results and raises nothing. A fingerprint is a small
sample of the corpus by row: it catches a corpus that was replaced or reordered, and doesn't catch one edited row outside the sample.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

SAMPLE_ROWS = 32
MIN_COSINE = 0.999  # stored vectors may be reloaded at another precision (float16, float32): same vector, slightly different bytes


def sample_rows(n_docs: int, samples: int = SAMPLE_ROWS) -> np.ndarray:
    """Evenly spaced rows, always including the first and the last when there is room."""
    return np.unique(np.linspace(0, n_docs - 1, min(n_docs, samples)).astype(np.int64)) if n_docs > 0 else np.empty(0, dtype=np.int64)


def text_digest(index, rows: np.ndarray) -> str:
    h = hashlib.sha256()
    for title, text in zip(index.titles(rows), index.texts(rows)):
        h.update((title or "").encode() + b"\x00" + (text or "").encode() + b"\x01")
    return h.hexdigest()


def text_fingerprint(index) -> dict[str, Any]:
    """Row count and a digest of sampled text: all a BM25 index depends on."""
    rows = sample_rows(len(index))
    return {"n_docs": len(index), "rows": rows.tolist(), "text_sha256": text_digest(index, rows)}


def check_text_fingerprint(fingerprint: Optional[dict], index, what: str) -> None:
    if fingerprint is None:  # saved before fingerprints existed: the caller checks what it can
        return
    if fingerprint["n_docs"] != len(index):
        raise ValueError(f"{what} was made for {fingerprint['n_docs']:,} documents and this index has {len(index):,}")
    if text_digest(index, np.asarray(fingerprint["rows"], dtype=np.int64)) != fingerprint["text_sha256"]:
        raise ValueError(f"{what} was made for another corpus or row order: the text of the sampled rows differs")


def vectors_match(a: np.ndarray, b: np.ndarray, min_cosine: float = MIN_COSINE) -> bool:
    """Whether two arrays hold the same vectors, row by row, up to storage precision."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return bool(np.all((a * b).sum(1) >= min_cosine * denominator))


@dataclass(frozen=True)
class CorpusFingerprint:
    """A sample of an index by row: its size, its text and its stored vectors."""

    n_docs: int
    dim: int
    rows: tuple[int, ...]
    text_sha256: str
    vectors: np.ndarray  # (len(rows), dim), float32

    @classmethod
    def of(cls, index) -> "CorpusFingerprint":
        rows = sample_rows(len(index))
        vectors = np.asarray(index.vectors(rows), dtype=np.float32)
        return cls(len(index), int(vectors.shape[1]) if vectors.ndim == 2 else 0, tuple(rows.tolist()), text_digest(index, rows), vectors)

    def check(self, index, what: str) -> None:
        if self.n_docs != len(index):
            raise ValueError(f"{what} was made for {self.n_docs:,} documents and this index has {len(index):,}")
        rows = np.asarray(self.rows, dtype=np.int64)
        if text_digest(index, rows) != self.text_sha256:
            raise ValueError(f"{what} was made for another corpus or row order: the text of the sampled rows differs")
        if not vectors_match(index.vectors(rows), self.vectors):
            raise ValueError(f"{what} was made for other vectors: the same text, but the sampled rows' stored vectors differ. Was the index re-embedded?")

    def to_meta(self) -> dict[str, Any]:
        return {"n_docs": self.n_docs, "dim": self.dim, "rows": list(self.rows), "text_sha256": self.text_sha256}

    @classmethod
    def from_saved(cls, meta: Optional[dict], vectors: Optional[np.ndarray]) -> Optional["CorpusFingerprint"]:
        if meta is None or vectors is None:
            return None
        return cls(meta["n_docs"], meta["dim"], tuple(meta["rows"]), meta["text_sha256"], vectors)


def encoder_identity(encoder) -> Optional[dict]:
    """What an encoder says it is, if anything: custom encoders may define ``identity``, a JSON-ready dict of model and settings."""
    identity = getattr(encoder, "identity", None)
    return dict(identity) if identity else None


def check_encoder(expected: Optional[dict], encoder, what: str) -> None:
    """Raises if an encoder is known to differ from the one an artifact was fitted with. Silent when either side doesn't say."""
    actual = encoder_identity(encoder)
    if expected is None or actual is None:
        return
    diff = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
    if diff:
        detail = "; ".join(f"{k}: {expected[k]!r} when fitted, {actual[k]!r} now" for k in diff)
        raise ValueError(f"{what} was fitted with a different query encoder ({detail}). Mount again with this encoder.")
