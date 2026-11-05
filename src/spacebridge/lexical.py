"""The lexical channel (design §4.6): BM25 over the index's own text, a quarter of the rerank pool.

The map loses detail that can pick out the right document, and the calibration changes tested did not recover it (HotpotQA study).
BM25 doesn't use the map at all, so it holds much of what the map loses. This implementation leaves comparing the two lists to a
reranker rather than fusing their scores, and only queries in the index's language can match its words.
"""

import array
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from . import durable, storage
from .identity import check_text_fingerprint, text_fingerprint

LEXICAL_SHARE = 4  # BM25 gets a quarter of the budget
BUILD_BATCH_ROWS = 65_536
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be but by for from has he in is it its of on or that the to was were which who what when where why how with".split())


def tokens(text: str) -> list[str]:
    """Lowercased ``[a-z0-9]+`` words without stop words. English only: Chinese text yields nothing."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class LexicalIndex:
    """BM25 (k1 = 0.9, b = 0.4) over title and text, as one sparse matrix of ready-weighted term frequencies."""

    def __init__(self, X: sp.csc_matrix, idf: np.ndarray, words: list[str], corpus: Optional[dict] = None):
        self.X, self.idf = X, idf
        self.vocab = {w: i for i, w in enumerate(words)}
        self.corpus = corpus  # text_fingerprint of the corpus it was built on; None if saved before these existed

    def __len__(self) -> int:
        return self.X.shape[0]

    @classmethod
    def build(cls, index, k1: float = 0.9, b: float = 0.4, batch_rows: int = BUILD_BATCH_ROWS) -> "LexicalIndex":
        """Reads an ``IndexReader``'s titles and text in batches. On the CPU, with no GPU route: 14-22 s at 1M, 128 s at 8.84M."""
        N = len(index)
        vocab: dict[str, int] = {}
        indptr, idx, tf = array.array("q", [0]), array.array("i"), array.array("f")
        for start in range(0, N, batch_rows):
            ids = np.arange(start, min(start + batch_rows, N))
            for title, text in zip(index.titles(ids), index.texts(ids)):
                for w, n in Counter(tokens(f"{title} {text}")).items():
                    idx.append(vocab.setdefault(w, len(vocab)))
                    tf.append(n)
                indptr.append(len(idx))
        X = sp.csr_matrix((np.frombuffer(tf, np.float32), np.frombuffer(idx, np.int32), np.frombuffer(indptr, np.int64)), shape=(N, len(vocab)), copy=True)
        del idx, tf, indptr
        # Replace each term frequency by its BM25 weight, tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / mean_length)), so a
        # query only sums columns times idf. The sparse data runs row by row, so each entry needs its own row's length.
        dl = np.asarray(X.sum(1)).ravel()
        length_of_entry = np.repeat(dl, np.diff(X.indptr))
        X.data = X.data * (k1 + 1) / (X.data + k1 * (1 - b + b * length_of_entry / dl.mean()))
        X = X.tocsc()
        df = np.diff(X.indptr)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        return cls(X, idf, list(vocab), corpus=text_fingerprint(index))

    def check_corpus(self, index) -> None:
        """Raises if this BM25 index was built on another corpus or row order."""
        if len(self) != len(index):
            raise ValueError(f"the lexical index has {len(self):,} documents and the index {len(index):,}")
        check_text_fingerprint(self.corpus, index, "this lexical index")

    def top(self, query: str, k: int, exclude=None) -> np.ndarray:
        """Up to ``k`` rows, best first, ties in row order, so ``top(q, m)`` is the head of ``top(q, k)``; a query with no
        word in the index gets none. ``exclude`` is a row or rows never returned; negative ones are ignored."""
        # Sorted: the order of a set of strings changes between processes, and with it the rounding of the sum.
        cols = sorted({self.vocab[w] for w in tokens(query) if w in self.vocab})
        if not cols or k <= 0:
            return np.empty(0, dtype=np.int64)
        s = np.asarray(self.X[:, cols] @ self.idf[cols]).ravel()
        if exclude is not None:
            rows = np.atleast_1d(exclude)
            s[rows[rows >= 0]] = 0
        hits = np.flatnonzero(s)
        if len(hits) > k:
            cut = len(hits) - k
            hits = hits[s[hits] >= np.partition(s[hits], cut)[cut]]  # all ties at the k-th score stay for the sort below
        return hits[np.argsort(-s[hits], kind="stable")[:k]].astype(np.int64)

    def save(self, path: str | Path) -> None:
        words = "\n".join(self.vocab).encode()  # no word holds a newline, so there's nothing to pickle
        arrays = {"data": self.X.data, "indices": self.X.indices, "indptr": self.X.indptr, "idf": self.idf, "words": np.frombuffer(words, np.uint8)}
        storage.save_dir(path, "lexical-index", arrays, {"shape": list(self.X.shape), "corpus": self.corpus})

    @classmethod
    def load(cls, path: str | Path) -> "LexicalIndex":
        arrays, meta = storage.load_dir(path, "lexical-index")
        words = arrays["words"].tobytes().decode()
        return cls(sp.csc_matrix((arrays["data"], arrays["indices"], arrays["indptr"]), shape=tuple(meta["shape"])), arrays["idf"], words.split("\n") if words else [], corpus=meta.get("corpus"))


def timed_build(index, label: str = "lexical index") -> LexicalIndex:
    t0 = time.perf_counter()
    out = LexicalIndex.build(index)
    durable.progress(f"{label}: built in {time.perf_counter() - t0:.1f} s, {len(out):,} documents, {out.X.nnz:,} postings")
    return out


def merge_pool(dense: np.ndarray, lexical: Optional[np.ndarray], budget: int) -> np.ndarray:
    """All of ``lexical``, then the top of ``dense`` up to ``budget``, no repeats and never over budget (a lexical list longer than the
    budget is cut). Dense rows come first, so ties in the reranker keep first-stage order. Ask ``dense`` for the whole budget, not the
    rest of it, since the two lists overlap."""
    keep = [] if lexical is None else list(dict.fromkeys(int(x) for x in lexical if x >= 0))[:max(budget, 0)]
    have = set(keep)
    room = budget - len(keep)
    head = []
    for d in np.asarray(dense).tolist():
        if room <= 0:
            break
        if d >= 0 and d not in have:
            head.append(d)
            have.add(d)
            room -= 1
    return np.array(head + keep, dtype=np.int64)
