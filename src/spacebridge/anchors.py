"""Anchor selection, query-proxy extraction, and fixed calibration splits (design §4.1)."""

import re

import numpy as np
from sklearn.cluster import MiniBatchKMeans

FPS_THRESHOLD = 5_000
MAX_DISTRACTORS = 50_000

# One token per CJK character, word, or punctuation mark: a tokenizer-free length proxy.
_TOKEN = re.compile(r"[㐀-鿿豈-﫿]|[^\W_]+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def select_anchors(E: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Returns K distinct row ids of E: k-means medoids when N >= 5,000, farthest-point sampling below."""
    N = E.shape[0]
    if not 0 < K <= N:
        raise ValueError(f"K={K} must be in [1, N={N}]")
    if K == N:
        return np.arange(N)
    E = _l2(np.asarray(E, dtype=np.float32))
    if N < FPS_THRESHOLD:
        return _farthest_point(E, K, seed)
    km = MiniBatchKMeans(n_clusters=K, batch_size=max(4096, 4 * K), n_init=1, random_state=seed)
    km.fit(E)
    return _unique_medoids(E, _l2(km.cluster_centers_.astype(np.float32)))


def _unique_medoids(E: np.ndarray, C: np.ndarray, n_cand: int = 8, chunk: int = 256) -> np.ndarray:
    """Maps each centroid to its most similar document, never assigning one document twice."""
    K, N = C.shape[0], E.shape[0]
    n_cand = min(n_cand, N)
    cand_ids = np.empty((K, n_cand), dtype=np.int64)
    cand_best = np.empty(K, dtype=np.float32)
    for s in range(0, K, chunk):
        sims = C[s : s + chunk] @ E.T
        top = np.argpartition(-sims, n_cand - 1, axis=1)[:, :n_cand]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1)
        cand_ids[s : s + chunk] = np.take_along_axis(top, order, axis=1)
        cand_best[s : s + chunk] = np.take_along_axis(top_sims, order[:, :1], axis=1)[:, 0]

    taken = np.zeros(N, dtype=bool)
    out = np.empty(K, dtype=np.int64)
    # Tightest clusters pick first; collisions fall through to the next candidate.
    for k in np.argsort(-cand_best):
        free = cand_ids[k][~taken[cand_ids[k]]]
        if free.size:
            j = int(free[0])
        else:
            sims = E @ C[k]
            sims[taken] = -np.inf
            j = int(np.argmax(sims))
        taken[j] = True
        out[k] = j
    return out


def _farthest_point(E: np.ndarray, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = [int(rng.integers(E.shape[0]))]
    min_dist = 1.0 - E @ E[ids[0]]
    min_dist[ids[0]] = -np.inf
    for _ in range(K - 1):
        j = int(np.argmax(min_dist))
        ids.append(j)
        min_dist = np.minimum(min_dist, 1.0 - E @ E[j])
        min_dist[ids] = -np.inf
    return np.asarray(ids, dtype=np.int64)


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _truncate_tokens(text: str, max_tokens: int) -> str:
    for i, m in enumerate(_TOKEN.finditer(text)):
        if i == max_tokens - 1:
            return text[: m.end()]
    return text


def extract_query_proxy(title: str, text: str, min_tokens: int = 32, max_tokens: int = 64) -> str:
    """Title plus lead sentence(s), extended to at least ``min_tokens`` and cut at ``max_tokens``."""
    parts = [title.strip()] if title and title.strip() else []
    n = sum(count_tokens(p) for p in parts)
    has_sentence = False
    for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
        if not sentence:
            continue
        if has_sentence and n >= min_tokens:
            break
        parts.append(sentence)
        n += count_tokens(sentence)
        has_sentence = True
    return _truncate_tokens("\n".join(parts), max_tokens)


def make_split(K: int, val_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed train/val split over anchor positions; K=2048 gives 1,638 / 410."""
    perm = np.random.default_rng(seed).permutation(K)
    n_val = round(val_frac * K)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def sample_distractors(N: int, exclude: np.ndarray, n: int = MAX_DISTRACTORS, seed: int = 0) -> np.ndarray:
    """Uniformly samples up to ``n`` index rows outside ``exclude`` for the rho_val pool."""
    mask = np.ones(N, dtype=bool)
    mask[exclude] = False
    pool = np.flatnonzero(mask)
    if pool.size <= n:
        return pool
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
