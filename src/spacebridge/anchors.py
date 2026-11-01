"""Anchor selection, query-proxy extraction, and fixed calibration splits (design §4.1)."""

import re
from collections import Counter
from typing import Optional

import numpy as np
import torch

FPS_THRESHOLD = 5_000
# Above this index size a fixed query error costs clearly more rank (MS MARCO, same map, Tier 3: 0.85 at 200K, 0.79 at
# 500K, 0.73 at 1M), so large indexes get more anchors, and random ones: k-means over 1M docs did worse than a random
# sample (-1.5 points Recall@30, 48/54 cells) while k-means wins on smaller sets (trec-covid, 171K: +1.3).
LARGE_INDEX_DOCS = 200_000
DEFAULT_ANCHORS = 2_048
LARGE_INDEX_ANCHORS = 8_192
MAX_DISTRACTORS = 50_000

# One token per CJK character, word, or punctuation mark: a tokenizer-free length proxy.
_TOKEN = re.compile(r"[㐀-鿿豈-﫿]|[^\W_]+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def default_anchor_count(N: int) -> int:
    """The anchor budget K for an index of N documents (design §4.1)."""
    return min(LARGE_INDEX_ANCHORS if N > LARGE_INDEX_DOCS else DEFAULT_ANCHORS, N)


def resolve_anchor_method(N: int, method: str = "auto") -> str:
    """``auto``: ``random`` above ``LARGE_INDEX_DOCS``, else ``kmeans``."""
    if method == "auto":
        return "random" if N > LARGE_INDEX_DOCS else "kmeans"
    return method


def select_anchors(E: np.ndarray, K: int, seed: int = 0, device: str = "cuda", method: str = "auto") -> np.ndarray:
    """Returns K distinct row ids of E. ``kmeans``: k-means medoids when N >= 5,000, farthest-point sampling
    below. ``random``: a uniform sample. ``auto``: :func:`resolve_anchor_method`."""
    N = E.shape[0]
    if not 0 < K <= N:
        raise ValueError(f"K={K} must be in [1, N={N}]")
    if K == N:
        return np.arange(N)
    method = resolve_anchor_method(N, method)
    if method == "random":
        return np.sort(np.random.default_rng(seed).choice(N, K, replace=False))
    if method != "kmeans":
        raise ValueError(f"unknown anchor selection {method!r}")
    E = _l2(np.asarray(E, dtype=np.float32))
    if N < FPS_THRESHOLD:
        return _farthest_point(E, K, seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return _unique_medoids(E, _spherical_kmeans(E, K, seed, device))


def _spherical_kmeans(E: np.ndarray, K: int, seed: int, device: str, iters: int = 20, chunk: int = 16_384) -> np.ndarray:
    """Unit-norm k-means centroids: k-means++ seeding, then full-batch Lloyd steps in torch.

    Replaces sklearn MiniBatchKMeans, whose greedy k-means++ init (2 + ln K trials per center over
    3 x batch_size >= 12K samples) grows with K^2 and took 46 of 47 s at K = 2,048 on fiqa.
    """
    X = torch.as_tensor(E, device=device)
    N = X.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.empty(K, dtype=torch.long, device=device)
    ids[0] = torch.randint(N, (1,), generator=gen, device=device)
    dist = (1 - X @ X[ids[0]]).clamp_min_(0)  # squared Euclidean / 2 for unit vectors
    for k in range(1, K):
        ids[k] = torch.multinomial(dist, 1, generator=gen)[0]
        dist = torch.minimum(dist, (1 - X @ X[ids[k]]).clamp_min_(0))
    C = X[ids].clone()
    assign = torch.full((N,), -1, dtype=torch.long, device=device)
    for _ in range(iters):
        new = torch.cat([(X[s : s + chunk] @ C.T).argmax(1) for s in range(0, N, chunk)])
        if torch.equal(new, assign):
            break
        assign = new
        sums = torch.zeros_like(C).index_add_(0, assign, X)
        filled = sums.norm(dim=1) > 0  # empty clusters keep their previous centroid
        C[filled] = torch.nn.functional.normalize(sums[filled], dim=-1)
    return C.cpu().numpy()


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


def extract_query_proxy(
    title: str, text: str, min_tokens: int = 32, max_tokens: int = 64,
    title_min_tokens: Optional[int] = 4, title_max_tokens: int = 16,
) -> str:
    """The title alone when it has at least ``title_min_tokens`` tokens (cut at ``title_max_tokens``);
    otherwise title plus lead sentence(s), extended to at least ``min_tokens`` and cut at ``max_tokens``.

    Titles are human-written summaries shaped like queries. With title proxies Stage 2 learns "query-like
    text -> its document" instead of "lead sentence -> its document": +4.7 to +8.6 points Recall@30 on
    trec-covid, +0.4 to +3.4 on scidocs / nq / hotpotqa, about 0 on arguana (M3). ``title_min_tokens=None``
    gives the title + lead proxy for every document.
    """
    if title_min_tokens is not None and title and count_tokens(title) >= title_min_tokens:
        return _truncate_tokens(title.strip(), title_max_tokens)
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


def make_query_proxies(titles: list[str], texts: list[str], **kwargs) -> list[str]:
    """Query proxies for a set of anchor documents: :func:`extract_query_proxy`, except that a title shared by
    another anchor is not used alone (it would pair one query with several documents; on SQuAD paragraphs,
    96% of usable titles were shared, and title-only proxies cost xquad-zh up to 6 points)."""
    counts = Counter(t.strip() for t in titles if t and t.strip())
    return [
        extract_query_proxy(t, x, **kwargs) if counts[t.strip()] <= 1 else extract_query_proxy(t, x, **{**kwargs, "title_min_tokens": None})
        for t, x in zip(titles, texts)
    ]


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
