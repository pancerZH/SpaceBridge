"""Exact inner-product search (no ANN, so benchmark numbers carry no approximation noise)."""

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from . import durable
from .store import EmbeddingStore

RANKING_CACHE = Path("data/cache/rankings")
RANKING_K = 1_000  # native and re-index rankings are cached this deep; top-1,000 costs what top-100 does


def _scan(
    Q: np.ndarray, shards: Iterable[tuple[int, torch.Tensor]], N: int, k: int, device: str, chunk: int,
    exclude: Optional[np.ndarray], snapshots: Sequence[int] = (),
) -> dict[int, np.ndarray]:
    """Running top-k over consecutive shards; returns row ids at the end of the index (key N) and at each snapshot."""
    Q_t = torch.as_tensor(Q, dtype=torch.float32)
    k = min(k, N)
    best_v, best_i, out = None, None, {}
    for start, shard in shards:
        n = shard.shape[0]
        block = max(1, min(chunk, 2**28 // n))  # score block <= 1 GB of fp32 (1M docs: 268 queries)
        vs, ids = [], []
        for s in range(0, Q_t.shape[0], block):
            scores = Q_t[s : s + block].to(device) @ shard.T
            if exclude is not None:
                ex = torch.as_tensor(exclude[s : s + block], device=device) - start
                rows = torch.nonzero((ex >= 0) & (ex < n)).squeeze(1)
                scores[rows, ex[rows]] = -torch.inf
            v, i = scores.topk(min(k, n), dim=1)
            vs.append(v)
            ids.append(i + start)
        v, i = torch.cat(vs), torch.cat(ids)
        if best_v is None:
            best_v, best_i = v, i
        else:
            top = torch.cat([best_v, v], dim=1).topk(min(k, best_v.shape[1] + v.shape[1]), dim=1)
            best_v, best_i = top.values, torch.cat([best_i, i], dim=1).gather(1, top.indices)
        del shard
        end = start + n
        if end in snapshots or end == N:
            out[end] = best_i.cpu().numpy()
    return out


def search(
    Q: np.ndarray, E: np.ndarray | EmbeddingStore, k: int = 100, device: str = "cuda", chunk: int = 1024,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns (n_queries, k) row ids of E ordered by descending inner product.

    ``exclude[i]`` (if >= 0) is a row never returned for query i, e.g. the query's own corpus entry.
    """
    device = device if torch.cuda.is_available() else "cpu"
    shards = E.shards(device) if isinstance(E, EmbeddingStore) else [(0, torch.as_tensor(E, dtype=torch.float32, device=device))]
    return _scan(Q, shards, E.shape[0], k, device, chunk, exclude)[E.shape[0]]


def search_nested(
    Q: np.ndarray, E: EmbeddingStore, sizes: Sequence[int], k: int = 1_000, device: str = "cuda", chunk: int = 1024,
    exclude: Optional[np.ndarray] = None,
) -> dict[int, np.ndarray]:
    """The top-k of each prefix ``E[:size]`` in one pass over E, so a retention-vs-size curve costs one search.
    Prefixes are fair samples only if E's rows are shuffled."""
    device = device if torch.cuda.is_available() else "cpu"
    sizes = sorted(s for s in sizes if s <= len(E))
    out = _scan(Q, E.shards(device, boundaries=sizes), len(E), k, device, chunk, exclude, snapshots=sizes)
    return {s: out[s] for s in sizes}


def cached_search(
    Q: np.ndarray, E: EmbeddingStore, k: int = 100, device: str = "cuda", exclude: Optional[np.ndarray] = None,
    cache_dir: Path = RANKING_CACHE,
) -> np.ndarray:
    """``search`` with the ranking kept on disk: native and re-index rankings repeat across every K and seed of a pair."""
    h = hashlib.sha256(f"{E.path.stem}|{E.dim}|{len(E)}".encode())
    h.update(np.asarray(Q, dtype=np.float32).tobytes())
    if exclude is not None:
        h.update(np.asarray(exclude, dtype=np.int64).tobytes())
    path = cache_dir / f"{h.hexdigest()[:24]}_k{RANKING_K}.npy"
    if path.exists():
        return np.load(path)[:, :k].astype(np.int64)
    ranked = search(Q, E, max(k, RANKING_K), device, exclude=exclude)
    durable.save_npy(path, ranked.astype(np.int32))
    return ranked[:, :k]
