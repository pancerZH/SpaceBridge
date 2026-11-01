"""Exact inner-product search (no ANN, so Phase-0 numbers carry no approximation noise)."""

from typing import Optional

import numpy as np
import torch


def search(
    Q: np.ndarray, E: np.ndarray, k: int = 100, device: str = "cuda", chunk: int = 1024,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns (n_queries, k) row ids of E ordered by descending inner product.

    ``exclude[i]`` (if >= 0) is a row never returned for query i, e.g. the query's own corpus entry.
    """
    device = device if torch.cuda.is_available() else "cpu"
    E_t = torch.as_tensor(E, dtype=torch.float32, device=device)
    k = min(k, E.shape[0])
    chunk = max(1, min(chunk, 2**28 // E.shape[0]))  # score block <= 1 GB of fp32 (1M docs: 268 queries)
    out = []
    for s in range(0, Q.shape[0], chunk):
        q = torch.as_tensor(Q[s : s + chunk], dtype=torch.float32, device=device)
        scores = q @ E_t.T
        if exclude is not None:
            ex = torch.as_tensor(exclude[s : s + chunk], device=device)
            rows = torch.nonzero(ex >= 0).squeeze(1)
            scores[rows, ex[rows]] = -torch.inf
        out.append(scores.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)
