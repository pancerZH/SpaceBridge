"""Exact inner-product search (no ANN, so Phase-0 numbers carry no approximation noise)."""

import numpy as np
import torch


def search(Q: np.ndarray, E: np.ndarray, k: int = 100, device: str = "cuda", chunk: int = 1024) -> np.ndarray:
    """Returns (n_queries, k) row ids of E ordered by descending inner product."""
    device = device if torch.cuda.is_available() else "cpu"
    E_t = torch.as_tensor(E, dtype=torch.float32, device=device)
    k = min(k, E.shape[0])
    out = []
    for s in range(0, Q.shape[0], chunk):
        q = torch.as_tensor(Q[s : s + chunk], dtype=torch.float32, device=device)
        out.append((q @ E_t.T).topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)
