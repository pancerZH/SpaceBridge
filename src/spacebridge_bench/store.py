"""Cached embeddings read lazily from disk, so an index larger than RAM or VRAM is never loaded whole."""

import gc
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import torch

RESIDENT_SHARE = 0.6  # an index stays on the GPU between searches if it fits in this share of VRAM
MIN_SHARD_ROWS = 65_536
CPU_SHARD_ROWS = 2_000_000
_RESIDENT: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def release_gpu() -> None:
    """Drops every resident index, e.g. before loading a model."""
    _RESIDENT.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class EmbeddingStore:
    """A float16 embedding file as L2-normalized float32 rows, the same values ``Encoder.encode`` returns.

    Converting and normalizing on the GPU takes 0.6 s per 1M x 1024-d rows, against 3.2 s on the CPU.
    """

    def __init__(self, path: str | Path, truncate_dim: Optional[int] = None):
        self.path = Path(path)
        self._mm = np.load(self.path, mmap_mode="r")
        self.dim = min(truncate_dim or self._mm.shape[1], self._mm.shape[1])
        self._mean: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._mm.shape[0], self.dim

    def __len__(self) -> int:
        return self._mm.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        x = np.asarray(self._mm[rows], dtype=np.float32)[..., : self.dim]
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    def prefix(self, n: int) -> "EmbeddingStore":
        """A view of the first ``n`` rows, for indexes whose prefixes are fair samples (``DatasetSpec.nested_docs``)."""
        if n == len(self):
            return self
        view = EmbeddingStore.__new__(EmbeddingStore)
        view.path, view.dim, view._mean, view._mm = self.path, self.dim, None, self._mm[:n]
        return view

    def mean(self, device: str = "cuda") -> np.ndarray:
        if self._mean is None:
            device = device if torch.cuda.is_available() else "cpu"
            total = torch.zeros(self.dim, dtype=torch.float64, device=device)
            for _, shard in self.shards(device):
                for s in range(0, shard.shape[0], MIN_SHARD_ROWS):  # a double copy of a whole shard may not fit
                    total += shard[s : s + MIN_SHARD_ROWS].double().sum(0)
                del shard
            self._mean = (total / len(self)).float().cpu().numpy()
        return self._mean

    def _rows(self, a: int, b: int, device: str) -> torch.Tensor:
        half = torch.from_numpy(np.array(self._mm[a:b, : self.dim]))
        rows = half.to(device, non_blocking=True).float()
        return rows.div_(rows.norm(dim=-1, keepdim=True).clamp_min_(1e-12))  # in place: a second copy may not fit in VRAM

    def _fits_on_gpu(self, device: str) -> bool:
        need = len(self) * self.dim * 4
        limit = RESIDENT_SHARE * torch.cuda.mem_get_info(device)[1]
        if need > limit:
            return False
        while _RESIDENT and sum(t.numel() * 4 for t in _RESIDENT.values()) + need > limit:
            _RESIDENT.popitem(last=False)
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(device)[0] > 1.6 * need  # float16 and float32 copies coexist while converting

    def _shard_rows(self, device: str) -> int:
        if not device.startswith("cuda"):
            return CPU_SHARD_ROWS
        release_gpu()
        return max(MIN_SHARD_ROWS, int(0.5 * torch.cuda.mem_get_info(device)[0] / (self.dim * 4)))

    def shards(self, device: str, boundaries: Sequence[int] = ()) -> Iterator[tuple[int, torch.Tensor]]:
        """Yields (first row, rows on ``device``). Shards end at ``boundaries``; without any, a whole index that fits
        on the GPU stays there for the next search, and a larger one is streamed."""
        key = (self.path, self.dim, len(self), device)
        if not boundaries and device.startswith("cuda"):
            if key in _RESIDENT:
                _RESIDENT.move_to_end(key)
                yield 0, _RESIDENT[key]
                return
            if self._fits_on_gpu(device):
                _RESIDENT[key] = self._rows(0, len(self), device)
                yield 0, _RESIDENT[key]
                return
        step = self._shard_rows(device)
        cuts = sorted({*range(0, len(self), step), *(b for b in boundaries if 0 < b < len(self)), len(self)})
        for a, b in zip(cuts, cuts[1:]):
            yield a, self._rows(a, b, device)
