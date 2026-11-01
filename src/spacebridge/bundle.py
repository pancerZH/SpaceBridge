"""Calibration bundle produced at index-freeze time (design §4.1, §6)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ARRAYS = ("anchor_ids", "X_doc_raw", "X_query_raw", "train_idx", "val_idx", "distractor_ids")


@dataclass
class CalibrationBundle:
    src_lang: str                  # e.g., "en" or "zh"
    anchor_ids: np.ndarray         # (K,) index row ids of the anchor docs
    anchor_doc_texts: list[str]
    anchor_query_texts: list[str]
    X_doc_raw: np.ndarray          # (K, d_s) anchor doc vectors exactly as stored in the index
    X_query_raw: np.ndarray        # (K, d_s) source query embeddings of the proxies (native query prompt)
    train_idx: np.ndarray          # (K_train,) positions into the anchor arrays, fixed at freeze time
    val_idx: np.ndarray            # (K_val,)
    distractor_ids: np.ndarray     # (<=50K,) index row ids used for rho_val
    mu_x_doc: Optional[np.ndarray] = None  # (d_s,) diagnostics only; the index is never re-centered
    index_size: Optional[int] = None       # rows in the index: the quality report warns outside its calibrated range

    @property
    def K(self) -> int:
        return len(self.anchor_ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        if self.mu_x_doc is not None:
            arrays["mu_x_doc"] = self.mu_x_doc
        np.savez(path / "arrays.npz", **arrays)
        meta = {
            "src_lang": self.src_lang,
            "anchor_doc_texts": self.anchor_doc_texts,
            "anchor_query_texts": self.anchor_query_texts,
            "index_size": self.index_size,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationBundle":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        with np.load(path / "arrays.npz") as z:
            arrays = {name: z[name] for name in _ARRAYS}
            mu_x_doc = z["mu_x_doc"] if "mu_x_doc" in z else None
        return cls(mu_x_doc=mu_x_doc, **meta, **arrays)
