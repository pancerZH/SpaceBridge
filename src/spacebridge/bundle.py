"""Calibration bundle produced at index-freeze time (design §4.1, §6)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import storage

_ARRAYS = ("anchor_ids", "X_doc_raw", "train_idx", "val_idx", "distractor_ids")
_OPTIONAL_ARRAYS = ("X_query_raw", "mu_x_doc")
_META = ("src_lang", "anchor_doc_texts", "anchor_query_texts", "anchor_doc_titles", "tgt_anchor_texts", "index_size")


@dataclass
class CalibrationBundle:
    src_lang: str                  # e.g., "en" or "zh"
    anchor_ids: np.ndarray         # (K,) index row ids of the anchor docs; -1 for a static sentence bank
    anchor_doc_texts: list[str]
    anchor_query_texts: list[str]
    X_doc_raw: np.ndarray          # (K, d_s) anchor doc vectors exactly as stored in the index
    X_query_raw: Optional[np.ndarray]  # (K, d_s) source query embeddings of the proxies; None without the source model
    train_idx: np.ndarray          # (K_train,) positions into the anchor arrays, fixed at freeze time
    val_idx: np.ndarray            # (K_val,)
    distractor_ids: np.ndarray     # (<=50K,) index row ids used for rho_val
    mu_x_doc: Optional[np.ndarray] = None  # (d_s,) diagnostics only; the index is never re-centered
    index_size: Optional[int] = None       # rows in the index: the quality report warns outside its calibrated range
    anchor_doc_titles: Optional[list[str]] = None  # empty or absent: the anchor docs have no titles
    tgt_anchor_texts: Optional[list[str]] = None   # static bank: the parallel sentences the target model encodes
    # {lang: {"src": sentences, "lang": sentences, "X_query": (n, d_s), "X_doc": (n, d_s)}}: held-out parallel sentences
    # with the source model's embeddings, for per-language validation at mount.
    language_validation: dict[str, dict] = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.anchor_ids)

    def save(self, path: str | Path) -> None:
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        arrays.update({name: getattr(self, name) for name in _OPTIONAL_ARRAYS if getattr(self, name) is not None})
        for lang, val in self.language_validation.items():
            arrays[f"lang_val/{lang}/X_query"], arrays[f"lang_val/{lang}/X_doc"] = val["X_query"], val["X_doc"]
        meta = {name: getattr(self, name) for name in _META}
        meta["language_validation"] = {lang: {"src": v["src"], "lang": v["lang"]} for lang, v in self.language_validation.items()}
        storage.save_dir(path, "calibration-bundle", arrays, meta)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationBundle":
        arrays, meta = storage.load_dir(path, "calibration-bundle")
        language_validation = {
            lang: {**v, "X_query": arrays[f"lang_val/{lang}/X_query"], "X_doc": arrays[f"lang_val/{lang}/X_doc"]}
            for lang, v in meta.pop("language_validation").items()
        }
        fixed = {name: arrays[name] for name in _ARRAYS}
        optional = {name: arrays.get(name) for name in _OPTIONAL_ARRAYS}
        return cls(language_validation=language_validation, **meta, **fixed, **optional)
