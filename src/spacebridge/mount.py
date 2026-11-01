"""The two calls a deployment makes (design §6): ``freeze`` once with the source model, ``mount`` with each new query model.

Every Phase 0 default lives here, so callers do not repeat the protocol: the anchor budget and selection for the index
size, query proxies, the fixed split, 128-token anchor docs, doc pairs by the query model's size class, language shifts.
"""

import time
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from .anchors import default_anchor_count, make_query_proxies, make_split, random_anchors, resolve_anchor_method, sample_distractors, select_anchors
from .bridge import DOC_PAIR_TOKENS, SpaceBridge
from .bundle import CalibrationBundle
from .language import load_flores, load_shift_bank
from .refine import RefineConfig


class QueryEncoder(Protocol):
    """An embedding model with its native prompts. Returns L2-normalized float32 rows.

    ``size_class`` is ``"large"`` for ~4B models and above, whose doc-pair pass is skipped; anything else keeps it.
    """

    size_class: str

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_docs(self, titles: list[str], texts: list[str], max_tokens: Optional[int] = None) -> np.ndarray: ...


class IndexReader(Protocol):
    """Read access to a frozen index: stored vectors and document text, by row."""

    def __len__(self) -> int: ...

    def vectors(self, ids: np.ndarray) -> np.ndarray: ...

    def texts(self, ids: np.ndarray) -> list[str]: ...

    def titles(self, ids: np.ndarray) -> list[str]: ...


class Index:
    """An ``IndexReader`` over an array-like of vectors (a numpy array, a memory map, ``EmbeddingStore``) and text lists."""

    def __init__(self, vectors, texts: Sequence[str], titles: Optional[Sequence[str]] = None):
        self._vectors, self._texts, self._titles = vectors, texts, titles

    def __len__(self) -> int:
        return len(self._vectors)

    def vectors(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self._vectors[np.asarray(ids)], dtype=np.float32)

    def texts(self, ids: np.ndarray) -> list[str]:
        return [self._texts[i] for i in ids]

    def titles(self, ids: np.ndarray) -> list[str]:
        return [self._titles[i] if self._titles is not None else "" for i in ids]


def freeze(
    index: IndexReader,
    src_encoder: Optional[QueryEncoder],
    src_lang: str = "en",
    seed: int = 0,
    K: Optional[int] = None,
    anchor_select: str = "auto",
    exclude: Optional[np.ndarray] = None,
    anchor_ids: Optional[np.ndarray] = None,
    make_proxies: Callable[[list[str], list[str]], list[str]] = make_query_proxies,
    parallel_bank: Optional[tuple[list[str], list[str]]] = None,
    validate_langs: Sequence[str] = (),
    device: str = "cuda",
) -> CalibrationBundle:
    """Builds the calibration bundle of an index, reading only its anchors and distractors (and, for k-means, its vectors).

    Args:
        src_encoder: the model the index was built with. ``None`` leaves ``X_query_raw`` empty (a source-model-free fit).
        K: anchor budget; default ``default_anchor_count(len(index))``. ``anchor_select``: ``auto`` follows the index size.
        exclude: rows that may not be anchors, e.g. rows known to be test queries.
        anchor_ids: use these rows as anchors instead of selecting (studies, cached k-means selections).
        parallel_bank: (source-language, target-language) parallel sentences replacing corpus anchors (Regime L2, for a
            query model that cannot read the corpus language). No anchor is then an index row.
        validate_langs: query languages to prepare per-language validation for (FLORES ``devtest``, English index).
    """
    N = len(index)
    allowed = np.arange(N) if exclude is None else np.setdiff1d(np.arange(N), exclude)
    tgt_anchor_texts, titles = None, None
    if parallel_bank is not None:
        src_side, tgt_side = parallel_bank
        K = min(K or len(src_side), len(src_side))
        pick = np.sort(np.random.default_rng(seed).permutation(len(src_side))[:K])
        doc_texts = proxies = [src_side[i] for i in pick]
        tgt_anchor_texts = [tgt_side[i] for i in pick]
        anchor_ids = np.full(K, -1, dtype=np.int64)
        X_doc = src_encoder.encode_docs([""] * K, doc_texts)
    else:
        if anchor_ids is None:
            K = min(K or default_anchor_count(N), len(allowed))
            if resolve_anchor_method(len(allowed), anchor_select) == "random":
                anchor_ids = allowed[random_anchors(len(allowed), K, seed)]
            else:
                anchor_ids = allowed[select_anchors(index.vectors(allowed), K, seed, device, method="kmeans")]
        K = len(anchor_ids)
        titles, doc_texts = index.titles(anchor_ids), index.texts(anchor_ids)
        proxies = make_proxies(titles, doc_texts)
        X_doc = index.vectors(anchor_ids)
    train_idx, val_idx = make_split(K, seed=seed)
    distractor_ids = sample_distractors(N, anchor_ids[anchor_ids >= 0], seed=seed)
    language_validation = {}
    for lang in validate_langs:
        devtest = load_flores(lang, belebele_filter=True)["devtest"]
        language_validation[lang] = {
            "src": devtest[src_lang], "lang": devtest[lang], "X_query": src_encoder.encode_queries(devtest[src_lang]),
            "X_doc": src_encoder.encode_docs([""] * len(devtest[src_lang]), devtest[src_lang]),
        }
    return CalibrationBundle(
        src_lang=src_lang,
        anchor_ids=anchor_ids,
        anchor_doc_texts=doc_texts,
        anchor_query_texts=proxies,
        X_doc_raw=X_doc,
        X_query_raw=None if src_encoder is None else src_encoder.encode_queries(proxies),
        train_idx=train_idx,
        val_idx=val_idx,
        distractor_ids=distractor_ids,
        # Diagnostics only, so a uniform sample will do; the index stays unread.
        mu_x_doc=index.vectors(distractor_ids).mean(0),
        index_size=N,
        anchor_doc_titles=titles,
        tgt_anchor_texts=tgt_anchor_texts,
        language_validation=language_validation,
    )


def mount(
    bundle: CalibrationBundle,
    index: IndexReader,
    tgt_encoder: QueryEncoder,
    query_langs: Sequence[str] = (),
    device: str = "cuda",
    doc_pair_tokens: Optional[int] = DOC_PAIR_TOKENS,
    doc_pairs: Optional[bool] = None,
    refine_config: Optional[RefineConfig] = None,
    shift_banks: Optional[Mapping[str, tuple[list[str], list[str]]]] = None,
) -> SpaceBridge:
    """Fits the query map of a new query model onto a frozen index; the quality report is in ``bridge.quality_report()``.

    Args:
        query_langs: languages queries will arrive in, other than the bundle's. Each gets a language shift and, if the
            bundle holds its validation sentences, a per-language score.
        doc_pair_tokens: length anchor docs are encoded at for doc pairs; ``None`` or 0 for the model's full length.
        doc_pairs: override the size-class rule (ablation).
        shift_banks: ``{lang: (source-language sentences, lang sentences)}``; default ``load_shift_bank`` (FLORES when the
            gated repo is accessible).
    """
    timing = {}

    def timed(name: str, call: Callable):
        t0 = time.perf_counter()
        out = call()
        timing[f"{name}_ms"] = 1e3 * (time.perf_counter() - t0)
        return out

    # Doc pairs are all a source-model-free fit has, whatever the model's size.
    use_doc_pairs = bundle.X_query_raw is None or tgt_encoder.size_class != "large" if doc_pairs is None else doc_pairs
    static = bundle.tgt_anchor_texts is not None
    Y_query = timed("encode_proxies", lambda: tgt_encoder.encode_queries(bundle.tgt_anchor_texts if static else bundle.anchor_query_texts))
    Y_doc = None
    if use_doc_pairs:
        titles = [""] * bundle.K if static or bundle.anchor_doc_titles is None else bundle.anchor_doc_titles
        texts = bundle.tgt_anchor_texts if static else bundle.anchor_doc_texts
        Y_doc = timed("encode_anchor_docs", lambda: tgt_encoder.encode_docs(titles, texts, max_tokens=doc_pair_tokens or None))
    shifts, validation = {}, {}
    for lang in query_langs:
        src_side, lang_side = (shift_banks or {}).get(lang) or load_shift_bank(lang)[:2]
        shifts[lang] = timed(f"encode_shift_{lang}", lambda: (tgt_encoder.encode_queries(src_side), tgt_encoder.encode_queries(lang_side)))
        val = bundle.language_validation.get(lang)
        if val is not None:
            validation[lang] = (tgt_encoder.encode_queries(val["lang"]), tgt_encoder.encode_queries(val["src"]), val["X_query"], val["X_doc"])
    unload = getattr(tgt_encoder, "unload", None)
    if unload is not None:
        unload()  # leaves the GPU to the solver
    X_distractor = timed("read_distractors", lambda: index.vectors(bundle.distractor_ids))

    bridge = SpaceBridge(device=device, doc_pairs=use_doc_pairs, refine_config=refine_config)
    if bundle.X_query_raw is None:
        if Y_doc is None:
            raise ValueError("a source-model-free mount needs doc pairs")
        timed("fit", lambda: bridge.fit_source_free(bundle, Y_query, Y_doc, X_distractor))
    else:
        timed("fit", lambda: bridge.fit(bundle, Y_query, X_distractor=X_distractor, Y_doc_raw=Y_doc))
    for lang, (Y_src, Y_lang) in shifts.items():
        bridge.register_query_language(lang, Y_src, Y_lang)
        if lang in validation:
            bridge.validate_query_language(lang, *validation[lang])
    bridge.metrics["mount"] = timing
    return bridge
