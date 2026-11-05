"""Benchmark runner: tier/pair x ablation rows -> JSONL (design §7.2–§7.4).

Example:
    uv run spacebridge-bench --dataset scifact --tier 1 3
    uv run spacebridge-bench --dataset fiqa --pair bge-large-en:qwen3-0.6b --rows 0a 0b 6 8 --K 1024 2048
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from spacebridge import (
    DOC_PAIR_TOKENS, Index, LexicalIndex, RefineConfig, SpaceBridge, extract_query_proxy, freeze, make_query_proxies, merge_pool, mount, random_anchors,
    default_anchor_count, resolve_anchor_method, select_anchors,
)
from spacebridge.lexical import LEXICAL_SHARE

from . import durable, provenance
from .datasets import DATASETS, FLORES_SPLITS, SHIFT_BANKS, RetrievalDataset, flores_available, load_beir, load_flores, load_shift_bank
from .encode import Encoder, MountEncoder
from .metrics import bootstrap_retention_ci, per_query, retention, spearman
from .models import MODELS
from .rerank import RERANKERS, Reranker, rerank, rerank_pools, topk_jaccard
from .search import cached_search, search
from .store import EmbeddingStore

ROWS = {
    "0a": "Oracle full re-index (M_tgt docs + M_tgt queries)",
    "0b": "M_src native (ceiling; parallel index-language queries for cross-lingual sets)",
    "0c": "M_src encoding the test queries directly (cross-lingual sets: no bridge)",
    "1": "Identity sanity (M_tgt := M_src, full default pipeline incl. doc<->doc pairs)",
    "2": "Uncentered full-SVD Procrustes, doc anchors",
    "3s": "+ centering + mean restore + scaled truncated Stiefel, doc anchors",
    "3r": "+ centering + mean restore + ridge (lambda sweep), doc anchors",
    "3b": "Row 3 (best Stage-1 op) without mean restore",
    "6": "+ query-proxy pairs (best Stage-1 op)",
    "6b": "Row 6 without mean restore",
    "6n": "Row 6 without residual norm matching",
    "7": "Row 8 + language-cone shift correction (Tatoeba stand-in for FLORES; cross-lingual sets)",
    "7b": "Row 8b + language-cone shift correction (default engine on cross-lingual sets)",
    "8": "+ Stage-2 InfoNCE + KL refinement",
    "8s": "Row 8 without mean restore",
    "8n": "Row 8 without residual norm matching",
    "8b": "Row 8 + doc<->doc pairs (anchor docs encoded by M_tgt; default engine)",
    "6f": "Source-model-free: doc<->doc pairs only, Stage 1 (no M_src at any point)",
    "8f": "Source-model-free: doc<->doc pairs, Stage 1 + InfoNCE-only Stage 2 (no M_src at any point)",
    "9t": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-10 + cross-encoder rerank",
    "9u": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-20 + cross-encoder rerank",
    "9n": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-30 + cross-encoder rerank",
    "9": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-60 + cross-encoder rerank",
    "9w": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-100 + cross-encoder rerank",
    "9p": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-150 + cross-encoder rerank",
    "9q": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-200 + cross-encoder rerank",
    "9x": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-300 + cross-encoder rerank",
    "9y": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-500 + cross-encoder rerank",
    "9z": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-1,000 + cross-encoder rerank",
    "9h": "Row 8b + BM25, a quarter of a budget of 60, + cross-encoder rerank (same-language sets only)",
    "9hw": "Row 8b + BM25, a quarter of a budget of 100, + cross-encoder rerank (same-language sets only)",
    "9hx": "Row 8b + BM25, a quarter of a budget of 300, + cross-encoder rerank (same-language sets only)",
}

# row -> (anchor embeddings used as Y, fit mode, pair type, restore source mean)
BRIDGE_ROWS = {
    "1": ("src_query", "auto", "query", True),
    "2": ("tgt_doc", "naive", "doc", True),
    "3s": ("tgt_doc", "stiefel", "doc", True),
    "3r": ("tgt_doc", "ridge", "doc", True),
    "3b": ("tgt_doc", "stage1", "doc", False),
    "6": ("tgt_query", "stage1", "query", True),
    "6b": ("tgt_query", "stage1", "query", False),
    "6n": ("tgt_query", "stage1", "query", True),
    "7": ("tgt_query", "auto", "query", True),  # + register_query_language
    "7b": ("tgt_query", "auto", "query", True),  # + Y_doc_raw + register_query_language
    "8": ("tgt_query", "auto", "query", True),
    "8s": ("tgt_query", "auto", "query", False),
    "8n": ("tgt_query", "auto", "query", True),
    "8b": ("tgt_query", "auto", "query", True),  # + Y_doc_raw
    "6f": ("tgt_query", "source_free", "doc", True),
    "8f": ("tgt_query", "source_free", "doc", True),
}

TIERS = {
    "1": [("bge-base-en", "bge-large-en"), ("bge-large-en", "bge-base-en")],
    "3": [
        ("bge-large-en", "qwen3-0.6b"), ("qwen3-0.6b", "bge-large-en"),
        ("bge-large-en", "harrier-0.6b"), ("harrier-0.6b", "bge-large-en"),
    ],
    "2": [("bge-large-zh", "me5-large"), ("me5-large", "bge-large-zh")],
    # 4a: index <- queries; ("qwen3-4b", "minilm") is up-projection (384d queries into a 2560d index).
    "4a": [("qwen3-4b", "minilm"), ("minilm", "qwen3-4b")],
    "4b": [("bge-large-en", f"{m}-{d}") for m in ("embeddinggemma-2", "qwen3-0.6b") for d in (128, 256, 512)]
    + [("bge-large-en", "embeddinggemma-2"), ("bge-large-en", "qwen3-0.6b-768")],
    "5a": [("bge-large-en", q) for q in ("qwen3-0.6b", "harrier-0.6b", "me5-large", "embeddinggemma-2")],
    # 5b (Regime L2): English queries from an English-only model into a Chinese-only index (--anchor-bank flores).
    "5b": [("bge-large-zh", "bge-large-en")],
}

SEARCH_K = 1_000  # as fast as top-100, and reports recall at rerank depth
VAL_K = 10  # rho_val's top-k (spacebridge.bridge.VAL_K)
ANCHOR_CACHE = Path("data/cache/anchors")
_warm = False
LEXICAL_ROWS = {"9h": 60, "9hw": 100, "9hx": 300}
RERANK_DEPTH = {"9t": 10, "9u": 20, "9n": 30, "9": 60, "9w": 100, "9p": 150, "9q": 200, "9x": 300, "9y": 500, "9z": 1_000, **LEXICAL_ROWS}
LEXICAL_CACHE = Path("data/cache/lexical")
_LEXICAL: dict[Path, LexicalIndex] = {}  # one copy per process: 2.2 GB at 8.84M, and every seed and pair reads it
CI_METRICS = ("recall@30", "ndcg@5", "ndcg@10")
PER_QUERY_METRICS = ("ndcg@5", "recall@5", "recall@30", "recall@100")
PROXY_STYLES = ("title", "title-any", "lead", "short")


def make_proxies(titles: list[str], texts: list[str], style: str = "title") -> list[str]:
    """``title`` (engine default since M3): the title alone when it has >= 4 tokens and no other anchor
    shares it, else title + lead sentence (32-64 tokens). ``title-any``: the title alone even when shared
    (the first M3 version). ``lead``: title + lead sentence for every document (the M2 default).
    ``short``: like ``title-any``, but the fallback is cut to 16 tokens."""
    if style == "title":
        return make_query_proxies(titles, texts)
    kwargs = {"lead": {"title_min_tokens": None}, "title-any": {}, "short": {"min_tokens": 4, "max_tokens": 16}}[style]
    return [extract_query_proxy(t, x, **kwargs) for t, x in zip(titles, texts)]


def cached_anchors(
    ds: RetrievalDataset, src_key: str, E_src: np.ndarray | EmbeddingStore, allowed: np.ndarray, K: int, seed: int, method: str = "auto",
) -> np.ndarray:
    """Anchor doc rows; k-means selections are cached on disk so every pair and row of a dataset reuses one."""
    if resolve_anchor_method(len(allowed), method) == "random":
        return allowed[random_anchors(len(allowed), K, seed)]
    path = ANCHOR_CACHE / f"{ds.name}_{src_key}_N{ds.N}_n{len(allowed)}_K{K}_s{seed}_tkm.npy"  # _tkm: torch k-means
    if path.exists():
        return np.load(path)
    ids = allowed[select_anchors(E_src[allowed], K, seed, method="kmeans")]
    path.parent.mkdir(parents=True, exist_ok=True)
    durable.save_npy(path, ids)
    return ids


def anchor_candidates(ds: RetrievalDataset) -> np.ndarray:
    """Rows a benchmark's anchors may come from (design §7.1 rule 3): corpus rows that are test queries (ArguAna) never become
    anchors. On a nested corpus neither do positives: at small prefixes they would inflate results."""
    allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
    if DATASETS[ds.name.split("@")[0]].nested_docs is not None:
        allowed = np.setdiff1d(allowed, [d for rel in ds.qrels for d in rel])
    return allowed


def cached_lexical(ds: RetrievalDataset, index: Index) -> LexicalIndex:
    """BM25 over a corpus's rows, kept on disk: it depends on nothing but those rows, in that order."""
    h = hashlib.blake2b(digest_size=12)
    for doc_id in ds.doc_ids:
        h.update(str(doc_id).encode() + b"\x00")
    path = LEXICAL_CACHE / f"{ds.name}_{h.hexdigest()}"
    if path not in _LEXICAL:
        if (path / "manifest.json").exists():
            _LEXICAL[path] = LexicalIndex.load(path)
        else:
            t0 = time.perf_counter()
            _LEXICAL[path] = LexicalIndex.build(index)
            _LEXICAL[path].save(path)
            durable.progress(f"lexical {ds.name}: built and saved in {time.perf_counter() - t0:.0f} s, {_LEXICAL[path].X.nnz:,} postings")
    return _LEXICAL[path]


def run_pair(
    ds: RetrievalDataset, src_key: str, tgt_key: str, K: int, seed: int, rows: list[str], device: str,
    rerankers: tuple[str, ...] = (), proxy: str = "title", stage2_batch: Optional[int] = None,
    doc_pair_tokens: int = DOC_PAIR_TOKENS, time_mount: bool = False, stage2_tau_scale: Optional[float] = None,
    stage2_curves: bool = False, stage2_rule: Optional[str] = None, shift_bank: str = "auto", anchor_bank: str = "corpus",
    anchor_select: str = "auto", full_index_diagnostics: bool = False, rerank_row: Optional[str] = None,
    no_bridge: bool = False, per_query_path: Optional[Path] = None,
) -> list[dict]:
    src, tgt = MODELS[src_key], MODELS[tgt_key]
    K = min(K or default_anchor_count(ds.N), ds.N - int((ds.self_rows >= 0).sum()))
    static = anchor_bank != "corpus"
    parallel_bank = None
    if static:  # Regime L2 (design §4.4): FLORES dev + devtest pairs replace corpus anchors; none is an index row
        fl = load_flores(ds.lang if ds.lang != "en" else ds.query_lang, belebele_filter=anchor_bank == "flores")
        parallel_bank = tuple([t for sp in FLORES_SPLITS for t in fl[sp][l]] for l in (ds.lang, ds.query_lang))

    enc = Encoder(src, device)
    E_src = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q_src = enc.queries(ds.index_lang_queries, ds.task)
    Q_src_direct = enc.queries(ds.queries, ds.task) if "0c" in rows else None
    # Per-language validation (design §5): held-out FLORES devtest pairs, Belebele-filtered (the canonical set) and
    # unfiltered (4x the sentences; nothing here evaluates on Belebele). M_src encodes the English side as the
    # teacher (query prompt) and as counterparts mixed into the validation pool (doc prompt).
    lang_val = {}
    if ds.query_lang and {"7", "7b"} & set(rows) and flores_available(ds.query_lang):
        for tag, filt in (("unfiltered", False), ("filtered", True)):
            dt = load_flores(ds.query_lang, belebele_filter=filt)["devtest"]
            lang_val[tag] = {"en": dt["en"], "lang": dt[ds.query_lang],
                             "X_query": enc.queries(dt["en"], ds.task), "X_doc": enc.docs([""] * len(dt["en"]), dt["en"])}
    enc.unload()
    allowed = anchor_candidates(ds)
    index = ds.index_over(E_src)
    bundle = freeze(
        index, MountEncoder(src, ds.task, device), ds.lang, seed, K,
        anchor_ids=None if static else cached_anchors(ds, src_key, E_src, allowed, K, seed, anchor_select),
        make_proxies=lambda titles, texts: make_proxies(titles, texts, proxy), parallel_bank=parallel_bank, device=device,
    )
    K = bundle.K
    anchor_ids, proxies = bundle.anchor_ids, bundle.anchor_query_texts
    tgt_proxies = bundle.tgt_anchor_texts if static else proxies

    enc = Encoder(tgt, device)
    E_tgt = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q_tgt = enc.queries(ds.queries, ds.task)
    # Mount timing re-encodes the anchors uncached (~70 s per run for a 4B model), so it is opt-in.
    mount_timing = None
    rerank_base = rerank_row or ("7b" if ds.query_lang else "8b")  # Row 9 reranks the default engine's first stage
    needs = set(rows) | ({rerank_base} if set(rows) & set(RERANK_DEPTH) else set())
    if time_mount:
        _, mount_timing = enc.timed_encode([tgt.format_query(p, ds.task) for p in tgt_proxies])
        # Corpus token estimate: the mount budget is relative to the encoding work of re-indexing (design §2.1).
        sample = np.random.default_rng(0).choice(ds.N, min(3_000, ds.N), replace=False)
        doc_tokens = enc.model.tokenizer([tgt.format_doc(ds.titles[i], ds.texts[i]) for i in sample],
                                         truncation=True, max_length=tgt.max_seq_length)["input_ids"]
        mount_timing["corpus_tokens_est"] = sum(map(len, doc_tokens)) / len(sample) * ds.N
        mount_timing["projected_encode_ms_K2048"] = mount_timing["encode_ms"] * 2048 / K
        if {"8b", "7b"} & needs:
            enc.unload()
            enc = Encoder(replace(tgt, max_seq_length=min(doc_pair_tokens or tgt.max_seq_length, tgt.max_seq_length)), device)
            doc_inputs = ([tgt.format_doc("", t) for t in tgt_proxies] if static
                          else [tgt.format_doc(ds.titles[i], ds.texts[i]) for i in anchor_ids])
            _, doc_mount = enc.timed_encode(doc_inputs)
            mount_timing.update({f"doc_{k}": v for k, v in doc_mount.items()})
    enc.unload()
    tgt_mount = MountEncoder(tgt, ds.task, device)
    Y_query = tgt_mount.encode_queries(tgt_proxies)
    if {"7", "7b"} & needs and ds.query_lang:
        bank_src, bank_q, bank_name = load_shift_bank(ds.query_lang, bank=shift_bank)
        Y_bank_src, Y_bank_q = tgt_mount.encode_queries(bank_src), tgt_mount.encode_queries(bank_q)
        for v in lang_val.values():
            v["Y_lang"], v["Y_src"] = tgt_mount.encode_queries(v["lang"]), tgt_mount.encode_queries(v["en"])
    tgt_mount.unload()

    search_k = max([SEARCH_K] + [RERANK_DEPTH[r] for r in rows if r in RERANK_DEPTH])
    native_ranked = cached_search(Q_src, E_src, search_k, device, exclude=ds.self_rows)
    oracle_ranked = cached_search(Q_tgt, E_tgt, search_k, device, exclude=ds.self_rows)
    native_pq, oracle_pq = per_query(native_ranked, ds.qrels), per_query(oracle_ranked, ds.qrels)
    anchor_Y = {"src_query": bundle.X_query_raw, "tgt_query": Y_query, "tgt_doc": None if static else E_tgt[anchor_ids]}
    X_distractor = index.vectors(bundle.distractor_ids)
    # Diagnostic (opt-in, 2 of a cell's searches): rho_val over the whole index. The engine's pool holds the anchors
    # + <= 50K distractors, so on a larger index it cannot see the extra crowding (msmarco-1m, trec-covid); this scores
    # the same validation proxies against the teacher's top 10 over all N rows.
    val_teacher_full = search(bundle.X_query_raw[bundle.val_idx], E_src, VAL_K, device) if full_index_diagnostics else None

    global _warm
    if not _warm:  # one untimed fit, so the first row's timing excludes CUDA / cuSOLVER warmup
        SpaceBridge(device=device, doc_pairs=False).fit(bundle, Y_query, X_distractor=X_distractor)
        _warm = True
    bridged: dict[str, tuple[np.ndarray, dict, float, SpaceBridge]] = {}

    def fit_row(row: str) -> tuple[np.ndarray, dict, float, SpaceBridge]:
        if row not in bridged:
            y_kind, mode, pairs, restore = BRIDGE_ROWS[row]
            config = RefineConfig(
                **({"batch_candidates": stage2_batch or None} if stage2_batch is not None else {}),
                **({"tau_scale": stage2_tau_scale, "tau_scale_below_sd": float("inf")} if stage2_tau_scale else {}),
                **({"stop_rule": stage2_rule} if stage2_rule else {}),
                keep_checkpoints=stage2_curves,
            )
            if row in ("6f", "8f"):
                bridge = SpaceBridge(device=device, doc_pairs=True, refine_config=config)
                free = replace(bundle, X_query_raw=None)
                Y_doc = tgt_mount.encode_docs(free.anchor_doc_titles, free.anchor_doc_texts, max_tokens=doc_pair_tokens or None)
                tgt_mount.unload()
                fit = bridge.fit_source_free(free, Y_query, Y_doc, X_distractor, stage2=row == "8f")
            elif row in ("8b", "7b"):  # the default engine, through the call that ships
                bridge = mount(
                    bundle, index, tgt_mount, [ds.query_lang] if row == "7b" else (), device, doc_pair_tokens or None,
                    doc_pairs=True, refine_config=config, shift_banks={ds.query_lang: (bank_src, bank_q)} if row == "7b" else None,
                )
                fit = bridge.metrics
            else:
                bridge = SpaceBridge(
                    device=device, restore_source_mean=restore, match_residual_norm=row not in ("6n", "8n"),
                    doc_pairs=row == "1", refine_config=config,
                )
                # The identity row pairs source docs with themselves.
                extra = {"Y_doc_raw": bundle.X_doc_raw} if row == "1" else {}
                fit = bridge.fit(bundle, anchor_Y[y_kind], mode=mode, pairs=pairs, X_distractor=X_distractor, **extra)
            Q = Q_src if row == "1" else Q_tgt
            lang = None
            if row in ("7", "7b"):
                lang = ds.query_lang
                if row == "7":
                    bridge.register_query_language(lang, Y_bank_src, Y_bank_q)
                fit["lang_val"] = {
                    tag: bridge.validate_query_language(lang, v["Y_lang"], v["Y_src"], v["X_query"], v["X_doc"])
                    for tag, v in lang_val.items()
                }
            ranked = search(bridge.transform_query(Q, query_lang=lang), E_src, search_k, device, exclude=ds.self_rows)
            if full_index_diagnostics:
                student = search(bridge.transform_query(anchor_Y[y_kind][bundle.val_idx]), E_src, VAL_K, device)
                fit["rho_val_full"] = float(np.mean([len(set(a) & set(b)) / VAL_K for a, b in zip(student, val_teacher_full)]))
                own = bundle.anchor_ids[bundle.val_idx]  # each validation proxy's own anchor doc (-1 for a static bank)
                if (own >= 0).all():  # exploratory: does the student find the proxy's own doc as often as the teacher does?
                    fit["owndoc_full"] = {
                        f"{who}_r{k}": float(np.mean((top[:, :k] == own[:, None]).any(1)))
                        for who, top in (("student", student), ("teacher", val_teacher_full)) for k in (1, 10)
                    }
            latency = _projection_latency_us(bridge, Q[0]) if time_mount else None
            bridged[row] = (ranked, fit, latency, bridge)
        return bridged[row]

    base = {"dataset": ds.name, "src": src_key, "tgt": tgt_key, "K": K, "seed": seed, "n_queries": len(ds.queries)}
    base["anchor_select"] = resolve_anchor_method(ds.N, anchor_select)
    base["proxy"] = proxy  # records without this field (before M3's title default) used lead proxies
    if static:
        base["anchor_bank"] = anchor_bank
    if stage2_batch is not None:
        base["stage2_batch"] = stage2_batch
    if stage2_tau_scale:
        base["stage2_tau_scale"] = stage2_tau_scale
    if stage2_rule:
        base["stage2_rule"] = stage2_rule
    if ds.query_lang and {"7", "7b"} & needs:
        base["shift_bank"], base["shift_bank_n"] = bank_name, len(bank_src)  # records without it used 256 Tatoeba pairs
    base["doc_pair_tokens"] = doc_pair_tokens  # records without this field (before the M3 budget decision) used 0
    records = []
    for row in (r for r in rows if r not in RERANK_DEPTH):
        if row in ("0c", "7", "7b") and not ds.query_lang:
            continue  # cross-lingual rows only
        rec = {**base, "row": row, "description": ROWS[row]}
        if row in ("0a", "0b"):
            pq = oracle_pq if row == "0a" else native_pq
        elif row == "0c":
            pq = per_query(search(Q_src_direct, E_src, SEARCH_K, device, exclude=ds.self_rows), ds.qrels)
        else:
            ranked, rec["fit"], latency, bridge = fit_row(row)
            if latency is not None:
                rec["projection_us"] = latency
            pq = per_query(ranked, ds.qrels)
            if row != "1" and mount_timing is not None:
                rec["mount"] = mount_timing
            if bridge.stage2_checkpoints:
                rec["stage2_real"] = _checkpoint_study(
                    bridge, Q_src if row == "1" else Q_tgt, ds.query_lang if row in ("7", "7b") else None,
                    lambda Q: per_query(search(Q, E_src, SEARCH_K, device, exclude=ds.self_rows), ds.qrels),
                    native_pq if row == "1" else None, native_pq, oracle_pq,
                )
            if bridge.quality is not None and row != "1":
                q = bridge.quality
                rec["quality"] = {"score": q.score, "band": q.band, "grade": q.grade, "rerank_pool": q.rerank_pool, "notes": q.notes,
                                  "by_language": q.by_language}
                if row not in ("6f", "8f"):
                    rec["confidence"] = _confidence_study(bridge, Q_tgt, ranked, native_ranked, pq, native_pq, oracle_pq)
        records.append(_score_record(rec, pq, native_pq, oracle_pq))

    # What the records can't hold, per query, for intervals over aligned queries: "<row>/<system>/<metric>".
    per_query_scores: dict[str, np.ndarray] = {}

    def keep_per_query(row: str, system: str, pq: dict) -> None:
        for m in PER_QUERY_METRICS:
            if m in pq:
                per_query_scores[f"{row}/{system}/{m}"] = np.asarray(pq[m], dtype=np.float32)

    # Same-language only: BM25 can't match a query against an index in another language.
    rerank_rows = [r for r in rows if r in RERANK_DEPTH and not (r in LEXICAL_ROWS and ds.query_lang)]
    if rerank_rows and rerankers:
        ranked8, fit8, _, _ = fit_row(rerank_base)
        docs = _DocTexts(ds.titles, ds.texts)
        lexical_top = None
        if set(rerank_rows) & set(LEXICAL_ROWS):
            lexical = cached_lexical(ds, index)
            deepest = max(LEXICAL_ROWS[r] for r in rerank_rows if r in LEXICAL_ROWS) // LEXICAL_SHARE
            lexical_top = [lexical.top(q, deepest if not no_bridge else deepest * LEXICAL_SHARE, exclude=s) for q, s in zip(ds.queries, ds.self_rows.tolist())]
        for rr_key in rerankers:
            rr = Reranker(RERANKERS[rr_key], device)
            for row in rerank_rows:
                depth = RERANK_DEPTH[row]
                out = {name: rerank(rr, r, depth, qs, docs, ds.task) for name, r, qs in
                       (("native", native_ranked, ds.index_lang_queries), ("oracle", oracle_ranked, ds.queries))}
                candidates = ranked8[0, :depth]
                extra = {}
                if row in LEXICAL_ROWS:
                    # The plain lists above are the ceiling every row uses; the same fix on native and re-index is the strict one.
                    share = depth // LEXICAL_SHARE
                    pools = {name: [merge_pool(a, b[:share], depth) for a, b in zip(r, lexical_top)]
                             for name, r in (("native", native_ranked), ("oracle", oracle_ranked), ("bridge", ranked8))}
                    fixed = {name: rerank_pools(rr, pools[name], qs, docs, ds.task) for name, qs in
                             (("native", ds.index_lang_queries), ("oracle", ds.queries), ("bridge", ds.queries))}
                    out["bridge"] = fixed["bridge"]
                    candidates = pools["bridge"][0]
                    extra = {"lexical_share": share, "pool_mean": float(np.mean([len(x) for x in pools["bridge"]])),
                             "lexical_in_dense": float(np.mean([len(set(a[:depth].tolist()) & set(b[:share].tolist())) for a, b in zip(ranked8, lexical_top)]))}
                else:
                    out["bridge"] = rerank(rr, ranked8, depth, ds.queries, docs, ds.task)
                pqs = {name: per_query(r, ds.qrels) for name, r in out.items()}
                rec = {**base, "row": row, "description": ROWS[row], "reranker": rr_key, "depth": depth, "fit": fit8,
                       "first_stage_row": rerank_base, **extra}
                rec["baselines"] = {name: {m: float(v.mean()) for m, v in pqs[name].items()} for name in ("native", "oracle")}
                rec["jaccard@5"] = {
                    "bridge_vs_oracle": float(topk_jaccard(out["bridge"], out["oracle"]).mean()),
                    "bridge_vs_native": float(topk_jaccard(out["bridge"], out["native"]).mean()),
                    "native_vs_oracle": float(topk_jaccard(out["native"], out["oracle"]).mean()),
                }
                # A warm microbenchmark of one query, the fastest of three repeats: not typical latency. scripts/query_timing.py measures that.
                rec["rerank_ms_first_query_min3"] = rr.timed_ms_per_query(ds.queries[0], [docs[j] for j in candidates], ds.task)
                for name in ("native", "oracle"):
                    keep_per_query(row, name, pqs[name])
                keep_per_query(row, "bridge", pqs["bridge"])
                if row in LEXICAL_ROWS:
                    fixed_pqs = {name: per_query(fixed[name], ds.qrels) for name in fixed}
                    rec["baselines_lexical"] = _strict_retention(fixed_pqs)
                    keep_per_query(row, "native_bm25", fixed_pqs["native"])
                    keep_per_query(row, "oracle_bm25", fixed_pqs["oracle"])
                    if no_bridge:  # BM25's top of the whole budget and the reranker: the old index isn't used at all
                        alone = rerank_pools(rr, [b[:depth] for b in lexical_top], ds.queries, docs, ds.task)
                        pq_alone = per_query(alone, ds.qrels)
                        rec["baseline_no_bridge"] = {m: float(v.mean()) for m, v in pq_alone.items()}
                        keep_per_query(row, "no_bridge", pq_alone)
                records.append(_score_record(rec, pqs["bridge"], pqs["native"], pqs["oracle"]))
            rr.unload()
    if per_query_path is not None:
        with durable.atomic_file(per_query_path) as f:
            np.savez_compressed(f, **per_query_scores)
    return records


def _strict_retention(pqs: dict[str, dict]) -> dict:
    """What native and re-index read through the same fix, and the bridge's retention against that stricter ceiling."""
    bridge, native, oracle = ({m: float(v.mean()) for m, v in pqs[name].items()} for name in ("bridge", "native", "oracle"))
    return {
        "native": native, "oracle": oracle, "rho_ceil": retention(bridge, native, oracle)[1],
        "ci_ceil": {m: bootstrap_retention_ci(pqs["bridge"][m], pqs["native"][m], pqs["oracle"][m]) for m in CI_METRICS},
    }


class _DocTexts:
    """Document text by row, built on demand: only the candidates are ever needed."""

    def __init__(self, titles: list[str], texts: list[str]):
        self.titles, self.texts = titles, texts

    def __getitem__(self, i: int) -> str:
        return f"{self.titles[i]}\n{self.texts[i]}" if self.titles[i] else self.texts[i]


def _first_docs(ds: RetrievalDataset, n: int) -> RetrievalDataset:
    """The dataset on its first ``n`` rows (``DatasetSpec.nested_docs`` corpora), sharing the whole corpus's embeddings."""
    if max(max(rel) for rel in ds.qrels) >= n:
        raise ValueError(f"{ds.name}: positives beyond row {n}; only nested corpora have fair prefixes")
    return replace(ds, name=f"{ds.name}@{n}", doc_ids=ds.doc_ids[:n])


def _first_queries(ds: RetrievalDataset, n: int) -> RetrievalDataset:
    """The dataset with its first ``n`` queries, for screening grids (CI about 2x wider at 500 of 2,000 queries)."""
    return replace(
        ds, queries=ds.queries[:n], qrels=ds.qrels[:n], self_rows=ds.self_rows[:n],
        src_queries=None if ds.src_queries is None else ds.src_queries[:n],
    )


def finished_cells(path: Path, config: str) -> set[str]:
    """Cells (seed, pair, K) fully recorded in a results file by a run of this exact configuration. A crash can tear the last line or
    leave a cell's records incomplete; those are removed from the file, so the rerun repeats that cell and nothing else. A file holding
    records of another configuration, or with none recorded, raises: a screening run must never count as the full run asked for now."""
    if not path.exists():
        return set()
    records, clean = durable.read_jsonl(path)
    other = {r.get("config_id") for r in records if "cell" in r} - {config}
    if other:
        raise ValueError(
            f"{path} holds records of another configuration ({sorted(map(str, other))}), not {config}. Use another --out, "
            "or --no-resume to add to it knowing that one file then holds two experiments"
        )
    counts = Counter(r["cell"] for r in records if "cell" in r)
    done = {r["cell"] for r in records if "cell" in r and counts[r["cell"]] == r["cell_records"]}
    kept = [r for r in records if "cell" not in r or r["cell"] in done]
    if not clean or len(kept) != len(records):
        with durable.atomic_file(path) as f:
            f.write("".join(json.dumps(r, default=_json_default) + "\n" for r in kept).encode())
    return done


def _score_record(rec: dict, pq: dict, native_pq: dict, oracle_pq: dict) -> dict:
    """Adds mean metrics, retention ratios and paired-bootstrap ceiling CIs to ``rec``."""
    native = {m: float(v.mean()) for m, v in native_pq.items()}
    oracle = {m: float(v.mean()) for m, v in oracle_pq.items()}
    rec["metrics"] = {m: float(v.mean()) for m, v in pq.items()}
    rec["ci_ceil"] = {m: bootstrap_retention_ci(pq[m], native_pq[m], oracle_pq[m]) for m in CI_METRICS}
    rec["ci_ceil_recall@30"] = rec["ci_ceil"]["recall@30"]
    rec["rho_oracle"], rec["rho_ceil"] = retention(rec["metrics"], native, oracle)
    rec["rho_native"] = {m: v / native[m] if native[m] > 0 else float("nan") for m, v in rec["metrics"].items()}
    return rec


def _checkpoint_study(bridge: SpaceBridge, Q: np.ndarray, lang: Optional[str], evaluate, ref_pq, native_pq, oracle_pq) -> dict:
    """Real test retention of every Stage-2 checkpoint and of checkpoint averages (with their label-free signals).

    Retention is against ``ref_pq`` (native, for the identity row) or the ceiling min(native, oracle)."""
    def rho(pq):
        out = {}
        for m in ("recall@30", "ndcg@10"):
            ref = ref_pq[m] if ref_pq is not None else (
                native_pq[m] if native_pq[m].mean() <= oracle_pq[m].mean() else oracle_pq[m])
            out[m] = float(pq[m].mean() / ref.mean())
        return out

    W_sel = bridge.W
    Ws = dict(bridge.stage2_checkpoints)
    averages = {"avg_all": np.mean(list(Ws.values()), 0)}
    for lo in (10, 20, 30):
        averages[f"avg_{lo}_50"] = np.mean([W for s, W in Ws.items() if s >= lo], 0)
    out = {}
    for name, W in {**{str(s): W for s, W in Ws.items()}, **averages}.items():
        bridge.W = W
        out[name] = rho(evaluate(bridge.transform_query(Q, query_lang=lang)))
        if name in averages and bridge.stage2_signals is not None:
            out[name]["signals"] = bridge.stage2_signals(W)
    bridge.W = W_sel
    return out


def _confidence_study(
    bridge: SpaceBridge, Q_tgt: np.ndarray, ranked: np.ndarray, native_ranked: np.ndarray,
    pq: dict, native_pq: dict, oracle_pq: dict, n_bins: int = 5,
) -> dict:
    """Real-query retention by quintile of per-query confidence (design §5.1)."""
    _, conf = bridge.transform_query(Q_tgt, return_confidence=True)
    support = bridge.query_support(Q_tgt)
    overlap = np.array([len(set(a[:30]) & set(b[:30])) / 30 for a, b in zip(ranked, native_ranked)])
    ceil = native_pq["recall@30"] if native_pq["recall@30"].mean() <= oracle_pq["recall@30"].mean() else oracle_pq["recall@30"]
    bins = np.array_split(np.argsort(support, kind="stable"), n_bins)
    return {
        "spearman_support_overlap30": spearman(support, overlap),
        "support_mean": float(support.mean()),
        "conf_mean": float(conf.mean()),
        "quintiles": [
            {
                "support": float(support[b].mean()),
                "conf": float(conf[b].mean()),
                "overlap@30_native": float(overlap[b].mean()),
                "rho_ceil_recall@30": float(pq["recall@30"][b].sum() / max(ceil[b].sum(), 1e-9)),
            }
            for b in bins
        ],
    }


def _projection_latency_us(bridge: SpaceBridge, q: np.ndarray, n: int = 1000) -> float:
    bridge.transform_query(q)
    t0 = time.perf_counter()
    for _ in range(n):
        bridge.transform_query(q)
    return 1e6 * (time.perf_counter() - t0) / n


def print_table(records: list[dict]) -> None:
    r0 = records[0]
    print(f"\n{r0['dataset']}  {r0['src']} (index) <- {r0['tgt']} (queries)  K={r0['K']}  n_q={r0['n_queries']}")
    header = f"{'row':>4}  {'R@10':>6} {'R@30':>6} {'nDCG@10':>7}  {'ρceil@30':>8} {'95% CI':^13} {'ρorac@30':>8}  {'ρ_val':>6}  {'solver':<16} status"
    print(header)
    print("-" * len(header))
    for r in (r for r in records if r["row"] not in RERANK_DEPTH):
        m, fit = r["metrics"], r.get("fit", {})
        solver = fit.get("solver", "")
        if "lambda_rel" in fit:
            solver += f" λ={fit['lambda_rel']}"
        if "stage2_best_step" in fit:
            solver += " +S2@tail" if fit.get("stage2_averaged") else f" +S2@{fit['stage2_best_step']}"
        rho_val = f"{fit['rho_val']:.3f}" if "rho_val" in fit else ""
        lo, hi = r["ci_ceil_recall@30"]
        print(
            f"{r['row']:>4}  {m['recall@10']:6.3f} {m['recall@30']:6.3f} {m['ndcg@10']:7.3f}  "
            f"{r['rho_ceil']['recall@30']:8.3f} [{lo:.3f},{hi:.3f}] {r['rho_oracle']['recall@30']:8.3f}  {rho_val:>6}  {solver:<16} {fit.get('status', '')}"
        )
    rr = [r for r in records if r["row"] in RERANK_DEPTH]
    if rr:
        header = f"{'row':>4} {'reranker':<20} {'depth':>5}  {'nDCG@5 nat/orc/bridge':^23}  {'ρceil nDCG@5':>12} {'95% CI':^13}  {'ρceil R@5':>9}  {'J@5 orc':>7}  {'ms/q':>6}"
        print(header)
        print("-" * len(header))
        for r in rr:
            b, m = r["baselines"], r["metrics"]
            lo, hi = r["ci_ceil"]["ndcg@5"]
            print(
                f"{r['row']:>4} {r['reranker']:<20} {r['depth']:>5}  {b['native']['ndcg@5']:.3f}/{b['oracle']['ndcg@5']:.3f}/{m['ndcg@5']:.3f}  "
                f"{r['rho_ceil']['ndcg@5']:12.3f} [{lo:.3f},{hi:.3f}]  {r['rho_ceil']['recall@5']:9.3f}  "
                f"{r['jaccard@5']['bridge_vs_oracle']:7.3f}  {r['rerank_ms_first_query_min3']:6.0f}"
            )
    timing = next((r["mount"] for r in records if "mount" in r), None)
    if timing:
        print(
            f"mount: encode {timing['encode_ms']:.0f} ms for {timing['n_inputs']} proxies "
            f"({timing['tokens_per_s']:.0f} tok/s; ~{timing['projected_encode_ms_K2048']:.0f} ms at K=2048)"
        )
        if "doc_encode_ms" in timing:
            print(f"       + anchor docs for row 8b: {timing['doc_encode_ms']:.0f} ms ({timing['doc_tokens_per_s']:.0f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    p.add_argument("--tier", nargs="*", default=[], choices=sorted(TIERS))
    p.add_argument("--pair", nargs="*", default=[], help="src:tgt model keys, e.g. bge-large-en:qwen3-0.6b")
    p.add_argument("--rows", nargs="*", default=[r for r in ROWS if r not in RERANK_DEPTH], choices=list(ROWS),
                   help="ablation rows; rerank rows 9n/9/9w are opt-in")
    p.add_argument("--rerankers", nargs="*", default=["bge-reranker-v2-m3", "qwen3-reranker-0.6b"], choices=list(RERANKERS))
    p.add_argument("--K", type=int, nargs="*", default=[0],
                   help="anchor budget(s); one run per value (0: spacebridge.default_anchor_count of the index)")
    p.add_argument("--seed", type=int, nargs="*", default=[0], help="one run per seed, in one process")
    p.add_argument("--time-mount", action="store_true",
                   help="re-encode anchors uncached to time the mount, and time the projection (use an idle GPU)")
    p.add_argument("--proxy", default="title", choices=PROXY_STYLES, help="query-proxy style")
    p.add_argument("--doc-pair-tokens", type=int, default=DOC_PAIR_TOKENS,
                   help="truncate anchor docs encoded for doc pairs to this many tokens (default: spacebridge.DOC_PAIR_TOKENS; 0: full length)")
    p.add_argument("--stage2-batch", type=int, default=None, help="Stage-2 candidate mini-batch size (default: the engine's; 0: full batch)")
    p.add_argument("--stage2-tau-scale", type=float, default=None, help="Stage-2 temperature relative to the teacher score s.d., on every index (ablation)")
    p.add_argument("--stage2-curves", action="store_true",
                   help="score every Stage-2 checkpoint on the test queries (stop-rule studies; records stage2_real)")
    p.add_argument("--stage2-rule", default=None, help="Stage-2 checkpoint selection rule (default: the engine's)")
    p.add_argument("--shift-bank", default="auto", choices=SHIFT_BANKS,
                   help="parallel sentences for the language-cone shift (auto: filtered FLORES dev if accessible, else Tatoeba)")
    p.add_argument("--anchor-bank", default="corpus", choices=("corpus", "flores", "flores-unfiltered"),
                   help="calibration anchors: corpus medoids, or the static FLORES dev + devtest bank (Tier 5b, Regime L2)")
    p.add_argument("--anchor-select", default="auto", choices=("auto", "kmeans", "random"),
                   help="how corpus anchors are chosen (auto: spacebridge.resolve_anchor_method)")
    p.add_argument("--full-index-diagnostics", action="store_true",
                   help="also score the validation proxies over the whole index (rho_val_full, owndoc_full; two extra searches per fit)")
    p.add_argument("--max-queries", type=int, default=None,
                   help="evaluate on the first N queries only: for screening grids, confirm winners on all queries")
    p.add_argument("--rerank-row", default=None, choices=sorted(BRIDGE_ROWS),
                   help="first stage the rerank rows reorder (default: the default engine)")
    p.add_argument("--max-docs", type=int, default=None,
                   help="index only the first N rows of a nested corpus (msmarco-full, nq-full), sharing its embeddings")
    p.add_argument("--no-bridge", action="store_true", help="BM25 rows also rerank BM25's top of the whole budget alone: no bridge, no old index")
    p.add_argument("--per-query-dir", type=Path, default=None, help="save per-query scores of each cell here, for scripts/retention_ci.py")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip cells already complete in --out by a run of this exact configuration, so rerunning a crashed command continues it (default: on)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/results.jsonl"))
    args = p.parse_args()

    pairs = [tuple(s.split(":")) for s in args.pair] + [pr for t in args.tier for pr in TIERS[t]]
    if not pairs:
        p.error("give --tier and/or --pair")
    for src, tgt in pairs:
        for key in (src, tgt):
            if key not in MODELS:
                p.error(f"unknown model {key!r}; choose from {sorted(MODELS)}")

    ds = load_beir(args.dataset)
    if args.max_docs is not None:
        ds = _first_docs(ds, args.max_docs)
    if args.max_queries is not None:
        ds = _first_queries(ds, args.max_queries)
    # Everything that can change a number; seed, pair and K are in the cell.
    config = {
        "dataset": ds.name, "n_docs": ds.N, "queries_sha256": provenance.query_digest(ds.queries), "rows": sorted(args.rows), "rerankers": sorted(args.rerankers),
        "proxy": args.proxy, "doc_pair_tokens": args.doc_pair_tokens, "stage2_batch": args.stage2_batch, "stage2_tau_scale": args.stage2_tau_scale,
        "stage2_rule": args.stage2_rule, "stage2_curves": args.stage2_curves, "shift_bank": args.shift_bank, "anchor_bank": args.anchor_bank,
        "anchor_select": args.anchor_select, "rerank_row": args.rerank_row, "full_index_diagnostics": args.full_index_diagnostics,
        "time_mount": args.time_mount, "no_bridge": args.no_bridge,
    }
    run_id = provenance.config_id(config)
    done = finished_cells(args.out, run_id) if args.resume else set()
    # Beside the results: what this run was, what it ran on, and when. One line per start, so a resumed run shows every commit it used.
    durable.append_lines(args.out.with_name(f"{args.out.name}.manifest.jsonl"), [json.dumps({
        "config_id": run_id, "config": config, "started": time.strftime("%F %T"), "device": args.device,
        **provenance.collect(
            [MODELS[k].hf_id for pr in pairs for k in pr] + [RERANKERS[r].hf_id for r in args.rerankers],
            [DATASETS[ds.name.split("@")[0]].repo],
        ),
    }, default=str)])
    cells = [(s, pr, K) for s in args.seed for pr in pairs for K in args.K]
    for n, (seed, (src, tgt), K) in enumerate(cells, 1):
        cell = f"{ds.name}|{src}|{tgt}|K{K}|s{seed}"
        if cell in done:
            durable.progress(f"cell {n}/{len(cells)} {cell}: already recorded, skipped")
            continue
        t0 = time.perf_counter()
        durable.progress(f"cell {n}/{len(cells)} {cell}: start")
        records = run_pair(ds, src, tgt, K, seed, args.rows, args.device, tuple(args.rerankers), args.proxy, args.stage2_batch,
                           args.doc_pair_tokens, args.time_mount, args.stage2_tau_scale, args.stage2_curves, args.stage2_rule,
                           args.shift_bank, args.anchor_bank, args.anchor_select, args.full_index_diagnostics, args.rerank_row,
                           args.no_bridge, args.per_query_dir / f"{cell.replace('|', '__')}.npz" if args.per_query_dir else None)
        durable.append_lines(args.out, [
            json.dumps({**rec, "cell": cell, "cell_records": len(records), "config_id": run_id}, default=_json_default) for rec in records
        ])
        durable.progress(f"cell {n}/{len(cells)} {cell}: done in {time.perf_counter() - t0:.0f} s")
        print_table(records)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
