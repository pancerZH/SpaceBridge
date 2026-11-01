"""Benchmark runner: tier/pair x ablation rows -> JSONL (design §7.2–§7.4).

Example:
    uv run spacebridge-bench --dataset scifact --tier 1 3
    uv run spacebridge-bench --dataset fiqa --pair bge-large-en:qwen3-0.6b --rows 0a 0b 6 8 --K 1024 2048
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from spacebridge import (
    DOC_PAIR_TOKENS, CalibrationBundle, RefineConfig, SpaceBridge, extract_query_proxy, make_query_proxies, make_split, sample_distractors,
    select_anchors,
)

from .datasets import DATASETS, FLORES_SPLITS, SHIFT_BANKS, RetrievalDataset, flores_available, load_beir, load_flores, load_shift_bank
from .encode import Encoder
from .metrics import bootstrap_ratio_ci, per_query, retention
from .models import MODELS
from .rerank import RERANKERS, Reranker, rerank, topk_jaccard
from .search import search

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
    "9n": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-30 + cross-encoder rerank",
    "9": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-60 + cross-encoder rerank",
    "9w": "Default engine (Row 8b; 7b on cross-lingual sets) bridged top-100 + cross-encoder rerank",
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

SEARCH_K = 100
VAL_K = 10  # rho_val's top-k (spacebridge.bridge.VAL_K)
ANCHOR_CACHE = Path("data/cache/anchors")
_warm = False
RERANK_DEPTH = {"9n": 30, "9": 60, "9w": 100}
CI_METRICS = ("recall@30", "ndcg@5", "ndcg@10")
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


def cached_anchors(ds: RetrievalDataset, src_key: str, E_src: np.ndarray, allowed: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Anchor doc rows, cached on disk so every pair and row of a dataset reuses one selection."""
    path = ANCHOR_CACHE / f"{ds.name}_{src_key}_N{ds.N}_n{len(allowed)}_K{K}_s{seed}_tkm.npy"  # _tkm: torch k-means
    if path.exists():
        return np.load(path)
    ids = allowed[select_anchors(E_src[allowed], K, seed)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    return ids


def run_pair(
    ds: RetrievalDataset, src_key: str, tgt_key: str, K: int, seed: int, rows: list[str], device: str,
    rerankers: tuple[str, ...] = (), proxy: str = "title", stage2_batch: Optional[int] = None,
    doc_pair_tokens: int = DOC_PAIR_TOKENS, time_mount: bool = False, stage2_tau_scale: Optional[float] = None,
    stage2_curves: bool = False, stage2_rule: Optional[str] = None, shift_bank: str = "auto", anchor_bank: str = "corpus",
) -> list[dict]:
    src, tgt = MODELS[src_key], MODELS[tgt_key]
    K = min(K, ds.N - int((ds.self_rows >= 0).sum()))
    static = anchor_bank != "corpus"
    if static:  # Regime L2 (design §4.4): FLORES dev + devtest pairs replace corpus anchors; none is an index row
        fl = load_flores(ds.lang if ds.lang != "en" else ds.query_lang, belebele_filter=anchor_bank == "flores")
        bank = {l: [t for sp in FLORES_SPLITS for t in fl[sp][l]] for l in (ds.lang, ds.query_lang)}
        K = min(K, len(bank[ds.lang]))
        pick = np.sort(np.random.default_rng(seed).permutation(len(bank[ds.lang]))[:K])
        bank = {l: [v[i] for i in pick] for l, v in bank.items()}

    enc = Encoder(src, device)
    E_src = enc.docs(ds.titles, ds.texts)
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
    # Split hygiene (design §7.1 rule 3): corpus rows that are test queries (ArguAna) never become anchors.
    allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
    if static:  # sentences serve as both the source proxies and the anchor docs; the target encodes the parallel side
        anchor_ids = np.full(K, -1, dtype=np.int64)
        proxies, tgt_proxies = bank[ds.lang], bank[ds.query_lang]
        X_doc_anchor = enc.docs([""] * K, proxies)
    else:
        anchor_ids = cached_anchors(ds, src_key, E_src, allowed, K, seed)
        proxies = make_proxies([ds.titles[i] for i in anchor_ids], [ds.texts[i] for i in anchor_ids], proxy)
        tgt_proxies, X_doc_anchor = proxies, E_src[anchor_ids]
    train_idx, val_idx = make_split(K, seed=seed)
    bundle = CalibrationBundle(
        src_lang=ds.lang,
        anchor_ids=anchor_ids,
        anchor_doc_texts=proxies if static else [ds.texts[i] for i in anchor_ids],
        anchor_query_texts=proxies,
        X_doc_raw=X_doc_anchor,
        X_query_raw=enc.queries(proxies, ds.task),
        train_idx=train_idx,
        val_idx=val_idx,
        distractor_ids=sample_distractors(ds.N, anchor_ids[anchor_ids >= 0], seed=seed),
        mu_x_doc=E_src.mean(0),
    )
    enc.unload()

    enc = Encoder(tgt, device)
    E_tgt = enc.docs(ds.titles, ds.texts)
    Q_tgt = enc.queries(ds.queries, ds.task)
    # Mount timing re-encodes the anchors uncached (~70 s per run for a 4B model), so it is opt-in.
    mount = None
    if time_mount:
        Y_query, mount = enc.timed_encode([tgt.format_query(p, ds.task) for p in tgt_proxies])
        mount["projected_encode_ms_K2048"] = mount["encode_ms"] * 2048 / K
    else:
        Y_query = enc.queries(tgt_proxies, ds.task)
    rerank_base = "7b" if ds.query_lang else "8b"  # Row 9 reranks the default engine's first stage
    needs = set(rows) | ({rerank_base} if set(rows) & set(RERANK_DEPTH) else set())
    if {"7", "7b"} & needs and ds.query_lang:
        bank_src, bank_q, bank_name = load_shift_bank(ds.query_lang, bank=shift_bank)
        Y_bank_src, Y_bank_q = enc.queries(bank_src, ds.task), enc.queries(bank_q, ds.task)
        for v in lang_val.values():
            v["Y_lang"], v["Y_src"] = enc.queries(v["lang"], ds.task), enc.queries(v["en"], ds.task)
    Y_doc_pairs = None
    if {"8b", "7b"} & needs and (time_mount or doc_pair_tokens):
        if doc_pair_tokens:  # cheaper mount: anchor docs for doc pairs encoded truncated
            enc.unload()
            enc = Encoder(replace(tgt, max_seq_length=min(doc_pair_tokens, tgt.max_seq_length)), device)
        doc_inputs = ([tgt.format_doc("", t) for t in tgt_proxies] if static
                      else [tgt.format_doc(ds.titles[i], ds.texts[i]) for i in anchor_ids])
        if time_mount:
            Y_doc_pairs, doc_mount = enc.timed_encode(doc_inputs)
            mount.update({f"doc_{k}": v for k, v in doc_mount.items()})
        else:
            Y_doc_pairs = enc.encode(doc_inputs)
    enc.unload()

    native_ranked = search(Q_src, E_src, SEARCH_K, device, exclude=ds.self_rows)
    oracle_ranked = search(Q_tgt, E_tgt, SEARCH_K, device, exclude=ds.self_rows)
    native_pq, oracle_pq = per_query(native_ranked, ds.qrels), per_query(oracle_ranked, ds.qrels)
    anchor_Y = {"src_query": bundle.X_query_raw, "tgt_query": Y_query, "tgt_doc": None if static else E_tgt[anchor_ids]}
    # Rows that run the default engine (doc<->doc pairs on); the identity row pairs source docs with themselves.
    Y_doc = Y_doc_pairs if doc_pair_tokens else anchor_Y["tgt_doc"]
    doc_rows = {"1": bundle.X_doc_raw, "8b": Y_doc, "7b": Y_doc}
    X_distractor = E_src[bundle.distractor_ids]
    # Diagnostic: rho_val over the whole index. The engine's pool holds the anchors + <= 50K distractors, so on a
    # larger index it cannot see the extra crowding (msmarco-1m, trec-covid); this scores the same validation proxies
    # against the teacher's top 10 over all N rows.
    val_teacher_full = search(bundle.X_query_raw[bundle.val_idx], E_src, VAL_K, device)

    global _warm
    if not _warm:  # one untimed fit, so the first row's timing excludes CUDA / cuSOLVER warmup
        SpaceBridge(device=device, doc_pairs=False).fit(bundle, Y_query, X_distractor=X_distractor)
        _warm = True
    bridged: dict[str, tuple[np.ndarray, dict, float, SpaceBridge]] = {}

    def fit_row(row: str) -> tuple[np.ndarray, dict, float, SpaceBridge]:
        if row not in bridged:
            y_kind, mode, pairs, restore = BRIDGE_ROWS[row]
            bridge = SpaceBridge(
                device=device, restore_source_mean=restore, match_residual_norm=row not in ("6n", "8n"),
                doc_pairs=row in doc_rows,
                refine_config=RefineConfig(
                    **({"batch_candidates": stage2_batch or None} if stage2_batch is not None else {}),
                    **({"tau_scale": stage2_tau_scale, "tau_scale_below_sd": float("inf")} if stage2_tau_scale else {}),
                    **({"stop_rule": stage2_rule} if stage2_rule else {}),
                    keep_checkpoints=stage2_curves,
                ),
            )
            extra = {"Y_doc_raw": doc_rows[row]} if row in doc_rows else {}
            fit = bridge.fit(bundle, anchor_Y[y_kind], mode=mode, pairs=pairs, X_distractor=X_distractor, **extra)
            Q = Q_src if row == "1" else Q_tgt
            lang = None
            if row in ("7", "7b"):
                lang = ds.query_lang
                bridge.register_query_language(lang, Y_bank_src, Y_bank_q)
                fit["lang_val"] = {
                    tag: bridge.validate_query_language(lang, v["Y_lang"], v["Y_src"], v["X_query"], v["X_doc"])
                    for tag, v in lang_val.items()
                }
            ranked = search(bridge.transform_query(Q, query_lang=lang), E_src, SEARCH_K, device, exclude=ds.self_rows)
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
            if row != "1" and mount is not None:
                rec["mount"] = mount
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
                rec["confidence"] = _confidence_study(bridge, Q_tgt, ranked, native_ranked, pq, native_pq, oracle_pq)
        records.append(_score_record(rec, pq, native_pq, oracle_pq))

    rerank_rows = [r for r in rows if r in RERANK_DEPTH]
    if rerank_rows and rerankers:
        ranked8, fit8, _, _ = fit_row(rerank_base)
        docs = [f"{t}\n{x}" if t else x for t, x in zip(ds.titles, ds.texts)]
        for rr_key in rerankers:
            rr = Reranker(RERANKERS[rr_key], device)
            for row in rerank_rows:
                depth = RERANK_DEPTH[row]
                out = {name: rerank(rr, r, depth, qs, docs, ds.task) for name, r, qs in
                       (("native", native_ranked, ds.index_lang_queries), ("oracle", oracle_ranked, ds.queries),
                        ("bridge", ranked8, ds.queries))}
                pqs = {name: per_query(r, ds.qrels) for name, r in out.items()}
                rec = {**base, "row": row, "description": ROWS[row], "reranker": rr_key, "depth": depth, "fit": fit8,
                       "first_stage_row": rerank_base}
                rec["baselines"] = {name: {m: float(v.mean()) for m, v in pqs[name].items()} for name in ("native", "oracle")}
                rec["jaccard@5"] = {
                    "bridge_vs_oracle": float(topk_jaccard(out["bridge"], out["oracle"]).mean()),
                    "bridge_vs_native": float(topk_jaccard(out["bridge"], out["native"]).mean()),
                    "native_vs_oracle": float(topk_jaccard(out["native"], out["oracle"]).mean()),
                }
                rec["rerank_ms_per_query"] = rr.timed_ms_per_query(ds.queries[0], [docs[j] for j in ranked8[0, :depth]], ds.task)
                records.append(_score_record(rec, pqs["bridge"], pqs["native"], pqs["oracle"]))
            rr.unload()
    return records


def _score_record(rec: dict, pq: dict, native_pq: dict, oracle_pq: dict) -> dict:
    """Adds mean metrics, retention ratios and paired-bootstrap ceiling CIs to ``rec``."""
    native = {m: float(v.mean()) for m, v in native_pq.items()}
    oracle = {m: float(v.mean()) for m, v in oracle_pq.items()}
    rec["metrics"] = {m: float(v.mean()) for m, v in pq.items()}
    rec["ci_ceil"] = {
        m: bootstrap_ratio_ci(pq[m], native_pq[m] if native[m] <= oracle[m] else oracle_pq[m]) for m in CI_METRICS
    }
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
        "spearman_support_overlap30": _spearman(support, overlap),
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


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
                f"{r['jaccard@5']['bridge_vs_oracle']:7.3f}  {r['rerank_ms_per_query']:6.0f}"
            )
    mount = next((r["mount"] for r in records if "mount" in r), None)
    if mount:
        print(
            f"mount: encode {mount['encode_ms']:.0f} ms for {mount['n_inputs']} proxies "
            f"({mount['tokens_per_s']:.0f} tok/s; ~{mount['projected_encode_ms_K2048']:.0f} ms at K=2048)"
        )
        if "doc_encode_ms" in mount:
            print(f"       + anchor docs for row 8b: {mount['doc_encode_ms']:.0f} ms ({mount['doc_tokens_per_s']:.0f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    p.add_argument("--tier", nargs="*", default=[], choices=sorted(TIERS))
    p.add_argument("--pair", nargs="*", default=[], help="src:tgt model keys, e.g. bge-large-en:qwen3-0.6b")
    p.add_argument("--rows", nargs="*", default=[r for r in ROWS if r not in RERANK_DEPTH], choices=list(ROWS),
                   help="ablation rows; rerank rows 9n/9/9w are opt-in")
    p.add_argument("--rerankers", nargs="*", default=list(RERANKERS), choices=list(RERANKERS))
    p.add_argument("--K", type=int, nargs="*", default=[2048], help="anchor budget(s); one run per value")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed, (src, tgt), K in ((s, pr, K) for s in args.seed for pr in pairs for K in args.K):
        records = run_pair(ds, src, tgt, K, seed, args.rows, args.device, tuple(args.rerankers), args.proxy, args.stage2_batch,
                           args.doc_pair_tokens, args.time_mount, args.stage2_tau_scale, args.stage2_curves, args.stage2_rule,
                           args.shift_bank, args.anchor_bank)
        with args.out.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        print_table(records)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
