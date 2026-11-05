"""Part B: does anything like BM25 help cross-language search when the old query model is gone?

Index in English, queries in Chinese through Row 7b, the reranker reading the original query. Candidates:

    X1  BM25 on the raw Chinese query, a quarter of the budget
    X2  BM25 on the opus-mt translation, a quarter of the budget
    X3  BM25 on the MADLAD translation, a quarter of the budget
    X4o, X4m  no bridge: BM25 on one translation, then the reranker
    X5  local residuals on Row 7b's first stage: the calibration pairs nearest the query correct it

and, for the comparison with translation while the old model is alive, no bridge at all:

    TMo, TMm  the old model searching on a translation: M1's translation baseline
    TBo, TBm  the same, with BM25 on that translation, a quarter of the budget

One JSON line per (set, query model, seed, candidate, budget) goes to --out as it finishes; a rerun skips those. Translations come
from the cache (scripts/translation_baseline.py) and no translator loads if they're all there.

    uv run --group translate python -u scripts/cross_lang_batch.py xquad-zh mlqa-zh --query-model qwen3-0.6b --seed 0 --out out.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from translation_baseline import translate_cached  # noqa: E402

from spacebridge import Index, RefineConfig, freeze, merge_pool, mount  # noqa: E402
from spacebridge.lexical import LEXICAL_SHARE  # noqa: E402
from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.datasets import load_beir, load_shift_bank  # noqa: E402
from spacebridge_bench.encode import Encoder, MountEncoder  # noqa: E402
from spacebridge_bench.metrics import per_query  # noqa: E402
from spacebridge_bench.models import MODELS  # noqa: E402
from spacebridge_bench.rerank import RERANKERS, Reranker, rerank, rerank_pools  # noqa: E402
from spacebridge_bench.run import _DocTexts, _first_queries, cached_anchors, cached_lexical  # noqa: E402
from spacebridge_bench.search import cached_search, search  # noqa: E402

MAX_QUERIES = {"mlqa-zh": 1_500}
LOCAL = {"k": 50, "tau": 20.0, "g": 0.5}  # from exp_local in scripts/hotpot_study.py, chosen there on same-language test queries
CANDIDATES = ("7b", "reindex", "native", "X1", "X2", "X3", "X4o", "X4m", "X5", "TMo", "TMm", "TBo", "TBm")


def local_first_stage(bridge, bundle, tgt_mount, Q_tgt: np.ndarray, Q_hat: np.ndarray, lang: str, device: str) -> np.ndarray:
    """``exp_local`` for queries in another language: the pairs are in the index's language, so a query first drops the
    language shift to find its neighbours, and the corrections are the same as for same-language queries."""
    tr = bundle.train_idx
    yq, xq = tgt_mount.encode_queries(bundle.anchor_query_texts)[tr], bundle.X_query_raw[tr]
    yd = tgt_mount.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts, max_tokens=128)[tr]
    xd = bundle.X_doc_raw[tr]
    tgt_mount.unload()
    W, mu_x, mu_y = bridge.W, bridge.mu_x_query, bridge.mu_y_by_lang[bundle.src_lang]
    pred_q, pred_d = (yq - mu_y) @ W + mu_x, (yd - yd.mean(0)) @ W + xd.mean(0)
    pairs = torch.as_tensor(np.vstack([yq, yd]), device=device)
    resid = torch.as_tensor(np.vstack([xq - pred_q, xd - pred_d]), device=device)
    aligned = F.normalize(torch.as_tensor(Q_tgt + mu_y - bridge.mu_y_by_lang[lang], device=device), dim=1)
    top, idx = (aligned @ pairs.T).topk(LOCAL["k"])
    corr = ((LOCAL["tau"] * top).softmax(1)[..., None] * resid[idx]).sum(1)
    return F.normalize(torch.as_tensor(Q_hat, device=device) + LOCAL["g"] * corr, dim=1).cpu().numpy()


def run(name: str, tgt_key: str, seed: int, budgets: list[int], K: int, candidates: tuple[str, ...], device: str, out: Path, done: set[str]) -> None:
    keys = {f"{name}|{tgt_key}|s{seed}|{c}|B{b}": (c, b) for c in candidates for b in budgets}
    if all(k in done for k in keys):
        return
    ds = load_beir(name)
    if name in MAX_QUERIES:
        ds = _first_queries(ds, MAX_QUERIES[name])
    src, tgt, lang = MODELS["bge-large-en"], MODELS[tgt_key], ds.query_lang
    enc = Encoder(src, device)
    E_src = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q_native = enc.queries(ds.index_lang_queries, ds.task)
    enc.unload()
    enc = Encoder(tgt, device)
    E_tgt = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q_tgt = np.asarray(enc.queries(ds.queries, ds.task), dtype=np.float32)
    enc.unload()
    index = ds.index_over(E_src)
    allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
    src_mount, tgt_mount = MountEncoder(src, ds.task, device), MountEncoder(tgt, ds.task, device)
    bundle = freeze(index, src_mount, ds.lang, seed, K, anchor_ids=cached_anchors(ds, src.key, E_src, allowed, K, seed), device=device)
    src_mount.unload()
    bank_src, bank_q, _ = load_shift_bank(lang)
    bridge = mount(bundle, index, tgt_mount, [lang], device, doc_pairs=True, refine_config=RefineConfig(), shift_banks={lang: (bank_src, bank_q)})
    tgt_mount.unload()
    Q_hat = bridge.transform_query(Q_tgt, query_lang=lang)
    bridged = search(Q_hat, E_src, 1000, device, exclude=ds.self_rows)
    native = cached_search(Q_native, E_src, 1000, device, exclude=ds.self_rows)
    reindexed = cached_search(Q_tgt, E_tgt, 1000, device, exclude=ds.self_rows)
    local = search(local_first_stage(bridge, bundle, tgt_mount, Q_tgt, Q_hat, lang, device), E_src, 1000, device, exclude=ds.self_rows) if "X5" in candidates else None
    lex = cached_lexical(ds, index)
    opus, madlad = translate_cached("opus", ds.queries, device), translate_cached("madlad", ds.queries, device)
    texts = {"X1": ds.queries, "X2": opus, "X3": madlad, "X4o": opus, "X4m": madlad, "TBo": opus, "TBm": madlad}
    old_model = {}
    if {"TMo", "TMm", "TBo", "TBm"} & set(candidates):
        enc = Encoder(src, device)
        for tag, translated in (("o", opus), ("m", madlad)):
            old_model[tag] = cached_search(enc.queries(translated, ds.task), E_src, 1000, device, exclude=ds.self_rows)
        enc.unload()
    rr, docs = Reranker(RERANKERS["bge-reranker-v2-m3"], device), _DocTexts(ds.titles, ds.texts)
    for key, (c, b) in keys.items():
        if key in done:
            continue
        t0 = time.time()
        first = None
        if c in ("7b", "reindex", "native", "X5", "TMo", "TMm"):
            ranked, qs = {"7b": (bridged, ds.queries), "reindex": (reindexed, ds.queries), "native": (native, ds.index_lang_queries), "X5": (local, ds.queries),
                          "TMo": (old_model.get("o"), ds.queries), "TMm": (old_model.get("m"), ds.queries)}[c]
            rows = rerank(rr, ranked, b, qs, docs, ds.task)
            first = per_query(ranked, ds.qrels)["recall@300"].mean()
            pool_mean = float(b)
        else:
            share = b if c.startswith("X4") else b // LEXICAL_SHARE
            lexical = [lex.top(t, share, exclude=s) for t, s in zip(texts[c], ds.self_rows.tolist())]
            dense = [np.empty(0, dtype=np.int64)] * len(lexical) if c.startswith("X4") else old_model[c[-1]] if c.startswith("TB") else bridged
            pools = [merge_pool(d, x, b) for d, x in zip(dense, lexical)]
            if not all(len(p) for p in pools):  # a translation with no word in the index, and no bridge to fall back on
                pools = [p if len(p) else np.array([0], dtype=np.int64) for p in pools]
            rows = rerank_pools(rr, pools, ds.queries, docs, ds.task)
            pool_mean = float(np.mean([len(p) for p in pools]))
        m = per_query(rows, ds.qrels)
        rec = {"key": key, "dataset": name, "tgt": tgt_key, "seed": seed, "K": bundle.K, "candidate": c, "budget": b, "n": len(ds.queries), "pool_mean": pool_mean, "seconds": time.time() - t0,
               "metrics": {k: float(v.mean()) for k, v in m.items()}, "recall@300_first_stage": None if first is None else float(first)}
        durable.append_lines(out, [json.dumps(rec)])
        durable.progress(f"{key}: nDCG@5 {rec['metrics']['ndcg@5']:.4f}, pool {pool_mean:.1f}")
    rr.unload()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+")
    p.add_argument("--query-model", default="qwen3-0.6b")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--budgets", type=int, nargs="+", default=[60, 100])
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES), choices=CANDIDATES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    done = {r["key"] for r in durable.read_jsonl(args.out)[0]} if args.out.exists() else set()
    for name in args.datasets:
        for seed in args.seed:
            run(name, args.query_model, seed, args.budgets, args.K, tuple(args.candidates), args.device, args.out, done)


if __name__ == "__main__":
    main()
