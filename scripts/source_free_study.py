"""W6: variants of the source-model-free fit on real queries, to choose its defaults. One JSON line per fit; a rerun skips it.

    uv run python -u scripts/source_free_study.py fiqa --pair bge-large-en:qwen3-0.6b --seed 0 1 2 --out x.jsonl
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from spacebridge import Index, RefineConfig, SpaceBridge, freeze
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.metrics import per_query, retention
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import _first_queries, cached_anchors
from spacebridge_bench.search import cached_search, search

VARIANTS = {
    "A_doc_select_s1": dict(center="doc", operator="select", stage2=False),
    "B_doc_stiefel_s1": dict(center="doc", operator="stiefel", stage2=False),
    "C_query_stiefel_s1": dict(center="query", operator="stiefel", stage2=False),
    "D_doc_select_s2both": dict(center="doc", operator="select", stage2=True),
    "E_doc_stiefel_s2both": dict(center="doc", operator="stiefel", stage2=True),
    "F_query_stiefel_s2both": dict(center="query", operator="stiefel", stage2=True),
    "G_doc_stiefel_s2prox": dict(center="doc", operator="stiefel", stage2=True, stage2_rows="proxies"),
    "H_query_stiefel_s2prox": dict(center="query", operator="stiefel", stage2=True, stage2_rows="proxies"),
    "I_doc_stiefel_s2docs": dict(center="doc", operator="stiefel", stage2=True, stage2_rows="docs"),
    "J_query_stiefel_s2both_lr1e-3": dict(center="query", operator="stiefel", stage2=True, lr=1e-3),
    "K_doc_stiefel_s2both_lr1e-3": dict(center="doc", operator="stiefel", stage2=True, lr=1e-3),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--pair", nargs="+", required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ds = load_beir(args.dataset)
    if args.max_queries:
        ds = _first_queries(ds, args.max_queries)
    done = {r["key"] for r in durable.read_jsonl(args.out)[0]} if args.out.exists() else set()
    for pair in args.pair:
        src_key, tgt_key = pair.split(":")
        src, tgt = MODELS[src_key], MODELS[tgt_key]
        enc = Encoder(src, args.device)
        E_src = enc.docs_store(ds.titles, ds.texts)
        Q_src = enc.queries(ds.index_lang_queries, ds.task)
        enc.unload()
        enc = Encoder(tgt, args.device)
        E_tgt = enc.docs_store(ds.titles, ds.texts)
        Q_tgt = enc.queries(ds.queries, ds.task)
        enc.unload()
        native = per_query(cached_search(Q_src, E_src, 1000, args.device, exclude=ds.self_rows), ds.qrels)
        oracle = per_query(cached_search(Q_tgt, E_tgt, 1000, args.device, exclude=ds.self_rows), ds.qrels)
        native_m, oracle_m = {m: float(v.mean()) for m, v in native.items()}, {m: float(v.mean()) for m, v in oracle.items()}
        index = ds.index_over(E_src)
        for seed in args.seed:
            allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
            K = args.K or None
            from spacebridge import default_anchor_count

            K_ = min(K or default_anchor_count(ds.N), len(allowed))
            bundle = freeze(index, None, ds.lang, seed, K_, anchor_ids=cached_anchors(ds, src_key, E_src, allowed, K_, seed), device=args.device)
            mt = MountEncoder(tgt, ds.task, args.device)
            Y_q = mt.encode_queries(bundle.anchor_query_texts)
            Y_d = mt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts, max_tokens=128)
            mt.unload()
            X_dis = index.vectors(bundle.distractor_ids)
            for name in args.variants:
                key = f"{ds.name}|{pair}|K{K_}|s{seed}|{name}"
                if key in done:
                    continue
                kw = dict(VARIANTS[name])
                lr = kw.pop("lr", None)
                bridge = SpaceBridge(device=args.device, refine_config=RefineConfig(**({"lr": lr} if lr else {})))
                fit = bridge.fit_source_free(bundle, Y_q, Y_d, X_dis, **kw)
                ranked = search(bridge.transform_query(Q_tgt), E_src, 1000, args.device, exclude=ds.self_rows)
                m = {k: float(v.mean()) for k, v in per_query(ranked, ds.qrels).items()}
                _, rho = retention(m, native_m, oracle_m)
                rec = {"key": key, "dataset": ds.name, "pair": pair, "K": K_, "seed": seed, "variant": name, "solver": fit.get("solver"), "rho": {k: rho[k] for k in ("recall@30", "recall@100", "ndcg@10")},
                       "owndoc_mrr": fit["quality"]["recoverability"]["owndoc_mrr"], "owndoc_mrr_s1": fit["quality"]["recoverability"]["owndoc_mrr_stage1"], "val_r2": fit["quality"]["fit"]["val_r2"], "val_resid_cos": fit["quality"]["fit"]["val_resid_cos"]}
                durable.append_lines(args.out, [json.dumps(rec)])
                durable.progress(f"{key}: R@30 {rho['recall@30']:.3f} R@100 {rho['recall@100']:.3f} mrr {rec['owndoc_mrr']:.3f}")


if __name__ == "__main__":
    main()
