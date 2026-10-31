"""Does Stage 2 help when it is not early-stopped, and which validation signal tracks its real gain?

For each pair, fits Stage 1 only, the default Stage 2 (early-stopped on held-out own-doc MRR), the v2 default (agree@10), and Stage 2 run for a
fixed number of steps (optionally with hard negatives), and prints real test retention next to the
label-free validation signals computed on held-out proxies.

    uv run python scripts/stage2_study.py --dataset fiqa --K 2048
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spacebridge import CalibrationBundle, RefineConfig, SpaceBridge, extract_query_proxy, make_split, sample_distractors
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import TIERS, cached_anchors
from spacebridge_bench.search import search

CONFIGS = {  # name -> (mode, RefineConfig kwargs, hard negatives, Stage-2 stopping signal)
    "stage1": ("stage1", {}, 0, "owndoc_mrr"),
    "default": ("auto", {}, 0, "owndoc_mrr"),
    "v2-agree": ("auto", {"steps": 150}, 0, "agree@10"),
    "fixed20": ("auto", {"steps": 20, "early_stop": False}, 0, "owndoc_mrr"),
    "fixed50": ("auto", {"steps": 50, "early_stop": False}, 0, "owndoc_mrr"),
    "fixed150": ("auto", {"steps": 150, "early_stop": False}, 0, "owndoc_mrr"),
    "fixed50+hn": ("auto", {"steps": 50, "early_stop": False}, 10, "owndoc_mrr"),
}
SIGNALS = ("agree@10", "nbr_recall@10", "nbr_recall@30", "nbr_recall@100", "score_corr", "owndoc_mrr")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--tiers", nargs="*", default=["1", "3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/stage2_study.jsonl"))
    args = p.parse_args()

    ds = load_beir(args.dataset)
    pairs = [pr for t in args.tiers for pr in TIERS[t]]
    emb = {}
    for key in sorted({k for pr in pairs for k in pr}):
        enc = Encoder(MODELS[key], args.device)
        emb[key] = {"docs": enc.docs(ds.titles, ds.texts), "q": enc.queries(ds.index_lang_queries, ds.task)}
        enc.unload()

    rows = []
    for src, tgt in pairs:
        s, t = emb[src], emb[tgt]
        E = s["docs"]
        allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
        ids = cached_anchors(ds, src, E, allowed, args.K, 0)
        proxies = [extract_query_proxy(ds.titles[i], ds.texts[i]) for i in ids]
        enc = Encoder(MODELS[src], args.device)
        Xq = enc.queries(proxies, ds.task)
        enc.unload()
        enc = Encoder(MODELS[tgt], args.device)
        Y = enc.queries(proxies, ds.task)
        enc.unload()
        tr, va = make_split(args.K)
        bundle = CalibrationBundle(ds.lang, ids, [], proxies, E[ids], Xq, tr, va, sample_distractors(ds.N, ids), E.mean(0))

        evaluate = lambda Q, E_: per_query(search(Q, E_, 100, args.device, exclude=ds.self_rows), ds.qrels)
        native, oracle = evaluate(s["q"], E), evaluate(t["q"], t["docs"])
        ceil = {m: native[m] if native[m].mean() <= oracle[m].mean() else oracle[m] for m in ("recall@30", "ndcg@10")}
        print(f"\n{src} <- {tgt}")
        print(f"{'config':<13} {'ρceil R@30':>10} {'nDCG@10':>8}  " + " ".join(f"{x:>14}" for x in SIGNALS) + "  step")
        for name, (mode, rc, hn, signal) in CONFIGS.items():
            bridge = SpaceBridge(device=args.device, refine_config=RefineConfig(**rc), hard_negatives=hn, stage2_signal=signal)
            fit = bridge.fit(bundle, Y, mode=mode, X_distractor=E[bundle.distractor_ids])
            pq = evaluate(bridge.transform_query(t["q"]), E)
            rec = {
                "dataset": args.dataset, "src": src, "tgt": tgt, "K": args.K, "config": name,
                "rho_ceil_recall@30": float(pq["recall@30"].mean() / ceil["recall@30"].mean()),
                "rho_ceil_ndcg@10": float(pq["ndcg@10"].mean() / ceil["ndcg@10"].mean()),
                "step": fit.get("stage2_best_step", 0), **{k: fit["val_metrics"][k] for k in SIGNALS},
            }
            rows.append(rec)
            print(f"{name:<13} {rec['rho_ceil_recall@30']:10.3f} {rec['rho_ceil_ndcg@10']:8.3f}  "
                  + " ".join(f"{rec[x]:14.4f}" for x in SIGNALS) + f"  {rec['step']}")

    with args.out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\nMean change vs. stage1 over pairs (real retention, and each validation signal):")
    base = {(r["src"], r["tgt"]): r for r in rows if r["config"] == "stage1"}
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        d = lambda k: np.mean([r[k] - base[(r["src"], r["tgt"])][k] for r in rs])
        print(f"{name:<13} R@30 {d('rho_ceil_recall@30'):+.4f}  nDCG@10 {d('rho_ceil_ndcg@10'):+.4f}  "
              + "  ".join(f"{x} {d(x):+.4f}" for x in SIGNALS))


if __name__ == "__main__":
    main()
