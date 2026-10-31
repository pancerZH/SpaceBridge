"""Which validation signal tracks real-query retention? (design §7.4 routing calibration study)

Fits many configurations per pair (anchor count x solver x mean restore), records every candidate
validation metric from ``SpaceBridge.fit`` next to the real test-query retention, and reports
Spearman correlations, pooled over all fits and averaged within pairs.

    uv run python scripts/rho_val_study.py --dataset scifact
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spacebridge import CalibrationBundle, SpaceBridge, extract_query_proxy, make_split, sample_distractors, select_anchors
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import evaluate
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import TIERS
from spacebridge_bench.search import search

KS = (128, 256, 512, 1024, 2048, 4096)
SETTINGS = [  # (mode, restore_source_mean)
    ("naive", True), ("stiefel", True), ("ridge", True), ("stage1", False), ("auto", True),
]
TARGET = "recall@30"


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--tiers", nargs="*", default=["1", "3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/rho_val_study.jsonl"))
    args = p.parse_args()

    ds = load_beir(args.dataset)
    N = ds.N
    pairs = [pr for t in args.tiers for pr in TIERS[t]]
    keys = sorted({k for pr in pairs for k in pr})
    proxies = [extract_query_proxy(ds.titles[i], ds.texts[i]) for i in range(N)]
    emb = {}
    for key in keys:
        enc = Encoder(MODELS[key], args.device)
        emb[key] = {"docs": enc.docs(ds.titles, ds.texts), "q": enc.queries(ds.queries, ds.task), "proxy": enc.queries(proxies, ds.task)}
        enc.unload()

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for src, tgt in pairs:
        s, t = emb[src], emb[tgt]
        native = evaluate(search(s["q"], s["docs"], 100, args.device, exclude=ds.self_rows), ds.qrels)
        oracle = evaluate(search(t["q"], t["docs"], 100, args.device, exclude=ds.self_rows), ds.qrels)
        ceiling = min(native[TARGET], oracle[TARGET])
        for K in (K for K in KS if K <= N):
            ids = select_anchors(s["docs"], K)
            train_idx, val_idx = make_split(K)
            bundle = CalibrationBundle(
                src_lang=ds.lang, anchor_ids=ids, anchor_doc_texts=[], anchor_query_texts=[],
                X_doc_raw=s["docs"][ids], X_query_raw=s["proxy"][ids], train_idx=train_idx, val_idx=val_idx,
                distractor_ids=sample_distractors(N, ids),
            )
            for mode, restore in SETTINGS:
                bridge = SpaceBridge(device=args.device, restore_source_mean=restore)
                fit = bridge.fit(bundle, t["proxy"][ids], mode=mode, X_distractor=s["docs"][bundle.distractor_ids])
                real = evaluate(search(bridge.transform_query(t["q"]), s["docs"], 100, args.device, exclude=ds.self_rows), ds.qrels)
                rows.append({
                    "src": src, "tgt": tgt, "K": K, "mode": mode, "restore": restore,
                    "retention": real[TARGET] / ceiling, **fit["val_metrics"],
                })
        print(f"done {src} <- {tgt}")
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    retention = np.array([r["retention"] for r in rows])
    signals = [k for k in rows[0] if k not in ("src", "tgt", "K", "mode", "restore", "retention")]
    print(f"\n{len(rows)} fits; Spearman correlation with real {TARGET} retention")
    print(f"{'signal':<16} {'pooled':>7} {'within-pair mean':>17} {'min':>6}")
    for sig in signals:
        vals = np.array([r[sig] for r in rows])
        within = [
            spearman(vals[m], retention[m])
            for m in (np.array([(r["src"], r["tgt"]) == pr for r in rows]) for pr in pairs)
        ]
        print(f"{sig:<16} {spearman(vals, retention):7.3f} {np.mean(within):17.3f} {min(within):6.3f}")


if __name__ == "__main__":
    main()
