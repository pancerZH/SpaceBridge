"""Which label-free signal should pick the Stage-1 operator? (M3 L5)

For each pair, every Stage-1 candidate (Stiefel, ridge at each lambda) is fitted alone on the default
pipeline (doc<->doc pairs on), once as Stage 1 only and once followed by Stage 2. Real test retention is
recorded next to the candidate's label-free validation signals, so a signal can be judged by the real
retention of the candidate it would have picked.

    uv run python scripts/stage1_study.py --dataset fiqa --K 2048 4096 --seeds 0 1 2
    uv run python scripts/stage1_study.py --summarize outputs/m3_stage1_study.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge import CalibrationBundle, SpaceBridge, extract_query_proxy, make_split, sample_distractors
from spacebridge.solvers import LAMBDAS_REL
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import TIERS, cached_anchors
from spacebridge_bench.search import search

CANDIDATES = [("stiefel", None)] + [("ridge", lam) for lam in LAMBDAS_REL]
SIGNALS = ("agree@10", "nbr_recall@10", "nbr_recall@30", "nbr_recall@100", "score_corr", "owndoc_mrr", "val_r2", "val_resid_cos")


def run(args) -> None:
    ds = load_beir(args.dataset)
    pairs = [pr for t in args.tiers for pr in TIERS[t]]
    emb = {}
    for key in sorted({k for pr in pairs for k in pr}):
        enc = Encoder(MODELS[key], args.device)
        emb[key] = {"docs": enc.docs(ds.titles, ds.texts), "q": enc.queries(ds.index_lang_queries if key in {s for s, _ in pairs} else ds.queries, ds.task)}
        if key in {t for _, t in pairs}:
            emb[key]["q_tgt"] = enc.queries(ds.queries, ds.task)
        enc.unload()
    evaluate = lambda Q, E: per_query(search(Q, E, 100, args.device, exclude=ds.self_rows), ds.qrels)
    allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)

    with args.out.open("a") as f:
        for (src, tgt), K, seed in ((pr, K, s) for pr in pairs for K in args.K for s in args.seeds):
            E, E_t = emb[src]["docs"], emb[tgt]["docs"]
            ids = cached_anchors(ds, src, E, allowed, K, seed)
            proxies = [extract_query_proxy(ds.titles[i], ds.texts[i]) for i in ids]
            enc = Encoder(MODELS[src], args.device)
            Xq = enc.queries(proxies, ds.task)
            enc.unload()
            enc = Encoder(MODELS[tgt], args.device)
            Y = enc.queries(proxies, ds.task)
            enc.unload()
            tr, va = make_split(K, seed=seed)
            bundle = CalibrationBundle(ds.lang, ids, [], proxies, E[ids], Xq, tr, va, sample_distractors(ds.N, ids, seed=seed), E.mean(0))
            X_dis = E[bundle.distractor_ids]
            native, oracle = evaluate(emb[src]["q"], E), evaluate(emb[tgt]["q_tgt"], E_t)
            ceil = {m: native[m] if native[m].mean() <= oracle[m].mean() else oracle[m] for m in ("recall@30", "ndcg@10")}
            ret = lambda pq: {m: float(pq[m].mean() / ceil[m].mean()) for m in ceil}
            print(f"\n{args.dataset} {src} <- {tgt} K={K} seed={seed}")
            for op, lam in CANDIDATES:
                kw = dict(device=args.device, stage1_operators=(op,), lambdas_rel=(lam,) if lam is not None else LAMBDAS_REL)
                rec = {"dataset": args.dataset, "src": src, "tgt": tgt, "K": K, "seed": seed, "op": op, "lambda_rel": lam}
                for mode in ("stage1", "auto"):
                    bridge = SpaceBridge(**kw)
                    fit = bridge.fit(bundle, Y, mode=mode, X_distractor=X_dis, Y_doc_raw=E_t[ids])
                    r = ret(evaluate(bridge.transform_query(emb[tgt]["q_tgt"]), E))
                    sig = {**fit["val_metrics"], **{k: fit["quality"]["fit"][k] for k in ("val_r2", "val_resid_cos")}}
                    rec[mode] = {"rho_ceil": r, "signals": {k: sig[k] for k in SIGNALS}, "stage2_step": fit.get("stage2_best_step")}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{op:<8} {str(lam):>5}  S1 {rec['stage1']['rho_ceil']['recall@30']:.3f}  S1+S2 {rec['auto']['rho_ceil']['recall@30']:.3f} "
                      f"(step {rec['auto']['stage2_step']})  " + " ".join(f"{k} {rec['stage1']['signals'][k]:.3f}" for k in SIGNALS))


def summarize(paths: list[Path], metric: str) -> None:
    fits = defaultdict(list)
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            fits[r["dataset"], r["src"], r["tgt"], r["K"], r["seed"]].append(r)
    by_ds = defaultdict(list)
    for key, cands in fits.items():
        if len(cands) == len(CANDIDATES):
            by_ds[key[0]].append(cands)
    for stage in ("stage1", "auto"):
        print(f"\n## Real {metric} ρceil of the candidate each signal picks ({'Stage 1 only' if stage == 'stage1' else 'after Stage 2'}), points vs. the agree@10 pick\n")
        print("Signals are measured on the Stage-1 map in both cases (that is when selection happens).\n")
        fixed = [f"always {op}" if lam is None else f"always ridge {lam}" for op, lam in CANDIDATES]
        print("| dataset | fits | agree@10 pick (current) | best candidate (oracle) | " + " | ".join(SIGNALS[1:] + tuple(fixed)) + " |")
        print("| --- | --- | --- | --- | " + " | ".join("---" for _ in SIGNALS[1:] + tuple(fixed)) + " |")
        for d, flist in sorted(by_ds.items()):
            real = lambda c: c[stage]["rho_ceil"][metric]
            pick = lambda cands, s: max(cands, key=lambda c: c["stage1"]["signals"][s])
            base = np.array([real(pick(c, "agree@10")) for c in flist])
            best = np.array([max(real(x) for x in c) for c in flist])
            cells = []
            for s in SIGNALS[1:]:
                v = np.array([real(pick(c, s)) for c in flist]) - base
                cells.append(f"{100 * v.mean():+.1f} ({(v > 1e-9).sum()}+/{(v < -1e-9).sum()}−)")
            for op, lam in CANDIDATES:
                v = np.array([real(next(x for x in c if x["op"] == op and x["lambda_rel"] == lam)) for c in flist]) - base
                cells.append(f"{100 * v.mean():+.1f} ({(v > 1e-9).sum()}+/{(v < -1e-9).sum()}−)")
            print(f"| {d} | {len(flist)} | {base.mean():.3f} | {best.mean():.3f} (+{100 * (best - base).mean():.1f}) | " + " | ".join(cells) + " |")
    print("\n## Mean within-fit Spearman of each signal with real Stage-1 retention\n")
    print("| dataset | " + " | ".join(SIGNALS) + " |")
    print("| --- | " + " | ".join("---" for _ in SIGNALS) + " |")
    for d, flist in sorted(by_ds.items()):
        cors = defaultdict(list)
        for c in flist:
            y = np.array([x["stage1"]["rho_ceil"][metric] for x in c])
            for s in SIGNALS:
                x = np.array([z["stage1"]["signals"][s] for z in c])
                cors[s].append(_spearman(x, y))
        print(f"| {d} | " + " | ".join(f"{np.nanmean(cors[s]):+.2f}" for s in SIGNALS) + " |")
    print("\n## Which candidate the agree@10 signal picks vs. the best one (counts over fits)\n")
    for d, flist in sorted(by_ds.items()):
        name = lambda c: c["op"] if c["lambda_rel"] is None else f"ridge {c['lambda_rel']}"
        picked = defaultdict(int)
        best = defaultdict(int)
        for c in flist:
            picked[name(max(c, key=lambda x: x["stage1"]["signals"]["agree@10"]))] += 1
            best[name(max(c, key=lambda x: x["auto"]["rho_ceil"][metric]))] += 1
        print(f"- {d}: agree@10 picks {dict(picked)}; best after Stage 2 {dict(best)}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--K", type=int, nargs="*", default=[2048])
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--tiers", nargs="*", default=["1", "3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/m3_stage1_study.jsonl"))
    p.add_argument("--summarize", type=Path, nargs="*")
    p.add_argument("--metric", default="recall@30")
    args = p.parse_args()
    if args.summarize:
        summarize(args.summarize, args.metric)
    else:
        run(args)


if __name__ == "__main__":
    main()
