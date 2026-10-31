"""Default-engine status against the approved first-stage targets (design §2.1), from spacebridge-bench JSONL.

Uses Row 8b (Row 7b for cross-lingual sets), de-duplicated per (pair, K, seed) with the last run winning;
mean ± s.d. over seeds of the mean over pairs, and the worst pair (its mean over seeds). ``--e2e`` reports the
rerank rows (9n / 9 / 9w: bridged top-30 / 60 / 100 + cross-encoder) on `nDCG@5` against the end-to-end targets.

    uv run python scripts/target_status.py outputs/m3_trec-covid_title.jsonl outputs/m3_t2_default.jsonl
    uv run python scripts/target_status.py outputs/m4_b3_*.jsonl --e2e
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.run import TIERS

TARGETS = {("1", 2048): 0.88, ("1", 4096): 0.90, ("2", 2048): 0.88, ("2", 4096): 0.88, ("3", 2048): 0.82,
           ("3", 4096): 0.87, ("4a", 2048): 0.80, ("4a", 4096): 0.80, ("5a", 2048): 0.78, ("5a", 4096): 0.78}
E2E_TARGETS = {"1": 0.98, "2": 0.96, "3": 0.94, "4a": 0.92, "5a": 0.90}  # nDCG@5 ρceil after reranking


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--K", type=int, nargs="*", help="only these anchor counts")
    p.add_argument("--e2e", action="store_true", help="end-to-end rerank rows on nDCG@5 instead of the first stage")
    args = p.parse_args()
    metric = "ndcg@5" if args.e2e else "recall@30"
    tier = {}
    for t, prs in TIERS.items():  # first tier wins: Tier 5a reuses Tier 3 pairs
        for pr in prs:
            tier.setdefault(pr, t)
    recs, first_stage = {}, {}
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if args.K and r["K"] not in args.K:
                continue
            t = "5a" if r["dataset"].startswith("xquad") else tier.get((r["src"], r["tgt"]), "?")
            default_row = "7b" if r["dataset"].startswith("xquad") else "8b"
            if "reranker" not in r and r["row"] == default_row:
                first_stage[r["dataset"], t, r["K"], r["src"], r["tgt"], r["seed"]] = r
            if args.e2e:
                if "reranker" not in r:
                    continue
                view = f"{r['row']} (top-{r['depth']}, {r['reranker']})"
            elif "reranker" in r or r["row"] != default_row:
                continue
            else:
                view = r["row"]
            recs[r["dataset"], t, r["K"], view, r["src"], r["tgt"], r["seed"]] = r
    groups = defaultdict(lambda: defaultdict(dict))
    for (ds, t, K, view, s, q, seed), r in recs.items():
        groups[ds, t, K, view][s, q][seed] = r
    m = "`nDCG@5`" if args.e2e else "`R@30`"
    print(f"| dataset | tier | K | row | {m} ρceil mean ± s.d. | worst pair | target | met: mean / every pair | "
          f"{'first-stage `nDCG@5` worst pair' if args.e2e else '`R@100` worst pair'} |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (ds, t, K, view), pairs in sorted(groups.items()):
        seeds = sorted({s for v in pairs.values() for s in v})
        per_seed = [np.mean([v[s]["rho_ceil"][metric] for v in pairs.values() if s in v]) for s in seeds]
        pair_mean = {pr: np.mean([r["rho_ceil"][metric] for r in v.values()]) for pr, v in pairs.items()}
        worst = min(pair_mean, key=pair_mean.get)
        if args.e2e:  # the first stage the reranker saw (Row 8b / 7b records of the same runs), worst pair
            first = [[first_stage[ds, t, K, pr[0], pr[1], sd]["rho_ceil"]["ndcg@5"] for sd in v
                      if (ds, t, K, pr[0], pr[1], sd) in first_stage] for pr, v in pairs.items()]
            extra = min((np.mean(f) for f in first if f), default=np.nan)
        else:
            extra = min(np.mean([r["rho_ceil"]["recall@100"] for r in v.values()]) for v in pairs.values())
        tg = E2E_TARGETS.get(t) if args.e2e else TARGETS.get((t, K))
        met = f"{'✓' if np.mean(per_seed) >= tg else '✗'} / {'✓' if pair_mean[worst] >= tg else '✗'}" if tg else ""
        sd = np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0
        print(f"| {ds} | {t} | {K} | {view} | {np.mean(per_seed):.3f} ± {sd:.3f} | {pair_mean[worst]:.3f} ({worst[0]} ← {worst[1]}) | "
              f"{tg or ''} | {met} | {extra:.3f} |")



if __name__ == "__main__":
    main()
