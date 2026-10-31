"""Default-engine status against the approved first-stage targets (design §2.1), from spacebridge-bench JSONL.

Uses Row 8b (Row 7b for cross-lingual sets), de-duplicated per (pair, K, seed) with the last run winning;
mean ± s.d. over seeds of the mean over pairs, and the worst pair (its mean over seeds).

    uv run python scripts/target_status.py outputs/m3_trec-covid_title.jsonl outputs/m3_t2_default.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.run import TIERS

TARGETS = {("1", 2048): 0.88, ("1", 4096): 0.90, ("2", 2048): 0.88, ("2", 4096): 0.88, ("3", 2048): 0.82,
           ("3", 4096): 0.87, ("4a", 2048): 0.80, ("4a", 4096): 0.80, ("5a", 2048): 0.78, ("5a", 4096): 0.78}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--K", type=int, nargs="*", help="only these anchor counts")
    args = p.parse_args()
    tier = {}
    for t, prs in TIERS.items():  # first tier wins: Tier 5a reuses Tier 3 pairs
        for pr in prs:
            tier.setdefault(pr, t)
    recs = {}
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if "reranker" in r or r["row"] != ("7b" if r["dataset"].startswith("xquad") else "8b"):
                continue
            if args.K and r["K"] not in args.K:
                continue
            t = "5a" if r["dataset"].startswith("xquad") else tier.get((r["src"], r["tgt"]), "?")
            recs[r["dataset"], t, r["K"], r["src"], r["tgt"], r["seed"]] = r
    groups = defaultdict(lambda: defaultdict(dict))
    for (ds, t, K, s, q, seed), r in recs.items():
        groups[ds, t, K][s, q][seed] = r
    print("| dataset | tier | K | `R@30` ρceil mean ± s.d. | worst pair | target | met: mean / every pair | `R@100` worst pair |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (ds, t, K), pairs in sorted(groups.items()):
        seeds = sorted({s for v in pairs.values() for s in v})
        per_seed = [np.mean([v[s]["rho_ceil"]["recall@30"] for v in pairs.values() if s in v]) for s in seeds]
        pair_mean = {pr: np.mean([r["rho_ceil"]["recall@30"] for r in v.values()]) for pr, v in pairs.items()}
        worst = min(pair_mean, key=pair_mean.get)
        r100 = min(np.mean([r["rho_ceil"]["recall@100"] for r in v.values()]) for v in pairs.values())
        tg = TARGETS.get((t, K))
        met = f"{'✓' if np.mean(per_seed) >= tg else '✗'} / {'✓' if pair_mean[worst] >= tg else '✗'}" if tg else ""
        sd = np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0
        print(f"| {ds} | {t} | {K} | {np.mean(per_seed):.3f} ± {sd:.3f} | {pair_mean[worst]:.3f} ({worst[0]} ← {worst[1]}) | "
              f"{tg or ''} | {met} | {r100:.3f} |")


if __name__ == "__main__":
    main()
