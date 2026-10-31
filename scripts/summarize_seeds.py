"""Seed variance of spacebridge-bench results: per-row means and row-vs-row gains across seeds.

Each seed changes the anchors (k-means init), the train/val split and the distractor pool; the bootstrap
CIs in single runs cover only query sampling.

    uv run python scripts/summarize_seeds.py outputs/m2e_fiqa_seeds.jsonl --compare 6:8 8:8b 6:8b
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.run import TIERS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--metric", default="recall@30")
    p.add_argument("--compare", nargs="*", default=["6:8", "8:8b"], help="base:new row pairs")
    p.add_argument("--by-tier", action="store_true", help="split every table by tier (pairs as in run.TIERS)")
    args = p.parse_args()
    tier_of = {}
    for t, prs in TIERS.items():  # first tier wins: Tier 5a reuses Tier 3 pairs
        for pr in prs:
            tier_of.setdefault(pr, t)
    by = {}
    for f in args.results:
        for line in f.open():
            r = json.loads(line)
            if "reranker" in r:
                continue
            ds = r["dataset"] + (f" T{tier_of.get((r['src'], r['tgt']), '?')}" if args.by_tier else "")
            by[(ds, r["K"], r["row"], r["src"], r["tgt"], r["seed"])] = r  # last run wins

    def ret(r):  # identity is judged against native, everything else against the ceiling
        return (r["rho_native"] if r["row"] == "1" else r["rho_ceil"])[args.metric]

    cells = defaultdict(lambda: defaultdict(dict))  # (dataset, K, row) -> seed -> pair -> value
    for (ds, K, row, s, t, seed), r in by.items():
        if row not in ("0a", "0b"):
            cells[ds, K, row][seed][s, t] = ret(r)

    print(f"## Mean over pairs, {args.metric} retention (ρnative for Row 1, ρceil otherwise)\n")
    print("| dataset | K | row | seeds | per-seed means | mean ± s.d. over seeds | worst pair (any seed) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (ds, K, row), seeds in sorted(cells.items()):
        means = [np.mean(list(v.values())) for _, v in sorted(seeds.items())]
        worst = min(min(v.values()) for v in seeds.values())
        print(f"| {ds} | {K} | {row} | {len(means)} | {' / '.join(f'{m:.3f}' for m in means)} | "
              f"{np.mean(means):.3f} ± {np.std(means, ddof=1) if len(means) > 1 else 0:.3f} | {worst:.3f} |")

    print("\n## Gains (points), new − base\n")
    print("| dataset | K | new − base | per-seed mean gains | mean ± s.d. over seeds | (pair, seed) cells with gain > 0 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for ds, K in sorted({(d, k) for d, k, _ in cells}):
        for cmp in args.compare:
            base, new = cmp.split(":")
            if (ds, K, base) not in cells or (ds, K, new) not in cells:
                continue
            per_seed, diffs = [], []
            for seed, vb in sorted(cells[ds, K, base].items()):
                vn = cells[ds, K, new].get(seed, {})
                d = [100 * (vn[k] - vb[k]) for k in vb if k in vn]
                if d:
                    per_seed.append(np.mean(d))
                    diffs += d
            if per_seed:
                sd = np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0
                print(f"| {ds} | {K} | {new} − {base} | {' / '.join(f'{g:+.1f}' for g in per_seed)} | "
                      f"{np.mean(per_seed):+.1f} ± {sd:.1f} | {sum(x > 0 for x in diffs)}/{len(diffs)} |")


if __name__ == "__main__":
    main()
