"""Paired comparison of two spacebridge-bench result files: same dataset, pairs, K, rows and seeds, one
setting changed (proxy style, an engine option, a code version). Per tier, mean ± s.d. over seeds.

    uv run python scripts/compare_runs.py outputs/m3_trec-covid_seeds.jsonl outputs/m3_trec-covid_title.jsonl
    uv run python scripts/compare_runs.py before.jsonl after.jsonl --rows 1 6 8 8b --detail
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.run import TIERS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--metric", default="recall@30")
    p.add_argument("--rows", nargs="*", default=["6", "8", "8b"])
    p.add_argument("--detail", action="store_true", help="also list every configuration with solver and Stage-2 step")
    args = p.parse_args()
    tier = {}
    for t, prs in TIERS.items():  # first tier wins: Tier 5a reuses Tier 3 pairs
        for pr in prs:
            tier.setdefault(pr, t)

    def load(path):  # identity (Row 1) is judged against native, everything else against the ceiling
        out = {}
        for line in path.open():
            r = json.loads(line)
            if "reranker" not in r and r["row"] in args.rows:
                key = (r["dataset"], tier.get((r["src"], r["tgt"]), "?"), r["K"], r["row"], r["src"], r["tgt"], r.get("seed", 0))
                out[key] = r  # last run wins
        return out

    ret = lambda r: (r["rho_native"] if r["row"] == "1" else r["rho_ceil"])[args.metric]
    base, new = load(args.base), load(args.new)
    groups = defaultdict(lambda: defaultdict(list))
    for k, r in new.items():
        if k in base:
            groups[k[:4]][k[6]].append((ret(base[k]), ret(r)))
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0
    print(f"| dataset | tier | K | row | base mean | new mean ± s.d. (seeds) | new − base, points | (pair, seed) cells + |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, seeds in sorted(groups.items()):
        b = [np.mean([x[0] for x in v]) for v in seeds.values()]
        n = [np.mean([x[1] for x in v]) for v in seeds.values()]
        d = [100 * (y - x) for x, y in zip(b, n)]
        cells = [y > x for v in seeds.values() for x, y in v]
        print(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {np.mean(b):.3f} | {np.mean(n):.3f} ± {sd(n):.3f} | "
              f"{np.mean(d):+.1f} ± {sd(d):.1f} | {sum(cells)}/{len(cells)} |")

    if args.detail:
        print("\n| index ← queries | K | row | seed | base | new | Δ | solver base → new | S2 step base → new |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in sorted(k for k in new if k in base):
            fa, fb = base[k].get("fit", {}), new[k].get("fit", {})
            va, vb = ret(base[k]), ret(new[k])
            print(f"| {k[4]} ← {k[5]} | {k[2]} | {k[3]} | {k[6]} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} | "
                  f"{fa.get('solver', '')} → {fb.get('solver', '')} | {fa.get('stage2_best_step', '')} → {fb.get('stage2_best_step', '')} |")


if __name__ == "__main__":
    main()
