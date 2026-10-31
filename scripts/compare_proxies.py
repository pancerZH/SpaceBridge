"""Paired comparison of query-proxy styles (spacebridge-bench --proxy), per tier, mean ± s.d. over seeds.

    uv run python scripts/compare_proxies.py outputs/m3_trec-covid_seeds.jsonl outputs/m3_trec-covid_title.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.run import TIERS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path, help="results with the base proxy style (usually lead)")
    p.add_argument("new", type=Path)
    p.add_argument("--metric", default="recall@30")
    p.add_argument("--rows", nargs="*", default=["6", "8", "8b"])
    args = p.parse_args()
    tier = {}
    for t, prs in TIERS.items():
        for pr in prs:
            tier.setdefault(pr, t)

    def load(path):
        out = {}
        for line in path.open():
            r = json.loads(line)
            if "reranker" not in r and r["row"] in args.rows:
                out[r["dataset"], tier[r["src"], r["tgt"]], r["K"], r["row"], r["src"], r["tgt"], r["seed"]] = r["rho_ceil"][args.metric]
        return out

    base, new = load(args.base), load(args.new)
    groups = defaultdict(lambda: defaultdict(list))
    for k, v in new.items():
        if k in base:
            groups[k[:4]][k[6]].append((base[k], v))
    print(f"| dataset | tier | K | row | base mean | new mean ± s.d. (seeds) | new − base (points) | cells + |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, seeds in sorted(groups.items()):
        b = [np.mean([x[0] for x in v]) for v in seeds.values()]
        n = [np.mean([x[1] for x in v]) for v in seeds.values()]
        d = [100 * (y - x) for x, y in zip(b, n)]
        cells = [y > x for v in seeds.values() for x, y in v]
        sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0
        print(f"| {g[0]} | {g[1]} | {g[2]} | {g[3]} | {np.mean(b):.3f} | {np.mean(n):.3f} ± {sd(n):.3f} | "
              f"{np.mean(d):+.1f} ± {sd(d):.1f} | {sum(cells)}/{len(cells)} |")


if __name__ == "__main__":
    main()
