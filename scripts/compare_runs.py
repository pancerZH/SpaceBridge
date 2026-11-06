"""Paired comparison of two spacebridge-bench result files: same dataset, pairs, K, rows and seeds, one
setting changed (proxy style, an engine option, a code version). Per tier, mean ± s.d. over seeds, the
median and a 95% bootstrap CI of the mean difference over (pair, seed) cells; ``--pool`` also pools each
dataset and row over tiers and K.

    uv run python scripts/compare_runs.py outputs/m3_trec-covid_seeds.jsonl outputs/m3_trec-covid_title.jsonl
    uv run python scripts/compare_runs.py before.jsonl after.jsonl --rows 1 6 8 8b --detail
    uv run python scripts/compare_runs.py outputs/closeout/ms50k_main.jsonl outputs/closeout/ms1m_main.jsonl --ignore-dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.provenance import unique_records

from spacebridge_bench.run import TIERS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--metric", default="recall@30")
    p.add_argument("--rows", nargs="*", default=["6", "8", "8b"])
    p.add_argument("--detail", action="store_true", help="also list every configuration with solver and Stage-2 step")
    p.add_argument("--pool", action="store_true", help="also pool each dataset and row over tiers and K")
    p.add_argument("--ignore-dataset", action="store_true",
                   help="pair configurations across datasets (e.g. one corpus subsampled at two sizes)")
    p.add_argument("--allow-duplicates", action="store_true", help="when a file repeats a cell, keep the last record instead of refusing")
    p.add_argument("--allow-unmatched", action="store_true", help="compare on the cells in both files when some are in only one")
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
                key = ("*" if args.ignore_dataset else r["dataset"], tier.get((r["src"], r["tgt"]), "?"), r["K"], r["row"], r["src"], r["tgt"], r.get("seed", 0))
                if key in out and not args.allow_duplicates:
                    raise ValueError(f"{path}: two records for {key}; pass --allow-duplicates to keep the last")
                out[key] = r
        return out

    ret = lambda r: (r["rho_native"] if r["row"] == "1" else r["rho_ceil"])[args.metric]
    base, new = load(args.base), load(args.new)
    unmatched = sorted(set(base) ^ set(new), key=str)
    if unmatched and not args.allow_unmatched:
        raise SystemExit(f"{len(unmatched)} cells are in only one file, e.g. {unmatched[0]}: the comparison would be over an incomplete set. "
                         "Pass --allow-unmatched to compare on the cells in both")
    if unmatched:
        print(f"Compared on cells in both files only: {len(unmatched)} cells are in just one of them.\n")
    protocol = ("proxy", "doc_pair_tokens", "anchor_select", "stage2_rule", "anchor_bank", "shift_bank")
    for name, side in (("base", base), ("new", new)):
        try:
            unique_records(list(side.values()), lambda r: id(r), protocol=protocol)
        except ValueError as e:
            raise SystemExit(f"the {name} file mixes protocols: {e}")
    groups = defaultdict(lambda: defaultdict(list))
    for k, r in new.items():
        if k in base:
            groups[k[:4]][k[6]].append((ret(base[k]), ret(r)))
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else 0.0
    rng = np.random.default_rng(0)

    def cell_ci(diffs):  # 95% bootstrap CI of the mean over (pair, seed) cells, points
        diffs = np.asarray(diffs)
        boot = diffs[rng.integers(0, len(diffs), (2000, len(diffs)))].mean(1)
        return np.percentile(boot, [2.5, 97.5])

    def row(label, seeds):
        b = [np.mean([x[0] for x in v]) for v in seeds.values()]
        n = [np.mean([x[1] for x in v]) for v in seeds.values()]
        d = [100 * (y - x) for x, y in zip(b, n)]
        cells = [100 * (y - x) for v in seeds.values() for x, y in v]
        lo, hi = cell_ci(cells)
        print(f"| {label} | {np.mean(b):.3f} | {np.mean(n):.3f} ± {sd(n):.3f} | {np.mean(d):+.1f} ± {sd(d):.1f} | "
              f"{np.median(cells):+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sum(c > 0 for c in cells)}/{len(cells)} |")

    print(f"| dataset | tier | K | row | base mean | new mean ± s.d. (seeds) | new − base, points | median cell | "
          f"bootstrap over this table's cells (not a CI for new queries or corpora) | (pair, seed) cells + |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for g, seeds in sorted(groups.items()):
        row(f"{g[0]} | {g[1]} | {g[2]} | {g[3]}", seeds)
    if args.pool:
        pooled = defaultdict(lambda: defaultdict(list))
        for g, seeds in groups.items():
            for s_, v in seeds.items():
                pooled[g[0], g[3]][s_].extend(v)
        for (ds, r), seeds in sorted(pooled.items()):
            row(f"{ds} | all | all | {r}", seeds)

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
