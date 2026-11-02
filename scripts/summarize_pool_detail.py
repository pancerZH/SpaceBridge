"""End-to-end retention by pool, per pair and per tier, from Row 9 records: the mean over 3 seeds, the worst pair, a 95% CI on
each pair (the seeds' query-bootstrap bounds, averaged) and the absolute nDCG@5. "Passes" is on the tier mean, as the targets
define it, and the smallest passing pool is only ever the smallest pool tested.

Records of the same cell may come from several files: that is the regression check. The largest disagreement is printed.

    uv run python scripts/summarize_pool_detail.py outputs/phase1/w3_1m/*.jsonl outputs/phase1/m2_pools/*.jsonl --pools 100 150 200 300
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TIER1 = {"bge-base-en", "bge-large-en"}
TARGETS = {"1": 0.98, "3": 0.94}
RERANKER = "bge-reranker-v2-m3"


def tier(src: str, tgt: str) -> str:
    return "1" if {src, tgt} <= TIER1 else "3"


def load(paths: list[Path]) -> tuple[dict, float, int]:
    """(dataset, K, tier, pair, pool) -> {seed: record}, the largest difference between two records of a cell, in points."""
    cells, worst, repeats = defaultdict(dict), 0.0, 0
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            if r.get("reranker") != RERANKER:
                continue
            key = (r["dataset"], r["K"], tier(r["src"], r["tgt"]), f"{r['src']} ← {r['tgt']}", r["depth"])
            old = cells[key].get(r["seed"])
            if old is not None:
                repeats += 1
                worst = max(worst, 100 * abs(old["rho_ceil"]["ndcg@5"] - r["rho_ceil"]["ndcg@5"]))
            cells[key][r["seed"]] = r
    return cells, worst, repeats


def pair_cell(seeds: dict) -> dict:
    rs = list(seeds.values())
    vals = [r["rho_ceil"]["ndcg@5"] for r in rs]
    return {
        "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "seeds": len(vals),
        "lo": float(np.mean([r["ci_ceil"]["ndcg@5"][0] for r in rs])), "hi": float(np.mean([r["ci_ceil"]["ndcg@5"][1] for r in rs])),
        "abs": float(np.mean([r["metrics"]["ndcg@5"] for r in rs])),
        "n_q": rs[0]["n_queries"], "anchor": rs[0].get("anchor_select"),
        "native": float(np.mean([r["baselines"]["native"]["ndcg@5"] for r in rs])),
        "oracle": float(np.mean([r["baselines"]["oracle"]["ndcg@5"] for r in rs])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--pools", type=int, nargs="+", default=[60, 100, 150, 200, 300])
    args = p.parse_args()

    cells, worst, repeats = load(args.files)
    pools = args.pools
    print(f"Regression check: {repeats} records seen twice; largest difference {worst:.3f} points of nDCG@5 retention\n")
    for ds, K in sorted({k[:2] for k in cells}):
        for t in ("1", "3"):
            pairs = sorted({k[3] for k in cells if k[:3] == (ds, K, t)})
            if not pairs:
                continue
            info = next(pair_cell(cells[k]) for k in cells if k[:3] == (ds, K, t))
            print(f"### {ds}, K = {K:,} {info['anchor']}, Tier {t}, {info['n_q']} queries; line {TARGETS[t]}")
            print("| Index ← query | " + " | ".join(f"Pool {q}" for q in pools) + f" | Absolute at {pools[-1]} (native, re-index) | s.d. at {pools[-1]} |")
            print("| --- | " + " | ".join("---" for _ in pools) + " | --- | --- |")
            found = {q: {pr: pair_cell(cells[(ds, K, t, pr, q)]) for pr in pairs if (ds, K, t, pr, q) in cells} for q in pools}
            for pr in pairs:
                row = [f"{found[q][pr]['mean']:.3f} [{found[q][pr]['lo']:.3f}, {found[q][pr]['hi']:.3f}]" if pr in found[q] else "-" for q in pools]
                last = found[pools[-1]].get(pr)
                extra = f"{last['abs']:.3f} ({last['native']:.3f}, {last['oracle']:.3f}) | {last['sd']:.3f}" if last else "- | -"
                print(f"| `{pr}` | " + " | ".join(row) + f" | {extra} |")
            means = {q: np.mean([c["mean"] for c in found[q].values()]) for q in pools if found[q]}
            print("| **Mean** | " + " | ".join(f"**{means[q]:.3f}**" if q in means else "-" for q in pools) + " | | |")
            print("| Worst pair | " + " | ".join(f"{min(c['mean'] for c in found[q].values()):.3f}" if found[q] else "-" for q in pools) + " | | |")
            print("| Margin, points | " + " | ".join(f"{100 * (means[q] - TARGETS[t]):+.1f}" if q in means else "-" for q in pools) + " | | |")
            passing = [q for q in pools if q in means and means[q] >= TARGETS[t]]
            print(f"\nSmallest pool tested that passes on the mean: {passing[0] if passing else 'none tested'}\n")


if __name__ == "__main__":
    main()
