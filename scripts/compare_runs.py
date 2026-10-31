"""Paired comparison of two spacebridge-bench result files (same dataset, pairs, K and rows).

    uv run python scripts/compare_runs.py outputs/fiqa_v2.jsonl outputs/m2_fiqa.jsonl --rows 1 6 8 8b
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    recs = [json.loads(line) for line in path.open()]
    return {(r["src"], r["tgt"], r["K"], r["row"]): r for r in recs if "reranker" not in r}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--rows", nargs="*", default=["1", "6", "8", "8b"])
    p.add_argument("--metric", default="recall@30")
    args = p.parse_args()
    a, b = load(args.before), load(args.after)
    keys = sorted(k for k in a.keys() & b.keys() if k[3] in args.rows)

    print(f"| index ← queries | K | row | before ρceil {args.metric} | after | Δ | solver before → after | S2 step before → after | solver ms before → after |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    deltas: dict[str, list[float]] = {}
    for k in keys:
        ra, rb = a[k], b[k]
        va, vb = ra["rho_ceil"][args.metric], rb["rho_ceil"][args.metric]
        deltas.setdefault(k[3], []).append(vb - va)
        fa, fb = ra["fit"], rb["fit"]
        ms = lambda f: f.get("stage1_ms", 0) + f.get("stage2_ms", 0) + f.get("val_ms", 0)
        print(
            f"| {k[0]} ← {k[1]} | {k[2]} | {k[3]} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} | {fa['solver']} → {fb['solver']} | "
            f"{fa.get('stage2_best_step', '')} → {fb.get('stage2_best_step', '')} | {ms(fa):.0f} → {ms(fb):.0f} |"
        )
    print()
    for row, d in deltas.items():
        print(f"- row {row}: mean Δ {np.mean(d):+.4f}, max |Δ| {np.max(np.abs(d)):.3f} over {len(d)} configurations")


if __name__ == "__main__":
    main()
