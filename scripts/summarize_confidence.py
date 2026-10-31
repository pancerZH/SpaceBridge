"""Per-query confidence (support -> isotonic agreement) vs. real per-query outcomes, from Row 8 records.

    uv run python scripts/summarize_confidence.py outputs/m2_fiqa_p4.jsonl outputs/m2_nq.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--row", default="8")
    args = p.parse_args()
    print("| dataset | fits | Spearman(support, overlap@30 w/ native) | overlap@30 Q1 → Q5 | ρceil R@30 Q1 → Q5 | fits with Q5 > Q1 retention |")
    print("| --- | --- | --- | --- | --- | --- |")
    for f in args.files:
        recs = [r for r in map(json.loads, f.open()) if "reranker" not in r and r["row"] == args.row and r.get("confidence")]
        if not recs:
            continue
        c = [r["confidence"] for r in recs]
        q = lambda key, i: np.mean([x["quintiles"][i][key] for x in c])
        up = sum(x["quintiles"][4]["rho_ceil_recall@30"] > x["quintiles"][0]["rho_ceil_recall@30"] for x in c)
        print(f"| {recs[0]['dataset']} | {len(c)} | {np.mean([x['spearman_support_overlap30'] for x in c]):.2f} | "
              f"{q('overlap@30_native', 0):.2f} → {q('overlap@30_native', 4):.2f} | "
              f"{q('rho_ceil_recall@30', 0):.3f} → {q('rho_ceil_recall@30', 4):.3f} | {up}/{len(c)} |")


if __name__ == "__main__":
    main()
