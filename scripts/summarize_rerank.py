"""Markdown tables for the end-to-end rerank rows (9n / 9 / 9w) of spacebridge-bench JSONL results.

    uv run python scripts/summarize_rerank.py outputs/fiqa_m2_rerank.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

RERANK_ROWS = ("9n", "9", "9w")
FIRST_STAGE = ("6", "8")


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--K", type=int, default=2048)
    args = p.parse_args()
    recs = [json.loads(line) for f in args.results for line in f.open()]
    recs = [r for r in recs if r["K"] == args.K]
    by = {(r["dataset"], r["src"], r["tgt"], r["row"], r.get("reranker")): r for r in recs}  # last run wins
    pairs = list(dict.fromkeys((r["dataset"], r["src"], r["tgt"]) for r in recs if r["row"] in RERANK_ROWS))
    rerankers = list(dict.fromkeys(r["reranker"] for r in recs if r["row"] in RERANK_ROWS))

    for rr in rerankers:
        print(f"\n## {rr}: end-to-end nDCG@5 (K = {args.K})\n")
        rows = []
        for ds, s, t in pairs:
            first = by.get((ds, s, t, "8", None))
            cells = [ds, f"{s} ← {t}"]
            cells.append(f"{first['rho_ceil']['ndcg@5']:.3f}" if first else "")
            for row in RERANK_ROWS:
                r = by.get((ds, s, t, row, rr))
                if r is None:
                    cells.append("")
                    continue
                lo, hi = r["ci_ceil"]["ndcg@5"]
                cells.append(f"{r['rho_ceil']['ndcg@5']:.3f} [{lo:.2f},{hi:.2f}]")
            r9 = by.get((ds, s, t, "9", rr))
            if r9:
                b = r9["baselines"]
                cells += [f"{b['native']['ndcg@5']:.3f} / {b['oracle']['ndcg@5']:.3f} / {r9['metrics']['ndcg@5']:.3f}",
                          f"{r9['rho_ceil']['recall@5']:.3f}", f"{r9['jaccard@5']['bridge_vs_oracle']:.2f} ({r9['jaccard@5']['native_vs_oracle']:.2f})"]
            rows.append(cells)
        means = ["**mean**", ""]
        first = [by[(ds, s, t, "8", None)]["rho_ceil"]["ndcg@5"] for ds, s, t in pairs if (ds, s, t, "8", None) in by]
        means.append(f"**{np.mean(first):.3f}**" if first else "")
        for row in RERANK_ROWS:
            v = [by[(ds, s, t, row, rr)]["rho_ceil"]["ndcg@5"] for ds, s, t in pairs if (ds, s, t, row, rr) in by]
            means.append(f"**{np.mean(v):.3f}**" if v else "")
        rows.append(means)
        print(table(
            ["dataset", "index ← queries", "Row 8 first stage", "9n (top-30)", "9 (top-60)", "9w (top-100)",
             "nDCG@5 native / oracle / bridge (top-60)", "ρceil R@5 (top-60)", "J@5 bridge–oracle (native–oracle)"],
            rows,
        ))

    print(f"\n## Reranked baselines and latency (K = {args.K})\n")
    rows = []
    for rr in rerankers:
        for row in RERANK_ROWS:
            rs = [by[(ds, s, t, row, rr)] for ds, s, t in pairs if (ds, s, t, row, rr) in by]
            if not rs:
                continue
            rows.append([
                rr, row, str(rs[0]["depth"]),
                f"{np.mean([r['rho_ceil']['ndcg@5'] for r in rs]):.3f}",
                f"{min(r['rho_ceil']['ndcg@5'] for r in rs):.3f}",
                f"{np.mean([r['rho_ceil']['recall@5'] for r in rs]):.3f}",
                f"{np.mean([r['rho_ceil']['ndcg@10'] for r in rs]):.3f}",
                f"{np.mean([r['rerank_ms_per_query'] for r in rs]):.0f}",
            ])
    print(table(["reranker", "row", "depth", "mean ρceil nDCG@5", "worst pair", "mean ρceil R@5", "mean ρceil nDCG@10", "ms / query"], rows))


if __name__ == "__main__":
    main()
