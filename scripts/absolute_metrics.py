"""Absolute first-stage metrics next to retention: per (dataset, pair, K), the native (Row 0b), re-index (Row 0a)
and bridged rows, mean over seeds. Retention hides how hard a corpus is; this shows it (e.g. one corpus
subsampled at 50K and at 1M passages, where the ceilings differ).

    uv run python scripts/absolute_metrics.py outputs/closeout/ms50k_main.jsonl outputs/closeout/ms1m_main.jsonl --rows 8b
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.provenance import unique_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--rows", nargs="*", default=["8b"])
    p.add_argument("--metrics", nargs="*", default=["recall@30", "ndcg@10"])
    args = p.parse_args()
    cells = defaultdict(lambda: defaultdict(list))  # (dataset, src, tgt, K) -> row -> [record]
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if "reranker" not in r and r["row"] in ("0a", "0b", *args.rows):
                bucket = cells[r["dataset"], r["src"], r["tgt"], r["K"]][r["row"]]
                if any(x.get("seed", 0) == r.get("seed", 0) for x in bucket):
                    raise ValueError(f"{path}: two records for {r['dataset']}, {r['src']} <- {r['tgt']}, K {r['K']}, row {r['row']}, seed {r.get('seed', 0)}")
                bucket.append(r)
    protocol = ("proxy", "doc_pair_tokens", "anchor_select", "stage2_rule", "anchor_bank", "shift_bank")
    for rows in cells.values():  # each table row compares one protocol's records
        for recs in rows.values():
            unique_records(recs, lambda r: r.get("seed", 0), protocol=protocol)
    for m in args.metrics:
        cols = ["native", "re-index"] + [f"Row {r}" for r in args.rows] + [f"ρceil {r}" for r in args.rows] + [f"ρnative {r}" for r in args.rows]
        print(f"\n### {m} (mean over seeds)\n")
        print("| dataset | index ← queries | K | " + " | ".join(cols) + " |")
        print("| --- | --- | --- | " + " | ".join("---" for _ in cols) + " |")
        for (ds, src, tgt, K), rows in sorted(cells.items()):
            mean = lambda row, f: np.mean([f(r) for r in rows[row]]) if rows.get(row) else float("nan")
            vals = [mean("0b", lambda r: r["metrics"][m]), mean("0a", lambda r: r["metrics"][m])]
            vals += [mean(row, lambda r: r["metrics"][m]) for row in args.rows]
            vals += [mean(row, lambda r: r["rho_ceil"][m]) for row in args.rows]
            vals += [mean(row, lambda r: r["rho_native"][m]) for row in args.rows]
            print(f"| {ds} | {src} ← {tgt} | {K} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")


if __name__ == "__main__":
    main()
