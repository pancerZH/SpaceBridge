"""W3: first-stage retention by index size and K from scale_curve.py records: mean over pairs and seeds, by tier.

    uv run python scripts/summarize_curve.py outputs/phase1/m1_nested/curve_ms.jsonl outputs/phase1/m1_nested/curve_nq.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

TIER1 = {"bge-base-en", "bge-large-en"}
DEPTHS = ("recall@30", "recall@100", "recall@300", "recall@1000")


def tier(src: str, tgt: str) -> str:
    return "1" if {src, tgt} <= TIER1 else "3"


def main() -> None:
    cells = defaultdict(list)
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            cells[r["dataset"], tier(r["src"], r["tgt"]), r["K"], r["size"], r["src"], r["tgt"]].append(r)

    groups = defaultdict(dict)
    for (dataset, t, K, size, src, tgt), rows in cells.items():
        groups[dataset, t, K, size][src, tgt] = rows

    print(f"{'dataset':<20}{'tier':<5}{'K':>6}{'size':>9}  " + "  ".join(f"{d.replace('recall@', 'R@'):>7}" for d in DEPTHS)
          + f"  {'worst R@300':>11}  {'s.d.':>6}  {'overlap@30':>10}  {'proxy':>6}")
    for (dataset, t, K, size), pairs in sorted(groups.items()):
        pair_means = {d: [np.mean([r["rho_ceil"][d] for r in rows]) for rows in pairs.values()] for d in DEPTHS}
        sds = [np.std([r["rho_ceil"]["recall@300"] for r in rows], ddof=1) for rows in pairs.values()]
        everyone = [r for rows in pairs.values() for r in rows]
        print(f"{dataset:<20}{t:<5}{K:>6}{size:>9}  " + "  ".join(f"{np.mean(pair_means[d]):>7.3f}" for d in DEPTHS)
              + f"  {min(pair_means['recall@300']):>11.3f}  {max(sds):>6.3f}"
              + f"  {np.mean([r['overlap']['30'] for r in everyone]):>10.3f}  {np.mean([r['agree_val_full'] for r in everyone]):>6.3f}")


if __name__ == "__main__":
    main()
