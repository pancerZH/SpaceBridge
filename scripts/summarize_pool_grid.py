"""W5: smallest rerank pool meeting the end-to-end targets at each index size, from Row 9 records.

    uv run python scripts/summarize_pool_grid.py outputs/phase1/w5_grid/*.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

TIER1 = {"bge-base-en", "bge-large-en"}
TARGETS = {"1": 0.98, "3": 0.94}


def tier(src: str, tgt: str) -> str:
    return "1" if {src, tgt} <= TIER1 else "3"


def main() -> None:
    cells = defaultdict(list)
    meta = {}
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            if "depth" not in r:
                continue
            key = (r["dataset"], r["K"], tier(r["src"], r["tgt"]), f"{r['src']}<-{r['tgt']}", r["depth"])
            cells[key].append(r["rho_ceil"]["ndcg@5"])
            meta[r["dataset"]] = (r["n_queries"],)
    datasets = sorted({k[:2] for k in cells})
    print(f"{'dataset':<22} {'K':>6} {'n_q':>5} {'tier':>4}  " + " ".join(f"{'p' + str(d):>7}" for d in (10, 20, 30, 60, 100, 300)) + "   worst@smallest  smallest pool (mean)")
    for ds, K in datasets:
        for t in ("1", "3"):
            pairs = sorted({k[3] for k in cells if k[:3] == (ds, K, t)})
            if not pairs:
                continue
            line, smallest, worst = [], None, None
            for pool in (10, 20, 30, 60, 100, 300):
                vals = [np.mean(cells[(ds, K, t, p, pool)]) for p in pairs if (ds, K, t, p, pool) in cells]
                if not vals:
                    line.append(f"{'-':>7}")
                    continue
                line.append(f"{np.mean(vals):7.3f}")
                if smallest is None and np.mean(vals) >= TARGETS[t]:
                    smallest, worst = pool, min(vals)
            print(f"{ds:<22} {K:>6} {meta[ds][0]:>5} {t:>4}  " + " ".join(line) + f"   {worst if worst else float('nan'):8.3f}        {smallest or '> 300'}")


if __name__ == "__main__":
    main()
