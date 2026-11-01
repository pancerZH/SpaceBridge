"""Table of source-free variants: R@30 retention by pair, mean ± s.d. over seeds, plus the mean over pairs."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        rows[(r["dataset"], r["variant"], r["pair"])].append(r)
datasets = sorted({k[0] for k in rows})
for ds in datasets:
    pairs = sorted({k[2] for k in rows if k[0] == ds})
    print(f"\n{ds}: R@30 retention (s.d. over seeds); last columns: mean over pairs of R@30, R@100, nDCG@10; own-doc MRR")
    print(f"{'variant':<30} " + " ".join(f"{p.replace('bge-', '').replace('-en', ''):>20}" for p in pairs))
    for v in sorted({k[1] for k in rows if k[0] == ds}):
        cells, allv = [], {"recall@30": [], "recall@100": [], "ndcg@10": [], "mrr": []}
        for p in pairs:
            rs = rows.get((ds, v, p), [])
            if not rs:
                cells.append(f"{'-':>20}")
                continue
            vals = np.array([r["rho"]["recall@30"] for r in rs])
            cells.append(f"{vals.mean():>12.3f} ±{vals.std(ddof=1) if len(vals) > 1 else 0:5.3f} ")
            for m in ("recall@30", "recall@100", "ndcg@10"):
                allv[m].append(np.mean([r["rho"][m] for r in rs]))
            allv["mrr"].append(np.mean([r["owndoc_mrr"] for r in rs]))
        print(f"{v:<30} " + " ".join(cells) + f"  {np.mean(allv['recall@30']):.3f} {np.mean(allv['recall@100']):.3f} {np.mean(allv['ndcg@10']):.3f}  {np.mean(allv['mrr']):.3f}")
