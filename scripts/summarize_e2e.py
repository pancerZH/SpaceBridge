"""End-to-end table from Row 9 records: nDCG@5 retention per pair and pool, mean ± s.d. over seeds, worst pair, CIs.

    uv run python scripts/summarize_e2e.py outputs/phase1/w3_1m/ms1m_e2e.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np

TIER1 = {"bge-base-en", "bge-large-en"}


def tier(src: str, tgt: str) -> str:
    return "1" if {src, tgt} <= TIER1 else "3"


def main() -> None:
    rows: dict = defaultdict(list)
    first: dict = {}
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            key = (r["dataset"], r["src"], r["tgt"], r["K"])
            if r["row"] == "8b":
                first.setdefault(key, []).append(r)
            elif "depth" in r:
                rows[key + (r["depth"],)].append(r)
    print(f"{'dataset':<14} {'index <- queries':<30} {'K':>6} {'pool':>5} {'n':>2}  {'rho_ceil nDCG@5':<15} {'CI of 1st seed':<16} {'bridge':>6} {'native':>6} {'reindex':>7}  rho_oracle  R@pool(first stage)")
    groups: dict = defaultdict(list)
    for key in sorted(rows, key=lambda k: (k[0], tier(k[1], k[2]), k[1], k[2], k[3], k[4])):
        ds, src, tgt, K, depth = key
        rs = rows[key]
        rho = np.array([r["rho_ceil"]["ndcg@5"] for r in rs])
        groups[(ds, tier(src, tgt), K, depth)].append((rho.mean(), f"{src}<-{tgt}", rho))
        fs = first.get((ds, src, tgt, K), [])
        rec = np.mean([f["metrics"].get(f"recall@{depth}", np.nan) for f in fs]) if fs else np.nan
        rec_ret = np.mean([f["rho_ceil"].get(f"recall@{depth}", np.nan) for f in fs]) if fs else np.nan
        lo, hi = rs[0]["ci_ceil"]["ndcg@5"]
        b = rs[0]["baselines"]
        print(
            f"{ds:<14} {src + ' <- ' + tgt:<30} {K:>6} {depth:>5} {len(rs):>2}  {rho.mean():.4f} ± {rho.std(ddof=1) if len(rho) > 1 else 0:.4f}  [{lo:.3f},{hi:.3f}]  "
            f"{np.mean([r['metrics']['ndcg@5'] for r in rs]):6.3f} {b['native']['ndcg@5']:6.3f} {b['oracle']['ndcg@5']:7.3f}  {np.mean([r['rho_oracle']['ndcg@5'] for r in rs]):8.3f}  {rec:.3f} (ret {rec_ret:.3f})"
        )
    print(f"\n{'dataset':<14} {'tier':>4} {'K':>6} {'pool':>5}  mean over pairs (mean ± s.d. over seeds)  worst pair")
    for (ds, t, K, depth), vals in sorted(groups.items()):
        seeds = np.mean([v[2] for v in vals if len(v[2]) == len(vals[0][2])], axis=0)
        worst = min(vals, key=lambda v: v[0])
        print(f"{ds:<14} {t:>4} {K:>6} {depth:>5}  {seeds.mean():.4f} ± {seeds.std(ddof=1) if len(seeds) > 1 else 0:.4f}   {worst[0]:.4f} ({worst[1]})")


if __name__ == "__main__":
    main()
