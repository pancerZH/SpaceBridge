"""A4: the rows' merge and rerank against the study's records, on the study's own fits. Any difference left is the code's.

The rows fit with positives allowed among the anchors, and the study didn't, so the rows themselves differ a little. This takes the
fit out: the study's ``Ctx`` gives the first stage, then ``merge_pool`` and ``rerank_pools`` give the result. Scores are cached.

    uv run python -u scripts/lexical_regress.py --out outputs/phase1/m3_lib/regress.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hotpot_study import DEV, Ctx  # noqa: E402

from spacebridge import Index, merge_pool  # noqa: E402
from spacebridge.lexical import LEXICAL_SHARE  # noqa: E402
from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.metrics import per_query  # noqa: E402
from spacebridge_bench.rerank import RERANKERS, Reranker, rerank_pools  # noqa: E402
from spacebridge_bench.run import _DocTexts, cached_lexical  # noqa: E402
from spacebridge_bench.search import search  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="hotpotqa-1m")
    p.add_argument("--tiers", nargs="+", default=["T1", "T3"])
    p.add_argument("--budgets", type=int, nargs="+", default=[60, 300])
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study", type=Path, default=Path("outputs/phase1/hp_fixes/pools.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    records = {(r["tier"], r["budget"]): r for r in durable.read_jsonl(args.study)[0]
               if r["corpus"] == args.corpus and r["seed"] == args.seed and r["queries"] == args.queries and r["config"] == "bm25_a" and r.get("K", 8192) == 8192}
    rr = Reranker(RERANKERS["bge-reranker-v2-m3"], DEV)
    for tier in args.tiers:
        rr.unload()
        c = Ctx(Path("outputs/phase1/hp_study"), args.corpus, tier, args.seed)
        ds, n = c.ds, min(args.queries, len(c.ds.queries))
        lexical = cached_lexical(ds, ds.index_over(c.E_src))
        bridged = search(c.Q_hat[:n], c.E_src, 1000, DEV, exclude=ds.self_rows[:n])
        docs = _DocTexts(ds.titles, ds.texts)
        for budget in args.budgets:
            share = budget // LEXICAL_SHARE
            pools = [merge_pool(r, lexical.top(q, share, exclude=s), budget) for r, q, s in zip(bridged, ds.queries[:n], ds.self_rows[:n].tolist())]
            pq = per_query(rerank_pools(rr, pools, ds.queries[:n], docs, ds.task), ds.qrels[:n])
            got, want = float(pq["ndcg@5"].mean()), records[(tier, budget)]["ndcg5"]
            rec = {"corpus": args.corpus, "tier": tier, "budget": budget, "queries": n, "ndcg5": got, "ndcg5_study": want, "difference_points": 100 * (got - want), "pool_mean": float(np.mean([len(x) for x in pools]))}
            durable.append_lines(args.out, [json.dumps(rec)])
            durable.progress(f"{tier} budget {budget}: {got:.6f} against the study's {want:.6f}, {rec['difference_points']:+.3f} points")
    rr.unload()


if __name__ == "__main__":
    main()
