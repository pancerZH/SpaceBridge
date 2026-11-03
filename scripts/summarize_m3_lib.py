"""M3 Part A's regression tables: rows 9h and 9hx against the study, row 9x against M2, Searcher against the row, budget at K = 4,096.

    uv run python scripts/summarize_m3_lib.py > outputs/phase1/m3_tables/a_lib.md
"""
import json
from pathlib import Path

from spacebridge_bench import durable

ROOT = Path("outputs/phase1")
PAIRS = {("bge-large-en", "bge-base-en"): "T1", ("bge-base-en", "bge-large-en"): "T1b", ("bge-large-en", "qwen3-0.6b"): "T3", ("qwen3-0.6b", "bge-large-en"): "T3b"}
BUDGET = {"9h": 60, "9hx": 300}


def read(path: str) -> list[dict]:
    return durable.read_jsonl(ROOT / path)[0]


def study(queries: int) -> dict:
    return {(r["tier"], r["budget"]): r for r in read("hp_fixes/pools.jsonl")
            if r["corpus"] == "hotpotqa-1m" and r["seed"] == 0 and r["queries"] == queries and r["config"] == "bm25_a" and r.get("K", 8192) == 8192}


def regression(path: str, queries: int) -> None:
    ref = study(queries)
    print(f"\n### Rows 9h and 9hx against the study, HotpotQA 1M, seed 0, {queries:,} queries (`{path}`, `hp_fixes/pools.jsonl`)\n")
    print("| Row | Pair | Retention, row | Study | Difference, points | Absolute, row | Study |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    by_tier: dict = {}
    for r in read(path):
        if r["row"] not in BUDGET:
            continue
        tier = PAIRS[(r["src"], r["tgt"])]
        s = ref.get((tier, BUDGET[r["row"]]))
        if s is None:
            continue
        got, want = r["rho_ceil"]["ndcg@5"], s["ret"]
        by_tier.setdefault((r["row"], tier[:2]), []).append((got, want))
        print(f"| {r['row']} | {tier} | {got:.4f} | {want:.4f} | {100 * (got - want):+.2f} | {r['metrics']['ndcg@5']:.4f} | {s['ndcg5']:.4f} |")
    print("\n| Row | Tier | Mean, row | Mean, study | Difference, points |\n| --- | --- | --- | --- | --- |")
    for (row, tier), v in sorted(by_tier.items()):
        a, b = sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)
        print(f"| {row} | {tier} ({len(v)} pair{'s' if len(v) > 1 else ''}) | {a:.4f} | {b:.4f} | {100 * (a - b):+.2f} |")


def main() -> None:
    print("## Part A's regression\n\n### The study's own fits through merge_pool and rerank_pools (`m3_lib/regress.jsonl`)\n")
    print("| Pair | Budget | nDCG@5 | Study | Difference, points |\n| --- | --- | --- | --- | --- |")
    for r in read("m3_lib/regress.jsonl"):
        print(f"| {r['tier']} | {r['budget']} | {r['ndcg5']:.6f} | {r['ndcg5_study']:.6f} | {r['difference_points']:+.3f} |")
    regression("m3_lib/hp1m_reg.jsonl", 500)
    regression("m3_lib/hp1m_reg2k.jsonl", 2000)

    print("\n### Row 9x against M2's record on the same 500 queries (`m3_lib/hp1m_reg.jsonl`, `hp_scale/p1000_500q.jsonl`)\n")
    print("| Pair | Row 9x | Record | Difference, points |\n| --- | --- | --- | --- |")
    ref = {(r["src"], r["tgt"]): r for r in read("hp_scale/p1000_500q.jsonl") if r["row"] == "9x" and r["seed"] == 0}
    for r in read("m3_lib/hp1m_reg.jsonl"):
        if r["row"] == "9x":
            s = ref[(r["src"], r["tgt"])]
            print(f"| {PAIRS[(r['src'], r['tgt'])]} | {r['rho_ceil']['ndcg@5']:.5f} | {s['rho_ceil']['ndcg@5']:.5f} | {100 * (r['rho_ceil']['ndcg@5'] - s['rho_ceil']['ndcg@5']):+.3f} |")

    print("\n### A5: the real Searcher against the row, HotpotQA 1M, 200 queries, `bge-large-en` ← `bge-base-en` (`m3_lib/searcher.jsonl`, `m3_lib/hp1m_200q.jsonl`)\n")
    print("| Budget | Searcher nDCG@5 | Row | Row's nDCG@5 | Difference, points | Library's own budget | Grade |\n| --- | --- | --- | --- | --- | --- | --- |")
    rows = {r["depth"]: r for r in read("m3_lib/hp1m_200q.jsonl") if r["row"] in ("9h", "9hw")}
    for r in read("m3_lib/searcher.jsonl"):
        row = rows[r["budget"]]
        print(f"| {r['budget']} | {r['metrics']['ndcg@5']:.5f} | {row['row']} | {row['metrics']['ndcg@5']:.5f} | {100 * (r['metrics']['ndcg@5'] - row['metrics']['ndcg@5']):+.3f} | {r['default_budget']} | {r['grade']} |")

    print("\n### Identity, FiQA, K = 2,048, seed 0 (`m3_lib/identity.jsonl`)\n")
    print("| Pair | Row 1 Recall@30 | Native | Retention |\n| --- | --- | --- | --- |")
    recs = read("m3_lib/identity.jsonl")
    native = {(r["src"], r["tgt"]): r["metrics"]["recall@30"] for r in recs if r["row"] == "0b"}
    for r in recs:
        if r["row"] == "1":
            print(f"| `{r['src']}` | {r['metrics']['recall@30']:.4f} | {native[(r['src'], r['tgt'])]:.4f} | {r['rho_native']['recall@30']:.4f} |")

    print("\n### A3: the mount budget at the bottom of K = 4,096's range (`m3_lib/k4096_budget.jsonl`)\n")
    for r in read("m3_lib/k4096_budget.jsonl"):
        m = r.get("mount")
        if m:
            total = m["n_tokens"] + m.get("doc_n_tokens", 0)
            print(f"`{r['dataset']}`, `{r['src']}` ← `{r['tgt']}`, K = {r['K']:,}: {m['n_tokens']:,} proxy tokens + {m.get('doc_n_tokens', 0):,} anchor-doc tokens = {total:,}, "
                  f"against {m['corpus_tokens_est']:,.0f} estimated to re-index: **{100 * total / m['corpus_tokens_est']:.1f}%** of the budget's base.")


if __name__ == "__main__":
    main()
