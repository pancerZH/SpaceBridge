"""M2 A4: end-to-end retention through a real HNSW index, from hnsw_study.py records.

Retention through HNSW is bridged-through-HNSW / min(native through the same HNSW setting, re-index exact); a re-indexed model has
no HNSW index here, so its exact score is the ceiling's other side, as in M1. Exact retention is the same ratio with exact search.

    uv run python scripts/summarize_hnsw.py outputs/phase1/w5_ann/ms1m.jsonl outputs/phase1/m2_ann/ms1m_base.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

from spacebridge_bench import durable

EFS = (300, 500, 1000)
POOLS = (60, 100, 150, 200, 300)
INDEX_MODEL = {"ms1m.jsonl": "bge-large-en", "ms1m_base.jsonl": "bge-base-en"}


def retention(rec: dict, tgt: str, ef: int, pool: int) -> tuple:
    native = rec.get(f"hnsw|native|ef{ef}|pool{pool}")
    oracle = rec.get(f"oracle|{tgt}|pool{pool}")
    seeds = [rec.get(f"hnsw|{tgt}|s{s}|ef{ef}|pool{pool}") for s in (0, 1, 2)]
    if native is None or oracle is None or not all(seeds):
        return None
    through = [s["ndcg5_approx"] / min(native["ndcg5_approx"], oracle["ndcg5_exact"]) for s in seeds]
    exact = [s["ndcg5_exact"] / min(native["ndcg5_exact"], oracle["ndcg5_exact"]) for s in seeds]
    return float(np.mean(through)), float(np.mean(exact)), float(np.mean([s["recall_of_exact"] for s in seeds])), float(np.mean([s["ndcg5_approx"] for s in seeds])), native["ndcg5_approx"], native["recall_of_exact"]


def main() -> None:
    rows = {}
    builds = {}
    for path in map(Path, sys.argv[1:]):
        records, _ = durable.read_jsonl(path)
        rec = {r["key"]: r for r in records}
        index = INDEX_MODEL[path.name]
        builds[index] = rec.get("build")
        for tgt in sorted({k.split("|")[1] for k in rec if k.startswith("oracle|")}):
            for ef in EFS:
                for pool in POOLS:
                    got = retention(rec, tgt, ef, pool)
                    if got:
                        rows[index, tgt, ef, pool] = got
    pairs = sorted({k[:2] for k in rows})
    print("| Index ← query | ef | " + " | ".join(f"Pool {p}" for p in POOLS) + " | Recall of exact top-300 | Native recall |\n| --- | --- | " + " | ".join("---" for _ in POOLS) + " | --- | --- |")
    for idx, tgt in pairs:
        print(f"| `{idx} ← {tgt}` | exact | " + " | ".join(f"{rows[idx, tgt, 500, p][1]:.3f}" if (idx, tgt, 500, p) in rows else "-" for p in POOLS) + " | | |")
        for ef in EFS:
            cells = [f"{rows[idx, tgt, ef, p][0]:.3f}" if (idx, tgt, ef, p) in rows else "-" for p in POOLS]
            last = rows.get((idx, tgt, ef, 300))
            print(f"| | {ef} | " + " | ".join(cells) + (f" | {last[2]:.3f} | {last[5]:.3f} |" if last else " | | |"))
    print("\n**Tier 1, mean over both directions, through HNSW** (line 0.98):\n")
    print("| ef | " + " | ".join(f"Pool {p}" for p in POOLS) + " |\n| --- | " + " | ".join("---" for _ in POOLS) + " |")
    t1 = [("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")]
    print("| exact | " + " | ".join(f"{np.mean([rows[a, b, 500, p][1] for a, b in t1]):.3f}" if all((a, b, 500, p) in rows for a, b in t1) else "-" for p in POOLS) + " |")
    for ef in EFS:
        print(f"| {ef} | " + " | ".join(f"{np.mean([rows[a, b, ef, p][0] for a, b in t1]):.3f}" if all((a, b, ef, p) in rows for a, b in t1) else "-" for p in POOLS) + " |")
    print("\n**Loss through HNSW, `bge-large-en` ← `qwen3-0.6b`** (the only Tier 3 pair with an index), points, which also gives the estimate for the Tier 3 mean:\n")
    print("| ef | " + " | ".join(f"Pool {p}" for p in POOLS) + " |\n| --- | " + " | ".join("---" for _ in POOLS) + " |")
    for ef in EFS:
        print(f"| {ef} | " + " | ".join(f"{100 * (rows['bge-large-en', 'qwen3-0.6b', ef, p][1] - rows['bge-large-en', 'qwen3-0.6b', ef, p][0]):+.2f}" for p in POOLS) + " |")
    print("\nBuilds:", json.dumps({k: {f: v[f] for f in ("build_s_this_run", "file_bytes", "threads")} for k, v in builds.items() if v}))


if __name__ == "__main__":
    main()
