"""M2 Part E, from records on disk: lighter rerankers by pool (absolute nDCG@5 as well as retention, since a smaller reranker lowers
both), bge-reranker-v2-m3 int8 against the reference on the same candidates, each one's time, and Row 8f against Row 8b on nq-1m.

A reranker's retention is against a ceiling measured with that same reranker, so it can look fine while the answers get worse. The
last table of each corpus therefore also holds every reranker against the fp32 reference's ceiling at the same pool.

    uv run python scripts/summarize_rerankers.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("outputs/phase1")
REFERENCE = "bge-reranker-v2-m3"
LIGHT = (REFERENCE, "bge-reranker-base", "ms-marco-minilm")
POOLS = (60, 100, 200, 300)
TIER1 = {"bge-base-en", "bge-large-en"}
TIMINGS = {"bge-reranker-v2-m3 fp32": None, "bge-reranker-v2-m3 int8": "int8", "bge-reranker-base": "bge-reranker-base", "ms-marco-MiniLM-L6-v2": "MiniLM"}
SIZES = {"msmarco-1m": "short", "trec-covid": "long"}


def tier(r: dict) -> str:
    return "1" if {r["src"], r["tgt"]} <= TIER1 else "3"


def pair(r: dict) -> str:
    return f"{r['src']} ← {r['tgt']}"


def read(paths) -> list[dict]:
    return [json.loads(line) for path in paths for line in Path(path).open()]


def by_cell(records: list[dict]) -> dict:
    """(dataset, tier, reranker, pool, pair) -> records of that cell, one per seed. A cell seen twice keeps the later record."""
    cells = defaultdict(dict)
    for r in records:
        if r.get("reranker") and r.get("depth"):
            cells[(r["dataset"], tier(r), r["reranker"], r["depth"], pair(r))][r["seed"]] = r
    return {k: list(v.values()) for k, v in cells.items()}


def stat(rs: list[dict]) -> dict:
    """Means over seeds: absolute nDCG@5, its ceiling (as in retention, the lower of native and re-index, same reranker), retention, CI."""
    native = np.mean([r["baselines"]["native"]["ndcg@5"] for r in rs])
    oracle = np.mean([r["baselines"]["oracle"]["ndcg@5"] for r in rs])
    return {
        "abs": float(np.mean([r["metrics"]["ndcg@5"] for r in rs])), "ceil": float(min(native, oracle)), "native": float(native), "oracle": float(oracle),
        "ret": float(np.mean([r["rho_ceil"]["ndcg@5"] for r in rs])),
        "lo": float(np.mean([r["ci_ceil"]["ndcg@5"][0] for r in rs])), "hi": float(np.mean([r["ci_ceil"]["ndcg@5"][1] for r in rs])),
        "seeds": len(rs), "n_q": rs[0]["n_queries"], "ms": float(np.mean([r.get("rerank_ms_first_query_min3", r.get("rerank_ms_per_query", np.nan)) for r in rs])),
    }


def light_tables(cells: dict) -> None:
    print("## Lighter rerankers\n\nEach cell: absolute nDCG@5 / retention / worst pair's retention. Means over pairs of the tier and 3 seeds.\n")
    for ds, t in sorted({(k[0], k[1]) for k in cells}):
        pairs = sorted({k[4] for k in cells if k[:2] == (ds, t)})
        n_q = next(stat(v)["n_q"] for k, v in cells.items() if k[:2] == (ds, t))
        pools = [q for q in POOLS if any(k[:2] == (ds, t) and k[3] == q for k in cells)]
        print(f"### {ds}, Tier {t}, {len(pairs)} pairs, {n_q} queries\n")
        print("| Reranker | " + " | ".join(f"Pool {q}" for q in pools) + " |")
        print("| --- | " + " | ".join("---" for _ in pools) + " |")
        tiered = {}
        for rr in LIGHT:
            row = []
            for q in pools:
                found = [stat(cells[(ds, t, rr, q, pr)]) for pr in pairs if (ds, t, rr, q, pr) in cells]
                if not found:
                    row.append("-")
                    continue
                tiered[rr, q] = {"abs": np.mean([c["abs"] for c in found]), "ceil": np.mean([c["ceil"] for c in found])}
                row.append(f"{tiered[rr, q]['abs']:.3f} / {np.mean([c['ret'] for c in found]):.3f} / {min(c['ret'] for c in found):.3f}")
            print(f"| `{rr}` | " + " | ".join(row) + " |")
        print("\nAgainst the reference's ceiling at the same pool (absolute ÷ reference's ceiling), and in points of absolute nDCG@5 from the reference's bridged result:\n")
        print("| Reranker | " + " | ".join(f"Pool {q}" for q in pools) + " |")
        print("| --- | " + " | ".join("---" for _ in pools) + " |")
        for rr in LIGHT:
            row = []
            for q in pools:
                mine, ref = tiered.get((rr, q)), tiered.get((REFERENCE, q))
                row.append(f"{mine['abs'] / ref['ceil']:.3f} ({100 * (mine['abs'] - ref['abs']):+.1f})" if mine and ref else "-")
            print(f"| `{rr}` | " + " | ".join(row) + " |")
        print()


def int8_table(records: list[dict]) -> None:
    print("## int8 on the CPU against the reference in bf16 on the GPU, bridged candidates, one seed: a screen, not a claim\n")
    print("| Corpus | Index ← query | Queries | Pool | Absolute bf16 | Absolute int8 | Difference, points | Retention bf16 | Retention int8 | Difference, points |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cells = by_cell(records)
    for (ds, t, rr, q, pr), rs in sorted(cells.items()):
        if rr != REFERENCE or (ds, t, rr + "-int8", q, pr) not in cells:
            continue
        a, b = stat(rs), stat(cells[(ds, t, rr + "-int8", q, pr)])
        print(f"| {ds} | `{pr}` | {a['n_q']} | {q} | {a['abs']:.3f} | {b['abs']:.3f} | {100 * (b['abs'] - a['abs']):+.2f} | {a['ret']:.3f} | {b['ret']:.3f} | {100 * (b['ret'] - a['ret']):+.2f} |")
    print()


def precision_table(records: list[dict]) -> None:
    print("## fp32, bf16 and int8 on the same candidates: the index model's own first stage, no bridge\n")
    print("| Corpus | Queries | Pool | fp32 | bf16 | bf16 - fp32, points [95% CI] | int8 | int8 - fp32, points [95% CI] | Top-5 overlap with fp32, bf16 / int8 | Distinct scores of 100, fp32 / bf16 / int8 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cells = {(r["dataset"], r["n_queries"], r["pool"], r["variant"]): r for r in records}
    for ds, n, pool in sorted({k[:3] for k in cells}):
        a, b, c = (cells[ds, n, pool, v] for v in ("fp32", "bf16", "int8"))
        ci = lambda r: f"{100 * r['difference_from_fp32']:+.2f} [{100 * r['ci_difference'][0]:+.2f}, {100 * r['ci_difference'][1]:+.2f}]"
        share = lambda r: f"{100 * r['distinct_scores'] / r['scores']:.0f}"
        print(f"| {ds} | {n} | {pool} | {a['ndcg@5']:.3f} | {b['ndcg@5']:.3f} | {ci(b)} | {c['ndcg@5']:.3f} | {ci(c)} | {b['top5_jaccard_with_fp32']:.3f} / {c['top5_jaccard_with_fp32']:.3f} | {share(a)} / {share(b)} / {share(c)} |")
    print()


def length_table(records: list[dict]) -> None:
    print("## Precision and document length, by reranker. Index model's own first stage, no bridge\n")
    print("| Reranker | Corpus | Queries | Pool | fp32 | bf16 | bf16 - fp32, points [95% CI] | Tokens per pair, top 5 (fp32) | Tokens per pair, pool |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cells = {(r["reranker"], r["dataset"], r["n_queries"], r["pool"], r["variant"]): r for r in records}
    for rr, ds, n, pool in sorted({k[:4] for k in cells}):
        a, b = cells[rr, ds, n, pool, "fp32"], cells[rr, ds, n, pool, "bf16"]
        print(f"| `{rr}` | {ds} | {n} | {pool} | {a['ndcg@5']:.3f} | {b['ndcg@5']:.3f} | {100 * b['difference_from_fp32']:+.2f} [{100 * b['ci_difference'][0]:+.2f}, {100 * b['ci_difference'][1]:+.2f}] | {a['top5_tokens']:.0f} | {a['pool_tokens']:.0f} |")
    print()


def timing_name(r: dict) -> str:
    rr = r.get("reranker") or "BAAI/bge-reranker-v2-m3"
    return next(name for name, mark in reversed(TIMINGS.items()) if mark is None or mark in rr)


def timing_table(records: list[dict]) -> None:
    print("## Time, one query, idle machine\n")
    print("| Reranker | Device | Threads | Documents | Pool | Rerank p50, s | Total p50, s | Total p95, s | ms per candidate | Tokens per pair | Runs |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = {}
    for r in records:
        if r["dataset"] in SIZES and r.get("rerank") and r["pool"] in (60, 100, 300) and r["device"] in ("cuda", "cpu"):
            rows[timing_name(r), r["device"], r["threads"] if r["device"] == "cpu" else "-", SIZES[r["dataset"]], r["pool"]] = r
    for (name, device, threads, kind, pool), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][3], list(TIMINGS).index(kv[0][0]), kv[0][4], str(kv[0][2]))):
        s = r["stages_ms"]
        print(f"| {name} | {device} | {threads} | {kind} | {pool} | {s['rerank']['p50'] / 1e3:.2f} | {s['total']['p50'] / 1e3:.2f} | {s['total']['p95'] / 1e3:.2f} | "
              f"{s['rerank']['p50'] / pool:.1f} | {r.get('tokens_per_pair', float('nan')):.0f} | {r['n']} |")
    print()


def free_tables(free: list[dict], with_source: list[dict]) -> None:
    print("## Row 8f (source-free fit) against Row 8b (with the teacher term), nq-1m, K = 8,192, 3 seeds; report only\n")
    mine = lambda rs, row: {(pair(r), r["seed"]): r for r in rs if r["row"] == row and r["K"] == 8192}
    a, b = mine(free, "8f"), mine(with_source, "8b")
    print("| Index ← query | Queries | 8b R@300 | 8f R@300 | 8b retention R@300 | 8f retention R@300 | Difference, points | 8b fit, ms | 8f fit, ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pr in sorted({k[0] for k in a}):
        x, y = [b[k] for k in b if k[0] == pr], [a[k] for k in a if k[0] == pr]
        if not x:
            continue
        assert {r["n_queries"] for r in x} == {r["n_queries"] for r in y}, f"{pr}: not the same queries"
        rx, ry = np.mean([r["rho_ceil"]["recall@300"] for r in x]), np.mean([r["rho_ceil"]["recall@300"] for r in y])
        fit = lambda rs: np.mean([r["fit"]["stage1_ms"] + r["fit"]["stage2_ms"] for r in rs])
        print(f"| `{pr}` | {y[0]['n_queries']} | {np.mean([r['metrics']['recall@300'] for r in x]):.3f} | {np.mean([r['metrics']['recall@300'] for r in y]):.3f} | {rx:.3f} | {ry:.3f} | {100 * (ry - rx):+.2f} | {fit(x):.0f} | {fit(y):.0f} |")
    print("\nEnd to end, `bge-reranker-v2-m3`: retention [95% CI] and absolute nDCG@5 on the same queries, mean of 3 seeds\n")
    print("| Index ← query | Pool | 8b retention | 8f retention | Difference, points | 8b absolute | 8f absolute | Seeds 8b, 8f |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    keep = lambda rs, base: by_cell([r for r in rs if r.get("reranker") == REFERENCE and r.get("first_stage_row", "8b") == base and r["K"] == 8192])
    ca, cb = keep(free, "8f"), keep(with_source, "8b")
    differences = defaultdict(list)
    for k in sorted(ca, key=lambda k: (k[1], k[4], k[3])):
        if k not in cb:
            continue
        x, y = stat(cb[k]), stat(ca[k])
        assert x["n_q"] == y["n_q"], f"{k}: not the same queries"
        differences[k[1], k[3]].append(100 * (y["ret"] - x["ret"]))
        print(f"| `{k[4]}` | {k[3]} | {x['ret']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] | {y['ret']:.3f} [{y['lo']:.3f}, {y['hi']:.3f}] | {100 * (y['ret'] - x['ret']):+.2f} | {x['abs']:.3f} | {y['abs']:.3f} | {x['seeds']}, {y['seeds']} |")
    print("\n| Tier | Pool | Mean difference 8f - 8b, points | Worst pair, points |\n| --- | --- | --- | --- |")
    for (t, q), v in sorted(differences.items()):
        print(f"| {t} | {q} | {np.mean(v):+.2f} | {min(v):+.2f} |")
    print()


def main() -> None:
    light_tables(by_cell(read(sorted((ROOT / "m2_e").glob("rr_*.jsonl")))))
    int8_table(read(sorted((ROOT / "m2_e_int8").glob("*.jsonl"))))
    precision_table(read([ROOT / "m2_e_more/precision.jsonl"]))
    length_table(read([ROOT / "m2_e_more/length.jsonl"]))
    timing_table(read([ROOT / "m1_timing/timing.jsonl", ROOT / "m2_timing/timing.jsonl", ROOT / "m2_e_timing/timing.jsonl"]))
    free_tables(read([ROOT / "m2_e_more/free_nq2k.jsonl"]), read([*sorted((ROOT / "w3_1m").glob("nq1m_e2e*.jsonl")), ROOT / "m2_pools/nq1m.jsonl"]))


if __name__ == "__main__":
    main()
