"""M3's tables, from the records on disk. Retention is ρceil on nDCG@5: mean over pairs of the mean over seeds, as the targets define it.

    uv run python scripts/summarize_m3.py a6 > outputs/phase1/m3_tables/a6.md
    uv run python scripts/summarize_m3.py full > outputs/phase1/m3_tables/full.md
    uv run python scripts/summarize_m3.py b > outputs/phase1/m3_tables/b.md
"""
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from spacebridge_bench import durable

ROOT = Path("outputs/phase1")
T1 = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")}
LINES = {"T1": 0.98, "T3": 0.94}
STRETCH = {"T1": 0.95, "T3": 0.90}
BUDGET = {"9": 60, "9w": 100, "9x": 300, "9h": 60, "9hw": 100, "9hx": 300}


def tier(r: dict) -> str:
    if (r["src"], r["tgt"]) in T1:
        return "T1"
    return "T4" if r["src"] == "minilm" else "T3"


def short(r: dict) -> str:
    return f"{r['src']} ← {r['tgt']}"


def read(*paths: str) -> list[dict]:
    out = []
    for path in paths:
        if (ROOT / path).exists():
            out += durable.read_jsonl(ROOT / path)[0]
    return out


def rho(r: dict) -> float:
    return r["rho_ceil"]["ndcg@5"]


def strict(r: dict) -> float:
    return r["baselines_lexical"]["rho_ceil"]["ndcg@5"]


def absolute(r: dict) -> float:
    return r["metrics"]["ndcg@5"]


def by_pair(recs: list[dict], f) -> dict[str, float]:
    pairs = defaultdict(list)
    for r in recs:
        pairs[short(r)].append(f(r))
    return {p: sum(v) / len(v) for p, v in pairs.items()}


def mean_over_pairs(recs: list[dict], f=rho) -> float:
    return st.mean(by_pair(recs, f).values())


def seed_sd(recs: list[dict], f=rho) -> float:
    seeds = defaultdict(list)
    for r in recs:
        seeds[r["seed"]].append(f(r))
    means = [st.mean(v) for v in seeds.values()]
    return st.stdev(means) if len(means) > 1 else float("nan")


def ci(recs: list[dict]) -> tuple[float, float]:
    """The query-bootstrap bounds, averaged over cells, as M2's tables give them."""
    return st.mean(r["ci_ceil"]["ndcg@5"][0] for r in recs), st.mean(r["ci_ceil"]["ndcg@5"][1] for r in recs)


def worst(recs: list[dict], f=rho) -> tuple[str, float]:
    return min(by_pair(recs, f).items(), key=lambda kv: kv[1])


def group(records: list[dict]) -> dict:
    out = defaultdict(list)
    for r in records:
        if "depth" in r:
            out[(r["dataset"], tier(r), r["row"])].append(r)
    return out


def cells_positive(a: list[dict], b: list[dict], fa=rho, fb=rho) -> str:
    """(pair, seed) cells where a beats b, and the difference in points as mean ± s.d. over cells."""
    ka = {(r["src"], r["tgt"], r["seed"]): fa(r) for r in a}
    kb = {(r["src"], r["tgt"], r["seed"]): fb(r) for r in b}
    d = [100 * (ka[k] - kb[k]) for k in ka if k in kb]
    if not d:
        return "n/a"
    return f"{st.mean(d):+.1f} ± {st.stdev(d) if len(d) > 1 else 0:.1f}, {sum(x > 0 for x in d)}/{len(d)}"


def mark(v: float, line: float) -> str:
    return f"**{v:.3f}**" if v >= line else f"{v:.3f}"


def size_key(dataset: str) -> int:
    return int(dataset.split("@")[1]) if "@" in dataset else 0


def table(records: list[dict], rows: tuple[str, ...], lines: dict, title: str) -> None:
    """One line per (set, tier): each row's retention, then for the BM25 rows the strict retention, worst pair and absolute."""
    g = group(records)
    print(f"\n{title}\n")
    head = " | ".join(f"{r} ({BUDGET[r]})" for r in rows)
    print(f"| Set | Tier | {head} | Plain → BM25 at 60, points, (pair, seed) cells | 9h strict | 9h worst pair | 9h absolute | Native, re-index at 60 |")
    print("| --- | --- | " + " | ".join("---" for _ in rows) + " | --- | --- | --- | --- | --- |")
    for ds, t in sorted({(k[0], k[1]) for k in g}, key=lambda x: (x[0].split("@")[0], size_key(x[0]), x[1])):
        cols = []
        for r in rows:
            recs = g.get((ds, t, r))
            cols.append(mark(mean_over_pairs(recs), lines.get(t, 0.0)) if recs else "")
        h, plain = g.get((ds, t, "9h")), g.get((ds, t, "9"))
        if h:
            w = worst(h)
            base = h[0]["baselines"]
            n = st.mean(r["baselines"]["native"]["ndcg@5"] for r in h)
            o = st.mean(r["baselines"]["oracle"]["ndcg@5"] for r in h)
            tail = f"{cells_positive(h, plain) if plain else ''} | {mean_over_pairs(h, strict):.3f} | {w[1]:.3f} | {mean_over_pairs(h, absolute):.3f} | {n:.3f}, {o:.3f}"
        else:
            tail = " |  |  |  | "
        print(f"| {ds} | {t} | " + " | ".join(cols) + f" | {tail} |")


def a6() -> None:
    a = "m3_a6/"
    print("# M3 A6: the checks the default needs\n")
    print("Bold reaches the approved line: 0.98 Tier 1, 0.94 Tier 3. Each cell is the mean over pairs, 3 seeds. Tier 1 is one direction at the mid sizes, as `hp_scale` ran it.")
    sizes = [f"{c}{n}.jsonl" for c in ("ms", "nq", "hp") for n in (75000, 100000, 150000)]
    table(read(*[a + s for s in sizes]), ("9", "9h"), LINES, "## K = 4,096 and BM25 together: 75K, 100K, 150K, 500 queries")
    sizes = [f"{c}{n}.jsonl" for c in ("ms", "nq", "hp") for n in (200000, 500000)]
    table(read(*[a + s for s in sizes]), ("9", "9x", "9h"), LINES, "## 3 seeds at the sizes the default covers: 200K, 500K, K = 8,192, 500 queries")
    table(read(a + "fiqa.jsonl", a + "trec_covid.jsonl"), ("9", "9h"), LINES, "## Small corpora: FiQA (648 queries), TREC-COVID (50 queries)")

    print("\n## Tier 4: MS MARCO 1M, K = 8,192, 2,000 queries. Design's line is 92%\n")
    print("| Pair | 9 (60) | 9x (300) | 9h (60) | 9hx (300) | 9h strict | 9hx strict | 9h absolute | 9hx absolute | Native, re-index at 60 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    g = defaultdict(list)
    for r in read(a + "t4.jsonl"):
        if "depth" in r:
            g[(short(r), r["row"])].append(r)
    for p in sorted({k[0] for k in g}):
        c = {row: g.get((p, row)) for row in ("9", "9x", "9h", "9hx")}
        if all(c.values()):
            m = lambda row, f=rho: st.mean(f(r) for r in c[row])
            n, o = st.mean(r["baselines"]["native"]["ndcg@5"] for r in c["9h"]), st.mean(r["baselines"]["oracle"]["ndcg@5"] for r in c["9h"])
            print(f"| {p} | {m('9'):.3f} | {m('9x'):.3f} | {m('9h'):.3f} | {m('9hx'):.3f} | {m('9h', strict):.3f} | {m('9hx', strict):.3f} | {m('9h', absolute):.3f} | {m('9hx', absolute):.3f} | {n:.3f}, {o:.3f} |")

    print("\n## Source-free fit with BM25: 1M, K = 8,192, 500 queries, first stage Row 8f\n")
    print("| Set | Pair | 9h (60) | 9hw (100) | 9h strict | 9hw strict | 9h absolute | 9hw absolute |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    g = defaultdict(list)
    for r in read(a + "sf_hp.jsonl", a + "sf_ms.jsonl"):
        if "depth" in r:
            g[(r["dataset"], short(r), r["row"])].append(r)
    for ds, p in sorted({k[:2] for k in g}):
        c = {row: g.get((ds, p, row)) for row in ("9h", "9hw")}
        if all(c.values()):
            m = lambda row, f=rho: st.mean(f(r) for r in c[row])
            print(f"| {ds} | {p} | {m('9h'):.3f} | {m('9hw'):.3f} | {m('9h', strict):.3f} | {m('9hw', strict):.3f} | {m('9h', absolute):.3f} | {m('9hw', absolute):.3f} |")

    print("\n## HNSW at 1M with BM25: MS MARCO, ef 500, M = 32, 2,000 queries, 3 seeds\n")
    print("Retention here is absolute `nDCG@5` over the plain exact ceiling at the same budget, min(native, re-index), both reranked. `hnsw_study.py` read the two existing indexes and built nothing.\n")
    print("| Index ← query | Budget | Plain, exact | Plain, HNSW | BM25, exact | BM25, HNSW | HNSW's cost with BM25, points | Retention, BM25 through HNSW |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for path, index in (("m3_a6/hnsw_large.jsonl", "bge-large-en"), ("m3_a6/hnsw_base.jsonl", "bge-base-en")):
        recs = {r["key"]: r for r in read(path)}
        native = {r["pool"]: r for r in recs.values() if r["key"].startswith("hnsw|native|ef500")}
        for tgt in sorted({k.split("|")[1] for k in recs if k.startswith("lexical|")}):
            for pool in sorted({r["pool"] for k, r in recs.items() if k.startswith(f"lexical|{tgt}|")}):
                plain = [r for k, r in recs.items() if k.startswith(f"hnsw|{tgt}|s") and r["pool"] == pool]
                lex = [r for k, r in recs.items() if k.startswith(f"lexical|{tgt}|s") and r["pool"] == pool]
                oracle = recs[f"oracle|{tgt}|pool{pool}"]["ndcg5_exact"]
                ceil = min(native[pool]["ndcg5_exact"], oracle)
                m = lambda rs, k: st.mean(r[k] for r in rs)
                print(f"| {index} ← {tgt} | {pool} | {m(plain, 'ndcg5_exact'):.4f} | {m(plain, 'ndcg5_approx'):.4f} | {m(lex, 'ndcg5_exact'):.4f} | {m(lex, 'ndcg5_approx'):.4f} | {100 * (m(lex, 'ndcg5_approx') - m(lex, 'ndcg5_exact')):+.2f} | {m(lex, 'ndcg5_approx') / ceil:.3f} |")


def full() -> None:
    print("# M3 C4: end to end above 1M\n")
    print("Bold reaches the stretch line: 0.95 Tier 1, 0.90 Tier 3. Mean over pairs, 3 seeds, 2,000 queries, K = 8,192.")
    recs = read("m3_full/ms_t1.jsonl", "m3_full/ms_t3.jsonl", "m3_full/hp.jsonl")
    table(recs, ("9", "9w", "9x", "9h", "9hw", "9hx"), STRETCH, "## By budget")
    g = group(recs)
    print("\n## Each pair, each budget: retention [95% CI], strict, absolute\n")
    print("| Set | Pair | Row | Budget | Retention | CI | Strict | Absolute | Native, re-index |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (ds, t, row), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], BUDGET[kv[0][2]], kv[0][2])):
        pairs = defaultdict(list)
        for r in rs:
            pairs[short(r)].append(r)
        for p, v in sorted(pairs.items()):
            c = ci(v)
            s = st.mean(strict(r) for r in v) if "baselines_lexical" in v[0] else None
            print(f"| {ds} | {p} | {row} | {BUDGET[row]} | {st.mean(rho(r) for r in v):.3f} | [{c[0]:.3f}, {c[1]:.3f}] | {'' if s is None else f'{s:.3f}'} | {st.mean(absolute(r) for r in v):.3f} | {st.mean(r['baselines']['native']['ndcg@5'] for r in v):.3f}, {st.mean(r['baselines']['oracle']['ndcg@5'] for r in v):.3f} |")
    print("\n## Seed s.d., points, on the tier means\n")
    print("| Set | Tier | Row | Mean | Seed s.d. | Worst pair |\n| --- | --- | --- | --- | --- | --- |")
    for (ds, t, row), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], BUDGET[kv[0][2]], kv[0][2])):
        w = worst(rs)
        print(f"| {ds} | {t} | {row} | {mean_over_pairs(rs):.3f} | {100 * seed_sd(rs):.2f} | {w[0]}: {w[1]:.3f} |")


def b() -> None:
    recs = read("m3_b/screen.jsonl")
    print("# M3 Part B: cross-language candidates, Chinese to English, `bge-large-en` ← `qwen3-0.6b`, K = 2,048, seed 0\n")
    print("Absolute `nDCG@5`, reranker reading the original query. Gain is against Row 7b at the same budget. A candidate works if it gains at least 2 points at budget 60 on both sets.\n")
    names = {"7b": "Row 7b", "X1": "X1: BM25, raw query", "X2": "X2: BM25, opus-mt", "X3": "X3: BM25, MADLAD", "X4o": "X4: no bridge, BM25 on opus-mt", "X4m": "X4: no bridge, BM25 on MADLAD", "X5": "X5: local residuals", "reindex": "Re-index", "native": "Native (human English)"}
    sets = sorted({r["dataset"] for r in recs})
    print("| Candidate | Budget | " + " | ".join(f"{s} | gain" for s in sets) + " | Pool, mean |\n| --- | --- | " + " | ".join("--- | ---" for _ in sets) + " | --- |")
    idx = {(r["dataset"], r["candidate"], r["budget"]): r for r in recs}
    for c, name in names.items():
        for budget in (60, 100):
            cols = []
            pools = []
            for s in sets:
                r, base = idx.get((s, c, budget)), idx.get((s, "7b", budget))
                if r is None:
                    cols.append(" | ")
                    continue
                pools.append(f"{r['pool_mean']:.0f}")
                cols.append(f"{r['metrics']['ndcg@5']:.3f} | {100 * (r['metrics']['ndcg@5'] - base['metrics']['ndcg@5']):+.1f}")
            print(f"| {name} | {budget} | " + " | ".join(cols) + f" | {', '.join(pools)} |")


if __name__ == "__main__":
    {"a6": a6, "full": full, "b": b}[sys.argv[1]]()
