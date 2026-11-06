"""Historical: M3's tables, from the records on disk. The current documents' tables come from scripts/review_tables.py. Retention is ρceil on nDCG@5: mean over pairs of the mean over seeds, as the targets define it.

    uv run python scripts/summarize_m3.py a6 > outputs/phase1/m3_tables/a6.md
    uv run python scripts/summarize_m3.py full > outputs/phase1/m3_tables/full.md
    uv run python scripts/summarize_m3.py b > outputs/phase1/m3_tables/b.md
    uv run python scripts/summarize_m3.py table1 60 > outputs/phase1/m3_tables/table1.md
"""
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def read(*paths: str, required: bool = False) -> list[dict]:
    """Records of these files, skipping files not there, unless ``required``: a decision must not read missing evidence as nothing to say."""
    out = []
    for path in paths:
        if (ROOT / path).exists():
            out += durable.read_jsonl(ROOT / path)[0]
        elif required:
            raise FileNotFoundError(ROOT / path)
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
    """The cells' query-bootstrap bounds, averaged. This describes a typical cell: it is not a confidence interval for the mean over pairs
    and seeds, whose cells share queries. scripts/retention_ci.py computes that one from aligned per-query scores."""
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
    print(f"| Set | Tier | {head} | Plain → BM25 at 60, points, (pair, seed) cells | 9h strict | 9h worst pair | Absolute, plain → 9h | Native, re-index at 60 |")
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
            tail = f"{cells_positive(h, plain) if plain else ''} | {mean_over_pairs(h, strict):.3f} | {w[1]:.3f} | {f'{mean_over_pairs(plain, absolute):.3f} → ' if plain else ''}{mean_over_pairs(h, absolute):.3f} | {n:.3f}, {o:.3f}"
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


def one_m() -> None:
    print("# What ships at 1M, through the rows\n")
    print("Bold reaches the approved line: 0.98 Tier 1, 0.94 Tier 3. Mean over pairs, 3 seeds, 2,000 queries, K = 8,192, Tier 1 both directions.")
    table(read("m3_1m/hp.jsonl", "m3_1m/ms.jsonl", "m3_1m/nq2k.jsonl"), ("9", "9w", "9x", "9h", "9hw", "9hx"), LINES, "## By budget")


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


# Size, label, BM25 on by the library's rule, then (corpus, record files, K) as the library's profile sets it.
TABLE1 = [
    (5_183, "5.2K", True, 2048, [("SciFact", ["closeout/b1_scifact.jsonl"])]),
    (10_000, "10K", True, 2048, [("MS MARCO", ["closeout/b1_ms10000.jsonl"]), ("NQ", ["closeout/b1_nq10000.jsonl"]), ("HotpotQA", ["followup/hp10000.jsonl"])]),
    (20_000, "20K", True, 2048, [("MS MARCO", ["closeout/b1_ms20000.jsonl"]), ("NQ", ["closeout/b1_nq20000.jsonl"]), ("HotpotQA", ["followup/hp20000.jsonl"])]),
    (25_657, "25.7K", True, 2048, [("SciDocs", ["followup/scidocs.jsonl"])]),
    (30_000, "30K", True, 2048, [("MS MARCO", ["closeout/b1_ms30000.jsonl"]), ("NQ", ["closeout/b1_nq30000.jsonl"]), ("HotpotQA", ["followup/hp30000.jsonl"])]),
    (50_000, "50K", True, 2048, [("MS MARCO", ["closeout/b1_ms50000.jsonl"]), ("NQ", ["closeout/b1_nq50000.jsonl"]), ("HotpotQA", ["followup/hp50000.jsonl"])]),
    (100_000, "100K", True, 2048, [("MS MARCO", ["closeout/b2_ms100000.jsonl"]), ("NQ", ["closeout/b2_nq100000.jsonl"]), ("HotpotQA", ["closeout/b2_hp100000.jsonl"])]),
    (150_000, "150K", True, 2048, [("MS MARCO", ["closeout/b2_ms150000.jsonl"]), ("NQ", ["closeout/b2_nq150000.jsonl"]), ("HotpotQA", ["closeout/b2_hp150000.jsonl"])]),
    (160_000, "160K", True, 2048, [("MS MARCO", ["final/ms160000.jsonl"]), ("NQ", ["final/nq160000.jsonl"]), ("HotpotQA", ["final/hp160000.jsonl"])]),
    (160_001, "160K + 1", True, 8192, [("MS MARCO", ["final/ms160001.jsonl"]), ("NQ", ["final/nq160001.jsonl"]), ("HotpotQA", ["final/hp160001.jsonl"])]),
    (171_332, "171K", True, 8192, [("TREC-COVID", ["m3_a6/trec_covid.jsonl"])]),
    (200_000, "200K", True, 8192, [("MS MARCO", ["m3_a6/ms200000.jsonl"]), ("NQ", ["m3_a6/nq200000.jsonl"]), ("HotpotQA", ["m3_a6/hp200000.jsonl"])]),
    (500_000, "500K", True, 8192, [("MS MARCO", ["m3_a6/ms500000.jsonl"]), ("NQ", ["m3_a6/nq500000.jsonl"]), ("HotpotQA", ["m3_a6/hp500000.jsonl"])]),
    (1_000_000, "1M", True, 8192, [("MS MARCO", ["m3_1m/ms.jsonl"]), ("NQ", ["m3_1m/nq2k.jsonl"]), ("HotpotQA", ["m3_1m/hp.jsonl"])]),
    (2_000_000, "2M", True, 8192, [("MS MARCO", ["closeout/b4_ms2000000.jsonl"]), ("HotpotQA", ["closeout/b4_hp2000000.jsonl"])]),
    (4_000_000, "4M", True, 8192, [("MS MARCO", ["closeout/b4_ms4000000.jsonl"])]),
    (5_233_329, "5.23M", True, 8192, [("HotpotQA", ["m3_full/hp.jsonl"])]),
    (8_841_823, "8.84M", True, 8192, [("MS MARCO", ["m3_full/ms_t1.jsonl", "m3_full/ms_t3.jsonl"])]),
]
ROW_AT = {60: "9h", 100: "9hw", 300: "9hx"}


class Timing:
    """A query's time by size: the stages measured at one size, with search and BM25 replaced by their time at this size. On the
    GPU, sizes above 1M search on the CPU, since the GPU can't hold them."""

    def __init__(self):
        self.records = read("m1_timing/timing.jsonl", "m2_timing/timing.jsonl", "m3_timing/timing.jsonl", "closeout/timing.jsonl")  # oldest first: the latest reading of a size stands
        self.lexical_points = sorted({r["N"]: r["stages_ms"]["lexical"]["p50"] for r in self.records if r.get("lexical") and r["device"] == "cpu" and r["dataset"].startswith("msmarco") and r["backend"] == "exact" or r.get("lexical") and r["device"] == "cpu" and r["N"] == 1_000_000}.items())

    def search_ms(self, device: str, size: int) -> float:
        if size > 1_000_000:
            device = "cpu"
        hnsw = [r for r in self.records if r["device"] == device and not r["rerank"] and r["backend"] != "exact" and r["threads"] == 10 and r["ef"] == 500 and r["N"] == size
                and r.get("index_model", "bge-large-en") == "bge-large-en" and r.get("query_model", "qwen3-0.6b") == "qwen3-0.6b"]
        if size == 1_000_000 and hnsw:
            values = {round(r["stages_ms"]["search"]["p50"], 3) for r in hnsw}
            if len(values) > 1:  # two measurements of one configuration: record order is not a reason to prefer one
                raise ValueError(f"{len(hnsw)} HNSW search timings match {device} at {size:,} documents ({sorted(values)}): remove the ones that don't apply")
            return hnsw[0]["stages_ms"]["search"]["p50"]
        pts = sorted({r["N"]: r["stages_ms"]["search"]["p50"] for r in self.records if r["device"] == device and r["backend"] == "exact" and r.get("search_device", device) == device and (not r["rerank"] or r["N"] > 1_000_000)}.items())
        return float(np.interp(size, *zip(*pts)))

    def lexical_ms(self, size: int) -> float:
        return float(np.interp(size, *zip(*([(0, 0.0)] + self.lexical_points)))) if self.lexical_points else 0.0

    def run(self, device: str, corpus: str, pool: int, lexical: bool, hnsw: bool = False):
        """The latest record at this pool; a BM25 run may stand in for a plain one, since BM25's own time is taken off. At 8.84M, exact search unless ``hnsw``."""
        runs = [r for r in self.records if r["device"] == device and r["threads"] == 10 and r["rerank"] and r["dataset"] == corpus and r["pool"] == pool and not r.get("reranker", "").startswith("onnx") and r.get("reranker", "BAAI/bge-reranker-v2-m3") == "BAAI/bge-reranker-v2-m3"]
        if corpus == "msmarco-full":
            runs = [r for r in runs if r["ef"] == 500 and (r["backend"] != "exact") == hnsw]
        chosen = [r for r in runs if bool(r.get("lexical")) == lexical] or runs
        return chosen[-1] if chosen else None

    def is_measured(self, device: str, corpus: str, size: int, pool: int, lexical: bool) -> bool:
        """Whether a run of this device, corpus, size, pool and BM25 setting exists. Otherwise ``query_s`` is a model, and its p95 especially:
        percentiles don't add, so swapping medians into a measured total estimates a tail, it doesn't observe one."""
        r = self.run(device, corpus, pool, lexical)
        return r is not None and r["N"] == size and bool(r.get("lexical")) == lexical

    def query_s(self, device: str, corpus: str, size: int, pool: int, q: str, lexical: bool) -> float:
        r = self.run(device, corpus, pool, lexical)
        if r is None:
            return float("nan")
        stage = r["stages_ms"]
        if r["N"] == size:  # measured at this size: nothing to adjust
            return stage["total"][q] / 1e3
        rest = stage["total"][q] - stage["search"]["p50"] - stage.get("lexical", {"p50": 0.0})["p50"]  # the p50s, so a p95 can't fall under its p50
        return (rest + self.search_ms(device, size) + (self.lexical_ms(size) if lexical else 0.0)) / 1e3


def table1(budget_above_1m: int = 60) -> None:
    timing = Timing()
    print(f"**Table 1: what the library does.** Seconds for one query, p50 (p95), through `Searcher`, and retention on the hardest corpus measured at each size. A cell starting with ≈ is modeled, not measured: another size's run with its search and BM25 medians replaced. The CPU columns stop at the first size over 60 s. Budget above 1M: {budget_above_1m}.\n")
    print("| Size | BM25 | Budget | K | Tier 1 (corpus) | Tier 3 (corpus) | CPU, short | CPU, long | GPU, short | GPU, long |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    stopped = {"short": False, "long": False}
    for size, label, lexical, K, corpora in TABLE1:
        budget = budget_above_1m if size > 1_000_000 else 60
        row = ROW_AT[budget] if lexical else "9"
        by_tier = defaultdict(list)
        for name, files in corpora:
            recs = [r for r in read(*files) if r.get("row") == row and r.get("reranker") == "bge-reranker-v2-m3" and r["K"] == K]
            for t in ("T1", "T3"):
                rs = [r for r in recs if tier(r) == t]
                if rs:
                    by_tier[t].append((mean_over_pairs(rs), name))
        lines = LINES if size <= 1_000_000 else STRETCH
        cell = {}
        for t in ("T1", "T3"):
            if by_tier[t]:
                v, name = min(by_tier[t])
                cell[t] = f"{v:.3f}{'' if v >= lines[t] else ' **miss**'} ({name})"
            else:
                cell[t] = "-"
        times = {}
        for kind, corpus, full in (("short", "msmarco-1m", "msmarco-full"), ("long", "trec-covid", None)):
            big = size > 1_000_000 and kind == "short" and timing.run("cpu", full, budget, True) is not None
            source = full if big else corpus
            p50 = timing.query_s("cpu", source, size, budget, "p50", lexical)
            if stopped[kind]:
                times[f"cpu_{kind}"] = "stop"
            else:
                over = p50 > 60
                text = f"{'' if timing.is_measured('cpu', source, size, budget, lexical) else '≈'}{p50:.1f} ({timing.query_s('cpu', source, size, budget, 'p95', lexical):.1f})"
                times[f"cpu_{kind}"] = f"**{text}**" if over else text
                stopped[kind] = over
            times[f"gpu_{kind}"] = f"{'' if timing.is_measured('cuda', source, size, budget, lexical) else '≈'}{timing.query_s('cuda', source, size, budget, 'p50', lexical):.2f} ({timing.query_s('cuda', source, size, budget, 'p95', lexical):.2f})"
        print(f"| {label} | {'on' if lexical else 'off'} | {budget} | {K:,} | {cell['T1']} | {cell['T3']} | {times['cpu_short']} | {times['cpu_long']} | {times['gpu_short']} | {times['gpu_long']} |")
    hnsw_row(timing)


def hnsw_row(timing: Timing) -> None:
    """The reference deployment at 8.84M: one index, ef 500, retention through it, one direction per tier. Long is TREC-COVID's rerank with this search."""
    res = {r["key"]: r for r in read("closeout/b7_hnsw.jsonl")}
    runs = {d: timing.run(d, "msmarco-full", 60, True, hnsw=True) for d in ("cpu", "cuda")}
    if not res or None in runs.values():
        return
    cell = {}
    for t, tgt in (("T1", "bge-base-en"), ("T3", "qwen3-0.6b")):
        lex = [res[f"lexical|{tgt}|s{seed}|ef500|pool60"]["ndcg5_approx"] for seed in (0, 1, 2)]
        v = st.mean(lex) / min(res["hnsw|native|ef500|pool60"]["ndcg5_exact"], res[f"oracle|{tgt}|pool60"]["ndcg5_exact"])
        cell[t] = f"{v:.3f}{'' if v >= STRETCH[t] else ' **miss**'} (MS MARCO)"
    times = {}
    for d in ("cpu", "cuda"):
        s_ = runs[d]["stages_ms"]
        long_ = timing.run(d, "trec-covid", 60, True)["stages_ms"]
        fmt = (lambda x: f"{x:.1f}") if d == "cpu" else (lambda x: f"{x:.2f}")
        add = s_["search"]["p50"] + s_["lexical"]["p50"] - long_["search"]["p50"] - long_["lexical"]["p50"]
        times[d] = f"{fmt(s_['total']['p50'] / 1e3)} ({fmt(s_['total']['p95'] / 1e3)}) | {fmt((long_['total']['p50'] + add) / 1e3)} ({fmt((long_['total']['p95'] + add) / 1e3)})"
    print(f"| 8.84M, HNSW | on | 60 | 8,192 | {cell['T1']} | {cell['T3']} | {times['cpu']} | {times['cuda']} |")


def break_even() -> None:
    """A scenario estimate, not a measured saving: encoding seconds of a re-index (token rate, extrapolated) / what the bridge adds per query. It leaves out
    index construction and the bridge's own setup; scripts/review_tables.py gives the fuller equation. It depends on what the re-indexed system does. If it also
    reranks about 60, the bridge adds BM25 and the projection. If it doesn't rerank, it adds the reranking too."""
    timing = Timing()
    rate = {(r["model"], r["device"]): r for r in read("m1_timing/mount.jsonl") if r.get("mode") == "rate"}
    sizes = {1_000_000: "msmarco-1m", 8_841_823: "msmarco-full"}
    stage = {(size, d): timing.run(d, corpus, 60, True)["stages_ms"] for size, corpus in sizes.items() for d in ("cuda", "cpu")}
    with_rerank = {key: (timing.lexical_ms(key[0]) + s["project"]["p50"]) / 1e3 for key, s in stage.items()}
    no_rerank = {key: (s["total"]["p50"] - s["encode"]["p50"] - s["search"]["p50"]) / 1e3 for key, s in stage.items()}
    short = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e4 else f"{n / 1e3:.1f}K" if n >= 1e3 else f"{n:.0f}"
    print("\n**Break-even at budget 60, short passages.** What the bridge adds per query, p50, through `Searcher`, 1M and 8.84M. If the re-indexed system also reranks about 60: BM25 and the projection, "
          f"{1e3 * with_rerank[1_000_000, 'cuda']:.0f} and {1e3 * with_rerank[8_841_823, 'cuda']:.0f} ms on the GPU, {1e3 * with_rerank[1_000_000, 'cpu']:.0f} and {1e3 * with_rerank[8_841_823, 'cpu']:.0f} ms on the CPU. "
          f"If it doesn't rerank: {no_rerank[1_000_000, 'cuda']:.2f} and {no_rerank[8_841_823, 'cuda']:.2f} s on the GPU, {no_rerank[1_000_000, 'cpu']:.1f} and {no_rerank[8_841_823, 'cpu']:.1f} s on the CPU. Each pair of cells below is GPU, CPU.\n")
    print("| Query model | Re-index 1M | It also reranks | It doesn't | Re-index 8.84M | It also reranks | It doesn't |\n| --- | --- | --- | --- | --- | --- | --- |")
    for m in ("minilm", "bge-base-en", "bge-large-en", "qwen3-0.6b"):
        cells = []
        for size in sizes:
            sec = {d: size * rate[m, d]["tokens_per_doc"] / rate[m, d]["tokens_per_s"] for d in ("cuda", "cpu")}
            cells += [", ".join(f"{sec[d] / 60:.0f} min" if sec[d] < 3600 else f"{sec[d] / 3600:.1f} h" for d in ("cuda", "cpu")),
                      ", ".join(short(sec[d] / with_rerank[size, d]) for d in ("cuda", "cpu")), ", ".join(short(sec[d] / no_rerank[size, d]) for d in ("cuda", "cpu"))]
        print(f"| `{m}` | " + " | ".join(cells) + " |")


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


def b_confirm() -> None:
    """The rule's confirmation: X2 and X3 against Row 7b, 3 seeds, 2 query models, then the comparators with the old model alive."""
    recs = read("m3_b/confirm.jsonl")
    print("\n## Confirmation: 3 seeds, K = 2,048, `bge-large-en` index, absolute `nDCG@5`\n")
    print("Mean ± s.d. over seeds. Gain is points over Row 7b, paired by seed, with the seeds that gain.\n")
    print("| Set | Query model | Budget | Row 7b | X2: BM25, opus-mt | Gain | X3: BM25, MADLAD | Gain |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    idx = defaultdict(dict)
    for r in recs:
        idx[(r["dataset"], r["tgt"], r["budget"], r["candidate"])][r["seed"]] = r["metrics"]["ndcg@5"]
    for ds, tgt, budget in sorted({k[:3] for k in idx}):
        base = idx[(ds, tgt, budget, "7b")]
        cols = [f"{st.mean(base.values()):.3f} ± {st.stdev(base.values()):.3f}"]
        for c in ("X2", "X3"):
            v = idx[(ds, tgt, budget, c)]
            d = [100 * (v[s] - base[s]) for s in v]
            cols += [f"{st.mean(v.values()):.3f} ± {st.stdev(v.values()):.3f}", f"{st.mean(d):+.1f} ± {st.stdev(d):.1f}, {sum(x > 0 for x in d)}/{len(d)}"]
        print(f"| {ds} | {tgt} | {budget} | " + " | ".join(cols) + " |")
    print("\n**Re-index on the same query model, for scale:**\n")
    for (ds, tgt, budget, c), v in sorted(idx.items()):
        if c == "reindex" and budget == 60:
            print(f"- {ds}, {tgt}, budget 60: {st.mean(v.values()):.3f}")
    old = {(r["dataset"], r["candidate"], r["budget"]): r["metrics"]["ndcg@5"] for r in read("m3_b/old_model.jsonl")}
    print("\n## With the old model alive, for D4: absolute `nDCG@5`. No bridge, no seed, reranker reading the original query\n")
    print("| Set | Budget | Native (human English) | Old model on opus-mt | + BM25 on it | Old model on MADLAD | + BM25 on it |\n| --- | --- | --- | --- | --- | --- | --- |")
    for ds, budget in sorted({(k[0], k[2]) for k in old}):
        print(f"| {ds} | {budget} | " + " | ".join(f"{old[(ds, c, budget)]:.3f}" for c in ("native", "TMo", "TBo", "TMm", "TBm")) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "b_confirm":
        b_confirm()
        sys.exit()
    {"a6": a6, "1m": one_m, "full": full, "break_even": break_even, "b": b, "table1": lambda: table1(int(sys.argv[2]) if len(sys.argv) > 2 else 60)}[sys.argv[1]]()
