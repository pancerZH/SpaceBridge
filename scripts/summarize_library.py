"""What a user of the library gets, by index size: M2 Table 1 (the library's pool, retention on the hardest corpus measured there,
one query's time on CPU and GPU) and Table 2 (the smallest pool that passes), the rule of thumb and the break-even.

Retention comes from Row 9 records. Times come from query_timing.py records, added up: a query's time is the rerank run at its
pool plus the search time at its size, because rerank depends on pool and document length, not on index size. "Short" and "long"
are MS MARCO passages and TREC-COVID abstracts through the reranker; the long column combines TREC-COVID's rerank time with
pools from short-passage corpora, because no long-document corpus has been measured above 171K.

    uv run python scripts/summarize_library.py outputs/phase1/m1_timing outputs/phase1/m2_timing
"""
import json
import sys
from pathlib import Path

import numpy as np

from spacebridge.quality import size_rerank_pool
from spacebridge_bench import durable

ROOT = Path("outputs/phase1")
LINES = {"1": 0.98, "3": 0.94}
TIER1 = {"bge-base-en", "bge-large-en"}
# Size, label, then for each corpus measured there: its name, the K of the library's profile at that size, and the record files.
SETS = [
    (5_183, "5.2K", [("SciFact", 2048, ["w5_grid/grid_scifact"])]),
    (8_674, "8.7K", [("ArguAna", 2048, ["w5_grid/grid_arguana"])]),
    (10_000, "10K", [("MS MARCO", 2048, ["m1_grid/ms_10k"])]),
    (20_000, "20K", [("MS MARCO", 2048, ["m1_grid/ms_20k"])]),
    (25_657, "25.7K", [("SciDocs", 2048, ["w5_grid/grid_scidocs"])]),
    (50_000, "50K", [("MS MARCO", 2048, ["w5_grid/grid_msmarco"]), ("NQ", 2048, ["w5_grid/grid_nq"]), ("HotpotQA", 2048, ["m1_switch/hotpot_k2048"])]),
    (57_638, "57.6K", [("FiQA", 2048, ["w5_grid/grid_fiqa"])]),
    (100_000, "100K", [("MS MARCO", 2048, ["m1_grid/ms_100k"]), ("NQ", 2048, ["m1_grid/nq_100k"])]),
    (150_000, "150K", [("MS MARCO", 2048, ["m2_lib/ms150k", "m2_lib/ms150k_deep"]), ("NQ", 2048, ["m2_lib/nq150k", "m2_lib/nq150k_deep"])]),
    (171_332, "171K", [("TREC-COVID", 8192, ["m1_switch/trec_k8192"])]),
    (200_000, "200K", [("MS MARCO", 8192, ["m1_grid/ms_200k_k8192"]), ("NQ", 8192, ["m1_grid/nq_200k_k8192"])]),
    (500_000, "500K", [("MS MARCO", 8192, ["m1_grid/ms_500k", "m2_pools/ms500k"]), ("NQ", 8192, ["m1_grid/nq_500k", "m2_pools/nq500k"])]),
    (1_000_000, "1M", [
        ("MS MARCO", 8192, ["w3_1m/ms1m_e2e", "w3_1m/ms1m_e2e_more", "m2_pools/ms1m"]),
        ("NQ", 8192, ["w3_1m/nq1m_e2e", "w3_1m/nq1m_e2e_more", "m2_pools/nq1m"]),
        ("HotpotQA", 8192, ["m2_hotpot/e2e"]),
    ]),
]
TABLE1_SIZES = {5_183, 10_000, 20_000, 50_000, 100_000, 150_000, 171_332, 200_000, 500_000, 1_000_000}


def retention_by_pool(files: list[str], K: int) -> dict:
    """tier -> pool -> pair -> records over seeds, with later files replacing earlier ones."""
    out = {"1": {}, "3": {}}
    for stem in files:
        path = ROOT / f"{stem}.jsonl"
        if not path.exists():
            continue
        for r in durable.read_jsonl(path)[0]:
            if r.get("reranker") != "bge-reranker-v2-m3" or r["K"] != K:
                continue
            tier = "1" if {r["src"], r["tgt"]} <= TIER1 else "3"
            out[tier].setdefault(r["depth"], {}).setdefault(f"{r['src']} ← {r['tgt']}", {})[r["seed"]] = r
    return out


def summary(pairs: dict) -> dict:
    """Mean over pairs, the worst pair, and that pair's 95% CI (the seeds' query-bootstrap bounds, averaged)."""
    rho = {p: np.mean([r["rho_ceil"]["ndcg@5"] for r in s.values()]) for p, s in pairs.items()}
    worst = min(rho, key=rho.get)
    seeds = pairs[worst].values()
    return {
        "mean": float(np.mean(list(rho.values()))), "worst": float(rho[worst]), "worst_pair": worst, "n_pairs": len(rho),
        "lo": float(np.mean([r["ci_ceil"]["ndcg@5"][0] for r in seeds])), "hi": float(np.mean([r["ci_ceil"]["ndcg@5"][1] for r in seeds])),
        "abs": float(np.mean([r["metrics"]["ndcg@5"] for s in pairs.values() for r in s.values()])),
        "n_q": next(iter(seeds))["n_queries"],
    }


def table2(sets: list) -> None:
    print("\n**Table 2: the smallest pool that passes.** Setting this by hand needs `pool=`; the library doesn't choose it.\n")
    print("| Size | Corpus | Queries | Tier 1 | Tier 3 | Pool for both | Tier 1 there [worst pair's 95% CI] | Tier 3 there [worst pair's 95% CI] |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, label, corpora in sets:
        for name, K, files in corpora:
            by = retention_by_pool(files, K)
            if not by["1"] and not by["3"]:
                continue
            smallest = {}
            for t in by:
                passing = [q for q in sorted(by[t]) if summary(by[t][q])["mean"] >= LINES[t]]
                smallest[t] = passing[0] if passing else None
            both = max(smallest["1"] or 10_000, smallest["3"] or 10_000)
            cell = lambda t: (lambda s: f"{s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")(summary(by[t][both])) if both in by[t] else "-"
            n_q = summary(next(iter(by["3"].values())))["n_q"] if by["3"] else "-"
            fmt = lambda v: v if v else "none tested"
            print(f"| {label} | {name} | {n_q} | {fmt(smallest['1'])} | {fmt(smallest['3'])} | {both if both < 10_000 else 'none tested'} | {cell('1')} | {cell('3')} |")


def stages(r: dict, q: str) -> dict:
    return {k: v[q] for k, v in r["stages_ms"].items()}


class Timing:
    def __init__(self, records: list[dict]):
        self.records = records

    def search_ms(self, device: str, size: int) -> float:
        hnsw = [r for r in self.records if r["device"] == device and not r["rerank"] and r["backend"] != "exact" and r["threads"] == 10 and r["ef"] == 500]
        if size == 1_000_000 and hnsw:
            return stages(hnsw[-1], "p50")["search"]
        pts = sorted((r["N"], stages(r, "p50")["search"]) for r in self.records if r["device"] == device and not r["rerank"] and r["backend"] == "exact")
        return float(np.interp(size, *zip(*pts)))

    def run(self, device: str, corpus: str, pool: int, threads: int = 10) -> dict | None:
        runs = [r for r in self.records if r["device"] == device and r["threads"] == threads and r["rerank"] and r["dataset"] == corpus and r["pool"] == pool]
        return runs[-1] if runs else None

    def query_s(self, device: str, corpus: str, size: int, pool: int, q: str) -> float:
        r = self.run(device, corpus, pool)
        if r is None:
            raise KeyError((device, corpus, pool))
        s = stages(r, q)
        return (s["total"] - s["search"] + self.search_ms(device, size)) / 1e3


def table1(timing: Timing, sets: list, pools_override: dict) -> None:
    print("**Table 1: what the library does.** Seconds for one query, p50 (p95). The CPU columns stop at the first size over 60 s.\n")
    print("| Size | Library's pool | Hardest corpus | Tier 1 (line 0.98) | Tier 3 (line 0.94) | CPU, short | CPU, long | GPU, short | GPU, long |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    stopped = {"short": False, "long": False}
    for size, label, corpora in sets:
        if size not in TABLE1_SIZES:
            continue
        pool = pools_override.get(size, size_rerank_pool(size))
        reads = {}
        for name, K, files in corpora:
            by = retention_by_pool(files, K)
            for t in by:
                if pool in by[t]:
                    reads.setdefault(t, []).append((summary(by[t][pool]), name))
        retention = {}
        for t in ("1", "3"):
            if t in reads:
                s, name = min(reads[t], key=lambda x: x[0]["mean"])
                retention[t] = (s, name)
        names = sorted({n for _, n in retention.values()})
        cell = lambda t: (lambda s: f"{s['mean']:.3f}{' **miss**' if s['mean'] < LINES[t] else ''}")(retention[t][0]) if t in retention else "-"
        times = {}
        for kind, corpus in (("short", "msmarco-1m"), ("long", "trec-covid")):
            cpu50 = timing.query_s("cpu", corpus, size, pool, "p50")
            if stopped[kind]:
                times[f"cpu_{kind}"] = "stop"
            else:
                over = cpu50 > 60
                text = f"{cpu50:.1f} ({timing.query_s('cpu', corpus, size, pool, 'p95'):.1f})"
                times[f"cpu_{kind}"] = f"**{text}**" if over else text
                stopped[kind] = over
            times[f"gpu_{kind}"] = f"{timing.query_s('cuda', corpus, size, pool, 'p50'):.2f} ({timing.query_s('cuda', corpus, size, pool, 'p95'):.2f})"
        print(f"| {label} | {pool} | {', '.join(names)} | {cell('1')} | {cell('3')} | {times['cpu_short']} | {times['cpu_long']} | {times['gpu_short']} | {times['gpu_long']} |")


def table2_times(timing: Timing) -> None:
    print("\n**Time by pool, measured:** seconds for one query at 1M through HNSW (short) and at 171K exact (long), p50 (p95).\n")
    pools = sorted({r["pool"] for r in timing.records if r["rerank"] and r["threads"] == 10})
    print("| Pool | CPU, short | CPU, long | GPU, short | GPU, long |\n| --- | --- | --- | --- | --- |")
    for pool in pools:
        cells = []
        for device, corpus in (("cpu", "msmarco-1m"), ("cpu", "trec-covid"), ("cuda", "msmarco-1m"), ("cuda", "trec-covid")):
            r = timing.run(device, corpus, pool)
            cells.append(f"{r['stages_ms']['total']['p50'] / 1e3:.2f} ({r['stages_ms']['total']['p95'] / 1e3:.2f})" if r else "-")
        print(f"| {pool} | " + " | ".join(cells) + " |")


def tokens(timing: Timing) -> None:
    print("\n**Columns, measured tokens** (candidates at each record's pool, truncated at 512):\n")
    print("| Corpus | Tokens per document | Tokens per pair | Pools | Rerank ms per candidate, CPU 10 threads |\n| --- | --- | --- | --- | --- |")
    for corpus in sorted({r["dataset"] for r in timing.records if "tokens_per_pair" in r}):
        rs = [r for r in timing.records if r["dataset"] == corpus and "tokens_per_pair" in r and r["device"] == "cpu" and r["threads"] == 10]
        if not rs:
            continue
        doc, pair = np.mean([r["tokens_per_doc"] for r in rs]), np.mean([r["tokens_per_pair"] for r in rs])
        per = np.mean([stages(r, "p50")["rerank"] / r["pool"] for r in rs])
        print(f"| {corpus} | {doc:.0f} | {pair:.0f} | {sorted(r['pool'] for r in rs)} | {per:.0f} |")


def rule_of_thumb(timing: Timing) -> None:
    """Rerank seconds = a × pool × tokens per pair, fitted over every CPU record at 10 threads; error per record."""
    rs = [r for r in timing.records if "tokens_per_pair" in r and r["device"] == "cpu" and r["threads"] == 10]
    x = np.array([r["pool"] * r["tokens_per_pair"] for r in rs])
    y = np.array([stages(r, "p50")["rerank"] / 1e3 for r in rs])
    a = float((x * y).sum() / (x * x).sum())
    err = (a * x - y) / y
    leave_out = {}
    for corpus in sorted({r["dataset"] for r in rs}):
        keep = np.array([r["dataset"] != corpus for r in rs])
        a_lo = (x[keep] * y[keep]).sum() / (x[keep] ** 2).sum()
        leave_out[corpus] = float(np.mean(np.abs(a_lo * x[~keep] - y[~keep]) / y[~keep]))
    print(f"\n**Rule of thumb:** rerank s = {a * 1e3:.2f} ms × pool × tokens per pair. Over {len(rs)} records: mean absolute error {100 * np.abs(err).mean():.1f}%, worst {100 * np.abs(err).max():.1f}%. "
          f"One query reaches 60 s at pool × tokens ≈ {60 / a:,.0f}. Leave one corpus out: " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in leave_out.items()))
    for r, e in zip(rs, err):
        print(f"  {r['dataset']:<12} pool {r['pool']:>3}  tokens/pair {r['tokens_per_pair']:6.1f}  measured {stages(r, 'p50')['rerank'] / 1e3:7.2f} s  fit error {100 * e:+.1f}%")


def threads_gain(timing: Timing) -> None:
    print("\n**16 threads against 10, CPU, p50:**\n")
    print("| Corpus | Pool | 10 threads | 16 threads | Gain |\n| --- | --- | --- | --- | --- |")
    for r in timing.records:
        if r["device"] == "cpu" and r["threads"] == 16 and r["rerank"]:
            base = timing.run("cpu", r["dataset"], r["pool"])
            if base:
                a, b = stages(base, "p50")["total"] / 1e3, stages(r, "p50")["total"] / 1e3
                print(f"| {r['dataset']} | {r['pool']} | {a:.2f} s | {b:.2f} s | {100 * (a - b) / a:.0f}% |")


def break_even(timing: Timing, mount: list[dict], new_pool: int) -> None:
    rate = {(r["model"], r["device"]): r for r in mount if r.get("mode") == "rate"}
    extra = {d: (stages(timing.run(d, "msmarco-1m", new_pool), "p50")["rerank"] - stages(timing.run(d, "msmarco-1m", 60), "p50")["rerank"]) / 1e3 for d in ("cuda", "cpu")}
    print(f"\n**Break-even, pool {new_pool} against 60:** extra rerank GPU {extra['cuda']:.3f} s, CPU {extra['cpu']:.1f} s per query.\n")
    print("| Query model | 500K: GPU, CPU | 1M: GPU, CPU |\n| --- | --- | --- |")
    for m in ("minilm", "bge-base-en", "bge-large-en", "qwen3-0.6b"):
        cell = lambda N: ", ".join(f"{N * rate[m, d]['tokens_per_doc'] / rate[m, d]['tokens_per_s'] / extra[d]:,.0f}" for d in ("cuda", "cpu"))
        print(f"| `{m}` | {cell(500_000)} | {cell(1_000_000)} |")


def main() -> None:
    roots = [Path(p) for p in sys.argv[1:]]
    records = [r for root in roots if (root / "timing.jsonl").exists() for r in durable.read_jsonl(root / "timing.jsonl")[0]]
    mount = [r for root in roots if (root / "mount.jsonl").exists() for r in durable.read_jsonl(root / "mount.jsonl")[0]]
    timing = Timing(records)
    table1(timing, SETS, {})
    table2(SETS)
    table2_times(timing)
    tokens(timing)
    rule_of_thumb(timing)
    threads_gain(timing)
    break_even(timing, mount, size_rerank_pool(1_000_000))


if __name__ == "__main__":
    main()
