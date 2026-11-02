"""W5: the CPU vs. GPU table by index size and the mount table, from query_timing.py and mount_timing.py records.

A query's time is the rerank run at its pool plus the search time at its size: rerank depends on pool and document length, not on
index size. Pools between two measured ones are interpolated, since rerank time is close to linear in pool, and so is an exact
scan's time in size. CPU mounts not measured are tokens / measured rate + measured load and fit time; "~" marks them.

    uv run python scripts/summarize_timing.py outputs/phase1/m1_timing
"""
import json
import sys
from pathlib import Path

import numpy as np

# Smallest pool meeting the targets on the hardest corpus measured at each size (phase1-m1.md, smallest-pool table).
POOLS = {5_183: 30, 10_000: 10, 20_000: 20, 50_000: 60, 100_000: 100, 171_332: 60, 200_000: 60, 500_000: 300, 1_000_000: 300}
QUERY_TOKENS_ADDED = {"minilm": 0, "bge-base-en": 12, "bge-large-en": 12, "qwen3-0.6b": 20}  # per proxy, for the model's prompt
QUERY_MODELS = ("minilm", "bge-base-en", "bge-large-en", "qwen3-0.6b")


def stages(r: dict, q: str) -> dict:
    return {k: v[q] for k, v in r["stages_ms"].items()}


def query_table(records: list[dict]) -> None:
    def search_ms(device: str, size: int) -> float:
        if size == 1_000_000:
            r = next(r for r in records if r["device"] == device and not r["rerank"] and r["backend"] != "exact")
            return stages(r, "p50")["search"]
        pts = sorted((r["N"], stages(r, "p50")["search"]) for r in records if r["device"] == device and not r["rerank"] and r["backend"] == "exact")
        return float(np.interp(size, *zip(*pts)))

    def query_s(device: str, corpus: str, size: int, pool: int, q: str) -> float:
        runs = sorted((r for r in records if r["device"] == device and r["threads"] == 10 and r["rerank"] and r["dataset"] == corpus), key=lambda r: r["pool"])
        without_search = [stages(r, q)["total"] - stages(r, q)["search"] for r in runs]
        return (float(np.interp(pool, [r["pool"] for r in runs], without_search)) + search_ms(device, size)) / 1e3

    print("| Index size | Pool | CPU, short | CPU, long | GPU, short | GPU, long |\n| --- | --- | --- | --- | --- | --- |")
    for size, pool in POOLS.items():
        cells = []
        for device, fmt in (("cpu", "{:.1f}"), ("cpu", "{:.1f}"), ("cuda", "{:.2f}"), ("cuda", "{:.2f}")):
            corpus = "msmarco-1m" if len(cells) % 2 == 0 else "trec-covid"
            p50, p95 = query_s(device, corpus, size, pool, "p50"), query_s(device, corpus, size, pool, "p95")
            cells.append(f"{fmt.format(p50)} ({fmt.format(p95)})")
        print(f"| {size:,} | {pool} | " + " | ".join(cells) + " |")


def mount_table(records: list[dict]) -> None:
    rate = {(r["model"], r["device"]): r for r in records if r["mode"] == "rate"}
    mounts = {(r["model"], r["device"], r["K"]): r for r in records if r["mode"] == "mount"}
    tokens = {K: sum(mounts["minilm", "cpu", K]["stages_ms"][s] for s in ("encode_proxies_ms", "encode_anchor_docs_ms")) / 1e3 * rate["minilm", "cpu"]["tokens_per_s"] for K in (2048, 8192)}
    fit = {K: mounts["minilm", "cpu", K]["stages_ms"]["fit_ms"] / 1e3 for K in (2048, 8192)}

    def cpu(model: str, K: int) -> str:
        if (model, "cpu", K) in mounts:
            return f"{mounts[model, 'cpu', K]['mount_s'] / 60:.1f} min"
        t = (tokens[K] + QUERY_TOKENS_ADDED[model] * K) / rate[model, "cpu"]["tokens_per_s"]
        return f"~{(t + fit[K] + 4) / 60:.0f} min"

    print(f"\nTokens per mount, from the minilm CPU runs: { {K: round(v) for K, v in tokens.items()} }")
    print("\n| Query model | GPU, K 2,048 | GPU, K 8,192 | CPU, K 2,048 | CPU, K 8,192 | Peak VRAM, K 8,192 |\n| --- | --- | --- | --- | --- | --- |")
    for m in QUERY_MODELS:
        g = lambda K: f"{mounts[m, 'cuda', K]['mount_s']:.0f} s"
        print(f"| `{m}` | {g(2048)} | {g(8192)} | {cpu(m, 2048)} | {cpu(m, 8192)} | {mounts[m, 'cuda', 8192]['peak_vram_gb']:.1f} GB |")
    print("\n| Query model | GPU rate | CPU rate | Re-index 200K: GPU, CPU | 500K: GPU, CPU | 1M: GPU, CPU |\n| --- | --- | --- | --- | --- | --- |")
    span = lambda s: f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"
    for m in QUERY_MODELS:
        g, c = rate[m, "cuda"], rate[m, "cpu"]
        cost = lambda r, N: span(N * r["tokens_per_doc"] / r["tokens_per_s"])
        print(f"| `{m}` | {g['tokens_per_s'] / 1e3:.0f}K tok/s | {c['tokens_per_s'] / 1e3:.2f}K tok/s | " + " | ".join(f"{cost(g, N)}, {cost(c, N)}" for N in (200_000, 500_000, 1_000_000)) + " |")
    return rate


def break_even(records: list[dict], rate: dict) -> None:
    def rerank_ms(device: str, pool: int) -> float:
        runs = sorted((r for r in records if r["device"] == device and r["threads"] == 10 and r["rerank"] and r["dataset"] == "msmarco-1m"), key=lambda r: r["pool"])
        return float(np.interp(pool, [r["pool"] for r in runs], [stages(r, "p50")["rerank"] for r in runs]))

    extra = {d: (rerank_ms(d, 300) - rerank_ms(d, 60)) / 1e3 for d in ("cuda", "cpu")}
    print(f"\nExtra rerank, pool 300 vs. 60: GPU {extra['cuda']:.3f} s, CPU {extra['cpu']:.1f} s per query")
    print("\n| Query model | 500K: GPU, CPU | 1M: GPU, CPU |\n| --- | --- | --- |")
    for m in QUERY_MODELS:
        cell = lambda N: ", ".join(f"{N * rate[m, d]['tokens_per_doc'] / rate[m, d]['tokens_per_s'] / extra[d]:,.0f}" for d in ("cuda", "cpu"))
        print(f"| `{m}` | {cell(500_000)} | {cell(1_000_000)} |")


def main() -> None:
    root = Path(sys.argv[1])
    timing = [json.loads(line) for line in open(root / "timing.jsonl")]
    query_table(timing)
    rate = mount_table([json.loads(line) for line in open(root / "mount.jsonl")])
    break_even(timing, rate)


if __name__ == "__main__":
    main()
