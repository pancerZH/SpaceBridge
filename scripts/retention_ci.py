"""Retention with real intervals, from aligned per-query scores (``spacebridge_bench.run --per-query-dir``).

For one corpus and size, and one model family: the point estimate is the mean over pairs of the mean over seeds of bridge / min(native, re-index), as the
lines define it. The interval is a percentile bootstrap over queries: each replicate draws one resample of the queries, used by every pair and seed, retakes the
weaker baseline in each, and recomputes that whole average. It is conditional on these models, this corpus and these seeds. The seeds' own spread is reported
separately: it does not vary the evaluation sample, only the anchors, split and distractors.

    python scripts/retention_ci.py outputs/review/per_query/<name> ... --row 9h --out table.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SAME_FAMILY = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")}
SYSTEMS = ("bridge", "native", "oracle", "native_bm25", "oracle_bm25", "no_bridge")
BOOTSTRAPS = 2_000
LINES = {"up_to_1m": {"same": 0.98, "across": 0.94}, "above_1m": {"same": 0.95, "across": 0.90}}


def family(src: str, tgt: str) -> str:
    return "same" if (src, tgt) in SAME_FAMILY else "other" if src == "minilm" else "across"


def load_cells(directory: Path, row: str, metric: str = "ndcg@5") -> list[dict]:
    cells = []
    for path in sorted(directory.glob("*.npz")):
        dataset, src, tgt, _, seed = path.stem.split("__")
        with np.load(path) as z:
            scores = {s: z[f"{row}/{s}/{metric}"] for s in SYSTEMS if f"{row}/{s}/{metric}" in z.files}
        cells.append({"dataset": dataset, "src": src, "tgt": tgt, "seed": int(seed[1:]), "scores": scores})
    return cells


def verdict(point: float, low: float, line: float) -> str:
    return "confirmed" if low >= line else "point pass" if point >= line else "miss"


def summarize(cells: list[dict], n_docs: int, boot: int = BOOTSTRAPS, seed: int = 0) -> list[dict]:
    """One row per family. Means here are of per-query scores, so a replicate's mean is a weighted mean of the same scores."""
    out = []
    n = len(cells[0]["scores"]["bridge"])
    # Each replicate is one draw of the queries: counts per query, shared by every pair and seed.
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=boot).astype(np.float64) / n
    for fam in ("same", "across", "other"):
        chosen = [c for c in cells if family(c["src"], c["tgt"]) == fam]
        if not chosen:
            continue

        def average(ratio):
            """The mean over pairs of the mean over seeds of ``ratio(cell's mean scores)``; returns (point, replicates, by pair, by seed)."""
            pairs, seeds = defaultdict(list), defaultdict(list)
            reps = defaultdict(list)
            for c in chosen:
                m = {s: float(v.mean()) for s, v in c["scores"].items()}
                value = ratio(m)
                pairs[c["src"], c["tgt"]].append(value)
                seeds[c["seed"]].append(value)
                r = {s: counts @ v.astype(np.float64) for s, v in c["scores"].items()}
                reps[c["src"], c["tgt"]].append(ratio(r))
            point = float(np.mean([np.mean(v) for v in pairs.values()]))
            replicate = np.mean([np.mean(v, axis=0) for v in reps.values()], axis=0)
            return point, replicate, {k: float(np.mean(v)) for k, v in pairs.items()}, {k: float(np.mean(v)) for k, v in seeds.items()}

        def interval(replicate) -> tuple[float, float]:
            return float(np.nanpercentile(replicate, 2.5)), float(np.nanpercentile(replicate, 97.5))

        def weaker(m, a, b):
            return np.minimum(m[a], m[b])

        ret, ret_b, by_pair, by_seed = average(lambda m: m["bridge"] / weaker(m, "native", "oracle"))
        row = {
            "family": fam, "n_docs": n_docs, "n_queries": n, "pairs": len(by_pair), "seeds": len(by_seed), "cells": len(chosen),
            "retention": ret, "retention_low": interval(ret_b)[0], "retention_high": interval(ret_b)[1],
            "seed_sd": float(np.std(list(by_seed.values()), ddof=1)) if len(by_seed) > 1 else None,
            "worst_pair": min(by_pair.values()),
            "worst_pair_name": " <- ".join(min(by_pair, key=by_pair.get)),
        }
        has_bm25 = all("native_bm25" in c["scores"] for c in chosen)
        if has_bm25:
            v, b, _, _ = average(lambda m: m["bridge"] / weaker(m, "native_bm25", "oracle_bm25"))
            row.update(strict=v, strict_low=interval(b)[0], strict_high=interval(b)[1])
            v, b, _, _ = average(lambda m: m["bridge"] / m["oracle_bm25"])
            row.update(vs_reindex=v, vs_reindex_low=interval(b)[0], vs_reindex_high=interval(b)[1])
            v, b, _, _ = average(lambda m: m["bridge"] - m["oracle_bm25"])
            row.update(gap_reindex=v, gap_reindex_low=interval(b)[0], gap_reindex_high=interval(b)[1])
        if all("no_bridge" in c["scores"] for c in chosen):
            v, b, _, _ = average(lambda m: m["no_bridge"] / weaker(m, "native", "oracle"))
            row.update(no_bridge_retention=v, no_bridge_low=interval(b)[0], no_bridge_high=interval(b)[1])
            v, b, _, _ = average(lambda m: m["bridge"] - m["no_bridge"])
            row.update(bridge_gain=v, bridge_gain_low=interval(b)[0], bridge_gain_high=interval(b)[1])
        for system in SYSTEMS:
            if all(system in c["scores"] for c in chosen):
                row[f"abs_{system}"] = average(lambda m, s=system: m[s])[0]
        if fam in ("same", "across"):
            line = LINES["up_to_1m" if n_docs <= 1_000_000 else "above_1m"][fam]
            row.update(line=line, verdict=verdict(row["retention"], row["retention_low"], line))
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--row", default="9h")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    rows = []
    for d in a.directories:
        cells = load_cells(d, a.row)
        if not cells:
            continue
        name = cells[0]["dataset"]
        manifest = d.parent.parent / f"{d.name}.jsonl.manifest.jsonl"
        n_docs = json.loads(manifest.read_text().splitlines()[0])["config"]["n_docs"]
        for r in summarize(cells, n_docs):
            rows.append({"source": d.name, "dataset": name, "row": a.row, **r})
            print(f"{d.name:42} {r['family']:6} n={r['n_queries']:5} {r['retention']:.3f} [{r['retention_low']:.3f}, {r['retention_high']:.3f}] {r.get('verdict', ''):11} vs re-index {r.get('vs_reindex', float('nan')):.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
