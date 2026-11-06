"""Retention with real intervals, from aligned per-query scores (``spacebridge_bench.run --per-query-dir``).

For one corpus and size, and one model family: the point estimate is the mean over pairs of the mean over seeds of bridge / min(native, re-index), as the
lines define it. The interval is a percentile bootstrap over queries: each replicate draws one resample of the queries, used by every pair and seed, retakes the
weaker baseline in each, and recomputes that whole average. It is conditional on these models, this corpus and these seeds. The seeds' own spread is reported
separately: it does not vary the evaluation sample, only the anchors, split and distractors.

    python scripts/retention_ci.py outputs/review/per_query/<name> ... --row 9h --out table.jsonl
    python scripts/retention_ci.py outputs/review/per_query/<name> --plan results/manifests.jsonl   for runs whose manifest has no plan

A family missing any planned cell reads "incomplete", and the command then exits with an error, so it never confirms a partial run.
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


def verdict(point: float, low: float, line: float, complete: bool = True) -> str:
    if not complete:
        return "incomplete"
    if not np.isfinite(point):
        return "undefined"
    return "confirmed" if low >= line else "point pass" if point >= line else "miss"


def ratio(num, den):
    """num / den, and NaN where the denominator is zero: a replicate with no baseline quality has no retention. Intervals drop those."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def summarize(cells: list[dict], n_docs: int, boot: int = BOOTSTRAPS, seed: int = 0, expected: set | None = None) -> list[dict]:
    """One row per family. Means here are of per-query scores, so a replicate's mean is a weighted mean of the same scores.

    ``expected`` is the planned set of (index model, query model, seed): a family missing any of its planned cells reads "incomplete",
    never a verdict from the cells that happen to be there. Every cell must score the same queries, in the same order."""
    out = []
    n = len(cells[0]["scores"]["bridge"])
    lengths = {len(v) for c in cells for v in c["scores"].values()}
    if lengths != {n}:
        raise ValueError(f"cells score different numbers of queries ({sorted(lengths)}): they aren't aligned on one query list")
    present = {(c["src"], c["tgt"], c["seed"]) for c in cells}
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
                m = {s: float(np.mean(v, dtype=np.float64)) for s, v in c["scores"].items()}  # float64: float32 sums drift in the 8th digit
                value = ratio(m)
                pairs[c["src"], c["tgt"]].append(value)
                seeds[c["seed"]].append(value)
                r = {s: counts @ v.astype(np.float64) for s, v in c["scores"].items()}
                reps[c["src"], c["tgt"]].append(ratio(r))
            point = float(np.mean([np.mean(v) for v in pairs.values()]))
            replicate = np.mean([np.mean(v, axis=0) for v in reps.values()], axis=0)
            return point, replicate, {k: float(np.mean(v)) for k, v in pairs.items()}, {k: float(np.mean(v)) for k, v in seeds.items()}

        planned = {k for k in (expected or set()) if family(k[0], k[1]) == fam}
        missing = sorted(planned - present)

        def interval(replicate) -> tuple[float, float]:
            return float(np.nanpercentile(replicate, 2.5)), float(np.nanpercentile(replicate, 97.5))

        def weaker(m, a, b):
            return np.minimum(m[a], m[b])

        ret, ret_b, by_pair, by_seed = average(lambda m: ratio(m["bridge"], weaker(m, "native", "oracle")))
        row = {
            "family": fam, "n_docs": n_docs, "n_queries": n, "pairs": len(by_pair), "seeds": len(by_seed), "cells": len(chosen),
            "retention": ret, "retention_low": interval(ret_b)[0], "retention_high": interval(ret_b)[1],
            "seed_sd": float(np.std(list(by_seed.values()), ddof=1)) if len(by_seed) > 1 else None,
            "worst_pair": min(by_pair.values()),
            "worst_pair_name": " <- ".join(min(by_pair, key=by_pair.get)),
            "missing_cells": ["|".join(map(str, k)) for k in missing],
        }
        has_bm25 = all("native_bm25" in c["scores"] for c in chosen)
        if has_bm25:
            v, b, _, _ = average(lambda m: ratio(m["bridge"], weaker(m, "native_bm25", "oracle_bm25")))
            row.update(strict=v, strict_low=interval(b)[0], strict_high=interval(b)[1])
            v, b, _, _ = average(lambda m: ratio(m["bridge"], m["oracle_bm25"]))
            row.update(vs_reindex=v, vs_reindex_low=interval(b)[0], vs_reindex_high=interval(b)[1])
            v, b, _, _ = average(lambda m: m["bridge"] - m["oracle_bm25"])
            row.update(gap_reindex=v, gap_reindex_low=interval(b)[0], gap_reindex_high=interval(b)[1])
        if all("no_bridge" in c["scores"] for c in chosen):
            v, b, _, _ = average(lambda m: ratio(m["no_bridge"], weaker(m, "native", "oracle")))
            row.update(no_bridge_retention=v, no_bridge_low=interval(b)[0], no_bridge_high=interval(b)[1])
            v, b, _, _ = average(lambda m: m["bridge"] - m["no_bridge"])
            row.update(bridge_gain=v, bridge_gain_low=interval(b)[0], bridge_gain_high=interval(b)[1])
        for system in SYSTEMS:
            if all(system in c["scores"] for c in chosen):
                row[f"abs_{system}"] = average(lambda m, s=system: m[s])[0]
        if fam in ("same", "across"):
            line = LINES["up_to_1m" if n_docs <= 1_000_000 else "above_1m"][fam]
            row.update(line=line, verdict=verdict(row["retention"], row["retention_low"], line, complete=not missing))
        out.append(row)
    return out


def parse_cell(cell: str) -> tuple:
    """(index model, query model, seed) of a planned cell: "dataset|src|tgt|K|s0" from a run's manifest, or "src|tgt|0" from results/."""
    parts = cell.split("|")
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    return parts[1], parts[2], int(parts[4].removeprefix("s"))


def run_plan(manifest: Path, plan_file: Path | None, name: str) -> tuple[set, int]:
    """The planned cells and corpus size of one run. Its manifest must name one configuration, so its cells share one query list. The plan
    comes from the manifest, or else from a published plan (results/manifests.jsonl) for runs made before manifests recorded it."""
    if not manifest.exists():
        raise SystemExit(f"{name}: no manifest at {manifest}, so nothing says which cells were planned or that they share one query list")
    lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    configs = [m for m in lines if "config" in m]
    ids = {m["config_id"] for m in configs}
    if len(ids) != 1:
        raise SystemExit(f"{name}: its manifest names {len(ids)} configurations, so its cells may not share one query list")
    planned = {parse_cell(c) for m in configs for c in m.get("planned_cells", [])}
    if not planned and plan_file is not None:
        published = {m["run"]: m for m in (json.loads(line) for line in plan_file.read_text().splitlines() if line.strip())}
        if published.get(name, {}).get("expected_cells"):
            if published[name]["config_id"] != configs[0]["config_id"]:
                raise SystemExit(f"{name}: the published plan is for configuration {published[name]['config_id']}, and this run is {configs[0]['config_id']}")
            planned = {parse_cell(c) for c in published[name]["expected_cells"]}
    return planned, configs[0]["config"]["n_docs"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--row", default="9h")
    p.add_argument("--plan", type=Path, default=None, help="a published plan, results/manifests.jsonl, for runs whose own manifest has no planned cells")
    p.add_argument("--allow-unplanned", action="store_true", help="with no plan, judge the cells present; verdicts then say nothing about missing cells")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    rows, incomplete = [], []
    for d in a.directories:
        cells = load_cells(d, a.row)
        if not cells:
            continue
        name = cells[0]["dataset"]
        planned, n_docs = run_plan(d.parent.parent / f"{d.name}.jsonl.manifest.jsonl", a.plan, d.name)
        if not planned and not a.allow_unplanned:
            raise SystemExit(f"{d.name}: no planned cells in its manifest or in --plan, so a missing cell couldn't be detected. Pass --plan, or --allow-unplanned")
        for r in summarize(cells, n_docs, expected=planned or None):
            rows.append({"source": d.name, "dataset": name, "row": a.row, **r})
            if r.get("verdict") == "incomplete":
                incomplete.append(f"{d.name} {r['family']}: missing {', '.join(r['missing_cells'])}")
            print(f"{d.name:42} {r['family']:6} n={r['n_queries']:5} {r['retention']:.3f} [{r['retention_low']:.3f}, {r['retention_high']:.3f}] {r.get('verdict', ''):11} vs re-index {r.get('vs_reindex', float('nan')):.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if incomplete:
        raise SystemExit("incomplete runs, so no verdict:\n" + "\n".join(incomplete))


if __name__ == "__main__":
    main()
