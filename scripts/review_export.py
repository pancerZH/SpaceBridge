"""Writes results/ from the local records in outputs/: everything the published result tables need, and nothing else.

    python scripts/review_export.py              results/ from outputs/review/ and outputs/phase1/ (needs the local records)
    python scripts/review_export.py --datasets   also recounts each run's corpus from the cached datasets (a few minutes)

After this, `scripts/review_tables.py` fills the documents from results/ alone, and `scripts/verify_results.py` checks results/ against
itself, so a public checkout can do both. results/README.md describes every file.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retention_ci import load_cells, summarize  # noqa: E402

REVIEW = Path("outputs/review")
PHASE1 = Path("outputs/phase1")
RESULTS = Path("results")
QUANTUM = 10_000  # per-query scores are stored as integers in units of 1/10,000
FRESH_PAIRS = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en"), ("bge-large-en", "qwen3-0.6b"), ("qwen3-0.6b", "bge-large-en")}
FRESH_SEEDS = (0, 1, 2)
RETENTION_KEYS = [
    "source", "dataset", "row", "family", "n_docs", "n_queries", "pairs", "seeds", "cells", "missing_cells", "retention", "retention_low", "retention_high", "line",
    "verdict", "seed_sd", "worst_pair", "worst_pair_name", "strict", "strict_low", "strict_high", "vs_reindex", "vs_reindex_low", "vs_reindex_high", "gap_reindex",
    "gap_reindex_low", "gap_reindex_high", "no_bridge_retention", "no_bridge_low", "no_bridge_high", "bridge_gain", "bridge_gain_low", "bridge_gain_high",
    "abs_bridge", "abs_native", "abs_oracle", "abs_native_bm25", "abs_oracle_bm25", "abs_no_bridge",
]
HNSW_TIERS = (("bge-base-en", "same", 0.95), ("qwen3-0.6b", "across", 0.90))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.open():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def runs() -> list[Path]:
    """Every run directory of per-query scores behind a published table: the recomputed cells and the fresh confirmation."""
    return sorted(d for d in (REVIEW / "per_query").iterdir() if d.name.startswith(("recompute_", "fresh_")) and any(d.glob("*.npz")))


def expected_cells(name: str) -> set[tuple]:
    """The planned (index model, query model, seed) cells of a run: the original record file's, or the confirmation plan's grid."""
    if name.startswith("fresh_"):
        return {(s, t, seed) for s, t in FRESH_PAIRS for seed in FRESH_SEEDS}
    import review_recompute

    file = next(f for f in review_recompute.FILES if review_recompute.name_of(f) == name.removeprefix("recompute_"))
    return {(r["src"], r["tgt"], r["seed"]) for r in review_recompute.records(PHASE1 / f"{file}.jsonl")}


def check_one_query_list(name: str) -> dict:
    """A run's manifest lines must agree on one configuration, so every cell in it scored the same query list."""
    lines = [m for m in read_jsonl(REVIEW / f"{name}.jsonl.manifest.jsonl") if "config" in m]
    ids = {m["config_id"] for m in lines}
    if len(ids) != 1:
        raise ValueError(f"{name}: its manifest names {len(ids)} configurations, so its cells may not share one query list")
    return lines[0]


def export_retention() -> tuple[list[dict], dict]:
    rows, plans = [], {}
    for d in runs():
        manifest = check_one_query_list(d.name)
        plan = expected_cells(d.name)
        plans[d.name] = sorted("|".join(map(str, k)) for k in plan)
        for row in ("9h", "9"):
            cells = [c for c in load_cells(d, row) if c["scores"]]
            if not cells:
                continue
            for r in summarize(cells, manifest["config"]["n_docs"], expected=plan):
                r.update(source=d.name, dataset=cells[0]["dataset"], row=row, missing_cells=";".join(r["missing_cells"]))
                rows.append(r)
    rows += hnsw_rows()
    with (RESULTS / "retention.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, RETENTION_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)  # full precision: tables round once, when they print
    return rows, plans


def hnsw_rows() -> list[dict]:
    """The fresh HNSW check, with the same joint query bootstrap: per-query scores from the rerun that saved them. Its records match the
    first run's exactly, which verify_results.py checks."""
    res = {r["key"]: r for r in read_jsonl(REVIEW / "followup/hnsw.jsonl")}
    with np.load(REVIEW / "followup/hnsw_per_query.npz") as z:
        pq = {k: z[k].astype(np.float64) for k in z.files}
    nat = pq["hnsw|native|ef500|pool60/exact"]
    n = len(nat)
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1.0 / n), size=2_000) / n
    out = []
    for tgt, fam, line in HNSW_TIERS:
        seeds = [pq[f"lexical|{tgt}|s{s}|ef500|pool60/approx"] for s in FRESH_SEEDS]
        ora = pq[f"oracle|{tgt}|pool60/exact"]
        point = float(np.mean([x.mean() for x in seeds])) / min(nat.mean(), ora.mean())
        record_point = float(np.mean([res[f"lexical|{tgt}|s{s}|ef500|pool60"]["ndcg5_approx"] for s in FRESH_SEEDS])) / min(
            res["hnsw|native|ef500|pool60"]["ndcg5_exact"], res[f"oracle|{tgt}|pool60"]["ndcg5_exact"])
        assert abs(point - record_point) < 1e-9, (point, record_point)
        reps = np.mean([w @ x for x in seeds], axis=0) / np.minimum(w @ nat, w @ ora)
        low, high = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
        out.append({
            "source": "fresh_msmarco_hnsw", "dataset": "msmarco-full-fresh", "row": "9h", "family": fam, "n_docs": 8_841_823, "n_queries": n, "pairs": 1,
            "seeds": len(FRESH_SEEDS), "cells": len(FRESH_SEEDS), "missing_cells": "", "retention": point, "retention_low": low, "retention_high": high, "line": line,
            "verdict": "miss" if point < line else "confirmed" if low >= line else "point pass", "abs_bridge": float(np.mean([x.mean() for x in seeds])),
            "abs_native": float(nat.mean()), "abs_oracle": float(ora.mean()),
        })
    return out


def export_cells() -> None:
    """Each cell's mean nDCG@5 by row and system, at full precision: what point estimates and cell counts are computed from."""
    with (RESULTS / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "cell", "row", "system", "ndcg5_mean", "n_queries"])
        for d in runs():
            for path in sorted(d.glob("*.npz")):
                with np.load(path) as z:
                    for name in sorted(z.files):
                        row, system, metric = name.split("/")
                        if metric == "ndcg@5" and row in ("9", "9h"):
                            w.writerow([d.name, path.stem, row, system, repr(float(z[name].astype(np.float64).mean())), len(z[name])])


def export_per_query() -> None:
    """Per-query nDCG@5 of each published cell and of the HNSW check, in units of 1/10,000: scores only, in each run's query order."""
    out = RESULTS / "per_query"
    out.mkdir(parents=True, exist_ok=True)
    for d in runs():
        arrays = {}
        for path in sorted(d.glob("*.npz")):
            with np.load(path) as z:
                for name in z.files:
                    row, system, metric = name.split("/")
                    if metric == "ndcg@5" and row in ("9", "9h"):
                        arrays[f"{path.stem}/{row}/{system}"] = np.round(z[name] * QUANTUM).astype(np.uint16)
        np.savez_compressed(out / f"{d.name}.npz", **arrays)
    with np.load(REVIEW / "followup/hnsw_per_query.npz") as z:
        np.savez_compressed(out / "fresh_msmarco_hnsw.npz", **{k: np.round(z[k] * QUANTUM).astype(np.uint16) for k in z.files if k.endswith("pool60/approx") or k.endswith("pool60/exact")})


def export_records() -> None:
    """Small study records that tables read directly, copied as they are, with the configuration files that tie them to their runs."""
    copies = {
        REVIEW / "followup/hnsw.jsonl": "fresh_msmarco_hnsw.jsonl", REVIEW / "followup/hnsw.jsonl.config.json": "fresh_msmarco_hnsw.config.json",
        REVIEW / "followup/precision_cpu.jsonl": "precision_cpu.jsonl",
    }
    for src, dst in copies.items():
        shutil.copyfile(src, RESULTS / dst)
    with (RESULTS / "stage1.jsonl").open("w") as f:
        for name in ("fiqa", "scidocs", "nq"):
            for r in read_jsonl(REVIEW / f"stage1_{name}.jsonl"):
                f.write(json.dumps(r) + "\n")


def export_manifests(plans: dict) -> None:
    with (RESULTS / "manifests.jsonl").open("w") as f:
        for path in sorted(REVIEW.glob("*.jsonl.manifest.jsonl")):
            first = read_jsonl(path)
            if first:
                run = path.name.split(".jsonl")[0]
                f.write(json.dumps({"run": run, **first[0], "expected_cells": plans.get(run)}) + "\n")


def export_timing() -> None:
    with (RESULTS / "timing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "device", "dataset", "documents", "backend", "pool", "queries", "stage", "p50_ms", "p95_ms", "tokens_per_pair", "peak_rss_gb", "peak_vram_gb", "load_s"])
        for path in sorted(REVIEW.glob("timing_*.jsonl")):
            for r in read_jsonl(path):
                for stage, v in r["stages_ms"].items():
                    w.writerow([path.stem, r["device"] if r.get("search_device") in (None, r["device"]) else f"{r['device']}, search on {r['search_device']}", r["dataset"], r["N"],
                                "HNSW" if r["backend"] != "exact" else "exact", r["pool"], r["n"], stage, v["p50"], v["p95"],
                                r.get("tokens_per_pair", 0), r.get("peak_rss_gb", 0), r.get("peak_vram_gb", 0) or 0, r.get("load_s", 0)])


def export_setup() -> None:
    """What the break-even estimate reads: encoding rates, mount times, and the HNSW and BM25 builds."""
    with (RESULTS / "setup.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model", "device", "documents", "K", "tokens_per_doc", "tokens_per_s", "seconds", "source"])
        for r in read_jsonl(PHASE1 / "m1_timing/mount.jsonl"):
            if r.get("mode") == "rate":
                w.writerow(["encode_rate", r["model"], r["device"], r["N"], "", r["tokens_per_doc"], r["tokens_per_s"], "", "outputs/phase1/m1_timing/mount.jsonl"])
            elif r.get("mode") == "mount":
                w.writerow(["mount", r["model"], r["device"], r["N"], r["K"], "", "", r["mount_s"], "outputs/phase1/m1_timing/mount.jsonl"])
        build = next((r for r in read_jsonl(PHASE1 / "w5_ann/ms1m.jsonl") if r.get("key") == "build"), None)
        if build:
            w.writerow(["hnsw_build", "bge-large-en", "cpu", 1_000_000, "", "", "", build["build_s_this_run"], "outputs/phase1/w5_ann/ms1m.jsonl"])
        w.writerow(["hnsw_build", "bge-large-en", "cpu", 8_841_823, "", "", "", 1.8 * 3600, "the 8.84M build, about 1.8 hours on 12 threads"])
        w.writerow(["bm25_build", "", "cpu", 1_000_000, "", "", "", 22.0, "LexicalIndex.build, 14 to 22 s at 1M; the upper end"])
        w.writerow(["bm25_build", "", "cpu", 8_841_823, "", "", "", 128.0, "LexicalIndex.build at 8.84M"])


def export_datasets() -> None:
    """Per run: how its corpus is made, and digests of its rows, judgments and queries. The query digest must match the run's manifest."""
    import shlex

    sys.path.insert(0, "src")
    from spacebridge_bench import provenance
    from spacebridge_bench.datasets import load_beir
    from spacebridge_bench.run import _first_docs, _first_queries, dataset_counts

    commands = dict(line.split(" ", 1) for line in (REVIEW / "recompute_commands.txt").read_text().splitlines() if line.strip())
    plan = {f"recompute_{name}": shlex.split(args) for name, args in commands.items()}
    plan.update({"fresh_msmarco": ["--dataset", "msmarco-full-fresh"], "fresh_hotpotqa": ["--dataset", "hotpotqa-full-fresh"], "fresh_lotte": ["--dataset", "lotte-technology"],
                 "fresh_fever": ["--dataset", "fever-1m"], "fresh_touche": ["--dataset", "touche2020"]})
    loaded: dict = {}
    rows = []
    base_of = lambda argv: argv[argv.index("--dataset") + 1]
    for name, argv in sorted(plan.items(), key=lambda kv: (base_of(kv[1]), kv[0])):  # each corpus loads once
        opt = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None
        base = opt("--dataset")
        if base not in loaded:
            loaded.clear()
            loaded[base] = load_beir(base)
        ds = loaded[base]
        if opt("--max-docs"):
            ds = _first_docs(ds, int(opt("--max-docs")))
        if opt("--max-queries"):
            ds = _first_queries(ds, int(opt("--max-queries")))
        manifest = check_one_query_list(name)
        digest = provenance.query_digest(ds.queries)
        if digest != manifest["config"]["queries_sha256"]:
            raise ValueError(f"{name}: the queries recounted here aren't the ones the run scored")
        counts = dataset_counts(ds)
        rows.append({"run": name, "dataset": ds.name, **counts, "queries_sha256": digest, "corpus_sha256": provenance.corpus_digest(ds.doc_ids, ds.titles[: ds.N], ds.texts[: ds.N]),
                     "qrels_sha256": provenance.qrels_digest(ds.qrels)})
        print(f"{name}: {counts['documents']:,} documents, {counts['positive_documents']:,} positive", flush=True)
    with (RESULTS / "datasets.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_checksums() -> None:
    lines = []
    for path in sorted(p for p in RESULTS.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "README.md")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(RESULTS)}")
    (RESULTS / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", action="store_true", help="also rewrite results/datasets.csv from the cached datasets")
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    _, plans = export_retention()
    export_cells()
    export_per_query()
    export_records()
    export_manifests(plans)
    export_timing()
    export_setup()
    if a.datasets or not (RESULTS / "datasets.csv").exists():
        export_datasets()
    write_checksums()
    print(f"wrote {RESULTS}/")


if __name__ == "__main__":
    main()
