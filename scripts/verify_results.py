"""Checks results/ against itself and against the documents, with nothing but a public checkout: no caches, no models, no outputs/.

    uv run python scripts/verify_results.py

1. Every file matches results/SHA256SUMS.
2. Each run's per-query archive holds exactly the cells its manifest planned, every cell scores the same number of queries, and that
   number and the query digest agree with results/datasets.csv.
3. Every field of every retention row is recomputed. Point estimates, absolute scores, differences, seed spread and the worst pair come
   from the full-precision cell means in results/cells.csv and must match exactly. Every interval bound, the difference intervals
   included, comes from the per-query scores, stored to 1/10,000, and must match within 0.001. Counts, plans and names must match
   exactly. Each verdict must follow from its row's point estimate, low end and line.
4. The HNSW rows: point estimates from their records, intervals from their own per-query archive, and verdicts, the same way.
5. Every table in the README, the report and the design doc is regenerated from results/ and must match the documents exactly. What the
   tables show beyond retention.csv (timings, setup figures, the precision and Stage-1 studies) is regenerated from the records here,
   which are themselves checksummed, but not recomputed from anything rawer.

Exits 1 on the first kind of failure it finds, saying what failed.
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retention_ci import summarize, verdict  # noqa: E402

RESULTS = Path("results")
QUANTUM = 10_000
POINT_TOL, INTERVAL_TOL = 1e-9, 1e-3
POINT_FIELDS = ("retention", "strict", "vs_reindex", "gap_reindex", "no_bridge_retention", "bridge_gain", "abs_bridge", "abs_native", "abs_oracle",
                "abs_native_bm25", "abs_oracle_bm25", "abs_no_bridge", "seed_sd", "worst_pair")
INTERVAL_FIELDS = tuple(f"{base}_{end}" for base in ("retention", "strict", "vs_reindex", "gap_reindex", "no_bridge", "bridge_gain") for end in ("low", "high"))
EXACT_FIELDS = ("n_queries", "pairs", "seeds", "cells", "worst_pair_name", "missing_cells")
failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print("FAIL", message)


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def number(v: str):
    return None if v in ("", None) else float(v)


def check_sums() -> None:
    expected = dict(reversed(line.split("  ", 1)) for line in (RESULTS / "SHA256SUMS").read_text().splitlines() if line)
    present = {str(p.relative_to(RESULTS)) for p in RESULTS.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "README.md")}
    for name in sorted(present - expected.keys()):
        fail(f"{name} isn't in SHA256SUMS")
    for name, digest in sorted(expected.items()):
        path = RESULTS / name
        if not path.exists():
            fail(f"{name} is in SHA256SUMS and missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            fail(f"{name} doesn't match its checksum")
    print(f"checksums: {len(expected)} files")


def load_archive(name: str, row: str, from_cells: dict | None = None) -> list[dict]:
    """Cells in the shape retention_ci reads: from the per-query archive, or as one-element arrays of the exact cell means."""
    cells = defaultdict(dict)
    if from_cells is not None:
        for (cell, r, system), mean in from_cells.items():
            if r == row:
                cells[cell][system] = np.array([mean])
    else:
        with np.load(RESULTS / "per_query" / f"{name}.npz") as z:
            for key in z.files:
                cell, r, system = key.split("/")
                if r == row:
                    cells[cell][system] = z[key].astype(np.float64) / QUANTUM
    out = []
    for cell, scores in sorted(cells.items()):
        dataset, src, tgt, _, seed = cell.split("__")
        out.append({"dataset": dataset, "src": src, "tgt": tgt, "seed": int(seed[1:]), "scores": scores})
    return out


def check_runs() -> None:
    manifests = {m["run"]: m for m in (json.loads(line) for line in (RESULTS / "manifests.jsonl").open() if line.strip())}
    datasets = {r["run"]: r for r in read_csv("datasets.csv")}
    cell_means = defaultdict(dict)
    for r in read_csv("cells.csv"):
        cell_means[r["source"]][r["cell"], r["row"], r["system"]] = float(r["ndcg5_mean"])
    rows = defaultdict(dict)
    for r in read_csv("retention.csv"):
        rows[r["source"], r["row"]][r["family"]] = r
    checked = 0
    for (source, row), by_family in sorted(rows.items()):
        if source == "fresh_msmarco_hnsw":
            continue
        manifest, counts = manifests.get(source), datasets.get(source)
        if manifest is None or counts is None:
            fail(f"{source}: no manifest or no dataset counts")
            continue
        planned = {tuple(c.rsplit("|", 2)[:2]) + (int(c.rsplit("|", 1)[1]),) for c in manifest["expected_cells"]}
        cells = load_archive(source, row)
        found = {(c["src"], c["tgt"], c["seed"]) for c in cells}
        if found != planned:
            fail(f"{source} row {row}: the archive holds {sorted(found ^ planned)} that the plan doesn't, or misses them")
        lengths = {len(v) for c in cells for v in c["scores"].values()}
        if lengths != {int(counts["queries"])}:
            fail(f"{source}: cells score {sorted(lengths)} queries, and the corpus has {counts['queries']}")
        if counts["queries_sha256"] != manifest["config"]["queries_sha256"]:
            fail(f"{source}: the query digest of datasets.csv isn't the manifest's")
        n_docs = int(manifest["config"]["n_docs"])
        exact = {r["family"]: r for r in summarize(load_archive(source, row, cell_means[source]), n_docs, boot=10, expected=planned)}
        approx = {r["family"]: r for r in summarize(cells, n_docs, expected=planned)}
        for fam, published in by_family.items():
            check_row(f"{source} {row} {fam}", published, exact[fam], approx[fam])
            checked += 1
    print(f"retention rows: {checked}, every field checked against cells.csv and the per-query archives")


def check_row(where: str, published: dict, exact: dict, approx: dict) -> None:
    for key in POINT_FIELDS:
        mine, theirs = exact.get(key), number(published.get(key))
        if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > POINT_TOL):
            fail(f"{where}: {key} {published.get(key)} published, {mine} from cells.csv")
    for key in INTERVAL_FIELDS:
        mine, theirs = approx.get(key), number(published.get(key))
        if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > INTERVAL_TOL):
            fail(f"{where}: {key} {published.get(key)} published, {mine} from per-query scores")
    for key in EXACT_FIELDS:
        mine = ";".join(approx[key]) if key == "missing_cells" else approx.get(key)
        if str(mine) != str(published.get(key)):
            fail(f"{where}: {key} {published.get(key)!r} published, {mine!r} recomputed")
    if published.get("line"):
        line = float(published["line"])
        if abs(line - approx["line"]) > 0:
            fail(f"{where}: line {line} published, {approx['line']} by the rule")
        expected = verdict(float(published["retention"]), float(published["retention_low"]), line, complete=not published["missing_cells"])
        if published["verdict"] != expected:
            fail(f"{where}: verdict {published['verdict']} published, but its numbers give {expected}")


def check_hnsw() -> None:
    records = {r["key"]: r for r in (json.loads(line) for line in (RESULTS / "fresh_msmarco_hnsw.jsonl").open() if line.strip())}
    published = {r["family"]: r for r in read_csv("retention.csv") if r["source"] == "fresh_msmarco_hnsw"}
    with np.load(RESULTS / "per_query" / "fresh_msmarco_hnsw.npz") as z:
        pq = {k: z[k].astype(np.float64) / QUANTUM for k in z.files}
    nat = pq["hnsw|native|ef500|pool60/exact"]
    n = len(nat)
    # Its configuration file predates query digests, so the query list is checked another way: its exact native and re-index rankings,
    # reranked, must score every query exactly as the fresh MS MARCO run's do on the same index model.
    with np.load(RESULTS / "per_query" / "fresh_msmarco.npz") as z:
        native_cells = [z[k] for k in z.files if k.endswith("/9/native") and k.split("__")[1] == "bge-large-en"]
        oracle_cells = {t: [z[k] for k in z.files if k.endswith("/9/oracle") and k.split("__")[1] == "bge-large-en" and k.split("__")[2] == t] for t in ("bge-base-en", "qwen3-0.6b")}
    aligned = native_cells and all(np.array_equal(np.round(c * QUANTUM) if c.dtype.kind == "f" else c, np.round(nat * QUANTUM)) for c in native_cells)
    aligned = aligned and all(all(np.array_equal(c, np.round(pq[f"oracle|{t}|pool60/exact"] * QUANTUM)) for c in cells) and cells for t, cells in oracle_cells.items())
    if not aligned:
        fail("HNSW: its per-query scores don't line up with the fresh MS MARCO run's, so the two may not share one query list")
    w = np.random.default_rng(0).multinomial(n, np.full(n, 1.0 / n), size=2_000) / n
    for tgt, fam in (("bge-base-en", "same"), ("qwen3-0.6b", "across")):
        point = np.mean([records[f"lexical|{tgt}|s{s}|ef500|pool60"]["ndcg5_approx"] for s in (0, 1, 2)]) / min(
            records["hnsw|native|ef500|pool60"]["ndcg5_exact"], records[f"oracle|{tgt}|pool60"]["ndcg5_exact"])
        seeds = [pq[f"lexical|{tgt}|s{s}|ef500|pool60/approx"] for s in (0, 1, 2)]
        reps = np.mean([w @ x for x in seeds], axis=0) / np.minimum(w @ nat, w @ pq[f"oracle|{tgt}|pool60/exact"])
        low, high = np.percentile(reps, [2.5, 97.5])
        r = published[fam]
        if abs(point - float(r["retention"])) > POINT_TOL:
            fail(f"HNSW {fam}: retention {r['retention']} published, {point} from its records")
        if abs(low - float(r["retention_low"])) > INTERVAL_TOL or abs(high - float(r["retention_high"])) > INTERVAL_TOL:
            fail(f"HNSW {fam}: interval [{r['retention_low']}, {r['retention_high']}] published, [{low:.5f}, {high:.5f}] from per-query scores")
        expected = verdict(float(r["retention"]), float(r["retention_low"]), float(r["line"]))
        if r["verdict"] != expected:
            fail(f"HNSW {fam}: verdict {r['verdict']} published, but its numbers give {expected}")
    print("HNSW rows: query list, points, intervals and verdicts checked against their records and per-query scores")


def check_documents() -> None:
    done = subprocess.run([sys.executable, str(Path(__file__).parent / "review_tables.py"), "--check"], capture_output=True, text=True)
    print(done.stdout.strip())
    if done.returncode:
        fail("a document's tables don't match results/")


def main() -> None:
    check_sums()
    check_runs()
    check_hnsw()
    check_documents()
    if failures:
        print(f"\n{len(failures)} failures")
        sys.exit(1)
    print("\nresults/ verified")


if __name__ == "__main__":
    main()
