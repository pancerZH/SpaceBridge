"""Every result table in the README, the report and the design doc, between <!-- table:NAME --> and <!-- /table:NAME --> markers, from
results/ alone, so a public checkout can regenerate them. `scripts/review_export.py` writes results/ from the local records.

    python scripts/review_tables.py          fills the tables in place
    python scripts/review_tables.py --check  exits 1, naming the documents, if any table differs from results/
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results")
DOCS = ("README.md", "spacebridge_final_report.md", "spacebridge_design_and_benchmark.md")
CORPUS = {"msmarco": "MS MARCO", "nq": "NQ", "hotpotqa": "HotpotQA", "scifact": "SciFact", "arguana": "ArguAna", "scidocs": "SciDocs", "fiqa": "FiQA", "trec-covid": "TREC-COVID",
          "lotte-technology": "LoTTE technology", "fever": "FEVER", "touche2020": "Touché 2020"}
ORDER = ["MS MARCO", "NQ", "HotpotQA", "SciFact", "ArguAna", "SciDocs", "FiQA", "TREC-COVID"]
FAMILIES = (("same", "Same family"), ("across", "Across families"))
SAME_FAMILY = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")}
TIMED = [  # label, corpus, documents, search, documents' length
    ("100k", "MS MARCO", 100_000, "Exact", "Short"), ("long171k", "TREC-COVID", 171_332, "Exact", "Long"), ("long382k", "Touché 2020", 382_545, "Exact", "Long"),
    ("1m_hnsw", "MS MARCO", 1_000_000, "HNSW", "Short"), ("2m", "MS MARCO", 2_000_000, "Exact", "Short"), ("4m", "MS MARCO", 4_000_000, "Exact", "Short"),
    ("5m", "HotpotQA", 5_233_329, "Exact", "Medium"), ("8m_exact", "MS MARCO", 8_841_823, "Exact", "Short"), ("8m_hnsw", "MS MARCO", 8_841_823, "HNSW", "Short"),
]


def corpus_name(dataset: str) -> str:
    base = dataset.split("@")[0]
    for key, name in sorted(CORPUS.items(), key=lambda kv: -len(kv[0])):
        if base.startswith(key):
            return name
    return base


def docs_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M".replace(".00M", "M").replace("0M", "M") if n % 1_000_000 else f"{n // 1_000_000}M"
    if n == 160_001:
        return "160,001"
    return f"{n / 1e3:.1f}K".replace(".0K", "K")


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def f3(v) -> str:
    return "" if v is None else f"{v:.3f}"


def pts(v) -> str:
    return "" if v is None else f"{100 * v:+.1f}"


# Names, not numbers: row "9" is the BM25-off row, never the integer 9.
TEXT_COLUMNS = {"source", "dataset", "row", "family", "verdict", "missing_cells", "worst_pair_name", "cell", "system", "run", "kind", "model", "device", "stage", "backend", "source_note"}


def read_csv(name: str) -> list[dict]:
    def typed(v: str):
        if v == "":
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    with (RESULTS / name).open(newline="") as f:
        return [{k: (v if k in TEXT_COLUMNS else typed(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / name).open() if line.strip()]


class Results:
    """results/, indexed the ways the tables read it."""

    def __init__(self):
        self.retention = defaultdict(lambda: defaultdict(dict))  # source -> row -> family -> record
        for r in read_csv("retention.csv"):
            r["corpus"] = corpus_name(r["dataset"])
            self.retention[r["source"]][r["row"]][r["family"]] = r
        self.cells = defaultdict(lambda: defaultdict(dict))  # source -> cell -> (row, system) -> mean
        for r in read_csv("cells.csv"):
            self.cells[r["source"]][r["cell"]][r["row"], r["system"]] = r["ndcg5_mean"]
        self.timing = defaultdict(lambda: {"stages": {}})  # (label, device) -> record
        for r in read_csv("timing.csv"):
            label, device = r["run"].removeprefix("timing_").rsplit("_", 1)
            t = self.timing[label, device]
            t.update(n=r["queries"], tokens_per_pair=r["tokens_per_pair"])
            t["stages"][r["stage"]] = {"p50": r["p50_ms"], "p95": r["p95_ms"]}
        self.setup = read_csv("setup.csv")

    def published(self, row: str = "9h") -> dict:
        """One entry per corpus and size: the 8.84M MS MARCO has two runs, one per family."""
        merged = defaultdict(dict)
        for source, rows in self.retention.items():
            if source.startswith("recompute_") and row in rows:
                by = rows[row]
                r = next(iter(by.values()))
                merged[r["corpus"], r["n_docs"]].update(by)
        return dict(sorted(merged.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 99, kv[0][1])))


def retention_tables(res: Results, t: dict) -> None:
    ordered = list(res.published().values())

    def ret_cols(by):
        cols = []
        for fam, _ in FAMILIES:
            r = by.get(fam)
            cols += [f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"])] if r else ["", "", ""]
        return cols

    head = ["Corpus", "Documents", "Queries", "Same family", "Low", "High", "Across families", "Low", "High"]
    sizes = (100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 5_233_329, 8_841_823)
    t["retention_short"] = table(head, [[r["corpus"], docs_label(r["n_docs"]), r["n_queries"], *ret_cols(by)] for by in ordered for r in [next(iter(by.values()))] if r["n_docs"] in sizes])
    # One row per family, so each row's native and re-index are that family's comparators.
    rows = []
    for by in ordered:
        for fam, label in FAMILIES:
            r = by.get(fam)
            if r:
                rows.append([r["corpus"], docs_label(r["n_docs"]), r["n_queries"], label, f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"]), f3(r["strict"]),
                             f3(r["abs_bridge"]), f3(r["abs_native"]), f3(r["abs_oracle"])])
    t["retention"] = table(["Corpus", "Documents", "Queries", "Models", "Retention", "Low", "High", "Strict retention", "Bridge, absolute", "Native, absolute", "Re-index, absolute"], rows)

    def by_family(sizes, cols, need):
        out = []
        for by in ordered:
            for fam, label in FAMILIES:
                r = by.get(fam)
                if r and r["n_docs"] in sizes and r.get(need) is not None:
                    out.append([r["corpus"], docs_label(r["n_docs"]), label, *cols(r)])
        return out

    reindex = lambda r: [f3(r["abs_bridge"]), f3(r["abs_oracle_bm25"]), pts(r["gap_reindex"]), pts(r["gap_reindex_low"]), pts(r["gap_reindex_high"]), f3(r["vs_reindex"]), f3(r["vs_reindex_low"]), f3(r["vs_reindex_high"])]
    rh = ["Corpus", "Documents", "Models", "Bridge", "Re-index with BM25", "Difference, absolute points", "Low", "High", "Direct retention", "Low", "High"]
    t["reindex"] = table(rh, by_family(set(sizes), reindex, "gap_reindex"))
    t["reindex_short"] = table(rh, by_family({1_000_000, 5_233_329, 8_841_823}, reindex, "gap_reindex"))
    nobridge = lambda r: [f3(r["no_bridge_retention"]), f3(r["no_bridge_low"]), f3(r["no_bridge_high"]), f3(r["retention"]), pts(r["bridge_gain"]), pts(r["bridge_gain_low"]), pts(r["bridge_gain_high"])]
    nh = ["Corpus", "Documents", "Models", "No bridge, retention", "Low", "High", "Bridge, retention", "Bridge adds, absolute points", "Low", "High"]
    t["nobridge"] = table(nh, by_family({1_000_000, 2_000_000, 4_000_000, 5_233_329, 8_841_823}, nobridge, "no_bridge_retention"))
    t["nobridge_short"] = table(nh, by_family({1_000_000, 5_233_329, 8_841_823}, nobridge, "no_bridge_retention"))


def bm25_table(res: Results, t: dict) -> None:
    rows = []
    for source, by_row in sorted(res.retention.items(), key=lambda kv: (kv[0])):
        if not source.startswith("recompute_") or "9" not in by_row:
            continue
        on_by, off_by = by_row["9h"], by_row["9"]
        r0 = next(iter(on_by.values()))
        if r0["corpus"] not in ("SciFact", "ArguAna", "SciDocs", "FiQA", "TREC-COVID"):
            continue
        cells = res.cells[source]
        for fam, label in FAMILIES:
            on, off = on_by.get(fam), off_by.get(fam)
            if not on or not off:
                continue
            keys = [c for c in cells if (tuple(c.split("__")[1:3]) in SAME_FAMILY) == (fam == "same")]
            gain = sum(cells[c][("9h", "bridge")] > cells[c][("9", "bridge")] for c in keys)
            rows.append((ORDER.index(on["corpus"]), [on["corpus"], f"{on['n_docs']:,}", on["n_queries"], label, f3(off["retention"]), f3(on["retention"]), f3(off["abs_bridge"]), f3(on["abs_bridge"]), gain, len(keys)]))
    rows.sort(key=lambda x: x[0])
    t["bm25"] = table(["Corpus", "Documents", "Queries", "Models", "Retention, BM25 off", "Retention, BM25 on", "Absolute, BM25 off", "Absolute, BM25 on", "Cells that gain", "Cells"], [r for _, r in rows])


def speed_tables(res: Results, t: dict) -> None:
    fmt = {"cpu": lambda v: f"{v / 1e3:.1f}", "gpu": lambda v: f"{v / 1e3:.2f}"}

    def measured(short_only: bool) -> list[list]:
        rows = []
        for label, corpus, n, search, kind in TIMED:
            if short_only and label in ("2m", "4m"):
                continue
            cols, tokens = [docs_label(n), corpus, kind, search], None
            for device in ("cpu", "gpu"):
                r = res.timing.get((label, device))
                if r is None:
                    cols += ["not run", "not run", ""]
                else:
                    total = r["stages"]["total"]
                    cols += [fmt[device](total["p50"]), fmt[device](total["p95"]), r["n"]]
                    tokens = r["tokens_per_pair"] or tokens
            if not short_only:
                cols += [round(tokens) if tokens else "", "Measured"]
            rows.append(cols)
        return rows

    def estimated() -> list[list]:
        """Sizes with no run of their own: another run's total, with its search and BM25 medians replaced by those of a run nearer this size. A model of the tail, not an observation."""
        out = []
        for label, n, search in (("1m_hnsw", 1_000_000, "HNSW"), ("8m_hnsw", 8_841_823, "HNSW")):
            cols = [docs_label(n), "TREC-COVID's rerank time, this size's search", "Long", search]
            for device in ("cpu", "gpu"):
                long_, here = res.timing.get(("long171k", device)), res.timing.get((label, device))
                if long_ is None or here is None:
                    cols += ["", "", ""]
                    continue
                s, b = here["stages"], long_["stages"]
                add = s["search"]["p50"] + s["lexical"]["p50"] - b["search"]["p50"] - b["lexical"]["p50"]
                cols += [fmt[device](b["total"]["p50"] + add), fmt[device](b["total"]["p95"] + add), ""]
            out.append(cols + ["", "**Estimated**"])
        return out

    sh = ["Documents", "Corpus", "Documents' length", "Search", "CPU p50, s", "CPU p95, s", "CPU queries", "GPU p50, s", "GPU p95, s", "GPU queries"]
    t["speed"] = table([*sh, "Tokens per pair", "Status"], measured(False) + estimated())
    t["speed_short"] = table(sh, measured(True))


def breakeven_table(res: Results, t: dict) -> None:
    """(re-index setup - bridge setup) / (bridge query cost - re-indexed query cost): a scenario estimate on this machine, one job at a time."""
    rate = {(r["model"], r["device"]): r for r in res.setup if r["kind"] == "encode_rate"}
    mount = {(r["model"], r["device"]): r["seconds"] for r in res.setup if r["kind"] == "mount" and r["K"] == 8192}
    hnsw_build = {r["documents"]: r["seconds"] for r in res.setup if r["kind"] == "hnsw_build"}
    bm25_build = {r["documents"]: r["seconds"] for r in res.setup if r["kind"] == "bm25_build"}
    out = []
    short = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e4 else f"{n / 1e3:.1f}K" if n >= 1e3 else f"{n:.0f}"
    hours = lambda x: f"{x / 3600:.1f} h" if x >= 3600 else f"{x / 60:.0f} min"
    for m, name in (("minilm", "all-MiniLM-L6-v2"), ("bge-base-en", "bge-base-en-v1.5"), ("bge-large-en", "bge-large-en-v1.5"), ("qwen3-0.6b", "Qwen3-Embedding-0.6B")):
        for size, run in ((1_000_000, "1m_hnsw"), (8_841_823, "8m_hnsw")):
            for device, key in (("cuda", "gpu"), ("cpu", "cpu")):
                rec = res.timing.get((run, key))
                if rec is None or (m, device) not in rate:
                    continue
                reindex = size * rate[m, device]["tokens_per_doc"] / rate[m, device]["tokens_per_s"] + (hnsw_build.get(size) or 0)
                timed_mount = mount.get((m, device))
                bridge = (timed_mount or 0) + bm25_build[size]  # an untimed mount counts as 0: minutes, against hours of re-encoding
                s = rec["stages"]
                both = s["project"]["p50"] / 1e3  # both have BM25 and a reranker: only the projection differs
                none = (s["total"]["p50"] - s["encode"]["p50"] - s["search"]["p50"]) / 1e3  # the other system does neither
                gain = reindex - bridge
                out.append([f"`{name}`", docs_label(size), key.upper(), hours(reindex), (hours(bridge) if bridge >= 60 else f"{bridge:.0f} s") + ("" if timed_mount else ", BM25 only: mount not timed here"),
                            f"{1e3 * both:.1f} ms", short(gain / both), f"{none:.2f} s", short(gain / none)])
    t["breakeven"] = table(["Query model", "Documents", "Device", "Re-index setup", "Bridge setup", "Per query, if it also reranks", "Break-even queries", "Per query, if it doesn't", "Break-even queries"], out)


def study_tables(t: dict) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from spacebridge_bench.metrics import spearman

    rows = []
    records = read_jsonl("stage1.jsonl")
    for name, label in (("fiqa", "FiQA"), ("scidocs", "SciDocs"), ("nq", "NQ")):
        fits = defaultdict(dict)
        for r in records:
            if r["dataset"] == name:
                fits[r["src"], r["tgt"], r["K"], r["seed"]][r["op"], r["lambda_rel"]] = r
        full = [list(v.values()) for v in fits.values() if len(v) == 6]
        if not full:
            continue
        corr = [spearman([c["stage1"]["signals"]["agree@10"] for c in cands], [c["stage1"]["rho_ceil"]["recall@30"] for c in cands]) for cands in full]
        pick = [max(c, key=lambda x: x["stage1"]["signals"]["agree@10"])["auto"]["rho_ceil"]["recall@30"] for c in full]
        best = [max(x["auto"]["rho_ceil"]["recall@30"] for x in c) for c in full]
        corr = [c for c in corr if c == c]
        rows.append([label, len(full), f"{np.mean(corr):.2f}", f"{min(corr):.2f}", f"{max(corr):.2f}", f"{np.mean(pick):.3f}", f"{np.mean(best):.3f}", f"{100 * (np.mean(best) - np.mean(pick)):.2f}"])
    t["stage1"] = table(["Corpus", "Fits", "Mean Spearman, agreement against real retention", "Lowest", "Highest", "Agreement's pick, after Stage 2", "Best candidate, after Stage 2", "Retention points a perfect selector would gain"], rows)

    prec = [r for r in read_jsonl("precision_cpu.jsonl") if r["pool"] == 60]
    names = {"msmarco-1m": "MS MARCO 1M", "nq-1m": "NQ 1M", "fiqa": "FiQA", "trec-covid": "TREC-COVID"}
    rows = []
    for ds, label in names.items():
        by = {r["variant"]: r for r in prec if r["dataset"] == ds}
        if {"fp32", "bf16", "cpu"} <= set(by):
            rows.append([label, by["cpu"]["n_queries"], f"{by['cpu']['pool_tokens']:.0f}", f"{by['fp32']['ndcg@5']:.4f}", f"{by['cpu']['ndcg@5']:.4f}", f"{100 * by['cpu']['difference_from_fp32']:+.2f}",
                         f"{100 * by['cpu']['ci_difference'][0]:+.2f}", f"{100 * by['cpu']['ci_difference'][1]:+.2f}", f"{by['cpu']['top5_jaccard_with_fp32']:.3f}", f"{by['bf16']['ndcg@5']:.4f}", f"{100 * by['bf16']['difference_from_fp32']:+.2f}"])
    t["precision"] = table(["Corpus", "Queries", "Tokens per pair", "GPU fp32", "CPU fp32", "CPU minus GPU, absolute points", "Low", "High", "Top-5 overlap, CPU and GPU", "GPU bf16", "bf16 minus GPU fp32, absolute points"], rows)


def fresh_table(res: Results, t: dict) -> None:
    names = {"fresh_msmarco": "MS MARCO, unused queries", "fresh_hotpotqa": "HotpotQA, unused queries", "fresh_lotte": "LoTTE technology", "fresh_fever": "FEVER", "fresh_touche": "Touché 2020"}
    rows = []
    for source, name in names.items():
        by = res.retention.get(source, {}).get("9h")
        if not by:
            rows.append([name, "", "", "not run", "", "", "", "", "Not run", "", ""])
            continue
        for fam, label in FAMILIES:
            r = by[fam]
            rows.append([name, docs_label(r["n_docs"]), r["n_queries"], label, f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"]), f"{r['line']:.2f}", r["verdict"].capitalize(), f3(r["no_bridge_retention"]), f3(r["abs_bridge"])])
    hnsw = res.retention.get("fresh_msmarco_hnsw", {}).get("9h", {})
    for fam, label in FAMILIES:
        r = hnsw.get(fam)
        if r:
            rows.append(["MS MARCO, unused queries, HNSW", docs_label(r["n_docs"]), r["n_queries"], f"{label}, one direction", f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"]), f"{r['line']:.2f}", r["verdict"].capitalize(), "", f3(r["abs_bridge"])])
    t["fresh"] = table(["Data", "Documents", "Queries", "Models", "Retention", "Low", "High", "Line", "Verdict", "No bridge, retention", "Bridge, absolute `nDCG@5`"], rows)


def build_tables() -> dict[str, str]:
    res, t = Results(), {}
    retention_tables(res, t)
    bm25_table(res, t)
    speed_tables(res, t)
    breakeven_table(res, t)
    study_tables(t)
    fresh_table(res, t)
    return t


def fill(text: str, tables: dict[str, str]) -> str:
    def replace(m):
        name = m.group(1)
        return f"<!-- table:{name} -->\n{tables[name]}\n<!-- /table:{name} -->" if name in tables else m.group(0)

    return re.sub(r"<!-- table:(\w+) -->(?:.*?<!-- /table:\1 -->)?", replace, text, flags=re.S)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    tables = build_tables()
    empty = sorted(name for name, text in tables.items() if len(text.splitlines()) < 3)
    if empty:  # a table with a header and no rows means its inputs weren't found: never write or accept it
        print(f"tables with no rows: {', '.join(empty)}")
        sys.exit(1)
    stale = []
    for name in DOCS:
        path = Path(name)
        new = fill(path.read_text(), tables)
        if new != path.read_text():
            stale.append(name)
            if not a.check:
                path.write_text(new)
    for name in stale:
        print(("differs from results/: " if a.check else "updated ") + name)
    if a.check and stale:
        sys.exit(1)
    if a.check:
        print(f"every table in {', '.join(DOCS)} matches results/")


if __name__ == "__main__":
    main()
