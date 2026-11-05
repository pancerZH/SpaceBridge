"""Every result table in the README, the report and the design doc, from the records, between <!-- table:NAME --> and <!-- /table:NAME --> markers. Also writes results/.

    python scripts/review_tables.py        rewrites results/ and the tables in place
    python scripts/review_tables.py --check  says which documents would change
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retention_ci import load_cells, summarize  # noqa: E402

REVIEW = Path("outputs/review")
PHASE1 = Path("outputs/phase1")
RESULTS = Path("results")
DOCS = ("README.md", "spacebridge_final_report.md", "spacebridge_design_and_benchmark.md")
CORPUS = {"msmarco": "MS MARCO", "nq": "NQ", "hotpotqa": "HotpotQA", "scifact": "SciFact", "arguana": "ArguAna", "scidocs": "SciDocs", "fiqa": "FiQA", "trec-covid": "TREC-COVID",
          "lotte-technology": "LoTTE technology", "fever": "FEVER", "touche2020": "Touché 2020"}
ORDER = ["MS MARCO", "NQ", "HotpotQA", "SciFact", "ArguAna", "SciDocs", "FiQA", "TREC-COVID"]


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


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.open():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def summarize_dirs(prefix: str, row: str = "9h") -> dict:
    """{source name: {family: row}} for every run directory with this prefix."""
    out = {}
    for d in sorted((REVIEW / "per_query").glob(f"{prefix}*")):
        cells = load_cells(d, row) if any(d.glob("*.npz")) else []
        cells = [c for c in cells if c["scores"]]
        if not cells:
            continue
        manifest = read_jsonl(REVIEW / f"{d.name}.jsonl.manifest.jsonl")
        n_docs = manifest[0]["config"]["n_docs"]
        rows = summarize(cells, n_docs)
        for r in rows:
            r.update(source=d.name, dataset=cells[0]["dataset"], corpus=corpus_name(cells[0]["dataset"]), row=row)
        out[d.name] = {r["family"]: r for r in rows}
    return out


def sort_key(by_family: dict):
    r = next(iter(by_family.values()))
    return (ORDER.index(r["corpus"]) if r["corpus"] in ORDER else 99, r["n_docs"])


def write_retention_csv(all_rows: list[dict]) -> None:
    keys = ["source", "dataset", "corpus", "row", "family", "n_docs", "n_queries", "pairs", "seeds", "cells", "retention", "retention_low", "retention_high", "line", "verdict", "seed_sd",
            "worst_pair", "worst_pair_name", "strict", "strict_low", "strict_high", "vs_reindex", "vs_reindex_low", "vs_reindex_high", "gap_reindex", "gap_reindex_low", "gap_reindex_high",
            "no_bridge_retention", "no_bridge_low", "no_bridge_high", "bridge_gain", "bridge_gain_low", "bridge_gain_high", "abs_bridge", "abs_native", "abs_oracle", "abs_native_bm25", "abs_oracle_bm25", "abs_no_bridge"]
    with (RESULTS / "retention.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})


def export_per_query() -> None:
    """Per-query nDCG@5 of each published cell, in units of 1/10,000. No text and no embeddings: only scores, in the order of each run's query list."""
    out = RESULTS / "per_query"
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted((REVIEW / "per_query").iterdir()):
        arrays = {}
        for path in sorted(d.glob("*.npz")):
            with np.load(path) as z:
                for name in z.files:
                    row, system, metric = name.split("/")
                    if metric == "ndcg@5" and row in ("9", "9h"):
                        arrays[f"{path.stem}/{row}/{system}"] = np.round(z[name] * 10_000).astype(np.uint16)
        if arrays:
            np.savez_compressed(out / f"{d.name}.npz", **arrays)


def export_manifests() -> None:
    with (RESULTS / "manifests.jsonl").open("w") as f:
        for path in sorted(REVIEW.glob("*.jsonl.manifest.jsonl")):
            first = read_jsonl(path)
            if first:
                f.write(json.dumps({"run": path.name.split(".jsonl")[0], **first[0]}) + "\n")


TIMED = [  # label, corpus, documents, search, kind, record file stem or None
    ("100k", "MS MARCO", 100_000, "Exact", "Short"), ("long171k", "TREC-COVID", 171_332, "Exact", "Long"), ("long382k", "Touché 2020", 382_545, "Exact", "Long"),
    ("1m_hnsw", "MS MARCO", 1_000_000, "HNSW", "Short"), ("2m", "MS MARCO", 2_000_000, "Exact", "Short"), ("4m", "MS MARCO", 4_000_000, "Exact", "Short"),
    ("5m", "HotpotQA", 5_233_329, "Exact", "Short"), ("8m_exact", "MS MARCO", 8_841_823, "Exact", "Short"), ("8m_hnsw", "MS MARCO", 8_841_823, "HNSW", "Short"),
]


def timing_records() -> dict:
    out = {}
    for label, *_ in TIMED:
        for device in ("gpu", "cpu"):
            recs = read_jsonl(REVIEW / f"timing_{label}_{device}.jsonl")
            if recs:
                out[label, device] = recs[-1]
    return out


def speed_rows(short_only: bool = False) -> list[list]:
    t = timing_records()
    rows = []
    fmt = {"cpu": lambda v: f"{v / 1e3:.1f}", "gpu": lambda v: f"{v / 1e3:.2f}"}
    for label, corpus, n, search, kind in TIMED:
        if short_only and label in ("2m", "4m"):
            continue
        cols = [docs_label(n), corpus, kind, search]
        tokens = None
        for device in ("cpu", "gpu"):
            r = t.get((label, device))
            if r is None:
                cols += ["not run", "not run", ""]
            else:
                total = r["stages_ms"]["total"]
                cols += [fmt[device](total["p50"]), fmt[device](total["p95"]), r["n"]]
                tokens = r.get("tokens_per_pair", tokens)
        if not short_only:
            cols.append(round(tokens) if tokens else "")
            cols.append("Measured")
        rows.append(cols)
    return rows


def estimated_rows(t: dict) -> list[list]:
    """Sizes with no run of their own: another run's total, with its search and BM25 medians replaced by those of a run nearer this size. A model of the tail, not an observation."""
    out = []
    base = {("cpu"): t.get(("long171k", "cpu")), ("gpu"): t.get(("long171k", "gpu"))}
    for label, n, search in (("1m_hnsw", 1_000_000, "HNSW"), ("8m_hnsw", 8_841_823, "HNSW")):
        cols = [docs_label(n), "TREC-COVID's rerank time, this size's search", "Long", search]
        for device in ("cpu", "gpu"):
            long_, here = base[device], t.get((label, device))
            if long_ is None or here is None:
                cols += ["", "", ""]
                continue
            add = here["stages_ms"]["search"]["p50"] + here["stages_ms"]["lexical"]["p50"] - long_["stages_ms"]["search"]["p50"] - long_["stages_ms"]["lexical"]["p50"]
            fm = (lambda v: f"{v / 1e3:.1f}") if device == "cpu" else (lambda v: f"{v / 1e3:.2f}")
            cols += [fm(long_["stages_ms"]["total"]["p50"] + add), fm(long_["stages_ms"]["total"]["p95"] + add), ""]
        cols += ["", "**Estimated**"]
        out.append(cols)
    return out


def write_timing_csv() -> None:
    with (RESULTS / "timing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "device", "dataset", "documents", "backend", "pool", "queries", "stage", "p50_ms", "p95_ms", "tokens_per_pair", "peak_rss_gb", "peak_vram_gb", "load_s"])
        for path in sorted(REVIEW.glob("timing_*.jsonl")):
            for r in read_jsonl(path):
                for stage, v in r["stages_ms"].items():
                    w.writerow([path.stem, r["device"] if r.get("search_device") in (None, r["device"]) else f"{r['device']}, search on {r['search_device']}", r["dataset"], r["N"], "HNSW" if r["backend"] != "exact" else "exact", r["pool"], r["n"], stage, round(v["p50"], 2), round(v["p95"], 2),
                                round(r.get("tokens_per_pair", 0), 1), round(r.get("peak_rss_gb", 0), 2), round(r.get("peak_vram_gb", 0) or 0, 2), round(r.get("load_s", 0), 1)])


def breakeven_rows() -> list[list]:
    """(re-index setup - bridge setup) / (bridge query cost - re-indexed query cost): a scenario estimate on this machine, one job at a time."""
    t = timing_records()
    rate, mount = {}, {}
    for r in read_jsonl(PHASE1 / "m1_timing/mount.jsonl"):
        if r.get("mode") == "rate":
            rate[r["model"], r["device"]] = r
        elif r.get("K") == 8192:
            mount[r["model"], r["device"]] = r["mount_s"]
    hnsw_build_s = {1_000_000: None, 8_841_823: 1.8 * 3600}
    for r in read_jsonl(PHASE1 / "w5_ann/ms1m.jsonl"):
        if r.get("key") == "build":
            hnsw_build_s[1_000_000] = r.get("build_s_this_run")
    lexical_s = {1_000_000: 22.0, 8_841_823: 128.0}
    runs = {1_000_000: "1m_hnsw", 8_841_823: "8m_hnsw"}
    out = []
    for m, name in (("minilm", "all-MiniLM-L6-v2"), ("bge-base-en", "bge-base-en-v1.5"), ("bge-large-en", "bge-large-en-v1.5"), ("qwen3-0.6b", "Qwen3-Embedding-0.6B")):
        for size, run in runs.items():
            for device, key in (("cuda", "gpu"), ("cpu", "cpu")):
                rec = t.get((run, key))
                if rec is None or (m, device) not in rate:
                    continue
                encode = size * rate[m, device]["tokens_per_doc"] / rate[m, device]["tokens_per_s"]
                reindex = encode + (hnsw_build_s[size] or 0)
                bridge = (mount.get((m, device)) or 0) + lexical_s[size]
                s = rec["stages_ms"]
                both = s["project"]["p50"] / 1e3  # both have BM25 and a reranker: only the projection differs
                none = (s["total"]["p50"] - s["encode"]["p50"] - s["search"]["p50"]) / 1e3  # the other system does neither
                gain = reindex - bridge
                short = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e4 else f"{n / 1e3:.1f}K" if n >= 1e3 else f"{n:.0f}"
                hours = lambda x: f"{x / 3600:.1f} h" if x >= 3600 else f"{x / 60:.0f} min"
                out.append([f"`{name}`", docs_label(size), key.upper(), hours(reindex), hours(bridge) if bridge >= 60 else f"{bridge:.0f} s", f"{1e3 * both:.1f} ms", short(gain / both), f"{none:.2f} s", short(gain / none)])
    return out


def stage1_rows() -> list[list]:
    from spacebridge_bench.metrics import spearman

    rows = []
    for name, label in (("fiqa", "FiQA"), ("scidocs", "SciDocs"), ("nq", "NQ")):
        fits = defaultdict(dict)
        for r in read_jsonl(REVIEW / f"stage1_{name}.jsonl"):
            fits[r["src"], r["tgt"], r["K"], r["seed"]][r["op"], r["lambda_rel"]] = r
        full = [list(v.values()) for v in fits.values() if len(v) == 6]
        if not full:
            continue
        corr = [spearman([c["stage1"]["signals"]["agree@10"] for c in cands], [c["stage1"]["rho_ceil"]["recall@30"] for c in cands]) for cands in full]
        pick = [max(c, key=lambda x: x["stage1"]["signals"]["agree@10"])["auto"]["rho_ceil"]["recall@30"] for c in full]
        best = [max(x["auto"]["rho_ceil"]["recall@30"] for x in c) for c in full]
        corr = [c for c in corr if c == c]
        rows.append([label, len(full), f"{np.mean(corr):.2f}", f"{min(corr):.2f}", f"{max(corr):.2f}", f"{np.mean(pick):.3f}", f"{np.mean(best):.3f}", f"{100 * (np.mean(best) - np.mean(pick)):.2f}"])
    return rows


def fresh_rows(fresh: dict, hnsw: dict) -> list[list]:
    rows = []
    order = ["fresh_msmarco", "fresh_hotpotqa", "fresh_lotte", "fresh_fever", "fresh_touche"]
    names = {"fresh_msmarco": "MS MARCO, unused queries", "fresh_hotpotqa": "HotpotQA, unused queries", "fresh_lotte": "LoTTE technology", "fresh_fever": "FEVER", "fresh_touche": "Touché 2020"}
    for name in order:
        by = fresh.get(name)
        if by is None:
            rows.append([names[name], "", "", "not run", "", "", "", "", "Not run", "", ""])
            continue
        for fam, label in (("same", "Same family"), ("across", "Across families")):
            r = by[fam]
            rows.append([names[name], docs_label(r["n_docs"]), r["n_queries"], label, f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"]), f"{r['line']:.2f}", r["verdict"].capitalize(), f3(r.get("no_bridge_retention")), f3(r.get("abs_bridge"))])
    rows += hnsw
    return rows


def hnsw_fresh() -> list[list]:
    res = {r["key"]: r for r in read_jsonl(REVIEW / "fresh_msmarco_hnsw.jsonl")}
    rows = []
    for tgt, label, line in (("bge-base-en", "Same family, one direction", 0.95), ("qwen3-0.6b", "Across families, one direction", 0.90)):
        lex = [res.get(f"lexical|{tgt}|s{s}|ef500|pool60") for s in (0, 1, 2)]
        native, oracle = res.get("hnsw|native|ef500|pool60"), res.get(f"oracle|{tgt}|pool60")
        if None in lex or native is None or oracle is None:
            rows.append(["MS MARCO, unused queries, HNSW", "8.84M", "", label, "", "", "", f"{line:.2f}", "Not run", "", ""])
            continue
        v = float(np.mean([r["ndcg5_approx"] for r in lex])) / min(native["ndcg5_exact"], oracle["ndcg5_exact"])
        rows.append(["MS MARCO, unused queries, HNSW", "8.84M", 2000, label, f3(v), "", "", f"{line:.2f}", "Point pass" if v >= line else "Miss", "", f3(float(np.mean([r["ndcg5_approx"] for r in lex])))])
    return rows


def build_tables() -> dict[str, str]:
    published = summarize_dirs("recompute_")
    plain = summarize_dirs("recompute_", "9")
    fresh = summarize_dirs("fresh_")
    fresh = {k: v for k, v in fresh.items() if not k.endswith("hnsw")}
    all_rows = [r for d in (published, fresh) for by in d.values() for r in by.values()]
    RESULTS.mkdir(exist_ok=True)
    write_retention_csv(all_rows)
    export_per_query()
    export_manifests()
    write_timing_csv()

    # One row per size: the 8.84M MS MARCO has two runs, one per family.
    merged = defaultdict(dict)
    for name, by in published.items():
        r = next(iter(by.values()))
        merged[r["corpus"], r["n_docs"]].update(by)
    ordered = sorted(merged.values(), key=sort_key)
    t = {}

    def ret_cols(by):
        cols = []
        for fam in ("same", "across"):
            r = by.get(fam)
            cols += [f3(r["retention"]), f3(r["retention_low"]), f3(r["retention_high"])] if r else ["", "", ""]
        return cols

    head = ["Corpus", "Documents", "Queries", "Same family", "Low", "High", "Across families", "Low", "High"]
    t["retention_short"] = table(head, [[r["corpus"], docs_label(r["n_docs"]), r["n_queries"], *ret_cols(by)] for by in ordered for r in [next(iter(by.values()))] if r["n_docs"] in (100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 5_233_329, 8_841_823)])
    t["retention"] = table(
        [*head, "Same family, strict", "Across families, strict", "Same family, absolute", "Across families, absolute", "Native, absolute", "Re-index, absolute"],
        [[r["corpus"], docs_label(r["n_docs"]), r["n_queries"], *ret_cols(by), f3(by.get("same", {}).get("strict")), f3(by.get("across", {}).get("strict")), f3(by.get("same", {}).get("abs_bridge")), f3(by.get("across", {}).get("abs_bridge")), f3(r.get("abs_native")), f3(r.get("abs_oracle"))]
         for by in ordered for r in [by.get("across", next(iter(by.values())))]],
    )

    def reindex(sizes):
        rows = []
        for by in ordered:
            for fam, label in (("same", "Same family"), ("across", "Across families")):
                r = by.get(fam)
                if r and r["n_docs"] in sizes and "gap_reindex" in r:
                    rows.append([r["corpus"], docs_label(r["n_docs"]), label, f3(r["abs_bridge"]), f3(r["abs_oracle_bm25"]), pts(r["gap_reindex"]), pts(r["gap_reindex_low"]), pts(r["gap_reindex_high"]), f3(r["vs_reindex"]), f3(r["vs_reindex_low"]), f3(r["vs_reindex_high"])])
        return rows

    rh = ["Corpus", "Documents", "Models", "Bridge", "Re-index with BM25", "Difference, absolute points", "Low", "High", "Direct retention", "Low", "High"]
    t["reindex"] = table(rh, reindex({100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 5_233_329, 8_841_823}))
    t["reindex_short"] = table(rh, reindex({1_000_000, 5_233_329, 8_841_823}))

    def nobridge(sizes):
        rows = []
        for by in ordered:
            for fam, label in (("same", "Same family"), ("across", "Across families")):
                r = by.get(fam)
                if r and r["n_docs"] in sizes and "no_bridge_retention" in r:
                    rows.append([r["corpus"], docs_label(r["n_docs"]), label, f3(r["no_bridge_retention"]), f3(r["no_bridge_low"]), f3(r["no_bridge_high"]), f3(r["retention"]), pts(r["bridge_gain"]), pts(r["bridge_gain_low"]), pts(r["bridge_gain_high"])])
        return rows

    nh = ["Corpus", "Documents", "Models", "No bridge, retention", "Low", "High", "Bridge, retention", "Bridge adds, absolute points", "Low", "High"]
    t["nobridge"] = table(nh, nobridge({1_000_000, 2_000_000, 4_000_000, 5_233_329, 8_841_823}))
    t["nobridge_short"] = table(nh, nobridge({1_000_000, 5_233_329, 8_841_823}))

    rows = []
    for name, by in sorted(published.items(), key=lambda kv: sort_key(kv[1])):
        r0 = next(iter(by.values()))
        if r0["corpus"] not in ("SciFact", "ArguAna", "SciDocs", "FiQA", "TREC-COVID"):
            continue
        for fam, label in (("same", "Same family"), ("across", "Across families")):
            on, off = by.get(fam), plain.get(name, {}).get(fam)
            if not on or not off:
                continue
            cells_on = {(c["src"], c["tgt"], c["seed"]): float(c["scores"]["bridge"].mean()) for c in load_cells(REVIEW / "per_query" / name, "9h") if c["src"] is not None}
            cells_off = {(c["src"], c["tgt"], c["seed"]): float(c["scores"]["bridge"].mean()) for c in load_cells(REVIEW / "per_query" / name, "9")}
            same_fam = [k for k in cells_on if (k[0], k[1]) in (("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en"))]
            keys = same_fam if fam == "same" else [k for k in cells_on if k not in same_fam]
            rows.append([on["corpus"], f"{on['n_docs']:,}", on["n_queries"], label, f3(off["retention"]), f3(on["retention"]), f3(off["abs_bridge"]), f3(on["abs_bridge"]), sum(cells_on[k] > cells_off[k] for k in keys), len(keys)])
    t["bm25"] = table(["Corpus", "Documents", "Queries", "Models", "Retention, BM25 off", "Retention, BM25 on", "Absolute, BM25 off", "Absolute, BM25 on", "Cells that gain", "Cells"], rows)

    tm = timing_records()
    sh = ["Documents", "Corpus", "Documents' length", "Search", "CPU p50, s", "CPU p95, s", "CPU queries", "GPU p50, s", "GPU p95, s", "GPU queries"]
    t["speed"] = table([*sh, "Tokens per pair", "Status"], speed_rows() + estimated_rows(tm))
    t["speed_short"] = table(sh, speed_rows(short_only=True))
    t["breakeven"] = table(["Query model", "Documents", "Device", "Re-index setup", "Bridge setup", "Per query, if it also reranks", "Break-even queries", "Per query, if it doesn't", "Break-even queries"], breakeven_rows())
    t["stage1"] = table(["Corpus", "Fits", "Mean Spearman, agreement against real retention", "Lowest", "Highest", "Agreement's pick, after Stage 2", "Best candidate, after Stage 2", "Retention points a perfect selector would gain"], stage1_rows())
    t["fresh"] = table(["Data", "Documents", "Queries", "Models", "Retention", "Low", "High", "Line", "Verdict", "No bridge, retention", "Bridge, absolute `nDCG@5`"], fresh_rows(fresh, hnsw_fresh()))
    return t


def fill(text: str, tables: dict[str, str]) -> str:
    def replace(m):
        name = m.group(1)
        return f"<!-- table:{name} -->\n{tables[name]}\n<!-- /table:{name} -->" if name in tables else m.group(0)

    return re.sub(r"<!-- table:(\w+) -->(?:.*?<!-- /table:\1 -->)?", replace, text, flags=re.S)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    tables = build_tables()
    for name in DOCS:
        path = Path(name)
        new = fill(path.read_text(), tables)
        if new != path.read_text():
            print(("would change " if a.check else "updated ") + name)
            if not a.check:
                path.write_text(new)


if __name__ == "__main__":
    main()
