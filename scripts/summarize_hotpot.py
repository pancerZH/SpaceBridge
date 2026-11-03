"""Tables of the HotpotQA study, from the records of ``hotpot_study.py``, ``hotpot_fixes.py`` and ``outputs/phase1/hp_scale.sh``.

    uv run python scripts/summarize_hotpot.py > outputs/phase1/m2_tables/hp_study.md
"""
import collections
import json
import statistics as st
from pathlib import Path

D = Path("outputs/phase1")
STUDY, SCALE, FIXES = D / "hp_study", D / "hp_scale", D / "hp_fixes"
T1 = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")}
NAMES = {"hotpotqa-1m": "HotpotQA", "msmarco-1m": "MS MARCO", "nq-1m": "NQ"}
TIER = {"T1": "1", "T3": "3"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def table(head: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")
    print()


def f3(x) -> str:
    return "-" if x is None or x != x else f"{x:.3f}"


def by_key(records: list[dict]) -> dict:
    return {(r["corpus"], r["tier"]): r for r in records if r.get("seed", 0) == 0}


def label(corpus: str) -> str:
    return NAMES.get(corpus, corpus)


def noise_table() -> None:
    print("### The same noise, on tighter scores\n\n`dose.jsonl`, 300 queries, seed 0: among native's top 1,000, per query.\n")
    rows = []
    for (corpus, tier), r in sorted(by_key(read(STUDY / "dose.jsonl")).items()):
        t, at1 = r["top1000"], next(c for c in r["curve"] if c["alpha"] == 1.0)
        rows.append([label(corpus), TIER[tier], f3(t["s@1"]), f3(t["s@10"] - t["s@300"]), f"{t['spread']:.4f}", f"{t['noise']:.4f}", f3(t["noise_over_spread"]), f3(t["spearman"]), f3(at1["cos"]), f3(at1["lost300"])])
    table(["Corpus", "Tier", "Top score", "Score gap, rank 10 to 300", "Spread", "Map's noise", "Noise / spread", "Rank correlation", "Cosine", "Positives lost from top 300"], rows)


def dose_table() -> None:
    print("### The map's error, scaled\n\n`dose.jsonl`, `q + a (q_hat - q)`, seed 0, 2,000 queries: cosine and positives lost from the top 300.\n")
    rows = []
    for (corpus, tier), r in sorted(by_key(read(STUDY / "dose.jsonl")).items()):
        rows.append([label(corpus), TIER[tier]] + [f"{c['cos']:.3f}, {c['lost300']:.3f}" for c in r["curve"]])
    table(["Corpus", "Tier", "a = 0.25", "0.5", "0.75", "1 (the map)", "1.25"], rows)


def margin_table() -> None:
    print("### One curve?\n\n`margin.jsonl`, 800 queries, seed 0: loss by the positive's native margin over rank 300, in units of the map's noise, and each bucket's share of positives.\n")
    recs = by_key(read(STUDY / "margin.jsonl"))
    edges = [c["z"] for c in next(iter(recs.values()))["curve"]] if recs else []
    rows = []
    for (corpus, tier), r in sorted(recs.items()):
        rows.append([label(corpus), TIER[tier], f3(r["z_median"]), f3(r["lost"])] + [("-" if c["lost"] is None else f"{c['lost']:.2f}") + f" ({c['share']:.2f})" for c in r["curve"]])
    table(["Corpus", "Tier", "Median margin", "Lost"] + [f"{a}-{b}" for a, b in edges], rows)
    print("Loss each corpus would have with another's curve at its own margins (Tier, then corpus):\n")
    out = []
    for tier in ("T1", "T3"):
        hp = recs.get(("hotpotqa-1m", tier))
        for other in ("msmarco-1m", "nq-1m"):
            o = recs.get((other, tier))
            if hp and o:
                pred = sum(a["share"] * (b["lost"] or 0) for a, b in zip(hp["curve"], o["curve"]))
                out.append([TIER[tier], label(other), f3(o["lost"]), f3(pred), f3(hp["lost"]), f"{100 * (pred - o['lost']) / max(hp['lost'] - o['lost'], 1e-9):.0f}%"])
    table(["Tier", "Other corpus", "Its loss", "HotpotQA's margins, its curve", "HotpotQA's loss", "Share of the gap from margins"], out)


def signal_table() -> None:
    print("### What survives the map\n\n`signal.jsonl`, 800 queries, seed 0: among native's top 1,000 for a query, bridged scores against native scores, each centred on the query's mean.\n")
    rows = []
    for (corpus, tier), r in sorted(by_key(read(STUDY / "signal.jsonl")).items()):
        rows.append([label(corpus), TIER[tier], f3(r["slope"]), f"{r['resid']:.4f}", f"{r['spread']:.4f}", f3(r["resid_over_signal"]), f3(r["positive_ratio"]), f"{100 * r['share_below_trend']:.0f}%", f3(r["positive_excess_z"])])
    table(["Corpus", "Tier", "Slope", "Noise around it", "Native spread", "Noise / carried signal", "Positive's lead kept (median)", "Positives below the trend", "Median excess, in noise units"], rows)


def cover_table() -> None:
    print("### Calibration and input\n\n`cover.jsonl`, `offset.jsonl`, seed 0.\n")
    off = by_key(read(STUDY / "offset.jsonl"))
    rows = []
    for (corpus, tier), r in sorted(by_key(read(STUDY / "cover.jsonl")).items()):
        o = off.get((corpus, tier), {})
        b1000 = (o.get("by_n") or {}).get("1000", {}).get("both", {})
        rows.append([label(corpus), TIER[tier], f3(r["query_cos"]), f3(r["val_proxy_cos"]), f3(r["energy_queries"]["256"]), f3(r["energy_val_proxies"]["256"]), f"{r['pr_src_anchor_docs']:.0f}",
                     f"{r['pr_src_queries']:.0f}", f3(o.get("shared_share")), f3(o.get("baseline", {}).get("ret@300")), f3(b1000.get("ret@300"))])
    table(["Corpus", "Tier", "Cosine, queries", "Cosine, val proxies", "Energy in top 256 dims, queries", "Same, proxies", "Effective dim, docs", "Effective dim, queries", "Shared offset's share of the error", "`recall@300` retention", "Recentred on 1,000 queries"], rows)


def cross_table() -> None:
    print("### Whose map, whose text\n\n`cross.jsonl`, seed 0: cosine to the native side, each row's map on each column's input.\n")
    for r in read(STUDY / "cross.jsonl"):
        names = list(NAMES)
        rows = []
        for a in names:
            rows.append([f"{NAMES[a]} map"] + [f"{r['table'][f'{a}->{b}']['queries']:.3f}, {r['table'][f'{a}->{b}']['proxies']:.3f}, {r['table'][f'{a}->{b}']['docs']:.3f}" for b in names])
        print(f"Tier {TIER.get(r['tier'], r['tier'])}: queries, proxies, documents.\n")
        table(["Map"] + [NAMES[b] for b in names], rows)


def form_table() -> None:
    print("### Query form\n\n`form.jsonl`, seed 0: cosine under the corpus's own map. MS MARCO and NQ are lower-cased, so entity variants don't apply to them.\n")
    rows = []
    keys = ("original", "concat_two", "first_half", "second_half", "no_qmark_lower", "entities_only", "one_entity")
    for (corpus, tier), r in sorted(by_key(read(STUDY / "form.jsonl")).items()):
        rows.append([label(corpus), TIER[tier], f3(r["val_proxy_cos"])] + [f3(r["variants"][k]["cos"]) for k in keys])
    table(["Corpus", "Tier", "Val proxies", "Original", "Two queries joined", "First half", "Second half", "Lower-cased", "Entities only", "One entity"], rows)


def features_table() -> None:
    print("### Which positives are lost\n\n`features.jsonl`, seed 0, 2,000 queries: share lost from the top 300, of positives native had there.\n")
    recs = by_key(read(STUDY / "features.jsonl"))
    keys = [("all", "All"), ("title_in_query", "Title in the query"), ("title_not_in_query", "Title not in the query"), ("query_has_2_titles", "Query names both titles"), ("query_has_0_titles", "Query names neither"),
            ("title_rarity_low", "Common title words"), ("title_rarity_high", "Rare title words"), ("shared_words_low", "Few words shared with the query"), ("shared_words_high", "Many words shared"),
            ("query_words_low", "Short queries"), ("query_words_high", "Long queries"), ("doc_words_low", "Short documents"), ("doc_words_high", "Long documents"), ("query_cos_low", "Lowest-cosine third of queries"), ("query_cos_high", "Highest-cosine third")]
    cols = sorted(recs)
    rows = []
    for key, name in keys:
        row = [name]
        for c in cols:
            g = recs[c]["groups"].get(key) or {}
            row.append(f"{g['lost300']:.3f}" if g.get("lost300") is not None and g.get("n") else "-")
        rows.append(row)
    rows.append(["Lost that re-index also misses"] + [f3(recs[c]["groups"]["reindex"]["lost_and_reindex_gt300"]) for c in cols])
    rows.append(["Kept that re-index misses"] + [f3(recs[c]["groups"]["reindex"]["kept_and_reindex_gt300"]) for c in cols])
    table(["Positives"] + [f"{label(a)}, Tier {TIER[b]}" for a, b in cols], rows)


def fits_table() -> None:
    print("### Better calibration, and the map class\n\n`fits.jsonl` shrunk and chosen by cosine; `fits2.jsonl` norm-matched and chosen by agreement with native on dev queries. `recall@300` retention (cosine). Seed 0.\n")
    fit1, fit2 = by_key(read(STUDY / "fits.jsonl")), by_key(read(STUDY / "fits2.jsonl"))
    rows = []
    for key, r in sorted(fit2.items()):
        cell = lambda d, k: f"{d[k]['ret@300']:.3f} ({d[k]['cos']:.3f})" if k in d else "-"
        o = fit1.get(key, {})
        rows.append([label(key[0]), TIER[key[1]], cell(r, "baseline"), cell(o, "real_10000"), cell(r, "real_10000"), cell(r, "mix_1000"), cell(r, "mix_10000"), cell(r, "mix_84000"), cell(r, "real_84000"), cell(r, "mlp_84000")])
    table(["Corpus", "Tier", "Shipped", "10,000 real, by cosine", "10,000 real", "Shipped pairs + 1,000 real", "+ 10,000", "+ 84,000", "84,000 real alone", "MLP on 84,000"], rows)


def candidates_table() -> None:
    print("### Cheap first-stage candidates\n\n`scale.jsonl`, `tta.jsonl`, `local.jsonl`, `prf.jsonl`, seed 0: `recall@300` retention. Best of each grid, chosen on these test queries, so optimistic.\n")
    scale, tta, local = (by_key(read(STUDY / f"{n}.jsonl")) for n in ("scale", "tta", "local"))
    prf = by_key(read(FIXES / "prf.jsonl"))
    rows = []
    for key, r in sorted(scale.items()):
        best = lambda d: max((v["ret@300"], k) for k, v in d.items() if isinstance(v, dict) and "cos" in v and k != "baseline")
        s = max((v["ret@300"], k) for k, v in r["grid"].items())
        rows.append([label(key[0]), TIER[key[1]], f3(r["grid"]["s1.0_t1.0"]["ret@300"]), f"{s[0]:.3f} ({s[1]})"] + [f"{best(d[key])[0]:.3f} ({best(d[key])[1]})" for d in (tta, prf, local)])
    table(["Corpus", "Tier", "Shipped", "Scale", "Query variants", "Feedback from top docs", "Local residuals"], rows)


def curve_table() -> None:
    print("### HotpotQA's first stage by size and K\n\n`hp_scale/curve.jsonl` (K = 8,192, 3 seeds), `curve_k.jsonl` (one seed). 4 pairs, 2,000 queries. `recall@300` and `recall@1000` retention.\n")
    recs = read(SCALE / "curve.jsonl") + read(SCALE / "curve_k.jsonl")
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        g[(r["size"], r["K"])][(r["src"], r["tgt"])].append(r)
    rows = []
    for (size, K), pairs in sorted(g.items()):
        mean = lambda m, t: st.mean(st.mean(x["rho_ceil"][m] for x in v) for k, v in pairs.items() if (k in T1) == t)
        worst = min(st.mean(x["rho_ceil"]["recall@300"] for x in v) for v in pairs.values())
        rows.append([f"{size:,}", f"{K:,}", max(len(v) for v in pairs.values()), f3(mean("recall@300", True)), f3(mean("recall@300", False)), f3(worst), f3(mean("recall@1000", True)), f3(mean("recall@1000", False)),
                     f3(st.mean(st.mean(x["native"]["recall@300"] for x in v) for v in pairs.values()))])
    table(["Size", "K", "Seeds", "Tier 1 `R@300`", "Tier 3 `R@300`", "Worst pair", "Tier 1 `R@1000`", "Tier 3 `R@1000`", "Native `R@300`, absolute"], rows)


def e2e_tiers(records: list[dict], row: str) -> tuple:
    d = collections.defaultdict(list)
    for r in records:
        if r["row"] == row and r.get("reranker") == "bge-reranker-v2-m3":
            d[(r["src"], r["tgt"])].append(r["rho_ceil"]["ndcg@5"])
    m = {k: st.mean(v) for k, v in d.items()}
    t1, t3 = [v for k, v in m.items() if k in T1], [v for k, v in m.items() if k not in T1]
    return (st.mean(t1) if t1 else None, st.mean(t3) if t3 else None, min(m.values()) if m else None, max((len(v) for v in d.values()), default=0))


def k4096_table() -> None:
    print("### K = 4,096 at 100K and 150K\n\nLibrary-profile k-means anchors, 500 queries, 3 seeds. Tier 1 is one direction, as in A2. Lines are 0.98 and 0.94. HotpotQA is the corpus this wasn't chosen on.\n")
    sets = [("MS MARCO 100K", D / "m1_grid/ms_100k.jsonl", SCALE / "ms100000_k4096.jsonl"), ("NQ 100K", D / "m1_grid/nq_100k.jsonl", SCALE / "nq100000_k4096.jsonl"), ("HotpotQA 100K", SCALE / "hp100000_k2048.jsonl", SCALE / "hp100000_k4096.jsonl"),
            ("MS MARCO 150K", D / "m2_lib/ms150k.jsonl", SCALE / "ms150000_k4096.jsonl"), ("NQ 150K", D / "m2_lib/nq150k.jsonl", SCALE / "nq150000_k4096.jsonl"), ("HotpotQA 150K", SCALE / "hp150000_k2048.jsonl", SCALE / "hp150000_k4096.jsonl")]
    rows = []
    for name, a, b in sets:
        ra, rb = read(a), read(b)
        for row, pool in (("9", 60), ("9w", 100)):
            x, y = e2e_tiers(ra, row), e2e_tiers(rb, row)
            rows.append([name, pool, f3(x[0]), f3(x[1]), f3(y[0]), f3(y[1]), f3(y[2])])
    table(["Set", "Pool", "K = 2,048, Tier 1", "Tier 3", "K = 4,096, Tier 1", "Tier 3", "Worst pair"], rows)


def pool1000_table() -> None:
    print("### Pool 1,000\n\n`hp_scale/p1000_500q.jsonl`: HotpotQA 1M, K = 8,192, 500 queries, 3 seeds, 4 pairs. Lines are 0.98 and 0.94.\n")
    recs = read(SCALE / "p1000_500q.jsonl")
    rows = []
    for row, pool in (("9x", 300), ("9y", 500), ("9z", 1000)):
        t = e2e_tiers(recs, row)
        rows.append([pool, f3(t[0]), f3(t[1]), f3(t[2]), t[3]])
    table(["Pool", "Tier 1", "Tier 3", "Worst pair", "Seeds"], rows)
    print("`hp_scale/p1000_2000q.jsonl`: the same on 2,000 queries, one seed.\n")
    recs = read(SCALE / "p1000_2000q.jsonl")
    rows = []
    for row, pool in (("9x", 300), ("9y", 500), ("9z", 1000)):
        t = e2e_tiers(recs, row)
        rows.append([pool, f3(t[0]), f3(t[1]), f3(t[2]), t[3]])
    table(["Pool", "Tier 1", "Tier 3", "Worst pair", "Seeds"], rows)


def pool_records() -> list[dict]:
    """ArguAna's first run kept each query's own corpus row in BM25's list, which dense lists exclude: use its rerun."""
    return [r for r in read(FIXES / "pools.jsonl") if r["corpus"] != "arguana"] + read(D / "hp_fixes_arguana/pools.jsonl")


def pools(records: list[dict], corpus: str, queries: int, seeds=None) -> dict:
    """(budget, config) -> {(src, tgt): [records]}."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        if r["corpus"] == corpus and r["queries"] == queries and (seeds is None or r["seed"] in seeds):
            g[(r["budget"], r["config"])][(r["src"], r["tgt"])].append(r)
    return g


def tier_means(pairs: dict, field: str, tier1: bool):
    vals = [st.mean(x[field] for x in v if field in x) for k, v in pairs.items() if (k in T1) == tier1 and any(field in x for x in v)]
    return st.mean(vals) if vals else None


def bm25_tables() -> None:
    recs = pool_records()
    print("### Screen of the fixes\n\n`hp_fixes/pools.jsonl`: HotpotQA 1M, budget 300, seed 0, 500 queries; Tier 1 is `bge-large-en` ← `bge-base-en`, Tier 3 `bge-large-en` ← `qwen3-0.6b`. Retention against plain native and re-index at the same budget.\n")
    rows = []
    screen = {(r["tier"], r["config"]): r for r in recs if r["corpus"] == "hotpotqa-1m" and r["budget"] == 300 and r["queries"] == 500 and r["seed"] == 0}
    for name in ("plain", "bm25_a", "bm25_b", "bm25_c", "graph_a", "graph_b", "graph_c", "graph_2r", "graph_bm25"):
        a, b = screen.get(("T1", name)), screen.get(("T3", name))
        if a and b:
            rows.append([name, f3(a["ret"]), f3(a["ndcg5"]), f3(b["ret"]), f3(b["ndcg5"])])
    table(["Candidates", "Tier 1", "Absolute", "Tier 3", "Absolute"], rows)

    print("### BM25 in the pool, on 2,000 queries and 3 seeds\n\n`hp_fixes/pools.jsonl`. `bm25_a` gives BM25 a quarter of the budget, `bm25_b` a half. \"Strict\" gives native and re-index BM25 too. Lines are 0.98 and 0.94.\n")
    rows = []
    for corpus in ("hotpotqa-1m", "msmarco-1m", "nq-1m"):
        g = pools(recs, corpus, 2000)
        for (budget, config), pairs in sorted(g.items()):
            if config not in ("plain", "bm25_a", "bm25_b"):
                continue
            worst = min(st.mean(x["ret"] for x in v) for v in pairs.values())
            rows.append([label(corpus), budget, config, f3(tier_means(pairs, "ret", True)), f3(tier_means(pairs, "ret", False)), f3(worst), f3(tier_means(pairs, "ret_strict", True)), f3(tier_means(pairs, "ret_strict", False)), f3(tier_means(pairs, "ndcg5", True)), f3(tier_means(pairs, "ndcg5", False)),
                         len(pairs), max(len(v) for v in pairs.values())])
    table(["Corpus", "Budget", "Candidates", "Tier 1", "Tier 3", "Worst pair", "Strict, Tier 1", "Strict, Tier 3", "Absolute, Tier 1", "Absolute, Tier 3", "Pairs", "Seeds"], rows)

    print("### With no bridge at all\n\n`bm25_only`: BM25's top-B and the reranker, against the bridge alone and with BM25. Absolute `nDCG@5`, seed 0; native is plain.\n")
    rows = []
    for corpus in ("hotpotqa-1m", "msmarco-1m", "nq-1m"):
        for budget in (60, 100, 300):
            cell = {r["config"]: r for r in recs if r["corpus"] == corpus and r["budget"] == budget and r["seed"] == 0 and r["tier"] == "T1" and r["queries"] == 500}
            t3 = {r["config"]: r for r in recs if r["corpus"] == corpus and r["budget"] == budget and r["seed"] == 0 and r["tier"] == "T3" and r["queries"] == 500}
            if "bm25_only" in cell:
                rows.append([label(corpus), budget, f3(cell["bm25_only"]["ndcg5"]), f3(cell.get("plain", {}).get("ndcg5")), f3(cell.get("bm25_a", {}).get("ndcg5")), f3(t3.get("plain", {}).get("ndcg5")), f3(t3.get("bm25_a", {}).get("ndcg5")), f3(cell["bm25_only"]["native"])])
    table(["Corpus", "Budget", "BM25 and reranker alone", "Bridge, Tier 1", "Bridge + BM25, Tier 1", "Bridge, Tier 3", "Bridge + BM25, Tier 3", "Native"], rows)

    print("### 8.84M\n\n`hp_fixes/pools.jsonl`, MS MARCO full, 2,000 queries, K = 8,192, both Tier 1 directions.\n")
    g = pools(recs, "msmarco-full", 2000)
    rows = []
    for (budget, config), pairs in sorted(g.items()):
        rows.append([budget, config, f3(tier_means(pairs, "ret", True)), f3(tier_means(pairs, "ret_strict", True)), f3(tier_means(pairs, "ndcg5", True)), f3(st.mean(st.mean(x["native"] for x in v) for v in pairs.values())), len(pairs), max(len(v) for v in pairs.values())])
    table(["Budget", "Candidates", "Tier 1", "Strict", "Absolute", "Native", "Pairs", "Seeds"], rows)

    print("### Other sizes and corpora\n\n`hp_fixes/pools.jsonl`, library-default K and anchors, 500 queries or all, seed 0. Tier 1 and Tier 3 retention; each is one pair.\n")
    rows = []
    corpora = [c for c in dict.fromkeys(r["corpus"] for r in recs) if "@" in c or c in ("fiqa", "trec-covid", "scifact", "scidocs", "arguana")]
    for corpus in corpora:
        for budget in sorted({r["budget"] for r in recs if r["corpus"] == corpus}):
            cell = {(r["tier"], r["config"]): r for r in recs if r["corpus"] == corpus and r["budget"] == budget and r["seed"] == 0}
            get = lambda t, c, f: f3(cell.get((t, c), {}).get(f))
            rows.append([corpus.replace("-1m-nested@", " "), budget, cell.get(("T1", "plain"), {}).get("queries", "-"), get("T1", "plain", "ret"), get("T1", "bm25_a", "ret"), get("T1", "bm25_a", "ret_strict"), get("T3", "plain", "ret"), get("T3", "bm25_a", "ret"), get("T3", "bm25_a", "ret_strict"), get("T1", "plain", "ndcg5"), get("T1", "bm25_a", "ndcg5")])
    table(["Set", "Budget", "Queries", "Tier 1, plain", "+ BM25", "Strict", "Tier 3, plain", "+ BM25", "Strict", "Absolute, Tier 1 plain", "+ BM25"], rows)

    print("### What BM25 holds\n\n`rescue.jsonl`, seed 0: of the positives the map loses from the top 300, the share inside BM25's top k.\n")
    rows = []
    for r in read(FIXES / "rescue.jsonl"):
        h = r["bm25_holds_lost"]
        rows.append([label(r["corpus"]), TIER[r["tier"]], r["n_lost"], f3(h["15"]), f3(h["30"]), f3(h["75"]), f3(h["300"]), f3(r["bm25_holds_kept"]["15"])])
    table(["Corpus", "Tier", "Positives lost", "Top 15", "Top 30", "Top 75", "Top 300", "Top 15, positives the map kept"], rows)
    print("BM25's time per query, `bm25time.jsonl`: idle machine, one CPU thread, 200 queries, top 60.\n")
    table(["Corpus", "Documents", "Postings", "p50, ms", "p95, ms", "Max, ms"], [[label(r["corpus"]), f"{r['docs']:,}", f"{r['postings']:,}", f"{r['p50_ms']:.1f}", f"{r['p95_ms']:.1f}", f"{r['max_ms']:.1f}"] for r in read(FIXES / "bm25time.jsonl")])
    print("BM25 alone, `bm25.jsonl`:\n")
    table(["Corpus", "`R@5`", "`R@10`", "`R@30`", "`R@100`", "`R@300`", "`R@1000`", "`nDCG@5`"], [[label(r["corpus"])] + [f3(r[f"recall@{k}"]) for k in (5, 10, 30, 100, 300, 1000)] + [f3(r["ndcg@5"])] for r in read(FIXES / "bm25.jsonl")])


def main() -> None:
    print("## HotpotQA study: tables\n")
    for section in (noise_table, dose_table, margin_table, signal_table, cover_table, cross_table, form_table, features_table, fits_table, candidates_table, curve_table, k4096_table, pool1000_table, bm25_tables):
        section()


if __name__ == "__main__":
    main()
