"""The closeout's tables, from the records on disk, into outputs/phase1/closeout/tables/. Rules are as fixed in docs/reports/phase1-closeout.md.

    uv run python scripts/summarize_closeout.py a1 > outputs/phase1/closeout/tables/a1.md
    ... b1, b2, b3, b4, b5, b6, b7
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from summarize_m3 import LINES, STRETCH, absolute, ci, cells_positive, mean_over_pairs, read, rho, short, strict, table, tier, worst  # noqa: E402

BUDGET_LINE = 0.10
HNSW_LINE = {"T1": 0.95, "T3": 0.90}


def a1() -> None:
    print("# A1: the mount budget at K = 4,096's new line, 80,001 MS MARCO documents\n")
    print("Real mounts, `--time-mount`, `outputs/phase1/closeout/a1_budget.jsonl`. Tokens as M3's A3 counted them. The line is 10% of re-index tokens.\n")
    print("| Index ← query | K | Proxy tokens | Anchor-doc tokens | Re-index tokens | Share | Within 10%? | Reaches 10% at |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in read("closeout/a1_budget.jsonl"):
        m = r.get("mount")
        if m:
            total = m["n_tokens"] + m.get("doc_n_tokens", 0)
            share = total / m["corpus_tokens_est"]
            size = 80_001 * share / BUDGET_LINE
            print(f"| {short(r)} | {r['K']:,} | {m['n_tokens']:,} | {m.get('doc_n_tokens', 0):,} | {m['corpus_tokens_est']:,.0f} | {100 * share:.1f}% | {'yes' if share <= BUDGET_LINE else '**no**'} | {size / 1e3:.1f}K |")
    print("\n## Identity after the change, FiQA, Row 1 (`closeout/identity.jsonl`)\n")
    print("| Index ← query | Row 1 `Recall@30` | Native | Of native |\n| --- | --- | --- | --- |")
    recs = read("closeout/identity.jsonl")
    native = {(r["src"], r["tgt"]): r["metrics"]["recall@30"] for r in recs if r["row"] == "0b"}
    for r in recs:
        if r["row"] == "1":
            print(f"| {short(r)} | {r['metrics']['recall@30']:.4f} | {native[(r['src'], r['tgt'])]:.4f} | {r['rho_native']['recall@30']:.4f} |")


def b1_table(rows: list[tuple], lines: dict) -> tuple[list[dict], str]:
    """One line per (corpus, size, tier); returns them for the rule, and the markdown."""
    head = "| Corpus | Size | Queries | K | Tier | Off | On | Off, worst pair | On, worst pair | On: mean CI | Points on - off, (pair, seed) cells | On, strict | Absolute, off → on | Native, re-index at 60 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    out, text = [], head
    for corpus, size, path in rows:
        recs = [r for r in read(path) if "depth" in r]
        for t in ("T1", "T3"):
            on = [r for r in recs if r["row"] == "9h" and tier(r) == t]
            off = [r for r in recs if r["row"] == "9" and tier(r) == t]
            if not on or not off:
                continue
            m_on, m_off = mean_over_pairs(on), mean_over_pairs(off)
            lo, hi = ci(on)
            line = lines[t]
            d = {(r["src"], r["tgt"], r["seed"]): 100 * rho(r) for r in on}
            e = {(r["src"], r["tgt"], r["seed"]): 100 * rho(r) for r in off}
            diff = [d[k] - e[k] for k in d if k in e]
            out.append({"corpus": corpus, "size": size, "tier": t, "on": m_on, "off": m_off, "pass": m_on >= line, "off_pass": m_off >= line, "points": st.mean(diff), "gains": sum(x > 0 for x in diff), "cells": len(diff)})
            n = on[0]["baselines"]["native"]["ndcg@5"], on[0]["baselines"]["oracle"]["ndcg@5"]
            nat, orc = st.mean(r["baselines"]["native"]["ndcg@5"] for r in on), st.mean(r["baselines"]["oracle"]["ndcg@5"] for r in on)
            text += (f"| {corpus} | {size:,} | {on[0]['n_queries']} | {on[0]['K']:,} | {t[1]} | {m_off:.3f}{'' if m_off >= line else ' **miss**'} | {m_on:.3f}{'' if m_on >= line else ' **miss**'} | {worst(off)[1]:.3f} | {worst(on)[1]:.3f} "
                     f"| [{lo:.3f}, {hi:.3f}] | {cells_positive(on, off)} | {mean_over_pairs(on, strict):.3f} | {mean_over_pairs(off, absolute):.3f} → {mean_over_pairs(on, absolute):.3f} | {nat:.3f}, {orc:.3f} |\n")
    return out, text


def b1() -> None:
    print("# B1: BM25 on and off, every corpus by size, budget 60, library default K, 3 seeds, Tier 1 one direction here except at 1M\n")
    print("Lines: 0.98 and 0.94. Off is row `9`, on is row `9h`. Strict gives native and re-index BM25 too. **Retention above 1 isn't a gain over native.** Held out from the choice of BM25: SciFact, ArguAna, SciDocs, FiQA, TREC-COVID. HotpotQA chose it, and MS MARCO and NQ helped.\n")
    rows = [("SciFact", 5_183, "closeout/b1_scifact.jsonl"), ("ArguAna", 8_674, "closeout/b1_arguana.jsonl")]
    for n in (10_000, 20_000, 30_000, 50_000):
        rows += [("MS MARCO", n, f"closeout/b1_ms{n}.jsonl"), ("NQ", n, f"closeout/b1_nq{n}.jsonl"), ("HotpotQA", n, f"followup/hp{n}.jsonl")]
    rows += [("SciDocs", 25_657, "followup/scidocs.jsonl"), ("FiQA", 57_638, "m3_a6/fiqa.jsonl")]
    for n in (75_000, 100_000, 150_000, 200_000, 500_000):
        rows += [("MS MARCO", n, f"m3_a6/ms{n}.jsonl"), ("NQ", n, f"m3_a6/nq{n}.jsonl"), ("HotpotQA", n, f"m3_a6/hp{n}.jsonl")]
    rows += [("TREC-COVID", 171_332, "m3_a6/trec_covid.jsonl"), ("MS MARCO", 1_000_000, "m3_1m/ms.jsonl"), ("NQ", 1_000_000, "m3_1m/nq2k.jsonl"), ("HotpotQA", 1_000_000, "m3_1m/hp.jsonl")]
    rows.sort(key=lambda x: (x[1], x[0]))
    out, text = b1_table(rows, LINES)
    print(text)
    print("\n## The rule: the lowest size at which every corpus measured at or above it passes the approved lines with BM25 on, up to 1M\n")
    sizes = sorted({r["size"] for r in out})
    fails = [r for r in out if not r["pass"]]
    lowest = next((s for s in sizes if not [r for r in fails if r["size"] >= s]), None)
    print(f"Sizes read: {', '.join(f'{s:,}' for s in sizes)}. Cells below a line with BM25 on: {'; '.join(f'''{r['corpus']} {r['size']:,} Tier {r['tier'][1]} {r['on']:.3f}''' for r in fails) or 'none'}.\n")
    print(f"**Lowest size by the rule: {lowest:,}.**\n" if lowest else "**No size meets the rule.**\n")
    print("\n## Who loses from BM25, at each size: mean on - off in points, every tier\n")
    print("| Corpus | Size | Tier | Off | On | Points | Cells that gain |\n| --- | --- | --- | --- | --- | --- | --- |")
    for r in out:
        if r["points"] < 0:
            print(f"| {r['corpus']} | {r['size']:,} | {r['tier'][1]} | {r['off']:.3f} | {r['on']:.3f} | {r['points']:+.1f} | {r['gains']} of {r['cells']} |")
    print("\n## Who misses with BM25 off, at the sizes where it is off today (up to 50K)\n")
    print("| Corpus | Size | Tier | Off | On |\n| --- | --- | --- | --- | --- |")
    for r in out:
        if r["size"] <= 50_000 and not r["off_pass"]:
            print(f"| {r['corpus']} | {r['size']:,} | {r['tier'][1]} | {r['off']:.3f} | {r['on']:.3f} |")
    print("\n## How much of each prefix is positives (small subsets flatter results)\n")
    print("| Corpus | Positives kept in every prefix | Share of 10K | 20K | 30K | 50K |\n| --- | --- | --- | --- | --- | --- |")
    from spacebridge_bench.datasets import load_beir

    for name, label in (("msmarco-1m-nested", "MS MARCO"), ("nq-1m-nested", "NQ"), ("hotpotqa-1m-nested", "HotpotQA")):
        ds = load_beir(name)
        pos = len({d for rel in ds.qrels for d in rel})
        print(f"| {label} | {pos:,} | " + " | ".join(f"{100 * pos / n:.0f}%" for n in (10_000, 20_000, 30_000, 50_000)) + " |")


def b2() -> None:
    print("# B2: does K = 4,096 still earn its budget with BM25 on?\n")
    print("K = 2,048 (`closeout/b2_*.jsonl`) against K = 4,096 (`m3_a6/*.jsonl`), rows `9` and `9h`, 500 queries, 3 seeds. Budget 60. Rule fixed in advance: it earns its budget if it beats K = 2,048 by at least 0.5 points at Tier 3, in all 6 cells, on at least 2 of the 3 corpora, at either size. I read that as each of the 6 cells gaining at least 0.5 points; the looser reading, all 6 positive with a mean of at least 0.5, is alongside.\n")
    print("| Corpus | Size | Tier | K = 2,048, BM25 | K = 4,096, BM25 | Points, (pair, seed) cells | Lowest cell | K = 2,048, strict | K = 4,096, strict | Plain, K = 2,048 | Plain, K = 4,096 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    results = defaultdict(dict)
    for n in (100_000, 150_000):
        for corpus, tag in (("MS MARCO", "ms"), ("NQ", "nq"), ("HotpotQA", "hp")):
            lo_recs = [r for r in read(f"closeout/b2_{tag}{n}.jsonl") if "depth" in r]
            hi_recs = [r for r in read(f"m3_a6/{tag}{n}.jsonl") if "depth" in r]
            for t in ("T1", "T3"):
                pick = lambda recs, row: [r for r in recs if r["row"] == row and tier(r) == t]
                a, b = pick(lo_recs, "9h"), pick(hi_recs, "9h")
                if not a or not b:
                    continue
                d = {(r["src"], r["tgt"], r["seed"]): 100 * rho(r) for r in b}
                e = {(r["src"], r["tgt"], r["seed"]): 100 * rho(r) for r in a}
                diff = [d[k] - e[k] for k in d if k in e]
                print(f"| {corpus} | {n:,} | {t[1]} | {mean_over_pairs(a):.3f} | {mean_over_pairs(b):.3f} | {cells_positive(b, a)} | {min(diff):+.2f} | {mean_over_pairs(a, strict):.3f} | {mean_over_pairs(b, strict):.3f} | {mean_over_pairs(pick(lo_recs, '9')):.3f} | {mean_over_pairs(pick(hi_recs, '9')):.3f} |")
                if t == "T3":
                    results[n][corpus] = {"strict": len(diff) == 6 and min(diff) >= 0.5, "loose": len(diff) == 6 and min(diff) > 0 and st.mean(diff) >= 0.5, "mean": st.mean(diff)}
    print("\n## The rule\n")
    earned = {reading: [n for n, c in results.items() if sum(v[reading] for v in c.values()) >= 2] for reading in ("strict", "loose")}
    for n, c in results.items():
        print(f"- {n:,}: " + "; ".join(f"{k}: mean {v['mean']:+.2f}, each cell at least 0.5: {'yes' if v['strict'] else 'no'}, all positive with mean at least 0.5: {'yes' if v['loose'] else 'no'}" for k, v in c.items()))
    print(f"\n**By the rule: {'K = 4,096 earns its budget, at ' + ', '.join(f'{n:,}' for n in earned['strict']) if earned['strict'] else 'K = 4,096 does not earn its budget with BM25 on'}.** Looser reading: {'earns it, at ' + ', '.join(f'{n:,}' for n in earned['loose']) if earned['loose'] else 'does not earn it'}.")


def b3() -> None:
    print("# B3: the real `Searcher` above 1M, against row `9h` on its own map\n")
    print("`bge-large-en` ← `bge-base-en`, seed 0, K = 8,192, 200 queries, budget 60, exact search on the CPU, models and reranker on the GPU. Line: within ±0.1 points, as A5's. `closeout/b3_searcher_*.jsonl` and `closeout/b3_row_*.jsonl`.\n")
    print("| Corpus | `Searcher` `nDCG@5` | Row `9h` | Points | Within ±0.1? | Grade | The library's own budget | Pool, mean |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = {r["dataset"]: r for tag in ("hp", "ms") for r in read(f"closeout/b3_row_{tag}.jsonl") if r.get("row") == "9h"}
    for r in read("closeout/b3_searcher_hp.jsonl", "closeout/b3_searcher_ms.jsonl"):
        row = rows.get(r["dataset"])
        if row:
            d = 100 * (r["metrics"]["ndcg@5"] - row["metrics"]["ndcg@5"])
            print(f"| {r['dataset']} | {r['metrics']['ndcg@5']:.5f} | {row['metrics']['ndcg@5']:.5f} | {d:+.3f} | {'yes' if abs(d) <= 0.1 else '**no**'} | {r['grade']} | {r['default_budget']} | {r['pool_mean']:.1f} |")


def b4() -> None:
    print("# B4: end to end at the sizes between 1M and the top\n")
    print("Bold reaches the stretch line: 0.95 Tier 1, 0.90 Tier 3. Mean over pairs, 3 seeds, 2,000 queries, K = 8,192. Tier 1 both directions, Tier 3 `bge-large-en` both ways with `qwen3-0.6b`.")
    paths = ["closeout/b4_ms2000000.jsonl", "closeout/b4_ms4000000.jsonl", "closeout/b4_hp2000000.jsonl", "m3_1m/ms.jsonl", "m3_1m/hp.jsonl", "m3_full/ms_t1.jsonl", "m3_full/ms_t3.jsonl", "m3_full/hp.jsonl"]
    recs = [r for r in read(*paths) if r.get("K") == 8192 and r.get("row") in ("9", "9h") and not (r["dataset"].endswith("-1m") and r["src"] == "harrier-0.6b")]
    table(recs, ("9", "9h"), STRETCH, "## By size")
    print("\n## Each pair, budget 60 with BM25: retention [95% CI], strict, absolute\n")
    print("| Set | Pair | Retention | CI | Straddles the line? | Strict | Absolute | Native, re-index |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    pairs = defaultdict(list)
    for r in recs:
        if r["row"] == "9h":
            pairs[(r["dataset"], short(r))].append(r)
    for (ds, p), v in sorted(pairs.items()):
        lo, hi = ci(v)
        line = STRETCH[tier(v[0])]
        print(f"| {ds} | {p} | {st.mean(rho(r) for r in v):.3f} | [{lo:.3f}, {hi:.3f}] | {'yes' if lo < line <= hi else 'no'} | {st.mean(strict(r) for r in v):.3f} | {st.mean(absolute(r) for r in v):.3f} | {st.mean(r['baselines']['native']['ndcg@5'] for r in v):.3f}, {st.mean(r['baselines']['oracle']['ndcg@5'] for r in v):.3f} |")
    print("\n## The first stage, `Recall@300` retention: row `8b`\n")
    print("| Set | Tier | Row `8b` `Recall@300` retention |\n| --- | --- | --- |")
    first = defaultdict(list)
    for r in read(*paths):
        if r.get("row") == "8b" and r.get("K") == 8192 and not (r["dataset"].endswith("-1m") and r["src"] == "harrier-0.6b"):
            first[(r["dataset"], tier(r))].append(r)
    for (ds, t), v in sorted(first.items()):
        print(f"| {ds} | {t[1]} | {mean_over_pairs(v, lambda r: r['rho_ceil']['recall@300']):.3f} |")
    print("\n## No bridge: BM25's top 60 and the reranker (`closeout/hp_fixes/pools.jsonl`, `hp_fixes/pools.jsonl`)\n")
    print("Tier 1 and Tier 3 are one direction each here, and differ only in their ceilings.\n")
    print("| Set | Budget | Tier 1 | Tier 3 | Absolute `nDCG@5` |\n| --- | --- | --- | --- | --- |")
    nob = [r for r in read("closeout/hp_fixes/pools.jsonl", "hp_fixes/pools.jsonl") if r.get("config") == "bm25_only" and r.get("queries") == 2000]
    seen = {}
    for r in nob:
        seen[(r["corpus"], r["budget"], r["tier"])] = r
    for corpus, budget in sorted({k[:2] for k in seen}):
        a, b = seen.get((corpus, budget, "T1")), seen.get((corpus, budget, "T3"))
        if a and b:
            print(f"| {corpus} | {budget} | {a['ret']:.3f} | {b['ret']:.3f} | {a['ndcg5']:.3f} |")


# Fixed before any reading of these bins: words in the query. A bin needs 30 queries in both corpora to count as shared.
LENGTH_EDGES = (6, 8, 10, 12, 15, 20, 30)
LENGTH_BINS = ("<= 6", "7-8", "9-10", "11-12", "13-15", "16-20", "21-30", "> 30")
MIN_BIN = 30


def b5() -> None:
    from hotpot_identity import _tokens, fame_path, ratio_ci
    from hotpot_study import dataset

    print("# B5: does query length explain why titles transfer worse on HotpotQA than on NQ?\n")
    print("From `outputs/phase1/hp_identity/fame_rows/`, the title-transfer records of Test 6, with each query's length in words (`_tokens`, as the title matching uses). HotpotQA's and NQ's gap in title transfer, HotpotQA reweighted to NQ's mix of lengths, as Test 6 reweighted by mentions. Bins fixed before reading: "
          + ", ".join(LENGTH_BINS) + f". A bin is shared at {MIN_BIN} queries or more in both corpora. Rule fixed in advance: it explains the gap if reweighting removes at least 50% of it in both tiers. Seeds change the anchors, not the queries. First stage, 1M, K = 8,192, `bge-large-en` ← `bge-base-en` (Tier 1) and ← `qwen3-0.6b` (Tier 3).\n")
    out = Path("outputs/phase1/hp_identity")
    corpora = ("hotpotqa-1m", "nq-1m")
    lengths = {c: np.array([len(_tokens(q)) for q in dataset(c)[1].queries]) for c in corpora}
    print("| Corpus | Queries with a matched title | Words, median | p10 | p90 |\n| --- | --- | --- | --- | --- |")
    z0 = {c: np.load(fame_path(out, c, "T1", 0)) for c in corpora}
    for c in corpora:
        ln = lengths[c][z0[c]["query"].astype(int)]
        print(f"| {c} | {len(ln):,} | {np.median(ln):.0f} | {np.percentile(ln, 10):.0f} | {np.percentile(ln, 90):.0f} |")
    summary = {}
    for t in ("T1", "T3"):
        print(f"\n## Tier {t[1]}: title transfer by query length, mean over seeds\n")
        print("| Words | HotpotQA: queries | Transfer | Share of its queries | NQ: queries | Transfer | Share of its queries | Gap, NQ - HotpotQA | Shared? |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        per_seed = []
        for seed in (0, 1, 2):
            z = {c: np.load(fame_path(out, c, t, seed)) for c in corpora}
            bins = {c: np.digitize(lengths[c][z[c]["query"].astype(int)], LENGTH_EDGES, right=True) for c in corpora}
            shared = [b for b in range(len(LENGTH_BINS)) if all((bins[c] == b).sum() >= MIN_BIN for c in corpora)]

            def rows_(c, b):
                m = bins[c] == b
                return (float(z[c]["num_t"][m].sum() / z[c]["den_t"][m].sum()) if m.any() else None), int(m.sum())

            def gap(pick, which):
                h, n = pick["hotpotqa-1m"], pick["nq-1m"]
                hp, nq = z["hotpotqa-1m"], z["nq-1m"]
                raw = nq["num_t"][n].sum() / nq["den_t"][n].sum() - hp["num_t"][h].sum() / hp["den_t"][h].sum()
                total, adjusted, used = 0.0, 0.0, 0.0
                for b in which:
                    mh, mn = h[bins["hotpotqa-1m"][h] == b], n[bins["nq-1m"][n] == b]
                    if len(mh) and len(mn):
                        w = nq["den_t"][mn].sum()
                        adjusted += w * hp["num_t"][mh].sum() / hp["den_t"][mh].sum()
                        total += w * nq["num_t"][mn].sum() / nq["den_t"][mn].sum()
                        used += w
                return float(raw), float((total - adjusted) / used)

            everything = {c: np.arange(len(bins[c])) for c in corpora}
            raw, adjusted = gap(everything, shared)
            _, adjusted_all = gap(everything, range(len(LENGTH_BINS)))
            rng = np.random.default_rng(0)
            draws = [gap({c: rng.choice(everything[c], len(everything[c])) for c in corpora}, shared) for _ in range(500)]
            removed = [1 - a / r for r, a in draws]
            trend = {}
            for c in corpora:
                occupied = [b for b in range(len(LENGTH_BINS)) if (bins[c] == b).sum() >= MIN_BIN]
                lo, hi = bins[c] == occupied[0], bins[c] == occupied[-1]
                trend[c] = {"bottom": LENGTH_BINS[occupied[0]], "top": LENGTH_BINS[occupied[-1]], **ratio_ci(z[c]["num_t"][hi], z[c]["den_t"][hi], z[c]["num_t"][lo], z[c]["den_t"][lo], paired=False)}
            per_seed.append({"raw": raw, "adjusted": adjusted, "adjusted_all": adjusted_all, "removed": 1 - adjusted / raw, "removed_ci": [float(np.percentile(removed, q)) for q in (2.5, 97.5)],
                             "trend": trend, "shared": shared, "cells": {(c, b): rows_(c, b) for c in corpora for b in range(len(LENGTH_BINS))},
                             "share_shared": {c: float(np.isin(bins[c], shared).mean()) for c in corpora}})
        for b, label in enumerate(LENGTH_BINS):
            vals = {c: [s["cells"][(c, b)] for s in per_seed] for c in corpora}
            f = lambda c: st.mean(v[0] for v in vals[c]) if vals[c][0][0] is not None else None
            n = {c: vals[c][0][1] for c in corpora}
            th, tn = f("hotpotqa-1m"), f("nq-1m")
            total = {c: sum(s for (cc, _), (_, s) in per_seed[0]["cells"].items() if cc == c) for c in corpora}
            cell = lambda v: "" if v is None else f"{v:.3f}"
            print(f"| {label} | {n['hotpotqa-1m']} | {cell(th)} | {100 * n['hotpotqa-1m'] / total['hotpotqa-1m']:.0f}% | {n['nq-1m']} | {cell(tn)} | {100 * n['nq-1m'] / total['nq-1m']:.0f}% | {'' if th is None or tn is None else f'{tn - th:+.3f}'} | {'yes' if b in per_seed[0]['shared'] else 'no'} |")
        print("\n| Seed | Raw gap | After reweighting, shared bins | Removed [95% CI over queries] | With every bin both corpora occupy | HotpotQA's queries in shared bins | NQ's | HotpotQA, top bin - bottom [CI] | NQ, top bin - bottom [CI] |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for seed, s in enumerate(per_seed):
            th, tn = s["trend"]["hotpotqa-1m"], s["trend"]["nq-1m"]
            print(f"| {seed} | {s['raw']:.3f} | {s['adjusted']:.3f} | {100 * s['removed']:.0f}% [{100 * s['removed_ci'][0]:.0f}%, {100 * s['removed_ci'][1]:.0f}%] | {100 * (1 - s['adjusted_all'] / s['raw']):.0f}% | {100 * s['share_shared']['hotpotqa-1m']:.0f}% | {100 * s['share_shared']['nq-1m']:.0f}% "
                  f"| {th['top']} vs {th['bottom']}: {th['diff']:+.3f} [{th['ci'][0]:+.3f}, {th['ci'][1]:+.3f}] | {tn['top']} vs {tn['bottom']}: {tn['diff']:+.3f} [{tn['ci'][0]:+.3f}, {tn['ci'][1]:+.3f}] |")
        summary[t] = st.mean(s["removed"] for s in per_seed)
    both = all(v >= 0.5 for v in summary.values())
    print(f"\n**By the rule: mean removed {100 * summary['T1']:.0f}% at Tier 1 and {100 * summary['T3']:.0f}% at Tier 3, so query length {'explains' if both else 'does not explain'} the gap.**")


def b6() -> None:
    print("# B6: budget and grade, at 1M and above, from records only\n")
    print("The grade is row `8b`'s, on the record. `Serves` is what `Searcher` gives that grade with BM25 on (`lexical_rerank_pool`): 60 for A or B, 100 for C or D. Gain is points of retention at budget 100 (row `9hw`) over 60 (row `9h`), mean ± s.d. over seeds. 3 seeds, 2,000 queries. Budget 100 was not run at 2M and 4M, whose rows are `0a 0b 8b 9 9h`.\n")
    sets = [("MS MARCO", 1_000_000, ["m3_1m/ms.jsonl"]), ("NQ", 1_000_000, ["m3_1m/nq2k.jsonl"]), ("HotpotQA", 1_000_000, ["m3_1m/hp.jsonl"]), ("MS MARCO", 2_000_000, ["closeout/b4_ms2000000.jsonl"]), ("HotpotQA", 2_000_000, ["closeout/b4_hp2000000.jsonl"]), ("MS MARCO", 4_000_000, ["closeout/b4_ms4000000.jsonl"]),
            ("HotpotQA", 5_233_329, ["m3_full/hp.jsonl"]), ("MS MARCO", 8_841_823, ["m3_full/ms_t1.jsonl", "m3_full/ms_t3.jsonl"])]
    print("| Corpus | Size | Pair | Tier | Grades by seed | Score | Serves | At 60 | At 100 | Gain | Strict at 60 → 100 | Absolute at 60 → 100 | Line | Margin at 60 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for corpus, size, files in sets:
        recs = read(*files)
        pairs = sorted({(r["src"], r["tgt"]) for r in recs if r.get("row") == "9h" and r.get("K") == 8192})
        for src, tgt in pairs:
            pick = lambda row: {r["seed"]: r for r in recs if r.get("row") == row and r["src"] == src and r["tgt"] == tgt and r.get("K") == 8192}
            g, h, w = pick("8b"), pick("9h"), pick("9hw")
            grades = [g[s]["quality"]["grade"] for s in sorted(g)]
            score = st.mean(g[s]["quality"]["score"] for s in g)
            serves = 100 if any(x in "CD" for x in grades) else 60
            t = tier(h[0])
            line = (LINES if size <= 1_000_000 else STRETCH)[t]
            at60 = st.mean(rho(r) for r in h.values())
            if w:
                gains = [100 * (rho(w[s]) - rho(h[s])) for s in h if s in w]
                at100 = f"{st.mean(rho(r) for r in w.values()):.3f}"
                gain = f"{st.mean(gains):+.2f} ± {st.stdev(gains):.2f}"
                strict_s = f"{st.mean(strict(r) for r in h.values()):.3f} → {st.mean(strict(r) for r in w.values()):.3f}"
                abs_s = f"{st.mean(absolute(r) for r in h.values()):.3f} → {st.mean(absolute(r) for r in w.values()):.3f}"
            else:
                gains, at100, gain = [], "", ""
                strict_s = f"{st.mean(strict(r) for r in h.values()):.3f}"
                abs_s = f"{st.mean(absolute(r) for r in h.values()):.3f}"
            print(f"| {corpus} | {size:,} | {src} ← {tgt} | {t[1]} | {' '.join(grades)} | {score:.1f} | {serves} | {at60:.3f} | {at100} | {gain} | {strict_s} | {abs_s} | {line:.2f} | {100 * (at60 - line):+.1f} |")
            rows.append({"corpus": corpus, "size": size, "pair": f"{src} ← {tgt}", "tier": t, "serves": serves, "at60": at60, "gain": st.mean(gains) if gains else None, "margin": 100 * (at60 - line), "score": score})
    print("\n## Does the grade point at the pairs budget 100 helps most?\n")
    have = [r for r in rows if r["gain"] is not None]
    if have:
        c = [r for r in have if r["serves"] == 100]
        ab = [r for r in have if r["serves"] == 60]
        if c:
            print(f"- Graded C or D, so served 100: {len(c)} pairs, mean gain {st.mean(r['gain'] for r in c):+.2f}, mean margin over the line {st.mean(r['margin'] for r in c):+.1f}.")
        if ab:
            print(f"- Graded A or B, so served 60: {len(ab)} pairs, mean gain {st.mean(r['gain'] for r in ab):+.2f}, mean margin over the line {st.mean(r['margin'] for r in ab):+.1f}.")
        xs, ys = np.array([r["score"] for r in have]), np.array([r["gain"] for r in have])
        ranks = lambda a: np.argsort(np.argsort(a))
        print(f"- Spearman of score with what budget 100 gains, over {len(have)} pairs: {float(np.corrcoef(ranks(xs), ranks(ys))[0, 1]):+.2f}. Of margin with gain: {float(np.corrcoef(ranks(np.array([r['margin'] for r in have])), ranks(ys))[0, 1]):+.2f}.")


def b7() -> None:
    print("# B7: HNSW at 8.84M, MS MARCO, one index, `bge-large-en`, M = 32, efConstruction = 200\n")
    print("Budget 60 is the reading; 2,000 queries, 3 seeds. Tier 1 is one direction here (`bge-large-en` ← `bge-base-en`) and Tier 3 one direction (`bge-large-en` ← `qwen3-0.6b`). Retention is absolute `nDCG@5` over the plain exact ceiling at the same budget, min(native, re-index), both reranked, as the rows define it. `closeout/b7_hnsw.jsonl`, `scripts/hnsw_study.py`.\n")
    res = {r["key"]: r for r in read("closeout/b7_hnsw.jsonl")}
    rows = {(r["src"], r["tgt"], r["seed"], r["row"]): r for r in read("m3_full/ms_t1.jsonl", "m3_full/ms_t3.jsonl") if r.get("K") == 8192 and "depth" in r}
    tiers = {"bge-base-en": "T1", "qwen3-0.6b": "T3"}
    pools = sorted({r["pool"] for r in res.values() if "pool" in r})
    efs = sorted({int(k.split("|ef")[1].split("|")[0]) for k in res if k.startswith("lexical|")})
    row_at = {60: "9h", 100: "9hw", 300: "9hx"}
    print("## Do the fits match the rows? `hnsw_study.py`'s exact list with BM25 against row `9h`, `9hw`, `9hx`, same pair and seed\n")
    print("| Pair | Budget | Largest difference over seeds, points of `nDCG@5` |\n| --- | --- | --- |")
    for tgt in tiers:
        for pool in pools:
            diffs = [100 * (res[f"lexical|{tgt}|s{s}|ef{efs[0]}|pool{pool}"]["ndcg5_exact"] - rows[("bge-large-en", tgt, s, row_at[pool])]["metrics"]["ndcg@5"]) for s in (0, 1, 2) if f"lexical|{tgt}|s{s}|ef{efs[0]}|pool{pool}" in res]
            if diffs:
                print(f"| bge-large-en ← {tgt} | {pool} | {max(diffs, key=abs):+.3f} |")
    print("\n## End to end\n")
    print("| Tier | Query model | ef | Budget | Exact, BM25 | HNSW, BM25 | HNSW's cost, points, cells it loses | Retention, HNSW | Line | Plain exact | Plain HNSW | Recall of exact top-budget, bridged | Native |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    verdict = {}
    for tgt, t in tiers.items():
        for ef in efs:
            for pool in pools:
                lex = [res.get(f"lexical|{tgt}|s{s}|ef{ef}|pool{pool}") for s in (0, 1, 2)]
                plain = [res.get(f"hnsw|{tgt}|s{s}|ef{ef}|pool{pool}") for s in (0, 1, 2)]
                native = res.get(f"hnsw|native|ef{ef}|pool{pool}")
                oracle = res.get(f"oracle|{tgt}|pool{pool}")
                if None in lex or None in plain or native is None or oracle is None:
                    continue
                m = lambda rs, k: st.mean(r[k] for r in rs)
                ceil = min(native["ndcg5_exact"], oracle["ndcg5_exact"])
                ret = m(lex, "ndcg5_approx") / ceil
                diffs = [100 * (r["ndcg5_approx"] - r["ndcg5_exact"]) for r in lex]
                print(f"| {t[1]} | {tgt} | {ef} | {pool} | {m(lex, 'ndcg5_exact'):.4f} | {m(lex, 'ndcg5_approx'):.4f} | {st.mean(diffs):+.2f}, {sum(d < 0 for d in diffs)} of 3 | {ret:.3f} | {HNSW_LINE[t]:.2f} | {m(plain, 'ndcg5_exact'):.4f} | {m(plain, 'ndcg5_approx'):.4f} | {m(plain, 'recall_of_exact'):.3f} | {native['recall_of_exact']:.3f} |")
                if pool == 60:
                    verdict[(t, ef)] = ret
    print("\n## The reading, fixed in advance, budget 60, mean over seeds\n")
    if verdict:
        at = lambda ef: all(verdict[(t, ef)] >= HNSW_LINE[t] for t in ("T1", "T3") if (t, ef) in verdict)
        v = "**The supported range holds on HNSW**" if at(500) else "**It holds at a higher ef: ef 1000**" if 1000 in efs and at(1000) else "**It doesn't hold on HNSW**"
        print("; ".join(f"Tier {t[1]} at ef {ef}: {x:.3f}" for (t, ef), x in sorted(verdict.items())) + f". {v}.")


if __name__ == "__main__":
    {"a1": a1, "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7}[sys.argv[1]]()
