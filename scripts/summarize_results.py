"""Aggregate spacebridge-bench JSONL results into Markdown tables for the performance report.

    uv run python scripts/summarize_results.py outputs/fiqa.jsonl
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

BRIDGE_ROWS = ["2", "3s", "3r", "3b", "6", "6b", "6n", "8", "8s", "8n", "8b"]
TIER_OF = {("bge-base-en", "bge-large-en"): "1", ("bge-large-en", "bge-base-en"): "1"}
HIGH, MODERATE = 0.58, 0.40


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_name(r: dict) -> str:
    return f"{r['src']} ← {r['tgt']}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--K", type=int, default=2048, help="anchor count for the headline tables")
    args = p.parse_args()
    recs = [r for r in (json.loads(line) for line in args.results.open()) if "reranker" not in r]
    by = {(r["src"], r["tgt"], r["K"], r["row"]): r for r in recs}  # last run wins
    pairs = list(dict.fromkeys((r["src"], r["tgt"]) for r in recs))
    Ks = sorted({r["K"] for r in recs})
    K = args.K if args.K in Ks else Ks[-1]
    ds = recs[0]["dataset"]
    print(f"# {ds}: {len(pairs)} pairs, K ∈ {Ks}, {recs[0]['n_queries']} test queries\n")

    print(f"## Baselines and best rows at K = {K} (Recall@30 / nDCG@10, ceiling retention with 95% CI)\n")
    rows = []
    for src, tgt in pairs:
        g = lambda row: by.get((src, tgt, K, row))
        nat, orc = g("0b")["metrics"], g("0a")["metrics"]
        cells = [f"{TIER_OF.get((src, tgt), '3')}", f"{src} ← {tgt}", f"{nat['recall@30']:.3f} / {nat['ndcg@10']:.3f}", f"{orc['recall@30']:.3f} / {orc['ndcg@10']:.3f}", f"{g('1')['rho_ceil']['recall@30']:.3f}"]
        for row in ("2", "6", "6n", "8", "8n", "8b"):
            r = g(row)
            if r is None:
                cells.append("")
                continue
            lo, hi = r["ci_ceil_recall@30"]
            cells.append(f"{r['rho_ceil']['recall@30']:.3f} [{lo:.2f},{hi:.2f}]")
        rows.append(cells)
    print(table(["Tier", "index ← queries", "native R@30 / nDCG", "oracle R@30 / nDCG", "Row 1 identity", "Row 2 naive", "Row 6", "Row 6n no-match", "Row 8 +S2", "Row 8n no-match", "Row 8b +docs"], rows))

    print(f"\n## Ablation means over pairs at K = {K}\n")
    rows = []
    for row in ["1"] + BRIDGE_ROWS:
        rs = [by[(s, t, K, row)] for s, t in pairs if (s, t, K, row) in by]
        if not rs:
            continue
        mean = lambda f: np.mean([f(r) for r in rs])
        wins = sum(r["rho_ceil"]["recall@30"] >= 0.95 for r in rs)
        rows.append([
            row, rs[0]["description"],
            f"{mean(lambda r: r['rho_ceil']['recall@10']):.3f}", f"{mean(lambda r: r['rho_ceil']['recall@30']):.3f}",
            f"{mean(lambda r: r['rho_ceil']['ndcg@10']):.3f}", f"{mean(lambda r: r['rho_oracle']['recall@30']):.3f}",
            f"{mean(lambda r: r['fit']['rho_val']):.3f}", f"{wins}/{len(rs)}",
        ])
    print(table(["Row", "configuration", "ρceil R@10", "ρceil R@30", "ρceil nDCG@10", "ρoracle R@30", "ρ_val", "pairs ≥ 95% R@30"], rows))

    print("\n## Anchor-count sweep (ceiling Recall@30 retention)\n")
    for row in ("6", "6n", "8", "8s", "8b"):
        rows = []
        for src, tgt in pairs:
            rows.append([f"{src} ← {tgt}"] + [f"{by[(src, tgt, k, row)]['rho_ceil']['recall@30']:.3f}" if (src, tgt, k, row) in by else "" for k in Ks])
        if not any((s, t, k, row) in by for s, t in pairs for k in Ks):
            continue
        rows.append(["**mean**"] + [f"**{np.mean([by[(s, t, k, row)]['rho_ceil']['recall@30'] for s, t in pairs if (s, t, k, row) in by]):.3f}**" for k in Ks])
        print(f"Row {row}\n")
        print(table(["index ← queries"] + [f"K={k}" for k in Ks], rows))
        print()

    print("## ρ_val vs. real retention (all bridged rows, all K)\n")
    fits = [r for r in recs if r["row"] in BRIDGE_ROWS]
    a = np.array([r["fit"]["rho_val"] for r in fits])
    y = np.array([r["rho_ceil"]["recall@30"] for r in fits])
    slope, inter = np.polyfit(a, y, 1)
    resid = y - (slope * a + inter)
    within = [spearman(a[m], y[m]) for m in (np.array([(r["src"], r["tgt"]) == pr for r in fits]) for pr in pairs)]
    print(f"- {len(fits)} fits; Spearman pooled {spearman(a, y):.3f}, within-pair mean {np.mean(within):.3f} (min {min(within):.3f})")
    print(f"- linear fit: retention ≈ {slope:.3f}·ρ_val + {inter:.3f} (R² {1 - resid.var() / y.var():.3f}, residual s.d. {resid.std():.3f})")
    print(f"- implied ρ_val for 90% / 75% retention: {(0.90 - inter) / slope:.3f} / {(0.75 - inter) / slope:.3f}\n")
    status = collections.Counter()
    for r in fits:
        band = "≥90%" if r["rho_ceil"]["recall@30"] >= 0.90 else "75–90%" if r["rho_ceil"]["recall@30"] >= 0.75 else "<75%"
        status[(r["fit"]["status"], band)] += 1
    rows = [[st] + [str(status[(st, b)]) for b in ("≥90%", "75–90%", "<75%")] for st in ("HIGH_CONFIDENCE", "MODERATE_DRIFT", "LOW_ALIGNMENT_WARN")]
    print(table(["status (thresholds 0.58 / 0.40)", "real ≥90%", "real 75–90%", "real <75%"], rows))

    print(f"\n## Timing at K = {K}\n")
    rows = []
    for src, tgt in pairs:
        r8b, r8 = by.get((src, tgt, K, "8b")), by.get((src, tgt, K, "8"))
        if r8 is None:
            continue
        m, f = r8["mount"], r8["fit"]
        doc = (r8b or {}).get("mount", {})
        rows.append([
            f"{src} ← {tgt}", f"{m['encode_ms']:.0f}", f"{m['tokens_per_s'] / 1e3:.0f}k",
            f"{doc['doc_encode_ms']:.0f}" if "doc_encode_ms" in doc else "",
            f"{f['stage1_ms']:.0f}", f"{f['stage2_ms']:.0f}", f"{f['val_ms']:.1f}", f"{r8['projection_us']:.0f}",
        ])
    print(table(["index ← queries", "proxy encode ms", "tok/s", "anchor-doc encode ms (8b)", "Stage 1 ms", "Stage 2 ms", "ρ_val ms", "projection µs"], rows))


if __name__ == "__main__":
    main()
