"""M3 C3: first-stage retention by size above 1M, from scale_curve.py records, and the 1M prefix against earlier curves.

    uv run python scripts/summarize_m3_curves.py > outputs/phase1/m3_tables/curves.md
"""
import statistics as st
from collections import defaultdict
from pathlib import Path

from spacebridge_bench import durable

ROOT = Path("outputs/phase1")
T1 = {("bge-large-en", "bge-base-en"), ("bge-base-en", "bge-large-en")}


def read(path: str) -> list[dict]:
    return durable.read_jsonl(ROOT / path)[0]


def tier(r: dict) -> str:
    return "T1" if (r["src"], r["tgt"]) in T1 else "T3"


def main() -> None:
    print("## First stage above 1M: retention, K = 8,192, positives kept out of the anchors, 3 seeds, mean over pairs\n")
    print("| Corpus | Tier | Size | `Recall@30` | `Recall@100` | `Recall@300` | `Recall@1000` | Absolute `Recall@300`, bridged (native, re-index) | Worst pair `Recall@300` |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, path in (("MS MARCO", "m3_curves/ms_t3.jsonl"), ("HotpotQA", "m3_curves/hp.jsonl")):
        recs = read(path)
        g = defaultdict(list)
        for r in recs:
            g[(tier(r), r["size"])].append(r)
        for (t, size), rs in sorted(g.items()):
            m = lambda k: st.mean(r["rho_ceil"][k] for r in rs)
            pairs = defaultdict(list)
            for r in rs:
                pairs[f"{r['src']} ← {r['tgt']}"].append(r["rho_ceil"]["recall@300"])
            worst = min(st.mean(v) for v in pairs.values())
            a = lambda f: st.mean(r[f]["recall@300"] for r in rs)
            print(f"| {name} | {t} | {size:,} | {m('recall@30'):.3f} | {m('recall@100'):.3f} | {m('recall@300'):.3f} | {m('recall@1000'):.3f} | {st.mean(r['metrics']['recall@300'] for r in rs):.3f} ({a('native'):.3f}, {a('oracle'):.3f}) | {worst:.3f} |")
    print("\n## The 1M prefix against the earlier curves, `Recall@300` retention, by (pair, seed), points\n")
    print("| Corpus | Earlier curve | Pairs compared | Largest difference | Mean difference |\n| --- | --- | --- | --- | --- |")
    for name, path, ref in (("HotpotQA", "m3_curves/hp.jsonl", "hp_scale/curve.jsonl"),):
        now = {(r["src"], r["tgt"], r["seed"]): r for r in read(path) if r["size"] == 1_000_000}
        old = {(r["src"], r["tgt"], r["seed"]): r for r in read(ref) if r["size"] == 1_000_000 and r["K"] == 8192}
        d = {k: 100 * (now[k]["rho_ceil"]["recall@300"] - old[k]["rho_ceil"]["recall@300"]) for k in now if k in old}
        print(f"| {name} | `{ref}` | {len(d)} | {max(abs(v) for v in d.values()):.2f} | {st.mean(d.values()):+.2f} |")
    now = {(r["src"], r["tgt"], r["seed"]): r for r in read("m3_curves/ms_t3.jsonl") if r["size"] == 1_000_000}
    for ref in [ROOT / "m1_nested/curve_ms.jsonl"]:
        old = {(r["src"], r["tgt"], r["seed"]): r for r in durable.read_jsonl(ref)[0] if r.get("size") == 1_000_000 and r.get("K") == 8192}
        d = {k: 100 * (now[k]["rho_ceil"]["recall@300"] - old[k]["rho_ceil"]["recall@300"]) for k in now if k in old}
        if d:
            print(f"| MS MARCO | `{ref.relative_to(ROOT)}` | {len(d)} | {max(abs(v) for v in d.values()):.2f} | {st.mean(d.values()):+.2f} |")


if __name__ == "__main__":
    main()
