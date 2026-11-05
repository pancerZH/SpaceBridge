"""Per-language validation rho_val,Lq (design §5) against real cross-lingual retention.

Reads Row 7 / 7b records (registered query language) that carry ``fit.lang_val`` (FLORES devtest pairs, see
SpaceBridge.validate_query_language) and compares candidate predictors of real ceiling ``Recall@30`` retention:

* ``en``: the shipped score (English-proxy rho_val through the calibration), blind to the language shift;
* ``abs``: the calibration applied to the zh FLORES agreement directly (FLORES sentences are not proxies);
* ``drop``: the calibration applied to rho_val minus the agreement drop src -> lang on the same FLORES pairs;
* ``ratio``: the calibration applied to rho_val x (lang agreement / src agreement);
* ``rratio``: the English score x (lang counterpart Recall@10 / src counterpart Recall@10);
* ``mratio``: the English score x (lang counterpart MRR / src counterpart MRR).

Per dataset: mean signed error (predicted − real, points), MAE, RMSE and the Spearman correlation over fits.

    uv run python scripts/lang_val_study.py outputs/flores_b2_*.jsonl [--set unfiltered]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge.quality import load_calibration

CANDIDATES = ("en", "abs", "drop", "ratio", "rratio", "mratio")


from spacebridge_bench.metrics import spearman as tie_aware_spearman  # noqa: E402


def predictors(r: dict, cal: dict, tag: str) -> dict:
    lv = r["fit"]["lang_val"][tag]
    f = lambda a: 100.0 * (cal["intercept"] + cal["coef"][0] * (a - cal["mean"][0]) / cal["scale"][0])
    a0 = r["fit"]["quality"]["fidelity"]["agree@10"]
    en = f(a0)
    return {
        "en": en,
        "abs": f(lv["agree@10_lang"]),
        "drop": f(a0 - (lv["agree@10_src"] - lv["agree@10_lang"])),
        "ratio": f(a0 * lv["agree@10_lang"] / max(lv["agree@10_src"], 1e-9)),
        "rratio": en * lv["recall@10_lang"] / max(lv["recall@10_src"], 1e-9),
        "mratio": en * lv["mrr_lang"] / max(lv["mrr_src"], 1e-9),
    }


def spearman(a, b) -> float:
    return float("nan") if len(a) < 3 else tie_aware_spearman(a, b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--set", default="filtered", choices=("filtered", "unfiltered"), help="FLORES devtest set")
    p.add_argument("--rows", nargs="*", default=["7b"])
    p.add_argument("--detail", action="store_true", help="also list every fit")
    args = p.parse_args()
    cal = load_calibration()
    recs = {}
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if r["row"] in args.rows and "reranker" not in r and args.set in r.get("fit", {}).get("lang_val", {}):
                key = (r["dataset"], r["src"], r["tgt"], r["K"], r["row"], r.get("seed", 0), r.get("shift_bank"))
                recs[key] = r  # last run wins
    by_ds = defaultdict(list)
    for k, r in sorted(recs.items()):
        by_ds[(k[0], k[6])].append(r)

    print(f"FLORES devtest set: {args.set}; real = rho_ceil recall@30 (points); error = predicted − real")
    print("| dataset | Δμ bank | fits | real mean | " + " | ".join(f"{c} err / MAE / ρs" for c in CANDIDATES) + " |")
    print("| --- | --- | --- | --- | " + " | ".join("---" for _ in CANDIDATES) + " |")
    pooled = defaultdict(list)
    for (ds, bank), rs in sorted(by_ds.items()):
        real = np.array([100 * r["rho_ceil"]["recall@30"] for r in rs])
        cells = []
        for c in CANDIDATES:
            pred = np.array([predictors(r, cal, args.set)[c] for r in rs])
            err = pred - real
            pooled[c].extend(err.tolist())
            cells.append(f"{err.mean():+.1f} / {np.abs(err).mean():.1f} / {spearman(pred, real):+.2f}")
        print(f"| {ds} | {bank} | {len(rs)} | {real.mean():.1f} | " + " | ".join(cells) + " |")
    print("\npooled over datasets: " + ", ".join(
        f"{c} RMSE {np.sqrt(np.mean(np.square(e))):.1f} (mean {np.mean(e):+.1f})" for c, e in pooled.items()))

    if args.detail:
        print("\n| dataset | index ← queries | K | seed | real | " + " | ".join(CANDIDATES) +
              " | agree src / lang | R@10 src / lang / teacher |")
        print("| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in CANDIDATES) + " | --- | --- |")
        for k, r in sorted(recs.items()):
            pr, lv = predictors(r, cal, args.set), r["fit"]["lang_val"][args.set]
            print(f"| {k[0]} | {k[1]} ← {k[2]} | {k[3]} | {k[5]} | {100 * r['rho_ceil']['recall@30']:.1f} | "
                  + " | ".join(f"{pr[c]:.1f}" for c in CANDIDATES)
                  + f" | {lv['agree@10_src']:.3f} / {lv['agree@10_lang']:.3f} | "
                  f"{lv['recall@10_src']:.3f} / {lv['recall@10_lang']:.3f} / {lv['recall@10_teacher']:.3f} |")


if __name__ == "__main__":
    main()
