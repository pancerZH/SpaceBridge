"""Fit the quality score (design §5.1): calibration-time measurements -> real Recall@30 ceiling retention.

Uses every bridged fit with quality measurements in the given spacebridge-bench JSONL files, validates
by leave-one-dataset-out (LODO), compares against a rho_val-only baseline, and writes the coefficients
to src/spacebridge/quality_calibration.json.

    uv run python scripts/calibrate_quality.py outputs/m2_*.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from difficulty_features import feature_key
from spacebridge_bench.provenance import unique_records
from spacebridge.quality import CALIBRATION_PATH

TARGET = "recall@30"
# Fields that change what a fit means: the training set must agree on all of them (the shipped calibration's 306 fits do).
PROTOCOL = ("proxy", "doc_pair_tokens", "anchor_select", "stage2_rule", "stage2_batch", "stage2_tau_scale", "anchor_bank", "shift_bank")
KEEP_LAST = False
ROWS = ("6", "6b", "6n", "8", "8s", "8n", "8b")
# Candidate measurements; ``fidelity.agree@10`` alone is the rho_val baseline.
FEATURES = [
    "fidelity.agree@10",
    "fidelity.nbr_recall@30",
    "recoverability.nbr_recall@100",
    "fit.val_resid_cos",
    "fit.train_val_gap",
    "fit.rank_ratio",
    "difficulty.teacher_gap_sd",
    "difficulty.cone_angle_deg",
    "coverage.K_over_dt",
]
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def load(paths: list[Path], extra: dict | None = None) -> list[dict]:
    recs = []
    for path in paths:
        for line in path.open():
            r = json.loads(line)
            if r["row"] in ROWS and "quality" in r.get("fit", {}) and "reranker" not in r:
                if extra is not None:  # index-only difficulty features (scripts/difficulty_features.py)
                    f = extra.get(feature_key(r, path))
                    if f is None:
                        continue
                    r["fit"]["quality"]["corpus"] = f
                recs.append(r)
    # One record per configuration, all under one protocol. A repeated configuration is refused unless --keep-last names the choice,
    # and then each replaced record is listed.
    key = lambda r: (r["dataset"], r["src"], r["tgt"], r["K"], r["row"], r.get("seed", 0))
    return unique_records(recs, key, protocol=PROTOCOL, allow_duplicates=KEEP_LAST)


def features(r: dict, names: list[str]) -> list[float]:
    flat = {f"{g}.{k}": v for g, vals in r["fit"]["quality"].items() for k, v in vals.items()}
    return [flat[n] for n in names]


def fit_ridge(X, y, w, alpha):
    mean, scale = np.average(X, axis=0, weights=w), X.std(0) + 1e-12
    Z = (X - mean) / scale
    sw = np.sqrt(w)[:, None]
    A = np.hstack([np.ones((len(Z), 1)), Z]) * sw
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0  # intercept is not penalized
    beta = np.linalg.solve(A.T @ A + reg, A.T @ (y * sw[:, 0]))
    return {"mean": mean, "scale": scale, "intercept": beta[0], "coef": beta[1:]}


def predict(m, X):
    return m["intercept"] + ((X - m["mean"]) / m["scale"]) @ m["coef"]


def lodo(X, y, w, datasets, alpha):
    pred = np.empty_like(y)
    for d in np.unique(datasets):
        test = datasets == d
        pred[test] = predict(fit_ridge(X[~test], y[~test], w[~test], alpha), X[test])
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--write", action="store_true", help=f"write {CALIBRATION_PATH.name}")
    p.add_argument("--keep-last", action="store_true", help="when a configuration repeats, keep the last record and list each replaced one")
    p.add_argument("--features", type=Path, help="difficulty features JSONL: also test rho_val + each corpus feature")
    args = p.parse_args()
    global KEEP_LAST
    KEEP_LAST = args.keep_last
    extra = None
    if args.features:
        extra = {}
        for line in args.features.open():
            d = json.loads(line)
            extra[tuple(d.pop("key"))] = d
    recs = load(args.results, extra)
    datasets = np.array([r["dataset"] for r in recs])
    y = np.array([r["rho_ceil"][TARGET] for r in recs])
    # Inverse-variance weights from each fit's paired-bootstrap CI (50-query sets count less).
    se = np.array([(r["ci_ceil"][TARGET][1] - r["ci_ceil"][TARGET][0]) / 3.92 for r in recs])
    w = 1.0 / np.maximum(se, 0.01) ** 2
    print(f"{len(recs)} fits from {len(set(datasets))} datasets: {dict(zip(*np.unique(datasets, return_counts=True)))}\n")

    results = {}
    candidates = [("rho_val only", FEATURES[:1]), ("all measurements", FEATURES)]
    if extra:
        corpus = [f"corpus.{k}" for k in next(iter(extra.values()))]
        candidates += [(f"rho_val + {c}", FEATURES[:1] + [c]) for c in corpus]
        candidates += [("rho_val + all corpus features", FEATURES[:1] + corpus)]
    for name, feats in candidates:
        X = np.array([features(r, feats) for r in recs])
        best = min(ALPHAS, key=lambda a: np.average((lodo(X, y, w, datasets, a) - y) ** 2, weights=w))
        pred = lodo(X, y, w, datasets, best)
        results[name] = (feats, X, best, pred)
        resid = pred - y
        print(f"## {name} (alpha={best}): LODO weighted RMSE {100 * np.sqrt(np.average(resid**2, weights=w)):.1f} points\n")
        print("| held-out dataset | fits | observed mean | predicted mean | bias | MAE | within ±RMSE band |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        band = np.sqrt(np.average(resid**2, weights=w))
        for d in np.unique(datasets):
            m = datasets == d
            print(
                f"| {d} | {m.sum()} | {100 * y[m].mean():.1f} | {100 * pred[m].mean():.1f} | {100 * resid[m].mean():+.1f} | "
                f"{100 * np.abs(resid[m]).mean():.1f} | {np.mean(np.abs(resid[m]) <= band):.0%} |"
            )
        print()

    # The errors above help pick the alpha and feature set, so they are model-selection errors, optimistic about a procedure that
    # selects. Nested: for each held-out dataset, the alpha and features are chosen on the other datasets alone.
    nested = np.empty_like(y)
    for d in np.unique(datasets):
        out_ = datasets == d
        rest = ~out_
        def inner(feats_, a):
            X_ = np.array([features(r, feats_) for r in recs])[rest]
            return np.average((lodo(X_, y[rest], w[rest], datasets[rest], a) - y[rest]) ** 2, weights=w[rest])
        feats_, a_ = min(((feats_, a) for _, feats_ in candidates for a in ALPHAS), key=lambda c: inner(*c))
        X_all = np.array([features(r, feats_) for r in recs])
        m_ = fit_ridge(X_all[rest], y[rest], w[rest], a_)
        nested[out_] = m_["intercept"] + ((X_all[out_] - m_["mean"]) / m_["scale"]) @ m_["coef"]
    print(f"## Nested leave-one-dataset-out, selection inside each fold: weighted RMSE {100 * np.sqrt(np.average((nested - y) ** 2, weights=w)):.1f} points\n")

    # Ship whichever feature set transfers better to unseen datasets (lower LODO error).
    chosen = min(results, key=lambda k: np.average((results[k][3] - y) ** 2, weights=w))
    feats, X, alpha, pred = results[chosen]
    m = fit_ridge(X, y, w, alpha)
    resid = pred - y
    band = float(np.sqrt(np.average(resid**2, weights=w)))
    per_ds = {d: float(np.abs(resid[datasets == d]).mean()) for d in np.unique(datasets)}
    transfers = sum(v <= band for v in per_ds.values())
    print(f"## final model: {chosen} (all fits)\n")
    for f, c in sorted(zip(feats, m["coef"]), key=lambda t: -abs(t[1])):
        print(f"- {f}: {100 * c:+.2f} points per s.d.")
    print(f"- intercept {100 * m['intercept']:.1f}; band ±{100 * band:.1f} (LODO RMSE); "
          f"transfers on {transfers}/{len(per_ds)} held-out datasets")
    if args.write:
        out = {
            "target": f"rho_ceil {TARGET}",
            "features": feats,
            "mean": m["mean"].tolist(),
            "scale": m["scale"].tolist(),
            "coef": m["coef"].tolist(),
            "intercept": float(m["intercept"]),
            "alpha": alpha,
            "resid_sd": band,
            "lodo_mae_by_dataset": per_ds,
            "n_fits": len(recs),
            # Design §5.1: experimental until LODO error is within the band on >= 4 datasets.
            "experimental": transfers < 4 or len(per_ds) < 4,
            "experimental_means": "False only means the dataset-level error was within the band on at least 4 datasets. Not a reliability claim: the band is not a prediction interval.",
            "calibrated_on": sorted(per_ds),
            "applies_to": "English queries, indexes up to about 171K documents",
            "fitted_by": "scripts/calibrate_quality.py",
        }
        CALIBRATION_PATH.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
