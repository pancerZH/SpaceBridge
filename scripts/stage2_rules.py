"""Offline evaluation of Stage-2 checkpoint-selection rules on recorded checkpoint curves.

Input: spacebridge-bench JSONL written with ``--stage2-curves``. Each Stage-2 record has the label-free signals of
every checkpoint (``fit.stage2_curve``: teacher agree@10 and own-doc MRR on held-out proxies, and held-out doc-pair
agree@10 / MRR) and the real test retention of every checkpoint and of two checkpoint averages (``stage2_real``).
A rule maps the label-free curve to a checkpoint (or an average); this script looks up what that choice retains.

Per dataset x row kind (identity Row 1 vs. bridged Row 8b / 7b) x K, for every rule: mean real retention, the
difference to the shipped rule with a 95% bootstrap CI over (pair, seed) cells, the seed-to-seed swing (per-pair
s.d. over seeds, mean over pairs), and regret against the best checkpoint in hindsight.

    uv run python scripts/stage2_rules.py outputs/m4_b2_curves_*.jsonl
    uv run python scripts/stage2_rules.py outputs/m4_b2_curves_fiqa.jsonl --rules floor0.9 tail_mean --by-pair
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

Curve = list[dict]  # [{"step": 0, "agree": ..., "mrr": ..., "doc_agree": ..., "doc_mrr": ...}, ...]


def _floor(ratio: float) -> Callable[[Curve], str]:
    def rule(c: Curve) -> str:
        ok = [s for s in c if s["agree"] >= ratio * c[0]["agree"]]
        return str(max(ok, key=lambda s: (s["mrr"], -s["step"]))["step"])
    return rule


def _gap_floor(gamma: float) -> Callable[[Curve], str]:
    """Allowed agreement loss proportional to step 0's disagreement with the teacher, gamma * (1 - a0)."""
    def rule(c: Curve) -> str:
        floor = c[0]["agree"] - gamma * (1 - c[0]["agree"])
        ok = [s for s in c if s["agree"] >= floor]
        return str(max(ok, key=lambda s: (s["mrr"], -s["step"]))["step"])
    return rule


def _tradeoff(lam: float, rel: bool = True) -> Callable[[Curve], str]:
    """Soft trade-off without a threshold: own-doc MRR gain + lam * agreement change (relative to step 0)."""
    def rule(c: Curve) -> str:
        m0, a0 = c[0]["mrr"], c[0]["agree"]
        if rel:
            J = lambda s: (s["mrr"] - m0) / m0 + lam * (s["agree"] - a0) / a0
        else:
            J = lambda s: (s["mrr"] - m0) + lam * (s["agree"] - a0)
        return str(max(c, key=lambda s: (J(s), -s["step"]))["step"])
    return rule


def _smoothed_floor(ratio: float) -> Callable[[Curve], str]:
    """Floor rule on a 3-checkpoint moving average of both signals (endpoints use their one neighbour)."""
    def rule(c: Curve) -> str:
        sm = []
        for i, s in enumerate(c):
            nb = c[max(0, i - 1): i + 2] if i > 0 else c[:1]  # step 0 is not smoothed with Stage-2 checkpoints
            sm.append({"step": s["step"], "agree": np.mean([x["agree"] for x in nb]), "mrr": np.mean([x["mrr"] for x in nb])})
        return _floor(ratio)(sm)
    return rule


def _doc(signal: str) -> Callable[[Curve], str]:
    """Held-out doc pairs as the validation set."""
    def rule(c: Curve) -> str:
        return str(max(c, key=lambda s: (s[signal], -s["step"]))["step"])
    return rule


def _avg_unless_blocked(ratio: float) -> Callable[[Curve], str]:
    """Average of the Stage-2 checkpoints (10-50) when their mean agreement keeps ``ratio`` of step 0's, else step 0."""
    def rule(c: Curve) -> str:
        return "avg_10_50" if np.mean([s["agree"] for s in c[1:]]) >= ratio * c[0]["agree"] else "0"
    return rule


def _tail(lo: int, gamma: float = None, ratio: float = None, mrr_tol: float = None) -> Callable[[Curve, dict], str]:
    """Average of the checkpoints from step ``lo`` on, if its own agree@10 clears the floor (gap floor
    a0 - gamma * (1 - a0), or ratio * a0) and, with ``mrr_tol``, its own-doc MRR is within mrr_tol of step 0's;
    otherwise the best-MRR checkpoint above the same floor."""
    def rule(c: Curve, real: dict) -> str:
        a0, m0 = c[0]["agree"], c[0]["mrr"]
        floor = a0 - gamma * (1 - a0) if gamma is not None else ratio * a0
        sig = real[f"avg_{lo}_50"]["signals"]
        if sig["agree"] >= floor and (mrr_tol is None or sig["mrr"] >= m0 - mrr_tol):
            return f"avg_{lo}_50"
        ok = [x for x in c if x["agree"] >= floor]
        return str(max(ok, key=lambda x: (x["mrr"], -x["step"]))["step"])
    rule.needs_real = True
    return rule


RULES: dict[str, Callable[[Curve], str]] = {
    "floor0.9": _floor(0.9),  # shipped (M2)
    "floor0.8": _floor(0.8),
    "floor0.95": _floor(0.95),
    "floor0": _floor(0.0),  # own-doc MRR alone
    "step0": lambda c: "0",
    "step20": lambda c: "20",
    "step30": lambda c: "30",
    "step50": lambda c: "50",
    "avg_all": lambda c: "avg_all",
    "avg_10_50": lambda c: "avg_10_50",
    "avg_unless_blocked0.9": _avg_unless_blocked(0.9),
    "avg_unless_blocked0.8": _avg_unless_blocked(0.8),
    "gap_floor0.1": _gap_floor(0.1),
    "gap_floor0.25": _gap_floor(0.25),
    "gap_floor0.5": _gap_floor(0.5),
    "tradeoff_rel0.5": _tradeoff(0.5),
    "tradeoff_rel1": _tradeoff(1.0),
    "tradeoff_rel2": _tradeoff(2.0),
    "smooth_floor0.9": _smoothed_floor(0.9),
    "doc_mrr": _doc("doc_mrr"),
    "doc_agree": _doc("doc_agree"),
    "avg_20_50": lambda c: "avg_20_50",
    "avg_30_50": lambda c: "avg_30_50",
    "tail20_gap0.25": _tail(20, gamma=0.25),
    "tail20_gap0.5": _tail(20, gamma=0.5),
    "tail30_gap0.25": _tail(30, gamma=0.25),
    "tail20_floor0.9": _tail(20, ratio=0.9),
    "tail20_gap0.25_mrr0.01": _tail(20, gamma=0.25, mrr_tol=0.01),
    "tail20_gap0.25_mrr0": _tail(20, gamma=0.25, mrr_tol=0.0),
}


def pick(name: str, r: dict) -> str:
    rule = RULES[name]
    return rule(r["fit"]["stage2_curve"], r["stage2_real"]) if getattr(rule, "needs_real", False) else rule(r["fit"]["stage2_curve"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--metric", default="recall@30")
    p.add_argument("--rules", nargs="*", default=None, help="default: every rule the records support")
    p.add_argument("--base", default="floor0.9")
    p.add_argument("--by-pair", action="store_true", help="also list the chosen checkpoint per (pair, K, seed)")
    p.add_argument("--pool", action="store_true", help="pool K within each dataset and row kind")
    args = p.parse_args()

    recs = []
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if "stage2_real" in r and "reranker" not in r:
                recs.append(r)
    if args.rules is None:  # rules over tail averages need records from the round-2 harness
        has_tail = all("avg_20_50" in r["stage2_real"] for r in recs)
        args.rules = [k for k in RULES if has_tail or not (k.startswith("tail") or k in ("avg_20_50", "avg_30_50"))]
    groups = defaultdict(list)
    for r in recs:
        kind = "identity" if r["row"] == "1" else "bridge"
        groups[r["dataset"], kind, "all" if args.pool else r["K"]].append(r)

    rng = np.random.default_rng(0)
    print(f"Real ρ ({args.metric}; identity vs. native, bridge vs. ceiling) of the checkpoint each rule picks. "
          f"Δ vs. {args.base}, points, bootstrap over this table's (pair, seed) cells, which share queries (not a CI for new queries or corpora); swing = per-pair s.d. over seeds "
          f"(points, mean over pairs); regret = best checkpoint in hindsight − pick (points, mean / max).\n")
    for (ds, kind, K), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        best = np.array([max(v[args.metric] for k, v in r["stage2_real"].items() if not k.startswith("avg")) for r in rs])
        base = np.array([r["stage2_real"][pick(args.base, r)][args.metric] for r in rs])
        print(f"### {ds} · {kind} · K = {K} · {len(rs)} fits · best-checkpoint mean {best.mean():.3f}")
        print("| rule | mean | Δ vs. base | 95% CI | cells + / − | swing | regret mean / max | picks |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in args.rules:
            picks = [pick(name, r) for r in rs]
            v = np.array([r["stage2_real"][k][args.metric] for r, k in zip(rs, picks)])
            d = 100 * (v - base)
            boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            by_pair = defaultdict(list)
            for r, x in zip(rs, v):
                by_pair[r["src"], r["tgt"], r["K"]].append(x)
            swing = np.mean([100 * np.std(x, ddof=1) for x in by_pair.values() if len(x) > 1])
            regret = 100 * (best - v)
            counts = defaultdict(int)
            for k in picks:
                counts[k] += 1
            pick_str = " ".join(f"{k}:{n}" for k, n in sorted(counts.items(), key=lambda kv: (len(kv[0]), kv[0])))
            print(f"| {name} | {v.mean():.3f} | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(d > 0.05).sum()} / {(d < -0.05).sum()} | "
                  f"{swing:.2f} | {regret.mean():.2f} / {regret.max():.1f} | {pick_str} |")
        print()
        if args.by_pair:
            for r in sorted(rs, key=lambda r: (r["src"], r["tgt"], r["K"], r["seed"])):
                c = r["fit"]["stage2_curve"]
                real = " ".join(f"{k}:{v[args.metric]:.3f}" for k, v in r["stage2_real"].items())
                sig = " ".join(f"{s['step']}:{s['agree']:.3f}/{s['mrr']:.3f}" for s in c)
                print(f"  {r['src']} <- {r['tgt']} K={r['K']} s{r['seed']}  agree/mrr {sig}\n      real {real}")
            print()


if __name__ == "__main__":
    main()
