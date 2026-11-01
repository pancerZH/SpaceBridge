"""W7: does top-30 overlap between bridged and native results on real queries predict real retention?

Reads Row 8b / 7b records of the final engine (their per-query-confidence study holds overlap@30), fits retention on
overlap on the sets the signal was found on (MS MARCO, NQ), and judges it on sets it was not chosen on. rho_val is the
baseline. Exploratory until scale data is added: pass more JSONL files as arguments.

    uv run python scripts/overlap_study.py outputs/p0end/*.jsonl
"""
import json
import sys

import numpy as np

CHOSEN_ON = ("msmarco", "nq")


def load(paths: list[str]) -> list[dict]:
    seen = {}
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("row") not in ("8b",) or "confidence" not in r:
                continue
            key = (r["dataset"], r["src"], r["tgt"], r["K"], r["seed"], r.get("anchor_select", "kmeans"))
            q = r["confidence"]["quintiles"]
            seen[key] = {
                "dataset": r["dataset"], "corpus": r["dataset"].split("-")[0].split("@")[0], "K": r["K"], "pair": f"{r['src']}<-{r['tgt']}",
                "overlap": float(np.mean([b["overlap@30_native"] for b in q])), "rho": r["rho_ceil"]["recall@30"], "rho_native": r["rho_native"]["recall@30"],
                "rho_val": r["fit"].get("rho_val"), "score": (r.get("quality") or {}).get("score"), "n_q": r["n_queries"],
            }
    return list(seen.values())


def report(rows: list[dict], signal: str, train, test, target: str = "rho") -> str:
    tr, te = [r for r in rows if train(r)], [r for r in rows if test(r)]
    c = np.polyfit([r[signal] for r in tr], [r[target] for r in tr], 1)
    e = np.polyval(c, [r[signal] for r in te]) - np.array([r[target] for r in te])
    return f"RMSE {100 * np.sqrt((e ** 2).mean()):5.1f}  bias {100 * e.mean():+5.1f}  corr {np.corrcoef([r[signal] for r in te], [r[target] for r in te])[0, 1]:+.2f}  (fit {len(tr)}, judged {len(te)}; slope {c[0]:.3f}, intercept {c[1]:.3f})"


def main() -> None:
    rows = load(sys.argv[1:])
    corpora = sorted({r["dataset"] for r in rows})
    print(f"{len(rows)} fits on {len(corpora)} sets: " + ", ".join(f"{d} {sum(r['dataset'] == d for r in rows)}" for d in corpora))
    print("\nSet: mean overlap@30, retention R@30, rho_val, score")
    for d in corpora:
        s = [r for r in rows if r["dataset"] == d]
        print(f"  {d:<16} {np.mean([r['overlap'] for r in s]):.3f}  {np.mean([r['rho'] for r in s]):.3f}  {np.mean([r['rho_val'] for r in s]):.3f}  {np.mean([r['score'] or np.nan for r in s]):.1f}")
    chosen = lambda r: r["corpus"] in CHOSEN_ON
    for signal in ("overlap", "rho_val"):
        print(f"\n{signal}: all fits corr {np.corrcoef([r[signal] for r in rows], [r['rho'] for r in rows])[0, 1]:+.3f}")
        print(f"  fit on MS MARCO + NQ, judged on the rest:  {report(rows, signal, chosen, lambda r: not chosen(r))}")
        for d in sorted({r["corpus"] for r in rows}):
            print(f"  leave {d:<14} out: {report(rows, signal, lambda r: r['corpus'] != d, lambda r: r['corpus'] == d)}")


if __name__ == "__main__":
    main()
