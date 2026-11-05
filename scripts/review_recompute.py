"""The published cells again, with the same configuration, keeping per-query scores; then a comparison with the records they came from.

    python scripts/review_recompute.py --print-commands    the commands, one per record file, for scripts/review_experiments.sh
    python scripts/review_recompute.py --compare           each metric against the published record
"""
import argparse
import json
from pathlib import Path

ROOT = Path("outputs/phase1")
OUT = Path("outputs/review")
# Every file behind a published retention row. K follows the library's default in all of them.
FILES = """closeout/b1_scifact closeout/b1_arguana followup/scidocs m3_a6/fiqa m3_a6/trec_covid
closeout/b1_ms10000 closeout/b1_ms20000 closeout/b1_ms30000 closeout/b1_ms50000 closeout/b1_nq10000 closeout/b1_nq20000 closeout/b1_nq30000 closeout/b1_nq50000
followup/hp10000 followup/hp20000 followup/hp30000 followup/hp50000
closeout/b2_ms100000 closeout/b2_nq100000 closeout/b2_hp100000 closeout/b2_ms150000 closeout/b2_nq150000 closeout/b2_hp150000
final/ms160000 final/nq160000 final/hp160000 final/ms160001 final/nq160001 final/hp160001
m3_a6/ms200000 m3_a6/nq200000 m3_a6/hp200000 m3_a6/ms500000 m3_a6/nq500000 m3_a6/hp500000
m3_1m/ms m3_1m/nq2k m3_1m/hp closeout/b4_ms2000000 closeout/b4_hp2000000 closeout/b4_ms4000000 m3_full/hp m3_full/ms_t1 m3_full/ms_t3""".split()


def records(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "row" in r:
                out.append(r)
    return out


def name_of(file: str) -> str:
    return file.replace("/", "_")


def print_commands() -> None:
    for file in FILES:
        rs = records(ROOT / f"{file}.jsonl")
        dataset, _, max_docs = rs[0]["dataset"].partition("@")
        pairs = sorted({f"{r['src']}:{r['tgt']}" for r in rs})
        seeds = sorted({r["seed"] for r in rs})
        rows = sorted({r["row"] for r in rs})
        parts = ["--dataset", dataset, "--pair", *pairs, "--rows", *rows, "--K", "0", "--seed", *map(str, seeds)]
        if max_docs:
            parts += ["--max-docs", max_docs]
        n_queries = rs[0]["n_queries"]
        if dataset in ("nq-1m", "nq-1m-nested") or n_queries == 500:  # a file's first queries: the others load all, or a sample of 2,000
            parts += ["--max-queries", str(n_queries)]
        print(name_of(file), " ".join(parts))


def compare() -> None:
    worst, total, missing = 0.0, 0, []
    print("| Record file | Metrics compared | Largest difference |\n| --- | --- | --- |")
    for file in FILES:
        new_path = OUT / f"recompute_{name_of(file)}.jsonl"
        if not new_path.exists():
            missing.append(file)
            continue
        old = {(r["src"], r["tgt"], r["seed"], r["row"]): r for r in records(ROOT / f"{file}.jsonl")}
        diffs = []
        for r in records(new_path):
            o = old.get((r["src"], r["tgt"], r["seed"], r["row"]))
            if o is None or o.get("reranker") != r.get("reranker"):
                missing.append(f"{file}: {r['src']} <- {r['tgt']} seed {r['seed']} row {r['row']}")
                continue
            for m, v in r["metrics"].items():
                if m in o["metrics"]:
                    diffs.append(abs(v - o["metrics"][m]))
            if "baselines_lexical" in r and "baselines_lexical" in o:
                diffs.append(abs(r["baselines_lexical"]["rho_ceil"]["ndcg@5"] - o["baselines_lexical"]["rho_ceil"]["ndcg@5"]))
        total += len(diffs)
        worst = max(worst, max(diffs, default=0.0))
        print(f"| {file} | {len(diffs):,} | {max(diffs, default=0.0):.6f} |")
    print(f"\n{total:,} values compared. Largest difference {worst:.6f}. Not compared: {missing or 'none'}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--print-commands", action="store_true")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    print_commands() if a.print_commands else compare()
