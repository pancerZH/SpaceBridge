"""Mount time per target model (design §2.1 budgets): proxy encoding, anchor-doc encoding (doc pairs), solver.

Uses the default-engine rows (8b / 7b) of spacebridge-bench JSONL files. Timings are only meaningful for
runs that had the GPU to themselves.

    uv run python scripts/summarize_mount.py outputs/m3_fiqa_tier4a.jsonl outputs/m2e_fiqa_seeds.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from spacebridge_bench.models import MODELS

# Mount budget (design §2.1): encoding work relative to re-indexing the corpus with the same model, in tokens, so it
# holds on any hardware. Seconds are reference measurements of the machine the run used, not a pass / fail.
BUDGET_RATIO = 0.10


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    args = p.parse_args()
    cells, solver8 = defaultdict(list), defaultdict(list)
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if "mount" not in r or "reranker" in r:
                continue
            f = r["fit"]
            solver = f["stage1_ms"] + f.get("stage2_ms", 0) + f["val_ms"]
            if r["row"] in ("8b", "7b"):
                m = r["mount"]
                cells[r["dataset"], r["tgt"], r["K"]].append((
                    m["encode_ms"], m.get("doc_encode_ms", np.nan), solver, m["n_tokens"], m.get("doc_n_tokens", np.nan),
                    m.get("corpus_tokens_est", np.nan),
                ))
            elif r["row"] == "8":  # the mount without doc pairs (4B-class protocol) runs Row 8's solver
                solver8[r["dataset"], r["tgt"], r["K"]].append(solver)
    print("| dataset | query model (params) | K | runs | proxies s | anchor docs s | solver ms (8 / 8b) | mount s, this machine | "
          "protocol | mount / re-index tokens | within budget |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (ds, tgt, K), v in sorted(cells.items(), key=lambda kv: (kv[0][0], MODELS[kv[0][1]].params_b, kv[0])):
        enc, doc, solver, tok, doc_tok, corpus = np.array(v).mean(0)
        s8 = np.mean(solver8[ds, tgt, K]) if solver8[ds, tgt, K] else solver
        spec = MODELS[tgt]
        # Mount protocol (design §2.1): doc pairs up to ~0.6B query models, proxies only for the 4B class.
        proxies_only = spec.params_b > 0.6
        mount = (enc + s8) / 1e3 if proxies_only else (enc + doc + solver) / 1e3
        ratio = (tok if proxies_only else tok + doc_tok) / corpus
        status = "n/a (older record)" if np.isnan(ratio) else f"{'✓' if ratio <= BUDGET_RATIO else '✗'}"
        print(f"| {ds} | {tgt} ({spec.params_b:g}B) | {K} | {len(v)} | {enc / 1e3:.1f} | {doc / 1e3:.1f} | {s8:.0f} / {solver:.0f} | "
              f"{mount:.1f} | {'proxies only' if proxies_only else 'doc pairs'} | {ratio:.2%} | {status} |")


if __name__ == "__main__":
    main()
