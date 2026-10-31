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

# Size class (B params) -> end-to-end mount budget, s (design §2.1, set from M3 measurements: 128-token
# anchor docs for doc pairs up to 0.6B, no doc pairs for 4B-class query models).
BUDGET_S = ((0.35, 5.0), (0.6, 10.0), (4.0, 30.0))


def budget(params_b: float) -> float:
    return next(b for p, b in BUDGET_S if params_b <= p)


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
                cells[r["tgt"], r["K"]].append((r["mount"]["encode_ms"], r["mount"].get("doc_encode_ms", np.nan), solver))
            elif r["row"] == "8":  # the mount without doc pairs (4B-class protocol) runs Row 8's solver
                solver8[r["tgt"], r["K"]].append(solver)
    print("| query model (params) | K | runs | proxies encode s | anchor docs encode s | solver ms (8 / 8b) | "
          "mount s (no doc pairs) | mount s (doc pairs) | budget s | protocol | within budget |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (tgt, K), v in sorted(cells.items(), key=lambda kv: (MODELS[kv[0][0]].params_b, kv[0])):
        v = np.array(v)
        enc, doc, solver = v.mean(0)
        s8 = np.mean(solver8[tgt, K]) if solver8[tgt, K] else solver
        spec = MODELS[tgt]
        no_pairs, pairs = (enc + s8) / 1e3, (enc + doc + solver) / 1e3
        # Mount protocol (design §2.1): doc pairs up to ~0.6B query models, proxies only for the 4B class.
        protocol = "proxies only" if spec.params_b > 0.6 else "doc pairs"
        mount = no_pairs if spec.params_b > 0.6 else pairs
        b = budget(spec.params_b)
        print(f"| {tgt} ({spec.params_b:g}B) | {K} | {len(v)} | {enc / 1e3:.1f} | {doc / 1e3:.1f} | {s8:.0f} / {solver:.0f} | "
              f"{no_pairs:.1f} | {pairs:.1f} | {b:g} | {protocol} | {'✓' if mount < b else '✗'} {mount:.1f} s |")

if __name__ == "__main__":
    main()
