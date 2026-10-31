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
    cells = defaultdict(list)
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if r["row"] in ("8b", "7b") and "mount" in r and "reranker" not in r:
                f = r["fit"]
                solver = f["stage1_ms"] + f.get("stage2_ms", 0) + f["val_ms"]
                cells[r["tgt"], r["K"]].append((r["mount"]["encode_ms"], r["mount"].get("doc_encode_ms", np.nan), solver))
    print("| query model (params) | K | runs | proxies encode s | anchor docs encode s | solver ms | mount s (no doc pairs) | "
          "mount s (default, doc pairs) | budget s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (tgt, K), v in sorted(cells.items(), key=lambda kv: (MODELS[kv[0][0]].params_b, kv[0])):
        v = np.array(v)
        enc, doc, solver = v.mean(0)
        spec = MODELS[tgt]
        print(f"| {tgt} ({spec.params_b:g}B) | {K} | {len(v)} | {enc / 1e3:.1f} | {doc / 1e3:.1f} | {solver:.0f} | "
              f"{(enc + solver) / 1e3:.1f} | {(enc + doc + solver) / 1e3:.1f} | {budget(spec.params_b):g} |")


if __name__ == "__main__":
    main()
