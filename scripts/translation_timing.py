"""W6: wall-clock to translate one query, for the translation baseline's cost per query. Run it on an idle machine.

Use ``CUDA_VISIBLE_DEVICES=`` for CPU numbers. The first query only warms up and is not counted.

    uv run --group translate python -u scripts/translation_timing.py xquad-zh --translator madlad --device cuda --out t.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from translation_baseline import Translator

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--translator", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    key = f"{args.dataset}|{args.translator}|{args.device}|t{args.threads}"
    if args.out.exists() and key in {r["key"] for r in durable.read_jsonl(args.out)[0]}:
        durable.progress(f"{key}: already recorded")
        return
    torch.set_num_threads(args.threads)
    queries = load_beir(args.dataset).queries[: args.n + 1]
    t0 = time.perf_counter()
    tr = Translator(args.translator, args.device)
    load_s = time.perf_counter() - t0
    tr.translate(queries[:1])
    times = []
    for q in queries[1:]:
        t0 = time.perf_counter()
        tr.translate([q])
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
    rec = {"key": key, "dataset": args.dataset, "translator": args.translator, "device": args.device, "threads": args.threads, "n": len(times), "load_s": load_s,
           "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}
    if args.device.startswith("cuda"):
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    durable.append_lines(args.out, [json.dumps(rec)])
    durable.progress(f"{key}: p50 {rec['p50_ms']:.0f} ms, p95 {rec['p95_ms']:.0f} ms")


if __name__ == "__main__":
    main()
