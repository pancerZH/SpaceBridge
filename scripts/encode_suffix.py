"""Encodes a nested corpus reusing the file of its prefix, so the first rows aren't encoded twice. On the GPU.

1. Checks that the corpus's first rows are the prefix corpus's documents, in order.
2. Encodes the rest as their own input list, so ``Encoder`` keeps them resumable.
3. Checks the prefix file against 1,000 of its rows encoded again: cosine must be above 0.9999.
4. Writes the file ``Encoder`` expects for the whole list, atomically. The suffix file stays, so a rerun costs nothing.

A rerun continues: an encode in progress resumes from its last part, and a finished file is skipped.

    uv run python -u scripts/encode_suffix.py hotpotqa-full hotpotqa-1m-nested bge-base-en bge-large-en qwen3-0.6b
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder, concat_embeddings
from spacebridge_bench.models import MODELS

CHECK_ROWS = 1_000
MIN_COSINE = 0.9999


def main() -> None:
    full, prefix = load_beir(sys.argv[1]), load_beir(sys.argv[2])
    n = prefix.N
    if full.doc_ids[:n] != prefix.doc_ids or full.titles[:n] != prefix.titles or full.texts[:n] != prefix.texts:
        raise ValueError(f"the first {n:,} rows of {full.name} aren't {prefix.name}")
    if full.queries != prefix.queries:
        raise ValueError("the two datasets have different queries")
    durable.progress(f"{full.name}: {full.N:,} docs, {len(full.queries):,} queries; its first {n:,} rows are {prefix.name}")
    suffix_titles, suffix_texts = full.titles[n:], full.texts[n:]
    for key in sys.argv[3:]:
        enc = Encoder(MODELS[key], "cuda")
        inputs = [enc.spec.format_doc(a, b) for a, b in zip(full.titles, full.texts)]
        target = enc.cache_dir / f"{enc._key(inputs)}.npy"
        if target.exists():
            durable.progress(f"{full.name} {key}: already at {target}")
            continue
        prefix_path = enc.cache_dir / f"{enc._key(inputs[:n])}.npy"
        if not prefix_path.exists():
            raise FileNotFoundError(f"{prefix.name} {key}: no embeddings at {prefix_path}")
        t0 = time.perf_counter()
        suffix = enc.docs_store(suffix_titles, suffix_texts)
        durable.progress(f"{full.name} {key}: suffix {suffix.shape} in {(time.perf_counter() - t0) / 60:.1f} min")
        rows = np.sort(np.random.default_rng(0).choice(n, CHECK_ROWS, replace=False))
        fresh = enc.encode([inputs[i] for i in rows], cache=False)
        enc.unload()
        stored = np.asarray(np.load(prefix_path, mmap_mode="r")[rows], dtype=np.float32)
        cos = (fresh * (stored / np.linalg.norm(stored, axis=1, keepdims=True))).sum(1)
        check = {"model": key, "rows": CHECK_ROWS, "cos_min": float(cos.min()), "cos_mean": float(cos.mean())}
        durable.progress(f"{full.name} {key}: prefix check {json.dumps(check)}")
        if cos.min() <= MIN_COSINE:
            raise ValueError(f"{prefix_path} doesn't match a fresh encode: {check}")
        concat_embeddings([prefix_path, suffix.path], target)
        back = np.load(target, mmap_mode="r")
        if back.shape != (full.N, suffix.shape[1]):
            raise ValueError(f"{target}: shape {back.shape}")
        durable.progress(f"{full.name} {key}: {back.shape} at {target}")
        queries = enc.queries(full.queries, full.task)
        enc.unload()
        durable.progress(f"{full.name} {key}: queries {queries.shape}")


if __name__ == "__main__":
    main()
