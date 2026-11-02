"""How many anchors are test positives, by corpus and seed. A non-nested corpus doesn't keep positives out of its anchors (run_pair
does on nested ones), so this shows how much that could matter: K = 8,192 random anchors, as the large profile draws them.

    uv run python scripts/anchor_positive_overlap.py hotpotqa-1m msmarco-1m nq-1m
"""
import sys

import numpy as np

from spacebridge import random_anchors
from spacebridge_bench.datasets import load_beir

for name in sys.argv[1:]:
    ds = load_beir(name)
    pos = np.array(sorted({d for rel in ds.qrels for d in rel}))
    per_q = np.mean([len(rel) for rel in ds.qrels])
    allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
    hits = [int(np.isin(allowed[random_anchors(len(allowed), 8192, s)], pos).sum()) for s in (0, 1, 2)]
    print(f"{name}: {ds.N:,} docs, {len(ds.queries):,} queries, {per_q:.2f} positives per query, {len(pos):,} positives ({100 * len(pos) / ds.N:.2f}% of the index); positives among 8,192 anchors by seed {hits}")
