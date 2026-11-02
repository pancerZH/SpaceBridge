"""Writes a dataset's embedding file from another dataset's file holding the same documents in another order, with no
encoding. The vectors stay exactly the same, so results on the two orders compare without re-encoding noise.

    uv run python -u scripts/reorder_embeddings.py msmarco-1m msmarco-1m-nested bge-base-en bge-large-en qwen3-0.6b
"""
import sys

import numpy as np

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.models import MODELS


def doc_file(enc: Encoder, ds):
    return enc.cache_dir / f"{enc._key([enc.spec.format_doc(a, b) for a, b in zip(ds.titles, ds.texts)])}.npy"


def main() -> None:
    source, target = load_beir(sys.argv[1]), load_beir(sys.argv[2])
    if source.queries != target.queries:
        raise ValueError("the two datasets have different queries")
    row = {d: i for i, d in enumerate(source.doc_ids)}
    if len(row) != len(target.doc_ids) or set(row) != set(target.doc_ids):
        raise ValueError("the two datasets hold different documents")
    order = np.array([row[d] for d in target.doc_ids])
    for key in sys.argv[3:]:
        enc = Encoder(MODELS[key], "cpu")
        have, want = doc_file(enc, source), doc_file(enc, target)
        if want.exists():
            durable.progress(f"{target.name} {key}: already at {want}")
            continue
        if not have.exists():
            raise FileNotFoundError(f"{source.name} {key}: no embeddings at {have}")
        vectors = np.load(have, mmap_mode="r")
        with durable.atomic_file(want) as f:
            np.save(f, np.asarray(vectors[order]))
        durable.progress(f"{target.name} {key}: {vectors.shape} written to {want}")


if __name__ == "__main__":
    main()
