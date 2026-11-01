"""Encodes a dataset's documents and queries with the given models, one at a time (crash-safe: see ``Encoder``).

    uv run python -u scripts/encode_corpus.py msmarco-full bge-base-en bge-large-en qwen3-0.6b
"""
import sys

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.models import MODELS

ds = load_beir(sys.argv[1])
durable.progress(f"{ds.name}: {ds.N:,} docs, {len(ds.queries):,} queries")
for key in sys.argv[2:]:
    enc = Encoder(MODELS[key], "cuda")
    store = enc.docs_store(ds.titles, ds.texts)
    enc.queries(ds.queries, ds.task)
    enc.unload()
    durable.progress(f"{ds.name} {key}: {store.shape} at {store.path}")
