"""W5: wall-clock of one query through the shipped ``Searcher``, by stage. Run it on an idle machine.

``fit`` saves a bridge. ``run`` then times real queries with their real candidates, on the device given: use
``CUDA_VISIBLE_DEVICES=`` for CPU numbers. One JSON line per timed configuration goes to --out; a rerun skips it. The first
query of a configuration only warms up and is not counted. The time of a stage is its median and 95th percentile over queries.
Each record also has the candidates' mean tokens per (query, document) pair and per document, truncated at the reranker's
limit, counted between queries so they cost no timed work.

    uv run python -u scripts/query_timing.py fit msmarco-full --index-model bge-large-en --query-model qwen3-0.6b --max-docs 1000000 --device cuda --bridge b/
    CUDA_VISIBLE_DEVICES= uv run --extra hnsw python -u scripts/query_timing.py run msmarco-full --index-model bge-large-en \
        --query-model qwen3-0.6b --max-docs 1000000 --bridge b/ --pool 300 --backend exact --device cpu --threads 10 --n 20 --out t.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spacebridge import CrossEncoderReranker, ExactBackend, HnswBackend, Index, LexicalIndex, Searcher, SpaceBridge, freeze, mount
from spacebridge.encoders import CPU_BATCH_TOKENS, predict_pairs, token_lengths
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.models import MODELS
from spacebridge_bench.rerank import OnnxCrossEncoder
from spacebridge_bench.run import _first_docs

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


def load(args):
    ds = load_beir(args.dataset)
    if args.max_docs is not None:
        ds = _first_docs(ds, args.max_docs)
    enc = Encoder(MODELS[args.index_model], "cuda")
    store = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    return ds, store


class LiveQueryEncoder:
    """Encodes one query with no cache, as serving does."""

    size_class = "small"

    def __init__(self, key: str, task: str, device: str):
        self.spec, self.task = MODELS[key], task
        self.encoder = Encoder(self.spec, device)
        self.encoder.model  # loaded now, not during the first timed query

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode([self.spec.format_query(t, self.task) for t in texts], cache=False)


class OnnxReranker:
    """``--reranker onnx:PATH``: the reference reranker through an ONNX file, on the CPU."""

    def __init__(self, path: str, threads: int):
        self.max_length = 512
        self.model = OnnxCrossEncoder(path, DEFAULT_RERANKER, self.max_length, threads)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return predict_pairs(self.model, [query] * len(docs), docs, CPU_BATCH_TOKENS, self.max_length)


class RememberingReranker:
    """Passes scoring through, keeping the last candidates so their tokens can be counted outside the timed path."""

    def __init__(self, inner: CrossEncoderReranker):
        self.inner, self.last = inner, None

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        self.last = (query, docs)
        return self.inner.score(query, docs)

    def count_tokens(self) -> tuple[float, float]:
        query, docs = self.last
        tok, limit = self.inner.model.tokenizer, self.inner.max_length
        return float(token_lengths(tok, [query] * len(docs), docs, max_length=limit).mean()), float(token_lengths(tok, docs, max_length=limit).mean())


def percentile_table(rows: list[dict]) -> dict:
    return {k: {"p50": float(np.percentile([r[k] for r in rows], 50)), "p95": float(np.percentile([r[k] for r in rows], 95))} for k in rows[0]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("fit", "run"))
    p.add_argument("dataset")
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--query-model", default="qwen3-0.6b")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", type=int, nargs="*", default=[300])
    p.add_argument("--backend", default="exact", help="exact, or the path of a saved HNSW index")
    p.add_argument("--ef", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--search-device", default=None, help="where exact search runs, if not --device: an index too large for the GPU, with the models on it")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--no-rerank", action="store_true", help="time encode, project, search and fetch only")
    p.add_argument("--reranker", default=DEFAULT_RERANKER, help="Hugging Face id of the cross-encoder, or onnx:PATH (needs --extra onnx)")
    p.add_argument("--lexical", type=Path, default=None, help="a LexicalIndex to serve with: BM25 in the pool and in the breakdown (default: BM25 off)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ds, store = load(args)
    if args.mode == "fit":
        index = Index(store, ds.texts, ds.titles)
        bundle = freeze(index, MountEncoder(MODELS[args.index_model], ds.task, args.device), ds.lang, args.seed, device=args.device)
        mount(bundle, index, MountEncoder(MODELS[args.query_model], ds.task, args.device), device=args.device).save(args.bridge)
        durable.progress(f"bridge saved to {args.bridge}")
        return

    torch.set_num_threads(args.threads)
    done = {r["key"] for r in durable.read_jsonl(args.out)[0]} if args.out and args.out.exists() else set()
    bridge = SpaceBridge.load(args.bridge, device="cpu")
    index = Index(store, ds.texts, ds.titles)
    t0 = time.perf_counter()
    if args.backend == "exact":
        backend = ExactBackend(store[np.arange(len(store))] if len(store) < 3_000_000 else _load_all(store), device=args.search_device or args.device)
    else:
        import faiss

        faiss.omp_set_num_threads(args.threads)
        backend = HnswBackend(args.backend, ef_search=args.ef)
    load_s = time.perf_counter() - t0
    encoder = LiveQueryEncoder(args.query_model, ds.task, args.device)
    reranker = None
    if not args.no_rerank:
        onnx = args.reranker.startswith("onnx:")
        reranker = RememberingReranker(OnnxReranker(args.reranker[5:], args.threads) if onnx else CrossEncoderReranker(args.reranker, device=args.device))
    lexical = LexicalIndex.load(args.lexical) if args.lexical else None
    searcher = Searcher(bridge, encoder, backend, index, reranker, device=args.device, allow_no_reranker=args.no_rerank, lexical=lexical)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    queries = ds.queries[: args.n + 1]
    for pool in args.pool:
        key = f"{ds.name}|{args.backend}|{args.device}|t{args.threads}|pool{pool}|ef{args.ef}" + ("|norerank" if args.no_rerank else "") + (f"|{args.reranker}" if args.reranker != DEFAULT_RERANKER else "") + ("|lexical" if lexical else "") + (f"|search-{args.search_device}" if args.search_device else "")
        if key in done:
            continue
        searcher.search(queries[0], pool=pool)  # warmup
        rows, tokens = [], []
        for q in queries[1:]:
            out = searcher.search(q, pool=pool)
            rows.append(out.timing_ms)
            if reranker is not None:
                tokens.append(reranker.count_tokens())
        rec = {"key": key, "dataset": ds.name, "N": ds.N, "backend": args.backend, "ef": args.ef, "device": args.device, "threads": args.threads, "pool": pool, "n": len(rows), "rerank": not args.no_rerank, "reranker": args.reranker,
               "stages_ms": percentile_table(rows), "load_s": load_s, "lexical": lexical is not None, "search_device": args.search_device or args.device, "index_model": args.index_model, "query_model": args.query_model}
        if tokens:
            rec["tokens_per_pair"], rec["tokens_per_doc"] = (float(v) for v in np.mean(tokens, axis=0))
        if args.device.startswith("cuda"):
            rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if args.out:
            durable.append_lines(args.out, [json.dumps(rec)])
        durable.progress(f"{key}: total p50 {rec['stages_ms']['total']['p50'] / 1e3:.2f} s, p95 {rec['stages_ms']['total']['p95'] / 1e3:.2f} s")


def _load_all(store) -> np.ndarray:
    out = np.empty(store.shape, dtype=np.float32)
    for s in range(0, len(store), 500_000):
        out[s : s + 500_000] = store[np.arange(s, min(s + 500_000, len(store)))]
    return out


if __name__ == "__main__":
    main()
