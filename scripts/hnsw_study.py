"""W5 step 1: does a real index cost the bridge more than it costs native queries?

Builds a FAISS HNSW index over one model's stored vectors (crash-safe, resumable), then compares HNSW with exact search
for native and bridged queries: recall of the exact top-pool, and end-to-end nDCG@5 through the reranker. ``--ivfpq`` adds
one quantized index on retention only, with anchors read from the index's own lossy codes. One JSON line per unit of work
goes to --out as it finishes, and a rerun skips what is there.

    uv run --extra hnsw python -u scripts/hnsw_study.py msmarco-1m --index-model bge-large-en \
        --query-models bge-base-en qwen3-0.6b --out outputs/phase1/w5_ann/ms1m.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from spacebridge import Index, build_hnsw, freeze, mount
from spacebridge.serving import save_faiss_index
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.rerank import RERANKERS, Reranker, rerank
from spacebridge_bench.run import _DocTexts, _first_docs, _first_queries
from spacebridge_bench.search import cached_search, search

HNSW_CACHE = Path("data/cache/hnsw")
POOLS = (60, 100, 300)
EFS = (300, 500, 1000)


class Results:
    """Append-only records keyed by a string; a rerun skips keys already on disk."""

    def __init__(self, path: Path):
        self.path = path
        self.done = {}
        if path.exists():
            records, _ = durable.read_jsonl(path)
            self.done = {r["key"]: r for r in records}

    def add(self, key: str, **record) -> None:
        self.done[key] = {"key": key, **record}
        durable.append_lines(self.path, [json.dumps(self.done[key])])
        durable.progress(f"{key}: {json.dumps({k: v for k, v in record.items() if not isinstance(v, dict)})[:200]}")


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def rankings_report(res: Results, tag: str, approx: np.ndarray, exact: np.ndarray, ds, queries, rr, pools) -> None:
    """Recall of the exact top-pool, and absolute end-to-end nDCG@5, for one approximate ranking against the exact one."""
    docs = _DocTexts(ds.titles, ds.texts)
    for pool in pools:
        key = f"{tag}|pool{pool}"
        if key in res.done:
            continue
        pq_a = per_query(rerank(rr, approx, pool, queries, docs, ds.task), ds.qrels)
        pq_e = per_query(rerank(rr, exact, pool, queries, docs, ds.task), ds.qrels)
        first_a, first_e = per_query(approx[:, :pool], ds.qrels), per_query(exact[:, :pool], ds.qrels)
        res.add(
            key, tag=tag, pool=pool, recall_of_exact=overlap(approx[:, :pool], exact[:, :pool]),
            ndcg5_approx=float(pq_a["ndcg@5"].mean()), ndcg5_exact=float(pq_e["ndcg@5"].mean()),
            recall_pool_approx=float(first_a.get(f"recall@{pool}", first_a["recall@100"]).mean()),
            recall_pool_exact=float(first_e.get(f"recall@{pool}", first_e["recall@100"]).mean()),
        )


def hnsw_search(index, Q: np.ndarray, k: int, ef: int) -> np.ndarray:
    index.hnsw.efSearch = ef
    return index.search(np.asarray(Q, dtype=np.float32), k)[1]


class QuantizedIndex:
    """An ``IndexReader`` whose vectors are the quantized index's own reconstructions, as a deployment holding only codes would see."""

    def __init__(self, index, ds, dim: int):
        self.index, self.ds, self.dim = index, ds, dim

    def __len__(self) -> int:
        return self.index.ntotal

    def vectors(self, ids: np.ndarray) -> np.ndarray:
        out = np.vstack([self.index.reconstruct(int(i)) for i in ids]).astype(np.float32)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def texts(self, ids: np.ndarray) -> list[str]:
        return [self.ds.texts[i] for i in ids]

    def titles(self, ids: np.ndarray) -> list[str]:
        return [self.ds.titles[i] for i in ids]


def build_ivfpq(store, path: Path, factory: str):
    if path.exists():
        return faiss.read_index(str(path))
    index = faiss.index_factory(store.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
    sample = np.sort(np.random.default_rng(0).choice(len(store), min(400_000, len(store)), replace=False))
    durable.progress(f"ivfpq: training {factory} on {len(sample):,} rows")
    index.train(store[sample])
    for s in range(0, len(store), 200_000):
        index.add(store[s : s + 200_000])
        durable.progress(f"ivfpq: added {index.ntotal:,} / {len(store):,}")
    save_faiss_index(index, path)
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--query-models", nargs="+", required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--K", type=int, default=0, help="anchors (0: the index size's default)")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ivfpq", default=None, help="also test this FAISS factory string, e.g. IVF16384,PQ128")
    p.add_argument("--nprobe", type=int, nargs="+", default=[64, 256])
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir(args.dataset)
    if args.max_docs is not None:
        ds = _first_docs(ds, args.max_docs)
    if args.max_queries is not None:
        ds = _first_queries(ds, args.max_queries)
    res = Results(args.out)
    src = MODELS[args.index_model]
    enc = Encoder(src, args.device)
    store = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q_native = enc.queries(ds.index_lang_queries, ds.task)
    enc.unload()

    path = HNSW_CACHE / f"{ds.name}_{store.path.stem}_{store.dim}d_M{args.M}_efc{args.ef_construction}.faiss"
    t0 = time.perf_counter()
    was_complete = "build" in res.done
    index = build_hnsw(
        store, path, args.M, args.ef_construction,
        on_batch=lambda n, total: durable.progress(f"hnsw build: {n:,} / {total:,}, {time.perf_counter() - t0:.0f} s this run"),
    )
    if not was_complete:
        res.add("build", path=str(path), build_s_this_run=time.perf_counter() - t0, file_bytes=path.stat().st_size, ntotal=index.ntotal,
                threads=faiss.omp_get_max_threads(), M=args.M, ef_construction=args.ef_construction)
    if args.build_only:
        return

    rr = Reranker(RERANKERS["bge-reranker-v2-m3"], args.device)
    maxk = max(POOLS)
    exact_native = cached_search(Q_native, store, 1000, args.device, exclude=ds.self_rows)
    for ef in EFS:
        ranked = hnsw_search(index, Q_native, maxk, ef)
        rankings_report(res, f"hnsw|native|ef{ef}", ranked, exact_native, ds, ds.index_lang_queries, rr, POOLS)
    rr.unload()
    pool_index = Index(store, ds.texts, ds.titles)
    pq_index = None
    if args.ivfpq:
        pq_index = build_ivfpq(store, HNSW_CACHE / f"{ds.name}_{store.path.stem}_{store.dim}d_{args.ivfpq.replace(',', '_')}.faiss", args.ivfpq)
        faiss.extract_index_ivf(pq_index).make_direct_map()
        for nprobe in args.nprobe:
            faiss.extract_index_ivf(pq_index).nprobe = nprobe
            ranked = pq_index.search(np.asarray(Q_native, dtype=np.float32), maxk)[1]
            rankings_report(res, f"{args.ivfpq}|native|nprobe{nprobe}", ranked, exact_native, ds, ds.index_lang_queries, rr, POOLS)
        rr.unload()

    for tgt_key in args.query_models:
        enc = Encoder(MODELS[tgt_key], args.device)
        Q_tgt = enc.queries(ds.queries, ds.task)
        E_tgt = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
        enc.unload()
        oracle = cached_search(Q_tgt, E_tgt, 1000, args.device, exclude=ds.self_rows)
        docs = _DocTexts(ds.titles, ds.texts)
        for pool in POOLS:
            key = f"oracle|{tgt_key}|pool{pool}"
            if key not in res.done:
                pq = per_query(rerank(rr, oracle, pool, ds.queries, docs, ds.task), ds.qrels)
                res.add(key, tag="oracle", tgt=tgt_key, pool=pool, ndcg5_exact=float(pq["ndcg@5"].mean()))
        rr.unload()
        for seed in args.seed:
            if all(f"hnsw|{tgt_key}|s{seed}|ef{ef}|pool{pool}" in res.done for ef in EFS for pool in POOLS) and (
                not args.ivfpq or all(f"{args.ivfpq}|{tgt_key}|s{seed}|nprobe{n}|pool{pool}" in res.done for n in args.nprobe for pool in POOLS)
            ):
                continue
            bundle = freeze(pool_index, MountEncoder(src, ds.task, args.device), ds.lang, seed, args.K or None, device=args.device)
            bridge = mount(bundle, pool_index, MountEncoder(MODELS[tgt_key], ds.task, args.device), device=args.device)
            Q_hat = bridge.transform_query(Q_tgt)
            exact = search(Q_hat, store, 1000, args.device, exclude=ds.self_rows)
            for ef in EFS:
                ranked = hnsw_search(index, Q_hat, maxk, ef)
                rankings_report(res, f"hnsw|{tgt_key}|s{seed}|ef{ef}", ranked, exact, ds, ds.queries, rr, POOLS)
            if pq_index is not None:
                reader = QuantizedIndex(pq_index, ds, store.dim)
                pq_bundle = freeze(reader, MountEncoder(src, ds.task, args.device), ds.lang, seed, args.K or None, device=args.device)
                pq_bridge = mount(pq_bundle, reader, MountEncoder(MODELS[tgt_key], ds.task, args.device), device=args.device)
                Q_hat_pq = pq_bridge.transform_query(Q_tgt)
                for nprobe in args.nprobe:
                    faiss.extract_index_ivf(pq_index).nprobe = nprobe
                    ranked = pq_index.search(np.asarray(Q_hat_pq, dtype=np.float32), maxk)[1]
                    rankings_report(res, f"{args.ivfpq}|{tgt_key}|s{seed}|nprobe{nprobe}", ranked, exact, ds, ds.queries, rr, POOLS)
            rr.unload()


if __name__ == "__main__":
    main()
