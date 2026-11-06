"""Retention vs. index size (W3): one map per (pair, K, seed), evaluated on every prefix of a nested corpus in one pass.

Anchors and distractors come from the smallest size, positives excluded, so the anchors are in the index at every size
and a larger index only adds documents. One line per (cell, size) goes to --out as it finishes; a rerun skips finished
cells. The first-stage rows are Row 8b, through ``freeze`` and ``mount``.

    uv run python -u scripts/scale_curve.py msmarco-full --pair bge-large-en:bge-base-en --K 8192 --seed 0 1 2 --out x.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from spacebridge import Index, freeze, mount
from spacebridge.anchors import _truncate_tokens
from spacebridge_bench import durable
from spacebridge_bench.datasets import DATASETS, load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.metrics import bootstrap_retention_ci, per_query, retention
from spacebridge_bench.provenance import check_or_write_config, implementation_digest, query_digest, revisions
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import _first_queries
from spacebridge_bench.search import cached_search_nested, search_nested

SIZES = (200_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_841_823)


def finished(path: Path) -> set[str]:
    if not path.exists():
        return set()
    records, clean = durable.read_jsonl(path)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["cell"]] = counts.get(r["cell"], 0) + 1
    done = {c for c, n in counts.items() if n == next(r["cell_records"] for r in records if r["cell"] == c)}
    kept = [r for r in records if r["cell"] in done]
    if not clean or len(kept) != len(records):
        with durable.atomic_file(path) as f:
            f.write("".join(json.dumps(r) + "\n" for r in kept).encode())
    return done


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    return np.array([len(set(x[:k].tolist()) & set(y[:k].tolist())) / k for x, y in zip(a, b)])


def proxy_agreement(bridge, proxies: list[str], src, tgt, E_src, sizes, device) -> dict[int, float]:
    """Label-free candidate (W7): top-10 agreement of bridged and native proxies over the whole index, by size."""
    X, Y = src.encode_queries(proxies), tgt.encode_queries(proxies)
    src.unload()
    tgt.unload()
    teacher = search_nested(X, E_src, sizes, 10, device)
    student = search_nested(bridge.transform_query(Y), E_src, sizes, 10, device)
    return {size: float(top_overlap(student[size], teacher[size], 10).mean()) for size in sizes}


def ranked_for(key: str, ds, sizes, queries, device) -> tuple:
    enc = Encoder(MODELS[key], device)
    store = enc.docs_store(ds.titles, ds.texts)
    Q = enc.queries(queries, ds.task)
    enc.unload()
    return store, Q, cached_search_nested(Q, store, sizes, device, exclude=ds.self_rows)


def file_config(ds, sizes: list[int], device: str, model_keys: tuple = ()) -> dict:
    """What a results file is tied to, beyond each cell's pair, K and seed: the size grid, the queries, the device, the code including this
    script, and the resolved revisions of the models it uses and of the dataset."""
    return {"dataset": ds.name, "n_docs": ds.N, "sizes": list(sizes), "queries_sha256": query_digest(ds.queries), "device": device,
            "implementation": implementation_digest(Path(__file__)),
            "revisions": revisions([MODELS[k].hf_id for k in model_keys], [DATASETS[ds.name.split("@")[0]].repo])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--pair", nargs="+", required=True, help="src:tgt model keys")
    p.add_argument("--K", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir(args.dataset)
    if args.max_queries is not None:
        ds = _first_queries(ds, args.max_queries)
    sizes = sorted(s for s in args.sizes if s <= ds.N)
    # A cell's key names its pair, K and seed; the file is tied to the rest: sizes, queries, device and code.
    check_or_write_config(args.out, file_config(ds, sizes, args.device, tuple(sorted({k for pr in args.pair for k in pr.split(":")}))))
    done = finished(args.out)
    positives = np.array(sorted({d for rel in ds.qrels for d in rel}))
    if positives.max() >= sizes[0]:
        raise ValueError("positives beyond the smallest size: not a nested corpus")
    exclude = np.union1d(positives, ds.self_rows[ds.self_rows >= 0])
    cells = [(seed, *pr.split(":"), K) for seed in args.seed for pr in args.pair for K in args.K]
    for n, (seed, src_key, tgt_key, K) in enumerate(cells, 1):
        cell = f"{ds.name}|{src_key}|{tgt_key}|K{K}|s{seed}"
        if cell in done:
            durable.progress(f"cell {n}/{len(cells)} {cell}: already recorded, skipped")
            continue
        t0 = time.perf_counter()
        durable.progress(f"cell {n}/{len(cells)} {cell}: start")
        E_src, Q_src, native = ranked_for(src_key, ds, sizes, ds.index_lang_queries, args.device)
        E_tgt, Q_tgt, oracle = ranked_for(tgt_key, ds, sizes, ds.queries, args.device)
        pool = Index(E_src.prefix(sizes[0]), ds.texts[: sizes[0]], ds.titles[: sizes[0]])
        src, tgt = MountEncoder(MODELS[src_key], ds.task, args.device), MountEncoder(MODELS[tgt_key], ds.task, args.device)
        bundle = freeze(pool, src, ds.lang, seed, K, anchor_select="random", exclude=exclude, device=args.device)
        src.unload()
        bridge = mount(bundle, pool, tgt, device=args.device, doc_pairs=True)
        t_fit = time.perf_counter() - t0
        Q_hat = bridge.transform_query(Q_tgt)
        ranked = search_nested(Q_hat, E_src, sizes, 1_000, args.device, exclude=ds.self_rows)
        val = [bundle.anchor_query_texts[i] for i in bundle.val_idx[:1024]]
        agree_val = proxy_agreement(bridge, val, src, tgt, E_src, sizes, args.device)
        agree_short = proxy_agreement(bridge, [_truncate_tokens(t, 8) for t in val], src, tgt, E_src, sizes, args.device)
        records = []
        for size in sizes:
            pq, native_pq, oracle_pq = per_query(ranked[size], ds.qrels), per_query(native[size], ds.qrels), per_query(oracle[size], ds.qrels)
            mean = lambda d: {m: float(v.mean()) for m, v in d.items()}
            bridged_m, native_m, oracle_m = mean(pq), mean(native_pq), mean(oracle_pq)
            rho_oracle, rho_ceil = retention(bridged_m, native_m, oracle_m)
            records.append({
                "dataset": ds.name, "src": src_key, "tgt": tgt_key, "K": bundle.K, "seed": seed, "size": size, "n_queries": len(ds.queries),
                "metrics": bridged_m, "native": native_m, "oracle": oracle_m, "rho_ceil": rho_ceil, "rho_oracle": rho_oracle,
                "ci_ceil": {m: bootstrap_retention_ci(pq[m], native_pq[m], oracle_pq[m]) for m in ("recall@30", "recall@300", "ndcg@10")},
                "rho_val": bridge.metrics["rho_val"], "solver": bridge.metrics.get("solver"), "mount": bridge.metrics["mount"],
                "overlap": {k: float(top_overlap(ranked[size], native[size], k).mean()) for k in (10, 30, 100)},
                "overlap30_sd": float(top_overlap(ranked[size], native[size], 30).std()),
                "agree_val_full": agree_val[size], "agree_short_full": agree_short[size],
                "score": bridge.quality.score, "cell": cell, "cell_records": len(sizes),
            })
        durable.append_lines(args.out, [json.dumps(r) for r in records])
        r30 = " ".join(f"{r['rho_ceil']['recall@300']:.3f}" for r in records)
        durable.progress(f"cell {n}/{len(cells)} {cell}: done in {time.perf_counter() - t0:.0f} s (fit {t_fit:.0f} s); R@300 ret by size {r30}")


if __name__ == "__main__":
    main()
