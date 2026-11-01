"""MS MARCO at 1M documents: which anchors work (k-means over the index or over a 50K sample with / without test
positives, or random), and how deep a rerank pool must go (positive's rank over the whole index, no reranker).
Results are written to the JSON file given as the only argument.

    uv run python -u scripts/scale_anchors_depth.py outputs/scale_anchors_depth.json
"""

import json
import sys
from dataclasses import replace

import numpy as np
import torch

from spacebridge import CalibrationBundle, SpaceBridge, make_split, sample_distractors, select_anchors
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import cached_anchors, make_proxies
from spacebridge_bench.search import search

DEV = "cuda"
PAIRS = [("bge-large-en", "bge-base-en"), ("bge-large-en", "qwen3-0.6b"), ("harrier-0.6b", "bge-large-en")]
DEPTHS = (30, 60, 100, 200, 300, 500, 1000, 2000)
OUT = sys.argv[1]
results = []
ds = load_beir("msmarco-1m")
positives = np.array(sorted({d for rel in ds.qrels for d in rel}))
pos_set = set(positives.tolist())
rest = np.setdiff1d(np.arange(ds.N), positives)


def emit(**kw):
    results.append(kw)
    print(json.dumps(kw), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


def metrics(Q, E):
    pq = per_query(search(Q, E, 100, DEV), ds.qrels)
    return {m: float(pq[m].mean()) for m in ("recall@30", "recall@100", "ndcg@10")}


def fit(src, tgt, ids, E_src, seed):
    """The default engine (Row 8b) on the given anchor rows."""
    proxies = make_proxies([ds.titles[i] for i in ids], [ds.texts[i] for i in ids])
    enc_s, enc_t = Encoder(MODELS[src], DEV), Encoder(MODELS[tgt], DEV)
    X_q, Y_q = enc_s.queries(proxies, ds.task), enc_t.queries(proxies, ds.task)
    enc_s.unload()
    enc_t.unload()
    spec = MODELS[tgt]
    enc_d = Encoder(replace(spec, max_seq_length=min(128, spec.max_seq_length)), DEV)
    Y_d = enc_d.encode([spec.format_doc(ds.titles[i], ds.texts[i]) for i in ids])
    enc_d.unload()
    tr, va = make_split(len(ids), seed=seed)
    bundle = CalibrationBundle(
        ds.lang, ids, [], proxies, E_src[ids], X_q, tr, va, sample_distractors(ds.N, ids, seed=seed), E_src.mean(0)
    )
    bridge = SpaceBridge(device=DEV)
    info = bridge.fit(bundle, Y_q, X_distractor=E_src[bundle.distractor_ids], Y_doc_raw=Y_d)
    return bridge, info


def ranks(Q, E):
    """0-based rank of each query's best positive over the whole of E."""
    Et = torch.as_tensor(E, device=DEV)
    out = []
    for i in range(0, len(Q), 256):
        scores = torch.as_tensor(Q[i: i + 256], device=DEV) @ Et.T
        for j, rel in enumerate(ds.qrels[i: i + 256]):
            ids = torch.as_tensor(list(rel), device=DEV)
            out.append(int((scores[j] > scores[j, ids].max()).sum()))
    return np.array(out)


for src, tgt in PAIRS:
    enc = Encoder(MODELS[src], DEV)
    E_src, Q_src = enc.docs(ds.titles, ds.texts), enc.queries(ds.queries, ds.task)
    enc.unload()
    enc = Encoder(MODELS[tgt], DEV)
    E_tgt, Q_tgt = enc.docs(ds.titles, ds.texts), enc.queries(ds.queries, ds.task)
    enc.unload()
    N, O = metrics(Q_src, E_src), metrics(Q_tgt, E_tgt)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        sub_neg = rng.choice(rest, 50_000, replace=False)
        sub_pos = np.concatenate([positives, rng.choice(rest, 50_000 - len(positives), replace=False)])
        sets = {
            "kmeans_1M": cached_anchors(ds, src, E_src, np.arange(ds.N), 2048, seed, "kmeans"),
            "kmeans_50K_with_pos": sub_pos[select_anchors(E_src[sub_pos], 2048, seed)],
            "kmeans_50K_no_pos": sub_neg[select_anchors(E_src[sub_neg], 2048, seed)],
            "random_1M": rng.choice(ds.N, 2048, replace=False),
        }
        for name, ids in sets.items():
            bridge, info = fit(src, tgt, np.asarray(ids), E_src, seed)
            B = metrics(bridge.transform_query(Q_tgt), E_src)
            emit(test="anchors", src=src, tgt=tgt, seed=seed, set=name, pos_in_anchors=sum(int(i) in pos_set for i in ids),
                 rho={m: B[m] / min(N[m], O[m]) for m in B}, solver=info.get("solver"), step=info.get("stage2_best_step"),
                 rho_val=info["rho_val"])
    rank_n, rank_o = ranks(Q_src, E_src), ranks(Q_tgt, E_tgt)
    for K in (2048, 8192):
        ids = cached_anchors(ds, src, E_src, np.arange(ds.N), K, 0, "kmeans")
        bridge, _ = fit(src, tgt, np.asarray(ids), E_src, 0)
        rank_b = ranks(bridge.transform_query(Q_tgt), E_src)
        emit(test="depth", src=src, tgt=tgt, K=K, native={d: float((rank_n < d).mean()) for d in DEPTHS},
             oracle={d: float((rank_o < d).mean()) for d in DEPTHS}, bridged={d: float((rank_b < d).mean()) for d in DEPTHS})
    del E_src, E_tgt
    torch.cuda.empty_cache()
print("DONE", flush=True)
