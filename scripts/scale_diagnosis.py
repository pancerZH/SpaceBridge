"""Why does retention fall at 1M documents (MS MARCO)? Separates index density, map quality, proxy shape, and how
far more or better calibration pairs go: one map evaluated on growing subsets of the index, a map fitted at 50K
applied to the 1M index, map error on real queries vs. proxies, and Stage 1 fitted on proxies vs. real train queries.
Results are written to the JSON file given as the only argument (one seed).

    uv run python -u scripts/scale_diagnosis.py outputs/scale_diagnosis.json
"""
import json
import sys
from dataclasses import replace

import numpy as np
import torch
from datasets import load_dataset

from spacebridge import CalibrationBundle, SpaceBridge, make_split, sample_distractors
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import cached_anchors, make_proxies
from spacebridge_bench.search import search

DEV = "cuda"
SEED = 0
PAIRS = [("bge-large-en", "bge-base-en"), ("bge-large-en", "qwen3-0.6b"), ("harrier-0.6b", "bge-large-en")]
SIZES = (50_000, 100_000, 200_000, 500_000, 1_000_000)
OUT = sys.argv[1]
results = []


def emit(**kw):
    results.append(kw)
    print(json.dumps(kw), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


def metrics(Q, E, qrels, sel=None):
    ranked = search(Q, E if sel is None else E[sel], 100, DEV)
    if sel is not None:
        ranked = sel[ranked]
    pq = per_query(ranked, qrels)
    return {m: float(pq[m].mean()) for m in ("recall@30", "recall@100", "ndcg@10")}


def retention(b, n, o):
    return {m: b[m] / min(n[m], o[m]) for m in b}


def positive_ranks(Q, E, qrels):
    """0-based rank of each query's best positive over the whole of E."""
    Et = torch.as_tensor(E, device=DEV)
    out = []
    for i in range(0, len(Q), 256):
        q = torch.as_tensor(Q[i: i + 256], device=DEV)
        s = q @ Et.T
        for j, rel in enumerate(qrels[i: i + 256]):
            ids = torch.as_tensor(list(rel), device=DEV)
            out.append(int((s[j] > s[j, ids].max()).sum()))
    return np.array(out)


def bundle_for(ds, E_src, X_q, anchor_ids, proxies, K):
    tr, va = make_split(K, val_frac=min(0.2, 1000 / K), seed=SEED)
    return CalibrationBundle(
        src_lang=ds.lang, anchor_ids=anchor_ids, anchor_doc_texts=[], anchor_query_texts=proxies,
        X_doc_raw=E_src[anchor_ids], X_query_raw=X_q, train_idx=tr, val_idx=va,
        distractor_ids=sample_distractors(ds.N, anchor_ids, seed=SEED), mu_x_doc=E_src.mean(0),
    )


def load_side(ds, key):
    enc = Encoder(MODELS[key], DEV)
    E, Q = enc.docs(ds.titles, ds.texts), enc.queries(ds.queries, ds.task)
    return enc, E, Q


def default_fit(ds, src, tgt, E_src, K):
    """The shipped default (Row 8b) on ``ds``. Returns bridge, bundle, target proxy embeddings."""
    ids = cached_anchors(ds, src, E_src, np.arange(ds.N), K, SEED, "kmeans")
    proxies = make_proxies([ds.titles[i] for i in ids], [ds.texts[i] for i in ids])
    enc_s, enc_t = Encoder(MODELS[src], DEV), Encoder(MODELS[tgt], DEV)
    X_q, Y_q = enc_s.queries(proxies, ds.task), enc_t.queries(proxies, ds.task)
    spec = MODELS[tgt]
    enc_d = Encoder(replace(spec, max_seq_length=min(128, spec.max_seq_length)), DEV)
    Y_d = enc_d.encode([spec.format_doc(ds.titles[i], ds.texts[i]) for i in ids])
    b = bundle_for(ds, E_src, X_q, ids, proxies, K)
    br = SpaceBridge(device=DEV)
    br.fit(b, Y_q, X_distractor=E_src[b.distractor_ids], Y_doc_raw=Y_d)
    for e in (enc_s, enc_t, enc_d):
        e.unload()
    return br, b, Y_q, proxies


ds1m, ds50k = load_beir("msmarco-1m"), load_beir("msmarco")
qs = load_dataset("mteb/msmarco", "queries", split="queries")
qtext = dict(zip(map(str, qs["_id"]), qs["text"]))
tr_q = load_dataset("mteb/msmarco", "default", split="train")
tr_ids = sorted({str(q) for q, s in zip(tr_q["query-id"], tr_q["score"]) if s > 0})
dev_texts = set(ds1m.queries)
train_texts = [qtext[q] for q in tr_ids if q in qtext and qtext[q] not in dev_texts]
order = np.random.default_rng(0).permutation(len(train_texts))
TRAIN_Q = [train_texts[i] for i in order[:32_768]]
print(f"{len(train_texts)} train queries available, using {len(TRAIN_Q)}", flush=True)

rng = np.random.default_rng(0)
positives = np.array(sorted({d for rel in ds1m.qrels for d in rel}))
rest = np.setdiff1d(np.arange(ds1m.N), positives)
perm = rng.permutation(rest)
subsets = {n: np.sort(np.concatenate([positives, perm[: n - len(positives)]])) for n in SIZES[:-1]}
subsets[SIZES[-1]] = None

for src, tgt in PAIRS:
    print(f"\n=== {src} <- {tgt}", flush=True)
    enc, E_src, Q_src = load_side(ds1m, src)
    enc.unload()
    enc, E_tgt, Q_tgt = load_side(ds1m, tgt)
    enc.unload()
    base = dict(src=src, tgt=tgt)

    # 1. Same map, index density only.
    br, b, Y_q, proxies = default_fit(ds1m, src, tgt, E_src, 2048)
    Q_hat = br.transform_query(Q_tgt)
    for n in SIZES:
        sel = subsets[n]
        N, O, B = metrics(Q_src, E_src, ds1m.qrels, sel), metrics(Q_tgt, E_tgt, ds1m.qrels, sel), metrics(Q_hat, E_src, ds1m.qrels, sel)
        emit(test="density", **base, N=n, native=N, oracle=O, bridged=B, rho=retention(B, N, O))

    # 2. Map fitted on the 50K corpus, applied to the 1M index.
    enc, E50, _ = load_side(ds50k, src)
    enc.unload()
    br50, _, _, _ = default_fit(ds50k, src, tgt, E50, 2048)
    Q50 = br50.transform_query(Q_tgt)
    N, O = metrics(Q_src, E_src, ds1m.qrels), metrics(Q_tgt, E_tgt, ds1m.qrels)
    B50, B1m = metrics(Q50, E_src, ds1m.qrels), metrics(Q_hat, E_src, ds1m.qrels)
    emit(test="cross_fit", **base, fit50k_on_1m=retention(B50, N, O), fit1m_on_1m=retention(B1m, N, O),
         cos_maps=float(np.mean(np.sum(Q50 * Q_hat, axis=1))))
    del E50

    # 3. Map error: real queries vs. held-out proxies; positive ranks.
    va = b.val_idx
    P_hat = br.transform_query(Y_q[va])
    cos_q = np.sum(Q_hat * Q_src, axis=1)
    cos_p = np.sum(P_hat * b.X_query_raw[va], axis=1)
    mq, mp = Q_src.mean(0), b.X_query_raw.mean(0)
    def res_cos(a, x, m):
        ra, rx = a - m, x - m
        return float(np.mean(np.sum(ra * rx, 1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rx, axis=1))))
    rn, rb = positive_ranks(Q_src, E_src, ds1m.qrels), positive_ranks(Q_hat, E_src, ds1m.qrels)
    pct = lambda r: {str(p): float(np.percentile(r, p)) for p in (50, 75, 90, 95)}
    # Share of the loss by native rank: queries native gets in its top 30, by how far their positive falls.
    hit = rn < 30
    emit(test="error", **base, cos_query=float(cos_q.mean()), cos_proxy=float(cos_p.mean()),
         rescos_query=res_cos(Q_hat, Q_src, mq), rescos_proxy=res_cos(P_hat, b.X_query_raw[va], mp),
         len_query=float(np.mean([len(q.split()) for q in ds1m.queries])), len_proxy=float(np.mean([len(p.split()) for p in proxies])),
         ranks_native=pct(rn), ranks_bridged=pct(rb), native_top30=float(hit.mean()),
         lost_from_top30=float((rb[hit] >= 30).mean()), lost_within_100=float(((rb[hit] >= 30) & (rb[hit] < 100)).mean()),
         lost_within_1000=float(((rb[hit] >= 100) & (rb[hit] < 1000)).mean()), lost_beyond_1000=float((rb[hit] >= 1000).mean()))

    # 4/5. Stage 1 calibrated on proxies vs. on real train queries, by K.
    enc_s, enc_t = Encoder(MODELS[src], DEV), Encoder(MODELS[tgt], DEV)
    Xq_real, Yq_real = enc_s.queries(TRAIN_Q, ds1m.task), enc_t.queries(TRAIN_Q, ds1m.task)
    enc_s.unload()
    enc_t.unload()
    rand_docs = np.random.default_rng(1).choice(ds1m.N, len(TRAIN_Q), replace=False)
    for K in (2048, 8192, 32768):
        ids = rand_docs[:K]
        bq = bundle_for(ds1m, E_src, Xq_real[:K], ids, TRAIN_Q[:K], K)
        br_q = SpaceBridge(device=DEV, doc_pairs=False)
        fit = br_q.fit(bq, Yq_real[:K], mode="stage1", X_distractor=E_src[bq.distractor_ids])
        B = metrics(br_q.transform_query(Q_tgt), E_src, ds1m.qrels)
        emit(test="real_query_stage1", **base, K=K, solver=fit.get("solver"), lam=fit.get("lambda_rel"), rho=retention(B, N, O),
             cos_query=float(np.mean(np.sum(br_q.transform_query(Q_tgt) * Q_src, axis=1))))
    for K in (2048, 8192):
        ids = cached_anchors(ds1m, src, E_src, np.arange(ds1m.N), K, SEED, "kmeans")
        pr = make_proxies([ds1m.titles[i] for i in ids], [ds1m.texts[i] for i in ids])
        enc_s, enc_t = Encoder(MODELS[src], DEV), Encoder(MODELS[tgt], DEV)
        Xp, Yp = enc_s.queries(pr, ds1m.task), enc_t.queries(pr, ds1m.task)
        enc_s.unload()
        enc_t.unload()
        bp = bundle_for(ds1m, E_src, Xp, ids, pr, K)
        br_p = SpaceBridge(device=DEV, doc_pairs=False)
        fit = br_p.fit(bp, Yp, mode="stage1", X_distractor=E_src[bp.distractor_ids])
        B = metrics(br_p.transform_query(Q_tgt), E_src, ds1m.qrels)
        emit(test="proxy_stage1", **base, K=K, solver=fit.get("solver"), lam=fit.get("lambda_rel"), rho=retention(B, N, O),
             cos_query=float(np.mean(np.sum(br_p.transform_query(Q_tgt) * Q_src, axis=1))))
    del E_src, E_tgt
    torch.cuda.empty_cache()
print("DONE", flush=True)
