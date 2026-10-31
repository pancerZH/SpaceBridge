"""Diagnose where cross-model retention is lost for one index <- query-model pair.

Varies one factor at a time: anchor count (beyond the N/10 cap), proxy vs. real-query anchors,
and the amount of calibration data a linear query map can use, with paired-bootstrap 95% CIs.

    uv run python scripts/diagnose_alignment.py --pair bge-base-en:bge-large-en
"""

import argparse

import numpy as np

from spacebridge import CalibrationBundle, SpaceBridge, extract_query_proxy, make_split, sample_distractors, select_anchors
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import bootstrap_ratio_ci, per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.search import search

METRICS = ("recall@10", "recall@30", "ndcg@10")
STAGE2_MAX_ROWS = 16_384
KMEANS_MAX_K = 4096
MAX_VAL = 1024


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--pair", default="bge-base-en:bge-large-en", help="src(index):tgt(queries)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--K", type=int, nargs="*", help="proxy anchor counts (default: N/10, n_train, 1024..4096, N)")
    args = p.parse_args()
    src_key, tgt_key = args.pair.split(":")

    test, train = load_beir(args.dataset, "test"), load_beir(args.dataset, "train")
    N = test.N
    proxies = [extract_query_proxy(test.titles[i], test.texts[i]) for i in range(N)]
    train_pos = np.array([min(rel) for rel in train.qrels])  # one positive doc per train query

    emb = {}
    for side, key in (("src", src_key), ("tgt", tgt_key)):
        enc = Encoder(MODELS[key], args.device)
        emb[side] = {
            "docs": enc.docs(test.titles, test.texts),
            "test_q": enc.queries(test.queries, test.task),
            "train_q": enc.queries(train.queries, test.task),
            "proxy": enc.queries(proxies, test.task),
        }
        enc.unload()
    s, t = emb["src"], emb["tgt"]
    E_src = s["docs"]

    def evaluate(Q: np.ndarray, E: np.ndarray) -> dict[str, np.ndarray]:
        return per_query(search(Q, E, 100, args.device, exclude=test.self_rows), test.qrels)

    native, oracle = evaluate(s["test_q"], E_src), evaluate(t["test_q"], t["docs"])
    ceiling = {m: native[m] if native[m].mean() <= oracle[m].mean() else oracle[m] for m in METRICS}

    def report(name: str, pq: dict[str, np.ndarray], extra: str = "") -> None:
        rho = {m: pq[m].mean() / ceiling[m].mean() for m in METRICS}
        lo, hi = bootstrap_ratio_ci(pq["recall@30"], ceiling["recall@30"])
        print(
            f"{name:<44} {rho['recall@10']:6.3f} {rho['recall@30']:6.3f} [{lo:.3f},{hi:.3f}] "
            f"{rho['ndcg@10']:7.3f}  {extra}"
        )

    def fit_eval(name: str, Xq, Yq, Xd, anchor_ids, restore=False, mode="stage1") -> None:
        K = len(Xq)
        train_idx, val_idx = make_split(K, val_frac=min(0.2, MAX_VAL / K))  # bounds val x pool memory
        bundle = CalibrationBundle(
            src_lang=test.lang, anchor_ids=anchor_ids, anchor_doc_texts=[], anchor_query_texts=[],
            X_doc_raw=Xd, X_query_raw=Xq, train_idx=train_idx, val_idx=val_idx,
            distractor_ids=sample_distractors(N, np.unique(anchor_ids)),
        )
        bridge = SpaceBridge(device=args.device, restore_source_mean=restore)
        fit = bridge.fit(bundle, Yq, mode=mode, X_distractor=E_src[bundle.distractor_ids])
        solver = fit["solver"] + (f" λ={fit['lambda_rel']}" if "lambda_rel" in fit else "")
        solver += f" +S2@{fit['stage2_best_step']}" if "stage2_best_step" in fit else ""
        report(f"{name} (K={K})", evaluate(bridge.transform_query(t["test_q"]), E_src), f"ρ_val={fit['rho_val']:.3f} {solver}")

    def proxies_at(K: int, random: bool = False):
        # CPU k-means is impractically slow beyond a few thousand clusters, so large K is sampled.
        if random or K > KMEANS_MAX_K:
            ids = np.sort(np.random.default_rng(0).choice(N, K, replace=False))
        else:
            ids = select_anchors(E_src, K)
        return s["proxy"][ids], t["proxy"][ids], E_src[ids], ids

    n_tr = len(train.queries)
    print(f"{args.dataset}: {src_key} (index) <- {tgt_key} (queries); N={N}, test queries={len(test.queries)}, train queries={n_tr}")
    print(f"{'configuration':<44} {'R@10':>6} {'R@30':>6} {'95% CI':^15} {'nDCG@10':>7}  (retention vs ceiling)")
    report("oracle (M_tgt re-index)", oracle)
    report("native (M_src)", native)
    mu_q = s["proxy"].mean(0)
    stripped = s["test_q"] - mu_q
    report("native, source query mean stripped", evaluate(stripped / np.linalg.norm(stripped, axis=1, keepdims=True), E_src))

    print("-- anchor count (query proxies)")
    for K in args.K or (N // 10, n_tr, 1024, 2048, 4096, N):
        K = min(K, N)
        fit_eval("proxies" + (" (random)" if KMEANS_MAX_K < K < N else ""), *proxies_at(K))
    print("-- anchor selection: random instead of k-means medoids")
    for K in (2048, 4096):
        fit_eval("proxies (random)", *proxies_at(min(K, N), random=True))
    largest = min(max(args.K or [N]), N, STAGE2_MAX_ROWS)
    fit_eval("proxies + Stage-2", *proxies_at(largest), mode="auto")
    print("-- anchor type at matched count")
    fit_eval("real train queries", s["train_q"], t["train_q"], E_src[train_pos], train_pos)
    print("-- maximum calibration data (linear-map ceiling)")
    fit_eval("doc<->doc pairs, all docs", E_src, t["docs"], E_src, np.arange(N))
    union = (
        np.concatenate([s["proxy"], s["train_q"], E_src]),
        np.concatenate([t["proxy"], t["train_q"], t["docs"]]),
        np.concatenate([E_src, E_src[train_pos], E_src]),
        np.concatenate([np.arange(N), train_pos, np.arange(N)]),
    )
    fit_eval("all proxies + train queries + docs", *union)
    fit_eval("  same, with mean restore", *union, restore=True)
    if len(union[0]) <= STAGE2_MAX_ROWS:  # Stage 2 is full-batch: K_train x K_train logits
        fit_eval("  same, + Stage-2", *union, mode="auto")


if __name__ == "__main__":
    main()
