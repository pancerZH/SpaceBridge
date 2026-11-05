"""Retrieval metrics and retention ratios (design §2.1, §7.4)."""

import numpy as np
from scipy.stats import rankdata

RECALL_KS = (5, 10, 30, 100)
DEEP_RECALL_KS = (300, 1000)  # recall at rerank depth, for rankings that reach it


def evaluate(ranked: np.ndarray, qrels: list[dict[int, int]]) -> dict[str, float]:
    """Recall@{5,10,30,100}, MRR@10 and NDCG@{5,10}; ``qrels[i]`` maps doc row -> graded relevance.

    Per query, then averaged over queries with equal weight. Recall is the share of the query's positives (relevance > 0) in the top
    k rows given: on a list shorter than k, such as a reranked pool, that is recall of the list. nDCG has linear gain, log2(rank + 1)
    discount and an ideal order over the query's judged positives; unjudged documents count as not relevant. Every query kept by a
    loader has at least one positive in the corpus."""
    return {name: float(v.mean()) for name, v in per_query(ranked, qrels).items()}


def per_query(ranked: np.ndarray, qrels: list[dict[int, int]]) -> dict[str, np.ndarray]:
    """The metrics of ``evaluate`` for each query separately (for paired bootstrap intervals)."""
    recall_ks = RECALL_KS + tuple(k for k in DEEP_RECALL_KS if ranked.shape[1] >= k)
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in recall_ks}
    out.update({"mrr@10": [], "ndcg@5": [], "ndcg@10": []})
    for row, rel in zip(ranked, qrels):
        positives = {d for d, g in rel.items() if g > 0}
        for k in recall_ks:
            out[f"recall@{k}"].append(len(positives.intersection(row[:k].tolist())) / len(positives))
        first = next((i for i, d in enumerate(row[:10]) if d in positives), None)
        out["mrr@10"].append(0.0 if first is None else 1.0 / (first + 1))
        for k in (5, 10):
            out[f"ndcg@{k}"].append(_ndcg(row[:k], rel, k))
    return {name: np.asarray(v) for name, v in out.items()}


def recall_at(ranked: np.ndarray, qrels: list[dict[int, int]], k: int) -> np.ndarray:
    """Per-query Recall@k at any cutoff, over the whole of the first k rows given."""
    return np.array([len({d for d, g in rel.items() if g > 0}.intersection(row[:k].tolist())) / len([g for g in rel.values() if g > 0]) for row, rel in zip(ranked, qrels)])


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% paired-bootstrap interval of mean(num) / mean(den) over queries, for one fixed denominator."""
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n, len(num)))
    ratios = num[idx].mean(1) / den[idx].mean(1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def bootstrap_retention_ci(bridge: np.ndarray, native: np.ndarray, oracle: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap interval of retention over queries, resampling (bridge, native, re-index) triples together and taking
    the weaker baseline again in every replicate: which baseline is weaker is itself part of the statistic. A replicate whose
    denominator is zero has no defined ratio and is dropped, so the interval is conditional on a nonzero baseline."""
    idx = np.random.default_rng(seed).integers(0, len(bridge), size=(n, len(bridge)))
    den = np.minimum(native[idx].mean(1), oracle[idx].mean(1))
    num = bridge[idx].mean(1)
    ok = den > 0
    if not ok.any():
        return float("nan"), float("nan")
    ratios = num[ok] / den[ok]
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def spearman(a, b) -> float:
    """Spearman correlation with average ranks for ties; undefined (NaN) when either side is constant or shorter than 2."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _ndcg(row: np.ndarray, rel: dict[int, int], k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = sum(rel.get(int(d), 0) * discounts[i] for i, d in enumerate(row))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g * discounts[i] for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def retention(bridge: dict, src_native: dict, oracle: dict) -> tuple[dict, dict]:
    """(rho_oracle, rho_ceil) per metric; the ceiling is min(src-native, oracle)."""
    rho_oracle = {m: _ratio(v, oracle[m]) for m, v in bridge.items()}
    rho_ceil = {m: _ratio(v, min(src_native[m], oracle[m])) for m, v in bridge.items()}
    return rho_oracle, rho_ceil


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")
