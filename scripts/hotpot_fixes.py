"""Fixes that go around the map's error on HotpotQA, using only what a deployment already has: stored vectors and text.

A stored document vector carries no map error, so a document the reranker has confirmed is a clean query in the index's
own space; BM25 doesn't use the map at all. Results go to --out/<exp>.jsonl, one synced line per unit, and a rerun skips
what's there.

    prf     first stage only: the bridged query plus the stored vectors of its own top documents, unconfirmed
    bm25    BM25 alone, recall by depth
    bm25time  BM25's time per query
    rescue  of the positives the map loses from the top 300, the share BM25 holds, by depth
    pools   end to end at a rerank budget B: the plain top-B against budgets shared with the neighbours of confirmed
            documents, or with BM25. Native and re-index stay plain top-B, the ceiling M2's retention uses; they also
            run through the same fix, so the report can say what the fix would do for them

    uv run python -u scripts/hotpot_fixes.py pools --out outputs/phase1/hp_fixes --corpora hotpotqa-1m --tiers T1 T3b --budgets 300
"""
import argparse
import array
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from hotpot_study import CORPORA, DEV, PAIRS, Ctx, Log, dataset  # noqa: E402

from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.metrics import per_query, retention  # noqa: E402
from spacebridge_bench.rerank import RERANKERS, Reranker  # noqa: E402
from spacebridge_bench.run import _DocTexts  # noqa: E402
from spacebridge_bench.search import search  # noqa: E402

WIDTH = 50  # final list kept per query: enough for nDCG@5 and recall@30
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be but by for from has he in is it its of on or that the to was were which who what when where why how with".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class BM25:
    """BM25 (k1 = 0.9, b = 0.4) over title and text, as one sparse matrix, kept on disk after the first build."""

    def __init__(self, ds, out: Path, k1: float = 0.9, b: float = 0.4):
        path = out / f"bm25_{ds.name}.npz"
        if not path.exists():
            t0 = time.time()
            vocab: dict[str, int] = {}
            indptr, idx, tf = array.array("q", [0]), array.array("i"), array.array("f")
            for title, text in zip(ds.titles[: ds.N], ds.texts[: ds.N]):
                for w, n in Counter(tokens(f"{title} {text}")).items():
                    idx.append(vocab.setdefault(w, len(vocab)))
                    tf.append(n)
                indptr.append(len(idx))
            X = sp.csr_matrix((np.frombuffer(tf, np.float32), np.frombuffer(idx, np.int32), np.frombuffer(indptr, np.int64)), shape=(ds.N, len(vocab)), copy=True)
            del idx, tf, indptr
            dl = np.asarray(X.sum(1)).ravel()
            X.data = X.data * (k1 + 1) / (X.data + k1 * (1 - b + b * np.repeat(dl, np.diff(X.indptr)) / dl.mean()))
            X = X.tocsc()
            df = np.diff(X.indptr)
            idf = np.log(1 + (ds.N - df + 0.5) / (df + 0.5)).astype(np.float32)
            out.mkdir(parents=True, exist_ok=True)
            with durable.atomic_file(path) as f:
                np.savez(f, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape), idf=idf, words=np.array(list(vocab), dtype=object))
            durable.progress(f"bm25 {ds.name}: built in {time.time() - t0:.0f} s, {X.nnz:,} postings")
        z = np.load(path, allow_pickle=True)
        self.X = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        self.idf = z["idf"]
        self.vocab = {w: i for i, w in enumerate(z["words"].tolist())}

    def top(self, queries: list[str], k: int, exclude: np.ndarray | None = None) -> np.ndarray:
        """``exclude[i]`` (if >= 0) is a row never returned for query i, as in ``search``: ArguAna's queries are corpus rows."""
        out = np.full((len(queries), k), -1, dtype=np.int64)
        for i, q in enumerate(queries):
            cols = [self.vocab[w] for w in set(tokens(q)) if w in self.vocab]
            if not cols:
                continue
            s = np.asarray(self.X[:, cols] @ self.idf[cols]).ravel()
            if exclude is not None and exclude[i] >= 0:
                s[exclude[i]] = 0
            hits = np.flatnonzero(s)
            if len(hits) > k:
                hits = np.sort(hits[np.argpartition(-s[hits], k - 1)[:k]])  # row order, so ties break as a full sort would
            order = hits[np.argsort(-s[hits], kind="stable")[:k]]
            out[i, : len(order)] = order
        return out


def merge(lists: list[np.ndarray], budget: int) -> np.ndarray:
    """Round robin over ranked lists, dropping repeats and -1, until ``budget``: earlier lists first at each depth."""
    seen, out = set(), []
    for r in range(max(len(x) for x in lists)):
        for x in lists:
            if r < len(x) and x[r] >= 0 and int(x[r]) not in seen:
                seen.add(int(x[r]))
                out.append(int(x[r]))
                if len(out) == budget:
                    return np.array(out)
    return np.array(out)


class Pools:
    """Candidates by query, scored by the reranker, against the plain top-B of native and re-index."""

    def __init__(self, c: Ctx, n_queries: int, rr: Reranker):
        self.c, self.n, self.rr = c, n_queries, rr
        ds = c.ds
        self.queries, self.native_queries = ds.queries[:n_queries], ds.index_lang_queries[:n_queries]
        self.qrels = ds.qrels[:n_queries]
        self.docs = _DocTexts(ds.titles, ds.texts)
        self.task = ds.task

    def score_ranked(self, cands: list[np.ndarray], queries: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        """(n, WIDTH) final rows by reranker score, ties in candidate order, and each query's candidates re-sorted."""
        q_idx = np.repeat(np.arange(len(cands)), [len(x) for x in cands])
        flat = np.concatenate(cands)
        scores = self.rr.score([queries[i] for i in q_idx], [self.docs[j] for j in flat.tolist()], self.task)
        rows = np.full((len(cands), WIDTH), -1, dtype=np.int64)
        resorted, at = [], 0
        for i, x in enumerate(cands):
            order = np.argsort(-scores[at : at + len(x)], kind="stable")
            rows[i, : min(WIDTH, len(x))] = x[order][:WIDTH]
            resorted.append(x[order])
            at += len(x)
        return rows, resorted

    def metrics(self, rows: np.ndarray) -> dict:
        return {k: float(v.mean()) for k, v in per_query(rows, self.qrels).items()}

    def neighbours(self, E, seeds: np.ndarray, L: int) -> np.ndarray:
        """(n, m, L): for each seed document, its nearest rows by stored vector."""
        n, m = seeds.shape
        return search(np.asarray(E[seeds.ravel()], dtype=np.float32), E, L, DEV).reshape(n, m, L)

    def run(self, ranked: np.ndarray, queries: list[str], E, budget: int, bm25_rows: np.ndarray, config: dict) -> tuple[dict, float]:
        """One candidate strategy on one first stage. Returns metrics and mean reranked pairs per query."""
        kind = config["kind"]
        if kind == "plain":
            rows, _ = self.score_ranked([r[:budget] for r in ranked], queries)
            return self.metrics(rows), float(budget)
        if kind == "bm25":
            cands = [_fill(a, b[: config["m"]], budget) for a, b in zip(ranked, bm25_rows)]
            rows, _ = self.score_ranked(cands, queries)
            return self.metrics(rows), float(np.mean([len(x) for x in cands]))
        n0, rounds = config["n0"], config.get("rounds", 1)
        base = [r[:n0] for r in ranked]
        if config.get("m_bm25"):
            base = [_fill(r, b[: config["m_bm25"]], n0) for r, b in zip(ranked, bm25_rows)]
        for r in range(1, rounds + 1):
            _, resorted = self.score_ranked(base, queries)
            seeds = np.array([x[: config["seeds"]] for x in resorted])
            neighbours = self.neighbours(E, seeds, min(budget, 1000))
            target = n0 + (budget - n0) * r // rounds
            base = [_extend(x, list(neighbours[i]), target) for i, x in enumerate(base)]
        rows, _ = self.score_ranked(base, queries)
        return self.metrics(rows), float(np.mean([len(x) for x in base]))


def _fill(dense: np.ndarray, extra: np.ndarray, budget: int) -> np.ndarray:
    """All of ``extra`` plus the top of ``dense``, up to ``budget``, dense order first."""
    keep = [int(x) for x in extra if x >= 0]
    have = set(keep)
    room = budget - len(keep)
    head = []
    for d in dense.tolist():
        if room <= 0:
            break
        if d not in have:
            head.append(d)
            have.add(d)
            room -= 1
    return np.array(head + keep, dtype=np.int64)


def _extend(base: np.ndarray, lists: list[np.ndarray], budget: int) -> np.ndarray:
    extra = merge(lists, len(base) + budget)
    have, out = set(base.tolist()), base.tolist()
    for d in extra.tolist():
        if len(out) >= budget:
            break
        if d not in have:
            have.add(d)
            out.append(d)
    return np.array(out, dtype=np.int64)


def exp_prf(out: Path, args) -> None:
    grid = [(n, b) for n in (3, 10, 30) for b in (0.25, 0.5, 1.0, 2.0)]
    log = Log(out, "prf", len(args.corpora) * len(args.tiers))
    for corpus in args.corpora:
        for tier in args.tiers:
            key = f"{corpus}|{tier}|s{args.seed}"
            if log.has(key):
                continue
            t0 = time.time()
            c = Ctx(out.parent / "hp_study", corpus, tier, args.seed)
            ranked = search(c.Q_hat, c.E_src, max(n for n, _ in grid), DEV, exclude=c.ds.self_rows)
            res = {"baseline": c.evaluate(c.Q_hat)}
            for n, b in grid:
                V = np.asarray(c.E_src[ranked[:, :n].ravel()], dtype=np.float32).reshape(len(ranked), n, -1).mean(1)
                q = c.Q_hat + b * V
                res[f"n{n}_b{b}"] = c.evaluate(q / np.linalg.norm(q, axis=1, keepdims=True))
            log.write(key, t0, corpus=corpus, tier=tier, seed=args.seed, **res)


def exp_bm25(out: Path, args) -> None:
    log = Log(out, "bm25", len(args.corpora))
    for corpus in args.corpora:
        if log.has(corpus):
            continue
        t0 = time.time()
        ds = dataset(corpus)[1]
        rows = BM25(ds, out).top(ds.queries, 1000, exclude=ds.self_rows)
        pq = per_query(rows, ds.qrels)
        log.write(corpus, t0, corpus=corpus, n_queries=len(ds.queries), **{k: float(v.mean()) for k, v in pq.items()})


def exp_rescue(out: Path, args) -> None:
    """Of the positives the map pushes out of the top 300, how many does BM25 hold, and how deep?"""
    log = Log(out, "rescue", len(args.corpora) * len(args.tiers))
    for corpus in args.corpora:
        bm = BM25(dataset(corpus)[1], out)
        for tier in args.tiers:
            key = f"{corpus}|{tier}|s{args.seed}"
            if log.has(key):
                continue
            t0 = time.time()
            c = Ctx(out.parent / "hp_study", corpus, tier, args.seed)
            ds = c.ds
            bridged = search(c.Q_hat, c.E_src, 1000, DEV, exclude=ds.self_rows)
            rows = bm.top(ds.queries, 1000, exclude=ds.self_rows)
            lost_ranks, kept_ranks = [], []
            for q, rel in enumerate(ds.qrels):
                nat, brg, bmr = (dict(zip(r[q].tolist(), range(1, 1001))) for r in (c.native, bridged, rows))
                for d in rel:
                    if nat.get(d, 10**6) <= 300:
                        (lost_ranks if brg.get(d, 10**6) > 300 else kept_ranks).append(bmr.get(d, 10**6))
            lost_ranks, kept_ranks = np.array(lost_ranks), np.array(kept_ranks)
            log.write(key, t0, corpus=corpus, tier=tier, seed=args.seed, n_lost=len(lost_ranks), n_kept=len(kept_ranks),
                      bm25_holds_lost={k: float((lost_ranks <= k).mean()) for k in (15, 30, 50, 75, 100, 150, 300, 1000)},
                      bm25_holds_kept={k: float((kept_ranks <= k).mean()) for k in (15, 30, 50, 75, 100, 150, 300, 1000)})


def exp_bm25time(out: Path, args) -> None:
    """BM25's time per query on an idle machine, one CPU thread, with the index in memory."""
    log = Log(out, "bm25time", len(args.corpora))
    for corpus in args.corpora:
        if log.has(corpus):
            continue
        t0 = time.time()
        ds = dataset(corpus)[1]
        load0 = time.perf_counter()
        bm = BM25(ds, out)
        load = time.perf_counter() - load0
        ms = []
        for q in ds.queries[:200]:
            t = time.perf_counter()
            bm.top([q], 60)
            ms.append(1e3 * (time.perf_counter() - t))
        log.write(corpus, t0, corpus=corpus, docs=ds.N, postings=int(bm.X.nnz), load_s=load, p50_ms=float(np.percentile(ms, 50)), p95_ms=float(np.percentile(ms, 95)), max_ms=float(np.max(ms)))


CONFIGS = {
    "plain": {"kind": "plain"},
    "bm25_a": {"kind": "bm25", "m": None},  # m is a quarter of the budget
    "bm25_b": {"kind": "bm25", "m": None},  # a half
    "bm25_c": {"kind": "bm25", "m": None},  # three quarters
    "bm25_only": {"kind": "bm25", "m": None},  # no bridge at all: BM25's top-B and the reranker
    "graph_a": {"kind": "graph", "n0": None, "seeds": 5},  # n0 is half of the budget
    "graph_b": {"kind": "graph", "n0": None, "seeds": 10},
    "graph_c": {"kind": "graph", "n0": None, "seeds": 5},  # n0 is a third
    "graph_bm25": {"kind": "graph", "n0": None, "seeds": 5, "m_bm25": None},  # half, a quarter of that from BM25
    "graph_2r": {"kind": "graph", "n0": None, "seeds": 5, "rounds": 2},  # a third, then two rounds
}


def sized(name: str, budget: int) -> dict:
    cfg = dict(CONFIGS[name])
    if name == "bm25_a":
        cfg["m"] = budget // 4
    if name == "bm25_b":
        cfg["m"] = budget // 2
    if name == "bm25_c":
        cfg["m"] = 3 * budget // 4
    if name == "bm25_only":
        cfg["m"] = budget
    if name in ("graph_a", "graph_b", "graph_bm25"):
        cfg["n0"] = budget // 2
    if name in ("graph_c", "graph_2r"):
        cfg["n0"] = budget // 3
    if name == "graph_bm25":
        cfg["m_bm25"] = budget // 8
    return cfg


def exp_pools(out: Path, args) -> None:
    log = Log(out, "pools", len(args.corpora) * len(args.tiers) * len(args.budgets) * len(args.configs))
    rr = Reranker(RERANKERS["bge-reranker-v2-m3"], DEV)
    for corpus in args.corpora:
        bm = BM25(dataset(corpus)[1], out) if any("bm25" in c for c in args.configs) else None
        for tier in args.tiers:
            K = args.K or None
            rr.unload()  # a fit needs the GPU, and another job may share it; the reranker reloads when it next scores
            c = Ctx(out.parent / "hp_study", corpus, tier, args.seed, K)
            n = min(args.queries, len(c.ds.queries))
            p = Pools(c, n, rr)
            ds = c.ds
            bridged = search(c.Q_hat[:n], c.E_src, 1000, DEV, exclude=ds.self_rows[:n])
            bm_rows = bm.top(p.queries, 1000, exclude=ds.self_rows[:n]) if bm else None
            for budget in args.budgets:
                native_m, _ = p.run(c.native[:n], p.native_queries, c.E_src, budget, bm_rows, {"kind": "plain"})
                oracle_m, _ = p.run(c.oracle[:n], p.queries, c.E_tgt, budget, bm_rows, {"kind": "plain"})
                for name in args.configs:
                    key = f"{corpus}|{tier}|s{args.seed}|q{n}|B{budget}|{name}" + ("" if args.K == 8192 else f"|K{args.K}")
                    if log.has(key):
                        continue
                    t0 = time.time()
                    cfg = sized(name, budget)
                    m, pairs = p.run(bridged, p.queries, c.E_src, budget, bm_rows, cfg)
                    _, ceil = retention(m, native_m, oracle_m)
                    rec = {"corpus": corpus, "tier": tier, "src": c.src, "tgt": c.tgt, "seed": args.seed, "K": args.K, "queries": n, "budget": budget, "config": name, "cfg": cfg,
                           "pairs": pairs, "ndcg5": m["ndcg@5"], "ret": ceil["ndcg@5"], "recall30": m["recall@30"],
                           "native": native_m["ndcg@5"], "oracle": oracle_m["ndcg@5"]}
                    if name != "plain":  # the same fix for native and re-index: the stricter ceiling
                        nm, _ = p.run(c.native[:n], p.native_queries, c.E_src, budget, bm_rows, cfg)
                        om, _ = p.run(c.oracle[:n], p.queries, c.E_tgt, budget, bm_rows, cfg)
                        rec.update(native_with_fix=nm["ndcg@5"], oracle_with_fix=om["ndcg@5"],
                                   ret_strict=m["ndcg@5"] / min(nm["ndcg@5"], om["ndcg@5"]))
                    log.write(key, t0, **rec)
                    durable.progress(f"  {key}: nDCG@5 {m['ndcg@5']:.3f}, retention {ceil['ndcg@5']:.3f}")
    rr.unload()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=["prf", "bm25", "pools", "rescue", "bm25time"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpora", nargs="+", default=list(CORPORA))
    p.add_argument("--tiers", nargs="+", default=["T1", "T3"], choices=list(PAIRS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=8192, help="0 for the library's default at this size")
    p.add_argument("--queries", type=int, default=500)
    p.add_argument("--budgets", type=int, nargs="+", default=[300])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"prf": exp_prf, "bm25": exp_bm25, "pools": exp_pools, "rescue": exp_rescue, "bm25time": exp_bm25time}[args.exp](args.out, args)


if __name__ == "__main__":
    main()
