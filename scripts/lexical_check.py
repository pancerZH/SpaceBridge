"""A1's check: ``LexicalIndex.top`` against the study script's ``BM25.top``, on every query of a corpus. CPU only.

Compares the top 1,000 and the heads the study used (a quarter of budgets 60, 100, 300), and reports what differs: the build
(postings, weights, idf), and any row that moved, with whether the scores tie there.

    uv run python -u scripts/lexical_check.py hotpotqa-1m --out outputs/phase1/m3_lib/lexical_check.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hotpot_fixes import BM25  # noqa: E402

from spacebridge import Index, LexicalIndex  # noqa: E402
from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.datasets import load_beir  # noqa: E402

HEADS = (15, 25, 75)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--study-dir", type=Path, default=Path("outputs/phase1/hp_fixes"))
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir(args.dataset)
    script = BM25(ds, args.study_dir)
    t0 = time.perf_counter()
    lib = LexicalIndex.build(Index(np.empty((ds.N, 0)), ds.texts, ds.titles))
    build_s = time.perf_counter() - t0
    durable.progress(f"built in {build_s:.0f} s, {lib.X.nnz:,} postings")
    same_build = {
        "vocab": lib.vocab == script.vocab, "shape": lib.X.shape == script.X.shape, "indptr": bool(np.array_equal(lib.X.indptr, script.X.indptr)),
        "indices": bool(np.array_equal(lib.X.indices, script.X.indices)), "data": bool(np.array_equal(lib.X.data, script.X.data)), "idf": bool(np.array_equal(lib.idf, script.idf)),
    }
    durable.progress(f"build matches: {same_build}")

    rows = script.top(ds.queries, args.k, exclude=ds.self_rows)
    exact, set_same, moved, tie_only, heads_exact = 0, 0, [], 0, {m: 0 for m in HEADS}
    self_consistent = 0
    for i, q in enumerate(ds.queries):
        a = rows[i][rows[i] >= 0]
        b = lib.top(q, args.k, exclude=ds.self_rows[i])
        exact += len(a) == len(b) and bool((a == b).all())
        set_same += set(a.tolist()) == set(b.tolist())
        for m in HEADS:
            heads_exact[m] += np.array_equal(a[:m], b[:m])
        self_consistent += all(np.array_equal(lib.top(q, m, exclude=ds.self_rows[i]), b[:m]) for m in HEADS)
        if not (len(a) == len(b) and (a == b).all()):
            cols = sorted({lib.vocab[w] for w in _tokens(q) if w in lib.vocab})
            s = np.asarray(lib.X[:, cols] @ lib.idf[cols]).ravel()
            first = int(np.argmax(a != b)) if len(a) == len(b) else min(len(a), len(b))
            ties = len(a) == len(b) and bool(np.allclose(s[a], s[b], rtol=1e-5, atol=0))
            tie_only += ties
            moved.append({"query": i, "first_differs_at": first, "tied_within_rounding": ties})
        if (i + 1) % 500 == 0:
            durable.progress(f"{i + 1:,} / {len(ds.queries):,} queries, {exact:,} exact")
    n = len(ds.queries)
    rec = {"dataset": ds.name, "queries": n, "k": args.k, "build_s": build_s, "postings": int(lib.X.nnz), "same_build": same_build,
           "exact_at_k": exact, "same_set_at_k": set_same, "differ_only_by_ties": tie_only, "heads_exact": heads_exact, "top_m_is_head_of_top_k": self_consistent, "differences": moved[:50]}
    durable.append_lines(args.out, [json.dumps(rec)])
    durable.progress(f"exact {exact:,} / {n:,}, same set {set_same:,}, differ only by ties {tie_only:,}, heads {heads_exact}, top(m) = head of top(k) {self_consistent:,}")


def _tokens(q: str) -> list[str]:
    from spacebridge.lexical import tokens

    return tokens(q)


if __name__ == "__main__":
    main()
