import numpy as np

from spacebridge_bench.datasets import _positive_preserving_subsample, self_match_rows
from spacebridge_bench.rerank import rerank, topk_jaccard
from spacebridge_bench.search import search


def test_search_excludes_self_rows():
    E = np.eye(4, dtype=np.float32)
    Q = np.array([[1, 0.5, 0, 0], [0, 0, 1, 0.2]], dtype=np.float32)
    assert search(Q, E, 2, "cpu").tolist() == [[0, 1], [2, 3]]
    assert search(Q, E, 2, "cpu", exclude=np.array([0, -1])).tolist()[0][0] == 1
    assert search(Q, E, 2, "cpu", exclude=np.array([0, -1])).tolist()[1] == [2, 3]


def test_positive_preserving_subsample_keeps_all_positives():
    ids = [str(i) for i in range(1000)]
    qrels = {"corpus-id": ["3", "500", "999", "7"], "score": [1, 2, 1, 0]}
    rows = _positive_preserving_subsample(ids, qrels, 100, seed=0)
    assert len(rows) == len(set(rows.tolist())) == 100
    assert {3, 500, 999} <= set(rows.tolist())


class _FakeReranker:
    """Scores a pair by the doc's trailing integer, so higher doc ids rank first."""

    def score(self, queries, docs, task):
        return np.array([float(d.split()[-1]) for d in docs])


def test_rerank_reorders_within_depth_only():
    docs = [f"doc {i}" for i in range(10)]
    ranked = np.array([[1, 5, 3, 9, 0]])
    out = rerank(_FakeReranker(), ranked, 3, ["q"], docs, "")
    assert out.tolist() == [[5, 3, 1]]


def test_topk_jaccard():
    a, b = np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 6, 7]])
    assert topk_jaccard(a, b, k=5)[0] == 3 / 7


def test_self_match_rows_uses_text_not_ids():
    texts = ["an argument", "another doc", ""]
    assert self_match_rows(["another doc", "unrelated query", ""], texts).tolist() == [1, -1, -1]


def test_mrl_slices_share_the_full_cache_and_renormalize(tmp_path):
    from spacebridge_bench.encode import Encoder
    from spacebridge_bench.models import MODELS

    full, sliced = MODELS["qwen3-0.6b"], MODELS["qwen3-0.6b-256"]
    assert sliced.dim == sliced.truncate_dim == 256 and sliced.hf_id == full.hf_id
    emb = np.random.default_rng(0).standard_normal((3, full.dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    enc = Encoder(sliced, "cpu", cache_dir=tmp_path)
    assert enc.cache_dir == Encoder(full, "cpu", cache_dir=tmp_path).cache_dir
    out = enc._slice(emb)
    assert out.shape == (3, 256)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(out[0] / out[0, 0], emb[0, :256] / emb[0, 0], rtol=1e-4)


def test_query_subsample_keeps_whole_queries():
    from spacebridge_bench.datasets import _query_subsample

    qids = ["a", "a", "b", "c", "c", "d"]
    rows = _query_subsample(qids, [1, 1, 1, 1, 0, 0], 2, seed=0)
    kept = {qids[i] for i in rows}
    assert len(kept) == 2 and "d" not in kept
    assert all(qids[i] in kept for i in rows) and sum(q in kept for q in qids) == len(rows)
