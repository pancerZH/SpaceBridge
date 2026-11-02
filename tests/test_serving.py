import numpy as np
import pytest

from spacebridge import LARGE_INDEX_DOCS, ExactBackend, Index, Searcher, build_hnsw, freeze, mount
from spacebridge.quality import size_rerank_pool

from conftest import l2
from test_mount import make_mount_world


class ReverseAlphabetReranker:
    def score(self, query, docs):
        return np.array([-sum(map(ord, d)) for d in docs], dtype=np.float32)


def test_exact_backend_returns_the_true_top_k():
    E = l2(np.random.default_rng(0).standard_normal((500, 16))).astype(np.float32)
    q = E[:1] + 0.01
    ids, scores = ExactBackend(E).search(q, 20)
    truth = np.argsort(-(q @ E.T)[0])[:20]
    np.testing.assert_array_equal(ids, truth)
    assert (np.diff(scores) <= 0).all()


def test_searcher_runs_the_whole_path_and_reports_each_stage():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu")
    searcher = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), pool=30)
    out = searcher.search("question 17", k=5)
    assert len(out.ids) == 5 and out.pool == 30 and 0 <= out.confidence <= 1
    assert set(out.timing_ms) == {"encode", "project", "search", "fetch", "rerank", "total"}
    assert out.timing_ms["total"] >= out.timing_ms["search"]
    assert (np.diff(out.scores) <= 0).all()
    assert out.texts[0].startswith("the title of document")
    # Reranking only reorders the first stage's candidates.
    q_hat = bridge.transform_query(tgt.encode_queries(["question 17"]))
    assert set(out.ids) <= set(ExactBackend(vectors).search(q_hat, 30)[0])


def test_searcher_without_a_reranker_keeps_the_first_stage_order():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu")
    out = Searcher(bridge, tgt, ExactBackend(vectors), index).search("question 3", k=7)
    assert out.pool == bridge.quality.rerank_pool
    q_hat = bridge.transform_query(tgt.encode_queries(["question 3"]))
    np.testing.assert_array_equal(out.ids, ExactBackend(vectors).search(q_hat, 7)[0])


class SizedIndex(Index):
    """An index that reports a size other than its rows, to test the rules that depend on size."""

    def __init__(self, inner: Index, size: int):
        super().__init__(inner._vectors, inner._texts, inner._titles)
        self.size = size

    def __len__(self):
        return self.size


def test_searcher_requires_a_reranker_above_the_large_index_size():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu")
    backend = ExactBackend(vectors)
    assert Searcher(bridge, tgt, backend, SizedIndex(index, LARGE_INDEX_DOCS)).reranker is None
    large = SizedIndex(index, LARGE_INDEX_DOCS + 1)
    with pytest.raises(ValueError, match="needs a reranker"):
        Searcher(bridge, tgt, backend, large)
    assert Searcher(bridge, tgt, backend, large, ReverseAlphabetReranker()).reranker is not None
    out = Searcher(bridge, tgt, backend, large, allow_no_reranker=True).search("question 3", k=7)
    q_hat = bridge.transform_query(tgt.encode_queries(["question 3"]))
    np.testing.assert_array_equal(out.ids, backend.search(q_hat, 7)[0])


def test_searcher_without_a_report_follows_the_index_size():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu")
    bridge.quality = None
    backend, rr = ExactBackend(vectors), ReverseAlphabetReranker()
    for size in (800, LARGE_INDEX_DOCS, LARGE_INDEX_DOCS + 1, 1_000_000, 1_000_001, 8_841_823):
        assert Searcher(bridge, tgt, backend, SizedIndex(index, size), rr).pool == size_rerank_pool(size)
    assert Searcher(bridge, tgt, backend, index, rr, pool=17).pool == 17


def test_hnsw_build_continues_after_a_crash_and_finds_neighbours(tmp_path):
    pytest.importorskip("faiss")
    import faiss

    E = l2(np.random.default_rng(1).standard_normal((3000, 16))).astype(np.float32)
    path = tmp_path / "index.faiss"

    class Crash(Exception):
        pass

    def crash_after_a_save(ntotal, n):
        if ntotal >= 1500:
            raise Crash

    with pytest.raises(Crash):
        build_hnsw(E, path, M=16, ef_construction=40, batch_rows=500, save_seconds=0, on_batch=crash_after_a_save)
    assert 0 < faiss.read_index(str(path)).ntotal < 3000  # saved work, and no partial file at the final path
    assert [p.name for p in tmp_path.iterdir()] == ["index.faiss"]
    seen = []
    index = build_hnsw(E, path, M=16, ef_construction=40, batch_rows=500, save_seconds=0, on_batch=lambda n, _: seen.append(n))
    assert index.ntotal == 3000 and seen[0] > 500 and faiss.read_index(str(path)).ntotal == 3000
    index.hnsw.efSearch = 100
    _, ids = index.search(E[:50], 1)
    assert (ids[:, 0] == np.arange(50)).mean() > 0.95


def test_hnsw_backend_serves_the_saved_index(tmp_path):
    pytest.importorskip("faiss")
    from spacebridge import HnswBackend

    E = l2(np.random.default_rng(2).standard_normal((1000, 16))).astype(np.float32)
    build_hnsw(E, tmp_path / "i.faiss", M=16, ef_construction=40)
    ids, scores = HnswBackend(tmp_path / "i.faiss", ef_search=100).search(E[3:4], 10)
    assert ids[0] == 3 and scores[0] == pytest.approx(1.0, abs=1e-5) and len(ids) == 10


def test_real_query_overlap_is_one_for_the_identity_and_lower_for_a_new_model():
    from spacebridge.serving import real_query_overlap

    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, K=300, device="cpu")
    queries = [f"question {i}" for i in range(40)]
    backend = ExactBackend(vectors)
    same = real_query_overlap(mount(bundle, index, src, device="cpu"), queries, src, src, backend)
    other = real_query_overlap(mount(bundle, index, tgt, device="cpu"), queries, src, tgt, backend)
    assert same.mean() > 0.85 and same.mean() > other.mean() > 0.1 and other.shape == (40,)
