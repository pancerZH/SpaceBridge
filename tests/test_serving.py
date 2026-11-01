import numpy as np
import pytest

from spacebridge import ExactBackend, Index, Searcher, build_hnsw, freeze, mount

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
