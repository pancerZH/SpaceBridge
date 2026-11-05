import numpy as np
import pytest

from spacebridge import LARGE_INDEX_DOCS, ExactBackend, Index, LexicalIndex, Searcher, SpaceBridge, build_hnsw, freeze, merge_pool, mount
from spacebridge.quality import QualityReport, lexical_rerank_pool, size_rerank_pool

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
    searcher = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), pool=30, lexical=None)
    out = searcher.search("question 17", k=5)
    assert len(out.ids) == 5 and out.pool == 30 and 0 <= out.confidence <= 1
    assert set(out.timing_ms) == {"encode", "project", "search", "lexical", "fetch", "rerank", "total"}
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
    """An index that reports a size other than its rows, to test the rules that depend on size. The bridge's corpus check would
    rightly refuse it, so these tests clear the bridge's fingerprint and use a backend that doesn't check."""

    def __init__(self, inner: Index, size: int):
        super().__init__(inner._vectors, inner._texts, inner._titles)
        self.size = size

    def __len__(self):
        return self.size


class UncheckedBackend(ExactBackend):
    check_index = None


def any_size(bridge):
    bridge.corpus = None
    return bridge


def test_searcher_requires_a_reranker_above_the_large_index_size():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = any_size(mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu"))
    backend = UncheckedBackend(vectors)
    assert Searcher(bridge, tgt, backend, SizedIndex(index, LARGE_INDEX_DOCS)).reranker is None
    large = SizedIndex(index, LARGE_INDEX_DOCS + 1)
    with pytest.raises(ValueError, match="needs a reranker"):
        Searcher(bridge, tgt, backend, large)
    assert Searcher(bridge, tgt, backend, large, ReverseAlphabetReranker(), lexical=None).reranker is not None
    out = Searcher(bridge, tgt, backend, large, allow_no_reranker=True).search("question 3", k=7)
    q_hat = bridge.transform_query(tgt.encode_queries(["question 3"]))
    np.testing.assert_array_equal(out.ids, backend.search(q_hat, 7)[0])


def test_searcher_without_a_report_follows_the_index_size():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = any_size(mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu"))
    bridge.quality = None
    backend, rr = UncheckedBackend(vectors), ReverseAlphabetReranker()
    for size in (800, LARGE_INDEX_DOCS, LARGE_INDEX_DOCS + 1, 1_000_000, 1_000_001, 8_841_823):
        assert Searcher(bridge, tgt, backend, SizedIndex(index, size), rr, lexical=None).pool == size_rerank_pool(size)
    assert Searcher(bridge, tgt, backend, index, rr, pool=17, lexical=None).pool == 17


@pytest.fixture(scope="module")
def world():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    return vectors, index, tgt, any_size(mount(freeze(index, src, K=300, device="cpu"), index, tgt, device="cpu"))


def counting_builds(monkeypatch, lexical):
    """``auto`` over ``SizedIndex`` would read rows that aren't there, so the build hands back a real index and counts."""
    built = []
    monkeypatch.setattr("spacebridge.serving.timed_build", lambda index: built.append(len(index)) or lexical)
    return built


def test_auto_builds_at_any_size_for_an_english_index_with_a_reranker(world, monkeypatch):
    vectors, index, tgt, bridge = world
    backend, rr, lex = UncheckedBackend(vectors), ReverseAlphabetReranker(), LexicalIndex.build(index)
    built = counting_builds(monkeypatch, lex)
    assert Searcher(bridge, tgt, backend, index, rr).lexical is lex
    assert Searcher(bridge, tgt, backend, SizedIndex(index, 5_000), rr).lexical is lex
    assert Searcher(bridge, tgt, backend, SizedIndex(index, 1_000_000), rr).lexical is lex
    assert built == [800, 5_000, 1_000_000]


def test_auto_stays_off_for_another_language_and_with_no_reranker(world, monkeypatch):
    vectors, index, tgt, bridge = world
    backend, large = UncheckedBackend(vectors), SizedIndex(index, 100_000)
    built = counting_builds(monkeypatch, LexicalIndex.build(index))
    assert Searcher(bridge, tgt, backend, large).lexical is None
    bridge.src_lang = "zh"
    try:
        assert Searcher(bridge, tgt, backend, large, ReverseAlphabetReranker()).lexical is None
    finally:
        bridge.src_lang = "en"
    assert built == []


def test_a_lexical_index_passed_in_is_used_whatever_the_size_and_language(world):
    vectors, index, tgt, bridge = world
    lex = LexicalIndex.build(index)
    assert Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=lex).lexical is lex  # 800 documents
    bridge.src_lang = "zh"
    try:
        assert Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=lex).lexical is lex
    finally:
        bridge.src_lang = "en"
    assert Searcher(bridge, tgt, ExactBackend(vectors), index, lexical=lex).lexical is None  # no reranker to merge the lists
    assert Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=None).lexical is None
    with pytest.raises(ValueError, match="has 800 documents"):
        Searcher(bridge, tgt, UncheckedBackend(vectors), SizedIndex(index, 801), ReverseAlphabetReranker(), lexical=lex)
    with pytest.raises(ValueError, match="must be a LexicalIndex"):
        Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical="on")


def test_lexical_rows_reach_the_reranker_through_merge_pool(world):
    vectors, index, tgt, bridge = world
    lex, backend = LexicalIndex.build(index), ExactBackend(vectors)

    class Remember(ReverseAlphabetReranker):
        def score(self, query, docs):
            self.docs = docs
            return super().score(query, docs)

    rr = Remember()
    searcher = Searcher(bridge, tgt, backend, index, rr, pool=60, lexical=lex)
    query = "question 7 about the title of document 123"
    out = searcher.search(query, k=3)
    q_hat = bridge.transform_query(tgt.encode_queries([query]))
    want = merge_pool(backend.search(q_hat, 60)[0], lex.top(query, 15), 60)
    texts = [f"{t}\n{x}" for t, x in zip(index.titles(want), index.texts(want))]
    assert rr.docs == texts and out.pool == 60 and out.timing_ms["lexical"] >= 0
    assert set(lex.top(query, 15).tolist()) <= set(want.tolist())
    # A pool asked for per call sets the share too.
    assert searcher.search(query, k=3, pool=100).pool == 100
    assert rr.docs[-25:] == [f"{t}\n{x}" for t, x in zip(index.titles(lex.top(query, 25)), index.texts(lex.top(query, 25)))]


def test_a_query_in_another_language_skips_bm25_and_takes_the_plain_pool(world):
    vectors, index, tgt, bridge = world
    lex = LexicalIndex.build(index)
    bridge.register_query_language("zh", tgt.encode_queries(["question 1", "question 2"]), tgt.encode_queries(["question 3", "question 4"]))
    backend = UncheckedBackend(vectors)
    large = SizedIndex(index, 1_000_000)
    large_lex = LexicalIndex(lex.X, lex.idf, list(lex.vocab))
    large_lex.__class__ = type("Sized", (LexicalIndex,), {"__len__": lambda self: 1_000_000})
    stored = bridge.quality
    bridge.quality = report("B", stored=300)  # as index_size_check leaves a good fit above 200K
    try:
        searcher = Searcher(bridge, tgt, backend, large, ReverseAlphabetReranker(), lexical=large_lex)
        assert searcher.pool == 60 and searcher.plain_pool == 300
        q = "question 9 about the title of document 5"
        assert searcher.search(q, k=3).pool == 60
        assert searcher.search(q, k=3, query_lang="en").pool == 60
        out = searcher.search(q, k=3, query_lang="zh")
        q_hat = bridge.transform_query(tgt.encode_queries([q]), query_lang="zh")
        assert out.pool == 300 and set(out.ids) <= set(backend.search(q_hat, 300)[0])
    finally:
        bridge.quality = stored


def test_translate_feeds_bm25_only_and_only_for_a_query_in_another_language(world):
    vectors, index, tgt, bridge = world
    lex, backend = LexicalIndex.build(index), ExactBackend(vectors)
    bridge.register_query_language("zh", tgt.encode_queries(["question 1", "question 2"]), tgt.encode_queries(["question 3", "question 4"]))

    class Remember(ReverseAlphabetReranker):
        def score(self, query, docs):
            self.query, self.docs = query, docs
            return super().score(query, docs)

    calls, rr = [], Remember()

    def translate(query, lang):
        calls.append((query, lang))
        return "the title of document 123"

    searcher = Searcher(bridge, tgt, backend, index, rr, pool=60, lexical=lex, translate=translate)
    query = "问题 7"
    out = searcher.search(query, k=3, query_lang="zh")
    assert calls == [(query, "zh")] and rr.query == query  # the reranker keeps the original
    assert out.pool == 60 and "translate" in out.timing_ms and out.timing_ms["translate"] >= 0
    q_hat = bridge.transform_query(tgt.encode_queries([query]), query_lang="zh")
    want = merge_pool(backend.search(q_hat, 60)[0], lex.top("the title of document 123", 15), 60)
    assert rr.docs == [f"{t}\n{x}" for t, x in zip(index.titles(want), index.texts(want))]

    calls.clear()
    assert "translate" not in searcher.search("question 7 about document 9", k=3).timing_ms  # same language: nothing to translate
    assert "translate" not in searcher.search("question 7 about document 9", k=3, query_lang="en").timing_ms
    assert calls == []
    no_translate = Searcher(bridge, tgt, backend, index, rr, pool=60, lexical=lex).search(query, k=3, query_lang="zh")
    assert "translate" not in no_translate.timing_ms and set(no_translate.ids) <= set(backend.search(q_hat, 60)[0])  # skipped, as before


def test_translate_needs_a_lexical_index_and_a_reranker(world):
    vectors, index, tgt, bridge = world
    lex, backend = LexicalIndex.build(index), ExactBackend(vectors)
    identity = lambda q, lang: q
    with pytest.raises(ValueError, match="needs a reranker and a lexical index"):
        Searcher(bridge, tgt, backend, index, ReverseAlphabetReranker(), lexical=None, translate=identity)
    assert Searcher(bridge, tgt, backend, index, ReverseAlphabetReranker(), translate=identity).lexical is not None  # auto, any size
    bridge.src_lang = "zh"
    try:
        with pytest.raises(ValueError, match="needs a reranker and a lexical index"):
            Searcher(bridge, tgt, backend, index, ReverseAlphabetReranker(), translate=identity)  # auto stays off for this index
    finally:
        bridge.src_lang = "en"
    with pytest.raises(ValueError, match="needs a reranker and a lexical index"):
        Searcher(bridge, tgt, backend, index, lexical=lex, translate=identity)


def report(grade=None, stored=60, **stage2):
    r = QualityReport({"stage2": stage2} if stage2 else {}, grade=grade, rerank_pool=stored)
    return r


def test_the_pool_with_bm25_ignores_the_grade_and_never_inherits_what_is_stored():
    assert lexical_rerank_pool(None) == 60
    assert lexical_rerank_pool(report("A", stored=300)) == lexical_rerank_pool(report("B", stored=300)) == 60
    assert lexical_rerank_pool(report("C", stored=300)) == lexical_rerank_pool(report("D", stored=100)) == 60
    assert lexical_rerank_pool(report("C", stored=100, health_flag=1.0)) == 100
    assert lexical_rerank_pool(report("A", stored=100, health_flag=1.0)) == 100
    assert lexical_rerank_pool(report("A", health_flag=0.0)) == 60
    assert lexical_rerank_pool(report(stored=300), source_free=True) == 100
    assert lexical_rerank_pool(report(stored=60)) == 60  # no calibration on hand: nothing says the fit is weak


def test_a_source_free_bridge_with_bm25_takes_pool_100(world):
    vectors, index, tgt, _ = world
    bridge = mount(freeze(index, None, K=300, device="cpu"), index, tgt, device="cpu")
    searcher = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=LexicalIndex.build(index))
    assert searcher.pool == 100 and searcher.search("question 5 document 3", k=3).pool == 100


def test_a_bridge_saved_earlier_serves_with_bm25_at_the_new_pool(world, tmp_path):
    vectors, index, tgt, bridge = world
    bridge.quality.rerank_pool, stored = 300, bridge.quality.rerank_pool  # saved by M2's code above 200K, or read from one
    try:
        bridge.save(tmp_path / "old")
    finally:
        bridge.quality.rerank_pool = stored
    loaded = SpaceBridge.load(tmp_path / "old", device="cpu")
    backend = ExactBackend(vectors)
    assert Searcher(loaded, tgt, backend, index, ReverseAlphabetReranker(), lexical=None).pool == 300  # routing is as it was
    with_bm25 = Searcher(loaded, tgt, backend, index, ReverseAlphabetReranker(), lexical=LexicalIndex.build(index))
    assert with_bm25.pool == lexical_rerank_pool(loaded.quality) and with_bm25.pool < 300


def test_an_explicit_pool_wins_with_or_without_bm25(world):
    vectors, index, tgt, bridge = world
    lex, backend, rr = LexicalIndex.build(index), ExactBackend(vectors), ReverseAlphabetReranker()
    assert Searcher(bridge, tgt, backend, index, rr, pool=17, lexical=lex).pool == 17
    assert Searcher(bridge, tgt, backend, index, rr, pool=17, lexical=None).pool == 17


def test_pool_counts_the_candidates_actually_sent(world):
    vectors, index, tgt, bridge = world

    class Short:
        def search(self, q, k):
            return np.arange(5), np.zeros(5)

    out = Searcher(bridge, tgt, Short(), index, ReverseAlphabetReranker(), pool=30, lexical=None).search("question 2")
    assert out.pool == 5


def test_exact_backend_on_a_gpu_too_small_names_what_to_use_instead(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2**30, 2**31))
    with pytest.raises(ValueError, match=r"HnswBackend.*device=\"cpu\""):
        ExactBackend(np.zeros((1_000_000, 1024), dtype=np.float32), device="cuda")
    assert ExactBackend(np.zeros((10, 4), dtype=np.float32), device="cpu").vectors.shape == (10, 4)


def test_a_source_free_bridge_serves_with_nan_confidence():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, None, K=300, device="cpu"), index, tgt, device="cpu")
    out = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), pool=30).search("question 5", k=3)
    assert len(out.ids) == 3 and np.isnan(out.confidence)


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
    stale = path.with_name(f"{path.name}.tmp123")  # what a crash during a save leaves behind
    stale.write_bytes(b"partial")
    index = build_hnsw(E, path, M=16, ef_construction=40, batch_rows=500, save_seconds=0, on_batch=lambda n, _: seen.append(n))
    assert not stale.exists()
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
