"""Regression tests for defects found in review: wrong results without an error, misleading statistics, and inputs that used to fail deep inside a fit."""
import json

import numpy as np
import pytest
import torch

from spacebridge import ExactBackend, Index, LexicalIndex, RefineConfig, Searcher, SpaceBridge, build_hnsw, freeze, merge_pool, mount, retrieval_stats, select_anchors
from spacebridge.bundle import CalibrationBundle
from spacebridge.language import load_shift_bank
from spacebridge.solvers import stiefel_procrustes
from spacebridge_bench.metrics import bootstrap_retention_ci, recall_at, spearman
from spacebridge_bench.provenance import check_or_write_config, unique_records
from spacebridge_bench.run import finished_cells

from conftest import l2
from test_mount import make_mount_world


class ReverseAlphabetReranker:
    def score(self, query, docs):
        return np.array([-sum(map(ord, d)) for d in docs], dtype=np.float32)


@pytest.fixture(scope="module")
def world():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, K=300, device="cpu")
    return vectors, index, src, tgt, bundle, mount(bundle, index, tgt, device="cpu")


def reordered(vectors, index):
    order = np.random.default_rng(1).permutation(len(vectors))
    rows = np.arange(len(vectors))
    return vectors[order], [index.texts(rows)[i] for i in order], [index.titles(rows)[i] for i in order]


# ---- artifacts belong to a corpus and an encoder

def test_a_bundle_or_bridge_refuses_the_same_size_corpus_in_another_order(world):
    vectors, index, src, tgt, bundle, bridge = world
    other_v, other_t, other_ti = reordered(vectors, index)
    other = Index(other_v, other_t, other_ti)
    with pytest.raises(ValueError, match="another corpus or row order"):
        mount(bundle, other, tgt, device="cpu")
    with pytest.raises(ValueError, match="another corpus or row order"):
        Searcher(bridge, tgt, ExactBackend(other_v), other, lexical=None)


def test_a_bridge_refuses_re_embedded_vectors_with_the_same_text(world):
    vectors, index, src, tgt, bundle, bridge = world
    rows = np.arange(len(vectors))
    other_v = l2(np.random.default_rng(2).standard_normal(vectors.shape)).astype(np.float32)
    with pytest.raises(ValueError, match="other vectors"):
        Searcher(bridge, tgt, ExactBackend(other_v), Index(other_v, index.texts(rows), index.titles(rows)), lexical=None)


def test_a_backend_or_lexical_index_refuses_another_corpus(world):
    vectors, index, src, tgt, bundle, bridge = world
    other_v, other_t, other_ti = reordered(vectors, index)
    with pytest.raises(ValueError, match="another corpus or row order"):
        Searcher(bridge, tgt, ExactBackend(other_v), index, lexical=None)
    other_lex = LexicalIndex.build(Index(other_v, other_t, other_ti))
    with pytest.raises(ValueError, match="another corpus or row order"):
        Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=other_lex)


def test_fingerprints_and_encoder_identity_survive_a_save(world, tmp_path):
    vectors, index, src, tgt, bundle, bridge = world
    bundle.save(tmp_path / "b")
    loaded = CalibrationBundle.load(tmp_path / "b")
    assert loaded.corpus is not None
    other_v, other_t, other_ti = reordered(vectors, index)
    with pytest.raises(ValueError):
        mount(loaded, Index(other_v, other_t, other_ti), tgt, device="cpu")
    bridge.tgt_encoder = {"model": "the-one-it-was-fitted-with"}
    try:
        bridge.save(tmp_path / "r")
        served = SpaceBridge.load(tmp_path / "r", device="cpu")
        assert served.corpus is not None
        tgt.identity = {"model": "another"}
        with pytest.raises(ValueError, match="different query encoder"):
            Searcher(served, tgt, ExactBackend(vectors), index, lexical=None)
        tgt.identity = {"model": "the-one-it-was-fitted-with"}
        assert Searcher(served, tgt, ExactBackend(vectors), index, lexical=None).pool > 0
    finally:
        bridge.tgt_encoder = None
        del tgt.identity


def test_a_lexical_index_keeps_its_fingerprint_through_a_save(world, tmp_path):
    vectors, index, *_ = world
    lex = LexicalIndex.build(index)
    lex.save(tmp_path / "l")
    assert LexicalIndex.load(tmp_path / "l").corpus == lex.corpus


def test_hnsw_refuses_to_continue_over_other_vectors_settings_or_a_smaller_corpus(tmp_path):
    pytest.importorskip("faiss")
    E = l2(np.random.default_rng(0).standard_normal((500, 16))).astype(np.float32)
    path = tmp_path / "g.faiss"
    build_hnsw(E, path, M=8, ef_construction=40)
    assert build_hnsw(E, path, M=8, ef_construction=40).ntotal == 500  # the same call continues
    other = l2(np.random.default_rng(1).standard_normal((500, 16))).astype(np.float32)
    with pytest.raises(ValueError, match="other vectors"):
        build_hnsw(other, path, M=8, ef_construction=40)
    with pytest.raises(ValueError, match="more than the 400"):
        build_hnsw(E[:400], path, M=8, ef_construction=40)
    with pytest.raises(ValueError, match="M = 8"):
        build_hnsw(E, path, M=16, ef_construction=40)
    with pytest.raises(ValueError, match="efConstruction = 40"):
        build_hnsw(E, path, M=8, ef_construction=80)
    with pytest.raises(ValueError, match="dimension 16"):
        build_hnsw(l2(np.random.default_rng(2).standard_normal((500, 8))).astype(np.float32), path, M=8, ef_construction=40)


def test_hnsw_backend_refuses_an_index_with_other_rows(tmp_path):
    pytest.importorskip("faiss")
    from spacebridge import HnswBackend

    E = l2(np.random.default_rng(0).standard_normal((300, 16))).astype(np.float32)
    build_hnsw(E, tmp_path / "g.faiss", M=8, ef_construction=40)
    backend = HnswBackend(tmp_path / "g.faiss")
    texts = [f"t{i}" for i in range(300)]
    backend.check_index(Index(E, texts))
    with pytest.raises(ValueError, match="other vectors"):
        backend.check_index(Index(E[::-1], texts))


def test_two_builders_cannot_share_a_path(tmp_path):
    pytest.importorskip("faiss")
    fcntl = pytest.importorskip("fcntl")
    E = l2(np.random.default_rng(0).standard_normal((50, 8))).astype(np.float32)
    path = tmp_path / "g.faiss"
    with (tmp_path / "g.faiss.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="another process"):
            build_hnsw(E, path)


# ---- resume identity

def test_a_finished_cell_of_another_configuration_is_refused_not_skipped(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"cell": "a", "cell_records": 1, "row": "0a", "config_id": "screen-500-queries"}) + "\n")
    with pytest.raises(ValueError, match="another configuration"):
        finished_cells(path, "full-2000-queries")
    assert finished_cells(path, "screen-500-queries") == {"a"}
    legacy = tmp_path / "old.jsonl"
    legacy.write_text(json.dumps({"cell": "a", "cell_records": 1, "row": "0a"}) + "\n")
    with pytest.raises(ValueError, match="another configuration"):
        finished_cells(legacy, "anything")


def test_a_keyed_results_file_is_tied_to_one_configuration(tmp_path):
    out = tmp_path / "t.jsonl"
    check_or_write_config(out, {"n": 20})
    check_or_write_config(out, {"n": 20})
    with pytest.raises(ValueError, match=r"differs in \['n'\]"):
        check_or_write_config(out, {"n": 500})
    old = tmp_path / "old.jsonl"
    old.write_text("{}\n")
    with pytest.raises(ValueError, match="no old.jsonl.config.json"):
        check_or_write_config(old, {"n": 20})


def test_duplicate_records_are_an_error_unless_allowed():
    recs = [{"k": 1, "v": 1}, {"k": 1, "v": 2}, {"k": 2, "v": 3}]
    with pytest.raises(ValueError, match="two records"):
        unique_records(recs, lambda r: r["k"])
    assert [r["v"] for r in unique_records(recs, lambda r: r["k"], allow_duplicates=True)] == [2, 3]
    with pytest.raises(ValueError, match="mix different proxy"):
        unique_records([{"k": 1, "proxy": "a"}, {"k": 2, "proxy": "b"}], lambda r: r["k"], protocol=("proxy",))


# ---- statistics

def test_the_retention_interval_retakes_the_weaker_baseline_in_each_replicate():
    rng = np.random.default_rng(0)
    native, oracle = rng.random(400), rng.random(400)  # about equal, so which is weaker changes between replicates
    bridge = (native + oracle) / 2
    lo, hi = bootstrap_retention_ci(bridge, native, oracle)
    point = bridge.mean() / min(native.mean(), oracle.mean())
    assert lo < point < hi
    # With the minimum retaken, retention can't be judged against one fixed baseline: every replicate's denominator is at most either mean.
    idx = np.random.default_rng(0).integers(0, 400, size=(2000, 400))
    fixed = bridge[idx].mean(1) / (native if native.mean() <= oracle.mean() else oracle)[idx].mean(1)
    assert hi >= np.percentile(fixed, 97.5)
    assert bootstrap_retention_ci(np.zeros(5), np.zeros(5), np.zeros(5)) != (0.0, 0.0)


def test_spearman_handles_ties_and_constant_sides():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))  # argsort of argsort gave this one a spurious value
    assert np.isnan(spearman([1], [1]))


def test_recall_at_any_cutoff_reads_the_whole_list():
    ranked = np.arange(300)[None, :]
    assert recall_at(ranked, [{150: 1}], 200)[0] == 1.0  # past 100, where the fallback stopped reading
    assert recall_at(ranked, [{150: 1}], 100)[0] == 0.0
    assert recall_at(ranked, [{10: 1, 250: 2}], 100)[0] == 0.5


# ---- ties and refits

def test_a_tied_target_ranks_after_the_rows_it_ties_with():
    Q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    pool = torch.ones((20, 2)) / np.sqrt(2)  # every score equal: only one row can truly come first
    out = retrieval_stats(Q, pool, torch.tensor([0, 1]))
    assert out["recall"] == 0.0 and out["mrr"] == 0.0
    pool = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert retrieval_stats(Q, pool, torch.tensor([0, 0]))["mrr"] == 1.0  # with no tie, as before


def test_a_source_free_refit_forgets_the_earlier_teacher(world):
    _, index, src, tgt, bundle, _ = world
    Y_q, Y_d = tgt.encode_queries(bundle.anchor_query_texts), tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts)
    distract = index.vectors(bundle.distractor_ids)
    bridge = SpaceBridge(device="cpu")
    bridge.fit(bundle, Y_q, X_distractor=distract, Y_doc_raw=Y_d)
    assert bridge._conf_map is not None and bridge.status != "NO_TEACHER"
    bridge.fit_source_free(bundle, Y_q, Y_d, distract)
    assert bridge._conf_map is None and bridge._val_pool is None and bridge.status == "NO_TEACHER"
    _, confidence = bridge.transform_query(tgt.encode_queries(["q 1"]), return_confidence=True)
    assert np.isnan(confidence).all()


def test_a_naive_refit_leaves_no_report_or_confidence_behind(world):
    _, index, src, tgt, bundle, _ = world
    Y_q, Y_d = tgt.encode_queries(bundle.anchor_query_texts), tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts)
    distract = index.vectors(bundle.distractor_ids)
    bridge = SpaceBridge(device="cpu")
    bridge.fit(bundle, Y_q, X_distractor=distract, Y_doc_raw=Y_d)
    bridge.fit(bundle, Y_q, mode="naive", X_distractor=distract)
    assert bridge.quality is None and bridge._support_anchors is None and bridge._conf_map is None


def test_a_failed_refit_leaves_nothing_usable(world):
    _, index, src, tgt, bundle, _ = world
    Y_q = tgt.encode_queries(bundle.anchor_query_texts)
    bridge = SpaceBridge(device="cpu", doc_pairs=False)
    bridge.fit(bundle, Y_q)
    with pytest.raises(ValueError, match="all the same vector"):
        bridge.fit(bundle, np.ones_like(Y_q))
    with pytest.raises(RuntimeError):
        bridge.transform_query(Y_q[:1])


def test_source_free_stage2_honors_restore_source_mean(world):
    _, index, src, tgt, bundle, _ = world
    Y_q, Y_d = tgt.encode_queries(bundle.anchor_query_texts), tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts)
    with_mean = SpaceBridge(device="cpu")
    with_mean.fit_source_free(bundle, Y_q, Y_d)
    without = SpaceBridge(device="cpu", restore_source_mean=False)
    without.fit_source_free(bundle, Y_q, Y_d)
    assert not np.allclose(with_mean.W, without.W)


# ---- solver

def test_objective_scale_minimizes_the_truncated_objective():
    rng = np.random.default_rng(0)
    Y = torch.as_tensor(rng.standard_normal((40, 60)), dtype=torch.float32)  # more dimensions than rows, and truncation
    X = torch.as_tensor(rng.standard_normal((40, 30)), dtype=torch.float32)
    Y, X = Y - Y.mean(0), X - X.mean(0)
    W, _ = stiefel_procrustes(Y, X, rel_tol=0.5, objective_scale=True)
    W = W.double()
    loss = lambda s: ((Y.double() @ W) * s - X.double()).pow(2).sum().item()
    assert loss(1.0) <= min(loss(0.9), loss(1.1))
    with pytest.raises(ValueError, match="no usable cross-covariance"):
        stiefel_procrustes(torch.zeros(5, 3), torch.zeros(5, 3))


# ---- inputs

def test_an_index_rejects_misaligned_rows_and_shapes():
    with pytest.raises(ValueError, match="row-aligned"):
        Index(np.zeros((3, 2)), ["a", "b"])
    with pytest.raises(ValueError, match="row-aligned"):
        Index(np.zeros((2, 2)), ["a", "b"], ["t"])
    with pytest.raises(ValueError, match="shape"):
        Index(np.zeros(3), ["a", "b", "c"])


def test_freeze_checks_size_anchors_method_and_vectors():
    v = l2(np.random.default_rng(0).standard_normal((60, 8))).astype(np.float32)
    texts = [f"title {i}. some text." for i in range(60)]
    index = Index(v, texts)

    class Src:
        size_class = "small"

        def encode_queries(self, t):
            return l2(np.random.default_rng(1).standard_normal((len(t), 8))).astype(np.float32)

        def encode_docs(self, ti, t, max_tokens=None):
            return self.encode_queries(t)

    with pytest.raises(ValueError, match="at least 50"):
        freeze(Index(v[:10], texts[:10]), Src(), device="cpu")
    with pytest.raises(ValueError, match="K must be at least 20"):
        freeze(index, Src(), K=5, device="cpu")
    with pytest.raises(ValueError, match="unknown anchor selection"):
        freeze(index, Src(), anchor_select="kmens", device="cpu")
    bad = v.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        freeze(Index(bad, texts), Src(), device="cpu")
    with pytest.warns(UserWarning, match="unit length"):
        freeze(Index(v * 3, texts), Src(), device="cpu")


def test_a_corpus_with_no_distractors_has_finite_diagnostics():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, device="cpu")  # K = N: every document is an anchor
    assert bundle.K == 800 and len(bundle.distractor_ids) == 0 and np.isfinite(bundle.mu_x_doc).all()
    bridge = mount(bundle, index, tgt, device="cpu")
    assert np.isfinite(bridge.quality.details["difficulty"]["cone_angle_deg"])
    assert bridge.quality.score is not None


def test_small_pools_are_bounded_everywhere():
    vectors, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, K=20, device="cpu")
    bundle.distractor_ids = bundle.distractor_ids[:0]
    bridge = mount(bundle, index, tgt, device="cpu")
    assert bridge.quality is not None


def test_duplicate_vectors_get_an_actionable_error_from_kmeans():
    E = np.repeat(l2(np.random.default_rng(0).standard_normal((3, 8))).astype(np.float32), 2000, axis=0)
    with pytest.raises(ValueError, match="distinct directions"):
        select_anchors(E, 50, device="cpu")


def test_search_validates_k_and_pool_and_covers_k(world):
    vectors, index, src, tgt, bundle, bridge = world
    searcher = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=None)
    for bad in ({"k": 0}, {"k": -1}, {"k": 1.5}, {"k": 5, "pool": 0}, {"k": 5, "pool": -3}):
        with pytest.raises(ValueError):
            searcher.search("q 1", **bad)
    with pytest.raises(ValueError, match="pool must be"):
        Searcher(bridge, tgt, ExactBackend(vectors), index, lexical=None, pool=0)
    assert len(searcher.search("question 3", k=100).ids) == 100
    assert searcher.search("question 3", k=5, pool=10).pool == 10


def test_a_registered_language_gets_its_own_advice_on_the_plain_path(world):
    vectors, index, src, tgt, bundle, bridge = world
    bridge.register_query_language("zh", tgt.encode_queries(["a 1", "b 2"]), tgt.encode_queries(["c 3", "d 4"]))
    bridge.quality.by_language["zh"] = {"score": 60.0, "band": (55.0, 65.0), "grade": "D", "rerank_pool": 100, "experimental": True}
    try:
        searcher = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=None)
        assert searcher.search("q 1", query_lang="zh").pool == 100
        assert searcher.search("q 1").pool == searcher.plain_pool
        assert searcher.search("q 1", query_lang="zh", pool=30).pool == 30
        fixed = Searcher(bridge, tgt, ExactBackend(vectors), index, ReverseAlphabetReranker(), lexical=None, pool=17)
        assert fixed.search("q 1", query_lang="zh").pool == 17
    finally:
        bridge.quality.by_language.pop("zh")
        bridge.mu_y_by_lang.pop("zh")


def test_merge_pool_never_repeats_or_exceeds_the_budget():
    out = merge_pool(np.array([1, 2, 3]), np.array([7, 7, 8, 9, 10, 11]), 3)
    assert len(out) == 3 and len(set(out.tolist())) == 3
    assert merge_pool(np.array([1, 2]), None, 5).tolist() == [1, 2]


def test_real_query_overlap_counts_the_results_there_are():
    from spacebridge.serving import real_query_overlap

    class Short:
        def search(self, q, k):
            return np.arange(5), np.zeros(5)

    class Enc:
        def encode_queries(self, t):
            return np.zeros((len(t), 3), dtype=np.float32)

    class Bridge:
        def transform_query(self, q, query_lang=None):
            return q

    assert real_query_overlap(Bridge(), ["q"], Enc(), Enc(), Short(), k=30)[0] == 1.0


def test_unknown_options_are_rejected_not_substituted(world):
    _, index, src, tgt, bundle, _ = world
    Y_q, Y_d = tgt.encode_queries(bundle.anchor_query_texts), tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts)
    for kwargs in ({"center": "qery"}, {"operator": "stefel"}, {"stage2_rows": "all"}):
        with pytest.raises(ValueError, match="unknown"):
            SpaceBridge(device="cpu").fit_source_free(bundle, Y_q, Y_d, **kwargs)
    with pytest.raises(ValueError, match="unknown fit mode"):
        SpaceBridge(device="cpu", doc_pairs=False).fit(bundle, Y_q, mode="autoo")
    with pytest.raises(ValueError, match="steps"):
        RefineConfig(steps=0)
    with pytest.raises(ValueError, match="eval_every"):
        RefineConfig(eval_every=0)
    with pytest.raises(ValueError, match="stop_rule"):
        RefineConfig(stop_rule="best")


# ---- language banks

def test_default_banks_refuse_an_index_not_in_english():
    with pytest.raises(ValueError, match="pass mount.shift_banks"):
        load_shift_bank("en", src_lang="zh")


def test_a_language_with_no_fallback_gets_an_actionable_error(monkeypatch):
    def out_of_reach(*_, **__):
        raise OSError("gated")

    monkeypatch.setattr("spacebridge.language.load_flores", out_of_reach)
    with pytest.raises(ValueError, match="Tatoeba fallback covers only") as e:
        load_shift_bank("de")
    assert isinstance(e.value.__cause__, OSError)
    with pytest.raises(ValueError, match="no default language bank"):
        load_shift_bank("xx")


def test_a_programming_error_in_the_bank_is_not_taken_for_no_access(monkeypatch):
    from spacebridge.language import flores_available

    def bug(*_, **__):
        raise KeyError("bug")

    monkeypatch.setattr("spacebridge.language.load_flores", bug)
    with pytest.raises(KeyError):
        flores_available("de")


# ---- encoder adapter

def test_the_encoder_adapter_rejects_misaligned_docs_and_returns_shaped_empties():
    pytest.importorskip("sentence_transformers")
    from spacebridge.encoders import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder.__new__(SentenceTransformerEncoder)
    enc.doc_template = "{body}"
    with pytest.raises(ValueError, match="row-aligned"):
        enc.encode_docs(["t"], ["a", "b"])
