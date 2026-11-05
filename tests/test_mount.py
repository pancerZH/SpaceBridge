import json
import re

import numpy as np
import pytest

import spacebridge
import spacebridge.identity
from spacebridge import CalibrationBundle, Index, SpaceBridge, freeze, mount
from spacebridge.anchors import make_query_proxies

from conftest import l2, overlap_at_k, top_k


class FakeEncoder:
    """A model of one latent world: a text's row number picks its latent vector, and queries add a fixed noise."""

    def __init__(self, Z, Z_q, A, doc_off, q_off, size_class="small"):
        self.Z, self.Z_q, self.A, self.doc_off, self.q_off, self.size_class = Z, Z_q, A, doc_off, q_off, size_class
        self.doc_calls, self.query_calls = [], 0

    @staticmethod
    def _rows(texts):
        return [int(re.search(r"(\d+)", t).group(1)) for t in texts]

    def encode_queries(self, texts):
        self.query_calls += 1
        return l2(self.Z_q[self._rows(texts)] @ self.A + self.q_off)

    def encode_docs(self, titles, texts, max_tokens=None):
        self.doc_calls.append(max_tokens)
        return l2(self.Z[self._rows(texts)] @ self.A + self.doc_off)


def make_mount_world(n_docs=3000, d_lat=32, d_s=48, d_t=40, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_docs, d_lat))
    Z_q = Z + noise * rng.standard_normal((n_docs, d_lat))
    src = FakeEncoder(Z, Z_q, rng.standard_normal((d_lat, d_s)) / np.sqrt(d_lat), rng.standard_normal(d_s), rng.standard_normal(d_s))
    tgt = FakeEncoder(Z, Z_q, rng.standard_normal((d_lat, d_t)) / np.sqrt(d_lat), rng.standard_normal(d_t), rng.standard_normal(d_t))
    titles = [f"the title of document {i}" for i in range(n_docs)]
    texts = [f"text {i}. More words about it." for i in range(n_docs)]
    vectors = src.encode_docs(titles, texts)
    return vectors, Index(vectors, texts, titles), src, tgt


@pytest.fixture(scope="module")
def mounted():
    vectors, index, src, tgt = make_mount_world()
    bundle = freeze(index, src, device="cpu", K=600)
    return vectors, index, src, tgt, bundle, mount(bundle, index, tgt, device="cpu")


def test_freeze_applies_the_defaults_for_a_small_index(mounted):
    _, index, _, _, bundle, _ = mounted
    assert bundle.K == 600 and bundle.index_size == len(index)
    assert len(bundle.train_idx) + len(bundle.val_idx) == 600 and not set(bundle.train_idx) & set(bundle.val_idx)
    assert bundle.anchor_query_texts == make_query_proxies(bundle.anchor_doc_titles, bundle.anchor_doc_texts)
    assert not set(bundle.distractor_ids) & set(bundle.anchor_ids)
    np.testing.assert_array_equal(bundle.X_doc_raw, index.vectors(bundle.anchor_ids))


def test_freeze_chooses_random_anchors_above_the_size_switch_without_reading_the_index():
    class Large:
        def __init__(self):
            self.rows_read = 0

        def __len__(self):
            return spacebridge.LARGE_INDEX_DOCS + 1

        def vectors(self, ids):
            self.rows_read += len(ids)
            return np.full((len(ids), 4), 0.5, dtype=np.float32)

        def texts(self, ids):
            return [f"text {i}." for i in ids]

        def titles(self, ids):
            return [""] * len(ids)

    class Encoder:
        def encode_queries(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    index = Large()
    bundle = freeze(index, Encoder(), seed=3)
    assert bundle.K == spacebridge.default_anchor_count(len(index)) == 8192
    np.testing.assert_array_equal(bundle.anchor_ids, spacebridge.random_anchors(len(index), 8192, 3))
    assert index.rows_read <= 8192 + 50_000 + 2 * spacebridge.identity.SAMPLE_ROWS  # plus the sample freeze checks and fingerprints


def test_freeze_keeps_excluded_rows_out_of_the_anchors():
    _, index, src, _ = make_mount_world(n_docs=800)
    excluded = np.arange(100)
    bundle = freeze(index, src, K=200, exclude=excluded, device="cpu")
    assert bundle.anchor_ids.min() >= 100


def test_mount_aligns_a_new_model_and_reports_quality(mounted):
    vectors, _, src, tgt, _, bridge = mounted
    q_rows = [f"question {i}" for i in np.random.default_rng(1).choice(len(vectors), 200, replace=False)]
    native, bridged = top_k(src.encode_queries(q_rows), vectors), top_k(bridge.transform_query(tgt.encode_queries(q_rows)), vectors)
    assert overlap_at_k(native, bridged) > 0.35
    assert bridge.quality_report().score is not None and bridge.metrics["mount"]["fit_ms"] > 0


def test_identity_through_mount_stays_close_to_native(mounted):
    vectors, index, src, _, bundle, _ = mounted
    q_rows = [f"question {i}" for i in np.random.default_rng(2).choice(len(vectors), 200, replace=False)]
    identity = mount(bundle, index, src, device="cpu")
    native, bridged = top_k(src.encode_queries(q_rows), vectors), top_k(identity.transform_query(src.encode_queries(q_rows)), vectors)
    assert overlap_at_k(native, bridged) > 0.9


def test_mount_decides_doc_pairs_from_the_size_class():
    _, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, K=300, device="cpu")
    tgt.doc_calls.clear()
    assert "doc_pairs" in mount(bundle, index, tgt, device="cpu").metrics
    assert tgt.doc_calls == [spacebridge.DOC_PAIR_TOKENS]
    tgt.size_class, tgt.doc_calls = "large", []
    assert "doc_pairs" not in mount(bundle, index, tgt, device="cpu").metrics
    assert tgt.doc_calls == []


def test_mount_registers_query_languages_from_given_banks():
    _, index, src, tgt = make_mount_world(n_docs=800)
    bundle = freeze(index, src, K=300, device="cpu")
    bank = ([f"sentence {i}" for i in range(40)], [f"句子 {i}" for i in range(40, 80)])
    bridge = mount(bundle, index, tgt, ["zh"], device="cpu", shift_banks={"zh": bank})
    assert set(bridge.mu_y_by_lang) == {"en", "zh"}
    shift = tgt.encode_queries(bank[1]).mean(0) - tgt.encode_queries(bank[0]).mean(0)
    np.testing.assert_allclose(bridge.mu_y_by_lang["zh"] - bridge.mu_y_by_lang["en"], shift, atol=1e-6)
    assert any("does not apply" in n for n in bridge.quality_report().notes)


def test_freeze_with_a_parallel_bank_has_no_index_anchors():
    _, index, src, tgt = make_mount_world(n_docs=800)
    n = 500
    bank = ([f"english {i}" for i in range(n)], [f"中文 {i}" for i in range(n)])
    bundle = freeze(index, src, K=300, parallel_bank=bank, seed=1, device="cpu")
    assert bundle.K == 300 and (bundle.anchor_ids == -1).all() and bundle.anchor_doc_titles is None
    assert bundle.tgt_anchor_texts[0].split()[-1] == bundle.anchor_query_texts[0].split()[-1]  # parallel
    assert mount(bundle, index, tgt, device="cpu").W.shape == (40, 48)


def test_bundle_round_trip(mounted, tmp_path):
    bundle = mounted[4]
    bundle.save(tmp_path / "b")
    loaded = CalibrationBundle.load(tmp_path / "b")
    for name in ("anchor_ids", "X_doc_raw", "X_query_raw", "train_idx", "val_idx", "distractor_ids", "mu_x_doc"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(bundle, name))
    assert loaded.anchor_doc_titles == bundle.anchor_doc_titles and loaded.anchor_query_texts == bundle.anchor_query_texts
    assert loaded.index_size == bundle.index_size and loaded.tgt_anchor_texts is None


def test_bundle_round_trip_without_the_source_model_and_with_language_validation(mounted, tmp_path):
    index = mounted[1]
    bundle = freeze(index, None, K=100, anchor_ids=np.arange(100))
    assert bundle.X_query_raw is None
    bundle.language_validation = {"zh": {"src": ["a"], "lang": ["甲"], "X_query": np.ones((1, 4)), "X_doc": np.zeros((1, 4))}}
    bundle.save(tmp_path / "b")
    loaded = CalibrationBundle.load(tmp_path / "b")
    assert loaded.X_query_raw is None and loaded.language_validation["zh"]["lang"] == ["甲"]
    np.testing.assert_array_equal(loaded.language_validation["zh"]["X_query"], np.ones((1, 4)))


def test_bridge_round_trip_serves_the_same_results(mounted, tmp_path):
    vectors, _, _, tgt, _, bridge = mounted
    bridge.mu_y_by_lang["zh"] = bridge.mu_y_by_lang["en"] + 0.01
    bridge.save(tmp_path / "bridge")
    loaded = SpaceBridge.load(tmp_path / "bridge", device="cpu")
    Y = tgt.encode_queries([f"question {i}" for i in range(50)])
    for lang in ("en", "zh"):
        a, conf_a = bridge.transform_query(Y, query_lang=lang, return_confidence=True)
        b, conf_b = loaded.transform_query(Y, query_lang=lang, return_confidence=True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(conf_a, conf_b)
    report, again = bridge.quality_report(), loaded.quality_report()
    assert (report.score, report.band, report.grade, report.rerank_pool, report.notes) == (again.score, again.band, again.grade, again.rerank_pool, again.notes)
    assert report.details == again.details and loaded.status == bridge.status
    del bridge.mu_y_by_lang["zh"]


def test_an_interrupted_or_corrupted_save_does_not_load(mounted, tmp_path):
    bridge = mounted[5]
    bridge.save(tmp_path / "bridge")
    manifest = json.loads((tmp_path / "bridge" / "manifest.json").read_text())
    assert manifest["kind"] == "fitted-bridge" and manifest["format_version"] == 1
    with pytest.raises(ValueError, match="fitted-bridge"):
        CalibrationBundle.load(tmp_path / "bridge")
    with (tmp_path / "bridge" / "arrays.npz").open("r+b") as f:
        f.seek(100)
        f.write(b"\xff\xff\xff")
    with pytest.raises(ValueError, match="checksum"):
        SpaceBridge.load(tmp_path / "bridge")
    (tmp_path / "bridge" / "manifest.json").unlink()  # the state a crash during a save leaves
    with pytest.raises(ValueError, match="did not finish"):
        SpaceBridge.load(tmp_path / "bridge")
    bridge.save(tmp_path / "bridge")
    assert SpaceBridge.load(tmp_path / "bridge", device="cpu").status == bridge.status


def test_saving_a_bridge_needs_a_fit(tmp_path):
    with pytest.raises(RuntimeError, match="fit"):
        SpaceBridge(device="cpu").save(tmp_path / "bridge")


def test_source_free_fit_needs_no_source_model_and_says_what_it_cannot_report():
    vectors, index, src, tgt = make_mount_world(n_docs=1500)
    bundle = freeze(index, None, K=400, device="cpu")
    assert bundle.X_query_raw is None
    with pytest.raises(ValueError, match="fit_source_free"):
        SpaceBridge(device="cpu").fit(bundle, tgt.encode_queries(bundle.anchor_query_texts), Y_doc_raw=tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts))
    bridge = mount(bundle, index, tgt, device="cpu")
    assert bridge.status == "NO_TEACHER" and tgt.doc_calls[-1] == spacebridge.DOC_PAIR_TOKENS
    report = bridge.quality_report()
    assert report.score is None and report.grade is None and report.rerank_pool >= 100 and "owndoc_mrr" in report.details["recoverability"]
    assert any("without the source model" in n for n in report.notes)
    q_rows = [f"question {i}" for i in np.random.default_rng(1).choice(len(vectors), 200, replace=False)]
    q_hat, conf = bridge.transform_query(tgt.encode_queries(q_rows), return_confidence=True)
    assert np.isnan(conf).all() and conf.shape == (200,)
    native = top_k(src.encode_queries(q_rows), vectors)
    assert overlap_at_k(native, top_k(q_hat, vectors)) > 0.2


def test_a_large_model_still_gets_doc_pairs_without_the_source_model():
    _, index, _, tgt = make_mount_world(n_docs=800)
    tgt.size_class = "large"
    bundle = freeze(index, None, K=300, device="cpu")
    assert mount(bundle, index, tgt, device="cpu").metrics["doc_pairs"] > 0


def test_source_free_bridge_round_trip(tmp_path):
    _, index, _, tgt = make_mount_world(n_docs=800)
    bridge = mount(freeze(index, None, K=300, device="cpu"), index, tgt, device="cpu")
    bridge.save(tmp_path / "b")
    loaded = SpaceBridge.load(tmp_path / "b", device="cpu")
    Y = tgt.encode_queries(["question 3"])
    np.testing.assert_array_equal(bridge.transform_query(Y), loaded.transform_query(Y))
    assert loaded.quality.score is None and loaded.status == "NO_TEACHER"
