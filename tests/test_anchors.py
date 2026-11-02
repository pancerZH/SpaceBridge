import numpy as np
import pytest

from spacebridge import CalibrationBundle, extract_query_proxy, make_split, sample_distractors, select_anchors
from spacebridge.anchors import count_tokens


@pytest.mark.parametrize("N,K", [(800, 50), (6000, 64)])  # farthest-point and k-means medoid paths
def test_select_anchors_returns_distinct_ids(N, K):
    E = np.random.default_rng(0).standard_normal((N, 16)).astype(np.float32)
    ids = select_anchors(E, K)
    assert ids.shape == (K,)
    assert len(set(ids.tolist())) == K
    assert ids.min() >= 0 and ids.max() < N


def test_select_anchors_random_is_a_seeded_uniform_sample():
    E = np.random.default_rng(0).standard_normal((6000, 8))
    ids = select_anchors(E, 64, seed=1, method="random")
    assert len(set(ids.tolist())) == 64
    np.testing.assert_array_equal(ids, select_anchors(E, 64, seed=1, method="random"))
    assert set(ids.tolist()) != set(select_anchors(E, 64, seed=2, method="random").tolist())
    with pytest.raises(ValueError):
        select_anchors(E, 64, method="nope")


def test_large_indexes_get_more_anchors_chosen_at_random():
    from spacebridge import LARGE_INDEX_DOCS, LARGE_PROFILE_DOCS, default_anchor_count, resolve_anchor_method

    assert default_anchor_count(1_000) == 1_000
    assert default_anchor_count(LARGE_PROFILE_DOCS) == 2_048
    assert default_anchor_count(LARGE_PROFILE_DOCS + 1) == 8_192
    assert default_anchor_count(LARGE_INDEX_DOCS) == 8_192  # the anchor switch comes before the reranker requirement
    assert resolve_anchor_method(LARGE_PROFILE_DOCS) == "kmeans"
    assert resolve_anchor_method(LARGE_PROFILE_DOCS + 1) == "random"
    assert resolve_anchor_method(LARGE_PROFILE_DOCS + 1, "kmeans") == "kmeans"


def test_select_anchors_rejects_oversized_k():
    with pytest.raises(ValueError):
        select_anchors(np.ones((10, 4)), 11)


def test_split_sizes_match_design():
    train, val = make_split(2048)
    assert (len(train), len(val)) == (1638, 410)
    assert not set(train) & set(val)


def test_distractors_exclude_anchors():
    d = sample_distractors(1000, np.arange(100), n=300)
    assert len(d) == 300 and d.min() >= 100
    assert len(sample_distractors(150, np.arange(100))) == 50


def test_query_proxy_title_lead_and_length_bounds():
    text = "Short one. " + " ".join(f"word{i}" for i in range(100)) + ". Third sentence."
    proxy = extract_query_proxy("A Title", text)
    assert proxy.startswith("A Title\nShort one.")
    assert 32 <= count_tokens(proxy) <= 64


def test_query_proxy_counts_cjk_characters():
    proxy = extract_query_proxy("", "这是第一句话。" + "这是很长的第二句话" * 10 + "。")
    assert proxy.startswith("这是第一句话。")
    assert count_tokens(proxy) <= 64


def test_bundle_roundtrip(tmp_path, world):
    world.bundle.save(tmp_path / "b")
    loaded = CalibrationBundle.load(tmp_path / "b")
    assert loaded.anchor_query_texts == world.bundle.anchor_query_texts
    np.testing.assert_array_equal(loaded.X_query_raw, world.bundle.X_query_raw)
    np.testing.assert_array_equal(loaded.mu_x_doc, world.bundle.mu_x_doc)


def test_query_proxy_is_the_title_when_it_is_long_enough():
    text = "Lead sentence of the abstract. " + " ".join(f"word{i}" for i in range(80))
    assert extract_query_proxy("Clinical features of a viral infection in adults", text) == "Clinical features of a viral infection in adults"
    long_title = " ".join(f"t{i}" for i in range(30))
    assert count_tokens(extract_query_proxy(long_title, text)) == 16
    assert extract_query_proxy("Meme", text).startswith("Meme\nLead sentence")  # short titles keep the lead
    assert extract_query_proxy("Clinical features of a viral infection", text, title_min_tokens=None).startswith(
        "Clinical features of a viral infection\nLead sentence")


def test_shared_titles_are_not_used_alone():
    from spacebridge import make_query_proxies

    titles = ["University of Notre Dame campus", "University of Notre Dame campus", "A unique paper title here", ""]
    texts = ["First paragraph. More.", "Second paragraph. More.", "Abstract text. More.", "No title here. More."]
    proxies = make_query_proxies(titles, texts)
    assert proxies[0].startswith("University of Notre Dame campus\nFirst paragraph.")
    assert proxies[1].startswith("University of Notre Dame campus\nSecond paragraph.")
    assert proxies[2] == "A unique paper title here"
    assert proxies[3].startswith("No title here.")
