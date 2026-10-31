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
