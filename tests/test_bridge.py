import dataclasses

import numpy as np
import pytest

from spacebridge import RefineConfig, SpaceBridge

from conftest import l2, make_world, overlap_at_k, top_k


def _fit(world, Y, **kwargs):
    fit_kwargs = {k: kwargs.pop(k) for k in ("mode", "pairs") if k in kwargs}
    bridge = SpaceBridge(device="cpu", **kwargs)
    info = bridge.fit(world.bundle, Y, X_distractor=world.E_src[world.bundle.distractor_ids], **fit_kwargs)
    return bridge, info


def test_identity_reproduces_native_retrieval(world):
    bridge, info = _fit(world, world.bundle.X_query_raw, mode="stage1")
    native = top_k(world.Q_src, world.E_src)
    bridged = top_k(bridge.transform_query(world.Q_src), world.E_src)
    assert overlap_at_k(native, bridged) > 0.97
    assert info["rho_val"] > 0.97


def test_cross_model_alignment_recovers_most_native_retrieval(world):
    bridge, info = _fit(world, world.Y_query, mode="stage1")
    native = top_k(world.Q_src, world.E_src)
    bridged = top_k(bridge.transform_query(world.Q_tgt), world.E_src)
    assert overlap_at_k(native, bridged) > 0.6
    assert info["rho_val"] > 0.8


def test_stage2_never_lowers_validation(world):
    _, stage1 = _fit(world, world.Y_query, mode="stage1")
    _, auto = _fit(world, world.Y_query, mode="auto", refine_config=RefineConfig(steps=30))
    assert auto["rho_val"] >= stage1["rho_val"] - 1e-9
    assert "stage2_ms" in auto


def test_mean_restore_tracks_native_rankings_under_a_strong_query_cone():
    world = make_world(seed=1)
    cone = np.random.default_rng(1).standard_normal(world.Q_src.shape[1])
    cone /= np.linalg.norm(cone)  # unit-norm offset, same scale as the embeddings themselves
    bundle = dataclasses.replace(world.bundle, X_query_raw=l2(world.bundle.X_query_raw + cone))
    world = dataclasses.replace(world, bundle=bundle, Q_src=l2(world.Q_src + cone))

    native = top_k(world.Q_src, world.E_src)
    overlaps = {}
    for restore in (True, False):
        bridge, _ = _fit(world, world.Y_query, mode="stage1", restore_source_mean=restore)
        overlaps[restore] = overlap_at_k(native, top_k(bridge.transform_query(world.Q_tgt), world.E_src))
    assert overlaps[True] > overlaps[False]


def test_naive_mode_runs_and_reports_status(world):
    _, info = _fit(world, world.Y_query, mode="naive")
    assert info["status"] in {"HIGH_CONFIDENCE", "MODERATE_DRIFT", "LOW_ALIGNMENT_WARN"}


def test_auto_requires_query_pairs(world):
    with pytest.raises(ValueError):
        _fit(world, world.Y_doc, mode="auto", pairs="doc")


def test_language_registration_and_fallback(world):
    bridge = SpaceBridge(device="cpu")
    with pytest.raises(RuntimeError):
        bridge.register_query_language("zh", world.Y_query, world.Y_query)
    bridge.fit(world.bundle, world.Y_query, mode="ridge")

    shift = np.full(world.Y_query.shape[1], 0.1)
    bridge.register_query_language("zh", world.Y_query, world.Y_query + shift)
    np.testing.assert_allclose(bridge.mu_y_by_lang["zh"] - bridge.mu_y_by_lang["en"], shift, rtol=1e-5)

    q_zh = world.Q_tgt[:5] + shift
    np.testing.assert_allclose(
        bridge.transform_query(q_zh, query_lang="zh"), bridge.transform_query(world.Q_tgt[:5]), atol=1e-5
    )
    with pytest.warns(UserWarning, match="not registered"):
        bridge.transform_query(world.Q_tgt[:5], query_lang="ja")
