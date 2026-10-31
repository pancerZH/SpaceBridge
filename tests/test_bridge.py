import dataclasses

import numpy as np
import pytest

from spacebridge import RefineConfig, SpaceBridge

from conftest import l2, make_world, overlap_at_k, top_k


def _fit(world, Y, **kwargs):
    fit_kwargs = {k: kwargs.pop(k) for k in ("mode", "pairs", "Y_doc_raw") if k in kwargs}
    kwargs.setdefault("doc_pairs", "Y_doc_raw" in fit_kwargs)  # most tests exercise query pairs alone
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


def test_stage2_never_lowers_its_stopping_signal(world):
    _, stage1 = _fit(world, world.Y_query, mode="stage1")
    _, auto = _fit(world, world.Y_query, mode="auto", refine_config=RefineConfig(steps=30))
    assert auto["val_metrics"]["owndoc_mrr"] >= stage1["val_metrics"]["owndoc_mrr"] - 1e-9
    assert "stage2_ms" in auto


def test_stage2_keeps_most_teacher_agreement_on_the_identity(world):
    _, stage1 = _fit(world, world.bundle.X_query_raw, mode="stage1")
    _, auto = _fit(world, world.bundle.X_query_raw, mode="auto", refine_config=RefineConfig(steps=30))
    assert auto["rho_val"] >= 0.9 * stage1["rho_val"] - 1e-9


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


def test_doc_pairs_augment_stage1_and_stage2(world):
    _, info = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc, refine_config=RefineConfig(steps=20))
    assert info["doc_pairs"] == len(world.bundle.train_idx)
    assert info["rho_val"] > 0.5
    with pytest.raises(ValueError):
        _fit(world, world.Y_query, mode="naive", Y_doc_raw=world.Y_doc)


def test_default_auto_fit_requires_doc_pairs(world):
    with pytest.raises(ValueError, match="Y_doc_raw"):
        SpaceBridge(device="cpu").fit(world.bundle, world.Y_query)
    SpaceBridge(device="cpu").fit(world.bundle, world.Y_query, mode="stage1")  # Stage 1 alone stays optional


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
    assert any("'zh'" in n for n in bridge.quality_report().notes)
    with pytest.warns(UserWarning, match="not registered"):
        bridge.transform_query(world.Q_tgt[:5], query_lang="ja")


def test_quality_report_measurements_and_scoring(world):
    from spacebridge.quality import QualityReport, score_report

    bridge, info = _fit(world, world.Y_query, mode="auto", refine_config=RefineConfig(steps=20))
    report = bridge.quality_report()
    assert set(report.details) == {"fidelity", "recoverability", "fit", "difficulty", "coverage", "stage2"}
    feats = report.features()
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["recoverability.nbr_recall@100"] >= feats["fidelity.nbr_recall@30"]
    assert 0 < feats["fit.rank_ratio"] <= 1

    calib = {"features": ["fidelity.agree@10"], "mean": [0.0], "scale": [1.0], "coef": [0.5],
             "intercept": 0.5, "resid_sd": 0.03}
    scored = score_report(QualityReport(report.details), calib)
    assert scored.score == pytest.approx(100 * (0.5 + 0.5 * feats["fidelity.agree@10"]))
    assert scored.band == pytest.approx((scored.score - 3, scored.score + 3))
    assert scored.grade in "ABCD"
    assert scored.rerank_pool == {"A": 60, "B": 60, "C": 100, "D": 100}[scored.grade]
    for intercept, pool in ((1.2, 60), (0.9, 60), (0.75, 100), (0.5, 100)):
        r = score_report(QualityReport(report.details), {**calib, "coef": [0.0], "intercept": intercept})
        assert r.rerank_pool == pool  # a high grade never shrinks the pool below the default
    assert QualityReport(report.details).rerank_pool == 60  # unscored reports keep the default


def test_per_query_confidence_is_bounded_and_lower_off_distribution(world):
    bridge, _ = _fit(world, world.Y_query, mode="stage1")
    q_hat, conf = bridge.transform_query(world.Q_tgt, return_confidence=True)
    np.testing.assert_allclose(q_hat, bridge.transform_query(world.Q_tgt))
    assert conf.shape == (len(world.Q_tgt),) and np.all((conf >= 0) & (conf <= 1))
    off = l2(np.random.default_rng(5).standard_normal(world.Q_tgt.shape))
    assert bridge.query_support(off).mean() < bridge.query_support(world.Q_tgt).mean() - 0.1


def test_stage1_candidates_can_be_restricted(world):
    _, info = _fit(world, world.Y_query, mode="stage1", stage1_operators=("ridge",), lambdas_rel=(5.0,))
    assert info["solver"] == "ridge" and info["lambda_rel"] == 5.0
    _, info = _fit(world, world.Y_query, mode="auto", stage1_operators=("stiefel",), refine_config=RefineConfig(steps=10))
    assert info["solver"] == "stiefel" and "stage2_ms" in info


def test_stage2_candidate_minibatches_match_full_batch(world):
    full, info_full = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc, refine_config=RefineConfig(steps=30))
    mini, info_mini = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc,
                           refine_config=RefineConfig(steps=30, batch_candidates=100))
    assert len(world.bundle.train_idx) > 100  # the mini-batch path is exercised
    a, b = top_k(full.transform_query(world.Q_tgt), world.E_src), top_k(mini.transform_query(world.Q_tgt), world.E_src)
    assert overlap_at_k(a, b) > 0.8
    assert info_mini["val_metrics"]["owndoc_mrr"] >= info_full["val_metrics"]["owndoc_mrr"] - 0.05


def test_stage2_temperature_follows_compressed_teacher_scores(world):
    _, default = _fit(world, world.Y_query, mode="auto", refine_config=RefineConfig(steps=10))
    Xq, Xd = world.bundle.X_query_raw[world.bundle.train_idx], world.bundle.X_doc_raw[world.bundle.train_idx]
    sd = (Xq @ Xd.T).std(axis=1, ddof=1).mean()
    assert sd > 0.04 and default["stage2_tau"] == pytest.approx(0.05)  # ordinary score spread: fixed tau
    _, scaled = _fit(world, world.Y_query, mode="auto",
                     refine_config=RefineConfig(steps=10, tau_scale_below_sd=float("inf")))
    assert scaled["stage2_tau"] == pytest.approx(0.6 * sd, rel=1e-3)


def test_stage2_health_flags_blocked_and_compressed_fits(world, monkeypatch):
    import spacebridge.bridge as bridge_mod

    cfg = RefineConfig(steps=20)
    healthy, info = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc, refine_config=cfg)
    assert info["stage2_best_step"] > 0 and not info["stage2_tau_scaled"]
    assert healthy.quality_report().details["stage2"]["health_flag"] == 0.0
    assert not any("stage 2" in n or "compressed" in n for n in healthy.quality_report().notes)

    monkeypatch.setattr(bridge_mod, "STAGE2_MIN_AGREE_RATIO", 2.0)  # the floor rejects every checkpoint
    blocked, info = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc, refine_config=cfg)
    h = blocked.quality_report().details["stage2"]
    assert info["stage2_best_step"] == 0 and info["stage2_floor_rejected"] == 1.0
    assert h["blocked"] == h["health_flag"] == 1.0 and h["compressed_scores"] == 0.0
    assert any("stage 2 blocked" in n for n in blocked.quality_report().notes)
    assert blocked.quality_report().rerank_pool == 100
    monkeypatch.undo()

    compressed, info = _fit(world, world.Y_query, mode="auto", Y_doc_raw=world.Y_doc,
                            refine_config=RefineConfig(steps=20, tau_scale_below_sd=float("inf")))
    h = compressed.quality_report().details["stage2"]
    assert info["stage2_tau_scaled"] and h["compressed_scores"] == h["health_flag"] == 1.0
    assert any("compressed index scores" in n for n in compressed.quality_report().notes)

    stage1, _ = _fit(world, world.Y_query, mode="stage1")
    assert "stage2" not in stage1.quality_report().details  # no Stage 2, no flag
