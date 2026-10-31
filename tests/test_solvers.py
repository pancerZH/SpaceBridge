import numpy as np
import pytest
import torch

from spacebridge.solvers import naive_procrustes, ridge, ridge_lambda, ridge_path, stiefel_procrustes


def _rand(*shape, seed=0):
    return torch.as_tensor(np.random.default_rng(seed).standard_normal(shape), dtype=torch.float64)


@pytest.mark.parametrize("K,d_t", [(80, 50), (50, 80)])
def test_primal_and_dual_ridge_agree(K, d_t):
    Yc, Xc = _rand(K, d_t), _rand(K, 30, seed=1)
    lam = ridge_lambda(Yc, 0.1)
    W_p = ridge(Yc, Xc, lam, form="primal")
    W_d = ridge(Yc, Xc, lam, form="dual")
    torch.testing.assert_close(W_p, W_d)
    (_, W_path), = ridge_path(Yc, Xc, (0.1,))
    torch.testing.assert_close(W_path, W_p)


def test_stiefel_recovers_scaled_rotation():
    Yc = _rand(200, 16)
    R, _ = torch.linalg.qr(_rand(16, 16, seed=1))
    W, info = stiefel_procrustes(Yc, 0.7 * Yc @ R)
    torch.testing.assert_close(W, 0.7 * R)
    assert info["rank"] == 16
    assert info["scale"] == pytest.approx(0.7)


def test_stiefel_truncates_nullspace_when_rank_deficient():
    Yc, Xc = _rand(20, 64), _rand(20, 48, seed=1)
    W, info = stiefel_procrustes(Yc, Xc)
    assert info["rank"] <= 20
    assert torch.linalg.matrix_rank(W) == info["rank"]


def test_naive_procrustes_is_semi_orthogonal():
    W = naive_procrustes(_rand(100, 24), _rand(100, 32, seed=1))
    torch.testing.assert_close(W @ W.T, torch.eye(24, dtype=W.dtype))


def test_subtracting_doc_mean_preserves_ranking():
    rng = np.random.default_rng(0)
    E = rng.standard_normal((500, 32))
    q = rng.standard_normal(32)
    mu_d = E.mean(0)
    assert np.array_equal(np.argsort(-(E @ q)), np.argsort(-((E - mu_d) @ q)))
