"""Stage-1 closed-form solvers (design §4.3, Operators A and B).

All solvers map centered target rows to centered source rows: ``Yc @ W ≈ Xc``.
"""

from typing import Literal

import torch

LAMBDAS_REL = (0.01, 0.1, 1.0, 5.0, 20.0)


def _svd(M: torch.Tensor):
    # The default cuSOLVER driver failed to converge on some of these cross-covariances and silently fell back to a ~10x
    # slower path. gesvd is slower per call and avoided that in these studies; a failure raises torch.linalg.LinAlgError.
    return torch.linalg.svd(M, full_matrices=False, driver="gesvd" if M.is_cuda else None)


def naive_procrustes(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Uncentered, untruncated, unscaled Procrustes (Ablation Row 2 baseline)."""
    U, _, Vh = _svd(Y.T @ X)
    return U @ Vh


def stiefel_procrustes(Yc: torch.Tensor, Xc: torch.Tensor, rel_tol: float = 1e-3, objective_scale: bool = False) -> tuple[torch.Tensor, dict]:
    """Operator A: scaled, spectrally truncated partial isometry ``s * U_r V_r^T``.

    The default scale, ``sum(sigma_r) / ||Yc||^2``, is a heuristic: it is the least-squares scale only when the retained
    modes span the target space. ``objective_scale`` gives the scale that minimizes ``||s Yc U_r V_r^T - Xc||_F`` for this
    map, ``sum(sigma_r) / ||Yc U_r||^2``. The default pipeline rescales W to the native residual norm afterwards, which cancels
    either scalar, so this only matters with ``match_residual_norm=False``.

    Computed as a polar decomposition from a float64 eigendecomposition of the smaller Gram matrix
    of ``M = Yc^T Xc``: identical to the SVD form (to ~1e-5) but several times faster than gesvd on
    GPU. float64 matters: kept modes reach sigma_r / sigma_1 = 1e-3, i.e. eigenvalue ratios of 1e-6.
    """
    M = (Yc.T @ Xc).double()
    left = M.shape[0] < M.shape[1]  # eigendecompose M M^T (d_t x d_t) when d_t < d_s
    evals, Q = torch.linalg.eigh(M @ M.T if left else M.T @ M)
    S, Q = evals.flip(0).clamp_min(0).sqrt(), Q.flip(1)
    if not (torch.isfinite(S).all() and S[0] > 0):
        raise ValueError("the calibration pairs have no usable cross-covariance: the embeddings are constant, identical or not finite")
    r = int((S >= rel_tol * S[0]).sum())
    if left:
        U, V = Q[:, :r], (M.T @ Q[:, :r]) / S[:r]
    else:
        U, V = (M @ Q[:, :r]) / S[:r], Q[:, :r]
    s = S[:r].sum() / ((Yc.double() @ U).pow(2).sum() if objective_scale else Yc.double().pow(2).sum())
    W = (s * (U @ V.T)).to(Yc.dtype)
    return W, {"rank": r, "scale": float(s), "singular_values": S.float().cpu()}


def ridge_lambda(Yc: torch.Tensor, lam_rel: float) -> float:
    """Absolute ridge penalty: ``lam_rel`` times a nominal mean eigenvalue, the trace over min(d_t, K). That is a chosen
    normalization, not the mean nonzero eigenvalue: centering and duplicate anchors can leave the numerical rank lower."""
    K, d_t = Yc.shape
    return lam_rel * float(Yc.pow(2).sum()) / min(d_t, K)


def ridge(
    Yc: torch.Tensor, Xc: torch.Tensor, lam: float, form: Literal["auto", "primal", "dual"] = "auto"
) -> torch.Tensor:
    """Operator B by direct solve; ``auto`` inverts the smaller of d_t x d_t and K x K."""
    K, d_t = Yc.shape
    if form == "auto":
        form = "primal" if d_t <= K else "dual"
    if form == "primal":
        A = Yc.T @ Yc + lam * torch.eye(d_t, dtype=Yc.dtype, device=Yc.device)
        return torch.linalg.solve(A, Yc.T @ Xc)
    G = Yc @ Yc.T + lam * torch.eye(K, dtype=Yc.dtype, device=Yc.device)
    return Yc.T @ torch.linalg.solve(G, Xc)


def ridge_path(
    Yc: torch.Tensor, Xc: torch.Tensor, lambdas_rel: tuple[float, ...] = LAMBDAS_REL
) -> list[tuple[float, torch.Tensor]]:
    """Operator B for a whole lambda grid from one eigendecomposition of the smaller Gram matrix."""
    K, d_t = Yc.shape
    if d_t <= K:
        evals, Q = torch.linalg.eigh(Yc.T @ Yc)
        B = Q.T @ (Yc.T @ Xc)
        solve = lambda lam: Q @ (B / (evals.clamp_min(0) + lam)[:, None])
    else:
        evals, Q = torch.linalg.eigh(Yc @ Yc.T)
        B = Q.T @ Xc
        solve = lambda lam: Yc.T @ (Q @ (B / (evals.clamp_min(0) + lam)[:, None]))
    return [(lam_rel, solve(ridge_lambda(Yc, lam_rel))) for lam_rel in lambdas_rel]
