"""Stage-2 contrastive-distillation refinement (design §4.3, Operator C)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class RefineConfig:
    steps: int = 50  # gains plateau by step 20-50 (M2 P4 study); the cosine schedule spans all steps
    lr: float = 3e-3
    tau: float = 0.05
    alpha: float = 0.5
    # Student and teacher share temperature tau, so the Hinton tau^2 factor would shrink the KL
    # term ~400x relative to InfoNCE; 1.0 keeps the two terms balanced (tau**2 reproduces §4.3).
    kl_scale: float = 1.0
    beta: Optional[float] = None  # None: 1e-3, or 1e-2 when d_t > K_train
    eval_every: int = 10
    patience: int = 50  # stop after this many steps without a validation improvement
    early_stop: bool = True  # False: run all steps and return the final W (validation is only logged)
    # Stage 2 is full-batch (rows x candidates logits). With more candidates than this, each step trains on
    # a random subset of this many candidates and the rows whose positive is in it, so the per-step cost stays
    # at the default K = 2,048 level (1,638 training anchors). At K = 4,096 this takes Row 8b from 454 to 199 ms
    # (fiqa) at -0.4 / +0.1 points Recall@30 (fiqa / scidocs, 3 seeds). None: always full batch.
    batch_candidates: Optional[int] = 1638


def refine(
    W0: torch.Tensor,
    mu_x: torch.Tensor,
    Yc_train: torch.Tensor,
    Xq_train: torch.Tensor,
    Xd_train: torch.Tensor,
    val_score: Callable[[torch.Tensor], float],
    cfg: RefineConfig = RefineConfig(),
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Optimizes W against anchor docs with the deployed affine map; returns the best-on-val W.

    Row i of ``Yc_train`` / ``Xq_train`` (and of ``mu_x`` when it is 2-D) is a training query whose
    positive candidate is ``Xd_train[labels[i]]``; ``labels`` defaults to the diagonal.
    ``val_score`` maps a candidate W to a higher-is-better validation score; step 0 (W0) is a
    candidate, so refinement can never make validation worse.
    """
    K_train, d_t = Yc_train.shape
    beta = cfg.beta if cfg.beta is not None else (1e-2 if d_t > K_train else 1e-3)
    n_cand, dev = Xd_train.shape[0], W0.device
    mu_x = mu_x.expand(K_train, -1) if mu_x.dim() == 1 else mu_x
    if labels is None:
        labels = torch.arange(K_train, device=dev)
    batch = cfg.batch_candidates if cfg.batch_candidates and n_cand > cfg.batch_candidates else None
    with torch.no_grad():
        teacher_logits = Xq_train @ Xd_train.T / cfg.tau
        teacher = F.log_softmax(teacher_logits, dim=-1) if batch is None else None
    gen = torch.Generator(device=dev).manual_seed(0)

    W = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))

    best_score, best_step, best_W = val_score(W0), 0, W0.clone()
    curve = [(0, best_score)]
    for step in range(1, cfg.steps + 1):
        mu_b, Y_b, cand, lab, t = mu_x, Yc_train, Xd_train, labels, teacher
        if batch is not None:
            sel = torch.randperm(n_cand, generator=gen, device=dev)[:batch]
            slot = torch.full((n_cand,), -1, dtype=torch.long, device=dev)
            slot[sel] = torch.arange(batch, device=dev)
            rows = (slot[labels] >= 0).nonzero().squeeze(1)
            mu_b, Y_b, cand, lab = mu_x[rows], Yc_train[rows], Xd_train[sel], slot[labels[rows]]
            t = F.log_softmax(teacher_logits[rows][:, sel], dim=-1)
        q = F.normalize(mu_b + Y_b @ W, dim=-1)
        log_p = F.log_softmax(q @ cand.T / cfg.tau, dim=-1)
        ce = F.nll_loss(log_p, lab)
        kl = F.kl_div(log_p, t, log_target=True, reduction="batchmean")
        loss = (1 - cfg.alpha) * ce + cfg.alpha * cfg.kl_scale * kl + beta * (W - W0).pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                score = val_score(W.detach())
            curve.append((step, score))
            if not cfg.early_stop:
                continue
            if score > best_score:
                best_score, best_step, best_W = score, step, W.detach().clone()
            elif step - best_step >= cfg.patience:
                break

    if not cfg.early_stop:
        best_step, best_W, best_score = step, W.detach().clone(), curve[-1][1]

    return best_W, {"best_step": best_step, "best_val": best_score, "val_curve": curve, "beta": beta, "steps_run": step}
