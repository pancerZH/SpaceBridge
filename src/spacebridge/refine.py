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

    with torch.no_grad():
        teacher = F.log_softmax(Xq_train @ Xd_train.T / cfg.tau, dim=-1)
    if labels is None:
        labels = torch.arange(K_train, device=W0.device)

    W = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))

    best_score, best_step, best_W = val_score(W0), 0, W0.clone()
    curve = [(0, best_score)]
    for step in range(1, cfg.steps + 1):
        q = F.normalize(mu_x + Yc_train @ W, dim=-1)
        log_p = F.log_softmax(q @ Xd_train.T / cfg.tau, dim=-1)
        ce = F.nll_loss(log_p, labels)
        kl = F.kl_div(log_p, teacher, log_target=True, reduction="batchmean")
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
