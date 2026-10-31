"""Stage-2 contrastive-distillation refinement (design §4.3, Operator C)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class RefineConfig:
    steps: int = 50  # gains plateau by step 20-50 (M2 P4 study); the cosine schedule spans all steps
    lr: float = 3e-3
    tau: float = 0.05
    # Compressed-score indexes: when the mean per-row s.d. of the teacher's cosine scores is below
    # tau_scale_below_sd, tau = tau_scale * that s.d. multilingual-e5 indexes have s.d. 0.020-0.027 (every other
    # index measured: 0.057-0.092), so tau = 0.05 is a ~3x softer softmax there and every Stage-2 step failed the
    # agreement floor (T2Retrieval Row 8b 0.684 -> 0.879 with the rule; held-out fiqa +2 to +5 points). Applied to
    # all indexes, tau ~= 0.036 moved the identity map on trec-covid, hence the guard. tau_scale=None disables it.
    tau_scale: Optional[float] = 0.6
    tau_scale_below_sd: float = 0.04
    alpha: float = 0.5
    # Student and teacher share temperature tau, so the Hinton tau^2 factor would shrink the KL
    # term ~400x relative to InfoNCE; 1.0 keeps the two terms balanced (tau**2 reproduces §4.3).
    kl_scale: float = 1.0
    beta: Optional[float] = None  # None: 1e-3, or 1e-2 when d_t > K_train
    eval_every: int = 10
    early_stop: bool = True  # False: run all steps and return the final W (validation is only logged)
    # Checkpoint selection (``select_checkpoint``): "floor" keeps the checkpoint with the best held-out own-doc
    # MRR among those keeping min_agree_ratio of step 0's teacher agree@10 (the M2 rule).
    stop_rule: str = "floor"
    min_agree_ratio: float = 0.9
    keep_checkpoints: bool = False  # studies: also return every checkpoint's W (info["checkpoints"])
    # Stage 2 is full-batch (rows x candidates logits). With more candidates than this, each step trains on
    # a random subset of this many candidates and the rows whose positive is in it, so the per-step cost stays
    # at the default K = 2,048 level (1,638 training anchors). At K = 4,096 this takes Row 8b from 454 to 199 ms
    # (fiqa) at -0.4 / +0.1 points Recall@30 (fiqa / scidocs, 3 seeds). None: always full batch.
    batch_candidates: Optional[int] = 1638


def floor_scores(curve: Sequence[tuple[int, dict]], min_agree_ratio: float) -> list[float]:
    """Own-doc MRR of each checkpoint, or -inf when it keeps less than ``min_agree_ratio`` of step 0's agree@10."""
    floor = min_agree_ratio * curve[0][1]["agree"]
    return [s["mrr"] if s["agree"] >= floor else -math.inf for _, s in curve]


def select_checkpoint(curve: Sequence[tuple[int, dict]], cfg: RefineConfig) -> int:
    """Index into ``curve`` (``(step, signals)`` with step 0 first) of the checkpoint Stage 2 returns."""
    if cfg.stop_rule == "floor":
        scores = floor_scores(curve, cfg.min_agree_ratio)
        return max(range(len(scores)), key=lambda i: (scores[i], -i))  # first best: ties keep the earlier step
    raise ValueError(f"unknown Stage-2 stop rule {cfg.stop_rule!r}")


def refine(
    W0: torch.Tensor,
    mu_x: torch.Tensor,
    Yc_train: torch.Tensor,
    Xq_train: torch.Tensor,
    Xd_train: torch.Tensor,
    val_signals: Callable[[torch.Tensor], dict],
    cfg: RefineConfig = RefineConfig(),
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """Optimizes W against anchor docs with the deployed affine map; returns the selected checkpoint's W.

    Row i of ``Yc_train`` / ``Xq_train`` (and of ``mu_x`` when it is 2-D) is a training query whose
    positive candidate is ``Xd_train[labels[i]]``; ``labels`` defaults to the diagonal.
    ``val_signals`` maps a candidate W to label-free validation signals (``agree``: teacher agree@10,
    ``mrr``: own-doc MRR, on held-out proxies); every ``eval_every`` steps a checkpoint is scored and
    ``select_checkpoint`` picks one from the whole curve. Step 0 (W0) is a candidate.
    """
    K_train, d_t = Yc_train.shape
    beta = cfg.beta if cfg.beta is not None else (1e-2 if d_t > K_train else 1e-3)
    n_cand, dev = Xd_train.shape[0], W0.device
    mu_x = mu_x.expand(K_train, -1) if mu_x.dim() == 1 else mu_x
    if labels is None:
        labels = torch.arange(K_train, device=dev)
    batch = cfg.batch_candidates if cfg.batch_candidates and n_cand > cfg.batch_candidates else None
    with torch.no_grad():
        teacher_scores = Xq_train @ Xd_train.T
        sd = teacher_scores.std(dim=1).mean().item()
        tau = cfg.tau_scale * sd if cfg.tau_scale is not None and sd < cfg.tau_scale_below_sd else cfg.tau
        teacher_logits = teacher_scores / tau
        teacher = F.log_softmax(teacher_logits, dim=-1) if batch is None else None
    gen = torch.Generator(device=dev).manual_seed(0)

    W = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))

    curve, Ws = [(0, val_signals(W0))], [W0.clone()]
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
        log_p = F.log_softmax(q @ cand.T / tau, dim=-1)
        ce = F.nll_loss(log_p, lab)
        kl = F.kl_div(log_p, t, log_target=True, reduction="batchmean")
        loss = (1 - cfg.alpha) * ce + cfg.alpha * cfg.kl_scale * kl + beta * (W - W0).pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                curve.append((step, val_signals(W.detach())))
            Ws.append(W.detach().clone())

    best = select_checkpoint(curve, cfg) if cfg.early_stop else len(curve) - 1
    floor = floor_scores(curve, cfg.min_agree_ratio)
    info = {
        "best_step": curve[best][0], "val_curve": curve, "beta": beta, "steps_run": step, "tau": tau,
        "teacher_sd": sd, "tau_scaled": tau != cfg.tau,
        "mrr_gain": curve[best][1]["mrr"] - curve[0][1]["mrr"],
        "floor_rejected": sum(v == -math.inf for v in floor[1:]) / max(len(floor) - 1, 1),
    }
    if cfg.keep_checkpoints:
        info["checkpoints"] = list(zip((s for s, _ in curve), Ws))
    return Ws[best], info
