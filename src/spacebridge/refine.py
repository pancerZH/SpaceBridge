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
    # True: return a checkpoint chosen by the stop rule. It doesn't end the loop early: all steps always run, so it saves no time.
    # False: return the final W (validation is only logged).
    early_stop: bool = True
    # Checkpoint selection. "floor" (M2): the checkpoint with the best held-out own-doc MRR among those keeping
    # min_agree_ratio of step 0's teacher agree@10. "tail" (M4): the average of the checkpoints from step
    # tail_from on, if its own agree@10 clears the gap floor a0 - floor_gap * (1 - a0) (a0: step 0's agree@10, so
    # the allowed loss scales with Stage 1's disagreement with the teacher and stays tight near the identity);
    # otherwise the "floor" choice under the gap floor. "tail" is the default since M4: vs. "floor" (3 seeds, both K)
    # it unblocks T2 with an mE5 index (+3.3 points, seed-to-seed s.d. 3.5 -> 0.8), +0.5 on trec-covid (s.d. 2.5 ->
    # 1.7), 0.0 / -0.1 on xquad-zh / fiqa; held out: hotpotqa +0.0, scidocs -0.0, fiqa with an mE5 index +0.4.
    stop_rule: str = "tail"
    min_agree_ratio: float = 0.9
    floor_gap: float = 0.25
    tail_from: int = 20
    keep_checkpoints: bool = False  # studies: also return every checkpoint's W (info["checkpoints"])
    # Stage 2 is full-batch (rows x candidates logits). With more candidates than this, each step trains on
    # a random subset of this many candidates and the rows whose positive is in it, so the per-step cost stays
    # at the default K = 2,048 level (1,638 training anchors). At K = 4,096 this takes Row 8b from 454 to 199 ms
    # (fiqa) at -0.4 / +0.1 points Recall@30 (fiqa / scidocs, 3 seeds). None: always full batch.
    batch_candidates: Optional[int] = 1638
    # The per-step cost is rows x candidates x d_s with rows proportional to candidates, so for indexes wider than
    # batch_ref_dim the cap shrinks by sqrt(batch_ref_dim / d_s) to keep the cost of a 1024-d index: 1,035 candidates
    # into the 2560-d Qwen3-4B index (M4: -0.1 / -0.4 points at K = 2,048 / 4,096, 3 seeds; a cap scaled by
    # batch_ref_dim / d_s, 655 candidates, lost 1.2 points at K = 4,096).
    batch_ref_dim: int = 1024

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"RefineConfig.steps must be at least 1, got {self.steps}")
        if self.eval_every < 1:
            raise ValueError(f"RefineConfig.eval_every must be at least 1, got {self.eval_every}")
        if self.stop_rule not in ("floor", "tail"):
            raise ValueError(f"RefineConfig.stop_rule must be 'floor' or 'tail', got {self.stop_rule!r}")


def agree_floor(a0: float, cfg: RefineConfig) -> float:
    """Lowest teacher agree@10 a Stage-2 candidate may have, given step 0's ``a0``."""
    if cfg.stop_rule == "floor":
        return cfg.min_agree_ratio * a0
    if cfg.stop_rule == "tail":
        return a0 - cfg.floor_gap * (1 - a0)
    raise ValueError(f"unknown Stage-2 stop rule {cfg.stop_rule!r}")


def floor_scores(curve: Sequence[tuple[int, dict]], floor: float) -> list[float]:
    """Own-doc MRR of each checkpoint, or -inf when its agree@10 is below ``floor``."""
    return [s["mrr"] if s["agree"] >= floor else -math.inf for _, s in curve]


def select_checkpoint(curve: Sequence[tuple[int, dict]], cfg: RefineConfig) -> int:
    """Index into ``curve`` (``(step, signals)`` with step 0 first) of the best-MRR checkpoint above the floor."""
    scores = floor_scores(curve, agree_floor(curve[0][1]["agree"], cfg))
    return max(range(len(scores)), key=lambda i: (scores[i], -i))  # first best: ties keep the earlier step


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
    n_cand, dev = Xd_train.shape[0], W0.device
    # The penalty follows the anchor count: doc pairs double the rows, but those rows are related, not twice the information.
    beta = cfg.beta if cfg.beta is not None else (1e-2 if d_t > n_cand else 1e-3)
    mu_x = mu_x.expand(K_train, -1) if mu_x.dim() == 1 else mu_x
    if labels is None:
        labels = torch.arange(K_train, device=dev)
    cap = None
    if cfg.batch_candidates:
        cap = int(cfg.batch_candidates * min(1.0, math.sqrt(cfg.batch_ref_dim / Xd_train.shape[1])))
    batch = cap if cap is not None and n_cand > cap else None
    with torch.no_grad():
        # In blocks: rows x candidates fp32 is 5.5 GB at K = 32,768 with doc pairs.
        sd = torch.cat([(Xq_train[s : s + 1024] @ Xd_train.T).std(dim=1) for s in range(0, Xq_train.shape[0], 1024)]).mean().item()
        tau = cfg.tau_scale * sd if cfg.tau_scale is not None and sd < cfg.tau_scale_below_sd else cfg.tau
        teacher = F.log_softmax(Xq_train @ Xd_train.T / tau, dim=-1) if batch is None else None
    gen = torch.Generator(device=dev).manual_seed(0)

    W = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))

    curve, Ws = [(0, val_signals(W0))], [W0.clone()]
    for step in range(1, cfg.steps + 1):
        mu_b, Y_b, cand, lab, t = mu_x, Yc_train, Xd_train, labels, teacher
        if batch is not None:
            # This step sees only some candidates, so it keeps only the rows whose positive is among them, and each label
            # becomes its positive's position in the sample: slot[d] is candidate d's position, or -1 if it wasn't drawn.
            sel = torch.randperm(n_cand, generator=gen, device=dev)[:batch]
            slot = torch.full((n_cand,), -1, dtype=torch.long, device=dev)
            slot[sel] = torch.arange(batch, device=dev)
            rows = (slot[labels] >= 0).nonzero().squeeze(1)
            mu_b, Y_b, cand, lab = mu_x[rows], Yc_train[rows], Xd_train[sel], slot[labels[rows]]
            t = F.log_softmax(Xq_train[rows] @ cand.T / tau, dim=-1)
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

    floor = agree_floor(curve[0][1]["agree"], cfg)
    best = select_checkpoint(curve, cfg) if cfg.early_stop else len(curve) - 1
    W_out, best_step, best_sig, averaged = Ws[best], curve[best][0], curve[best][1], False
    if cfg.early_stop and cfg.stop_rule == "tail":
        tail = [Wk for (s, _), Wk in zip(curve, Ws) if s >= cfg.tail_from]
        if tail:
            W_tail = torch.stack(tail).mean(0)
            with torch.no_grad():
                tail_sig = val_signals(W_tail)
            if tail_sig["agree"] >= floor:  # reported as the last averaged step
                W_out, best_step, best_sig, averaged = W_tail, curve[-1][0], tail_sig, True
    rejected = floor_scores(curve, floor)[1:]
    info = {
        "best_step": best_step, "averaged": averaged, "val_curve": curve, "beta": beta, "steps_run": step, "tau": tau,
        "teacher_sd": sd, "tau_scaled": tau != cfg.tau, "batch": batch,
        "mrr_gain": best_sig["mrr"] - curve[0][1]["mrr"],
        "floor_rejected": sum(v == -math.inf for v in rejected) / max(len(rejected), 1),
    }
    if cfg.keep_checkpoints:
        info["checkpoints"] = list(zip((s for s, _ in curve), Ws))
    return W_out, info
