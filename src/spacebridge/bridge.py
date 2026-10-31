"""The SpaceBridge query-side alignment engine (design §4.2–§6)."""

import time
import warnings
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bundle import CalibrationBundle
from .refine import RefineConfig, refine
from .solvers import LAMBDAS_REL, naive_procrustes, ridge_path, stiefel_procrustes

Mode = Literal["naive", "stiefel", "ridge", "stage1", "auto"]
Pairs = Literal["query", "doc"]

HIGH_CONFIDENCE = 0.90
MODERATE_DRIFT = 0.75
VAL_K = 10


def topk_agreement(Q: torch.Tensor, pool: torch.Tensor, teacher_top: torch.Tensor) -> float:
    """Mean overlap@k between each query's top-k in pool and the teacher's top-k ids (k = teacher_top width)."""
    top = (Q @ pool.T).topk(teacher_top.shape[1], dim=1).indices
    return (top[:, :, None] == teacher_top[:, None, :]).any(-1).float().mean().item()


def retrieval_stats(Q: torch.Tensor, pool: torch.Tensor, targets: torch.Tensor, k: int = VAL_K) -> dict:
    """Recall@k and MRR@k when each query in Q has exactly one relevant row ``targets[i]`` in pool."""
    scores = Q @ pool.T
    target_scores = scores.gather(1, targets[:, None])
    rank = (scores > target_scores).sum(1)
    hit = rank < k
    return {
        "recall": hit.float().mean().item(),
        "mrr": torch.where(hit, 1.0 / (rank + 1).float(), torch.zeros_like(rank, dtype=torch.float)).mean().item(),
    }


class SpaceBridge:
    def __init__(
        self,
        device: str = "cuda",
        restore_source_mean: bool = True,
        refine_config: Optional[RefineConfig] = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.restore_source_mean = restore_source_mean  # False reproduces v2.3 behavior (Ablation Row 3b)
        self.refine_config = refine_config or RefineConfig()
        self.src_lang: str = "en"
        self.mu_x_query: Optional[np.ndarray] = None
        self.mu_y_by_lang: Dict[str, np.ndarray] = {}
        self.W: Optional[np.ndarray] = None
        self.status: str = "uninitialized"
        self.metrics: Dict[str, float] = {}

    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def fit(
        self,
        bundle: CalibrationBundle,
        Y_raw: np.ndarray,
        mode: Mode = "auto",
        pairs: Pairs = "query",
        X_distractor: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fits the affine query map and computes rho_val / status.

        Args:
            Y_raw: (K, d_t) target embeddings of the anchors: of ``bundle.anchor_query_texts`` with
                the target query prompt when ``pairs="query"``, or of the anchor docs with the target
                doc prompt when ``pairs="doc"`` (doc-anchor ablation rows).
            mode: ``naive`` (uncentered full Procrustes), ``stiefel`` / ``ridge`` (one Stage-1 operator),
                ``stage1`` (best Stage-1 operator on val), ``auto`` (``stage1`` + Stage-2 refinement).
            X_distractor: index vectors at ``bundle.distractor_ids``, added to the rho_val pool.
        """
        if mode == "auto" and pairs != "query":
            raise ValueError("Stage-2 refinement needs query-proxy pairs")
        Y = self._t(Y_raw)
        Xq, Xd = self._t(bundle.X_query_raw), self._t(bundle.X_doc_raw)
        X_pairs = Xq if pairs == "query" else Xd
        tr = torch.as_tensor(bundle.train_idx, device=self.device)
        va = torch.as_tensor(bundle.val_idx, device=self.device)
        pool = Xd if X_distractor is None else torch.cat([Xd, self._t(X_distractor)])
        # Proxies trivially retrieve their own anchor doc, so validation scores agreement with the
        # native source ranking (teacher top-k over the pool) rather than own-doc recall.
        teacher_top = (Xq[va] @ pool.T).topk(min(VAL_K, pool.shape[0]), dim=1).indices

        def val_score(W: torch.Tensor) -> float:
            return topk_agreement(self._project(Y[va], mu_y, mu_x, W), pool, teacher_top)

        info: Dict[str, float] = {"mode": mode, "pairs": pairs}
        self._sync()
        t0 = time.perf_counter()
        if mode == "naive":
            mu_y = torch.zeros(Y.shape[1], device=self.device)
            mu_x = torch.zeros(X_pairs.shape[1], device=self.device)
            W, info["solver"] = naive_procrustes(Y[tr], X_pairs[tr]), "naive"
        else:
            mu_y, mu_x = Y[tr].mean(0), X_pairs[tr].mean(0)
            Yc, Xc = Y - mu_y, X_pairs - mu_x
            candidates = []
            if mode != "ridge":
                W_s, s_info = stiefel_procrustes(Yc[tr], Xc[tr])
                candidates.append((val_score(W_s), W_s, {"solver": "stiefel", "rank": s_info["rank"], "scale": s_info["scale"]}))
            if mode != "stiefel":
                for lam_rel, W_r in ridge_path(Yc[tr], Xc[tr], LAMBDAS_REL):
                    candidates.append((val_score(W_r), W_r, {"solver": "ridge", "lambda_rel": lam_rel}))
            _, W, chosen = max(candidates, key=lambda c: c[0])
            info.update(chosen)
        self._sync()
        info["stage1_ms"] = 1e3 * (time.perf_counter() - t0)

        if mode == "auto":
            t0 = time.perf_counter()
            mu_x_used = mu_x if self.restore_source_mean else torch.zeros_like(mu_x)
            W, r_info = refine(W, mu_x_used, Yc[tr], Xq[tr], Xd[tr], val_score, self.refine_config)
            self._sync()
            info["stage2_ms"] = 1e3 * (time.perf_counter() - t0)
            info["stage2_best_step"] = r_info["best_step"]

        self.src_lang = bundle.src_lang
        self.W = W.cpu().numpy()
        self.mu_x_query = mu_x.cpu().numpy()
        self.mu_y_by_lang = {self.src_lang: mu_y.cpu().numpy()}

        t0 = time.perf_counter()
        Q_val = self._project(Y[va], mu_y, mu_x, W)
        rho = topk_agreement(Q_val, pool, teacher_top)
        self._sync()
        info["val_ms"] = 1e3 * (time.perf_counter() - t0)
        student, teacher = retrieval_stats(Q_val, pool, va), retrieval_stats(Xq[va], pool, va)
        info.update(
            rho_val=rho,
            val_owndoc_recall_student=student["recall"],
            val_owndoc_recall_teacher=teacher["recall"],
            val_pool_size=int(pool.shape[0]),
        )
        self.status = (
            "HIGH_CONFIDENCE" if rho >= HIGH_CONFIDENCE
            else "MODERATE_DRIFT" if rho >= MODERATE_DRIFT
            else "LOW_ALIGNMENT_WARN"
        )
        info["status"] = self.status
        self.metrics = info
        return info

    def _project(self, Y: torch.Tensor, mu_y: torch.Tensor, mu_x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        q = (Y - mu_y) @ W
        if self.restore_source_mean:
            q = q + mu_x
        return F.normalize(q, dim=-1)

    def register_query_language(
        self, lang: str, Y_flores_src_lang: np.ndarray, Y_flores_query_lang: np.ndarray
    ) -> None:
        """Registers the language-cone shift from M_tgt query-prompt embeddings of the same
        256 FLORES dev sentences in src_lang and in lang (<80ms)."""
        if self.src_lang not in self.mu_y_by_lang:
            raise RuntimeError("fit() must be called before register_query_language()")
        shift = Y_flores_query_lang.mean(axis=0) - Y_flores_src_lang.mean(axis=0)
        self.mu_y_by_lang[lang] = self.mu_y_by_lang[self.src_lang] + shift

    def transform_query(self, q_tgt_raw: np.ndarray, query_lang: Optional[str] = None) -> np.ndarray:
        """Applies language-aware centering, linear projection W, source-mean restore, and L2 norm."""
        if self.W is None:
            raise RuntimeError("fit() must be called before transform_query()")
        lang = query_lang or self.src_lang
        mu_y = self.mu_y_by_lang.get(lang)
        if mu_y is None:
            warnings.warn(
                f"query language {lang!r} not registered; using {self.src_lang!r} centering "
                "without language-cone correction"
            )
            mu_y = self.mu_y_by_lang[self.src_lang]
        q_proj = (q_tgt_raw - mu_y) @ self.W
        if self.restore_source_mean:
            q_proj = q_proj + self.mu_x_query
        return q_proj / (np.linalg.norm(q_proj, axis=-1, keepdims=True) + 1e-8)
