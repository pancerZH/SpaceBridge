"""The SpaceBridge query-side alignment engine (design §4.2–§6)."""

import math
import time
import warnings
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from .bundle import CalibrationBundle
from .quality import QualityReport, load_calibration, score_report
from .refine import RefineConfig, refine
from .solvers import LAMBDAS_REL, naive_procrustes, ridge_path, stiefel_procrustes

Mode = Literal["naive", "stiefel", "ridge", "stage1", "auto"]
Pairs = Literal["query", "doc"]

# rho_val (teacher top-10 agreement) thresholds for ~90% / ~75% real Recall@30 retention, from a
# 180-fit calibration on scifact (retention ~= 0.85 * rho_val + 0.41, R^2 = 0.87).
HIGH_CONFIDENCE = 0.58
MODERATE_DRIFT = 0.40
VAL_K = 10
STAGE2_MIN_AGREE_RATIO = 0.9  # Stage-2 steps must keep this share of Stage 1's teacher agree@10
SUPPORT_K = 5


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


def validation_metrics(Q_student: torch.Tensor, Q_teacher: torch.Tensor, pool: torch.Tensor, own: torch.Tensor) -> dict:
    """Candidate rho_val signals for validation proxies whose own anchor doc is ``pool[own[i]]``.

    ``nbr_recall@k``: share of the teacher's top-10 *other* docs (own doc excluded, since proxies
    find it trivially) that the student ranks in its top-k. ``score_corr``: mean Pearson
    correlation of student and teacher scores over the whole pool. ``teacher_gap``: teacher score
    drop from the 1st to the 30th other doc (``_sd``: in units of the query's pool-score s.d.);
    small gaps mean crowded neighborhoods, where small query errors reorder many documents.
    """
    S, T = Q_student @ pool.T, Q_teacher @ pool.T
    rows = torch.arange(len(own), device=own.device)
    T_other = T.clone()
    T_other[rows, own] = -torch.inf
    nbrs = T_other.topk(VAL_K, dim=1).indices
    out = {"agree@10": topk_agreement(Q_student, pool, T.topk(VAL_K, dim=1).indices)}
    for k in (10, 30, 100):
        top = S.topk(min(k, pool.shape[0]), dim=1).indices
        out[f"nbr_recall@{k}"] = (nbrs[:, :, None] == top[:, None, :]).any(-1).float().mean().item()
    Sc, Tc = S - S.mean(1, keepdim=True), T - T.mean(1, keepdim=True)
    out["score_corr"] = ((Sc * Tc).sum(1) / (Sc.norm(dim=1) * Tc.norm(dim=1))).mean().item()
    top30 = T_other.topk(min(30, pool.shape[0] - 1), dim=1).values
    gap = top30[:, 0] - top30[:, -1]
    out["teacher_gap"] = gap.mean().item()
    out["teacher_gap_sd"] = (gap / T.std(dim=1)).mean().item()
    out["owndoc_mrr"] = retrieval_stats(Q_student, pool, own)["mrr"]
    return out


class SpaceBridge:
    def __init__(
        self,
        device: str = "cuda",
        restore_source_mean: bool = True,
        refine_config: Optional[RefineConfig] = None,
        match_residual_norm: bool = True,
        selection_pool: int = 10_000,
        doc_pairs: bool = True,
        stage1_operators: tuple[str, ...] = ("stiefel", "ridge"),
        lambdas_rel: tuple[float, ...] = LAMBDAS_REL,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.restore_source_mean = restore_source_mean  # False reproduces v2.3 behavior (Ablation Row 3b)
        # Least-squares maps shrink query residuals toward 0, so the restored mean dominates and
        # queries drift toward the average query; rescaling W to the native residual norm undoes it.
        self.match_residual_norm = match_residual_norm
        self.refine_config = refine_config or RefineConfig()
        self.selection_pool = selection_pool
        # Default mount (M2): ``auto`` fits also use doc<->doc pairs (Row 8b, best first stage on 5 of
        # 6 BEIR sets and on cross-lingual); False is the opt-out for ablation rows without them.
        self.doc_pairs = doc_pairs
        # Stage-1 candidates for the ``stage1`` / ``auto`` modes (restricting them is for selection studies).
        self.stage1_operators = stage1_operators
        self.lambdas_rel = lambdas_rel
        self.src_lang: str = "en"
        self.mu_x_query: Optional[np.ndarray] = None
        self.mu_y_by_lang: Dict[str, np.ndarray] = {}
        self.W: Optional[np.ndarray] = None
        self.status: str = "uninitialized"
        self.metrics: Dict[str, float] = {}
        self.quality: Optional[QualityReport] = None
        self._support_anchors: Optional[np.ndarray] = None  # (K_train, d_t) unit-norm centered target proxies
        self._conf_map: Optional[tuple[np.ndarray, np.ndarray]] = None  # isotonic support -> agreement knots

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
        Y_doc_raw: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fits the affine query map and computes rho_val / status.

        Args:
            Y_raw: (K, d_t) target embeddings of the anchors: of ``bundle.anchor_query_texts`` with
                the target query prompt when ``pairs="query"``, or of the anchor docs with the target
                doc prompt when ``pairs="doc"`` (doc-anchor ablation rows).
            mode: ``naive`` (uncentered full Procrustes), ``stiefel`` / ``ridge`` (one Stage-1 operator),
                ``stage1`` (best Stage-1 operator on val), ``auto`` (``stage1`` + Stage-2 refinement).
            X_distractor: index vectors at ``bundle.distractor_ids``, added to the rho_val pool.
            Y_doc_raw: optional (K, d_t) target doc embeddings of the anchor docs. Their training rows
                are added as doc<->doc pairs (each centered on its own doc-cone mean) to Stage 1 and 2;
                validation stays on query proxies.
        """
        if mode == "auto" and pairs != "query":
            raise ValueError("Stage-2 refinement needs query-proxy pairs")
        if Y_doc_raw is not None and (pairs != "query" or mode == "naive"):
            raise ValueError("doc<->doc pairs augment centered query-proxy fits only")
        if mode == "auto" and self.doc_pairs and Y_doc_raw is None:
            raise ValueError(
                "the default auto fit uses doc<->doc pairs: pass Y_doc_raw (anchor docs encoded with the "
                "target doc prompt), or construct SpaceBridge(doc_pairs=False)"
            )
        Y = self._t(Y_raw)
        Xq, Xd = self._t(bundle.X_query_raw), self._t(bundle.X_doc_raw)
        X_pairs = Xq if pairs == "query" else Xd
        tr = torch.as_tensor(bundle.train_idx, device=self.device)
        va = torch.as_tensor(bundle.val_idx, device=self.device)
        pool = Xd if X_distractor is None else torch.cat([Xd, self._t(X_distractor)])
        # Proxies trivially retrieve their own anchor doc, so validation scores agreement with the
        # native source ranking (teacher top-k over the pool) rather than own-doc recall.
        teacher_top = (Xq[va] @ pool.T).topk(min(VAL_K, pool.shape[0]), dim=1).indices
        # Model selection (Stage-1 candidates, Stage-2 early stopping) scores on a fixed subsample of
        # the pool: all anchor docs plus random distractors; the reported rho_val uses the full pool.
        sel_pool = pool
        if pool.shape[0] > self.selection_pool:
            gen = torch.Generator().manual_seed(0)
            extra = Xd.shape[0] + torch.randperm(pool.shape[0] - Xd.shape[0], generator=gen)
            keep = torch.cat([torch.arange(Xd.shape[0]), extra[: max(0, self.selection_pool - Xd.shape[0])]])
            sel_pool = pool[keep.to(self.device)]
        sel_teacher = (Xq[va] @ sel_pool.T).topk(min(VAL_K, sel_pool.shape[0]), dim=1).indices

        def val_score(W: torch.Tensor) -> float:
            return topk_agreement(self._project(Y[va], mu_y, mu_x, W), sel_pool, sel_teacher)

        def owndoc_score(W: torch.Tensor) -> float:  # anchor docs are the first rows of sel_pool
            # Own-doc MRR alone rewards pulling proxies onto their own docs (on the identity it trades
            # 20 points of agree@10 for +0.4 MRR), so steps that give up the source geometry are rejected.
            Q = self._project(Y[va], mu_y, mu_x, W)
            if topk_agreement(Q, sel_pool, sel_teacher) < STAGE2_MIN_AGREE_RATIO * stage1_agree:
                return -math.inf
            return retrieval_stats(Q, sel_pool, va)["mrr"]

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
            Y_fit, X_fit = Yc[tr], Xc[tr]
            if Y_doc_raw is not None:
                Yd = self._t(Y_doc_raw)[tr]
                mu_xd = Xd[tr].mean(0)
                Ydc, Xdc = Yd - Yd.mean(0), Xd[tr] - mu_xd
                Y_fit, X_fit = torch.cat([Y_fit, Ydc]), torch.cat([X_fit, Xdc])
                info["doc_pairs"] = len(tr)
            if self.match_residual_norm:
                target_norm = Xc[tr].norm(dim=1).mean()
                rescale = lambda W: W * (target_norm / (Yc[tr] @ W).norm(dim=1).mean())
            else:
                rescale = lambda W: W
            candidates = []
            ops = self.stage1_operators if mode in ("stage1", "auto") else (mode,)
            if "stiefel" in ops:
                W_s, s_info = stiefel_procrustes(Y_fit, X_fit)
                W_s = rescale(W_s)
                candidates.append((val_score(W_s), W_s, {"solver": "stiefel", "rank": s_info["rank"], "scale": s_info["scale"]}))
            if "ridge" in ops:
                for lam_rel, W_r in ridge_path(Y_fit, X_fit, self.lambdas_rel):
                    W_r = rescale(W_r)
                    candidates.append((val_score(W_r), W_r, {"solver": "ridge", "lambda_rel": lam_rel}))
            stage1_agree, W, chosen = max(candidates, key=lambda c: c[0])
            info.update(chosen)
        self._sync()
        info["stage1_ms"] = 1e3 * (time.perf_counter() - t0)

        if mode == "auto":
            t0 = time.perf_counter()
            n = len(tr)
            restore = 1.0 if self.restore_source_mean else 0.0
            mu_rows, Y_rows, teacher_rows = (restore * mu_x).expand(n, -1), Yc[tr], Xq[tr]
            labels = torch.arange(n, device=self.device)
            if Y_doc_raw is not None:
                mu_rows = torch.cat([mu_rows, (restore * mu_xd).expand(n, -1)])
                Y_rows, teacher_rows = torch.cat([Y_rows, Ydc]), torch.cat([teacher_rows, Xd[tr]])
                labels = torch.cat([labels, labels])
            # Stopping signal: held-out own-doc MRR under the agreement floor. Stage 2 lowers teacher agreement even
            # when it improves real retention, so stopping on agree@10 (the v2 signal) stopped it at step 0 (M2).
            W, r_info = refine(W, mu_rows, Y_rows, teacher_rows, Xd[tr], owndoc_score, self.refine_config, labels)
            self._sync()
            info["stage2_ms"] = 1e3 * (time.perf_counter() - t0)
            info["stage2_best_step"] = r_info["best_step"]
            info["stage2_steps_run"] = r_info["steps_run"]
            info["stage2_tau"] = r_info["tau"]
            info["stage2_gain"] = r_info["best_val"] - r_info["val_curve"][0][1]

        self.src_lang = bundle.src_lang
        self.W = W.cpu().numpy()
        self.mu_x_query = mu_x.cpu().numpy()
        self.mu_y_by_lang = {self.src_lang: mu_y.cpu().numpy()}

        t0 = time.perf_counter()
        Q_val = self._project(Y[va], mu_y, mu_x, W)
        rho = topk_agreement(Q_val, pool, teacher_top)
        self._sync()
        info["val_ms"] = 1e3 * (time.perf_counter() - t0)
        val_metrics = validation_metrics(Q_val, Xq[va], pool, va)
        info.update(rho_val=rho, val_pool_size=int(pool.shape[0]), val_metrics=val_metrics)
        if mode != "naive":
            self.quality = self._quality_report(
                val_metrics, Y, X_pairs, Xq, pool, tr, va, mu_y, mu_x, W, info.get("stage2_gain", 0.0), bundle
            )
            info["quality"] = self.quality.details
            self._fit_confidence(Y, mu_y, mu_x, W, Xq, pool, tr, va)
        self.status = (
            "HIGH_CONFIDENCE" if rho >= HIGH_CONFIDENCE
            else "MODERATE_DRIFT" if rho >= MODERATE_DRIFT
            else "LOW_ALIGNMENT_WARN"
        )
        info["status"] = self.status
        self.metrics = info
        return info

    def _quality_report(self, val_metrics, Y, X_pairs, Xq, pool, tr, va, mu_y, mu_x, W, stage2_gain, bundle) -> QualityReport:
        """Measurements of design §5.1 from the calibration data alone (no labeled queries)."""
        resid_true = X_pairs[va] - mu_x
        resid_pred = (Y[va] - mu_y) @ W
        tr_s = tr[: len(va)]
        teacher_tr = (Xq[tr_s] @ pool.T).topk(min(VAL_K, pool.shape[0]), dim=1).indices
        train_agree = topk_agreement(self._project(Y[tr_s], mu_y, mu_x, W), pool, teacher_tr)
        Yc = Y[tr] - mu_y
        # Participation ratio of the calibration inputs: how many target dimensions the anchors span.
        part_ratio = float(Yc.pow(2).sum() ** 2 / (Yc.T @ Yc).pow(2).sum())
        mu_doc = self._t(bundle.mu_x_doc) if bundle.mu_x_doc is not None else pool.mean(0)
        cone = F.cosine_similarity(Xq[tr].mean(0), mu_doc, dim=0).clamp(-1, 1)
        d_t, d_s = Y.shape[1], X_pairs.shape[1]
        details = {
            "fidelity": {"agree@10": val_metrics["agree@10"], "nbr_recall@30": val_metrics["nbr_recall@30"]},
            "recoverability": {"nbr_recall@100": val_metrics["nbr_recall@100"]},
            "fit": {
                "val_r2": float(1 - (resid_pred - resid_true).pow(2).sum() / resid_true.pow(2).sum()),
                "val_resid_cos": F.cosine_similarity(resid_pred, resid_true, dim=1).mean().item(),
                "train_val_gap": train_agree - val_metrics["agree@10"],
                "rank_ratio": part_ratio / min(d_t, d_s),
                "stage2_gain": float(stage2_gain),
            },
            "difficulty": {
                "teacher_gap": val_metrics["teacher_gap"],
                "teacher_gap_sd": val_metrics["teacher_gap_sd"],
                "cone_angle_deg": float(torch.rad2deg(torch.arccos(cone))),
            },
            "coverage": {"K_over_dt": len(bundle.anchor_ids) / d_t},
        }
        return score_report(QualityReport(details), load_calibration())

    def _fit_confidence(self, Y, mu_y, mu_x, W, Xq, pool, tr, va) -> None:
        """Per-query confidence: anchor support of the query in target space, mapped by an isotonic fit
        on validation proxies to their top-10 agreement with the native ranking."""
        anchors = F.normalize(Y[tr] - mu_y, dim=-1)
        self._support_anchors = anchors.cpu().numpy()
        support = self._support(Y[va] - mu_y, anchors)
        teacher = (Xq[va] @ pool.T).topk(VAL_K, dim=1).indices
        student = (self._project(Y[va], mu_y, mu_x, W) @ pool.T).topk(VAL_K, dim=1).indices
        agree = (student[:, :, None] == teacher[:, None, :]).any(-1).float().mean(1)
        iso = IsotonicRegression(out_of_bounds="clip").fit(support.cpu().numpy(), agree.cpu().numpy())
        self._conf_map = (iso.X_thresholds_, iso.y_thresholds_)

    @staticmethod
    def _support(Yc, anchors):
        sims = F.normalize(Yc, dim=-1) @ anchors.T
        return sims.topk(min(SUPPORT_K, anchors.shape[0]), dim=1).values.mean(1)

    def query_support(self, q_tgt_raw: np.ndarray, query_lang: Optional[str] = None) -> np.ndarray:
        """Mean cosine of each centered target query to its 5 nearest training proxies."""
        if self._support_anchors is None:
            raise RuntimeError("fit() must be called (with a centered mode) before query_support()")
        Yc = np.atleast_2d(q_tgt_raw) - self._mu_y(query_lang)
        Yc = Yc / (np.linalg.norm(Yc, axis=-1, keepdims=True) + 1e-8)
        sims = Yc @ self._support_anchors.T
        k = min(SUPPORT_K, sims.shape[1])
        return -np.partition(-sims, k - 1, axis=1)[:, :k].mean(1)

    def quality_report(self) -> QualityReport:
        if self.quality is None:
            raise RuntimeError("fit() must be called (with a centered mode) before quality_report()")
        return self.quality

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
        if self.quality is not None and lang != self.src_lang:
            # The score is fitted on same-language queries; held out on xquad-zh it over-predicted by 10.5 points.
            self.quality.notes.append(
                f"quality score covers {self.src_lang!r} queries only; it does not apply to {lang!r} queries"
            )

    def _mu_y(self, query_lang: Optional[str]) -> np.ndarray:
        lang = query_lang or self.src_lang
        mu_y = self.mu_y_by_lang.get(lang)
        if mu_y is None:
            warnings.warn(
                f"query language {lang!r} not registered; using {self.src_lang!r} centering "
                "without language-cone correction"
            )
            mu_y = self.mu_y_by_lang[self.src_lang]
        return mu_y

    def transform_query(
        self, q_tgt_raw: np.ndarray, query_lang: Optional[str] = None, return_confidence: bool = False
    ):
        """Applies language-aware centering, linear projection W, source-mean restore, and L2 norm.

        With ``return_confidence``, also returns a per-query confidence in [0, 1]: the predicted
        top-10 agreement with native retrieval, from the query's support among the training anchors.
        """
        if self.W is None:
            raise RuntimeError("fit() must be called before transform_query()")
        mu_y = self._mu_y(query_lang)
        q_proj = (q_tgt_raw - mu_y) @ self.W
        if self.restore_source_mean:
            q_proj = q_proj + self.mu_x_query
        q_hat = q_proj / (np.linalg.norm(q_proj, axis=-1, keepdims=True) + 1e-8)
        if not return_confidence:
            return q_hat
        if self._conf_map is None:
            raise RuntimeError("per-query confidence needs a centered fit")
        return q_hat, np.interp(self.query_support(q_tgt_raw, query_lang), *self._conf_map)
