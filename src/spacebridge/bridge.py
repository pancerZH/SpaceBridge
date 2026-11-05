"""The SpaceBridge query-side alignment engine (design §4.2–§6)."""

import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, get_args

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from . import storage
from .bundle import CalibrationBundle
from .identity import CorpusFingerprint
from .quality import RERANK_POOL_BY_GRADE, QualityReport, index_size_check, language_score, load_calibration, score_report, stage2_health
from .refine import RefineConfig, refine
from .solvers import LAMBDAS_REL, naive_procrustes, ridge_path, stiefel_procrustes

Mode = Literal["naive", "stiefel", "ridge", "stage1", "auto"]
Pairs = Literal["query", "doc"]

# rho_val (teacher top-10 agreement) thresholds for ~90% / ~75% real Recall@30 retention, from a
# 180-fit calibration on scifact (retention ~= 0.85 * rho_val + 0.41, R^2 = 0.87).
HIGH_CONFIDENCE = 0.58
MODERATE_DRIFT = 0.40
VAL_K = 10
SUPPORT_K = 5
SCORE_BLOCK = 1024  # query rows scored at a time against a calibration pool
# Mount protocol (design §2.1): encode the anchor docs passed as ``Y_doc_raw`` truncated to this many tokens.
# 2.3-2.5x faster than full length at -0.4 / -0.5 points Recall@30 pooled (trec-covid / fiqa, 3 seeds, M4); callers may
# choose another length (256 recovers most of the cost but misses the mount budgets for bge-large, EmbeddingGemma 2
# and the 0.6B decoders). 4B-class query models skip doc pairs (``doc_pairs=False``).
DOC_PAIR_TOKENS = 128


def _top_ids(Q: torch.Tensor, pool: torch.Tensor, k: int) -> torch.Tensor:
    """Row ids of each query's top-k in pool, in blocks: queries x pool fp32 is 2.2 GB at K = 32,768."""
    return torch.cat([(Q[s : s + SCORE_BLOCK] @ pool.T).topk(k, dim=1).indices for s in range(0, Q.shape[0], SCORE_BLOCK)])


def topk_agreement(Q: torch.Tensor, pool: torch.Tensor, teacher_top: torch.Tensor) -> float:
    """Mean overlap@k between each query's top-k in pool and the teacher's top-k ids (k = teacher_top width)."""
    top = _top_ids(Q, pool, teacher_top.shape[1])
    return (top[:, :, None] == teacher_top[:, None, :]).any(-1).float().mean().item()


def retrieval_stats(Q: torch.Tensor, pool: torch.Tensor, targets: torch.Tensor, k: int = VAL_K) -> dict:
    """Recall@k and MRR@k when each query in Q has exactly one relevant row ``targets[i]`` in pool. A target tied with other rows
    ranks after all of them: the pessimistic reading, since a top-k can't return every tied row, and collapsed scores read as a failure."""
    rank = []
    for s in range(0, Q.shape[0], SCORE_BLOCK):
        scores = Q[s : s + SCORE_BLOCK] @ pool.T
        rank.append((scores >= scores.gather(1, targets[s : s + SCORE_BLOCK, None])).sum(1) - 1)
    rank = torch.cat(rank)
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
    out = {"agree@10": 0.0, "nbr_recall@10": 0.0, "nbr_recall@30": 0.0, "nbr_recall@100": 0.0}
    corr, gap, gap_sd, n = [], [], [], Q_student.shape[0]
    val_k = min(VAL_K, pool.shape[0])
    others = min(VAL_K, pool.shape[0] - 1)  # the teacher's neighbours, own doc excluded
    for s in range(0, n, SCORE_BLOCK):
        S, T = Q_student[s : s + SCORE_BLOCK] @ pool.T, Q_teacher[s : s + SCORE_BLOCK] @ pool.T
        rows = torch.arange(S.shape[0], device=own.device)
        teacher_top = T.topk(val_k, dim=1).indices
        student_top = S.topk(min(100, pool.shape[0]), dim=1).indices  # sorted, so prefixes are the top-10 and top-30
        out["agree@10"] += (student_top[:, :val_k, None] == teacher_top[:, None, :]).any(-1).float().sum().item() / val_k
        T_other = T.clone()
        T_other[rows, own[s : s + SCORE_BLOCK]] = -torch.inf
        nbrs = T_other.topk(others, dim=1).indices
        for k in (10, 30, 100):
            out[f"nbr_recall@{k}"] += (nbrs[:, :, None] == student_top[:, None, : k]).any(-1).float().sum().item() / others
        Sc, Tc = S - S.mean(1, keepdim=True), T - T.mean(1, keepdim=True)
        corr.append((Sc * Tc).sum(1) / (Sc.norm(dim=1) * Tc.norm(dim=1)))
        top30 = T_other.topk(min(30, pool.shape[0] - 1), dim=1).values
        gap.append(top30[:, 0] - top30[:, -1])
        gap_sd.append(gap[-1] / T.std(dim=1))
    out = {k: v / n for k, v in out.items()}
    out["score_corr"] = torch.cat(corr).mean().item()
    out["teacher_gap"] = torch.cat(gap).mean().item()
    out["teacher_gap_sd"] = torch.cat(gap_sd).mean().item()
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
        self.restore_source_mean = restore_source_mean  # False strips the source query mean (ablation Row 3b)
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
        self.stage2_checkpoints: list[tuple[int, np.ndarray]] = []  # with RefineConfig(keep_checkpoints=True)
        self.stage2_signals = None  # with keep_checkpoints: W (numpy) -> the label-free Stage-2 signals
        self._support_anchors: Optional[np.ndarray] = None  # (K_train, d_t) unit-norm centered target proxies
        self._conf_map: Optional[tuple[np.ndarray, np.ndarray]] = None  # isotonic support -> agreement knots
        self._val_pool: Optional[tuple] = None  # (anchor docs, distractors) as passed to fit(), for language validation
        self.corpus: Optional[CorpusFingerprint] = None  # set by mount: the corpus and row order this map belongs to
        self.src_encoder: Optional[dict] = None
        self.tgt_encoder: Optional[dict] = None

    def _reset_fit_state(self) -> None:
        """Forgets every result of an earlier fit, so a fit that fails or differs in kind can't leave any of it behind."""
        self.mu_x_query, self.W = None, None
        self.mu_y_by_lang = {}
        self.status, self.metrics, self.quality = "uninitialized", {}, None
        self.stage2_checkpoints, self.stage2_signals = [], None
        self._support_anchors, self._conf_map, self._val_pool = None, None, None
        self.corpus, self.src_encoder, self.tgt_encoder = None, None, None

    @staticmethod
    def _check_calibration(**arrays: np.ndarray) -> None:
        """Rejects calibration embeddings no map can be fitted on, before they reach a solver as NaNs."""
        for name, a in arrays.items():
            if not np.isfinite(a).all():
                raise ValueError(f"calibration embeddings {name} aren't finite")
            if not np.ptp(a, axis=0).any():
                raise ValueError(f"calibration embeddings {name} are all the same vector: a collapsed or constant encoder?")

    def save(self, path: str | Path) -> None:
        """Writes what serving needs: the map, the means, the quality report and the confidence map (crash-safe)."""
        if self.W is None:
            raise RuntimeError("fit() must be called before save()")
        arrays = {"W": self.W, "mu_x_query": self.mu_x_query}
        arrays.update({f"mu_y/{lang}": mu for lang, mu in self.mu_y_by_lang.items()})
        if self._support_anchors is not None:
            arrays["support_anchors"] = self._support_anchors
        if self._conf_map is not None:
            arrays["conf_x"], arrays["conf_y"] = self._conf_map
        meta = {
            "src_lang": self.src_lang, "restore_source_mean": self.restore_source_mean, "status": self.status,
            "metrics": self.metrics, "quality": None if self.quality is None else asdict(self.quality),
            "src_encoder": self.src_encoder, "tgt_encoder": self.tgt_encoder,
        }
        if self.corpus is not None:
            meta["corpus"], arrays["corpus_vectors"] = self.corpus.to_meta(), self.corpus.vectors
        storage.save_dir(path, "fitted-bridge", arrays, meta)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "SpaceBridge":
        """A bridge ready for ``transform_query``. Fitting state stays behind, so ``validate_query_language`` needs a new fit."""
        arrays, meta = storage.load_dir(path, "fitted-bridge")
        bridge = cls(device=device, restore_source_mean=meta["restore_source_mean"])
        bridge.src_lang, bridge.status, bridge.metrics = meta["src_lang"], meta["status"], meta["metrics"]
        bridge.W, bridge.mu_x_query = arrays["W"], arrays["mu_x_query"]
        bridge.mu_y_by_lang = {name.split("/", 1)[1]: a for name, a in arrays.items() if name.startswith("mu_y/")}
        bridge.corpus = CorpusFingerprint.from_saved(meta.get("corpus"), arrays.get("corpus_vectors"))
        bridge.src_encoder, bridge.tgt_encoder = meta.get("src_encoder"), meta.get("tgt_encoder")
        bridge._support_anchors = arrays.get("support_anchors")
        if "conf_x" in arrays:
            bridge._conf_map = (arrays["conf_x"], arrays["conf_y"])
        if meta["quality"] is not None:
            bridge.quality = QualityReport(**meta["quality"])
            if bridge.quality.band is not None:
                bridge.quality.band = tuple(bridge.quality.band)
            for scored in bridge.quality.by_language.values():
                scored["band"] = tuple(scored["band"])
        return bridge

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
    ) -> Dict[str, Any]:
        """Fits the affine query map and computes rho_val / status.

        Args:
            Y_raw: (K, d_t) target embeddings of the anchors: of ``bundle.anchor_query_texts`` with
                the target query prompt when ``pairs="query"``, or of the anchor docs with the target
                doc prompt when ``pairs="doc"`` (doc-anchor ablation rows).
            mode: ``naive`` (uncentered full Procrustes), ``stiefel`` / ``ridge`` (one Stage-1 operator),
                ``stage1`` (best Stage-1 operator on val), ``auto`` (``stage1`` + Stage-2 refinement).
            X_distractor: index vectors at ``bundle.distractor_ids``, added to the rho_val pool.
            Y_doc_raw: optional (K, d_t) target doc embeddings of the anchor docs, encoded with the target doc
                prompt and truncated to ``DOC_PAIR_TOKENS`` tokens by default. Their training rows are added as
                doc<->doc pairs (each centered on its own doc-cone mean) to Stage 1 and 2; validation stays on
                query proxies.
        """
        if mode not in get_args(Mode):
            raise ValueError(f"unknown fit mode {mode!r}; choose from {get_args(Mode)}")
        if pairs not in get_args(Pairs):
            raise ValueError(f"unknown pair type {pairs!r}; choose from {get_args(Pairs)}")
        if bundle.X_query_raw is None:
            raise ValueError("this bundle has no source-model embeddings: use fit_source_free()")
        if mode == "auto" and pairs != "query":
            raise ValueError("Stage-2 refinement needs query-proxy pairs")
        if Y_doc_raw is not None and (pairs != "query" or mode == "naive"):
            raise ValueError("doc<->doc pairs augment centered query-proxy fits only")
        if mode == "auto" and self.doc_pairs and Y_doc_raw is None:
            raise ValueError(
                "the default auto fit uses doc<->doc pairs: pass Y_doc_raw (anchor docs encoded with the "
                "target doc prompt), or construct SpaceBridge(doc_pairs=False)"
            )
        self._reset_fit_state()
        self._check_calibration(Y_raw=Y_raw, X_query_raw=bundle.X_query_raw, X_doc_raw=bundle.X_doc_raw)
        Y = self._t(Y_raw)
        Xq, Xd = self._t(bundle.X_query_raw), self._t(bundle.X_doc_raw)
        X_pairs = Xq if pairs == "query" else Xd
        tr = torch.as_tensor(bundle.train_idx, device=self.device)
        va = torch.as_tensor(bundle.val_idx, device=self.device)
        pool = Xd if X_distractor is None else torch.cat([Xd, self._t(X_distractor)])
        # Proxies trivially retrieve their own anchor doc, so validation scores agreement with the
        # native source ranking (teacher top-k over the pool) rather than own-doc recall.
        teacher_top = _top_ids(Xq[va], pool, min(VAL_K, pool.shape[0]))
        # Model selection (Stage-1 candidates, Stage-2 early stopping) scores on a fixed subsample of
        # the pool: all anchor docs plus random distractors; the reported rho_val uses the full pool.
        sel_pool = pool
        if pool.shape[0] > self.selection_pool:
            gen = torch.Generator().manual_seed(0)
            extra = Xd.shape[0] + torch.randperm(pool.shape[0] - Xd.shape[0], generator=gen)
            keep = torch.cat([torch.arange(Xd.shape[0]), extra[: max(0, self.selection_pool - Xd.shape[0])]])
            sel_pool = pool[keep.to(self.device)]
        sel_teacher = _top_ids(Xq[va], sel_pool, min(VAL_K, sel_pool.shape[0]))

        def val_score(W: torch.Tensor) -> float:
            return topk_agreement(self._project(Y[va], mu_y, mu_x, W), sel_pool, sel_teacher)

        def stage2_signals(W: torch.Tensor) -> dict:  # anchor docs are the first rows of sel_pool
            # Own-doc MRR alone rewards pulling proxies onto their own docs (on the identity it trades
            # 20 points of agree@10 for +0.4 MRR), so the stop rule also watches teacher agreement.
            Q = self._project(Y[va], mu_y, mu_x, W)
            out = {"agree": topk_agreement(Q, sel_pool, sel_teacher), "mrr": retrieval_stats(Q, sel_pool, va)["mrr"]}
            if Y_doc_raw is not None and self.refine_config.keep_checkpoints:  # held-out doc pairs (studies)
                Qd = F.normalize((Yd_va - Yd_mu) @ W + (mu_xd if self.restore_source_mean else 0.0), dim=-1)
                out["doc_agree"] = topk_agreement(Qd, sel_pool, doc_teacher)
                out["doc_mrr"] = retrieval_stats(Qd, sel_pool, va)["mrr"]
            return out

        info: Dict[str, Any] = {"mode": mode, "pairs": pairs}
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
                mu_xd, Yd_mu = Xd[tr].mean(0), Yd.mean(0)
                Ydc, Xdc = Yd - Yd_mu, Xd[tr] - mu_xd
                Yd_va = self._t(Y_doc_raw)[va]
                doc_teacher = _top_ids(Xd[va], sel_pool, min(VAL_K, sel_pool.shape[0]))
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
                # Without residual norm matching the scale stays in W, so it has to be the right one for this map.
                W_s, s_info = stiefel_procrustes(Y_fit, X_fit, objective_scale=not self.match_residual_norm)
                W_s = rescale(W_s)
                candidates.append((val_score(W_s), W_s, {"solver": "stiefel", "rank": s_info["rank"], "scale": s_info["scale"]}))
            if "ridge" in ops:
                for lam_rel, W_r in ridge_path(Y_fit, X_fit, self.lambdas_rel):
                    W_r = rescale(W_r)
                    candidates.append((val_score(W_r), W_r, {"solver": "ridge", "lambda_rel": lam_rel}))
            _, W, chosen = max(candidates, key=lambda c: c[0])
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
            # when it improves real retention, so selecting on agree@10 alone would keep step 0.
            W, r_info = refine(W, mu_rows, Y_rows, teacher_rows, Xd[tr], stage2_signals, self.refine_config, labels)
            self._sync()
            info["stage2_ms"] = 1e3 * (time.perf_counter() - t0)
            info["stage2_best_step"] = r_info["best_step"]
            info["stage2_averaged"] = r_info["averaged"]
            info["stage2_steps_run"] = r_info["steps_run"]
            info["stage2_tau"] = r_info["tau"]
            info["stage2_gain"] = r_info["mrr_gain"]
            info["stage2_floor_rejected"] = r_info["floor_rejected"]
            info["stage2_curve"] = [{"step": s, **sig} for s, sig in r_info["val_curve"]]
            info["stage2_teacher_sd"] = r_info["teacher_sd"]
            info["stage2_tau_scaled"] = bool(r_info["tau_scaled"])
            self.stage2_checkpoints = [(s, Wk.cpu().numpy()) for s, Wk in r_info.get("checkpoints", [])]
            if self.refine_config.keep_checkpoints:  # studies score other candidate maps with the same signals
                self.stage2_signals = lambda W_np: stage2_signals(self._t(W_np))

        self.src_lang = bundle.src_lang
        self._val_pool = (bundle.X_doc_raw, X_distractor)
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
            if mode == "auto":
                stage2_health(self.quality, info["stage2_best_step"], info["stage2_floor_rejected"], info["stage2_tau_scaled"])
            if bundle.index_size is not None:
                index_size_check(self.quality, bundle.index_size)
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

    def fit_source_free(
        self, bundle: CalibrationBundle, Y_query_raw: np.ndarray, Y_doc_raw: np.ndarray, X_distractor: Optional[np.ndarray] = None,
        stage2: bool = True, center: str = "doc", operator: str = "stiefel", stage2_rows: str = "both",
    ) -> Dict[str, Any]:
        """Fits without the source model, for an index whose model is gone (design §4.5): only stored vectors and M_tgt.

        Queries stay in the target model's own geometry: they are centered on its *doc* mean and restored on the source
        doc mean, so a doc-pair map carries the target's query/doc relation across. With no teacher, Stage 2 uses the
        proxies' own-doc retrieval and has no KL term, and the quality report has no score. ``operator="select"`` also
        tries ridge maps and picks by own-doc MRR, which picked worse maps across families (FiQA, design §4.5).
        """
        if center not in ("doc", "query"):
            raise ValueError(f"unknown centering {center!r}; choose 'doc' or 'query'")
        if operator not in ("stiefel", "select"):
            raise ValueError(f"unknown source-free operator {operator!r}; choose 'stiefel' or 'select'")
        if stage2_rows not in ("both", "proxies", "docs"):
            raise ValueError(f"unknown Stage-2 rows {stage2_rows!r}; choose 'both', 'proxies' or 'docs'")
        self._reset_fit_state()
        self._check_calibration(Y_query_raw=Y_query_raw, Y_doc_raw=Y_doc_raw, X_doc_raw=bundle.X_doc_raw)
        Yq, Yd, Xd = self._t(Y_query_raw), self._t(Y_doc_raw), self._t(bundle.X_doc_raw)
        tr = torch.as_tensor(bundle.train_idx, device=self.device)
        va = torch.as_tensor(bundle.val_idx, device=self.device)
        pool = Xd if X_distractor is None else torch.cat([Xd, self._t(X_distractor)])
        sel_pool = pool
        if pool.shape[0] > self.selection_pool:
            gen = torch.Generator().manual_seed(0)
            extra = Xd.shape[0] + torch.randperm(pool.shape[0] - Xd.shape[0], generator=gen)
            keep = torch.cat([torch.arange(Xd.shape[0]), extra[: max(0, self.selection_pool - Xd.shape[0])]])
            sel_pool = pool[keep.to(self.device)]
        mu_yd, mu_x = Yd[tr].mean(0), Xd[tr].mean(0)
        mu_y = mu_yd if center == "doc" else Yq[tr].mean(0)
        Ydc, Xdc = Yd - mu_yd, Xd - mu_x
        Yqc = Yq - mu_y
        target_norm = Xdc[tr].norm(dim=1).mean()
        rescale = (lambda W: W * (target_norm / (Ydc[tr] @ W).norm(dim=1).mean())) if self.match_residual_norm else (lambda W: W)

        def mrr(W: torch.Tensor, pool_: torch.Tensor) -> float:
            return retrieval_stats(self._project(Yq[va], mu_y, mu_x, W), pool_, va)["mrr"]

        info: Dict[str, Any] = {"mode": "source_free", "pairs": "doc", "doc_pairs": len(tr)}
        t0 = time.perf_counter()
        candidates = []
        W_s, s_info = stiefel_procrustes(Ydc[tr], Xdc[tr], objective_scale=not self.match_residual_norm)
        W_s = rescale(W_s)
        candidates.append((mrr(W_s, sel_pool), W_s, {"solver": "stiefel", "rank": s_info["rank"], "scale": s_info["scale"]}))
        if operator == "select":
            for lam_rel, W_r in ridge_path(Ydc[tr], Xdc[tr], self.lambdas_rel):
                W_r = rescale(W_r)
                candidates.append((mrr(W_r, sel_pool), W_r, {"solver": "ridge", "lambda_rel": lam_rel}))
        _, W, chosen = max(candidates, key=lambda c: c[0])
        info.update(chosen)
        self._sync()
        info["stage1_ms"] = 1e3 * (time.perf_counter() - t0)
        step0_mrr = mrr(W, sel_pool)
        if stage2:
            t0 = time.perf_counter()
            n = len(tr)
            rows = {"both": [Yqc[tr], Ydc[tr]], "proxies": [Yqc[tr]], "docs": [Ydc[tr]]}[stage2_rows]
            restore = 1.0 if self.restore_source_mean else 0.0  # the map trained is the map served
            mu_rows = (restore * mu_x).expand(len(rows) * n, -1)
            Y_rows, labels = torch.cat(rows), torch.cat([torch.arange(n, device=self.device)] * len(rows))
            cfg = replace(self.refine_config, alpha=0.0, stop_rule="tail")

            def signals(W_: torch.Tensor) -> dict:  # no teacher: the agreement floor is always passed
                return {"agree": 1.0, "mrr": mrr(W_, sel_pool)}

            W, r_info = refine(W, mu_rows, Y_rows, Xd[tr][labels], Xd[tr], signals, cfg, labels)
            self._sync()
            info.update(stage2_ms=1e3 * (time.perf_counter() - t0), stage2_gain=r_info["mrr_gain"], stage2_tau=r_info["tau"],
                        stage2_averaged=r_info["averaged"], stage2_best_step=r_info["best_step"], stage2_curve=[{"step": s, **sig} for s, sig in r_info["val_curve"]])

        self.src_lang = bundle.src_lang
        self.W = W.cpu().numpy()
        self.mu_x_query = mu_x.cpu().numpy()
        self.mu_y_by_lang = {self.src_lang: mu_y.cpu().numpy()}
        resid_true, resid_pred = Xdc[va], Ydc[va] @ W
        stats = retrieval_stats(self._project(Yq[va], mu_y, mu_x, W), pool, va)
        self.quality = QualityReport({
            "fit": {
                "val_r2": float(1 - (resid_pred - resid_true).pow(2).sum() / resid_true.pow(2).sum()),
                "val_resid_cos": F.cosine_similarity(resid_pred, resid_true, dim=1).mean().item(),
                "stage2_gain": float(info.get("stage2_gain", 0.0)),
            },
            "recoverability": {"owndoc_mrr": stats["mrr"], "owndoc_recall@10": stats["recall"], "owndoc_mrr_stage1": step0_mrr},
            "coverage": {"K_over_dt": bundle.K / Y_query_raw.shape[1]},
        })
        self.quality.notes.append(
            "fitted without the source model: no teacher, so no rho_val, no calibrated score and no per-query confidence"
        )
        if bundle.index_size is not None:
            index_size_check(self.quality, bundle.index_size)
        self.quality.rerank_pool = max(self.quality.rerank_pool, RERANK_POOL_BY_GRADE["D"])
        self._support_anchors = F.normalize(Yqc[tr], dim=-1).cpu().numpy()
        info.update(val_pool_size=int(pool.shape[0]), quality=self.quality.details, status="NO_TEACHER")
        self.status = "NO_TEACHER"
        self.metrics = info
        return info

    def _quality_report(self, val_metrics, Y, X_pairs, Xq, pool, tr, va, mu_y, mu_x, W, stage2_gain, bundle) -> QualityReport:
        """Measurements of design §5.1 from the calibration data alone (no labeled queries)."""
        resid_true = X_pairs[va] - mu_x
        resid_pred = (Y[va] - mu_y) @ W
        tr_s = tr[: len(va)]
        teacher_tr = _top_ids(Xq[tr_s], pool, min(VAL_K, pool.shape[0]))
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
        val_k = min(VAL_K, pool.shape[0])
        teacher = _top_ids(Xq[va], pool, val_k)
        student = _top_ids(self._project(Y[va], mu_y, mu_x, W), pool, val_k)
        agree = (student[:, :, None] == teacher[:, None, :]).any(-1).float().mean(1)
        iso = IsotonicRegression(out_of_bounds="clip").fit(support.cpu().numpy(), agree.cpu().numpy())
        self._conf_map = (iso.X_thresholds_, iso.y_thresholds_)

    @staticmethod
    def _support(Yc, anchors):
        Yn, k = F.normalize(Yc, dim=-1), min(SUPPORT_K, anchors.shape[0])
        return torch.cat([(Yn[s : s + SCORE_BLOCK] @ anchors.T).topk(k, dim=1).values.mean(1) for s in range(0, Yn.shape[0], SCORE_BLOCK)])

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
        """Registers the language-cone shift from M_tgt query-prompt embeddings of the same parallel sentences in src_lang and
        in lang. By default those are the Belebele-filtered FLORES dev sentences (94 for Chinese), else 256 Tatoeba pairs (Chinese only)."""
        if self.src_lang not in self.mu_y_by_lang:
            raise RuntimeError("fit() must be called before register_query_language()")
        shift = Y_flores_query_lang.mean(axis=0) - Y_flores_src_lang.mean(axis=0)
        self.mu_y_by_lang[lang] = self.mu_y_by_lang[self.src_lang] + shift
        if self.quality is not None and lang != self.src_lang:
            # The score is fitted on same-language queries; it over-predicts other languages by 5-12 points (close-out,
            # 7 query languages) until validate_query_language() gives the language its own score.
            self.quality.notes.append(
                f"quality score covers {self.src_lang!r} queries only; it does not apply to {lang!r} queries"
            )

    def validate_query_language(
        self, lang: str, Y_val_query_lang: np.ndarray, Y_val_src_lang: np.ndarray,
        X_val_query: np.ndarray, X_val_doc: np.ndarray,
    ) -> dict:
        """Per-language validation rho_val,Lq (design §5) on held-out parallel sentences (FLORES devtest).

        Args:
            Y_val_query_lang / Y_val_src_lang: (n, d_t) M_tgt query-prompt embeddings of the sentences in
                ``lang`` and in the source language (``lang`` must be registered first).
            X_val_query / X_val_doc: (n, d_s) M_src embeddings of the source-language sentences with its query
                prompt (the teacher) and its doc prompt (the counterparts, mixed into the validation pool).

        Each projected sentence must retrieve its counterpart from the pool (all anchor docs + distractors +
        counterparts) and agree with the teacher's top 10, once in ``lang`` and once in the source language.
        The source-language pass shares the sentences' domain and length, so the difference between the two
        isolates the cost of the language shift. Stored in ``quality.details["language"][lang]``; the report also gets
        a per-language score (``quality.by_language[lang]``: the score scaled by the counterpart Recall@10 ratio,
        ±5.1 points on 7 held-out cross-lingual sets, experimental) with its own grade and advisory rerank pool.
        """
        if lang not in self.mu_y_by_lang or self._val_pool is None:
            raise RuntimeError("fit() and register_query_language(lang) must be called before validate_query_language()")
        Xc = self._t(X_val_doc)
        pool = torch.cat([*(self._t(p) for p in self._val_pool if p is not None), Xc])
        own = torch.arange(pool.shape[0] - Xc.shape[0], pool.shape[0], device=self.device)
        teacher_top = _top_ids(self._t(X_val_query), pool, VAL_K)
        out: dict = {"n": int(Xc.shape[0])}
        for key, Y, l in (("lang", Y_val_query_lang, lang), ("src", Y_val_src_lang, self.src_lang)):
            Q = self._t(self.transform_query(np.asarray(Y, dtype=np.float32), query_lang=l))
            stats = retrieval_stats(Q, pool, own)
            out[f"agree@10_{key}"] = topk_agreement(Q, pool, teacher_top)
            out[f"recall@10_{key}"], out[f"mrr_{key}"] = stats["recall"], stats["mrr"]
        # Teacher ceiling: the source model's own query embedding retrieving the counterpart.
        out["recall@10_teacher"] = retrieval_stats(self._t(X_val_query), pool, own)["recall"]
        if self.quality is not None:
            self.quality.details.setdefault("language", {})[lang] = out
            scored = language_score(self.quality, lang, out["recall@10_lang"], out["recall@10_src"])
            if scored is not None:
                self.quality.notes = [n for n in self.quality.notes if not n.endswith(f"does not apply to {lang!r} queries")]
                self.quality.notes.append(
                    f"{lang!r} queries: score {scored['score']:.0f} (grade {scored['grade']}, rerank pool {scored['rerank_pool']}) "
                    "from per-language validation, experimental"
                )
        return out

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
        if self._conf_map is None:  # a source-model-free fit has no teacher to calibrate it on
            return q_hat, np.full(q_hat.shape[:-1] if q_hat.ndim > 1 else (1,), np.nan)
        return q_hat, np.interp(self.query_support(q_tgt_raw, query_lang), *self._conf_map)
