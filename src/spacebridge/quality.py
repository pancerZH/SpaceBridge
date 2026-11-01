"""Conversion quality report (design §5.1): grouped measurements, calibrated score and grade."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CALIBRATION_PATH = Path(__file__).with_name("quality_calibration.json")
GRADES = ((95.0, "A"), (85.0, "B"), (70.0, "C"))
# Advisory routing (M2): the score's ±4.3-point transfer error is about half a grade and it over-predicts
# hard corpora (trec-covid by 8 points), so a grade may widen the rerank pool but never shrink it below
# the default of 60, where all Tier 1 and 3-4 of 4 Tier 3 fiqa pairs (by reranker) meet end-to-end targets.
DEFAULT_RERANK_POOL = 60
RERANK_POOL_BY_GRADE = {"A": 60, "B": 60, "C": 100, "D": 100}
# Stage-2 health (M4). The calibrated score cannot see a failed Stage 2: with the agreement floor rejecting
# every checkpoint on a multilingual-e5 index (T2Retrieval, K = 4,096) it predicted 89 for a real 74. A fit is
# flagged when Stage 2 kept step 0 while the floor rejected at least this share of its checkpoints, or when the
# compressed-score temperature rule fired (teacher score s.d. < 0.04: an index regime the calibration never saw).
STAGE2_BLOCKED_SHARE = 0.5
# Per-language score (close-out, FLORES): the English score times the counterpart Recall@10 ratio (query language /
# source language) of held-out FLORES devtest pairs. Chosen on xquad-zh; on 7 held-out sets (MLQA-zh, XQuAD
# de/es/ru/ar/hi/vi; 168 fits) it predicts real retention with RMSE 5.1 points (English score: 8.7, biased +8.0),
# so its band is ±5.1 points. Experimental: all evidence is Wikipedia QA (XQuAD / MLQA) into one English index.
LANGUAGE_SCORE_SD = 0.051


@dataclass
class QualityReport:
    details: dict[str, dict[str, float]]
    score: Optional[float] = None               # predicted % ceiling Recall@30 retention
    band: Optional[tuple[float, float]] = None  # score ± one residual s.d. of the calibration fit
    grade: Optional[str] = None
    rerank_pool: int = DEFAULT_RERANK_POOL       # advisory routing: candidates to send to the cross-encoder
    experimental: bool = True
    notes: list[str] = field(default_factory=list)
    by_language: dict[str, dict] = field(default_factory=dict)  # registered query languages with a validation score

    def features(self) -> dict[str, float]:
        return {f"{group}.{name}": v for group, vals in self.details.items() for name, v in vals.items()}


def stage2_health(report: QualityReport, best_step: int, floor_rejected: float, compressed: bool) -> QualityReport:
    """Adds the label-free Stage-2 health flag (``details["stage2"]``, a note, a widened rerank pool)."""
    blocked = best_step == 0 and floor_rejected >= STAGE2_BLOCKED_SHARE
    report.details["stage2"] = {
        "best_step": float(best_step), "floor_rejected": float(floor_rejected),
        "blocked": float(blocked), "compressed_scores": float(compressed), "health_flag": float(blocked or compressed),
    }
    if blocked:
        report.notes.append(
            f"stage 2 blocked: the agreement floor rejected {floor_rejected:.0%} of its checkpoints and step 0 was kept; "
            "the score likely over-predicts retention"
        )
    if compressed:
        report.notes.append(
            "compressed index scores (teacher score s.d. < 0.04): Stage 2 ran with a scaled temperature, outside the "
            "score's calibration data"
        )
    if blocked or compressed:
        report.rerank_pool = max(report.rerank_pool, RERANK_POOL_BY_GRADE["D"])
    return report


def language_score(report: QualityReport, lang: str, recall_lang: float, recall_src: float) -> Optional[dict]:
    """Score, band, grade and advisory rerank pool for queries in ``lang`` (``report.by_language[lang]``): the
    English score scaled by how much of the source language's counterpart Recall@10 the language keeps."""
    if report.score is None or recall_src <= 0:
        return None
    score = report.score * recall_lang / recall_src
    sd = 100.0 * LANGUAGE_SCORE_SD
    g = grade(score)
    pool = max(RERANK_POOL_BY_GRADE[g], report.rerank_pool)  # never narrower than the source-language routing
    report.by_language[lang] = {"score": score, "band": (score - sd, score + sd), "grade": g, "rerank_pool": pool,
                                "experimental": True}
    return report.by_language[lang]


def grade(score: float) -> str:
    return next((g for lo, g in GRADES if score >= lo), "D")


def load_calibration(path: Path = CALIBRATION_PATH) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def score_report(report: QualityReport, calibration: Optional[dict]) -> QualityReport:
    """Fills score / band / grade from a linear calibration over ``report.features()``."""
    if calibration is None:
        report.notes.append("no quality calibration available; score not computed")
        return report
    feats = report.features()
    missing = [f for f in calibration["features"] if f not in feats or not np.isfinite(feats[f])]
    if missing:
        report.notes.append(f"score not computed; missing measurements {missing}")
        return report
    x = np.array([feats[f] for f in calibration["features"]])
    z = (x - np.array(calibration["mean"])) / np.array(calibration["scale"])
    score = 100.0 * float(calibration["intercept"] + z @ np.array(calibration["coef"]))
    sd = 100.0 * calibration["resid_sd"]
    report.score, report.band, report.grade = score, (score - sd, score + sd), grade(score)
    report.experimental = calibration.get("experimental", True)
    report.rerank_pool = RERANK_POOL_BY_GRADE[report.grade]
    if report.grade in "CD":
        report.notes.append(f"grade {report.grade}: expect noticeable loss vs. re-indexing; rerank pool widened")
    if report.grade == "D":
        report.notes.append("recommend native re-indexing with the target model")
    return report
