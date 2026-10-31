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
# the default of 60, where Tier 1 and 3 of 4 Tier 3 pairs meet their end-to-end targets on fiqa.
DEFAULT_RERANK_POOL = 60
RERANK_POOL_BY_GRADE = {"A": 60, "B": 60, "C": 100, "D": 100}


@dataclass
class QualityReport:
    details: dict[str, dict[str, float]]
    score: Optional[float] = None               # predicted % ceiling Recall@30 retention
    band: Optional[tuple[float, float]] = None  # score ± one residual s.d. of the calibration fit
    grade: Optional[str] = None
    rerank_pool: int = DEFAULT_RERANK_POOL       # advisory routing: candidates to send to the cross-encoder
    experimental: bool = True
    notes: list[str] = field(default_factory=list)

    def features(self) -> dict[str, float]:
        return {f"{group}.{name}": v for group, vals in self.details.items() for name, v in vals.items()}


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
