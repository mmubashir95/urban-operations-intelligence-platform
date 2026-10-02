"""Select an XGBoost candidate deterministically and persist a write-once freeze.

PR-AUC ties are measured against the global maximum, avoiding order-dependent
pairwise comparisons. TEST evidence never enters the selection object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math

from urban_ops.models.evaluation import evaluate_ranking, evaluate_calibration, compare_capacity_levels
from urban_ops.models.gradient_boosting_tuning import GradientBoostingCandidate, GradientBoostingTuningContract, GradientBoostingTuningError

METRIC_NAMES = (
    "pr_auc", "roc_auc", "brier_score", "recall_at_05", "recall_at_10",
    "recall_at_20", "precision_at_05", "precision_at_10", "precision_at_20",
)
SELECTION_POLICY = "Maximize validation PR-AUC; within tolerance of global maximum, maximize Recall@10%, minimize Brier, minimize (max_depth, n_estimators), then lexical candidate ID."


@dataclass(frozen=True)
class CandidateEvaluation:
    """Immutable normalized metrics for one validation score vector."""

    candidate: GradientBoostingCandidate
    metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if tuple(name for name, _ in self.metrics) != METRIC_NAMES:
            raise GradientBoostingTuningError("Candidate evaluation metric schema differs.")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for _, value in self.metrics):
            raise GradientBoostingTuningError("Candidate metrics must be finite in [0, 1].")

    def record(self) -> dict[str, object]:
        """Return a deterministic flat comparison row."""
        return {"candidate_id": self.candidate.candidate_id, **self.candidate.parameters(), **dict(self.metrics)}


def evaluate_scores(y_true: object, scores: object) -> tuple[tuple[str, float], ...]:
    """Normalize existing ranking, raw probability-quality, and capacity metrics."""
    ranking = evaluate_ranking(y_true, scores).metrics
    calibration = evaluate_calibration(y_true, scores).metrics
    capacity = compare_capacity_levels(y_true, scores)
    values = {"pr_auc": ranking.pr_auc, "roc_auc": ranking.roc_auc, "brier_score": calibration.brier_score}
    for row in capacity:
        suffix = f"{round(row.capacity * 100):02d}"
        values[f"recall_at_{suffix}"] = row.recall
        values[f"precision_at_{suffix}"] = row.precision
    return tuple((name, values[name]) for name in METRIC_NAMES)


def select_candidate(evaluations: tuple[CandidateEvaluation, ...], *, tolerance: float) -> CandidateEvaluation:
    """Choose from the global PR-AUC tolerance band using predefined tie breaks."""
    if not evaluations or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise GradientBoostingTuningError("Selection requires evaluations and a valid tolerance.")
    if len({row.candidate.candidate_id for row in evaluations}) != len(evaluations):
        raise GradientBoostingTuningError("Duplicate evaluated candidate IDs.")
    maximum = max(dict(row.metrics)["pr_auc"] for row in evaluations)
    tied = [row for row in evaluations if maximum - dict(row.metrics)["pr_auc"] <= tolerance]
    return min(tied, key=lambda row: (
        -dict(row.metrics)["recall_at_10"], dict(row.metrics)["brier_score"],
        row.candidate.max_depth, row.candidate.n_estimators, row.candidate.candidate_id,
    ))


@dataclass(frozen=True)
class FrozenGradientBoostingDecision:
    """One immutable validation selection with complete candidate provenance."""

    candidate: GradientBoostingCandidate
    validation_metrics: tuple[tuple[str, float], ...]
    split_id: str
    feature_fingerprint: str
    feature_names: tuple[str, ...]
    random_state: int
    pr_auc_tolerance: float
    candidates: tuple[GradientBoostingCandidate, ...]
    selection_timestamp: str
    selection_version: int = 1
    selected_on_split: str = "validation"
    selection_metric: str = "pr_auc"
    selection_policy: str = SELECTION_POLICY
    final_fit_policy: str = "train_only"
    frozen: bool = True

    def payload(self) -> dict[str, object]:
        """Serialize immutable fields without mutable nested state in memory."""
        return asdict(self)


def freeze_decision(selected: CandidateEvaluation, *, contract: GradientBoostingTuningContract,
                    feature_names: tuple[str, ...], path: Path) -> FrozenGradientBoostingDecision:
    """Persist selection exclusively before the final TEST boundary is opened."""
    if selected.candidate not in contract.candidates:
        raise GradientBoostingTuningError("Cannot freeze a candidate outside the predefined set.")
    decision = FrozenGradientBoostingDecision(
        candidate=selected.candidate, validation_metrics=selected.metrics,
        split_id=contract.frozen_split_id, feature_fingerprint=contract.frozen_feature_fingerprint,
        feature_names=feature_names, random_state=contract.random_state,
        pr_auc_tolerance=contract.pr_auc_tolerance, candidates=contract.candidates,
        selection_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(decision.payload(), indent=2, allow_nan=False) + "\n")
    return decision


def verify_persisted_decision(decision: FrozenGradientBoostingDecision, path: Path) -> None:
    """Reject missing, edited, or unfrozen decisions before any TEST access."""
    expected = json.loads(json.dumps(decision.payload()))
    if not decision.frozen or decision.selected_on_split != "validation" or decision.final_fit_policy != "train_only":
        raise GradientBoostingTuningError("Final evaluation requires a frozen validation decision.")
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
        raise GradientBoostingTuningError("Persisted selection freeze is missing or changed.")
