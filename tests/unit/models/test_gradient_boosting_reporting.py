"""Unit tests for frozen Month 1 comparison loading and normalization."""

from pathlib import Path

import pandas as pd
import pytest

from urban_ops.models.evaluation import (
    CalibrationCurve,
    CalibrationEvaluation,
    CalibrationMetrics,
    RankingCurves,
    RankingEvaluation,
    RankingMetrics,
)
from urban_ops.models.gradient_boosting_reporting import (
    COMPARISON_METRICS,
    FrozenLogisticRegressionValidationEvidence,
    build_model_comparison_table,
    load_frozen_logistic_regression_validation_metrics,
)


def _frozen_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return internally consistent structured Month 1 fixture evidence."""
    validation = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                "row_count": 100,
                "positive_count": 40,
                "pr_auc": 0.40,
                "roc_auc": 0.60,
                "brier_score": 0.20,
            }
        ]
    )
    capacity = pd.DataFrame(
        {
            "model": ["Logistic Regression"] * 3,
            "evaluated_split": ["validation"] * 3,
            "capacity": [0.05, 0.10, 0.20],
            "selected_count": [5, 10, 20],
            "captured_positive_count": [3, 5, 10],
            "precision_at_k": [0.60, 0.50, 0.50],
            "recall_at_k": [0.075, 0.125, 0.25],
        }
    )
    return validation, capacity


def _write_frozen_frames(tmp_path) -> tuple[object, object]:
    """Persist structured fixture evidence and return both paths."""
    validation, capacity = _frozen_frames()
    validation_path = tmp_path / "validation.csv"
    capacity_path = tmp_path / "capacity.csv"
    validation.to_csv(validation_path, index=False)
    capacity.to_csv(capacity_path, index=False)
    return validation_path, capacity_path


def test_valid_frozen_evidence_loads_with_normalized_metrics_and_provenance(
    tmp_path,
) -> None:
    validation_path, capacity_path = _write_frozen_frames(tmp_path)

    evidence = load_frozen_logistic_regression_validation_metrics(
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
    )

    assert evidence.model_name == "Logistic Regression"
    assert evidence.evaluated_split == "validation"
    assert evidence.frozen is True
    assert evidence.validation_source == validation_path
    assert evidence.capacity_source == capacity_path
    assert tuple(evidence.metric_values()) == COMPARISON_METRICS
    assert evidence.metric_values()["PR-AUC"] == 0.40
    assert evidence.metric_values()["Precision@5%"] == 0.60
    assert evidence.metric_values()["Recall@20%"] == 0.25


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        ("validation", lambda frame: frame.drop(columns="brier_score"), "missing"),
        (
            "validation",
            lambda frame: frame.assign(pr_auc="invalid"),
            "PR-AUC must be numeric",
        ),
        (
            "validation",
            lambda frame: frame.assign(roc_auc=float("inf")),
            "ROC-AUC must be finite",
        ),
        (
            "validation",
            lambda frame: frame.assign(evaluated_split="test"),
            "exactly one",
        ),
        (
            "validation",
            lambda frame: frame.assign(model="Other Model"),
            "exactly one",
        ),
        ("capacity", lambda frame: frame.drop(columns="recall_at_k"), "missing"),
        (
            "capacity",
            lambda frame: frame.assign(evaluated_split="test"),
            "validation only",
        ),
    ],
)
def test_malformed_or_ambiguous_frozen_evidence_fails_clearly(
    tmp_path,
    artifact,
    mutation,
    message,
) -> None:
    validation, capacity = _frozen_frames()
    if artifact == "validation":
        validation = mutation(validation)
    else:
        capacity = mutation(capacity)
    validation_path = tmp_path / "validation.csv"
    capacity_path = tmp_path / "capacity.csv"
    validation.to_csv(validation_path, index=False)
    capacity.to_csv(capacity_path, index=False)

    with pytest.raises(ValueError, match=message):
        load_frozen_logistic_regression_validation_metrics(
            validation_results_path=validation_path,
            capacity_results_path=capacity_path,
        )


def test_missing_frozen_artifact_fails_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_frozen_logistic_regression_validation_metrics(
            validation_results_path=tmp_path / "missing.csv",
            capacity_results_path=tmp_path / "also-missing.csv",
        )


def test_comparison_difference_is_always_gradient_boosting_minus_logistic() -> None:
    frozen = FrozenLogisticRegressionValidationEvidence(
        model_name="Logistic Regression",
        evaluated_split="validation",
        frozen=True,
        validation_source=Path("validation.csv"),
        capacity_source=Path("capacity.csv"),
        row_count=100,
        positive_count=40,
        metrics=tuple(
            zip(
                COMPARISON_METRICS,
                (0.40, 0.60, 0.20, 0.60, 0.075, 0.50, 0.125, 0.50, 0.25),
            )
        ),
    )
    ranking = RankingEvaluation(
        metrics=RankingMetrics(100, 40, 60, 0.40, 0.65, 0.45),
        curves=RankingCurves((), (), (), (), (), ()),
    )
    calibration = CalibrationEvaluation(
        metrics=CalibrationMetrics(100, 40, 60, 0.40, 0.22),
        curve=CalibrationCurve((), (), 10, "uniform"),
    )
    capacity = pd.DataFrame(
        {
            "capacity": [0.05, 0.10, 0.20],
            "precision_at_k": [0.65, 0.55, 0.52],
            "recall_at_k": [0.08, 0.14, 0.27],
        }
    )

    comparison = build_model_comparison_table(
        ranking=ranking,
        calibration=calibration,
        capacity=capacity,
        frozen_logistic_regression=frozen,
    ).set_index("metric")

    assert comparison.loc["PR-AUC", "difference_gb_minus_lr"] == pytest.approx(0.05)
    assert comparison.loc["Brier Score", "difference_gb_minus_lr"] == pytest.approx(
        0.02
    )
