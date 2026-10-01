"""Unit tests for frozen Month 1 comparison loading and normalization."""

import json
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
    load_frozen_logistic_regression_evidence_snapshot,
    load_frozen_logistic_regression_validation_metrics,
    write_frozen_logistic_regression_evidence_snapshot,
    write_gradient_boosting_reports,
)


SPLIT_ID = "frozen-split"
FINGERPRINT = "frozen-phase-9-fingerprint"



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


def _write_snapshot(tmp_path) -> Path:
    """Freeze the fixture CSV evidence into a bound JSON snapshot."""
    validation_path, capacity_path = _write_frozen_frames(tmp_path)
    return write_frozen_logistic_regression_evidence_snapshot(
        split_id=SPLIT_ID,
        phase_9_contract_fingerprint=FINGERPRINT,
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
        output_path=tmp_path / "snapshot.json",
    )


def test_snapshot_round_trips_csv_evidence_with_split_binding(tmp_path) -> None:
    validation_path, capacity_path = _write_frozen_frames(tmp_path)
    from_csv = load_frozen_logistic_regression_validation_metrics(
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
    )
    snapshot_path = _write_snapshot(tmp_path)

    evidence = load_frozen_logistic_regression_evidence_snapshot(
        snapshot_path,
        expected_split_id=SPLIT_ID,
        expected_phase_9_contract_fingerprint=FINGERPRINT,
    )

    assert evidence.metrics == from_csv.metrics
    assert evidence.row_count == from_csv.row_count
    assert evidence.positive_count == from_csv.positive_count
    assert evidence.split_id == SPLIT_ID
    assert evidence.phase_9_contract_fingerprint == FINGERPRINT
    assert evidence.snapshot_source == snapshot_path


def test_snapshot_is_write_once(tmp_path) -> None:
    snapshot_path = _write_snapshot(tmp_path)
    original = snapshot_path.read_bytes()

    # Identical evidence is an idempotent no-op.
    assert _write_snapshot(tmp_path) == snapshot_path

    validation, capacity = _frozen_frames()
    validation.assign(pr_auc=0.41).to_csv(tmp_path / "validation.csv", index=False)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_frozen_logistic_regression_evidence_snapshot(
            split_id=SPLIT_ID,
            phase_9_contract_fingerprint=FINGERPRINT,
            validation_results_path=tmp_path / "validation.csv",
            capacity_results_path=tmp_path / "capacity.csv",
            output_path=snapshot_path,
        )
    assert snapshot_path.read_bytes() == original


@pytest.mark.parametrize(
    ("split_id", "fingerprint", "message"),
    [
        ("other-split", FINGERPRINT, "recorded for split"),
        (SPLIT_ID, "other-fingerprint", "fingerprint"),
    ],
)
def test_snapshot_rejects_mismatched_frozen_inputs(
    tmp_path,
    split_id,
    fingerprint,
    message,
) -> None:
    snapshot_path = _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match=message):
        load_frozen_logistic_regression_evidence_snapshot(
            snapshot_path,
            expected_split_id=split_id,
            expected_phase_9_contract_fingerprint=fingerprint,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("split_id"), "fields must be exactly"),
        (lambda payload: payload.update(evaluated_split="test"), "validation only"),
        (
            lambda payload: payload["validation"].update(pr_auc=1.5),
            "PR-AUC must be finite",
        ),
        (
            lambda payload: payload["capacity"][0].update(precision_at_k=0.9),
            "Precision@5% does not match",
        ),
        (lambda payload: payload.update(capacity=[]), "5%, 10%, and 20%"),
    ],
)
def test_tampered_snapshot_fails_clearly(tmp_path, mutation, message) -> None:
    snapshot_path = _write_snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    mutation(payload)
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_frozen_logistic_regression_evidence_snapshot(
            snapshot_path,
            expected_split_id=SPLIT_ID,
            expected_phase_9_contract_fingerprint=FINGERPRINT,
        )


def test_missing_snapshot_fails_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="snapshot does not exist"):
        load_frozen_logistic_regression_evidence_snapshot(
            tmp_path / "missing.json",
            expected_split_id=SPLIT_ID,
            expected_phase_9_contract_fingerprint=FINGERPRINT,
        )


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


def test_report_writer_produces_complete_deterministic_phase_2_contract(
    tmp_path,
) -> None:
    frozen_directory = tmp_path / "frozen"
    frozen_directory.mkdir()
    snapshot_path = _write_snapshot(frozen_directory)
    ranking = RankingEvaluation(
        metrics=RankingMetrics(100, 40, 60, 0.40, 0.55, 0.58),
        curves=RankingCurves((), (), (), (), (), ()),
    )
    calibration = CalibrationEvaluation(
        metrics=CalibrationMetrics(100, 40, 60, 0.40, 0.21),
        curve=CalibrationCurve((), (), 10, "uniform"),
    )
    calibration_table = pd.DataFrame(
        {
            "bin_index": [0, 1],
            "lower_bound": [0.0, 0.1],
            "upper_bound": [0.1, 0.2],
            "row_count": [60, 40],
            "mean_predicted_risk": [0.05, 0.15],
            "observed_positive_rate": [0.20, 0.70],
        }
    )
    capacity_table = pd.DataFrame(
        {
            "capacity": [0.05, 0.10, 0.20],
            "capacity_pct": [5.0, 10.0, 20.0],
            "selected_count": [5, 10, 20],
            "captured_positive_count": [2, 4, 9],
            "precision_at_k": [0.40, 0.40, 0.45],
            "recall_at_k": [0.05, 0.10, 0.225],
        }
    )
    configuration = {
        "objective": "binary:logistic",
        "n_estimators": 100,
        "random_state": 20260806,
    }

    outputs = []
    for directory_name in ("first", "second"):
        artifacts = write_gradient_boosting_reports(
            ranking=ranking,
            calibration=calibration,
            calibration_table=calibration_table,
            capacity_table=capacity_table,
            training_row_count=200,
            feature_count=2,
            split_id=SPLIT_ID,
            phase_9_contract_fingerprint=FINGERPRINT,
            model_implementation="XGBoost",
            model_class="GradientBoostedRiskModel",
            configuration_version=1,
            model_configuration=configuration,
            output_directory=tmp_path / directory_name,
            frozen_evidence_path=snapshot_path,
        )
        outputs.append(artifacts)

    first, second = outputs
    artifact_pairs = (
        (first.validation_path, second.validation_path),
        (first.calibration_path, second.calibration_path),
        (first.capacity_path, second.capacity_path),
        (first.comparison_path, second.comparison_path),
        (first.markdown_path, second.markdown_path),
    )
    for first_path, second_path in artifact_pairs:
        assert first_path.is_file()
        assert first_path.read_bytes() == second_path.read_bytes()

    validation_csv = pd.read_csv(first.validation_path)
    assert validation_csv.loc[0, "evaluated_split"] == "validation"
    assert validation_csv.loc[0, "probability_status"] == "raw_uncalibrated"
    assert validation_csv[["pr_auc", "roc_auc", "brier_score"]].notna().all().all()

    capacity_csv = pd.read_csv(first.capacity_path)
    assert capacity_csv["capacity"].tolist() == [0.05, 0.10, 0.20]
    assert len(capacity_csv) == 3

    comparison_csv = pd.read_csv(first.comparison_path)
    assert comparison_csv["metric"].tolist() == list(COMPARISON_METRICS)
    assert comparison_csv["difference_gb_minus_lr"].tolist() == pytest.approx(
        (
            comparison_csv["gradient_boosting"]
            - comparison_csv["logistic_regression"]
        ).tolist()
    )

    markdown = first.markdown_path.read_text(encoding="utf-8")
    required_sections = (
        "## 2. Model Trained",
        "## 3. Frozen Inputs Reused",
        "## 4. Evaluation Boundary",
        "## 5. Model Configuration",
        "## 6. Validation Metrics",
        "## 7. Raw Probability Calibration",
        "## 8. Operational Top-K Evaluation",
        "## 9. Frozen Logistic Regression Comparison",
        "## 10. Ranking Improvement Assessment",
        "## 11. Threshold Policy Status",
        "## 12. Test-Set Protection",
        "## 13. Phase 3 Readiness",
    )
    assert all(section in markdown for section in required_sections)
    for fact in (
        "GradientBoostedRiskModel",
        "frozen Month 1 inputs",
        "TRAIN",
        "VALIDATION",
        "PR-AUC",
        "ROC-AUC",
        "Brier Score",
        "Recall@K",
        "Logistic Regression is not retrained",
        "did not materially improve validation ranking",
        "region-specific under- and over-prediction",
        "threshold selection remains deferred",
        "`0.49` was not transferred",
        "No test labels were used",
        "technically ready to proceed to Phase 3",
    ):
        assert fact in markdown
