"""Integration tests for the dedicated Month 2 Gradient Boosting workflow."""

from dataclasses import replace
from math import ceil

import numpy as np
import pandas as pd
import pytest

from urban_ops.models import (
    gradient_boosting_calibration,
    gradient_boosting_capacity,
    gradient_boosting_ranking,
    gradient_boosting_workflow,
)
from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)
from urban_ops.models.gradient_boosting_reporting import COMPARISON_METRICS
from urban_ops.models.gradient_boosting_reporting import (
    load_frozen_logistic_regression_evidence_snapshot,
    write_frozen_logistic_regression_evidence_snapshot,
)
from urban_ops.models.gradient_boosting_workflow import (
    freeze_logistic_regression_validation_evidence,
    run_gradient_boosting_workflow,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


class ProtectedSplits(dict):
    """Record train/validation access and reject all test access."""

    def __init__(self, values):
        super().__init__(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "test":
            raise AssertionError("Phase 2 accessed the protected test split.")
        return super().__getitem__(key)


def _write_matching_frozen_csvs(tmp_path, inputs):
    """Write Month 1-style CSV evidence aligned to the synthetic validation split."""
    y_validation = inputs.targets["validation"]
    row_count = len(y_validation)
    positive_count = int(y_validation.sum())
    validation_path = tmp_path / "frozen_validation.csv"
    capacity_path = tmp_path / "frozen_capacity.csv"
    pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                "row_count": row_count,
                "positive_count": positive_count,
                "pr_auc": 0.50,
                "roc_auc": 0.50,
                "brier_score": 0.25,
            }
        ]
    ).to_csv(validation_path, index=False)
    pd.DataFrame(
        {
            "model": ["Logistic Regression"] * 3,
            "evaluated_split": ["validation"] * 3,
            "capacity": [0.05, 0.10, 0.20],
            "selected_count": [1, 1, 2],
            "captured_positive_count": [1, 1, 1],
            "precision_at_k": [1.0, 1.0, 0.5],
            "recall_at_k": [1 / positive_count] * 3,
        }
    ).to_csv(capacity_path, index=False)
    return validation_path, capacity_path


def _write_matching_frozen_evidence(tmp_path, fixture, inputs):
    """Freeze synthetic CSV evidence through the real snapshot entry point."""
    validation_path, capacity_path = _write_matching_frozen_csvs(tmp_path, inputs)
    snapshot_path = freeze_logistic_regression_validation_evidence(
        eda_config_path=fixture.config,
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
        output_path=tmp_path / "frozen_lr_evidence.json",
    )
    evidence = load_frozen_logistic_regression_evidence_snapshot(
        snapshot_path,
        expected_split_id=inputs.split_id,
        expected_phase_9_contract_fingerprint=inputs.phase_9_contract.fingerprint,
    )
    assert evidence.split_id == inputs.split_id
    return snapshot_path


def test_dedicated_workflow_uses_train_and_validation_only(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    matrices = ProtectedSplits(inputs.matrices)
    targets = ProtectedSplits(inputs.targets)
    protected_inputs = replace(inputs, matrices=matrices, targets=targets)
    frozen_evidence_path = _write_matching_frozen_evidence(tmp_path, fixture, inputs)
    load_calls = []
    fit_calls = []
    score_calls = []
    evaluation_calls = []

    def load_inputs(**kwargs):
        load_calls.append(kwargs)
        return protected_inputs

    original_fit = GradientBoostedRiskModel.fit

    def spy_fit(model, X_train, y_train, *, feature_names):
        fit_calls.append((X_train, y_train, tuple(feature_names)))
        return original_fit(
            model,
            X_train,
            y_train,
            feature_names=feature_names,
        )

    original_predict_score = GradientBoostedRiskModel.predict_score

    def spy_predict_score(model, X):
        score_calls.append(X)
        return original_predict_score(model, X)

    original_ranking = gradient_boosting_ranking.evaluate_ranking
    original_calibration = gradient_boosting_calibration.evaluate_calibration
    original_capacity = gradient_boosting_capacity.compare_capacity_levels

    def spy_ranking(y_true, y_score):
        evaluation_calls.append(("ranking", y_true, y_score))
        return original_ranking(y_true, y_score)

    def spy_calibration(y_true, y_score):
        evaluation_calls.append(("calibration", y_true, y_score))
        return original_calibration(y_true, y_score)

    def spy_capacity(y_true, y_score, *, capacities):
        evaluation_calls.append(("capacity", y_true, y_score))
        return original_capacity(y_true, y_score, capacities=capacities)

    monkeypatch.setattr(
        gradient_boosting_workflow,
        "load_and_verify_gradient_boosting_inputs",
        load_inputs,
    )
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", spy_fit)
    monkeypatch.setattr(
        GradientBoostedRiskModel,
        "predict_score",
        spy_predict_score,
    )
    monkeypatch.setattr(
        gradient_boosting_ranking,
        "evaluate_ranking",
        spy_ranking,
    )
    monkeypatch.setattr(
        gradient_boosting_calibration,
        "evaluate_calibration",
        spy_calibration,
    )
    monkeypatch.setattr(
        gradient_boosting_capacity,
        "compare_capacity_levels",
        spy_capacity,
    )
    monkeypatch.setattr(
        LogisticRegressionBaseline,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Phase 2.12 retrained Logistic Regression.")
        ),
    )

    result = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "reports" / "month_2",
        frozen_evidence_path=frozen_evidence_path,
    )

    assert load_calls == [
        {"eda_config_path": fixture.config, "split_run_path": None}
    ]
    assert len(fit_calls) == 1
    fitted_matrix, fitted_target, fitted_names = fit_calls[0]
    assert fitted_matrix is inputs.matrices["train"]
    assert fitted_target is inputs.targets["train"]
    assert fitted_names == inputs.feature_names
    assert len(score_calls) == 1
    assert score_calls[0] is inputs.matrices["validation"]
    assert matrices.accessed == ["train", "validation"]
    # Ranking, calibration, and capacity helpers each read validation once.
    assert targets.accessed == ["train", "validation", "validation", "validation"]
    assert [name for name, _, _ in evaluation_calls] == [
        "ranking",
        "calibration",
        "capacity",
    ]
    for _, evaluated_target, evaluated_scores in evaluation_calls:
        assert evaluated_target is inputs.targets["validation"]
        assert evaluated_scores is result.validation_scores
    assert result.training.metadata.training_split == "train"
    assert result.training.metadata.feature_names == inputs.feature_names
    assert result.feature_names == inputs.feature_names
    assert len(result.validation_scores) == len(inputs.targets["validation"])
    assert result.ranking.metrics.row_count == len(result.validation_scores)
    assert result.calibration.metrics.row_count == len(result.validation_scores)
    validation_count = len(inputs.targets["validation"])
    assert [row.capacity for row in result.capacity] == [0.05, 0.10, 0.20]
    assert [row.selected_count for row in result.capacity] == [
        ceil(validation_count * capacity) for capacity in (0.05, 0.10, 0.20)
    ]
    assert all(row.selected_count <= validation_count for row in result.capacity)

    artifact_paths = (
        result.report_artifacts.validation_path,
        result.report_artifacts.calibration_path,
        result.report_artifacts.capacity_path,
        result.report_artifacts.comparison_path,
        result.report_artifacts.markdown_path,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifact_paths)

    validation_report = pd.read_csv(result.report_artifacts.validation_path)
    assert {
        "model",
        "evaluated_split",
        "pr_auc",
        "roc_auc",
        "brier_score",
    }.issubset(validation_report.columns)
    assert validation_report.loc[0, "evaluated_split"] == "validation"
    validation_metrics = validation_report.loc[
        0, ["pr_auc", "roc_auc", "brier_score"]
    ].to_numpy(dtype=float)
    assert np.isfinite(validation_metrics).all()
    assert ((validation_metrics >= 0.0) & (validation_metrics <= 1.0)).all()

    capacity_report = pd.read_csv(result.report_artifacts.capacity_path)
    assert capacity_report["capacity"].tolist() == [0.05, 0.10, 0.20]
    assert capacity_report["selected_count"].tolist() == [
        ceil(validation_count * capacity) for capacity in (0.05, 0.10, 0.20)
    ]
    operational_metrics = capacity_report[
        ["precision_at_k", "recall_at_k"]
    ].to_numpy(dtype=float)
    assert np.isfinite(operational_metrics).all()
    assert ((operational_metrics >= 0.0) & (operational_metrics <= 1.0)).all()

    comparison_report = pd.read_csv(result.report_artifacts.comparison_path)
    assert comparison_report["metric"].tolist() == list(COMPARISON_METRICS)
    assert {
        "logistic_regression",
        "gradient_boosting",
        "difference_gb_minus_lr",
    }.issubset(comparison_report.columns)
    assert np.isfinite(
        comparison_report[
            [
                "logistic_regression",
                "gradient_boosting",
                "difference_gb_minus_lr",
            ]
        ].to_numpy(dtype=float)
    ).all()

    markdown_report = result.report_artifacts.markdown_path.read_text(
        encoding="utf-8"
    )
    assert "Frozen Logistic Regression Comparison" in markdown_report
    assert "No test probabilities were generated" in markdown_report
    assert "test labels were not used for fitting" in markdown_report
    report_directory = result.report_artifacts.markdown_path.parent
    assert not tuple(report_directory.glob("*gradient_boosting_test*"))
    assert not tuple(report_directory.glob("*phase_2_test*"))
    assert result.frozen_logistic_regression.frozen is True
    assert result.frozen_logistic_regression.evaluated_split == "validation"
    assert result.frozen_logistic_regression.split_id == inputs.split_id
    assert (
        result.frozen_logistic_regression.phase_9_contract_fingerprint
        == inputs.phase_9_contract.fingerprint
    )
    assert result.frozen_logistic_regression.row_count == len(
        result.validation_scores
    )
    assert result.comparison["metric"].tolist() == list(COMPARISON_METRICS)
    expected_difference = (
        result.comparison["gradient_boosting"]
        - result.comparison["logistic_regression"]
    )
    np.testing.assert_allclose(
        result.comparison["difference_gb_minus_lr"],
        expected_difference,
    )


def test_workflow_is_deterministic_for_fixed_configuration(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    frozen_evidence_path = _write_matching_frozen_evidence(tmp_path, fixture, inputs)
    first = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "first",
        frozen_evidence_path=frozen_evidence_path,
    )
    second = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "second",
        frozen_evidence_path=frozen_evidence_path,
    )

    np.testing.assert_array_equal(first.validation_scores, second.validation_scores)
    assert first.ranking == second.ranking
    assert first.calibration == second.calibration
    assert first.capacity == second.capacity
    assert first.feature_names == second.feature_names
    assert first.report_artifacts.comparison.equals(
        second.report_artifacts.comparison
    )


def test_workflow_rejects_frozen_evidence_from_a_different_split(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    validation_path, capacity_path = _write_matching_frozen_csvs(tmp_path, inputs)
    # Same population counts, but recorded against another frozen split.
    stale_snapshot = write_frozen_logistic_regression_evidence_snapshot(
        split_id="some-other-split",
        phase_9_contract_fingerprint=inputs.phase_9_contract.fingerprint,
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
        output_path=tmp_path / "stale_lr_evidence.json",
    )
    output_directory = tmp_path / "reports" / "month_2"

    with pytest.raises(ValueError, match="recorded for split"):
        run_gradient_boosting_workflow(
            eda_config_path=fixture.config,
            output_directory=output_directory,
            frozen_evidence_path=stale_snapshot,
        )
    assert not output_directory.exists()
