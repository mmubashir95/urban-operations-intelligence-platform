"""Unit tests for shared calibration evaluation of Gradient Boosting scores."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from urban_ops.models import gradient_boosting_calibration
from urban_ops.models.evaluation import (
    CALIBRATION_N_BINS,
    CalibrationEvaluation,
    build_calibration_table,
    evaluate_calibration,
)
from urban_ops.models.gradient_boosting_calibration import (
    GradientBoostingCalibrationResult,
    evaluate_gradient_boosting_validation_calibration,
)


class TestLabelsProtected(dict):
    """Allow validation labels while failing on test-label access."""

    __test__ = False

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Phase 2.8 attempted to access test labels.")
        return super().__getitem__(key)


def test_shared_evaluators_receive_unchanged_validation_inputs(monkeypatch) -> None:
    y_validation = pd.Series([0, 1, 0, 1])
    validation_scores = np.asarray([0.08, 0.74, 0.21, 0.91])
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=y_validation),
    )
    expected_calibration = evaluate_calibration(y_validation, validation_scores)
    expected_table = build_calibration_table(y_validation, validation_scores)
    calls = []

    def spy_evaluate_calibration(y_true, y_score):
        calls.append(("calibration", y_true, y_score))
        return expected_calibration

    def spy_build_calibration_table(y_true, y_score, *, n_bins):
        calls.append(("table", y_true, y_score, n_bins))
        return expected_table

    monkeypatch.setattr(
        gradient_boosting_calibration,
        "evaluate_calibration",
        spy_evaluate_calibration,
    )
    monkeypatch.setattr(
        gradient_boosting_calibration,
        "build_calibration_table",
        spy_build_calibration_table,
    )

    result = evaluate_gradient_boosting_validation_calibration(
        inputs,
        validation_scores,
    )

    assert result.calibration is expected_calibration
    assert result.table is expected_table
    assert calls == [
        ("calibration", y_validation, validation_scores),
        ("table", y_validation, validation_scores, CALIBRATION_N_BINS),
    ]


def test_calibration_result_preserves_shared_contract_and_all_rows() -> None:
    y_validation = pd.Series([0, 1, 0, 1, 1, 0])
    scores = np.asarray([0.05, 0.24, 0.36, 0.68, 0.84, 0.95])
    inputs = SimpleNamespace(targets={"validation": y_validation})

    result = evaluate_gradient_boosting_validation_calibration(inputs, scores)

    assert isinstance(result, GradientBoostingCalibrationResult)
    assert isinstance(result.calibration, CalibrationEvaluation)
    assert result.calibration == evaluate_calibration(y_validation, scores)
    assert np.isfinite(result.calibration.metrics.brier_score)
    assert 0.0 <= result.calibration.metrics.brier_score <= 1.0
    assert result.table.equals(build_calibration_table(y_validation, scores))
    assert tuple(result.table.columns) == (
        "bin_index",
        "lower_bound",
        "upper_bound",
        "row_count",
        "mean_predicted_risk",
        "observed_positive_rate",
        "positive_count",
    )
    assert len(result.table) == CALIBRATION_N_BINS
    assert int(result.table["row_count"].sum()) == len(y_validation)


def test_invalid_score_contract_fails_through_shared_evaluator() -> None:
    inputs = SimpleNamespace(targets={"validation": pd.Series([0, 1])})

    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        evaluate_gradient_boosting_validation_calibration(
            inputs,
            np.asarray([0.1, 1.1]),
        )


def test_test_labels_are_not_accessed() -> None:
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=pd.Series([0, 1, 0, 1]))
    )

    result = evaluate_gradient_boosting_validation_calibration(
        inputs,
        np.asarray([0.1, 0.8, 0.2, 0.9]),
    )

    assert result.calibration.metrics.row_count == 4