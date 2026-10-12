"""Unit tests for the Phase 7.3 median-duration baseline and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_ops.models.duration_baseline import (
    DurationBaselineError,
    MedianDurationBaseline,
    evaluate_duration_predictions,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 3.0, 9.0], 3.0),
        ([0.0, 2.0, 4.0, 10000.0], 3.0),
        ([0.0, 0.5, 1.25], 0.5),
    ],
)
def test_fit_uses_deterministic_training_median(
    values: list[float], expected: float
) -> None:
    source = pd.Series(values)
    original = source.copy(deep=True)
    first = MedianDurationBaseline().fit(source)
    second = MedianDurationBaseline().fit(source)
    assert first.median_duration_hours_ == expected
    assert second.median_duration_hours_ == expected
    pd.testing.assert_series_equal(source, original)


@pytest.mark.parametrize(
    "values",
    [[], [-1.0], [1.0, np.nan], [1.0, np.inf], [1.0, -np.inf]],
)
def test_fit_rejects_empty_negative_missing_or_infinite_targets(
    values: list[float]
) -> None:
    with pytest.raises(DurationBaselineError):
        MedianDurationBaseline().fit(values)


def test_predict_requires_fit_and_returns_constant_length() -> None:
    with pytest.raises(DurationBaselineError, match="fitted"):
        MedianDurationBaseline().predict(1)
    baseline = MedianDurationBaseline().fit([1.0, 4.0, 7.0])
    predictions = baseline.predict(4)
    np.testing.assert_array_equal(predictions, np.full(4, 4.0))
    assert predictions.dtype == np.float64
    assert baseline.predict(0).shape == (0,)
    for invalid in (-1, 1.5, True):
        with pytest.raises(DurationBaselineError, match="n_samples"):
            baseline.predict(invalid)  # type: ignore[arg-type]


def test_metrics_calculate_mae_and_rmse_in_hours() -> None:
    metrics = evaluate_duration_predictions([0.0, 10.0, 20.0], [5.0, 5.0, 5.0])
    assert metrics.sample_count == 3
    assert metrics.mae_hours == pytest.approx(25 / 3)
    assert metrics.rmse_hours == pytest.approx(np.sqrt(275 / 3))
    assert metrics.rmse_hours > metrics.mae_hours


def test_perfect_predictions_have_zero_error() -> None:
    metrics = evaluate_duration_predictions([0.0, 1.5, 9000.0], [0.0, 1.5, 9000.0])
    assert metrics.mae_hours == 0.0
    assert metrics.rmse_hours == 0.0


@pytest.mark.parametrize(
    ("actual", "predictions"),
    [
        ([], []),
        ([1.0], []),
        ([1.0], [np.nan]),
        ([1.0], [np.inf]),
        ([1.0], [-1.0]),
    ],
)
def test_metrics_reject_empty_mismatched_or_invalid_values(
    actual: list[float], predictions: list[float]
) -> None:
    with pytest.raises(DurationBaselineError):
        evaluate_duration_predictions(actual, predictions)
