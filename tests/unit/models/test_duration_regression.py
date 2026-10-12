"""Unit tests for controlled Phase 7.4 regression models and comparison math."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from urban_ops.models.duration_regression import (
    DurationRegressionError,
    build_duration_regressors,
    comparison_improvement,
    enforce_nonnegative_predictions,
    load_duration_regression_config,
)


def _data():
    X = sparse.csr_matrix([
        [0, 0, 1, 0], [6, 1, 2, 0], [12, 5, 3, 1], [18, 6, 4, 1],
        [23, 4, 5, 0], [3, 2, 6, 0],
    ], dtype="float64")
    y = np.asarray([0.0, 10.0, 30.0, 50.0, 70.0, 15.0])
    return X, y


def test_controlled_configuration_and_models_fit_deterministically() -> None:
    config = load_duration_regression_config()
    assert config.xgboost_parameters == {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
        "random_state": 20260806, "n_jobs": 1,
    }
    X, y = _data()
    names = ("created_hour", "created_day_of_week", "created_month", "is_weekend")
    first_models = build_duration_regressors(config)
    second_models = build_duration_regressors(config)
    for first, second in zip(first_models, second_models, strict=True):
        X_before, y_before = X.copy(), y.copy()
        first.fit(X, y, feature_names=names)
        second.fit(X, y, feature_names=names)
        first_predictions = first.predict(X)
        second_predictions = second.predict(X)
        np.testing.assert_allclose(first_predictions.raw, second_predictions.raw)
        assert first.feature_names_ == names
        assert first_predictions.nonnegative.shape == (len(y),)
        assert np.isfinite(first_predictions.nonnegative).all()
        assert (first_predictions.nonnegative >= 0).all()
        assert (X != X_before).nnz == 0
        np.testing.assert_array_equal(y, y_before)


def test_models_reject_unfitted_and_misaligned_inputs() -> None:
    config = load_duration_regression_config()
    model, _ = build_duration_regressors(config)
    X, y = _data()
    names = ("a", "b", "c", "d")
    with pytest.raises(DurationRegressionError, match="fitted"):
        model.predict(X)
    with pytest.raises(DurationRegressionError, match="row count"):
        model.fit(X, y[:-1], feature_names=names)
    with pytest.raises(DurationRegressionError, match="feature count"):
        model.fit(X, y, feature_names=names[:-1])
    model.fit(X, y, feature_names=names)
    with pytest.raises(DurationRegressionError, match="feature count"):
        model.predict(X[:, :3])


def test_nonnegative_policy_preserves_raw_and_counts_clipping() -> None:
    result = enforce_nonnegative_predictions([-2.5, 0.0, 4.0, -0.1])
    np.testing.assert_array_equal(result.raw, [-2.5, 0.0, 4.0, -0.1])
    np.testing.assert_array_equal(result.nonnegative, [0.0, 0.0, 4.0, 0.0])
    assert result.negative_raw_count == 2
    with pytest.raises(DurationRegressionError, match="finite"):
        enforce_nonnegative_predictions([np.inf])


def test_comparison_improvement_reports_positive_and_negative_results() -> None:
    assert comparison_improvement(100.0, 80.0) == (20.0, 20.0)
    absolute, percentage = comparison_improvement(100.0, 120.0)
    assert absolute == -20.0
    assert percentage == -20.0
