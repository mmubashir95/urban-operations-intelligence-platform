"""Deterministic median baseline and focused duration-regression metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DurationBaselineError(ValueError):
    """Raised when duration targets, predictions, or model state are invalid."""


def _duration_array(values: object, *, name: str) -> np.ndarray:
    """Return a validated non-empty one-dimensional duration array in hours."""
    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    else:
        try:
            array = np.asarray(values, dtype="float64").copy()
        except (TypeError, ValueError) as error:
            raise DurationBaselineError(f"{name} must contain numeric values.") from error
    if array.ndim != 1 or array.size == 0:
        raise DurationBaselineError(
            f"{name} must be a non-empty one-dimensional vector."
        )
    if not np.isfinite(array).all():
        raise DurationBaselineError(f"{name} must contain only finite values.")
    if (array < 0).any():
        raise DurationBaselineError(f"{name} must contain only nonnegative values.")
    return array


class MedianDurationBaseline:
    """Predict the unmodified median of eligible training durations."""

    median_duration_hours_: float

    def fit(self, y_train: object) -> "MedianDurationBaseline":
        """Fit and store the training-only median duration in hours."""
        target = _duration_array(y_train, name="y_train")
        self.median_duration_hours_ = float(np.median(target))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        """Return one constant fitted-median prediction per requested sample."""
        if not hasattr(self, "median_duration_hours_"):
            raise DurationBaselineError("MedianDurationBaseline must be fitted first.")
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise DurationBaselineError("n_samples must be a nonnegative integer.")
        if n_samples < 0:
            raise DurationBaselineError("n_samples must be a nonnegative integer.")
        return np.full(int(n_samples), self.median_duration_hours_, dtype="float64")


@dataclass(frozen=True)
class DurationRegressionMetrics:
    """Focused validation errors for a duration predictor, expressed in hours."""

    sample_count: int
    mae_hours: float
    rmse_hours: float


def evaluate_duration_predictions(
    y_true: object, predictions: object
) -> DurationRegressionMetrics:
    """Calculate MAE and RMSE after strict duration-vector validation."""
    actual = _duration_array(y_true, name="y_true")
    predicted = _duration_array(predictions, name="predictions")
    if len(actual) != len(predicted):
        raise DurationBaselineError("Targets and predictions must have equal lengths.")
    errors = actual - predicted
    return DurationRegressionMetrics(
        sample_count=len(actual),
        mae_hours=float(np.mean(np.abs(errors))),
        rmse_hours=float(np.sqrt(np.mean(np.square(errors)))),
    )
