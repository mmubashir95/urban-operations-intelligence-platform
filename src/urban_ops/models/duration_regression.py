"""Controlled Linear Regression and XGBoost duration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from urban_ops.models.baselines import DEFAULT_RANDOM_STATE
from urban_ops.utils.paths import PROJECT_ROOT


CONFIG_PATH: Final = PROJECT_ROOT / "configs/models/resolution_duration_regression.yaml"


class DurationRegressionError(ValueError):
    """Raised when regression configuration or model inputs are unsafe."""


@dataclass(frozen=True)
class DurationRegressionConfig:
    """Explicit initial configurations shared by the two regressors."""

    fit_intercept: bool
    objective: str
    eval_metric: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int
    n_jobs: int
    nonnegative_prediction_policy: str

    @property
    def xgboost_parameters(self) -> dict[str, object]:
        """Return the exact XGBRegressor constructor parameters."""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


@dataclass(frozen=True)
class DurationPredictions:
    """Raw and nonnegative prediction vectors plus clipping evidence."""

    raw: np.ndarray
    nonnegative: np.ndarray
    negative_raw_count: int


def load_duration_regression_config(
    path: Path | str = CONFIG_PATH,
) -> DurationRegressionConfig:
    """Load and strictly validate the controlled Phase 7.4 configuration."""
    try:
        root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        linear = root["linear_regression"]
        xgb = root["xgboost"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise DurationRegressionError("Invalid duration regression configuration.") from error
    expected_root = {
        "config_version", "random_state", "nonnegative_prediction_policy",
        "linear_regression", "xgboost",
    }
    expected_xgb = {
        "objective", "eval_metric", "n_estimators", "learning_rate",
        "max_depth", "random_state", "n_jobs",
    }
    if set(root) != expected_root or set(linear) != {"fit_intercept"} or set(xgb) != expected_xgb:
        raise DurationRegressionError("Duration regression configuration fields differ from contract.")
    if root["config_version"] != 1:
        raise DurationRegressionError("Only config_version 1 is supported.")
    if root["random_state"] != DEFAULT_RANDOM_STATE or xgb["random_state"] != DEFAULT_RANDOM_STATE:
        raise DurationRegressionError("Regression models must use the project random state.")
    if root["nonnegative_prediction_policy"] != "clip_at_zero":
        raise DurationRegressionError("Only clip_at_zero prediction policy is supported.")
    if xgb["objective"] != "reg:squarederror" or xgb["eval_metric"] != "rmse":
        raise DurationRegressionError("XGBoost must use squared-error regression and RMSE.")
    for name in ("n_estimators", "max_depth", "n_jobs"):
        if isinstance(xgb[name], bool) or not isinstance(xgb[name], int) or xgb[name] <= 0:
            raise DurationRegressionError(f"xgboost.{name} must be a positive integer.")
    learning_rate = float(xgb["learning_rate"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise DurationRegressionError("xgboost.learning_rate must be positive and finite.")
    if not isinstance(linear["fit_intercept"], bool):
        raise DurationRegressionError("linear_regression.fit_intercept must be boolean.")
    return DurationRegressionConfig(
        fit_intercept=linear["fit_intercept"],
        objective=xgb["objective"], eval_metric=xgb["eval_metric"],
        n_estimators=xgb["n_estimators"], learning_rate=learning_rate,
        max_depth=xgb["max_depth"], random_state=xgb["random_state"],
        n_jobs=xgb["n_jobs"],
        nonnegative_prediction_policy=root["nonnegative_prediction_policy"],
    )


def _matrix_shape(matrix: object, *, name: str) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise DurationRegressionError(f"{name} must be a two-dimensional matrix.")
    return int(shape[0]), int(shape[1])


def _target(values: object) -> np.ndarray:
    try:
        target = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as error:
        raise DurationRegressionError("y_train must contain numeric durations.") from error
    if target.ndim != 1 or target.size == 0 or not np.isfinite(target).all() or (target < 0).any():
        raise DurationRegressionError("y_train must be non-empty, finite, and nonnegative.")
    return target


def _names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise DurationRegressionError("feature_names must be unique non-empty strings.")
    return names


def enforce_nonnegative_predictions(values: object) -> DurationPredictions:
    """Validate raw predictions and explicitly clip negative hours to zero."""
    raw = np.asarray(values, dtype="float64").copy()
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise DurationRegressionError("Predictions must be a finite one-dimensional vector.")
    negative_count = int((raw < 0).sum())
    return DurationPredictions(raw, np.maximum(raw, 0.0), negative_count)


class DurationRegressor:
    """Small common interface around one sklearn-compatible regressor."""

    def __init__(self, estimator: object, *, model_name: str) -> None:
        self.estimator = estimator
        self.model_name = model_name

    def fit(self, X_train: object, y_train: object, *, feature_names: Sequence[str]) -> "DurationRegressor":
        """Validate aligned training data and fit the wrapped estimator."""
        rows, columns = _matrix_shape(X_train, name="X_train")
        target = _target(y_train)
        names = _names(feature_names)
        if rows != len(target):
            raise DurationRegressionError("X_train row count must match y_train.")
        if columns != len(names):
            raise DurationRegressionError("X_train feature count must match feature_names.")
        self.estimator.fit(X_train, target)  # type: ignore[attr-defined]
        self.feature_names_ = names
        self.feature_count_ = columns
        return self

    def predict(self, X: object) -> DurationPredictions:
        """Predict, validate shape, and apply the documented zero floor."""
        if not hasattr(self, "feature_count_"):
            raise DurationRegressionError(f"{self.model_name} must be fitted first.")
        rows, columns = _matrix_shape(X, name="X")
        if columns != self.feature_count_:
            raise DurationRegressionError("Prediction feature count differs from training.")
        raw = np.asarray(self.estimator.predict(X), dtype="float64")  # type: ignore[attr-defined]
        if raw.shape != (rows,):
            raise DurationRegressionError("Estimator returned an invalid prediction shape.")
        return enforce_nonnegative_predictions(raw)


def build_duration_regressors(config: DurationRegressionConfig) -> tuple[DurationRegressor, DurationRegressor]:
    """Construct the controlled linear and nonlinear regression models."""
    linear = DurationRegressor(
        LinearRegression(fit_intercept=config.fit_intercept),
        model_name="Linear Regression",
    )
    xgboost = DurationRegressor(
        XGBRegressor(**config.xgboost_parameters), model_name="XGBRegressor"
    )
    return linear, xgboost


def comparison_improvement(baseline: float, model: float) -> tuple[float, float]:
    """Return absolute and percentage improvement, positive when model is better."""
    if not np.isfinite([baseline, model]).all() or baseline <= 0 or model < 0:
        raise DurationRegressionError("Comparison metrics must be finite and baseline positive.")
    absolute = baseline - model
    return float(absolute), float(absolute / baseline * 100.0)
