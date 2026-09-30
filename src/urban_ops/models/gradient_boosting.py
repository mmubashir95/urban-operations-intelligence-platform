"""Project-facing XGBoost wrapper for resolution-risk classification.

The wrapper consumes already-preprocessed matrices. It does not load project
data, create features, split data, evaluate predictions, tune hyperparameters,
or apply an operational threshold policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from urban_ops.models.baselines import DEFAULT_RANDOM_STATE
from urban_ops.models.evaluation import EvaluationError, classify_scores_at_threshold
from urban_ops.utils.paths import PROJECT_ROOT


GRADIENT_BOOSTING_CONFIG_PATH: Final = (
    PROJECT_ROOT / "configs/models/resolution_risk_gradient_boosting.yaml"
)
SUPPORTED_SELECTION_VERSION: Final = 1
SUPPORTED_IMPLEMENTATION: Final = "xgboost"
SUPPORTED_ESTIMATOR: Final = "XGBClassifier"
SUPPORTED_OBJECTIVE: Final = "binary:logistic"
SUPPORTED_EVAL_METRIC: Final = "logloss"


class GradientBoostingModelError(ValueError):
    """Raised when the XGBoost wrapper receives unsafe model inputs."""


@dataclass(frozen=True)
class GradientBoostingConfig:
    """Validated Phase 2.4 starting configuration for XGBClassifier."""

    selection_version: int
    implementation: str
    estimator: str
    objective: str
    eval_metric: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    random_state: int
    n_jobs: int

    @property
    def model_parameters(self) -> dict[str, object]:
        """Return only parameters accepted by the selected estimator."""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


def _mapping(value: object, *, field: str) -> dict[str, object]:
    """Return a string-keyed configuration mapping."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise GradientBoostingModelError(f"{field} must be a mapping.")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    """Return a positive non-boolean integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GradientBoostingModelError(f"{field} must be a positive integer.")
    return value


def _positive_number(value: object, *, field: str) -> float:
    """Return a finite positive numeric configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GradientBoostingModelError(f"{field} must be a positive number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise GradientBoostingModelError(f"{field} must be a positive number.")
    return result


def load_gradient_boosting_config(
    path: Path | str = GRADIENT_BOOSTING_CONFIG_PATH,
) -> GradientBoostingConfig:
    """Load and strictly validate the authoritative Phase 2.4 YAML contract."""
    config_path = Path(path)
    if not config_path.is_file():
        raise GradientBoostingModelError(
            f"Gradient Boosting configuration does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            field="configuration root",
        )
    except (OSError, yaml.YAMLError) as error:
        raise GradientBoostingModelError(
            f"Gradient Boosting configuration is invalid: {config_path}"
        ) from error

    expected_root = {
        "selection_version",
        "implementation",
        "estimator",
        "starting_configuration",
    }
    if set(root) != expected_root:
        raise GradientBoostingModelError(
            "Gradient Boosting configuration fields must be exactly "
            f"{sorted(expected_root)}."
        )
    parameters = _mapping(
        root["starting_configuration"], field="starting_configuration"
    )
    expected_parameters = {
        "objective",
        "eval_metric",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_state",
        "n_jobs",
    }
    if set(parameters) != expected_parameters:
        raise GradientBoostingModelError(
            "starting_configuration fields must be exactly "
            f"{sorted(expected_parameters)}."
        )
    if root["selection_version"] != SUPPORTED_SELECTION_VERSION:
        raise GradientBoostingModelError("selection_version must be 1.")
    if root["implementation"] != SUPPORTED_IMPLEMENTATION:
        raise GradientBoostingModelError("implementation must be xgboost.")
    if root["estimator"] != SUPPORTED_ESTIMATOR:
        raise GradientBoostingModelError("estimator must be XGBClassifier.")
    if parameters["objective"] != SUPPORTED_OBJECTIVE:
        raise GradientBoostingModelError("objective must be binary:logistic.")
    if parameters["eval_metric"] != SUPPORTED_EVAL_METRIC:
        raise GradientBoostingModelError("eval_metric must be logloss.")
    random_state = _positive_integer(
        parameters["random_state"], field="random_state"
    )
    if random_state != DEFAULT_RANDOM_STATE:
        raise GradientBoostingModelError(
            f"random_state must use the project seed {DEFAULT_RANDOM_STATE}."
        )
    return GradientBoostingConfig(
        selection_version=SUPPORTED_SELECTION_VERSION,
        implementation=SUPPORTED_IMPLEMENTATION,
        estimator=SUPPORTED_ESTIMATOR,
        objective=SUPPORTED_OBJECTIVE,
        eval_metric=SUPPORTED_EVAL_METRIC,
        n_estimators=_positive_integer(
            parameters["n_estimators"], field="n_estimators"
        ),
        learning_rate=_positive_number(
            parameters["learning_rate"], field="learning_rate"
        ),
        max_depth=_positive_integer(parameters["max_depth"], field="max_depth"),
        random_state=random_state,
        n_jobs=_positive_integer(parameters["n_jobs"], field="n_jobs"),
    )


def _matrix_shape(X: object, *, name: str) -> tuple[int, int]:
    """Return a two-dimensional matrix shape without copying or densifying X."""
    shape = getattr(X, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise GradientBoostingModelError(
            f"{name} must be a two-dimensional matrix-like object."
        )
    return int(shape[0]), int(shape[1])


def _feature_names(values: Sequence[object]) -> tuple[str, ...]:
    """Validate and preserve the exact caller-provided feature-name order."""
    if isinstance(values, (str, bytes)):
        raise GradientBoostingModelError(
            "feature_names must be a non-empty sequence of names."
        )
    try:
        names = tuple(values)
    except TypeError as error:
        raise GradientBoostingModelError(
            "feature_names must be a non-empty sequence of names."
        ) from error
    if not names:
        raise GradientBoostingModelError("feature_names must not be empty.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise GradientBoostingModelError(
            "feature_names must not contain null, non-text, or blank values."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GradientBoostingModelError(
            f"feature_names must be unique; duplicates={duplicates}."
        )
    return names  # type: ignore[return-value]


def _binary_training_target(values: object) -> np.ndarray:
    """Return a validated binary target without accepting arbitrary classes."""
    if isinstance(values, pd.Series):
        target = values.to_numpy(copy=False)
    else:
        target = np.asarray(values)
    if target.ndim != 1 or target.size == 0:
        raise GradientBoostingModelError(
            "y_train must be a non-empty one-dimensional target."
        )
    if pd.isna(target).any():
        raise GradientBoostingModelError("y_train must not contain null values.")
    observed = set(target.tolist())
    if not observed.issubset({0, 1, False, True}):
        raise GradientBoostingModelError(
            f"y_train must contain only 0/1 values; observed={observed}."
        )
    if observed not in ({0, 1}, {False, True}):
        raise GradientBoostingModelError(
            "y_train must contain both target classes {0, 1}."
        )
    return target


class GradientBoostedRiskModel:
    """Stable project interface around XGBoost's binary classifier."""

    def __init__(
        self,
        *,
        config_path: Path | str = GRADIENT_BOOSTING_CONFIG_PATH,
    ) -> None:
        """Construct the wrapper from the frozen Phase 2.4 configuration."""
        self.config = load_gradient_boosting_config(config_path)
        self._model = XGBClassifier(**self.config.model_parameters)

    def fit(
        self,
        X_train: object,
        y_train: object,
        *,
        feature_names: Sequence[object],
    ) -> "GradientBoostedRiskModel":
        """Fit the wrapped estimator on caller-supplied preprocessed training data."""
        row_count, feature_count = _matrix_shape(X_train, name="X_train")
        target = _binary_training_target(y_train)
        if row_count != len(target):
            raise GradientBoostingModelError(
                "X_train row count does not match y_train length: "
                f"{row_count} != {len(target)}."
            )
        names = _feature_names(feature_names)
        if feature_count != len(names):
            raise GradientBoostingModelError(
                "X_train feature count does not match feature_names: "
                f"{feature_count} != {len(names)}."
            )

        self._model.fit(X_train, target)
        if tuple(self._model.classes_.tolist()) != (0, 1):
            raise GradientBoostingModelError(
                "Fitted XGBClassifier classes do not match binary classes (0, 1)."
            )
        self.feature_names_ = names
        self.feature_count_ = feature_count
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Return validated raw class probabilities from XGBClassifier."""
        row_count = self._validate_prediction_input(X)
        probabilities = np.asarray(self._model.predict_proba(X))
        if probabilities.shape != (row_count, 2):
            raise GradientBoostingModelError(
                "XGBClassifier probability output must have shape "
                f"({row_count}, 2); observed {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise GradientBoostingModelError(
                "XGBClassifier probability output must contain only finite values."
            )
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise GradientBoostingModelError(
                "XGBClassifier probability output must be within [0, 1]."
            )
        return probabilities

    def predict_score(self, X: object) -> np.ndarray:
        """Return the raw probability of positive missed-target class 1."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: object, *, threshold: float = 0.5) -> np.ndarray:
        """Threshold positive-class scores using a caller-supplied cutoff."""
        scores = self.predict_score(X)
        try:
            return classify_scores_at_threshold(scores, threshold=threshold)
        except EvaluationError as error:
            raise GradientBoostingModelError(str(error)) from error

    def _validate_prediction_input(self, X: object) -> int:
        """Require fitted state and the exact training feature width."""
        self._require_fitted()
        row_count, feature_count = _matrix_shape(X, name="X")
        if feature_count != self.feature_count_:
            raise GradientBoostingModelError(
                "Prediction feature count does not match training feature count: "
                f"{feature_count} != {self.feature_count_}."
            )
        return row_count

    def _require_fitted(self) -> None:
        """Raise a project-level error before consulting XGBoost state."""
        if not hasattr(self, "feature_names_"):
            raise GradientBoostingModelError(
                "GradientBoostedRiskModel must be fitted before prediction."
            )
