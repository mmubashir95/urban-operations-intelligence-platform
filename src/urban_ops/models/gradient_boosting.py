"""Project-facing XGBoost wrapper for resolution-risk classification.

The wrapper consumes already-preprocessed matrices. It does not load project
data, create features, split data, evaluate predictions, tune hyperparameters,
or apply an operational threshold policy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from urban_ops.models.baselines import DEFAULT_RANDOM_STATE
from urban_ops.models.evaluation import EvaluationError, classify_scores_at_threshold


class GradientBoostingModelError(ValueError):
    """Raised when the XGBoost wrapper receives unsafe model inputs."""


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
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        """Construct the deterministic initial wrapper configuration."""
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )

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
