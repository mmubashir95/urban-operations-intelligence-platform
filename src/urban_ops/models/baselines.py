"""Month 1 baseline classifiers for resolution-risk modelling.

These estimators consume frozen model inputs or approved creation-time grouping
metadata. They do not refit preprocessing, alter split membership, or inspect
validation/test labels during training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Final
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


DEFAULT_RANDOM_STATE: Final = 20260806
DEFAULT_THRESHOLD_CANDIDATES: Final = tuple(
    round(value, 2) for value in np.arange(0.05, 0.951, 0.01)
)


class BaselineModelError(ValueError):
    """Raised when a baseline is used with invalid inputs or before fitting."""


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-only threshold selection metadata."""

    selected_threshold: float
    selection_metric: str
    validation_metric_value: float
    tie_breaking: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


def _binary_series(values: object, *, name: str) -> pd.Series:
    """Return a non-empty binary integer series."""
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if series.ndim != 1 or series.empty:
        raise BaselineModelError(f"{name} must be a non-empty one-dimensional vector.")
    if series.isna().any():
        raise BaselineModelError(f"{name} must not contain null values.")
    observed = set(series.unique().tolist())
    if not observed.issubset({0, 1, False, True}):
        raise BaselineModelError(f"{name} must contain only 0/1 values.")
    return series.astype(int)


def _validate_row_count(X: object, expected: int) -> int:
    """Return feature row count or raise a clear mismatch error."""
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise BaselineModelError("X must be a two-dimensional feature matrix.")
    row_count = int(X.shape[0])
    if row_count != expected:
        raise BaselineModelError(
            f"X row count {row_count} does not match expected {expected}."
        )
    return row_count


def _score_array(scores: object, *, name: str) -> np.ndarray:
    """Validate finite risk/probability scores in [0, 1]."""
    array = np.asarray(scores, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise BaselineModelError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise BaselineModelError(f"{name} must contain only finite values.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise BaselineModelError(f"{name} values must be in [0, 1].")
    return array


class MajorityClassBaseline:
    """Deterministic classifier that always predicts the train majority class."""

    majority_class_: int
    class_distribution_: dict[int, int]
    training_positive_rate_: float

    def fit(self, X: object, y: object) -> "MajorityClassBaseline":
        """Learn the majority class from training labels only."""
        target = _binary_series(y, name="y_train")
        _validate_row_count(X, len(target))
        counts = target.value_counts().to_dict()
        zero_count = int(counts.get(0, 0))
        one_count = int(counts.get(1, 0))
        self.majority_class_ = 1 if one_count > zero_count else 0
        self.class_distribution_ = {0: zero_count, 1: one_count}
        self.training_positive_rate_ = float(one_count / len(target))
        return self

    def predict(self, X: object) -> np.ndarray:
        """Predict the learned majority class for every row."""
        self._require_fitted()
        row_count = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        return np.full(row_count, self.majority_class_, dtype=int)

    def predict_proba(self, X: object) -> np.ndarray:
        """Return the empirical train class distribution for every row."""
        self._require_fitted()
        row_count = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        positive = self.training_positive_rate_
        return np.column_stack(
            [
                np.full(row_count, 1.0 - positive, dtype=float),
                np.full(row_count, positive, dtype=float),
            ]
        )

    def _require_fitted(self) -> None:
        """Raise if the baseline has not been fitted."""
        if not hasattr(self, "majority_class_"):
            raise BaselineModelError("MajorityClassBaseline must be fitted first.")


class HistoricalRateBaseline:
    """Training-only group target-rate risk model with a global fallback."""

    def __init__(self, group_column: str) -> None:
        """Configure the approved grouping column used for rates."""
        if not group_column.strip():
            raise BaselineModelError("group_column must be non-empty.")
        self.group_column = group_column

    def fit(
        self, groups: pd.DataFrame | pd.Series | Sequence[object], y: object
    ) -> "HistoricalRateBaseline":
        """Aggregate training target rates by configured group only."""
        group_values = self._extract_group(groups)
        target = _binary_series(y, name="y_train")
        if len(group_values) != len(target):
            raise BaselineModelError("Training groups and target lengths must match.")
        if group_values.isna().any():
            raise BaselineModelError("Training groups must not contain null values.")
        table = (
            pd.DataFrame({"group": group_values.astype("string"), "target": target})
            .groupby("group", sort=True)["target"]
            .agg(["mean", "count"])
        )
        self.global_rate_ = float(target.mean())
        self.group_rates_ = {
            str(group): float(row["mean"]) for group, row in table.iterrows()
        }
        self.group_counts_ = {
            str(group): int(row["count"]) for group, row in table.iterrows()
        }
        return self

    def predict_score(
        self, groups: pd.DataFrame | pd.Series | Sequence[object]
    ) -> np.ndarray:
        """Return group risk scores with global fallback for unseen groups."""
        self._require_fitted()
        group_values = self._extract_group(groups)
        if group_values.isna().any():
            raise BaselineModelError("Scoring groups must not contain null values.")
        return np.asarray(
            [
                self.group_rates_.get(str(value), self.global_rate_)
                for value in group_values.astype("string")
            ],
            dtype=float,
        )

    def predict(
        self, groups: pd.DataFrame | pd.Series | Sequence[object], *, threshold: float
    ) -> np.ndarray:
        """Threshold group risk scores into binary predictions."""
        return (self.predict_score(groups) >= float(threshold)).astype(int)

    def _extract_group(
        self, groups: pd.DataFrame | pd.Series | Sequence[object]
    ) -> pd.Series:
        """Extract the configured group series without changing row order."""
        if isinstance(groups, pd.DataFrame):
            if self.group_column not in groups:
                raise BaselineModelError(
                    f"Group column {self.group_column!r} is missing."
                )
            return groups[self.group_column]
        if isinstance(groups, pd.Series):
            return groups
        return pd.Series(list(groups))

    def _require_fitted(self) -> None:
        """Raise if training rates have not been fitted."""
        if not hasattr(self, "global_rate_"):
            raise BaselineModelError("HistoricalRateBaseline must be fitted first.")


class RuleBasedHistoricalRateBaseline:
    """Binary classifier using validation-selected threshold on historical rates."""

    def __init__(self, risk_model: HistoricalRateBaseline) -> None:
        """Wrap a fitted historical risk model."""
        self.risk_model = risk_model

    def select_threshold(
        self,
        validation_groups: pd.DataFrame | pd.Series | Sequence[object],
        y_validation: object,
        *,
        candidates: Iterable[float] = DEFAULT_THRESHOLD_CANDIDATES,
    ) -> ThresholdSelection:
        """Choose the F1-maximizing threshold using validation labels only.

        Ties are resolved by choosing the highest threshold, which is stable and
        favors fewer positive alerts when F1 is indistinguishable.
        """
        y_true = _binary_series(y_validation, name="y_validation")
        scores = self.risk_model.predict_score(validation_groups)
        if len(scores) != len(y_true):
            raise BaselineModelError("Validation scores and labels must align.")
        best_threshold: float | None = None
        best_value = -1.0
        for raw_threshold in candidates:
            threshold = float(raw_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise BaselineModelError("Threshold candidates must be in [0, 1].")
            value = float(f1_score(y_true, scores >= threshold, zero_division=0))
            if value > best_value or (
                np.isclose(value, best_value)
                and (best_threshold is None or threshold > best_threshold)
            ):
                best_value = value
                best_threshold = threshold
        if best_threshold is None:
            raise BaselineModelError("At least one threshold candidate is required.")
        self.threshold_selection_ = ThresholdSelection(
            selected_threshold=best_threshold,
            selection_metric="validation_f1",
            validation_metric_value=best_value,
            tie_breaking="highest_threshold",
        )
        return self.threshold_selection_

    def predict_score(
        self, groups: pd.DataFrame | pd.Series | Sequence[object]
    ) -> np.ndarray:
        """Return underlying historical risk scores."""
        return self.risk_model.predict_score(groups)

    def predict(self, groups: pd.DataFrame | pd.Series | Sequence[object]) -> np.ndarray:
        """Apply the selected threshold to historical risk scores."""
        if not hasattr(self, "threshold_selection_"):
            raise BaselineModelError("Threshold must be selected before prediction.")
        return (
            self.predict_score(groups) >= self.threshold_selection_.selected_threshold
        ).astype(int)


class LogisticRegressionBaseline:
    """Sparse logistic regression over frozen preprocessed matrices."""

    def __init__(
        self,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = 1000,
    ) -> None:
        """Configure a deterministic sparse-compatible sklearn estimator."""
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = LogisticRegression(
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(
        self,
        X: csr_matrix,
        y: object,
        *,
        feature_names: Sequence[str],
    ) -> "LogisticRegressionBaseline":
        """Fit on train CSR features and verify coefficient/feature alignment."""
        if not sparse.isspmatrix_csr(X):
            raise BaselineModelError("Logistic regression baseline requires CSR X.")
        target = _binary_series(y, name="y_train")
        _validate_row_count(X, len(target))
        if X.shape[1] != len(feature_names):
            raise BaselineModelError("Feature count does not match feature_names.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(X, target)
        convergence = [
            warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
        ]
        if convergence:
            raise BaselineModelError(
                "Logistic regression failed to converge; increase max_iter explicitly."
            )
        coefficient_count = int(self.model.coef_.shape[1])
        if coefficient_count != len(feature_names):
            raise BaselineModelError(
                "Coefficient count does not match frozen feature count."
            )
        self.feature_names_ = tuple(feature_names)
        self.coefficient_count_ = coefficient_count
        return self

    def predict_score(self, X: csr_matrix) -> np.ndarray:
        """Return positive-class probabilities for CSR inputs."""
        self._require_fitted()
        if not sparse.isspmatrix_csr(X):
            raise BaselineModelError("Logistic regression baseline requires CSR X.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: csr_matrix, *, threshold: float = 0.5) -> np.ndarray:
        """Threshold positive-class probabilities into binary predictions."""
        return (self.predict_score(X) >= threshold).astype(int)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Return sklearn positive and negative class probabilities."""
        self._require_fitted()
        if not sparse.isspmatrix_csr(X):
            raise BaselineModelError("Logistic regression baseline requires CSR X.")
        return self.model.predict_proba(X)

    def _require_fitted(self) -> None:
        """Raise if sklearn state has not been fitted."""
        if not hasattr(self, "feature_names_"):
            raise BaselineModelError("LogisticRegressionBaseline must be fitted first.")


def predict_from_scores(scores: object, threshold: float) -> np.ndarray:
    """Threshold validated risk scores into binary predictions."""
    return (_score_array(scores, name="scores") >= float(threshold)).astype(int)
