"""Generate aligned validation risks from a fitted Gradient Boosting model.

This module does not fit models, access labels, apply thresholds, rank rows,
calculate metrics, persist artifacts, or score the test split.
"""

from __future__ import annotations

import numpy as np

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel


class GradientBoostingValidationScoreError(ValueError):
    """Raised when validation score generation violates its output contract."""


def generate_gradient_boosting_validation_scores(
    model: GradientBoostedRiskModel,
    inputs: FrozenBaselineInputs,
) -> np.ndarray:
    """Return positive-class risks for frozen validation rows in input order."""
    if "validation" not in inputs.matrices:
        raise GradientBoostingValidationScoreError(
            "Frozen Gradient Boosting inputs are missing the validation matrix."
        )
    X_validation = inputs.matrices["validation"]
    shape = getattr(X_validation, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise GradientBoostingValidationScoreError(
            "Frozen validation input must be a two-dimensional matrix."
        )

    scores = np.asarray(model.predict_score(X_validation))
    if scores.ndim != 1:
        raise GradientBoostingValidationScoreError(
            "Validation risk scores must be one-dimensional; "
            f"observed shape {scores.shape}."
        )
    row_count = int(shape[0])
    if len(scores) != row_count:
        raise GradientBoostingValidationScoreError(
            "Validation score count does not match validation row count: "
            f"{len(scores)} != {row_count}."
        )
    if not np.isfinite(scores).all():
        raise GradientBoostingValidationScoreError(
            "Validation risk scores must contain only finite values."
        )
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise GradientBoostingValidationScoreError(
            "Validation risk scores must be within [0, 1]."
        )
    return scores
