"""Evaluate Gradient Boosting validation scores with the Month 1 ranker.

This module contains no metric formulas. It delegates the frozen validation
target and existing Phase 2.6 scores directly to the shared evaluator.
"""

from __future__ import annotations

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.evaluation import RankingEvaluation, evaluate_ranking


def evaluate_gradient_boosting_validation_ranking(
    inputs: FrozenBaselineInputs,
    validation_scores: object,
) -> RankingEvaluation:
    """Return shared PR-AUC/ROC-AUC evaluation for validation scores."""
    if "validation" not in inputs.targets:
        raise ValueError(
            "Frozen Gradient Boosting inputs are missing the validation target."
        )
    y_validation = inputs.targets["validation"]
    return evaluate_ranking(y_validation, validation_scores)
