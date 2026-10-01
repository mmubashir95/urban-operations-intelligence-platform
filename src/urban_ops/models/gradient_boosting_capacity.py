"""Evaluate Gradient Boosting validation risks at standard review capacities.

This module delegates ranking, Top-K selection, counts, precision, and recall
to the shared Month 1 evaluation layer. It does not apply a score threshold,
select a capacity policy, refit a model, or access the test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.evaluation import (
    CapacityComparisonRow,
    CapacityLevel,
    compare_capacity_levels,
    get_standard_capacity_levels,
)


@dataclass(frozen=True)
class GradientBoostingCapacityResult:
    """Standard capacity definitions, shared results, and reporting table."""

    capacity_levels: tuple[CapacityLevel, ...]
    comparisons: tuple[CapacityComparisonRow, ...]
    table: pd.DataFrame


def build_gradient_boosting_capacity_table(
    comparisons: tuple[CapacityComparisonRow, ...],
) -> pd.DataFrame:
    """Map shared capacity results to the established Month 1 report schema."""
    table = pd.DataFrame(
        [
            {
                "model": "Gradient Boosting",
                "evaluated_split": "validation",
                "capacity": row.capacity,
                "capacity_pct": row.capacity * 100.0,
                "selected_count": row.selected_count,
                "captured_positive_count": row.captured_positive_count,
                "precision_at_k": row.precision,
                "recall_at_k": row.recall,
                "additional_selected_count": row.additional_selected_count,
                "additional_captured_positive_count": (
                    row.additional_captured_positive_count
                ),
                "additional_capacity_pct_points": (
                    None
                    if row.additional_capacity is None
                    else row.additional_capacity * 100.0
                ),
                "additional_recall": row.additional_recall,
                "additional_recall_pct_points": (
                    None
                    if row.additional_recall is None
                    else row.additional_recall * 100.0
                ),
            }
            for row in comparisons
        ]
    )
    for column in ("additional_selected_count", "additional_captured_positive_count"):
        table[column] = table[column].astype("Int64")
    return table


def evaluate_gradient_boosting_validation_capacity(
    inputs: FrozenBaselineInputs,
    validation_scores: object,
) -> GradientBoostingCapacityResult:
    """Evaluate unchanged raw validation scores at standard capacities."""
    if "validation" not in inputs.targets:
        raise ValueError(
            "Frozen Gradient Boosting inputs are missing the validation target."
        )
    y_validation = inputs.targets["validation"]
    capacity_levels = get_standard_capacity_levels(len(y_validation))
    comparisons = compare_capacity_levels(
        y_validation,
        validation_scores,
        capacities=tuple(level.capacity for level in capacity_levels),
    )
    return GradientBoostingCapacityResult(
        capacity_levels=capacity_levels,
        comparisons=comparisons,
        table=build_gradient_boosting_capacity_table(comparisons),
    )
