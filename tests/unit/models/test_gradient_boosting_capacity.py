"""Unit tests for shared Top-K evaluation of Gradient Boosting scores."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from urban_ops.models import gradient_boosting_capacity
from urban_ops.models.evaluation import (
    CapacityComparisonRow,
    compare_capacity_levels,
    get_standard_capacity_levels,
    rank_by_risk,
)
from urban_ops.models.gradient_boosting_capacity import (
    GradientBoostingCapacityResult,
    evaluate_gradient_boosting_validation_capacity,
)


class TestLabelsProtected(dict):
    """Allow validation labels while failing on test-label access."""

    __test__ = False

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Phase 2.9 attempted to access test labels.")
        return super().__getitem__(key)


def test_standard_capacity_comparison_receives_unchanged_inputs(monkeypatch) -> None:
    y_validation = pd.Series([0, 1, 0, 1, 1])
    validation_scores = np.asarray([0.10, 0.90, 0.30, 0.70, 0.50])
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=y_validation),
    )
    levels = get_standard_capacity_levels(len(y_validation))
    expected = compare_capacity_levels(y_validation, validation_scores)
    calls = []

    def spy_get_standard_capacity_levels(n_samples):
        calls.append(("levels", n_samples))
        return levels

    def spy_compare_capacity_levels(y_true, y_score, *, capacities):
        calls.append(("comparison", y_true, y_score, capacities))
        return expected

    monkeypatch.setattr(
        gradient_boosting_capacity,
        "get_standard_capacity_levels",
        spy_get_standard_capacity_levels,
    )
    monkeypatch.setattr(
        gradient_boosting_capacity,
        "compare_capacity_levels",
        spy_compare_capacity_levels,
    )

    result = evaluate_gradient_boosting_validation_capacity(
        inputs,
        validation_scores,
    )

    assert result.comparisons is expected
    assert calls == [
        ("levels", len(y_validation)),
        (
            "comparison",
            y_validation,
            validation_scores,
            (0.05, 0.10, 0.20),
        ),
    ]


def test_capacity_result_preserves_shared_contract_and_canonical_schema() -> None:
    y_validation = pd.Series([0, 1] * 10)
    scores = np.linspace(0.01, 0.99, len(y_validation))
    inputs = SimpleNamespace(targets={"validation": y_validation})

    result = evaluate_gradient_boosting_validation_capacity(inputs, scores)

    assert isinstance(result, GradientBoostingCapacityResult)
    assert all(isinstance(row, CapacityComparisonRow) for row in result.comparisons)
    assert [level.capacity for level in result.capacity_levels] == [
        0.05,
        0.10,
        0.20,
    ]
    assert [level.k for level in result.capacity_levels] == [1, 2, 4]
    assert result.comparisons == compare_capacity_levels(y_validation, scores)
    assert tuple(result.table.columns) == (
        "model",
        "evaluated_split",
        "capacity",
        "capacity_pct",
        "selected_count",
        "captured_positive_count",
        "precision_at_k",
        "recall_at_k",
        "additional_selected_count",
        "additional_captured_positive_count",
        "additional_capacity_pct_points",
        "additional_recall",
        "additional_recall_pct_points",
    )


def test_complete_ranking_and_capacity_invariants_are_preserved() -> None:
    y_validation = pd.Series([0, 1] * 10)
    scores = np.asarray(
        [
            0.20,
            0.90,
            0.80,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.19,
            0.18,
            0.17,
            0.16,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.10,
        ]
    )
    inputs = SimpleNamespace(targets={"validation": y_validation})

    result = evaluate_gradient_boosting_validation_capacity(inputs, scores)
    ranking = rank_by_risk(y_validation, scores)

    assert len(ranking) == len(y_validation)
    assert all(
        left.y_score >= right.y_score
        for left, right in zip(ranking, ranking[1:])
    )
    selected_counts = [row.selected_count for row in result.comparisons]
    recalls = [row.recall for row in result.comparisons]
    assert selected_counts == sorted(selected_counts)
    assert recalls == sorted(recalls)
    assert all(0.0 <= row.precision <= 1.0 for row in result.comparisons)
    assert all(0.0 <= row.recall <= 1.0 for row in result.comparisons)


def test_invalid_scores_fail_through_shared_ranking_validation() -> None:
    inputs = SimpleNamespace(targets={"validation": pd.Series([0, 1])})

    with pytest.raises(ValueError, match="finite"):
        evaluate_gradient_boosting_validation_capacity(
            inputs,
            np.asarray([0.1, float("nan")]),
        )


def test_test_labels_are_not_accessed() -> None:
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=pd.Series([0, 1] * 10))
    )

    result = evaluate_gradient_boosting_validation_capacity(
        inputs,
        np.linspace(0.01, 0.99, 20),
    )

    assert len(result.comparisons) == 3
