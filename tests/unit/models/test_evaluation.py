"""Unit tests for shared Month 1 evaluation metrics."""

import math

import numpy as np
import pytest

from urban_ops.models.evaluation import (
    EvaluationError,
    build_calibration_table,
    evaluate_binary_classifier,
    top_k_metrics,
)


def test_evaluate_binary_classifier_matches_manual_confusion_and_top_k() -> None:
    """Core metrics are computed from predictions and scores consistently."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.7])

    metrics = evaluate_binary_classifier(y_true, y_pred, y_score)

    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.precision_at_10_percent == 1.0
    assert metrics.recall_at_10_percent == 0.5


def test_top_k_uses_ceil_and_stable_tie_order() -> None:
    """Top-K selection uses max(1, ceil(n*fraction)) and stable order."""
    y_true = np.array([0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.1])

    metrics = top_k_metrics(y_true, y_score, fraction=0.50)

    assert metrics.selected_count == 2
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_build_calibration_table_uses_fixed_bins() -> None:
    """Calibration bins compare mean score with observed outcome rate."""
    table = build_calibration_table(
        [0, 1, 1, 0],
        [0.05, 0.15, 0.85, 0.95],
        n_bins=5,
    )

    assert table.shape[0] == 5
    assert table.loc[0, "row_count"] == 2
    assert table.loc[0, "positive_count"] == 1
    assert table.loc[4, "row_count"] == 2
    assert table.loc[4, "positive_count"] == 1
    assert table.loc[1, "row_count"] == 0
    assert np.isnan(table.loc[1, "observed_positive_rate"])


def test_single_class_auc_metrics_are_nan_but_other_metrics_defined() -> None:
    """Single-class targets do not produce misleading ROC/PR AUC values."""
    metrics = evaluate_binary_classifier([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])

    assert math.isnan(metrics.roc_auc)
    assert math.isnan(metrics.pr_auc)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_score", "message"),
    [
        ([], [], [], "must not be empty"),
        ([0, 1], [0], [0.1, 0.2], "lengths must match"),
        ([0, 2], [0, 1], [0.1, 0.2], "only 0/1"),
        ([0, 1], [0, 1], [0.1, 1.2], "in \\[0, 1\\]"),
        ([0, 1], [0, 1], [0.1, float("nan")], "finite"),
    ],
)
def test_evaluation_rejects_invalid_inputs(y_true, y_pred, y_score, message: str) -> None:
    """Malformed metric inputs fail explicitly."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_binary_classifier(y_true, y_pred, y_score)
