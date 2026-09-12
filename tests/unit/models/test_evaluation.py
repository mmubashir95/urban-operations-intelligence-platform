"""Unit tests for shared Month 1 evaluation metrics."""

import math

import numpy as np
import pytest

from urban_ops.models.evaluation import (
    PR_AUC_DEFINITION,
    BasicClassificationMetrics,
    EvaluationError,
    RankingEvaluation,
    build_calibration_table,
    evaluate_basic_classifier,
    evaluate_binary_classifier,
    evaluate_ranking,
    metrics_row,
    top_k_metrics,
)


def test_basic_evaluation_matches_hand_calculated_metric_definitions() -> None:
    """Small counts map directly to the documented Phase 1 formulas."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]

    metrics = evaluate_basic_classifier(y_true, y_pred)

    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_negative == 2
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.row_count == 6
    assert metrics.positive_count == 3
    assert metrics.negative_count == 3
    assert metrics.positive_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_counts", "expected_metrics"),
    [
        (
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            (2, 0, 0, 2),
            (1.0, 1.0, 1.0, 1.0),
        ),
        (
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            (0, 2, 2, 0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        (
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            (2, 0, 2, 0),
            (0.5, 0.0, 0.0, 0.0),
        ),
        (
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            (0, 2, 0, 2),
            (0.5, 0.5, 1.0, 2 / 3),
        ),
    ],
)
def test_basic_evaluation_edge_cases(
    y_true,
    y_pred,
    expected_counts: tuple[int, int, int, int],
    expected_metrics: tuple[float, float, float, float],
) -> None:
    """Perfect, incorrect, no-positive, and all-positive predictions are stable."""
    metrics = evaluate_basic_classifier(y_true, y_pred)

    assert (
        metrics.true_negative,
        metrics.false_positive,
        metrics.false_negative,
        metrics.true_positive,
    ) == expected_counts
    assert (
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    ) == pytest.approx(expected_metrics)


def test_basic_evaluation_uses_missed_target_as_positive_class() -> None:
    """Label 1 is a missed target, fixing TP/FP/FN/TN orientation."""
    metrics = evaluate_basic_classifier([1, 0, 1, 0], [1, 1, 0, 0])

    assert metrics.true_positive == 1  # Actual miss correctly flagged.
    assert metrics.false_positive == 1  # On-time complaint falsely flagged.
    assert metrics.false_negative == 1  # Actual miss not flagged.
    assert metrics.true_negative == 1  # On-time complaint correctly classified.


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        ([0, 0, 0], [0, 0, 0], (1.0, 0.0, 0.0, 0.0)),
        ([1, 1, 1], [0, 0, 0], (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_basic_evaluation_permits_single_class_actual_targets(
    y_true,
    y_pred,
    expected: tuple[float, float, float, float],
) -> None:
    """Single-class actual inputs have deterministic Phase 1 metrics."""
    metrics = evaluate_basic_classifier(y_true, y_pred)

    assert (
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
    ) == expected
    assert metrics.true_positive + metrics.false_negative == metrics.positive_count
    assert metrics.true_negative + metrics.false_positive == metrics.negative_count


def test_phase_1_result_schema_is_consistent_across_baselines() -> None:
    """Every baseline report row exposes the same Phase 1 fields."""
    predictions = {
        "Majority Class": [0, 0, 0, 0],
        "Historical Rate": [0, 1, 0, 1],
        "Rule Based": [1, 1, 1, 1],
        "Logistic Regression": [0, 1, 1, 0],
    }
    rows = [
        metrics_row(
            model_name,
            evaluate_basic_classifier([0, 1, 0, 1], y_pred),
            evaluated_split="validation",
        )
        for model_name, y_pred in predictions.items()
    ]

    assert all(set(row) == set(rows[0]) for row in rows)
    assert all(row["evaluated_split"] == "validation" for row in rows)
    assert isinstance(evaluate_basic_classifier([0], [0]), BasicClassificationMetrics)
    for row in rows:
        assert (
            row["true_positive"]
            + row["false_positive"]
            + row["false_negative"]
            + row["true_negative"]
            == row["row_count"]
        )


@pytest.mark.parametrize(
    ("y_score", "expected_roc_auc", "expected_pr_auc"),
    [
        ([0.9, 0.8, 0.2, 0.1], 1.0, 1.0),
        ([0.1, 0.2, 0.8, 0.9], 0.0, 5 / 12),
    ],
)
def test_ranking_evaluation_reflects_score_orientation(
    y_score,
    expected_roc_auc: float,
    expected_pr_auc: float,
) -> None:
    """Higher positive-class scores produce better ranking than reversed scores."""
    result = evaluate_ranking([1, 1, 0, 0], y_score)

    assert result.metrics.roc_auc == pytest.approx(expected_roc_auc)
    assert result.metrics.pr_auc == pytest.approx(expected_pr_auc)


def test_tied_scores_match_non_informative_ranking_references() -> None:
    """A constant score has chance ROC-AUC and prevalence Average Precision."""
    result = evaluate_ranking([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])

    assert result.metrics.roc_auc == pytest.approx(0.5)
    assert result.metrics.pr_auc == pytest.approx(0.5)
    assert result.metrics.positive_rate == pytest.approx(0.5)


def test_ranking_curves_have_authoritative_shapes_and_ordering() -> None:
    """ROC and PR arrays preserve sklearn's score-derived threshold contracts."""
    result = evaluate_ranking(
        [1, 0, 1, 0, 1, 0],
        [0.95, 0.80, 0.70, 0.40, 0.30, 0.10],
    )
    curves = result.curves

    assert len(curves.roc_false_positive_rate) == len(curves.roc_true_positive_rate)
    assert len(curves.roc_true_positive_rate) == len(curves.roc_thresholds)
    assert len(curves.pr_precision) == len(curves.pr_recall)
    assert len(curves.pr_precision) == len(curves.pr_thresholds) + 1
    assert curves.roc_false_positive_rate[0] == 0.0
    assert curves.roc_true_positive_rate[0] == 0.0
    assert curves.roc_false_positive_rate[-1] == 1.0
    assert curves.roc_true_positive_rate[-1] == 1.0
    assert all(
        left <= right
        for left, right in zip(
            curves.roc_false_positive_rate,
            curves.roc_false_positive_rate[1:],
        )
    )
    assert all(
        left >= right
        for left, right in zip(curves.pr_recall, curves.pr_recall[1:])
    )
    assert all(
        left < right
        for left, right in zip(curves.pr_thresholds, curves.pr_thresholds[1:])
    )


def test_ranking_result_exposes_population_and_average_precision_definition() -> None:
    """Phase 2 summary includes prevalence and documents the legacy PR name."""
    result = evaluate_ranking([1, 1, 0, 0, 0], [0.9, 0.7, 0.6, 0.2, 0.1])

    assert isinstance(result, RankingEvaluation)
    assert result.metrics.row_count == 5
    assert result.metrics.positive_count == 2
    assert result.metrics.negative_count == 3
    assert result.metrics.positive_rate == pytest.approx(0.4)
    assert PR_AUC_DEFINITION == "average_precision_score"


def test_ranking_summary_schema_is_consistent_across_baseline_scores() -> None:
    """Every baseline score vector produces the same ranking summary contract."""
    score_vectors = {
        "Majority Class": [0.5, 0.5, 0.5, 0.5],
        "Historical Rate": [0.2, 0.8, 0.3, 0.7],
        "Rule Based": [0.2, 0.8, 0.3, 0.7],
        "Logistic Regression": [0.1, 0.9, 0.4, 0.6],
    }
    summaries = [
        evaluate_ranking([0, 1, 0, 1], scores).metrics.to_dict()
        for scores in score_vectors.values()
    ]

    assert all(set(summary) == set(summaries[0]) for summary in summaries)


@pytest.mark.parametrize(
    ("y_true", "y_score", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0.1], "lengths must match"),
        ([0, 2], [0.1, 0.2], "only 0/1"),
        ([0, 1], [0.1, float("nan")], "finite"),
        ([0, 1], [0.1, float("inf")], "finite"),
        ([0, 1], [0.1, 1.1], "in \\[0, 1\\]"),
        ([0, 0], [0.1, 0.2], "both 0 and 1"),
        ([1, 1], [0.8, 0.9], "both 0 and 1"),
    ],
)
def test_ranking_evaluation_rejects_invalid_or_undefined_inputs(
    y_true,
    y_score,
    message: str,
) -> None:
    """Malformed scores and undefined single-class ranking fail explicitly."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_ranking(y_true, y_score)


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
    assert metrics.accuracy == 0.5
    assert metrics.positive_rate == 0.5
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
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0], "lengths must match"),
        ([0, 2], [0, 1], "only 0/1"),
        ([0, 1], [0, None], "null"),
    ],
)
def test_basic_evaluation_rejects_invalid_inputs(y_true, y_pred, message: str) -> None:
    """Phase 1 rejects malformed label and prediction vectors explicitly."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_basic_classifier(y_true, y_pred)


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
