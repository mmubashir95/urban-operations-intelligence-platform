"""Unit tests for shared Month 1 evaluation metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from urban_ops.models.evaluation import (
    CALIBRATION_N_BINS,
    CALIBRATION_STRATEGY,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MANUAL_CLASSIFICATION_THRESHOLDS,
    PR_AUC_DEFINITION,
    SWEEP_CLASSIFICATION_THRESHOLDS,
    THRESHOLD_POLICY_MAX_FLAGGED_RATE,
    THRESHOLD_POLICY_MIN_PRECISION,
    THRESHOLD_POLICY_MIN_RECALL,
    THRESHOLD_POLICY_TIE_BREAK,
    FROZEN_THRESHOLD_POLICY_NAME,
    FROZEN_THRESHOLD_SELECTED_THRESHOLD,
    FROZEN_THRESHOLD_SECONDARY_OBJECTIVE,
    BasicClassificationMetrics,
    CalibrationEvaluation,
    CapacityComparisonRow,
    EvaluationError,
    FrozenThresholdDecision,
    RankedRiskRecord,
    RankingEvaluation,
    ThresholdPolicyResult,
    ThresholdMetrics,
    build_calibration_table,
    capacity_to_k,
    classify_scores_at_threshold,
    compare_capacity_levels,
    evaluate_basic_classifier,
    evaluate_binary_classifier,
    evaluate_calibration,
    evaluate_manual_thresholds,
    evaluate_ranking,
    evaluate_threshold,
    evaluate_threshold_selection_policies,
    evaluate_threshold_sweep,
    freeze_workload_limited_threshold,
    metrics_row,
    rank_by_risk,
    select_max_f1_policy,
    select_with_max_flagged_rate_policy,
    select_with_min_precision_policy,
    select_with_min_recall_policy,
    select_top_k,
    top_k_metrics,
    validate_frozen_threshold_decision,
)


def test_rank_by_risk_orders_scores_and_retains_source_fields() -> None:
    """Ranking descends by risk while retaining labels and input positions."""
    ranked = rank_by_risk(
        [0, 1, 0, 1, 1],
        [0.20, 0.90, 0.40, 0.70, 0.60],
    )

    assert [record.rank for record in ranked] == [1, 2, 3, 4, 5]
    assert [record.original_position for record in ranked] == [1, 3, 4, 2, 0]
    assert [record.y_true for record in ranked] == [1, 1, 1, 0, 0]
    assert [record.y_score for record in ranked] == [0.90, 0.70, 0.60, 0.40, 0.20]


def test_rank_by_risk_preserves_input_order_for_equal_scores() -> None:
    """Stable score ties use original input position deterministically."""
    first = rank_by_risk([0, 1, 1], [0.80, 0.80, 0.70])
    repeated = rank_by_risk([0, 1, 1], [0.80, 0.80, 0.70])

    assert first == repeated
    assert [record.original_position for record in first] == [0, 1, 2]


@pytest.mark.parametrize(
    ("n_samples", "capacity", "expected_k"),
    [
        (100, 0.05, 5),
        (100, 0.10, 10),
        (100, 0.20, 20),
        (11, 0.10, 2),
        (5_499, 0.10, 550),
    ],
)
def test_capacity_to_k_uses_ceiling(
    n_samples: int,
    capacity: float,
    expected_k: int,
) -> None:
    """Operational fractions deterministically round upward to whole rows."""
    assert capacity_to_k(n_samples, capacity) == expected_k


def test_select_top_k_returns_exactly_first_k_ranked_records() -> None:
    """Selection is a rank prefix and never a probability cutoff."""
    ranked = rank_by_risk(
        [0, 1, 0, 1, 1],
        [0.20, 0.90, 0.40, 0.70, 0.60],
    )

    selected = select_top_k(ranked, capacity=0.40)

    assert len(selected) == capacity_to_k(len(ranked), 0.40) == 2
    assert selected == ranked[:2]
    assert [record.original_position for record in selected] == [1, 3]


def test_top_k_selection_does_not_use_a_probability_threshold() -> None:
    """Low absolute scores can be selected because membership uses rank only."""
    ranked = rank_by_risk([1, 0, 1], [0.20, 0.10, 0.05])

    selected = select_top_k(ranked, capacity=0.34)

    assert [record.y_score for record in selected] == [0.20, 0.10]
    assert all(record.y_score < FROZEN_THRESHOLD_SELECTED_THRESHOLD for record in selected)


@pytest.mark.parametrize(
    ("y_true", "y_score", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0.5], "lengths must match"),
        ([0, 1], [0.5, float("nan")], "finite"),
        ([0, 1], [0.5, float("inf")], "finite"),
        ([0, 1], [0.5, float("-inf")], "finite"),
        ([0, None], [0.5, 0.6], "null"),
        ([0, 2], [0.5, 0.6], "0/1"),
        ([0, 1], [0.5, 1.2], r"\[0, 1\]"),
        ([0, 1], [-0.1, 0.6], r"\[0, 1\]"),
    ],
)
def test_rank_by_risk_rejects_invalid_inputs(y_true, y_score, message: str) -> None:
    """Malformed foundational ranking inputs fail with evaluation errors."""
    with pytest.raises(EvaluationError, match=message):
        rank_by_risk(y_true, y_score)


@pytest.mark.parametrize(
    ("n_samples", "capacity", "message"),
    [
        (0, 0.10, "positive integer"),
        (-1, 0.10, "positive integer"),
        (1.5, 0.10, "positive integer"),
        (10, 0.0, "interval"),
        (10, -0.1, "interval"),
        (10, 1.01, "interval"),
        (10, float("nan"), "interval"),
        (10, float("inf"), "interval"),
        (10, float("-inf"), "interval"),
        (10, True, "interval"),
        (10, np.bool_(True), "interval"),
        (10, None, "interval"),
        (10, "high", "interval"),
        (True, 0.10, "positive integer"),
        (np.bool_(True), 0.10, "positive integer"),
        (10.0, 0.10, "positive integer"),
    ],
)
def test_capacity_to_k_rejects_invalid_inputs(
    n_samples,
    capacity,
    message: str,
) -> None:
    """Invalid counts and operational fractions fail clearly."""
    with pytest.raises(EvaluationError, match=message):
        capacity_to_k(n_samples, capacity)


def test_select_top_k_rejects_empty_or_malformed_rankings() -> None:
    """Selection requires nonempty, correctly ordered ranked records."""
    with pytest.raises(EvaluationError, match="must not be empty"):
        select_top_k([], capacity=0.10)
    with pytest.raises(EvaluationError, match="RankedRiskRecord"):
        select_top_k([{"rank": 1}], capacity=0.10)
    with pytest.raises(EvaluationError, match="contiguous rank"):
        select_top_k(
            [RankedRiskRecord(2, 0, 1, 0.9)],
            capacity=0.10,
        )


def test_capacity_to_k_accepts_numpy_integer_sample_counts() -> None:
    """NumPy integer counts follow the same ceiling rule as Python integers."""
    assert capacity_to_k(np.int64(100), 0.10) == 10
    assert capacity_to_k(np.int32(11), np.float64(0.10)) == 2


def test_ranked_risk_record_is_immutable() -> None:
    """Ranked records cannot be edited after ranking."""
    record = rank_by_risk([1], [0.9])[0]

    with pytest.raises(AttributeError):
        record.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [RankedRiskRecord(1, 0, 1, 0.9), RankedRiskRecord(1, 1, 0, 0.5)],
            "contiguous rank",
        ),
        (
            [RankedRiskRecord(1, 0, 1, 0.9), RankedRiskRecord(3, 1, 0, 0.5)],
            "contiguous rank",
        ),
        (
            [RankedRiskRecord(2, 1, 0, 0.5), RankedRiskRecord(1, 0, 1, 0.9)],
            "contiguous rank",
        ),
        (
            [RankedRiskRecord(1, 0, 0, 0.1), RankedRiskRecord(2, 1, 1, 0.9)],
            "descending risk",
        ),
        (
            [RankedRiskRecord(1, 1, 0, 0.8), RankedRiskRecord(2, 0, 1, 0.8)],
            "stable ties",
        ),
        (
            [RankedRiskRecord(1, 0, 1, 0.9), RankedRiskRecord(2, 0, 1, 0.8)],
            "original_position",
        ),
    ],
)
def test_select_top_k_rejects_rankings_that_break_the_rank_contract(
    records: list[RankedRiskRecord],
    message: str,
) -> None:
    """Hand-built rankings cannot silently yield a wrong top-K slice."""
    with pytest.raises(EvaluationError, match=message):
        select_top_k(records, capacity=0.50)


def test_select_top_k_accepts_rankings_with_tied_scores() -> None:
    """Stable ties produced by rank_by_risk satisfy the selection contract."""
    ranked = rank_by_risk([0, 1, 1, 0], [0.80, 0.80, 0.80, 0.10])

    selected = select_top_k(ranked, capacity=0.50)

    assert [record.original_position for record in selected] == [0, 1]


def test_compare_capacity_levels_calculates_metrics_and_adjacent_changes() -> None:
    """Capacity comparison reuses exact Top-K counts and adjacent deltas."""
    y_true = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    y_score = [0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, 0.19, 0.10]

    rows = compare_capacity_levels(
        y_true,
        y_score,
        capacities=(0.20, 0.05, 0.10),
    )

    assert all(isinstance(row, CapacityComparisonRow) for row in rows)
    assert [row.capacity for row in rows] == [0.05, 0.10, 0.20]
    assert [row.selected_count for row in rows] == [1, 1, 2]
    assert [row.captured_positive_count for row in rows] == [1, 1, 1]
    assert [row.precision for row in rows] == pytest.approx([1.0, 1.0, 0.5])
    assert [row.recall for row in rows] == pytest.approx([0.25, 0.25, 0.25])
    assert rows[0].additional_selected_count is None
    assert rows[0].additional_capacity is None
    assert rows[0].additional_recall is None
    assert [row.additional_selected_count for row in rows[1:]] == [0, 1]
    assert [row.additional_capacity for row in rows[1:]] == pytest.approx(
        [0.05, 0.10]
    )
    assert [row.additional_recall for row in rows[1:]] == pytest.approx([0.0, 0.0])
    for row in rows:
        assert isinstance(row.captured_positive_count, int)
        assert 0 <= row.captured_positive_count <= row.selected_count
        assert row.captured_positive_count <= sum(y_true)
        assert row.recall == pytest.approx(
            row.captured_positive_count / sum(y_true)
        )


def test_compare_standard_capacities_uses_scores_and_ties_deterministically() -> None:
    """Continuous score order and stable ties govern repeatable capacity rows."""
    y_true = [0] * 17 + [1, 0, 1]
    y_score = [0.01] * 17 + [0.20, 0.20, 0.10]

    first = compare_capacity_levels(y_true, y_score)
    repeated = compare_capacity_levels(y_true, y_score)

    assert first == repeated
    assert [row.capacity for row in first] == [0.05, 0.10, 0.20]
    assert [row.selected_count for row in first] == [1, 2, 4]
    assert [row.captured_positive_count for row in first] == [1, 1, 2]
    assert [row.precision for row in first] == pytest.approx([1.0, 0.5, 0.5])
    assert [row.recall for row in first] == pytest.approx([0.5, 0.5, 1.0])
    assert first[0].precision == 1.0  # A 0.20 score ranks first despite being < 0.49.


@pytest.mark.parametrize(
    ("y_true", "y_score", "capacities", "message"),
    [
        ([], [], (0.05,), "must not be empty"),
        ([0, 1], [0.5], (0.05,), "lengths must match"),
        ([0, 2], [0.5, 0.6], (0.05,), "0/1"),
        ([0, 1], [0.5, float("nan")], (0.05,), "finite"),
        ([0, 1], [0.5, 0.6], (), "must not be empty"),
        ([0, 1], [0.5, 0.6], (0.10, 0.10), "unique"),
        ([0, 1], [0.5, 0.6], (0.0,), "interval"),
    ],
)
def test_compare_capacity_levels_reuses_existing_input_validation(
    y_true,
    y_score,
    capacities,
    message: str,
) -> None:
    """Malformed comparison inputs fail through established evaluation rules."""
    with pytest.raises(EvaluationError, match=message):
        compare_capacity_levels(y_true, y_score, capacities=capacities)


def test_threshold_conversion_includes_score_equal_to_threshold() -> None:
    """The Phase 4.1 equality boundary assigns exact matches to class 1."""
    predictions = classify_scores_at_threshold(
        [0.12, 0.49, 0.50, 0.73],
        threshold=0.50,
    )

    np.testing.assert_array_equal(predictions, [0, 0, 1, 1])


def test_frozen_threshold_conversion_includes_score_equal_to_049() -> None:
    """Phase 4.7 applies the loaded 0.49 threshold with the Phase 4.1 rule."""
    predictions = classify_scores_at_threshold(
        [0.48, FROZEN_THRESHOLD_SELECTED_THRESHOLD, 0.50],
        threshold=FROZEN_THRESHOLD_SELECTED_THRESHOLD,
    )

    np.testing.assert_array_equal(predictions, [0, 1, 1])


def test_threshold_evaluation_matches_hand_calculated_operating_point() -> None:
    """One small example verifies every Phase 4.1 count and metric directly."""
    result = evaluate_threshold(
        [0, 1, 1, 0],
        [0.18, 0.42, 0.63, 0.91],
        threshold=0.50,
    )

    assert isinstance(result, ThresholdMetrics)
    assert result.threshold == DEFAULT_CLASSIFICATION_THRESHOLD == 0.5
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.true_negatives == 1
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.predicted_positive_count == 2
    assert result.predicted_positive_rate == pytest.approx(0.5)
    assert result.sample_count == 4
    assert result.actual_positive_count == 2
    assert result.actual_positive_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("scores", "expected_count", "expected_rate", "expected_precision"),
    [
        ([0.1, 0.2, 0.3, 0.4], 0, 0.0, 0.0),
        ([0.5, 0.6, 0.7, 1.0], 4, 1.0, 0.5),
    ],
)
def test_threshold_evaluation_handles_all_negative_or_positive_predictions(
    scores,
    expected_count: int,
    expected_rate: float,
    expected_precision: float,
) -> None:
    """Extreme prediction sets use Phase 1's explicit zero-division policy."""
    result = evaluate_threshold([0, 1, 0, 1], scores)

    assert result.predicted_positive_count == expected_count
    assert result.predicted_positive_rate == expected_rate
    assert result.precision == expected_precision


@pytest.mark.parametrize(
    ("y_true", "y_score", "threshold", "message"),
    [
        ([0, 1], [0.1, 0.9], -0.01, "threshold must be in"),
        ([0, 1], [0.1, 0.9], 1.01, "threshold must be in"),
        ([0, 1], [0.1], 0.5, "lengths must match"),
        ([0, 1], [-0.1, 0.9], 0.5, "in \\[0, 1\\]"),
        ([0, 1], [0.1, 1.1], 0.5, "in \\[0, 1\\]"),
        ([0, 1], [0.1, float("nan")], 0.5, "finite"),
        ([0, 1], [0.1, float("inf")], 0.5, "finite"),
        ([0, 1], [0.1, float("-inf")], 0.5, "finite"),
        ([0, 2], [0.1, 0.9], 0.5, "only 0/1"),
    ],
)
def test_threshold_evaluation_rejects_invalid_inputs(
    y_true,
    y_score,
    threshold: float,
    message: str,
) -> None:
    """Malformed labels, probabilities, lengths, and thresholds fail clearly."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_threshold(y_true, y_score, threshold=threshold)


def test_threshold_evaluation_is_deterministic_and_serializable() -> None:
    """Repeated scoring produces one stable flat result contract."""
    expected = evaluate_threshold([0, 1, 1], [0.2, 0.5, 0.8]).to_dict()

    assert evaluate_threshold([0, 1, 1], [0.2, 0.5, 0.8]).to_dict() == expected
    assert set(expected) == {
        "threshold",
        "sample_count",
        "actual_positive_count",
        "actual_positive_rate",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "predicted_positive_count",
        "predicted_positive_rate",
    }


def test_manual_threshold_comparison_matches_hand_calculated_counts() -> None:
    """Five explicit thresholds produce manually verifiable confusion counts."""
    results = evaluate_manual_thresholds(
        [1, 0, 1, 0, 1],
        [0.80, 0.65, 0.45, 0.35, 0.25],
    )

    assert tuple(result.threshold for result in results) == (
        MANUAL_CLASSIFICATION_THRESHOLDS
    ) == (0.30, 0.40, 0.50, 0.60, 0.70)
    assert len(results) == 5
    assert [
        (
            result.true_positives,
            result.false_positives,
            result.true_negatives,
            result.false_negatives,
            result.predicted_positive_count,
        )
        for result in results
    ] == [
        (2, 2, 0, 1, 4),
        (2, 1, 1, 1, 3),
        (1, 1, 1, 2, 2),
        (1, 1, 1, 2, 2),
        (1, 0, 2, 2, 1),
    ]
    assert all(result.sample_count == 5 for result in results)
    assert all(result.actual_positive_count == 3 for result in results)
    assert all(result.actual_positive_rate == pytest.approx(0.6) for result in results)


def test_manual_threshold_comparison_preserves_structural_invariants() -> None:
    """Raising a threshold can remove but cannot create positive predictions."""
    results = evaluate_manual_thresholds(
        [1, 0, 1, 0, 1],
        [0.80, 0.65, 0.45, 0.35, 0.25],
    )

    for field in (
        "predicted_positive_count",
        "predicted_positive_rate",
        "true_positives",
        "false_positives",
        "recall",
    ):
        values = [getattr(result, field) for result in results]
        assert all(left >= right for left, right in zip(values, values[1:]))
    for field in ("true_negatives", "false_negatives"):
        values = [getattr(result, field) for result in results]
        assert all(left <= right for left, right in zip(values, values[1:]))


def test_manual_threshold_comparison_reuses_phase_4_1_and_is_deterministic() -> None:
    """The 0.50 row and serialized schema remain identical to Phase 4.1."""
    y_true = [1, 0, 1, 0, 1]
    y_score = [0.80, 0.65, 0.45, 0.35, 0.25]
    first = evaluate_manual_thresholds(y_true, y_score)
    repeated = evaluate_manual_thresholds(y_true, y_score)
    reference = evaluate_threshold(y_true, y_score, threshold=0.50)

    assert first == repeated
    assert first[2] == reference
    schemas = [set(result.to_dict()) for result in first]
    assert all(schema == schemas[0] for schema in schemas)


def test_sweep_threshold_grid_has_91_ascending_unique_hundredths() -> None:
    """The default sweep grid runs 0.05 to 0.95 inclusive in 0.01 steps."""
    assert len(SWEEP_CLASSIFICATION_THRESHOLDS) == 91
    assert SWEEP_CLASSIFICATION_THRESHOLDS[0] == 0.05
    assert SWEEP_CLASSIFICATION_THRESHOLDS[-1] == 0.95
    assert list(SWEEP_CLASSIFICATION_THRESHOLDS) == sorted(
        SWEEP_CLASSIFICATION_THRESHOLDS
    )
    assert len(set(SWEEP_CLASSIFICATION_THRESHOLDS)) == 91
    steps = [
        round(right - left, 10)
        for left, right in zip(
            SWEEP_CLASSIFICATION_THRESHOLDS, SWEEP_CLASSIFICATION_THRESHOLDS[1:]
        )
    ]
    assert all(step == pytest.approx(0.01) for step in steps)


def test_sweep_threshold_grid_is_independent_of_input_scores() -> None:
    """The threshold grid itself does not depend on the supplied data."""
    first = evaluate_threshold_sweep([1, 0], [0.2, 0.8])
    second = evaluate_threshold_sweep([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.7])

    assert tuple(result.threshold for result in first) == tuple(
        result.threshold for result in second
    )


def test_evaluate_threshold_sweep_delegates_to_evaluate_threshold_per_row() -> None:
    """A small custom grid proves delegation instead of a second implementation."""
    y_true = [1, 0, 1, 0, 1]
    y_score = [0.80, 0.65, 0.45, 0.35, 0.25]
    thresholds = (0.30, 0.50, 0.70)

    results = evaluate_threshold_sweep(y_true, y_score, thresholds=thresholds)

    assert tuple(result.threshold for result in results) == thresholds
    assert results == tuple(
        evaluate_threshold(y_true, y_score, threshold=threshold)
        for threshold in thresholds
    )


def test_evaluate_threshold_sweep_default_matches_evaluate_threshold_per_row() -> None:
    """Every row of the default 91-threshold sweep matches standalone evaluation."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]

    results = evaluate_threshold_sweep(y_true, y_score)

    assert len(results) == 91
    assert tuple(result.threshold for result in results) == (
        SWEEP_CLASSIFICATION_THRESHOLDS
    )
    assert results == tuple(
        evaluate_threshold(y_true, y_score, threshold=threshold)
        for threshold in SWEEP_CLASSIFICATION_THRESHOLDS
    )


def test_evaluate_threshold_sweep_preserves_structural_invariants() -> None:
    """Raising the threshold across the full grid cannot create new positives."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]

    results = evaluate_threshold_sweep(y_true, y_score)

    for field in (
        "predicted_positive_count",
        "predicted_positive_rate",
        "true_positives",
        "false_positives",
        "recall",
    ):
        values = [getattr(result, field) for result in results]
        assert all(left >= right for left, right in zip(values, values[1:]))
    for field in ("true_negatives", "false_negatives"):
        values = [getattr(result, field) for result in results]
        assert all(left <= right for left, right in zip(values, values[1:]))
    assert all(result.sample_count == 8 for result in results)
    assert all(result.actual_positive_count == 4 for result in results)


def test_evaluate_threshold_sweep_includes_manual_checkpoints_exactly() -> None:
    """Phase 4.2's five thresholds appear in the sweep with identical metrics."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]

    results = evaluate_threshold_sweep(y_true, y_score)
    by_threshold = {result.threshold: result for result in results}

    for checkpoint in MANUAL_CLASSIFICATION_THRESHOLDS:
        assert checkpoint in by_threshold
        assert by_threshold[checkpoint] == evaluate_threshold(
            y_true, y_score, threshold=checkpoint
        )


def test_evaluate_threshold_sweep_is_deterministic_and_schema_stable() -> None:
    """Repeated sweeps produce identical rows and one consistent field set."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]

    first = evaluate_threshold_sweep(y_true, y_score)
    repeated = evaluate_threshold_sweep(y_true, y_score)

    assert first == repeated
    schemas = [set(result.to_dict()) for result in first]
    assert all(schema == schemas[0] for schema in schemas)
    assert schemas[0] == set(ThresholdMetrics.__dataclass_fields__)


def _policy_sweep_fixture() -> pd.DataFrame:
    """Return a tiny Phase 4.3-shaped sweep with manual policy expectations."""
    return pd.DataFrame(
        [
            {
                "threshold": 0.10,
                "sample_count": 10,
                "actual_positive_count": 5,
                "actual_positive_rate": 0.5,
                "true_positives": 5,
                "false_positives": 5,
                "true_negatives": 0,
                "false_negatives": 0,
                "precision": 0.30,
                "recall": 0.90,
                "f1": 0.45,
                "predicted_positive_count": 10,
                "predicted_positive_rate": 0.80,
            },
            {
                "threshold": 0.20,
                "sample_count": 10,
                "actual_positive_count": 5,
                "actual_positive_rate": 0.5,
                "true_positives": 4,
                "false_positives": 4,
                "true_negatives": 1,
                "false_negatives": 1,
                "precision": 0.50,
                "recall": 0.80,
                "f1": 0.60,
                "predicted_positive_count": 8,
                "predicted_positive_rate": 0.50,
            },
            {
                "threshold": 0.30,
                "sample_count": 10,
                "actual_positive_count": 5,
                "actual_positive_rate": 0.5,
                "true_positives": 3,
                "false_positives": 2,
                "true_negatives": 3,
                "false_negatives": 2,
                "precision": 0.60,
                "recall": 0.70,
                "f1": 0.65,
                "predicted_positive_count": 5,
                "predicted_positive_rate": 0.30,
            },
            {
                "threshold": 0.40,
                "sample_count": 10,
                "actual_positive_count": 5,
                "actual_positive_rate": 0.5,
                "true_positives": 3,
                "false_positives": 2,
                "true_negatives": 3,
                "false_negatives": 2,
                "precision": 0.60,
                "recall": 0.65,
                "f1": 0.65,
                "predicted_positive_count": 5,
                "predicted_positive_rate": 0.25,
            },
        ]
    )


def test_threshold_policy_max_f1_uses_highest_threshold_tie_break() -> None:
    """Max-F1 chooses highest F1 and breaks exact ties by higher threshold."""
    result = select_max_f1_policy(_policy_sweep_fixture())

    assert isinstance(result, ThresholdPolicyResult)
    assert result.policy_name == "Max F1"
    assert result.constraint_satisfied is True
    assert result.candidate_threshold == pytest.approx(0.40)
    assert result.f1 == pytest.approx(0.65)
    assert THRESHOLD_POLICY_TIE_BREAK == "highest_threshold"


def test_threshold_policy_min_recall_filters_then_maximizes_precision() -> None:
    """Recall >= 0.70 keeps boundary rows, then picks highest precision."""
    result = select_with_min_recall_policy(_policy_sweep_fixture())

    assert result.constraint_name == "minimum_recall"
    assert result.constraint_value == pytest.approx(THRESHOLD_POLICY_MIN_RECALL)
    assert result.constraint_satisfied is True
    assert result.candidate_threshold == pytest.approx(0.30)
    assert result.recall == pytest.approx(0.70)
    assert result.precision == pytest.approx(0.60)


def test_threshold_policy_min_precision_filters_then_maximizes_recall() -> None:
    """Precision >= 0.50 keeps boundary rows, then picks highest recall."""
    result = select_with_min_precision_policy(_policy_sweep_fixture())

    assert result.constraint_name == "minimum_precision"
    assert result.constraint_value == pytest.approx(THRESHOLD_POLICY_MIN_PRECISION)
    assert result.constraint_satisfied is True
    assert result.candidate_threshold == pytest.approx(0.20)
    assert result.precision == pytest.approx(0.50)
    assert result.recall == pytest.approx(0.80)


def test_threshold_policy_max_flagged_rate_filters_then_maximizes_recall() -> None:
    """Flagged-rate <= 0.30 keeps boundary rows, then picks highest recall."""
    result = select_with_max_flagged_rate_policy(_policy_sweep_fixture())

    assert result.constraint_name == "maximum_predicted_positive_rate"
    assert result.constraint_value == pytest.approx(THRESHOLD_POLICY_MAX_FLAGGED_RATE)
    assert result.constraint_satisfied is True
    assert result.candidate_threshold == pytest.approx(0.30)
    assert result.predicted_positive_rate == pytest.approx(0.30)
    assert result.recall == pytest.approx(0.70)


def test_threshold_policy_no_candidate_is_explicit_and_serializable() -> None:
    """Unsatisfied constraints return a stable no-candidate result."""
    result = select_with_min_recall_policy(_policy_sweep_fixture(), min_recall=0.95)

    assert result.constraint_satisfied is False
    assert result.candidate_threshold is None
    assert result.precision is None
    serialized = result.to_dict()
    assert serialized["constraint_satisfied"] is False
    assert set(serialized) == set(ThresholdPolicyResult.__dataclass_fields__)


def test_threshold_policy_tie_break_is_deterministic_for_constrained_policies() -> None:
    """Equal objective values prefer the highest threshold every time."""
    tied = _policy_sweep_fixture()
    tied.loc[tied["threshold"].eq(0.40), "recall"] = 0.70

    first = select_with_min_precision_policy(tied, min_precision=0.6)
    repeated = select_with_min_precision_policy(
        tied.sample(frac=1, random_state=7),
        min_precision=0.6,
    )

    assert first.candidate_threshold == pytest.approx(0.40)
    assert repeated == first


def test_evaluate_threshold_selection_policies_returns_stable_order_and_schema() -> None:
    """All four Phase 4.5 policies run in documented comparison order."""
    first = evaluate_threshold_selection_policies(_policy_sweep_fixture())
    repeated = evaluate_threshold_selection_policies(_policy_sweep_fixture())

    assert first == repeated
    assert tuple(result.policy_name for result in first) == (
        "Max F1",
        "Recall >= 0.70",
        "Precision >= 0.50",
        "Flagged rate <= 0.30",
    )
    assert all(
        set(result.to_dict()) == set(ThresholdPolicyResult.__dataclass_fields__)
        for result in first
    )


def test_freeze_workload_limited_threshold_uses_approved_policy_result() -> None:
    """Phase 4.6 freezes the approved workload-limited Phase 4.5 candidate."""
    policy_results = evaluate_threshold_selection_policies(_policy_sweep_fixture())

    decision = freeze_workload_limited_threshold(policy_results)

    assert isinstance(decision, FrozenThresholdDecision)
    assert decision.policy_name == FROZEN_THRESHOLD_POLICY_NAME
    assert decision.constraint_name == "predicted_positive_rate"
    assert decision.constraint_value == pytest.approx(0.30)
    assert decision.secondary_objective == FROZEN_THRESHOLD_SECONDARY_OBJECTIVE
    assert decision.selected_threshold == pytest.approx(0.30)
    assert decision.selected_on_split == "validation"
    assert decision.frozen is True
    assert decision.precision == pytest.approx(0.60)
    assert decision.recall == pytest.approx(0.70)
    assert decision.predicted_positive_rate == pytest.approx(0.30)


def test_freeze_workload_limited_threshold_is_deterministic_and_serializable() -> None:
    """Repeated freezing produces a stable JSON-safe decision contract."""
    policy_results = evaluate_threshold_selection_policies(_policy_sweep_fixture())
    first = freeze_workload_limited_threshold(policy_results)
    repeated = freeze_workload_limited_threshold(policy_results)

    assert first == repeated
    serialized = first.to_dict()
    assert set(serialized) == set(FrozenThresholdDecision.__dataclass_fields__)
    assert FrozenThresholdDecision(**serialized) == first


def _approved_frozen_decision() -> FrozenThresholdDecision:
    """Return a Phase 4.7-valid frozen threshold decision fixture."""
    return FrozenThresholdDecision(
        policy_name=FROZEN_THRESHOLD_POLICY_NAME,
        policy_description=(
            "Approved workload-limited policy: keep predicted_positive_rate "
            "<= 0.30, then maximize recall on validation."
        ),
        constraint_name="predicted_positive_rate",
        constraint_value=0.30,
        secondary_objective=FROZEN_THRESHOLD_SECONDARY_OBJECTIVE,
        selected_threshold=FROZEN_THRESHOLD_SELECTED_THRESHOLD,
        selected_on_split="validation",
        precision=0.5039106145251396,
        recall=0.3129770992366412,
        f1=0.3861301369863014,
        true_positives=902,
        false_positives=888,
        true_negatives=2992,
        false_negatives=1980,
        predicted_positive_count=1790,
        predicted_positive_rate=0.2647145814847678,
        frozen=True,
    )


def test_validate_frozen_threshold_decision_accepts_authoritative_contract() -> None:
    """Phase 4.7 can only use the approved validation-selected decision."""
    decision = _approved_frozen_decision()

    assert validate_frozen_threshold_decision(decision) == decision


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"frozen": False}, "frozen=true"),
        ({"selected_on_split": "test"}, "selected on validation"),
        ({"selected_threshold": 0.50}, "0.49"),
        ({"selected_threshold": float("nan")}, "threshold"),
        ({"policy_name": "Max F1"}, "policy name"),
        ({"constraint_name": "minimum_precision"}, "constraint name"),
        ({"constraint_value": 0.20}, "0.30"),
        ({"secondary_objective": "maximize_precision"}, "secondary objective"),
        ({"predicted_positive_rate": 0.31}, "workload constraint"),
    ],
)
def test_validate_frozen_threshold_decision_rejects_invalid_artifacts(
    updates: dict[str, object],
    message: str,
) -> None:
    """Invalid Phase 4.6 artifacts fail before the test split can be scored."""
    payload = _approved_frozen_decision().to_dict()
    payload.update(updates)

    with pytest.raises(EvaluationError, match=message):
        validate_frozen_threshold_decision(FrozenThresholdDecision(**payload))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.loc[
                ~frame["policy_name"].eq("Flagged rate <= 0.30")
            ],
            "exactly one approved",
        ),
        (
            lambda frame: frame.assign(
                policy_name=frame["policy_name"].replace(
                    {"Flagged rate <= 0.30": "Wrong policy"}
                )
            ),
            "exactly one approved",
        ),
        (
            lambda frame: frame.assign(
                predicted_positive_rate=np.where(
                    frame["policy_name"].eq("Flagged rate <= 0.30"), 0.31, frame["predicted_positive_rate"]
                )
            ),
            "violates",
        ),
        (
            lambda frame: frame.assign(
                selection_reason=np.where(
                    frame["policy_name"].eq("Flagged rate <= 0.30"),
                    "Selected arbitrarily.",
                    frame["selection_reason"],
                )
            ),
            "recall maximization",
        ),
    ],
)
def test_freeze_workload_limited_threshold_rejects_invalid_provenance(
    mutator,
    message: str,
) -> None:
    """Phase 4.6 fails if the approved Phase 4.5 candidate is not intact."""
    policy_table = pd.DataFrame(
        [result.to_dict() for result in evaluate_threshold_selection_policies(_policy_sweep_fixture())]
    )
    with pytest.raises(EvaluationError, match=message):
        freeze_workload_limited_threshold(mutator(policy_table))


@pytest.mark.parametrize(
    ("selector", "kwargs", "message"),
    [
        (select_with_min_recall_policy, {"min_recall": -0.01}, "min_recall"),
        (select_with_min_precision_policy, {"min_precision": 1.01}, "min_precision"),
        (
            select_with_max_flagged_rate_policy,
            {"max_flagged_rate": float("nan")},
            "max_flagged_rate",
        ),
    ],
)
def test_threshold_policies_reject_invalid_constraints(
    selector,
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Policy constraints must be finite values in [0, 1]."""
    with pytest.raises(EvaluationError, match=message):
        selector(_policy_sweep_fixture(), **kwargs)


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        (pd.DataFrame(), "must not be empty"),
        (_policy_sweep_fixture().drop(columns=["precision"]), "missing required"),
        (
            _policy_sweep_fixture().assign(threshold=[0.1, 0.2, 0.2, 0.4]),
            "unique thresholds",
        ),
        (_policy_sweep_fixture().assign(f1=[0.1, 0.2, float("inf"), 0.4]), "finite"),
    ],
)
def test_threshold_policies_reject_malformed_sweeps(
    sweep: pd.DataFrame,
    message: str,
) -> None:
    """Policy selection fails clearly on malformed Phase 4.3 tables."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_threshold_selection_policies(sweep)


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


def test_calibration_brier_matches_hand_calculated_example() -> None:
    """Brier score matches the mean of three explicit squared errors."""
    result = evaluate_calibration([1, 0, 1], [0.8, 0.3, 0.6])

    expected = (0.04 + 0.09 + 0.16) / 3
    assert result.metrics.brier_score == pytest.approx(expected)


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected_brier"),
    [
        ([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0], 0.0),
        ([1, 0], [0.0, 1.0], 1.0),
        ([1, 0], [0.5, 0.5], 0.25),
    ],
)
def test_calibration_brier_core_probability_cases(
    y_true,
    y_score,
    expected_brier: float,
) -> None:
    """Perfect, confidently wrong, and uncertain probabilities are deterministic."""
    result = evaluate_calibration(y_true, y_score)

    assert result.metrics.brier_score == pytest.approx(expected_brier)


def test_calibration_curve_uses_uniform_bins_and_omits_empty_bins() -> None:
    """Curve axes contain only sklearn-populated uniform-bin points."""
    result = evaluate_calibration([0, 1], [0.05, 0.95])

    assert result.curve.requested_bin_count == CALIBRATION_N_BINS == 10
    assert result.curve.strategy == CALIBRATION_STRATEGY == "uniform"
    assert result.curve.mean_predicted_probability == pytest.approx((0.05, 0.95))
    assert result.curve.observed_positive_rate == pytest.approx((0.0, 1.0))
    assert len(result.curve.mean_predicted_probability) == 2


def test_well_calibrated_grouped_probabilities_follow_ideal_diagonal() -> None:
    """Twenty and eighty percent groups yield matching observed frequencies."""
    y_score = [0.2] * 10 + [0.8] * 10
    y_true = [1, 1] + [0] * 8 + [1] * 8 + [0, 0]

    result = evaluate_calibration(y_true, y_score)

    assert result.curve.mean_predicted_probability == pytest.approx((0.2, 0.8))
    assert result.curve.observed_positive_rate == pytest.approx((0.2, 0.8))


def test_overconfident_group_lies_below_ideal_diagonal() -> None:
    """An 0.8 prediction group with 40% events is visibly overconfident."""
    result = evaluate_calibration([1] * 4 + [0] * 6, [0.8] * 10)

    assert result.curve.mean_predicted_probability == pytest.approx((0.8,))
    assert result.curve.observed_positive_rate == pytest.approx((0.4,))
    assert (
        result.curve.observed_positive_rate[0]
        < result.curve.mean_predicted_probability[0]
    )


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected_rate", "expected_brier"),
    [
        ([0, 0, 0], [0.1, 0.2, 0.3], 0.0, (0.01 + 0.04 + 0.09) / 3),
        ([1, 1, 1], [0.7, 0.8, 0.9], 1.0, (0.09 + 0.04 + 0.01) / 3),
    ],
)
def test_calibration_accepts_single_class_targets(
    y_true,
    y_score,
    expected_rate: float,
    expected_brier: float,
) -> None:
    """Brier and observed frequencies remain defined for one target class."""
    result = evaluate_calibration(y_true, y_score)

    assert result.metrics.positive_rate == expected_rate
    assert result.metrics.brier_score == pytest.approx(expected_brier)
    assert all(value == expected_rate for value in result.curve.observed_positive_rate)


def test_calibration_result_exposes_consistent_probability_contract() -> None:
    """Phase 3 keeps scalar metrics separate from populated curve coordinates."""
    result = evaluate_calibration([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert isinstance(result, CalibrationEvaluation)
    assert result.metrics.row_count == 4
    assert result.metrics.positive_count == 2
    assert result.metrics.negative_count == 2
    assert result.metrics.positive_rate == pytest.approx(0.5)
    assert len(result.curve.mean_predicted_probability) == len(
        result.curve.observed_positive_rate
    )


@pytest.mark.parametrize(
    ("y_true", "y_score", "message"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0.1], "lengths must match"),
        ([0, 2], [0.1, 0.2], "only 0/1"),
        ([0, 1], [0.1, float("nan")], "finite"),
        ([0, 1], [0.1, float("inf")], "finite"),
        ([0, 1], [0.1, float("-inf")], "finite"),
        ([0, 1], [-0.1, 0.2], "in \\[0, 1\\]"),
        ([0, 1], [0.1, 1.1], "in \\[0, 1\\]"),
    ],
)
def test_calibration_rejects_invalid_probability_inputs(
    y_true,
    y_score,
    message: str,
) -> None:
    """Invalid labels and probability values fail without clipping."""
    with pytest.raises(EvaluationError, match=message):
        evaluate_calibration(y_true, y_score)


def test_legacy_evaluator_reuses_phase_3_brier_score() -> None:
    """The legacy Brier field equals the isolated Phase 3 implementation."""
    y_true = [1, 0, 1]
    y_pred = [1, 0, 1]
    y_score = [0.8, 0.3, 0.6]

    legacy = evaluate_binary_classifier(y_true, y_pred, y_score)
    calibration = evaluate_calibration(y_true, y_score)

    assert legacy.brier_score == calibration.metrics.brier_score


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
