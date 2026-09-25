"""Unit coverage for Phase 5.7 final operational test evaluation helpers."""

import pandas as pd
import pytest

from urban_ops.models.baseline_workflow import (
    _describe_difference,
    _verify_test_top_k_matches_final_test_metrics,
    build_logistic_capacity_comparison_table,
)
from urban_ops.models.evaluation import (
    EvaluationError,
    compare_capacity_levels,
    evaluate_binary_classifier,
    metrics_row,
)

TEST_LABELS = [1, 0, 0, 1, 0, 1, 0, 0] * 5
TEST_SCORES = [1.0 - index / 40 for index in range(40)]


def _test_table() -> pd.DataFrame:
    """Return a 40-row test capacity table built by the shared Top-K path."""
    return build_logistic_capacity_comparison_table(
        compare_capacity_levels(TEST_LABELS, TEST_SCORES),
        evaluated_split="test",
    )


def _final_test_results() -> pd.DataFrame:
    """Return the Month 1 final test row computed from the same scores."""
    predictions = [int(score >= 0.5) for score in TEST_SCORES]
    metrics = evaluate_binary_classifier(TEST_LABELS, predictions, TEST_SCORES)
    return pd.DataFrame(
        [metrics_row("Logistic Regression", metrics, evaluated_split="test")]
    )


def test_test_table_uses_fixed_capacities_and_ceil_k() -> None:
    """Test rows keep the frozen 5/10/20% capacities and ceil(n × capacity)."""
    table = _test_table()

    assert table["evaluated_split"].tolist() == ["test"] * 3
    assert table["capacity"].tolist() == [0.05, 0.10, 0.20]
    assert table["selected_count"].tolist() == [2, 4, 8]
    assert table["captured_positive_count"].tolist() == [1, 2, 3]
    assert table["precision_at_k"].tolist() == pytest.approx([0.5, 0.5, 0.375])
    assert table["recall_at_k"].tolist() == pytest.approx([1 / 15, 2 / 15, 3 / 15])


def test_final_test_consistency_check_accepts_the_shared_evaluation() -> None:
    """The Phase 5.7 table agrees with the existing final test Top-K metrics."""
    _verify_test_top_k_matches_final_test_metrics(
        _test_table(), _final_test_results()
    )


@pytest.mark.parametrize("column", ["precision_at_10_percent", "recall_at_20_percent"])
def test_final_test_consistency_check_rejects_a_competing_result(column: str) -> None:
    """A second, different test result cannot be reported as final evidence."""
    final_results = _final_test_results()
    final_results.loc[0, column] = final_results.loc[0, column] + 0.01

    with pytest.raises(EvaluationError, match="does not match the final test"):
        _verify_test_top_k_matches_final_test_metrics(_test_table(), final_results)


def test_final_test_consistency_check_requires_one_logistic_test_row() -> None:
    """Missing or ambiguous final test rows are rejected."""
    final_results = _final_test_results()

    with pytest.raises(EvaluationError, match="exactly one final"):
        _verify_test_top_k_matches_final_test_metrics(
            _test_table(), final_results.iloc[0:0]
        )
    with pytest.raises(EvaluationError, match="exactly one final"):
        _verify_test_top_k_matches_final_test_metrics(
            _test_table(), pd.concat([final_results, final_results])
        )


@pytest.mark.parametrize(
    ("test_value", "validation_value", "expected"),
    [
        (0.1318, 0.1124, "1.94 percentage points above validation"),
        (0.4618, 0.4786, "1.68 percentage points below validation"),
        (0.25, 0.25, "equal to validation"),
    ],
)
def test_describe_difference_reports_percentage_points(
    test_value: float,
    validation_value: float,
    expected: str,
) -> None:
    """Validation-to-test changes are absolute percentage points with direction."""
    assert _describe_difference(test_value, validation_value).startswith(expected)
