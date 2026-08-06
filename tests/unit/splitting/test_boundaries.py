"""Tests for boundary validation, candidate gates, drift, ranking, and selection."""

from dataclasses import replace

import pandas as pd
import pytest

from urban_ops.splitting.boundaries import (
    boundaries_from_mapping, evaluate_candidates, selected_boundaries,
    validate_boundaries,
)


def evaluate(eligible_frame, candidates, selected="A", minimum_rows=1, minimum_class=1):
    """Evaluate candidates with the common test contract."""
    return evaluate_candidates(
        eligible_frame, candidates=candidates, selected_candidate_id=selected,
        timestamp_column="created_date", identifier_column="unique_key",
        target_column="missed_resolution_target", minimum_rows=minimum_rows,
        minimum_class_rows=minimum_class, maximum_target_rate_difference=None,
        drift_is_warning_only=True,
    )


def test_valid_contiguous_boundaries_and_naive_values_use_utc(boundaries) -> None:
    validate_boundaries(boundaries)
    parsed = boundaries_from_mapping({
        "train_start": "2024-01-01", "train_end_exclusive": "2024-04-01",
        "validation_start": "2024-04-01", "validation_end_exclusive": "2024-07-01",
        "test_start": "2024-07-01", "test_end_exclusive": "2025-01-01",
    }, interval_convention="left_closed_right_open")
    assert str(parsed.train_start.tz) == "UTC"
    assert parsed.train_end_exclusive == parsed.validation_start
    assert parsed.validation_end_exclusive == parsed.test_start


@pytest.mark.parametrize("changed", [
    {"validation_start": pd.Timestamp("2024-03-01", tz="UTC")},
    {"validation_start": pd.Timestamp("2024-05-01", tz="UTC")},
    {"train_end_exclusive": pd.Timestamp("2023-01-01", tz="UTC")},
])
def test_overlap_gap_and_reversed_boundaries_fail(boundaries, changed) -> None:
    with pytest.raises(ValueError, match="ordered, contiguous"):
        validate_boundaries(replace(boundaries, **changed))


def test_unsupported_convention_and_incomplete_source_coverage_fail(boundaries) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        validate_boundaries(replace(boundaries, interval_convention="closed"))
    with pytest.raises(ValueError, match="earliest"):
        validate_boundaries(boundaries, source_min=boundaries.train_start - pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="latest"):
        validate_boundaries(boundaries, source_max=boundaries.test_end_exclusive)


def test_multiple_candidates_rank_deterministically_and_select_once(
    eligible_frame, boundaries
) -> None:
    results = evaluate(eligible_frame, [(name, boundaries) for name in ("A", "B", "C")])
    assert len(results) == 3
    assert [item.rank for item in results] == [1, 2, 3]
    assert sum(item.recommended for item in results) == 1
    assert selected_boundaries(results, "A") == boundaries
    assert all(item.decision_reason for item in results)
    assert all(item.metrics["all_rows_assigned"] for item in results)


def test_insufficient_rows_or_one_class_candidate_is_not_qualified(
    eligible_frame, boundaries
) -> None:
    with pytest.raises(ValueError, match="selected candidate fails"):
        evaluate(
            eligible_frame, [(name, boundaries) for name in ("A", "B", "C")],
            minimum_rows=100,
        )
    one_class = eligible_frame.copy()
    one_class.loc[one_class["created_date"].lt(boundaries.train_end_exclusive), "missed_resolution_target"] = 0
    with pytest.raises(ValueError, match="selected candidate fails"):
        evaluate(one_class, [(name, boundaries) for name in ("A", "B", "C")])


def test_target_rate_drift_is_reported_but_warning_only(
    eligible_frame, boundaries
) -> None:
    frame = eligible_frame.copy()
    frame.loc[frame["created_date"].ge(boundaries.test_start), "missed_resolution_target"] = 1
    first_test = frame.index[frame["created_date"].ge(boundaries.test_start)][0]
    frame.loc[first_test, "missed_resolution_target"] = 0
    results = evaluate_candidates(
        frame, candidates=[(name, boundaries) for name in ("A", "B", "C")],
        selected_candidate_id="A", timestamp_column="created_date",
        identifier_column="unique_key", target_column="missed_resolution_target",
        minimum_rows=1, minimum_class_rows=0,
        maximum_target_rate_difference=0.01, drift_is_warning_only=True,
    )
    assert results[0].metrics["maximum_rate_difference"] > 0.01
    assert results[0].qualified


def test_selected_candidate_must_be_deterministic_top_rank(
    eligible_frame, boundaries
) -> None:
    with pytest.raises(ValueError, match="top-ranked"):
        evaluate(eligible_frame, [(name, boundaries) for name in ("A", "B", "C")], selected="B")
