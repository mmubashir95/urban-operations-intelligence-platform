"""Tests for half-open chronological assignment and stable ordering."""

import pandas as pd
import pytest

from urban_ops.splitting.assignment import SplitAssignmentError, assign_time_splits


def test_boundary_timestamps_follow_half_open_policy(boundaries) -> None:
    timestamps = [
        boundaries.train_start, boundaries.train_end_exclusive,
        boundaries.validation_end_exclusive,
        boundaries.test_end_exclusive - pd.Timedelta(microseconds=1),
    ]
    frame = pd.DataFrame({"unique_key": ["a", "b", "c", "d"], "created_date": timestamps, "missed_resolution_target": [0, 1, 0, 1]})
    result = assign_time_splits(frame, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    assert result.train["unique_key"].tolist() == ["a"]
    assert result.validation["unique_key"].tolist() == ["b"]
    assert result.test["unique_key"].tolist() == ["c", "d"]


def test_test_end_exclusive_is_unassigned_and_fails(boundaries) -> None:
    frame = pd.DataFrame({"unique_key": ["outside"], "created_date": [boundaries.test_end_exclusive]})
    with pytest.raises(SplitAssignmentError, match="outside"):
        assign_time_splits(frame, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")


def test_assignment_is_sorted_deterministic_non_mutating_and_preserves_target(
    eligible_frame, boundaries
) -> None:
    shuffled = eligible_frame.sample(frac=1, random_state=7).reset_index(drop=True)
    before = shuffled.copy(deep=True)
    first = assign_time_splits(shuffled, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    second = assign_time_splits(shuffled, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    for name in ("train", "validation", "test"):
        left, right = getattr(first, name), getattr(second, name)
        pd.testing.assert_frame_equal(left, right)
        assert left.equals(left.sort_values(["created_date", "unique_key"]).reset_index(drop=True))
    output = pd.concat([first.train, first.validation, first.test]).set_index("unique_key")
    expected = shuffled.set_index("unique_key")
    pd.testing.assert_series_equal(output["missed_resolution_target"].sort_index(), expected["missed_resolution_target"].sort_index())
    pd.testing.assert_frame_equal(shuffled, before)


def test_missing_null_or_naive_timestamp_fails(eligible_frame, boundaries) -> None:
    with pytest.raises(SplitAssignmentError, match="Missing"):
        assign_time_splits(eligible_frame.drop(columns="created_date"), boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    null = eligible_frame.copy()
    null.loc[0, "created_date"] = pd.NaT
    with pytest.raises(SplitAssignmentError, match="valid"):
        assign_time_splits(null, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
