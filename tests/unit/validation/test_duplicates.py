"""Tests for deterministic exact and conflicting duplicate classification."""

import pandas as pd
import pytest

from urban_ops.validation.duplicates import analyze_duplicates


def test_no_duplicates_returns_stable_empty_reports(raw_frame) -> None:
    result = analyze_duplicates(raw_frame)
    assert not result.exact_flags.any() and not result.conflicting_flags.any()
    assert result.exact_records.empty and result.conflicting_records.empty
    assert list(result.exact_records.columns) == [
        "unique_key", "duplicate_group_size", "canonical_row", "source_row_index"
    ]


def test_exact_duplicate_preserves_one_canonical_and_flags_redundant(raw_frame) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    original = frame.copy(deep=True)
    result = analyze_duplicates(frame)
    assert result.exact_flags.tolist() == [False, True]
    assert result.exact_records["canonical_row"].tolist() == [True, False]
    assert result.summary.set_index("metric").loc["duplicate_full_rows", "row_count"] == 2
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", [
    "due_date", "closed_date", "status", "agency", "complaint_type",
    "descriptor", "resolution_description",
])
def test_material_difference_is_conflicting(raw_frame, column: str) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    frame.loc[1, column] = "different"
    result = analyze_duplicates(frame)
    assert result.conflicting_flags.all()
    assert result.conflicting_records["conflicting_fields"].eq(column).all()


def test_all_conflicting_members_are_reported(raw_frame) -> None:
    frame = pd.concat([raw_frame.iloc[[0]]] * 3, ignore_index=True)
    frame.loc[2, "status"] = "Open"
    result = analyze_duplicates(frame)
    assert len(result.conflicting_records) == 3
    assert result.summary.set_index("metric").loc["conflicting_duplicate_keys", "row_count"] == 1


def test_detection_is_deterministic_under_row_reordering(raw_frame) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    first = analyze_duplicates(frame).summary
    second = analyze_duplicates(frame.iloc[::-1]).summary
    pd.testing.assert_frame_equal(first, second)


def test_missing_unique_key_is_counted_separately(raw_frame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "unique_key"] = None
    result = analyze_duplicates(frame)
    assert result.summary.set_index("metric").loc["missing_unique_key_rows", "row_count"] == 1
    assert not result.exact_flags.any()
