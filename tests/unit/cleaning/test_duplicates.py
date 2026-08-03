"""Tests for stable exact-copy retention and conflicting-group preservation."""

import pandas as pd
import pytest

from urban_ops.cleaning.duplicates import clean_duplicates


def test_no_duplicates_preserves_all_rows(raw_frame) -> None:
    original = raw_frame.copy(deep=True)
    result = clean_duplicates(raw_frame, identifier="unique_key")
    assert len(result.frame) == len(raw_frame)
    assert result.removed_exact_copies == result.conflicting_group_count == 0
    assert result.audit.empty
    pd.testing.assert_frame_equal(raw_frame, original)


def test_exact_duplicate_keeps_one_deterministic_canonical(raw_frame) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    first = clean_duplicates(frame, identifier="unique_key")
    second = clean_duplicates(frame.iloc[::-1].reset_index(drop=True), identifier="unique_key")
    assert len(first.frame) == len(second.frame) == 1
    assert first.removed_exact_copies == second.removed_exact_copies == 1
    assert first.frame.loc[0, "source_row_fingerprint"] == second.frame.loc[0, "source_row_fingerprint"]
    assert set(first.audit["action"]) == {"retain_canonical", "remove_copy"}


@pytest.mark.parametrize("column", ["due_date", "closed_date", "status"])
def test_conflicting_target_material_preserves_and_flags_whole_group(
    raw_frame, column: str
) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    frame.loc[1, column] = "different"
    result = clean_duplicates(frame, identifier="unique_key")
    assert len(result.frame) == 2
    assert result.conflicting_keys == frozenset({"1"})
    assert result.conflicting_group_count == 1
    assert result.audit["action"].eq("preserve_and_exclude").all()
    assert result.audit["conflicting_fields"].eq(column).all()


def test_missing_keys_are_not_duplicate_groups(raw_frame) -> None:
    frame = raw_frame.copy()
    frame["unique_key"] = None
    result = clean_duplicates(frame, identifier="unique_key")
    assert len(result.frame) == 2
    assert result.audit.empty


def test_duplicate_actions_reconcile(raw_frame) -> None:
    frame = pd.concat([raw_frame.iloc[[0]]] * 3, ignore_index=True)
    result = clean_duplicates(frame, identifier="unique_key")
    removed = int(result.audit["action"].eq("remove_copy").sum())
    assert removed == result.removed_exact_copies == 2
    assert len(frame) == len(result.frame) + removed
