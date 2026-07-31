"""Tests for row-level timestamp chronology evidence."""

import pandas as pd
import pytest

from urban_ops.validation.timestamps import analyze_timestamps, chronology_violations


@pytest.mark.parametrize("column", ["due_date", "closed_date"])
@pytest.mark.parametrize("value", ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])
def test_equal_or_after_created_is_valid(column: str, value: str) -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "created_date": "2024-01-01T00:00:00Z",
        "due_date": None, "closed_date": None, "resolution_action_updated_date": None,
        column: value,
    }])
    parsed = analyze_timestamps(
        frame, ["created_date", "due_date", "closed_date", "resolution_action_updated_date"]
    ).parsed
    assert chronology_violations(frame, parsed).empty


@pytest.mark.parametrize(("column", "violation"), [
    ("due_date", "due_before_created"),
    ("closed_date", "closed_before_created"),
    ("resolution_action_updated_date", "resolution_action_before_created"),
])
def test_before_created_is_flagged(column: str, violation: str) -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "created_date": "2024-01-02T00:00:00Z",
        "due_date": None, "closed_date": None, "resolution_action_updated_date": None,
        column: "2024-01-01T00:00:00Z",
    }])
    original = frame.copy(deep=True)
    parsed = analyze_timestamps(frame, list(frame.columns[1:])).parsed
    result = chronology_violations(frame, parsed)
    assert result["violation_type"].tolist() == [violation]
    pd.testing.assert_frame_equal(frame, original)


def test_missing_dates_do_not_create_false_violations() -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "created_date": "2024-01-01", "due_date": None,
        "closed_date": None, "resolution_action_updated_date": None,
    }])
    parsed = analyze_timestamps(frame, list(frame.columns[1:])).parsed
    assert chronology_violations(frame, parsed).empty


def test_invalid_created_with_present_target_inputs_preserves_multiple_violations() -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "created_date": "invalid", "due_date": "2024-01-02",
        "closed_date": "2024-01-03", "resolution_action_updated_date": None,
    }])
    parsed = analyze_timestamps(frame, list(frame.columns[1:])).parsed
    result = chronology_violations(frame, parsed)
    assert set(result["violation_type"]) == {
        "closed_present_created_invalid", "due_present_created_invalid"
    }
    assert list(result.columns) == [
        "unique_key", "created_date_raw", "due_date_raw", "closed_date_raw",
        "resolution_action_updated_date_raw", "violation_type", "severity",
        "recommended_step_7_action",
    ]
