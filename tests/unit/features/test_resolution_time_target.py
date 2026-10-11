"""Tests for deterministic, leakage-safe resolution-time target construction."""

from __future__ import annotations

import pandas as pd
import pytest

from urban_ops.features.target import (
    build_missed_resolution_target,
    build_resolution_time_target,
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a small target input with stable non-default indexes."""
    return pd.DataFrame(rows, index=range(10, 10 + len(rows)))


@pytest.mark.parametrize(("closed", "expected"), [
    ("2024-01-01T11:00:00Z", 1.0),
    ("2024-01-03T16:00:00Z", 54.0),
    ("2024-01-01T10:30:00Z", 0.5),
    ("2024-01-01T10:00:00Z", 0.0),
    ("2026-01-01T10:00:00Z", 17544.0),
])
def test_valid_durations_preserve_precision_and_long_values(
    closed: str, expected: float
) -> None:
    result = build_resolution_time_target(_frame([{
        "created_date": "2024-01-01T10:00:00Z",
        "closed_date": closed,
        "status": "Closed",
    }]))
    assert result.loc[10, "resolution_time_eligible"]
    assert result.loc[10, "resolution_duration_hours"] == pytest.approx(expected)
    assert result.loc[10, "resolution_time_exclusion_reason"] == "eligible"


@pytest.mark.parametrize(("changes", "reason", "category"), [
    ({"created_date": None}, "missing_created_date", "invalid"),
    ({"created_date": "not-a-date"}, "invalid_created_date", "invalid"),
    ({"closed_date": None}, "missing_closed_date", "not_observed"),
    ({"closed_date": "not-a-date"}, "invalid_closed_date", "invalid"),
    ({"closed_date": "2024-01-01T09:00:00Z"}, "invalid_timestamp_order", "invalid"),
    ({"status": "Open"}, "open_complaint", "not_observed"),
    ({"status": "Cancelled"}, "cancelled_complaint", "invalid"),
    ({"status": "Pending"}, "ambiguous_completion_status", "invalid"),
    ({"status": None}, "ambiguous_completion_status", "invalid"),
])
def test_ineligible_rows_have_deterministic_primary_reason(
    changes: dict[str, object], reason: str, category: str
) -> None:
    row = {
        "created_date": "2024-01-01T10:00:00Z",
        "closed_date": "2024-01-01T11:00:00Z",
        "status": "Closed",
    }
    result = build_resolution_time_target(_frame([{**row, **changes}]))
    assert not result.loc[10, "resolution_time_eligible"]
    assert pd.isna(result.loc[10, "resolution_duration_hours"])
    assert result.loc[10, "resolution_time_exclusion_reason"] == reason
    assert result.loc[10, "resolution_time_exclusion_category"] == category


@pytest.mark.parametrize(("closed", "eligible"), [
    ("2024-01-31T23:59:59Z", True),
    ("2024-02-01T00:00:00Z", True),
    ("2024-02-01T00:00:01Z", False),
])
def test_observation_cutoff_is_inclusive(closed: str, eligible: bool) -> None:
    result = build_resolution_time_target(
        _frame([{
            "created_date": "2024-01-01T00:00:00Z",
            "closed_date": closed,
            "status": "Closed",
        }]),
        observation_cutoff="2024-02-01T00:00:00Z",
    )
    assert bool(result.loc[10, "resolution_time_eligible"]) is eligible
    if not eligible:
        assert result.loc[10, "resolution_time_exclusion_reason"] == (
            "resolution_not_observed_by_cutoff"
        )
        assert result.loc[10, "resolution_time_exclusion_category"] == "not_observed"


def test_multiple_rows_preserve_alignment_input_and_reproducibility() -> None:
    frame = _frame([
        {"created_date": "2024-01-01T00:00:00Z", "closed_date": "2024-01-02T00:00:00Z", "status": "closed"},
        {"created_date": "2024-01-01T00:00:00Z", "closed_date": None, "status": "open"},
    ])
    original = frame.copy(deep=True)
    first = build_resolution_time_target(frame)
    second = build_resolution_time_target(frame)
    assert first.index.equals(frame.index)
    assert first["resolution_time_eligible"].tolist() == [True, False]
    assert first.loc[10, "resolution_duration_hours"] == 24.0
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, original)


def test_empty_input_returns_typed_aligned_output() -> None:
    result = build_resolution_time_target(
        pd.DataFrame(columns=["created_date", "closed_date", "status"])
    )
    assert result.empty
    assert list(result.columns) == [
        "resolution_time_eligible", "resolution_duration_hours",
        "resolution_time_exclusion_reason", "resolution_time_exclusion_category",
    ]
    assert str(result["resolution_duration_hours"].dtype) == "Float64"


def test_reason_precedence_is_stable() -> None:
    result = build_resolution_time_target(_frame([{
        "created_date": None, "closed_date": None, "status": "Open",
    }]))
    assert result.loc[10, "resolution_time_exclusion_reason"] == "missing_created_date"


def test_required_columns_and_invalid_cutoff_raise() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_resolution_time_target(pd.DataFrame({"created_date": []}))
    with pytest.raises(ValueError, match="observation_cutoff"):
        build_resolution_time_target(
            _frame([{"created_date": "2024-01-01", "closed_date": "2024-01-02", "status": "Closed"}]),
            observation_cutoff="not-a-cutoff",
        )


def test_existing_classification_target_is_unchanged() -> None:
    frame = pd.DataFrame({
        "due_date": ["2024-01-02", "2024-01-02"],
        "closed_date": ["2024-01-02", "2024-01-03"],
        "target_eligible": [True, True],
    })
    assert build_missed_resolution_target(frame).tolist() == [0, 1]
