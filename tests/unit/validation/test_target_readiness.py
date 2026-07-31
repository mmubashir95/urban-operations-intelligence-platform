"""Tests for provisional readiness derived through Step 4 governance."""

import pandas as pd
import pytest

from urban_ops.validation.duplicates import analyze_duplicates
from urban_ops.validation.target_readiness import analyze_target_readiness


def analyze(frame: pd.DataFrame, selected_scope) -> object:
    return analyze_target_readiness(
        frame,
        scope=selected_scope,
        extraction_timestamp="2026-12-31T00:00:00Z",
        duplicates=analyze_duplicates(frame),
    )


def test_valid_candidate_is_ready_and_no_target_is_created(raw_frame, selected_scope) -> None:
    original = raw_frame.copy(deep=True)
    result = analyze(raw_frame, selected_scope)
    assert result.flags["candidate_target_eligible"].all()
    assert "missed_resolution_target" not in result.flags
    pd.testing.assert_frame_equal(raw_frame, original)


@pytest.mark.parametrize(("changes", "failed_rule"), [
    ({"due_date": None}, "has_due_date"),
    ({"closed_date": None}, "has_closed_date"),
    ({"due_date": "2023-12-31"}, "valid_due_chronology"),
    ({"closed_date": "2023-12-31"}, "valid_closed_chronology"),
    ({"status": "Open", "closed_date": None}, "status_allowed"),
    ({"due_date": "2027-01-01"}, "outcome_mature"),
])
def test_governed_failure_is_not_candidate_ready(
    raw_frame, selected_scope, changes: dict[str, object], failed_rule: str
) -> None:
    frame = raw_frame.iloc[[0]].copy()
    for column, value in changes.items():
        frame.loc[frame.index[0], column] = value
    result = analyze(frame, selected_scope)
    assert not bool(result.flags.iloc[0].candidate_target_eligible)
    assert not bool(result.flags.iloc[0][failed_rule])


def test_conflicting_duplicate_is_not_candidate_ready(raw_frame, selected_scope) -> None:
    frame = pd.concat([raw_frame.iloc[[0]], raw_frame.iloc[[0]]], ignore_index=True)
    frame.loc[1, "resolution_description"] = "different material source value"
    result = analyze(frame, selected_scope)
    assert result.flags["is_conflicting_duplicate"].all()
    assert not result.flags["candidate_target_eligible"].any()


def test_unexpected_status_is_provisionally_excluded_not_silently_allowed(
    raw_frame, selected_scope
) -> None:
    frame = raw_frame.iloc[[0]].assign(status="Novel")
    result = analyze(frame, selected_scope)
    assert not result.flags["status_allowed"].iloc[0]
    assert not result.flags["candidate_target_eligible"].iloc[0]


def test_exclusion_counts_reconcile(raw_frame, selected_scope) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "due_date"] = None
    result = analyze(frame, selected_scope)
    assert result.exclusions["row_count"].sum() == len(frame)
    assert result.summary.set_index("readiness_rule").loc[
        "candidate_target_eligible", "pass_count"
    ] == 1
