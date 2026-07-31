"""Tests for deterministic target eligibility."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.features.eligibility import evaluate_target_eligibility

BASE = {
    "unique_key": "1", "created_date": "2024-01-01T12:00:00Z",
    "due_date": "2024-01-10T12:00:00Z",
    "closed_date": "2024-01-09T12:00:00Z", "agency": "DSNY",
    "complaint_type": "Graffiti", "status": "Closed",
}


def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_target_eligibility(
        frame, selected_agency="DSNY", selected_complaint_type="Graffiti",
        start_date="2024-01-01", end_date="2025-12-31",
        extraction_timestamp="2026-01-31T00:00:00Z",
    )


def test_valid_row_is_eligible_and_input_unchanged() -> None:
    frame = pd.DataFrame([BASE])
    original = frame.copy(deep=True)
    result = evaluate(frame)
    assert result.loc[0, "target_eligible"]
    assert result.loc[0, "primary_exclusion_reason"] == "eligible"
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(("changes", "reason", "flag"), [
    ({"created_date": None}, "missing_created_date", "has_created_date"),
    ({"due_date": None}, "missing_due_date", "has_due_date"),
    ({"closed_date": None}, "missing_closed_date", "has_closed_date"),
    ({"due_date": "2023-12-31"}, "due_before_created", "valid_due_chronology"),
    ({"closed_date": "2023-12-31"}, "closed_before_created", "valid_closed_chronology"),
    ({"due_date": "2026-02-01"}, "outcome_not_mature", "outcome_mature"),
    ({"status": "Cancelled"}, "excluded_status", "status_allowed"),
    ({"status": "Open", "closed_date": None}, "missing_closed_date", "has_closed_date"),
    ({"agency": "DOT"}, "outside_selected_scope", "within_selected_scope"),
    ({"complaint_type": "Noise"}, "outside_selected_scope", "within_selected_scope"),
    ({"created_date": "2023-12-31"}, "outside_selected_scope", "within_selected_scope"),
    ({"created_date": "2026-01-01"}, "outside_selected_scope", "within_selected_scope"),
])
def test_governance_failures(
    changes: dict[str, object], reason: str, flag: str
) -> None:
    result = evaluate(pd.DataFrame([{**BASE, **changes}]))
    assert not result.loc[0, "target_eligible"]
    assert result.loc[0, "primary_exclusion_reason"] == reason
    assert not result.loc[0, flag]


def test_multiple_failures_preserve_flags_and_precedence() -> None:
    result = evaluate(pd.DataFrame([{**BASE, "due_date": None,
                                     "closed_date": None, "status": "Cancelled"}]))
    assert not result.loc[0, "has_due_date"]
    assert not result.loc[0, "has_closed_date"]
    assert not result.loc[0, "status_allowed"]
    assert result.loc[0, "primary_exclusion_reason"] == "missing_due_date"


def test_exact_duplicate_keeps_stable_canonical_row() -> None:
    rows = pd.DataFrame([{**BASE, "unique_key": "same"}] * 2, index=[20, 10])
    result = evaluate(rows)
    assert result.loc[10, "target_eligible"]
    assert result.loc[20, "is_exact_duplicate"]
    assert not result["is_conflicting_duplicate"].any()


def test_conflicting_duplicate_excludes_whole_group() -> None:
    rows = pd.DataFrame([
        {**BASE, "unique_key": "same"},
        {**BASE, "unique_key": "same", "closed_date": "2024-01-11"},
    ])
    result = evaluate(rows)
    assert result["is_conflicting_duplicate"].all()
    assert not result["target_eligible"].any()
    assert result["primary_exclusion_reason"].eq(
        "conflicting_duplicate_unique_key"
    ).all()


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate(pd.DataFrame([BASE]).drop(columns="status"))
