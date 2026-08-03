"""Regression tests for Step 6 cleaning-action report attribution."""

from urban_ops.validation.models import Severity
from urban_ops.validation.reports import proposed_cleaning_actions
from urban_ops.validation.severity import make_check


def test_chronology_action_names_both_affected_source_columns() -> None:
    check = make_check(
        check_id="chronology.closed_before_created",
        area="chronology",
        check_name="Closed Before Created",
        severity=Severity.ERROR,
        passed=False,
        observed_value=6,
        expected_value=0,
        affected_rows=6,
        total_rows=100,
        message="Timestamp order conflicts with the contract.",
        recommended_action="Do not repair automatically.",
    )
    actions = proposed_cleaning_actions((check,))
    assert actions.loc[0, "source_column"] == "created_date|closed_date"


def test_due_chronology_action_names_both_affected_source_columns() -> None:
    check = make_check(
        check_id="chronology.due_before_created",
        area="chronology",
        check_name="Due Before Created",
        severity=Severity.ERROR,
        passed=False,
        observed_value=1,
        expected_value=0,
        affected_rows=1,
        total_rows=10,
        message="Timestamp order conflicts with the contract.",
        recommended_action="Do not repair automatically.",
    )
    actions = proposed_cleaning_actions((check,))
    assert actions.loc[0, "source_column"] == "created_date|due_date"
