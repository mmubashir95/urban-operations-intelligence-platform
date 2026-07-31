"""Tests for central severity ordering, summaries, and CLI thresholds."""

import pytest

from urban_ops.validation.models import Severity
from urban_ops.validation.severity import (
    checks_frame, make_check, overall_status, severity_counts, validation_exit_code,
)


def check(severity: Severity, passed: bool, identifier: str = "check"):
    return make_check(
        check_id=identifier, area="test", check_name="Test", severity=severity,
        passed=passed, observed_value=1, expected_value=0,
        affected_rows=0 if passed else 1, total_rows=10,
        message="message", recommended_action="action",
    )


def test_critical_failure_controls_overall_status() -> None:
    assert overall_status([check(Severity.WARNING, False), check(Severity.CRITICAL, False)]) == "CRITICAL"


def test_error_does_not_become_critical() -> None:
    assert overall_status([check(Severity.ERROR, False)]) == "ERROR"


def test_pass_only_status_is_pass() -> None:
    assert overall_status([check(Severity.CRITICAL, True)]) == "PASS"


def test_warning_does_not_fail_default_cli_mode() -> None:
    assert validation_exit_code([check(Severity.WARNING, False)], fail_on_error=False, fail_on_warning=False) == 0


def test_fail_on_error_changes_exit_behavior() -> None:
    assert validation_exit_code([check(Severity.ERROR, False)], fail_on_error=True, fail_on_warning=False) == 1


def test_fail_on_warning_changes_exit_behavior() -> None:
    assert validation_exit_code([check(Severity.WARNING, False)], fail_on_error=False, fail_on_warning=True) == 1


def test_critical_always_returns_distinct_exit_code() -> None:
    assert validation_exit_code([check(Severity.CRITICAL, False)], fail_on_error=False, fail_on_warning=False) == 2


def test_checks_are_sorted_and_pass_checks_retained() -> None:
    frame = checks_frame([
        check(Severity.INFO, True, "z"), check(Severity.CRITICAL, True, "b"),
        check(Severity.CRITICAL, False, "a"),
    ])
    assert frame["check_id"].tolist() == ["a", "b", "z"]
    assert set(frame["status"]) == {"PASS", "FAIL"}


def test_summary_counts_reconcile_and_actions_are_preserved() -> None:
    checks = [check(Severity.ERROR, False), check(Severity.WARNING, False, "other")]
    assert severity_counts(checks) == {"CRITICAL": 0, "ERROR": 1, "WARNING": 1, "INFO": 0}
    assert checks[0].recommended_step_7_action == "action"


def test_unknown_severity_is_rejected() -> None:
    with pytest.raises(ValueError):
        Severity("UNKNOWN")
