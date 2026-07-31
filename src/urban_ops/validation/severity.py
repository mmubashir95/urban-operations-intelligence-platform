"""Central severity semantics and command exit-code policy for validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd

from urban_ops.validation.models import CheckStatus, Severity, ValidationCheck


SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
}


def make_check(
    *,
    check_id: str,
    area: str,
    check_name: str,
    severity: Severity,
    passed: bool,
    observed_value: object,
    expected_value: object,
    affected_rows: int,
    total_rows: int,
    message: str,
    recommended_action: str,
) -> ValidationCheck:
    """Build a consistently-statused validation check."""
    if passed:
        status = CheckStatus.PASS
    elif severity in {Severity.CRITICAL, Severity.ERROR}:
        status = CheckStatus.FAIL
    elif severity is Severity.WARNING:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.INFO
    return ValidationCheck(
        check_id=check_id,
        area=area,
        check_name=check_name,
        severity=severity,
        status=status,
        observed_value=observed_value,
        expected_value=expected_value,
        affected_rows=int(affected_rows),
        affected_rate=float(affected_rows / total_rows) if total_rows else 0.0,
        message=message,
        recommended_step_7_action=recommended_action,
    )


def checks_frame(checks: Iterable[ValidationCheck]) -> pd.DataFrame:
    """Return checks in deterministic severity and identifier order."""
    rows = [check.to_dict() for check in checks]
    columns = [
        "check_id", "area", "check_name", "severity", "status",
        "observed_value", "expected_value", "affected_rows", "affected_rate",
        "message", "recommended_step_7_action",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    result["_rank"] = result["severity"].map(
        {severity.value: rank for severity, rank in SEVERITY_ORDER.items()}
    )
    return result.sort_values(["_rank", "check_id"], kind="stable").drop(
        columns="_rank"
    ).reset_index(drop=True)


def overall_status(checks: Iterable[ValidationCheck]) -> str:
    """Return the highest failed severity, or PASS when no issue is present."""
    failed = [
        check.severity for check in checks if check.status is not CheckStatus.PASS
    ]
    return min(failed, key=SEVERITY_ORDER.__getitem__).value if failed else "PASS"


def severity_counts(checks: Iterable[ValidationCheck]) -> dict[str, int]:
    """Count non-pass findings by severity with stable zero-valued keys."""
    counts = Counter(
        check.severity.value
        for check in checks
        if check.status is not CheckStatus.PASS
    )
    return {severity.value: counts[severity.value] for severity in Severity}


def validation_exit_code(
    checks: Iterable[ValidationCheck], *, fail_on_error: bool, fail_on_warning: bool
) -> int:
    """Map findings to CLI exit status according to the documented threshold."""
    failed = {
        check.severity for check in checks if check.status is not CheckStatus.PASS
    }
    if Severity.CRITICAL in failed:
        return 2
    if fail_on_warning and failed & {Severity.ERROR, Severity.WARNING}:
        return 1
    if fail_on_error and Severity.ERROR in failed:
        return 1
    return 0
