"""Typed contracts shared by the raw-data validation modules and CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class Severity(StrEnum):
    """Impact assigned to one validation finding."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class CheckStatus(StrEnum):
    """Outcome assigned to one validation check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationCheck:
    """One stable, reportable validation decision."""

    check_id: str
    area: str
    check_name: str
    severity: Severity
    status: CheckStatus
    observed_value: object
    expected_value: object
    affected_rows: int
    affected_rate: float
    message: str
    recommended_step_7_action: str

    def to_dict(self) -> dict[str, object]:
        """Return a CSV-safe representation with enum values serialized."""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class TimestampAnalysis:
    """Temporary parsed timestamp views and their report tables."""

    parsed: dict[str, "pd.Series"]
    summary: "pd.DataFrame"
    invalid_examples: "pd.DataFrame"


@dataclass(frozen=True)
class DuplicateAnalysis:
    """Duplicate flags and stable row-level evidence."""

    exact_flags: "pd.Series"
    conflicting_flags: "pd.Series"
    summary: "pd.DataFrame"
    exact_records: "pd.DataFrame"
    conflicting_records: "pd.DataFrame"


@dataclass(frozen=True)
class TargetReadinessAnalysis:
    """Provisional Step 6 readiness flags derived through Step 4 rules."""

    flags: "pd.DataFrame"
    summary: "pd.DataFrame"
    exclusions: "pd.DataFrame"


@dataclass(frozen=True)
class ValidationRunResult:
    """Artifacts and findings produced by a complete validation run."""

    raw_run_path: Path
    raw_file: Path
    metadata_file: Path
    query_file: Path
    report_root: Path
    checks: tuple[ValidationCheck, ...]
    tables: dict[str, "pd.DataFrame"]
    overall_status: str
    row_count: int
    column_count: int
    raw_sha256_before: str
    raw_sha256_after: str
    metadata: Any

    @property
    def raw_file_modified(self) -> bool:
        """Return whether the raw artifact bytes changed during validation."""
        return self.raw_sha256_before != self.raw_sha256_after
