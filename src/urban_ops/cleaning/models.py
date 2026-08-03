"""Typed contracts shared by cleaning transforms, persistence, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CleaningAction:
    """One aggregate, governed transformation or preservation action."""

    action_id: str
    cleaning_area: str
    source_column: str
    action_type: str
    input_value: object
    output_value: object
    affected_rows: int
    reason: str
    source_rule: str
    governance_status: str
    applied: bool

    def to_dict(self) -> dict[str, object]:
        """Return a report-ready action mapping."""
        return asdict(self)


@dataclass(frozen=True)
class CleaningCheck:
    """One cleaning execution or reconciliation check."""

    check_id: str
    area: str
    status: str
    observed_value: object
    expected_value: object
    affected_rows: int
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a report-ready check mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ProcessedOutputPaths:
    """Canonical files belonging to one immutable processed run."""

    run_directory: Path
    cleaned: Path
    eligible: Path
    excluded: Path
    metadata: Path
    rules_snapshot: Path
    latest_pointer: Path


@dataclass(frozen=True)
class ValidationEvidence:
    """Machine-read Step 6 facts required before cleaning may proceed."""

    report_root: Path
    raw_run_id: str
    raw_sha256: str
    row_count: int
    column_count: int
    critical_finding_count: int
    overall_status: str
    candidate_eligible_count: int
    candidate_ineligible_count: int
    chronology: "pd.DataFrame"


@dataclass(frozen=True)
class CleaningFrames:
    """In-memory cleaned, eligible, and excluded record collections."""

    cleaned: "pd.DataFrame"
    eligible: "pd.DataFrame"
    excluded: "pd.DataFrame"


@dataclass(frozen=True)
class CleaningRunResult:
    """Outcome, evidence, and artifact paths from one cleaning invocation."""

    dry_run: bool
    raw_run_path: Path
    raw_file: Path
    raw_sha256_before: str
    raw_sha256_after: str
    validation_evidence: ValidationEvidence
    output_paths: ProcessedOutputPaths
    input_row_count: int
    cleaned_row_count: int
    eligible_row_count: int
    excluded_row_count: int
    removed_exact_duplicate_copies: int
    cleaning_hash: str
    actions: tuple[CleaningAction, ...]
    checks: tuple[CleaningCheck, ...]
    tables: dict[str, "pd.DataFrame"]
    metadata: Any | None

    @property
    def raw_file_modified(self) -> bool:
        """Return whether raw bytes changed during cleaning."""
        return self.raw_sha256_before != self.raw_sha256_after
