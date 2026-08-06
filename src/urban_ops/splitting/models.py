"""Typed contracts shared by time-split analysis, persistence, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from urban_ops.cleaning.metadata import CleaningMetadata
    from urban_ops.data.selected_scope import SelectedScope


@dataclass(frozen=True)
class SplitBoundaries:
    """UTC half-open train, validation, and test boundaries."""

    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp
    interval_convention: str = "left_closed_right_open"

    def to_dict(self) -> dict[str, str]:
        """Return ISO-formatted boundary values suitable for reports."""
        return {
            name: value.isoformat()
            for name, value in asdict(self).items()
            if isinstance(value, pd.Timestamp)
        } | {"interval_convention": self.interval_convention}


@dataclass(frozen=True)
class CandidateSplitResult:
    """Metrics and decision evidence for one chronological candidate."""

    candidate_id: str
    boundaries: SplitBoundaries
    metrics: dict[str, Any]
    qualified: bool
    rank: int
    recommended: bool
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Flatten candidate boundaries and metrics into a report row."""
        return {
            "candidate_id": self.candidate_id,
            **self.boundaries.to_dict(),
            **self.metrics,
            "qualified": self.qualified,
            "rank": self.rank,
            "recommended": self.recommended,
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class SplitIntegrityCheck:
    """One source, assignment, chronology, target, or output check."""

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
class SplitFrames:
    """Deterministically ordered train, validation, and test frames."""

    train: "pd.DataFrame"
    validation: "pd.DataFrame"
    test: "pd.DataFrame"
    unassigned_row_count: int = 0
    multiply_assigned_row_count: int = 0


@dataclass(frozen=True)
class SplitRunPaths:
    """Canonical artifact locations for one immutable split run."""

    run_directory: Path
    train: Path
    validation: Path
    test: Path
    metadata: Path
    rules_snapshot: Path
    latest_pointer: Path


@dataclass(frozen=True)
class SourceEvidence:
    """Verified Step 7 source dataset and retained provenance."""

    run_path: Path
    metadata_path: Path
    eligible_path: Path
    eligible_sha256: str
    eligible_mtime_ns: int
    metadata: "CleaningMetadata"
    scope: "SelectedScope"
    frame: "pd.DataFrame"


@dataclass(frozen=True)
class SplitRunResult:
    """Complete in-memory and persisted outcome of one split invocation."""

    dry_run: bool
    source: SourceEvidence
    boundaries: SplitBoundaries
    candidates: tuple[CandidateSplitResult, ...]
    frames: SplitFrames
    checks: tuple[SplitIntegrityCheck, ...]
    tables: dict[str, "pd.DataFrame"]
    paths: SplitRunPaths
    source_sha256_after: str
    source_mtime_after_ns: int
    metadata: Any | None
