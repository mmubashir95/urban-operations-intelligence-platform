"""Validated provenance contract for immutable processed cleaning runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


FORBIDDEN_KEYS = frozenset({"token", "authorization", "x-app-token", "password", "secret"})


@dataclass(frozen=True)
class CleaningMetadata:
    """Serializable lineage, actions, counts, and integrity for one cleaning run."""

    source_raw_run_id: str
    source_raw_parquet_path: str
    source_raw_sha256: str
    validation_report_root: str
    validation_raw_run_id: str
    validation_evidence_status: str
    validation_critical_count: int
    cleaning_started_utc: str
    cleaning_completed_utc: str
    cleaning_config_hash: str
    cleaning_run_id: str
    input_row_count: int
    cleaned_row_count: int
    eligible_row_count: int
    excluded_row_count: int
    removed_exact_duplicate_copies: int
    conflicting_duplicate_groups: int
    timestamp_parse_failures: int
    due_before_created_exclusions: int
    closed_before_created_exclusions: int
    missing_due_date_exclusions: int
    missing_closed_date_exclusions: int
    on_time_target_count: int
    missed_target_count: int
    missed_target_rate: float
    zero_variance_columns: list[str]
    all_null_columns: list[str]
    output_paths: dict[str, str]
    output_hashes: dict[str, str]
    warnings: list[str]
    completion_status: str
    python_version: str
    package_versions: dict[str, str]
    schema_version: str

    def validate(self) -> None:
        """Raise when lineage, counts, timestamps, or output hashes are inconsistent."""
        required = {
            "source_raw_run_id": self.source_raw_run_id,
            "source_raw_sha256": self.source_raw_sha256,
            "validation_raw_run_id": self.validation_raw_run_id,
            "cleaning_config_hash": self.cleaning_config_hash,
            "cleaning_run_id": self.cleaning_run_id,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"Cleaning metadata fields must be non-empty: {missing}")
        if self.completion_status != "success":
            raise ValueError("Final cleaning metadata must have completion_status='success'.")
        if self.validation_critical_count != 0:
            raise ValueError("Successful cleaning requires zero critical validation findings.")
        if self.input_row_count != self.cleaned_row_count + self.removed_exact_duplicate_copies:
            raise ValueError("Input rows do not reconcile to cleaned rows and duplicate copies.")
        if self.cleaned_row_count != self.eligible_row_count + self.excluded_row_count:
            raise ValueError("Cleaned rows do not reconcile to eligible and excluded rows.")
        if self.eligible_row_count != self.on_time_target_count + self.missed_target_count:
            raise ValueError("Eligible rows do not reconcile to binary target counts.")
        if not self.output_paths or not self.output_hashes:
            raise ValueError("Successful cleaning metadata requires output paths and hashes.")
        for field_name in ("cleaning_started_utc", "cleaning_completed_utc"):
            value = pd.Timestamp(getattr(self, field_name))
            if pd.isna(value) or value.tzinfo is None:
                raise ValueError(f"{field_name} must be a timezone-aware timestamp.")
        keys = {key.casefold() for key in asdict(self)}
        if keys & FORBIDDEN_KEYS:
            raise ValueError("Cleaning metadata contains a forbidden secret field.")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-compatible dictionary."""
        self.validate()
        return asdict(self)

    @classmethod
    def current_python_version(cls) -> str:
        """Return the interpreter version without platform details."""
        return ".".join(str(value) for value in sys.version_info[:3])


def write_cleaning_metadata(metadata: CleaningMetadata, path: Path) -> None:
    """Write stable, validated JSON metadata."""
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_cleaning_metadata(path: Path) -> CleaningMetadata:
    """Read and validate cleaning metadata from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Cleaning metadata must contain a JSON object.")
        metadata = CleaningMetadata(**payload)
        metadata.validate()
        return metadata
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Unable to read cleaning metadata: {path}") from error
