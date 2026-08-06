"""Validated provenance contract for immutable chronological split runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


FORBIDDEN_KEYS = frozenset({"token", "authorization", "password", "secret"})


@dataclass(frozen=True)
class SplitMetadata:
    """Serializable source lineage, boundaries, counts, rates, and output integrity."""

    schema_version: str
    split_id: str
    completion_status: str
    created_at_utc: str
    source_cleaning_run_id: str
    source_cleaning_metadata_path: str
    source_eligible_dataset_path: str
    source_eligible_sha256: str
    source_eligible_mtime: int
    source_raw_run_id: str
    source_raw_sha256: str
    scope_agency: str
    scope_complaint_type: str
    scope_start_date: str
    scope_end_date: str
    timestamp_column: str
    identifier_column: str
    target_column: str
    interval_convention: str
    train_start_inclusive: str
    train_end_exclusive: str
    validation_start_inclusive: str
    validation_end_exclusive: str
    test_start_inclusive: str
    test_end_exclusive: str
    input_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    train_on_time_count: int
    train_missed_count: int
    validation_on_time_count: int
    validation_missed_count: int
    test_on_time_count: int
    test_missed_count: int
    train_missed_target_rate: float
    validation_missed_target_rate: float
    test_missed_target_rate: float
    train_min_created_date: str
    train_max_created_date: str
    validation_min_created_date: str
    validation_max_created_date: str
    test_min_created_date: str
    test_max_created_date: str
    train_month_count: int
    validation_month_count: int
    test_month_count: int
    unassigned_row_count: int
    multiply_assigned_row_count: int
    train_validation_identifier_overlap: int
    train_test_identifier_overlap: int
    validation_test_identifier_overlap: int
    config_hash: str
    output_paths: dict[str, str]
    output_hashes: dict[str, str]
    warnings: list[str]
    python_version: str
    package_versions: dict[str, str]

    def validate(self) -> None:
        """Raise when successful split metadata does not reconcile."""
        if self.completion_status != "success":
            raise ValueError("Final split metadata must have completion_status='success'.")
        required = {
            "split_id": self.split_id,
            "source_cleaning_run_id": self.source_cleaning_run_id,
            "source_eligible_sha256": self.source_eligible_sha256,
            "config_hash": self.config_hash,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"Split metadata fields must be non-empty: {missing}")
        if self.input_row_count != sum((
            self.train_row_count, self.validation_row_count, self.test_row_count,
        )):
            raise ValueError("Input rows do not reconcile to train, validation, and test.")
        for name in ("train", "validation", "test"):
            rows = getattr(self, f"{name}_row_count")
            on_time = getattr(self, f"{name}_on_time_count")
            missed = getattr(self, f"{name}_missed_count")
            rate = getattr(self, f"{name}_missed_target_rate")
            if rows != on_time + missed:
                raise ValueError(f"{name} target counts do not reconcile.")
            expected_rate = missed / rows if rows else 0.0
            if abs(rate - expected_rate) > 1e-12:
                raise ValueError(f"{name} target rate is inconsistent.")
        if self.unassigned_row_count or self.multiply_assigned_row_count:
            raise ValueError("Successful split metadata cannot record assignment failures.")
        if any((
            self.train_validation_identifier_overlap,
            self.train_test_identifier_overlap,
            self.validation_test_identifier_overlap,
        )):
            raise ValueError("Successful split metadata cannot record identifier overlap.")
        created = pd.Timestamp(self.created_at_utc)
        if pd.isna(created) or created.tzinfo is None or str(created.tz_convert("UTC").tz) != "UTC":
            raise ValueError("created_at_utc must be a timezone-aware UTC timestamp.")
        for field_name in (
            "train_start_inclusive", "train_end_exclusive",
            "validation_start_inclusive", "validation_end_exclusive",
            "test_start_inclusive", "test_end_exclusive",
            "train_min_created_date", "train_max_created_date",
            "validation_min_created_date", "validation_max_created_date",
            "test_min_created_date", "test_max_created_date",
        ):
            value = pd.Timestamp(getattr(self, field_name))
            if pd.isna(value) or value.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware.")
        if not self.output_paths or not self.output_hashes:
            raise ValueError("Successful split metadata requires output paths and hashes.")
        if {key.casefold() for key in asdict(self)} & FORBIDDEN_KEYS:
            raise ValueError("Split metadata contains a forbidden secret field.")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-compatible mapping."""
        self.validate()
        return asdict(self)

    @classmethod
    def current_python_version(cls) -> str:
        """Return the interpreter version without machine-specific details."""
        return ".".join(str(value) for value in sys.version_info[:3])


def write_split_metadata(metadata: SplitMetadata, path: Path) -> None:
    """Write stable sorted JSON after validating all metadata relationships."""
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_split_metadata(path: Path) -> SplitMetadata:
    """Read and validate one split metadata artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Split metadata must contain a JSON object.")
        metadata = SplitMetadata(**payload)
        metadata.validate()
        return metadata
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Unable to read split metadata: {path}") from error
