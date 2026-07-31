"""Structured metadata contract for immutable NYC 311 raw extractions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


FORBIDDEN_METADATA_KEYS = frozenset(
    {"authorization", "authorization_header", "x-app-token", "app_token", "token"}
)
COMPLETION_STATUSES = frozenset({"success", "failure"})


@dataclass(frozen=True)
class ExtractionMetadata:
    """Serializable provenance and integrity facts for one extraction run."""

    source_name: str
    dataset_id: str
    base_url: str
    scope_authority_path: str
    scope_authority_extraction_timestamp: str
    selected_agency: str
    selected_complaint_type: str
    selected_start_date: str
    selected_end_date: str
    extraction_start_utc: str
    extraction_completion_utc: str
    count_preflight_utc: str
    count_postflight_utc: str
    selected_source_columns: list[str]
    where_clause: str
    ordering: list[str]
    page_size: int
    timeout_seconds: float
    maximum_retries: int
    expected_source_count: int
    retrieved_row_count: int
    page_count: int
    page_row_counts: list[int]
    retry_count: int
    minimum_created_date: str | None
    maximum_created_date: str | None
    unique_key_count: int
    duplicate_unique_key_count: int
    raw_file_path: str
    raw_file_format: str
    raw_file_size: int
    schema_version: str
    query_hash: str
    run_id: str
    warnings: list[str]
    completion_status: str
    python_version: str
    package_version: str | None = None

    def validate(self) -> None:
        """Raise when metadata is incomplete, inconsistent, or unsafe."""
        required_text = {
            "source_name": self.source_name,
            "dataset_id": self.dataset_id,
            "base_url": self.base_url,
            "scope_authority_path": self.scope_authority_path,
            "selected_agency": self.selected_agency,
            "selected_complaint_type": self.selected_complaint_type,
            "where_clause": self.where_clause,
            "query_hash": self.query_hash,
            "run_id": self.run_id,
        }
        missing = [name for name, value in required_text.items() if not value]
        if missing:
            raise ValueError(f"Extraction metadata fields must be non-empty: {missing}")
        if self.completion_status not in COMPLETION_STATUSES:
            raise ValueError(f"Invalid metadata completion status: {self.completion_status}")
        if self.page_count != len(self.page_row_counts):
            raise ValueError("Metadata page count does not match page-row counts.")
        if sum(self.page_row_counts) != self.retrieved_row_count:
            raise ValueError("Metadata page-row total does not match retrieved row count.")
        if self.expected_source_count < 0 or self.retrieved_row_count < 0:
            raise ValueError("Metadata row counts cannot be negative.")
        for field_name in (
            "scope_authority_extraction_timestamp",
            "extraction_start_utc",
            "extraction_completion_utc",
            "count_preflight_utc",
            "count_postflight_utc",
        ):
            value = getattr(self, field_name)
            try:
                parsed = pd.Timestamp(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Metadata {field_name} is not a valid timestamp.") from error
            if pd.isna(parsed) or parsed.tzinfo is None:
                raise ValueError(f"Metadata {field_name} must be timezone-aware UTC.")
            if parsed.tz_convert("UTC").utcoffset().total_seconds() != 0:
                raise ValueError(f"Metadata {field_name} must use UTC.")
        lowered_keys = {key.casefold() for key in asdict(self)}
        if lowered_keys & FORBIDDEN_METADATA_KEYS:
            raise ValueError("Metadata contract contains a forbidden secret field.")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-compatible dictionary."""
        self.validate()
        return asdict(self)

    @classmethod
    def current_python_version(cls) -> str:
        """Return the interpreter version without platform-specific details."""
        return ".".join(str(value) for value in sys.version_info[:3])


def write_metadata(metadata: ExtractionMetadata, path: Path) -> None:
    """Write validated metadata as stable indented JSON."""
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_metadata(path: Path) -> ExtractionMetadata:
    """Read and validate metadata from a JSON artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read extraction metadata: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Extraction metadata JSON must contain an object.")
    forbidden = {str(key).casefold() for key in payload} & FORBIDDEN_METADATA_KEYS
    if forbidden:
        raise ValueError(f"Extraction metadata contains forbidden keys: {sorted(forbidden)}")
    try:
        metadata = ExtractionMetadata(**payload)
    except TypeError as error:
        raise ValueError("Extraction metadata fields do not match the contract.") from error
    metadata.validate()
    return metadata
