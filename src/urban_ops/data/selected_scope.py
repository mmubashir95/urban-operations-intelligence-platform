"""Load the Step 3 scope authority and its selected NYC 311 population."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from urban_ops.data.api_client import fetch_json_url
from urban_ops.data.nyc_311_config import (
    API_ENDPOINT, API_MAX_ATTEMPTS, API_TIMEOUT_SECONDS, SELECTED_COLUMNS,
)
from urban_ops.utils.paths import PROJECT_ROOT


DEFAULT_SCOPE_AUTHORITY_PATH = (
    PROJECT_ROOT / "reports/05_temporal_stability/tables/selected_scope_summary.csv"
)
DEFAULT_EXTRACTION_METADATA_PATH = (
    PROJECT_ROOT / "reports/05_temporal_stability/tables/extraction_metadata.csv"
)


@dataclass(frozen=True)
class SelectedScope:
    """Validated identifiers and boundaries owned by Step 3."""

    agency: str
    complaint_type: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    authority_extraction_timestamp: pd.Timestamp
    decision_status: str
    authority_path: Path
    extraction_metadata_path: Path


def _value(row: pd.Series, column: str) -> str:
    """Return one required non-empty artifact value."""
    if column not in row or pd.isna(row[column]) or not str(row[column]).strip():
        raise ValueError(f"Selected-scope authority requires {column}.")
    return str(row[column]).strip()


def load_selected_scope_authority(
    authority_path: Path = DEFAULT_SCOPE_AUTHORITY_PATH,
    extraction_metadata_path: Path = DEFAULT_EXTRACTION_METADATA_PATH,
) -> SelectedScope:
    """Load and validate exactly one temporal scope authority row."""
    if not authority_path.is_file():
        raise FileNotFoundError(f"Selected-scope authority is missing: {authority_path}")
    if not extraction_metadata_path.is_file():
        raise FileNotFoundError(
            f"Selected-scope extraction metadata is missing: {extraction_metadata_path}"
        )
    authority, metadata = pd.read_csv(authority_path), pd.read_csv(extraction_metadata_path)
    if len(authority) != 1 or len(metadata) != 1:
        raise ValueError("Scope authority and extraction metadata must each contain one row.")
    row = authority.iloc[0]
    try:
        start = pd.Timestamp(_value(row, "selected_start_date")).normalize()
        end = pd.Timestamp(_value(row, "selected_end_date")).normalize()
        extracted = pd.Timestamp(_value(row, "extraction_timestamp"))
    except (TypeError, ValueError) as error:
        raise ValueError("Selected-scope authority contains an invalid timestamp.") from error
    if start > end:
        raise ValueError("Selected-scope start date must not be after end date.")
    extracted = (
        extracted.tz_localize("UTC") if extracted.tzinfo is None
        else extracted.tz_convert("UTC")
    )
    for required in ("source", "dataset_identifier", "extraction_timestamp"):
        _value(metadata.iloc[0], required)
    try:
        metadata_timestamp = pd.Timestamp(_value(metadata.iloc[0], "extraction_timestamp"))
    except (TypeError, ValueError) as error:
        raise ValueError("Extraction metadata timestamp is invalid.") from error
    if pd.isna(metadata_timestamp):
        raise ValueError("Extraction metadata timestamp is invalid.")
    agency, complaint = _value(row, "selected_agency"), _value(row, "selected_complaint_type")
    if str(metadata.iloc[0].get("agency", agency)) != agency:
        raise ValueError("Extraction metadata agency disagrees with scope authority.")
    if str(metadata.iloc[0].get("complaint_type", complaint)) != complaint:
        raise ValueError("Extraction metadata complaint type disagrees with scope authority.")
    return SelectedScope(
        agency, complaint, start, end, extracted, _value(row, "decision_status"),
        authority_path, extraction_metadata_path,
    )


def _quote(value: str) -> str:
    """Quote a scope value as a SoQL literal."""
    return "'" + value.replace("'", "''") + "'"


def _fetch(query: str, headers: Mapping[str, str]) -> list[dict[str, object]]:
    """Execute and validate one read-only SoQL query."""
    payload = fetch_json_url(
        f"{API_ENDPOINT}?{urlencode({'$query': query})}",
        timeout_seconds=API_TIMEOUT_SECONDS,
        max_attempts=API_MAX_ATTEMPTS,
        headers=headers,
    )
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("NYC Open Data returned an invalid record collection.")
    return payload


def fetch_selected_scope_records(
    scope: SelectedScope,
    *,
    columns: Sequence[str] = SELECTED_COLUMNS,
    page_size: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch every selected row with stable ordering and count reconciliation."""
    required = {
        "unique_key", "created_date", "closed_date", "due_date", "agency",
        "complaint_type", "status",
    }
    missing = sorted(required.difference(columns))
    if missing or page_size < 1:
        raise ValueError(f"Invalid selected-scope extraction configuration: {missing}")
    end_exclusive = scope.end_date + pd.Timedelta(days=1)
    where = (
        f"agency = {_quote(scope.agency)} AND complaint_type = {_quote(scope.complaint_type)} "
        f"AND created_date >= '{scope.start_date:%Y-%m-%d}T00:00:00.000' "
        f"AND created_date < '{end_exclusive:%Y-%m-%d}T00:00:00.000'"
    )
    headers = {"User-Agent": "urban-operations-intelligence-target-governance/1.0"}
    if os.getenv("NYC_OPEN_DATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["NYC_OPEN_DATA_APP_TOKEN"]
    count_query = f"SELECT count(*) AS row_count WHERE {where}"
    before = _fetch(count_query, headers)
    if len(before) != 1 or "row_count" not in before[0]:
        raise RuntimeError("Selected-scope count query returned an invalid result.")
    expected = int(before[0]["row_count"])
    records: list[dict[str, object]] = []
    offset = 0
    while True:
        page = _fetch(
            f"SELECT {', '.join(columns)} WHERE {where} "
            f"ORDER BY created_date, unique_key LIMIT {page_size} OFFSET {offset}",
            headers,
        )
        records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    after = int(_fetch(count_query, headers)[0]["row_count"])
    if expected != after or len(records) != expected:
        raise RuntimeError("Live selected-scope count changed or pagination was incomplete.")
    frame = pd.DataFrame.from_records(records).reindex(columns=list(columns))
    extracted_at = datetime.now(timezone.utc)
    metadata = pd.DataFrame([{
        "source": API_ENDPOINT, "dataset_identifier": "erm2-nwe9",
        "extraction_timestamp": extracted_at.isoformat(),
        "scope_authority_extraction_timestamp": scope.authority_extraction_timestamp.isoformat(),
        "agency": scope.agency, "complaint_type": scope.complaint_type,
        "requested_start_date": scope.start_date.date().isoformat(),
        "requested_end_date": scope.end_date.date().isoformat(),
        "row_count": len(frame), "ordering": "created_date ASC, unique_key ASC",
        "page_size": page_size, "page_count": (len(frame) + page_size - 1) // page_size,
        "scope_authority_path": str(scope.authority_path.relative_to(PROJECT_ROOT)),
    }])
    return frame, metadata
