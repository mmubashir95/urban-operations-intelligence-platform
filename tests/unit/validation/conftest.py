"""Deterministic fixtures shared by Step 6 validation unit tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from urban_ops.data.metadata import ExtractionMetadata
from urban_ops.data.selected_scope import SelectedScope


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)


def make_row(key: str = "1", **changes: object) -> dict[str, object]:
    """Return one structurally complete raw NYC 311 fixture row."""
    row: dict[str, object] = {
        "unique_key": key,
        "created_date": "2024-01-01T12:00:00.000",
        "closed_date": "2024-01-09T12:00:00.000",
        "due_date": "2024-01-10T12:00:00.000",
        "agency": "DSNY",
        "agency_name": "Department of Sanitation",
        "complaint_type": "Graffiti",
        "descriptor": "Graffiti",
        "descriptor_2": None,
        "status": "Closed",
        "borough": "QUEENS",
        "incident_zip": "01234",
        "latitude": "40.75",
        "longitude": "-73.90",
        "location_type": "Residential",
        "open_data_channel_type": "ONLINE",
        "resolution_description": "Resolved",
        "resolution_action_updated_date": "2024-01-09T12:00:00.000",
    }
    row.update(changes)
    return row


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Return a small valid raw frame with all configured columns."""
    return pd.DataFrame([make_row("1"), make_row("2")], columns=COLUMNS)


@pytest.fixture
def extraction_metadata(tmp_path: Path) -> ExtractionMetadata:
    """Return valid metadata that reconciles to ``raw_frame``."""
    return ExtractionMetadata(
        source_name="NYC Open Data — 311 Service Requests",
        dataset_id="erm2-nwe9",
        base_url="https://example.test/resource/erm2-nwe9.json",
        scope_authority_path=str(tmp_path / "scope.csv"),
        scope_authority_extraction_timestamp="2026-01-01T00:00:00+00:00",
        selected_agency="DSNY",
        selected_complaint_type="Graffiti",
        selected_start_date="2024-01-01",
        selected_end_date="2025-12-31",
        extraction_start_utc="2026-01-01T00:00:00+00:00",
        extraction_completion_utc="2026-01-01T01:00:00+00:00",
        count_preflight_utc="2026-01-01T00:00:00+00:00",
        count_postflight_utc="2026-01-01T01:00:00+00:00",
        selected_source_columns=list(COLUMNS),
        where_clause="agency = 'DSNY'",
        ordering=["created_date ASC", "unique_key ASC"],
        page_size=10000,
        timeout_seconds=60,
        maximum_retries=5,
        expected_source_count=2,
        retrieved_row_count=2,
        page_count=1,
        page_row_counts=[2],
        retry_count=0,
        minimum_created_date="2024-01-01T12:00:00+00:00",
        maximum_created_date="2024-01-01T12:00:00+00:00",
        unique_key_count=2,
        duplicate_unique_key_count=0,
        raw_file_path=str(tmp_path / "service_requests.parquet"),
        raw_file_format="parquet",
        raw_file_size=100,
        schema_version="1.0",
        query_hash="abc123",
        run_id="20260101T000000Z_abc123",
        warnings=[],
        completion_status="success",
        python_version="3.13.7",
    )


@pytest.fixture
def selected_scope(tmp_path: Path) -> SelectedScope:
    """Return the governed DSNY/Graffiti calendar scope."""
    return SelectedScope(
        agency="DSNY",
        complaint_type="Graffiti",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2025-12-31"),
        authority_extraction_timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        decision_status="APPROVED_WITH_LIMITATIONS",
        authority_path=tmp_path / "scope.csv",
        extraction_metadata_path=tmp_path / "scope_metadata.csv",
    )
