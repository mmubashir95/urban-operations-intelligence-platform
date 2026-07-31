"""Unit tests for extraction metadata serialization and safety."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.data.metadata import ExtractionMetadata, read_metadata, write_metadata


def metadata(**overrides: object) -> ExtractionMetadata:
    values = {
        "source_name": "NYC Open Data — 311 Service Requests",
        "dataset_id": "erm2-nwe9",
        "base_url": "https://data.cityofnewyork.us/resource/erm2-nwe9.json",
        "scope_authority_path": "reports/05/selected.csv",
        "scope_authority_extraction_timestamp": "2026-07-27T17:04:43+00:00",
        "selected_agency": "DSNY",
        "selected_complaint_type": "Graffiti",
        "selected_start_date": "2024-01-01",
        "selected_end_date": "2025-12-31",
        "extraction_start_utc": "2026-07-31T10:00:00+00:00",
        "extraction_completion_utc": "2026-07-31T10:01:00+00:00",
        "count_preflight_utc": "2026-07-31T10:00:00+00:00",
        "count_postflight_utc": "2026-07-31T10:00:59+00:00",
        "selected_source_columns": ["unique_key", "created_date"],
        "where_clause": "agency = 'DSNY'",
        "ordering": ["created_date ASC", "unique_key ASC"],
        "page_size": 2,
        "timeout_seconds": 60.0,
        "maximum_retries": 5,
        "expected_source_count": 3,
        "retrieved_row_count": 3,
        "page_count": 2,
        "page_row_counts": [2, 1],
        "retry_count": 1,
        "minimum_created_date": "2024-01-01T00:00:00+00:00",
        "maximum_created_date": "2024-01-03T00:00:00+00:00",
        "unique_key_count": 3,
        "duplicate_unique_key_count": 0,
        "raw_file_path": "data/raw/run/service_requests.parquet",
        "raw_file_format": "parquet",
        "raw_file_size": 123,
        "schema_version": "1.0",
        "query_hash": "abc123",
        "run_id": "run-1",
        "warnings": ["one warning"],
        "completion_status": "success",
        "python_version": ExtractionMetadata.current_python_version(),
        "package_version": None,
    }
    values.update(overrides)
    return ExtractionMetadata(**values)


def test_required_scope_query_counts_and_runtime_are_serialized() -> None:
    payload = metadata().to_dict()
    assert payload["selected_agency"] == "DSNY"
    assert payload["where_clause"] == "agency = 'DSNY'"
    assert payload["expected_source_count"] == payload["retrieved_row_count"] == 3
    assert payload["page_count"] == len(payload["page_row_counts"])
    assert sum(payload["page_row_counts"]) == payload["retrieved_row_count"]
    assert payload["python_version"]


def test_metadata_json_round_trip_preserves_paths_warnings_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    expected = metadata()
    write_metadata(expected, path)
    actual = read_metadata(path)
    assert actual == expected
    assert actual.raw_file_path.endswith("service_requests.parquet")
    assert actual.warnings == ["one warning"]
    assert actual.query_hash == "abc123"


@pytest.mark.parametrize("field", [
    "scope_authority_extraction_timestamp", "extraction_start_utc",
    "extraction_completion_utc", "count_preflight_utc", "count_postflight_utc",
])
def test_metadata_timestamps_must_be_utc(field: str) -> None:
    with pytest.raises(ValueError, match="UTC"):
        replace(metadata(), **{field: "2026-07-31T10:00:00"}).validate()


def test_page_counts_must_reconcile() -> None:
    with pytest.raises(ValueError, match="page-row total"):
        replace(metadata(), page_row_counts=[2, 2]).validate()


def test_success_and_failure_status_are_distinct() -> None:
    assert metadata().completion_status == "success"
    failure = metadata(
        completion_status="failure", expected_source_count=0,
        retrieved_row_count=0, page_count=0, page_row_counts=[],
    )
    failure.validate()
    assert failure.completion_status == "failure"


@pytest.mark.parametrize("secret_key", ["app_token", "Authorization", "X-App-Token"])
def test_secret_fields_are_rejected_on_read(tmp_path: Path, secret_key: str) -> None:
    payload = metadata().to_dict()
    payload[secret_key] = "secret-token"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        read_metadata(path)


def test_metadata_json_contains_no_secret_fields(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_metadata(metadata(), path)
    content = path.read_text(encoding="utf-8").casefold()
    assert "secret-token" not in content
    assert "authorization" not in content
    assert "x-app-token" not in content
