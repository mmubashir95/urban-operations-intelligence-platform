"""Mocked end-to-end test for immutable multi-page API ingestion."""

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_ops.data.api_client import APIResponse
from urban_ops.data.ingest import (
    HttpConfig, IngestionConfig, OutputConfig, QueryConfig, ScopeConfig,
    SourceConfig, find_latest_successful_run, run_ingestion,
)
from urban_ops.data.metadata import read_metadata


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)


def record(key: str, created: str, descriptor: str) -> dict[str, object]:
    row = {column: None for column in COLUMNS}
    row.update({
        "unique_key": key, "created_date": created,
        "closed_date": "2024-01-05T00:00:00.000",
        "due_date": "2024-01-04T00:00:00.000", "agency": "DSNY",
        "agency_name": "Department of Sanitation", "complaint_type": "Graffiti",
        "descriptor": descriptor, "status": "Closed", "borough": "QUEENS",
        "open_data_channel_type": "ONLINE",
    })
    return row


def test_complete_mocked_ingestion_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NYC_OPEN_DATA_APP_TOKEN", "secret-token")
    authority = tmp_path / "selected.csv"
    scope_metadata = tmp_path / "scope_metadata.csv"
    pd.DataFrame([{
        "decision_status": "APPROVED_WITH_LIMITATIONS", "selected_agency": "DSNY",
        "selected_complaint_type": "Graffiti", "selected_start_date": "2024-01-01",
        "selected_end_date": "2025-12-31",
        "extraction_timestamp": "2026-07-27T17:04:43+00:00",
    }]).to_csv(authority, index=False)
    pd.DataFrame([{
        "source": "fixture", "dataset_identifier": "erm2-nwe9",
        "extraction_timestamp": "2026-07-27T17:04:43+00:00",
        "agency": "DSNY", "complaint_type": "Graffiti",
    }]).to_csv(scope_metadata, index=False)
    config = IngestionConfig(
        SourceConfig("erm2-nwe9", "https://example.test/resource/erm2-nwe9.json"),
        ScopeConfig(authority, scope_metadata, ("APPROVED_WITH_LIMITATIONS",)),
        QueryConfig(COLUMNS, ("created_date ASC", "unique_key ASC"), 2),
        HttpConfig(10, 3, 0, 0), OutputConfig(tmp_path / "raw", "parquet", "1.0"),
    )
    rows = [
        record("1", "2024-01-01T00:00:00.000", "unchanged"),
        record("duplicate", "2024-01-02T00:00:00.000", "  preserve spaces  "),
        record("duplicate", "2024-01-03T00:00:00.000", "different raw value"),
    ]
    offsets: list[int] = []
    count_calls = 0
    token_header_seen = False

    def request(url: str, **kwargs: object) -> APIResponse:
        nonlocal count_calls, token_header_seen
        headers = kwargs.get("headers")
        token_header_seen = token_header_seen or (
            isinstance(headers, dict) and headers.get("X-App-Token") == "secret-token"
        )
        query = parse_qs(urlsplit(url).query)["$query"][0]
        if query.startswith("SELECT count"):
            count_calls += 1
            return APIResponse([{"record_count": "3"}], retry_count=0)
        offset = int(query.rsplit("OFFSET ", 1)[1])
        limit = int(query.rsplit("LIMIT ", 1)[1].split()[0])
        offsets.append(offset)
        return APIResponse(rows[offset: offset + limit], retry_count=0)

    result = run_ingestion(
        config,
        report_directory=tmp_path / "reports",
        run_started_at=datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        request_json=request,
    )
    stored = pd.read_parquet(result.raw_file)
    metadata = read_metadata(result.metadata_file)
    page_summary = pd.read_csv(result.report_directory / "tables/page_summary.csv")
    validation = pd.read_csv(
        result.report_directory / "tables/extraction_validation.csv"
    )
    assert count_calls == 2
    assert token_header_seen
    assert offsets == [0, 2]
    assert stored["unique_key"].tolist() == ["1", "duplicate", "duplicate"]
    assert stored.loc[1, "descriptor"] == "  preserve spaces  "
    assert metadata.retrieved_row_count == metadata.expected_source_count == len(stored) == 3
    assert metadata.duplicate_unique_key_count == 1
    assert page_summary["offset"].tolist() == [0, 2]
    assert page_summary["returned_rows"].sum() == 3
    assert not validation["status"].eq("FAIL").any()
    assert find_latest_successful_run(config.output.root) == result.output_directory
    assert result.output_directory.is_dir()
    assert not list(result.output_directory.parent.glob(".*.tmp-*"))
    for artifact in (result.metadata_file, result.query_file,
                     result.report_directory / "ingestion_summary.md"):
        text = artifact.read_text(encoding="utf-8").casefold()
        assert "secret-token" not in text
        assert "x-app-token" not in text
        assert "authorization" not in text
    assert json.loads(result.metadata_file.read_text())["completion_status"] == "success"
