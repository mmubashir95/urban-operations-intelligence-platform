"""Unit tests for scope resolution, validation, CLI behavior, and atomic output."""

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.data.api_client import APIResponse
from urban_ops.data.ingest import (
    ExtractionValidationError,
    HttpConfig,
    IngestionConfig,
    InvalidScopeError,
    OutputAlreadyExistsError,
    OutputConfig,
    QueryConfig,
    ScopeConfig,
    SourceConfig,
    load_ingestion_config,
    load_ingestion_scope,
    run_ingestion,
    validate_extraction,
)
from urban_ops.data.metadata import read_metadata
from urban_ops.data.selected_scope import SelectedScope


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)
RUN_AT = datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc)


def write_authority(
    root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    metadata_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    authority = root / "selected_scope.csv"
    extraction = root / "extraction_metadata.csv"
    default = {
        "decision_status": "APPROVED_WITH_LIMITATIONS",
        "selected_agency": "DSNY", "selected_complaint_type": "Graffiti",
        "selected_start_date": "2024-01-01", "selected_end_date": "2025-12-31",
        "extraction_timestamp": "2026-07-27T17:04:43+00:00",
    }
    pd.DataFrame([default] if rows is None else rows).to_csv(authority, index=False)
    metadata_default = {
        "source": "test", "dataset_identifier": "erm2-nwe9",
        "extraction_timestamp": "2026-07-27T17:04:43+00:00",
        "agency": "DSNY", "complaint_type": "Graffiti",
    }
    pd.DataFrame(
        [metadata_default] if metadata_rows is None else metadata_rows
    ).to_csv(extraction, index=False)
    return authority, extraction


def config(tmp_path: Path) -> IngestionConfig:
    authority, extraction = write_authority(tmp_path)
    return IngestionConfig(
        SourceConfig("erm2-nwe9", "https://example.test/resource/erm2-nwe9.json"),
        ScopeConfig(authority, extraction, ("APPROVED_WITH_LIMITATIONS",)),
        QueryConfig(COLUMNS, ("created_date ASC", "unique_key ASC"), 2),
        HttpConfig(10, 2, 0, 0),
        OutputConfig(tmp_path / "raw", "parquet", "1.0"),
    )


def raw_row(key: str, created: str, **overrides: object) -> dict[str, object]:
    row = {column: None for column in COLUMNS}
    row.update({
        "unique_key": key, "created_date": created,
        "closed_date": "2024-01-10T00:00:00.000",
        "due_date": "2024-01-09T00:00:00.000", "agency": "DSNY",
        "agency_name": "Department of Sanitation", "complaint_type": "Graffiti",
        "descriptor": "raw descriptor", "status": "Closed",
        "borough": "BROOKLYN", "open_data_channel_type": "PHONE",
    })
    row.update(overrides)
    return row


def fixture_rows() -> list[dict[str, object]]:
    return [
        raw_row("1", "2024-01-01T01:00:00.000"),
        raw_row("same", "2024-01-02T01:00:00.000", descriptor="  Raw Value  "),
        raw_row("same", "2024-01-03T01:00:00.000", due_date=None),
    ]


def mocked_source(rows: list[dict[str, object]]):
    def request(url: str, **kwargs: object) -> APIResponse:
        query = parse_qs(urlsplit(url).query)["$query"][0]
        if query.startswith("SELECT count"):
            return APIResponse([{"record_count": str(len(rows))}], 0)
        offset = int(query.rsplit("OFFSET ", 1)[1])
        limit = int(query.rsplit("LIMIT ", 1)[1].split()[0])
        return APIResponse(rows[offset: offset + limit], 0)
    return request


def test_valid_authoritative_scope_loads_and_populates_query(tmp_path: Path) -> None:
    result = run_ingestion(config(tmp_path), dry_run=True, run_started_at=RUN_AT)
    assert result.scope.agency == "DSNY"
    assert "agency = 'DSNY'" in result.plan.where_clause


def test_missing_empty_multiple_and_invalid_authorities_fail(tmp_path: Path) -> None:
    missing = config(tmp_path)
    missing.scope.authority_file.unlink()
    with pytest.raises(InvalidScopeError, match="missing"):
        load_ingestion_scope(missing)

    cases = [
        [],
        [{"selected_agency": "DSNY"}, {"selected_agency": "DSNY"}],
        [{"decision_status": "APPROVED_WITH_LIMITATIONS", "selected_agency": "",
          "selected_complaint_type": "Graffiti", "selected_start_date": "2024-01-01",
          "selected_end_date": "2025-12-31", "extraction_timestamp": "2026-01-01Z"}],
    ]
    for index, rows in enumerate(cases):
        case_root = tmp_path / f"case-{index}"
        case_root.mkdir()
        authority, extraction = write_authority(case_root, rows=rows)
        candidate = config(tmp_path)
        candidate = IngestionConfig(candidate.source,
            ScopeConfig(authority, extraction, candidate.scope.acceptable_decision_statuses),
            candidate.query, candidate.http, candidate.output)
        with pytest.raises(InvalidScopeError):
            load_ingestion_scope(candidate)


@pytest.mark.parametrize(("field", "value"), [
    ("selected_complaint_type", ""),
    ("selected_start_date", "2026-01-01"),
])
def test_missing_complaint_type_and_invalid_boundaries_fail(
    tmp_path: Path, field: str, value: str
) -> None:
    row = {
        "decision_status": "APPROVED_WITH_LIMITATIONS", "selected_agency": "DSNY",
        "selected_complaint_type": "Graffiti", "selected_start_date": "2024-01-01",
        "selected_end_date": "2025-12-31",
        "extraction_timestamp": "2026-07-27T17:04:43+00:00",
    }
    row[field] = value
    candidate = config(tmp_path)
    authority, extraction = write_authority(tmp_path, rows=[row])
    candidate = IngestionConfig(candidate.source,
        ScopeConfig(authority, extraction, candidate.scope.acceptable_decision_statuses),
        candidate.query, candidate.http, candidate.output)
    with pytest.raises(InvalidScopeError):
        load_ingestion_scope(candidate)


def validation_scope() -> SelectedScope:
    return SelectedScope(
        "DSNY", "Graffiti", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"),
        pd.Timestamp("2026-07-27T17:04:43Z"), "APPROVED_WITH_LIMITATIONS",
        Path("authority.csv"), Path("metadata.csv"),
    )


@pytest.mark.parametrize(("changes", "failed_check"), [
    ({"agency": "DOT"}, "agency_scope"),
    ({"complaint_type": "Noise"}, "complaint_type_scope"),
    ({"created_date": "2023-12-31"}, "created_date_scope"),
    ({"created_date": "2026-01-01"}, "created_date_scope"),
])
def test_out_of_scope_rows_fail_validation(
    changes: dict[str, object], failed_check: str
) -> None:
    frame = pd.DataFrame([raw_row("1", "2024-01-01", **changes)])
    with pytest.raises(ExtractionValidationError, match=failed_check):
        validate_extraction(
            frame, scope=validation_scope(), selected_columns=COLUMNS,
            expected_count=1, final_source_count=1, query_hash="hash",
        )


def test_missing_columns_zero_rows_and_count_mismatch_fail() -> None:
    scope = validation_scope()
    with pytest.raises(ExtractionValidationError, match="required_source_columns"):
        validate_extraction(pd.DataFrame([{"unique_key": "1"}]), scope=scope,
            selected_columns=COLUMNS, expected_count=1, final_source_count=1,
            query_hash="hash")
    with pytest.raises(ExtractionValidationError, match="non_empty_extract"):
        validate_extraction(pd.DataFrame(columns=COLUMNS), scope=scope,
            selected_columns=COLUMNS, expected_count=0, final_source_count=0,
            query_hash="hash")
    with pytest.raises(ExtractionValidationError, match="preflight_count"):
        validate_extraction(pd.DataFrame([raw_row("1", "2024-01-01")]), scope=scope,
            selected_columns=COLUMNS, expected_count=2, final_source_count=2,
            query_hash="hash")


def test_dry_run_performs_no_http_and_writes_no_raw_artifact(tmp_path: Path) -> None:
    def forbidden_request(*args: object, **kwargs: object) -> APIResponse:
        raise AssertionError("dry run must not make HTTP requests")
    candidate = config(tmp_path)
    result = run_ingestion(candidate, dry_run=True, run_started_at=RUN_AT,
                           request_json=forbidden_request)
    assert not result.expected_output_directory.exists()


def test_atomic_raw_metadata_query_and_readback(tmp_path: Path) -> None:
    candidate = config(tmp_path)
    rows = fixture_rows()
    result = run_ingestion(
        candidate, report_directory=tmp_path / "reports", run_started_at=RUN_AT,
        request_json=mocked_source(rows),
    )
    stored = pd.read_parquet(result.raw_file)
    metadata = read_metadata(result.metadata_file)
    assert list(stored.columns) == list(COLUMNS)
    assert stored["unique_key"].tolist() == ["1", "same", "same"]
    assert stored.loc[1, "descriptor"] == "  Raw Value  "
    assert metadata.retrieved_row_count == len(stored) == 3
    assert metadata.duplicate_unique_key_count == 1
    assert result.query_file.is_file() and "LIMIT :limit OFFSET :offset" in result.query_file.read_text()
    assert (result.report_directory / "tables/page_summary.csv").is_file()
    assert not list(result.output_directory.parent.glob(".*.tmp-*"))


def test_sparse_json_omitted_null_field_is_materialized_and_reported(
    tmp_path: Path,
) -> None:
    candidate = config(tmp_path)
    rows = fixture_rows()
    for row in rows:
        row.pop("descriptor_2")
    result = run_ingestion(
        candidate,
        report_directory=tmp_path / "reports",
        run_started_at=RUN_AT,
        request_json=mocked_source(rows),
    )
    stored = pd.read_parquet(result.raw_file)
    assert "descriptor_2" in stored.columns
    assert stored["descriptor_2"].isna().all()
    assert any("sparse JSON" in warning for warning in result.metadata.warnings)


def test_successful_run_is_immutable_even_with_overwrite(tmp_path: Path) -> None:
    candidate, rows = config(tmp_path), fixture_rows()
    run_ingestion(candidate, report_directory=tmp_path / "reports", run_started_at=RUN_AT,
                  request_json=mocked_source(rows))
    with pytest.raises(OutputAlreadyExistsError, match="immutable"):
        run_ingestion(candidate, overwrite=True, report_directory=tmp_path / "reports",
                      run_started_at=RUN_AT, request_json=mocked_source(rows))


def test_overwrite_replaces_only_incomplete_matching_output(tmp_path: Path) -> None:
    candidate, rows = config(tmp_path), fixture_rows()
    dry = run_ingestion(candidate, dry_run=True, run_started_at=RUN_AT)
    dry.expected_output_directory.mkdir(parents=True)
    (dry.expected_output_directory / "partial.txt").write_text("partial")
    with pytest.raises(OutputAlreadyExistsError, match="pass --overwrite"):
        run_ingestion(candidate, report_directory=tmp_path / "reports",
                      run_started_at=RUN_AT, request_json=mocked_source(rows))
    result = run_ingestion(candidate, overwrite=True, report_directory=tmp_path / "reports",
                           run_started_at=RUN_AT, request_json=mocked_source(rows))
    assert not (result.output_directory / "partial.txt").exists()


def test_failed_write_leaves_no_final_success_or_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, rows = config(tmp_path), fixture_rows()
    dry = run_ingestion(candidate, dry_run=True, run_started_at=RUN_AT)
    monkeypatch.setattr(pd.DataFrame, "to_parquet",
                        lambda self, path, **kwargs: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        run_ingestion(candidate, report_directory=tmp_path / "reports",
                      run_started_at=RUN_AT, request_json=mocked_source(rows))
    assert not dry.expected_output_directory.exists()
    assert not list(dry.expected_output_directory.parent.glob(".*.tmp-*"))


def test_yaml_configuration_loads(tmp_path: Path) -> None:
    authority, extraction = write_authority(tmp_path)
    payload = {
        "source": {"dataset_id": "erm2-nwe9",
                   "base_url": "https://example.test/erm2-nwe9.json"},
        "scope": {"authority_file": str(authority),
                  "extraction_metadata_file": str(extraction),
                  "acceptable_decision_statuses": ["APPROVED_WITH_LIMITATIONS"]},
        "query": {"selected_columns": list(COLUMNS), "order_by": list(("created_date ASC", "unique_key ASC")), "page_size": 2},
        "http": {"timeout_seconds": 10, "max_retries": 2,
                 "initial_backoff_seconds": 0, "maximum_backoff_seconds": 0},
        "output": {"root": str(tmp_path / "raw"), "format": "parquet",
                   "schema_version": "1.0"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_ingestion_config(path).query.page_size == 2
