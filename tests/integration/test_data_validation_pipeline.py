"""End-to-end Step 6 validation test using only local immutable fixtures."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from urban_ops.data.metadata import ExtractionMetadata
from urban_ops.validation.pipeline import main, run_validation
from urban_ops.validation.reports import REQUIRED_REPORT_TABLES


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)


def row(key: str, **changes: object) -> dict[str, object]:
    base = {
        "unique_key": key, "created_date": "2024-01-01T00:00:00.000",
        "closed_date": "2024-01-09T00:00:00.000",
        "due_date": "2024-01-10T00:00:00.000", "agency": "DSNY",
        "agency_name": "Department of Sanitation", "complaint_type": "Graffiti",
        "descriptor": "Graffiti", "descriptor_2": None, "status": "Closed",
        "borough": "QUEENS", "incident_zip": "01234", "latitude": "40.75",
        "longitude": "-73.90", "location_type": "Residential",
        "open_data_channel_type": "ONLINE", "resolution_description": "Resolved",
        "resolution_action_updated_date": "2024-01-09T00:00:00.000",
    }
    base.update(changes)
    return base


def metadata(run: Path, count: int, *, completion: str = "success") -> ExtractionMetadata:
    return ExtractionMetadata(
        source_name="NYC Open Data — 311 Service Requests", dataset_id="erm2-nwe9",
        base_url="https://example.test/resource/erm2-nwe9.json",
        scope_authority_path=str(run.parents[2] / "scope.csv"),
        scope_authority_extraction_timestamp="2026-01-01T00:00:00+00:00",
        selected_agency="DSNY", selected_complaint_type="Graffiti",
        selected_start_date="2024-01-01", selected_end_date="2025-12-31",
        extraction_start_utc="2026-01-01T00:00:00+00:00",
        extraction_completion_utc="2026-01-01T01:00:00+00:00",
        count_preflight_utc="2026-01-01T00:00:00+00:00",
        count_postflight_utc="2026-01-01T01:00:00+00:00",
        selected_source_columns=list(COLUMNS), where_clause="agency = 'DSNY'",
        ordering=["created_date ASC", "unique_key ASC"], page_size=10000,
        timeout_seconds=60, maximum_retries=5, expected_source_count=count,
        retrieved_row_count=count, page_count=1, page_row_counts=[count], retry_count=0,
        minimum_created_date="2024-01-01T00:00:00+00:00",
        maximum_created_date="2024-01-01T00:00:00+00:00", unique_key_count=count,
        duplicate_unique_key_count=0, raw_file_path=str(run / "service_requests.parquet"),
        raw_file_format="parquet", raw_file_size=1, schema_version="1.0",
        query_hash="fixturehash", run_id=run.name.removeprefix("run_id="), warnings=[],
        completion_status=completion, python_version="3.13.7",
    )


def write_run(run: Path, frame: pd.DataFrame, *, completion: str = "success") -> None:
    run.mkdir(parents=True)
    frame.to_parquet(run / "service_requests.parquet", index=False)
    (run / "query.sql").write_text("SELECT fixture", encoding="utf-8")
    payload = asdict(metadata(run, len(frame), completion=completion))
    (run / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def setup_fixture(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    scope = tmp_path / "scope.csv"
    scope_metadata = tmp_path / "scope_metadata.csv"
    pd.DataFrame([{
        "decision_status": "APPROVED_WITH_LIMITATIONS", "selected_agency": "DSNY",
        "selected_complaint_type": "Graffiti", "selected_start_date": "2024-01-01",
        "selected_end_date": "2025-12-31",
        "extraction_timestamp": "2026-01-01T00:00:00+00:00",
    }]).to_csv(scope, index=False)
    pd.DataFrame([{
        "source": "fixture", "dataset_identifier": "erm2-nwe9",
        "extraction_timestamp": "2026-01-01T00:00:00+00:00",
        "agency": "DSNY", "complaint_type": "Graffiti",
    }]).to_csv(scope_metadata, index=False)
    rows = [
        row("valid"),
        row("missing-due", due_date=None),
        row("missing-closed", closed_date=None, status="Open"),
        row("invalid-created", created_date="invalid"),
        row("due-before", due_date="2023-12-31"),
        row("closed-before", closed_date="2023-12-31"),
        row("exact"), row("exact"),
        row("conflict", status="Closed"), row("conflict", status="Pending"),
        row("variant", status=" closed "),
        row("unexpected", status="Novel"),
        row("missing-coordinate", latitude=None, longitude=None),
        row("invalid-coordinate", latitude="bad", incident_zip="12345.0"),
        row("outside-nyc", latitude="35", longitude="-100"),
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    raw_root = tmp_path / "raw"
    failed = raw_root / "extraction_date=2026-01-02" / "run_id=20260102T000000Z_failed"
    successful = raw_root / "extraction_date=2026-01-01" / "run_id=20260101T000000Z_fixturehash"
    write_run(failed, frame, completion="failure")
    write_run(successful, frame)
    config = {
        "input": {
            "raw_root": str(raw_root), "expected_dataset_id": "erm2-nwe9",
            "supported_format": "parquet", "scope_authority_file": str(scope),
            "scope_extraction_metadata_file": str(scope_metadata),
        },
        "output": {"report_root": str(tmp_path / "reports")},
        "schema": {"require_column_order": True, "required_columns": list(COLUMNS)},
        "timestamps": {"columns": [
            "created_date", "closed_date", "due_date", "resolution_action_updated_date"
        ], "timezone": "UTC"},
        "categories": {
            "columns": ["agency", "complaint_type", "descriptor", "descriptor_2", "status", "borough", "location_type", "open_data_channel_type"],
            "rare_category_max_rows": 1, "high_cardinality_threshold": 100,
            "maximum_profile_values_per_column": 100,
        },
        "geography": {
            "latitude_valid_range": [-90, 90], "longitude_valid_range": [-180, 180],
            "nyc_bounding_box": {"min_latitude": 40.4, "max_latitude": 41.0, "min_longitude": -74.3, "max_longitude": -73.6},
        },
        "missingness": {"null_like_strings": ["", "null", "none", "nan", "n/a", "na", "unknown"]},
    }
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, successful, frame


def test_complete_local_validation_pipeline_is_non_mutating_and_reproducible(
    tmp_path: Path,
) -> None:
    config, successful, source = setup_fixture(tmp_path)
    raw_file = successful / "service_requests.parquet"
    before_bytes = raw_file.read_bytes()
    before_mtime = raw_file.stat().st_mtime_ns
    result = run_validation(config_path=config)
    assert result.raw_run_path == successful
    assert result.raw_sha256_before == result.raw_sha256_after == hashlib.sha256(before_bytes).hexdigest()
    assert raw_file.read_bytes() == before_bytes
    assert raw_file.stat().st_mtime_ns == before_mtime
    assert not list(result.report_root.rglob("*.parquet"))
    assert set(REQUIRED_REPORT_TABLES) == set(result.tables)
    assert all((result.report_root / "tables" / name).is_file() for name in REQUIRED_REPORT_TABLES)
    assert (result.report_root / "validation_summary.md").is_file()
    assert result.tables["chronology_violations.csv"]["violation_type"].isin(
        ["due_before_created", "closed_before_created"]
    ).sum() >= 2
    duplicate_metrics = result.tables["duplicate_summary.csv"].set_index("metric")["row_count"]
    assert duplicate_metrics["redundant_exact_duplicate_rows"] == 1
    assert duplicate_metrics["conflicting_duplicate_keys"] == 1
    assert (~result.tables["status_validation.csv"]["is_expected"]).any()
    assert result.tables["geographic_outliers.csv"]["issue_type"].isin(
        ["invalid_numeric", "outside_nyc_bounding_box", "decimal_like", "non_digit"]
    ).all()
    assert result.tables["candidate_exclusion_reason_summary.csv"]["row_count"].sum() == len(source)
    assert "missed_resolution_target" not in result.tables["target_readiness_summary.csv"].columns
    first_checks = (result.report_root / "tables" / "validation_checks.csv").read_bytes()
    repeated = run_validation(config_path=config)
    assert (repeated.report_root / "tables" / "validation_checks.csv").read_bytes() == first_checks
    assert main(["--config", str(config)]) == 2  # invalid creation time is critical


def test_metadata_scope_mismatch_is_a_critical_result(tmp_path: Path) -> None:
    config, successful, _ = setup_fixture(tmp_path)
    metadata_path = successful / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["selected_agency"] = "DOT"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    result = run_validation(config_path=config, raw_run_path=successful)
    check = next(item for item in result.checks if item.check_id == "metadata_scope.agency")
    assert check.severity.value == "CRITICAL" and check.status.value == "FAIL"
    assert main(["--config", str(config), "--raw-run", str(successful)]) == 2
