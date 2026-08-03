"""End-to-end local validation-to-cleaning integration with immutable fixtures."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from urban_ops.cleaning.outputs import sha256_file
from urban_ops.cleaning.pipeline import run_cleaning
from urban_ops.data.metadata import ExtractionMetadata
from urban_ops.validation.pipeline import run_validation


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)


def row(key: str, **changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "unique_key": key, "created_date": "2024-01-01T00:00:00.000",
        "closed_date": "2024-01-09T00:00:00.000",
        "due_date": "2024-01-10T00:00:00.000", "agency": "DSNY",
        "agency_name": "Department of Sanitation", "complaint_type": "Graffiti",
        "descriptor": "Graffiti", "descriptor_2": None, "status": "Closed",
        "borough": "QUEENS", "incident_zip": "01234", "latitude": "40.75",
        "longitude": "-73.90", "location_type": "Residential",
        "open_data_channel_type": "UNKNOWN", "resolution_description": "Resolved",
        "resolution_action_updated_date": "2024-01-09T00:00:00.000",
    }
    base.update(changes)
    return base


def write_scope(tmp_path: Path) -> tuple[Path, Path]:
    scope, metadata = tmp_path / "scope.csv", tmp_path / "scope_metadata.csv"
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
    }]).to_csv(metadata, index=False)
    return scope, metadata


def write_raw_run(tmp_path: Path, scope: Path, frame: pd.DataFrame) -> Path:
    run = tmp_path / "raw/extraction_date=2026-01-01/run_id=20260101T000000Z_fixture"
    run.mkdir(parents=True)
    raw = run / "service_requests.parquet"
    frame.to_parquet(raw, index=False)
    (run / "query.sql").write_text("SELECT fixture", encoding="utf-8")
    metadata = ExtractionMetadata(
        source_name="NYC Open Data — 311 Service Requests", dataset_id="erm2-nwe9",
        base_url="https://example.test/resource/erm2-nwe9.json",
        scope_authority_path=str(scope),
        scope_authority_extraction_timestamp="2026-01-01T00:00:00+00:00",
        selected_agency="DSNY", selected_complaint_type="Graffiti",
        selected_start_date="2024-01-01", selected_end_date="2025-12-31",
        extraction_start_utc="2026-01-01T00:00:00+00:00",
        extraction_completion_utc="2026-01-01T01:00:00+00:00",
        count_preflight_utc="2026-01-01T00:00:00+00:00",
        count_postflight_utc="2026-01-01T01:00:00+00:00",
        selected_source_columns=list(COLUMNS), where_clause="agency = 'DSNY'",
        ordering=["created_date ASC", "unique_key ASC"], page_size=10000,
        timeout_seconds=60, maximum_retries=5, expected_source_count=len(frame),
        retrieved_row_count=len(frame), page_count=1, page_row_counts=[len(frame)],
        retry_count=0, minimum_created_date="2024-01-01T00:00:00+00:00",
        maximum_created_date="2024-01-01T00:00:00+00:00",
        unique_key_count=int(frame["unique_key"].nunique()),
        duplicate_unique_key_count=int(frame["unique_key"].duplicated().sum()),
        raw_file_path=str(raw), raw_file_format="parquet", raw_file_size=raw.stat().st_size,
        schema_version="1.0", query_hash="fixture", run_id="20260101T000000Z_fixture",
        warnings=[], completion_status="success", python_version="3.13.7",
    )
    (run / "metadata.json").write_text(json.dumps(asdict(metadata)), encoding="utf-8")
    return run


def write_configs(
    tmp_path: Path, scope: Path, scope_metadata: Path
) -> tuple[Path, Path, Path, Path]:
    validation_report = tmp_path / "validation_reports"
    validation = yaml.safe_load(Path("configs/data/validation_rules.yaml").read_text())
    validation["input"].update({
        "raw_root": str(tmp_path / "raw"), "scope_authority_file": str(scope),
        "scope_extraction_metadata_file": str(scope_metadata),
    })
    validation["output"]["report_root"] = str(validation_report)
    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text(yaml.safe_dump(validation), encoding="utf-8")
    processed, cleaning_reports = tmp_path / "processed", tmp_path / "cleaning_reports"
    cleaning = yaml.safe_load(Path("configs/data/cleaning_rules.yaml").read_text())
    cleaning["input"].update({
        "raw_root": str(tmp_path / "raw"),
        "validation_report_root": str(validation_report),
        "scope_authority_file": str(scope),
        "scope_extraction_metadata_file": str(scope_metadata),
    })
    cleaning["output"].update({"root": str(processed), "report_root": str(cleaning_reports)})
    cleaning_path = tmp_path / "cleaning.yaml"
    cleaning_path.write_text(yaml.safe_dump(cleaning), encoding="utf-8")
    return validation_path, cleaning_path, processed, cleaning_reports


def test_validation_to_cleaning_flow_is_governed_atomic_and_non_mutating(
    tmp_path: Path,
) -> None:
    rows = [
        row("on-time"),
        row("late", closed_date="2024-01-11T00:00:00.000"),
        row("equal", closed_date="2024-01-10T00:00:00.000"),
        row("missing-due", due_date=None),
        row("missing-closed", closed_date=None, status="Open"),
        row("closed-before", closed_date="2023-12-31T00:00:00.000", status="Pending"),
        row("due-before", due_date="2023-12-31T00:00:00.000"),
        row("exact"), row("exact"),
        row("conflict"), row("conflict", status="Pending"),
        row("blank", descriptor="   ", location_type=" "),
        row("whitespace", status=" Closed "),
        row("case", status="closed"),
        row("geo", borough="Unspecified", latitude=None, longitude=None),
    ]
    source = pd.DataFrame(rows, columns=COLUMNS)
    scope, scope_metadata = write_scope(tmp_path)
    raw_run = write_raw_run(tmp_path, scope, source)
    validation_config, cleaning_config, processed, cleaning_reports = write_configs(
        tmp_path, scope, scope_metadata
    )
    raw_file = raw_run / "service_requests.parquet"
    before_bytes, before_mtime = raw_file.read_bytes(), raw_file.stat().st_mtime_ns
    validation = run_validation(config_path=validation_config)
    assert validation.overall_status in {"ERROR", "WARNING"}
    dry = run_cleaning(config_path=cleaning_config, dry_run=True)
    assert not processed.exists() and not cleaning_reports.exists()
    result = run_cleaning(
        config_path=cleaning_config,
        run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert result.raw_sha256_before == result.raw_sha256_after == sha256_file(raw_file)
    assert raw_file.read_bytes() == before_bytes and raw_file.stat().st_mtime_ns == before_mtime
    assert result.input_row_count == 15
    assert result.removed_exact_duplicate_copies == 1
    assert result.cleaned_row_count == 14
    assert result.cleaned_row_count == result.eligible_row_count + result.excluded_row_count
    assert dry.eligible_row_count == result.eligible_row_count
    cleaned = pd.read_parquet(result.output_paths.cleaned)
    eligible = pd.read_parquet(result.output_paths.eligible)
    excluded = pd.read_parquet(result.output_paths.excluded)
    assert cleaned["created_date"].dt.tz is not None
    assert cleaned.loc[cleaned["unique_key"].eq("missing-due"), "due_date"].isna().all()
    assert cleaned.loc[cleaned["unique_key"].eq("missing-closed"), "closed_date"].isna().all()
    assert cleaned.loc[cleaned["unique_key"].eq("closed-before"), "closed_date"].iloc[0] < cleaned.loc[cleaned["unique_key"].eq("closed-before"), "created_date"].iloc[0]
    assert pd.isna(cleaned.loc[cleaned["unique_key"].eq("blank"), "descriptor"].iloc[0])
    assert cleaned.loc[cleaned["unique_key"].eq("geo"), "borough"].iloc[0] == "Unspecified"
    assert cleaned["open_data_channel_type"].eq("UNKNOWN").all()
    assert set(eligible.set_index("unique_key").loc[["on-time", "late", "equal"], "missed_resolution_target"].astype(int)) == {0, 1}
    assert int(eligible.set_index("unique_key").loc["equal", "missed_resolution_target"]) == 0
    assert excluded.loc[excluded["unique_key"].eq("conflict"), "primary_exclusion_reason"].eq("conflicting_duplicate_unique_key").all()
    assert excluded["missed_resolution_target"].isna().all()
    assert eligible["unique_key"].is_unique
    assert result.output_paths.latest_pointer.is_file()
    assert (cleaning_reports / "cleaning_summary.md").is_file()
    assert all(table["status"].eq("PASS").all() for name, table in result.tables.items() if name == "output_reconciliation.csv")
    second = run_cleaning(
        config_path=cleaning_config,
        run_started_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    for key in ("cleaned", "eligible", "excluded", "rules_snapshot"):
        assert result.metadata.output_hashes[key] == second.metadata.output_hashes[key]
    assert json.loads(second.output_paths.latest_pointer.read_text())["run_id"] == second.metadata.cleaning_run_id
