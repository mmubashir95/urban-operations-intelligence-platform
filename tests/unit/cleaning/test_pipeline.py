"""Tests for Step 7 input authorization, dry runs, atomic runs, and CLI behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import urban_ops.cleaning.pipeline as pipeline_module
from urban_ops.cleaning.outputs import sha256_file
from urban_ops.cleaning.pipeline import (
    CleaningConfigurationError, CleaningInputError, load_cleaning_config,
    main, run_cleaning,
)
from urban_ops.data.metadata import ExtractionMetadata


@dataclass(frozen=True)
class PipelineFixture:
    config: Path
    raw_run: Path
    validation: Path
    output: Path
    reports: Path
    raw_file: Path


def extraction_metadata(run: Path, scope: Path, count: int) -> ExtractionMetadata:
    """Return valid Step 5 metadata for a temporary raw run."""
    return ExtractionMetadata(
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
        selected_source_columns=[
            "unique_key", "created_date", "closed_date", "due_date", "agency",
            "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
            "borough", "incident_zip", "latitude", "longitude", "location_type",
            "open_data_channel_type", "resolution_description",
            "resolution_action_updated_date",
        ],
        where_clause="agency = 'DSNY'", ordering=["created_date ASC", "unique_key ASC"],
        page_size=10000, timeout_seconds=60, maximum_retries=5,
        expected_source_count=count, retrieved_row_count=count, page_count=1,
        page_row_counts=[count], retry_count=0,
        minimum_created_date="2024-01-01T12:00:00+00:00",
        maximum_created_date="2024-01-01T12:00:00+00:00",
        unique_key_count=count, duplicate_unique_key_count=0,
        raw_file_path=str(run / "service_requests.parquet"), raw_file_format="parquet",
        raw_file_size=1, schema_version="1.0", query_hash="fixturehash",
        run_id=run.name.removeprefix("run_id="), warnings=[], completion_status="success",
        python_version="3.13.7",
    )


def build_fixture(tmp_path: Path, raw_frame: pd.DataFrame) -> PipelineFixture:
    """Write a complete local raw run, validation evidence, scope, and config."""
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
    raw_run = tmp_path / "raw/extraction_date=2026-01-01/run_id=20260101T000000Z_fixturehash"
    raw_run.mkdir(parents=True)
    raw_file = raw_run / "service_requests.parquet"
    raw_frame.to_parquet(raw_file, index=False)
    (raw_run / "query.sql").write_text("SELECT fixture", encoding="utf-8")
    metadata = extraction_metadata(raw_run, scope, len(raw_frame))
    (raw_run / "metadata.json").write_text(json.dumps(asdict(metadata)), encoding="utf-8")
    validation = tmp_path / "validation"
    tables = validation / "tables"
    tables.mkdir(parents=True)
    raw_hash = sha256_file(raw_file)
    (validation / "validation_summary.md").write_text(
        f"Raw run ID: `{metadata.run_id}`\nOverall validation status: **PASS**\n",
        encoding="utf-8",
    )
    checks = [
        {"check_id": "boundary.raw_immutable", "severity": "CRITICAL", "status": "PASS", "observed_value": raw_hash},
        *[
            {"check_id": f"metadata_scope.{field}", "severity": "CRITICAL", "status": "PASS", "observed_value": "ok"}
            for field in ("agency", "complaint_type", "start_date", "end_date", "authority_path")
        ],
    ]
    pd.DataFrame(checks).to_csv(tables / "validation_checks.csv", index=False)
    pd.DataFrame({"column_name": raw_frame.columns}).to_csv(
        tables / "schema_validation.csv", index=False
    )
    pd.DataFrame([{"scope_rule": "all", "status": "PASS"}]).to_csv(
        tables / "scope_validation.csv", index=False
    )
    pd.DataFrame([{
        "readiness_rule": "candidate_target_eligible", "pass_count": len(raw_frame),
        "fail_count": 0,
    }]).to_csv(tables / "target_readiness_summary.csv", index=False)
    pd.DataFrame(columns=["unique_key", "violation_type"]).to_csv(
        tables / "chronology_violations.csv", index=False
    )
    output, reports = tmp_path / "processed", tmp_path / "reports"
    config_payload = {
        "input": {
            "raw_root": str(tmp_path / "raw"), "validation_report_root": str(validation),
            "scope_authority_file": str(scope),
            "scope_extraction_metadata_file": str(scope_metadata),
            "require_validation_critical_count": 0,
        },
        "output": {"root": str(output), "report_root": str(reports), "format": "parquet", "schema_version": "1.0"},
        "timestamps": {"columns": ["created_date", "closed_date", "due_date", "resolution_action_updated_date"], "timezone": "UTC", "invalid_policy": "preserve_and_exclude_from_target", "impute": False},
        "categories": {"trim_columns": ["agency", "agency_name", "complaint_type", "descriptor", "descriptor_2", "status", "borough", "location_type", "open_data_channel_type"], "blank_to_null_columns": ["descriptor", "descriptor_2", "borough", "location_type"], "collapse_repeated_spaces_columns": [], "approved_mappings": {}},
        "duplicates": {"identifier": "unique_key", "exact_policy": "keep_deterministic", "conflict_policy": "exclude_entire_group"},
        "feature_decisions": {"all_null_columns": ["descriptor_2"], "zero_variance_columns": ["open_data_channel_type"]},
    }
    config = tmp_path / "cleaning.yaml"
    config.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    return PipelineFixture(config, raw_run, validation, output, reports, raw_file)


def test_config_loads_and_rejects_unapproved_imputation(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    assert load_cleaning_config(fixture.config).output_format == "parquet"
    payload = yaml.safe_load(fixture.config.read_text())
    payload["timestamps"]["impute"] = True
    fixture.config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(CleaningConfigurationError, match="imputation disabled"):
        load_cleaning_config(fixture.config)


def test_dry_run_locates_latest_and_writes_nothing(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    result = run_cleaning(config_path=fixture.config, dry_run=True)
    assert result.raw_run_path == fixture.raw_run and result.dry_run
    assert result.eligible_row_count == len(raw_frame)
    assert not fixture.output.exists() and not fixture.reports.exists()
    assert not result.output_paths.latest_pointer.exists()


def test_full_run_writes_outputs_reports_metadata_and_latest(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    result = run_cleaning(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert not result.dry_run and result.metadata is not None
    for path in (
        result.output_paths.cleaned, result.output_paths.eligible,
        result.output_paths.excluded, result.output_paths.metadata,
        result.output_paths.rules_snapshot, result.output_paths.latest_pointer,
    ):
        assert path.is_file()
    assert (fixture.reports / "cleaning_summary.md").is_file()
    assert set(result.metadata.output_hashes) == {"cleaned", "eligible", "excluded", "rules_snapshot"}
    assert pd.read_parquet(result.output_paths.eligible)["unique_key"].is_unique


def test_raw_bytes_and_mtime_remain_unchanged(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    before, mtime = fixture.raw_file.read_bytes(), fixture.raw_file.stat().st_mtime_ns
    result = run_cleaning(config_path=fixture.config)
    assert result.raw_sha256_before == result.raw_sha256_after
    assert fixture.raw_file.read_bytes() == before
    assert fixture.raw_file.stat().st_mtime_ns == mtime


def test_missing_validation_evidence_fails(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    (fixture.validation / "validation_summary.md").unlink()
    with pytest.raises(CleaningInputError, match="evidence is missing"):
        run_cleaning(config_path=fixture.config, dry_run=True)


def test_validation_run_or_hash_mismatch_fails(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    summary = fixture.validation / "validation_summary.md"
    summary.write_text("Raw run ID: `other`\nOverall validation status: **PASS**\n")
    with pytest.raises(CleaningInputError, match="another raw run"):
        run_cleaning(config_path=fixture.config, dry_run=True)
    summary.write_text(
        f"Raw run ID: `{fixture.raw_run.name.removeprefix('run_id=')}`\nOverall validation status: **PASS**\n"
    )
    checks = pd.read_csv(fixture.validation / "tables/validation_checks.csv")
    checks.loc[checks["check_id"].eq("boundary.raw_immutable"), "observed_value"] = "bad"
    checks.to_csv(fixture.validation / "tables/validation_checks.csv", index=False)
    with pytest.raises(CleaningInputError, match="SHA-256"):
        run_cleaning(config_path=fixture.config, dry_run=True)


def test_critical_validation_finding_fails(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    checks = pd.read_csv(fixture.validation / "tables/validation_checks.csv")
    checks.loc[len(checks)] = {
        "check_id": "scope.failure", "severity": "CRITICAL", "status": "FAIL",
        "observed_value": "1",
    }
    checks.to_csv(fixture.validation / "tables/validation_checks.csv", index=False)
    with pytest.raises(CleaningInputError, match="critical"):
        run_cleaning(config_path=fixture.config, dry_run=True)


def test_scope_mismatch_fails(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    path = fixture.raw_run / "metadata.json"
    payload = json.loads(path.read_text())
    payload["selected_agency"] = "DOT"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CleaningInputError, match="scope"):
        run_cleaning(config_path=fixture.config, dry_run=True)


def test_successful_run_is_immutable_for_same_run_id(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    started = datetime(2026, 1, 2, tzinfo=timezone.utc)
    run_cleaning(config_path=fixture.config, run_started_at=started)
    with pytest.raises(FileExistsError, match="immutable"):
        run_cleaning(config_path=fixture.config, run_started_at=started)


def test_partial_write_failure_leaves_no_run_or_latest(
    tmp_path: Path, raw_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    def fail(*args, **kwargs):
        raise RuntimeError("forced metadata failure")
    monkeypatch.setattr(pipeline_module, "write_cleaning_metadata", fail)
    with pytest.raises(RuntimeError, match="forced"):
        run_cleaning(
            config_path=fixture.config,
            run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    assert not list(fixture.output.glob("run_id=*"))
    assert not list(fixture.output.glob(".*.tmp-*"))
    assert not (fixture.output / "latest.json").exists()


def test_cli_success_and_failure_codes(tmp_path: Path, raw_frame) -> None:
    fixture = build_fixture(tmp_path, raw_frame)
    assert main(["--config", str(fixture.config), "--dry-run"]) == 0
    assert main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]) == 1
