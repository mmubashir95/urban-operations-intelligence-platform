"""Shared deterministic Step 7 and boundary fixtures for splitting tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from urban_ops.cleaning.metadata import CleaningMetadata, write_cleaning_metadata
from urban_ops.cleaning.outputs import sha256_file
from urban_ops.data.selected_scope import SelectedScope
from urban_ops.splitting.models import SplitBoundaries


@dataclass(frozen=True)
class SplitFixture:
    """Paths belonging to one isolated successful Step 7 fixture."""

    config: Path
    processed_root: Path
    latest_pointer: Path
    cleaning_run: Path
    eligible: Path
    output: Path
    reports: Path
    scope: Path
    scope_metadata: Path


def make_eligible_frame() -> pd.DataFrame:
    """Return two binary-target rows in each month across all three splits."""
    rows = []
    key = 0
    for month in range(1, 10):
        for target, day in ((0, 1), (1, 2)):
            key += 1
            rows.append({
                "unique_key": f"id-{key:03d}",
                "created_date": pd.Timestamp(2024, month, day, 12, tz="UTC"),
                "missed_resolution_target": target,
                "target_eligible": True,
                "agency": "DSNY", "complaint_type": "Graffiti",
            })
    frame = pd.DataFrame(rows)
    frame["missed_resolution_target"] = frame["missed_resolution_target"].astype("Int8")
    return frame


def fixture_boundaries() -> SplitBoundaries:
    """Return contiguous half-open boundaries for the small fixture."""
    return SplitBoundaries(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"),
    )


def write_scope(tmp_path: Path) -> tuple[Path, Path]:
    """Write a local Step 3 scope authority and matching extraction evidence."""
    scope, extraction = tmp_path / "scope.csv", tmp_path / "scope_metadata.csv"
    pd.DataFrame([{
        "decision_status": "APPROVED_WITH_LIMITATIONS", "selected_agency": "DSNY",
        "selected_complaint_type": "Graffiti", "selected_start_date": "2024-01-01",
        "selected_end_date": "2024-12-31",
        "extraction_timestamp": "2025-01-01T00:00:00+00:00",
    }]).to_csv(scope, index=False)
    pd.DataFrame([{
        "source": "fixture", "dataset_identifier": "erm2-nwe9",
        "extraction_timestamp": "2025-01-01T00:00:00+00:00",
        "agency": "DSNY", "complaint_type": "Graffiti",
    }]).to_csv(extraction, index=False)
    return scope, extraction


def write_cleaning_run(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    """Write a successful Step 7 run, metadata, and latest pointer."""
    processed = tmp_path / "processed"
    run = processed / "run_id=cleaning-fixture"
    run.mkdir(parents=True)
    eligible = run / "eligible_service_requests.parquet"
    frame.to_parquet(eligible, index=False)
    eligible_hash = sha256_file(eligible)
    missed = int(frame["missed_resolution_target"].eq(1).sum())
    # Cleaning metadata only reconciles aggregate eligible counts; malformed
    # target-domain fixtures are intentionally left for Step 8 to reject.
    on_time = len(frame) - missed
    metadata = CleaningMetadata(
        source_raw_run_id="raw-fixture", source_raw_parquet_path="data/raw/fixture.parquet",
        source_raw_sha256="raw-hash", validation_report_root="reports/validation",
        validation_raw_run_id="raw-fixture", validation_evidence_status="PASS",
        validation_critical_count=0, cleaning_started_utc="2025-01-01T00:00:00+00:00",
        cleaning_completed_utc="2025-01-01T00:01:00+00:00",
        cleaning_config_hash="clean-config", cleaning_run_id="cleaning-fixture",
        input_row_count=len(frame), cleaned_row_count=len(frame), eligible_row_count=len(frame),
        excluded_row_count=0, removed_exact_duplicate_copies=0,
        conflicting_duplicate_groups=0, timestamp_parse_failures=0,
        due_before_created_exclusions=0, closed_before_created_exclusions=0,
        missing_due_date_exclusions=0, missing_closed_date_exclusions=0,
        on_time_target_count=on_time, missed_target_count=missed,
        missed_target_rate=missed / len(frame), zero_variance_columns=[],
        all_null_columns=[], output_paths={"eligible": str(eligible)},
        output_hashes={"eligible": eligible_hash}, warnings=[],
        completion_status="success", python_version="3.13.7",
        package_versions={}, schema_version="1.0",
    )
    write_cleaning_metadata(metadata, run / "cleaning_metadata.json")
    latest = processed / "latest.json"
    latest.write_text(json.dumps({
        "run_id": "cleaning-fixture", "run_path": str(run),
        "metadata_path": str(run / "cleaning_metadata.json"),
        "updated_utc": "2025-01-01T00:01:00+00:00",
    }), encoding="utf-8")
    return processed, run, eligible, latest


def write_split_config(
    tmp_path: Path, processed: Path, latest: Path, scope: Path,
    scope_metadata: Path,
) -> tuple[Path, Path, Path]:
    """Write a complete local Step 8 configuration with three candidates."""
    output, reports = tmp_path / "splits", tmp_path / "split_reports"
    boundary = {
        "train": {"start": "2024-01-01", "end_exclusive": "2024-04-01"},
        "validation": {"start": "2024-04-01", "end_exclusive": "2024-07-01"},
        "test": {"start": "2024-07-01", "end_exclusive": "2025-01-01"},
    }
    flattened = {
        "train_start": "2024-01-01", "train_end_exclusive": "2024-04-01",
        "validation_start": "2024-04-01", "validation_end_exclusive": "2024-07-01",
        "test_start": "2024-07-01", "test_end_exclusive": "2025-01-01",
    }
    payload = {
        "input": {
            "processed_root": str(processed), "latest_pointer": str(latest),
            "dataset": "eligible_service_requests.parquet",
            "required_completion_status": "success",
            "scope_authority_file": str(scope),
            "scope_extraction_metadata_file": str(scope_metadata),
        },
        "split": {
            "timestamp_column": "created_date", "identifier_column": "unique_key",
            "target_column": "missed_resolution_target",
            "interval_convention": "left_closed_right_open",
        },
        "boundaries": boundary,
        "candidates": [
            {"candidate_id": candidate_id, **flattened}
            for candidate_id in ("A", "B", "C")
        ],
        "selected_candidate": "A",
        "integrity": {
            "require_all_rows_assigned": True, "require_no_date_overlap": True,
            "require_no_identifier_overlap": True, "require_unique_identifiers": True,
            "require_binary_target": True, "require_non_null_target": True,
            "require_both_classes_per_split": True,
            "require_source_immutability": True,
        },
        "minimums": {"rows_per_split": 2, "rows_per_class_per_split": 1},
        "candidate_analysis": {
            "evaluate_multiple_candidates": True,
            "maximum_target_rate_difference": None, "drift_is_warning_only": True,
        },
        "output": {
            "root": str(output), "report_root": str(reports), "format": "parquet",
            "schema_version": "1.0",
        },
    }
    config = tmp_path / "splits.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config, output, reports


def build_split_fixture(tmp_path: Path, frame: pd.DataFrame | None = None) -> SplitFixture:
    """Build a complete isolated Step 7 source and Step 8 configuration."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_frame = make_eligible_frame() if frame is None else frame
    scope, scope_metadata = write_scope(tmp_path)
    processed, run, eligible, latest = write_cleaning_run(tmp_path, source_frame)
    config, output, reports = write_split_config(
        tmp_path, processed, latest, scope, scope_metadata
    )
    return SplitFixture(
        config, processed, latest, run, eligible, output, reports, scope, scope_metadata
    )


@pytest.fixture
def eligible_frame() -> pd.DataFrame:
    """Provide a fresh eligible fixture per test."""
    return make_eligible_frame()


@pytest.fixture
def boundaries() -> SplitBoundaries:
    """Provide the canonical small-fixture boundaries."""
    return fixture_boundaries()


@pytest.fixture
def selected_scope(tmp_path: Path) -> SelectedScope:
    """Provide an in-memory scope equivalent to the fixture authority."""
    return SelectedScope(
        agency="DSNY", complaint_type="Graffiti",
        start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-12-31"),
        authority_extraction_timestamp=pd.Timestamp("2025-01-01", tz="UTC"),
        decision_status="APPROVED_WITH_LIMITATIONS", authority_path=tmp_path / "scope.csv",
        extraction_metadata_path=tmp_path / "scope_metadata.csv",
    )
