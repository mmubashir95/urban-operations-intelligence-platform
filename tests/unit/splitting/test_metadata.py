"""Tests for split metadata reconciliation, serialization, paths, and safety."""

from dataclasses import replace
from pathlib import Path

import pytest

from urban_ops.splitting.metadata import (
    SplitMetadata, read_split_metadata, write_split_metadata,
)
from urban_ops.splitting.models import SplitRunPaths


def metadata(*, paths: SplitRunPaths | None = None) -> SplitMetadata:
    """Return one fully reconciled successful metadata fixture."""
    path_map = ({
        "train": "data/splits/train.parquet", "validation": "data/splits/validation.parquet",
        "test": "data/splits/test.parquet", "metadata": "data/splits/split_metadata.json",
        "rules_snapshot": "data/splits/split_rules_snapshot.yaml",
    } if paths is None else {
        "train": str(paths.train), "validation": str(paths.validation),
        "test": str(paths.test), "metadata": str(paths.metadata),
        "rules_snapshot": str(paths.rules_snapshot),
    })
    return SplitMetadata(
        schema_version="1.0", split_id="split-fixture", completion_status="success",
        created_at_utc="2026-01-01T00:00:00+00:00",
        source_cleaning_run_id="cleaning-fixture",
        source_cleaning_metadata_path="data/processed/metadata.json",
        source_eligible_dataset_path="data/processed/eligible.parquet",
        source_eligible_sha256="eligible-hash", source_eligible_mtime=123,
        source_raw_run_id="raw-fixture", source_raw_sha256="raw-hash",
        scope_agency="DSNY", scope_complaint_type="Graffiti",
        scope_start_date="2024-01-01", scope_end_date="2024-12-31",
        timestamp_column="created_date", identifier_column="unique_key",
        target_column="missed_resolution_target",
        interval_convention="left_closed_right_open",
        train_start_inclusive="2024-01-01T00:00:00+00:00",
        train_end_exclusive="2024-04-01T00:00:00+00:00",
        validation_start_inclusive="2024-04-01T00:00:00+00:00",
        validation_end_exclusive="2024-07-01T00:00:00+00:00",
        test_start_inclusive="2024-07-01T00:00:00+00:00",
        test_end_exclusive="2025-01-01T00:00:00+00:00",
        input_row_count=18, train_row_count=6, validation_row_count=6,
        test_row_count=6, train_on_time_count=3, train_missed_count=3,
        validation_on_time_count=3, validation_missed_count=3,
        test_on_time_count=3, test_missed_count=3,
        train_missed_target_rate=0.5, validation_missed_target_rate=0.5,
        test_missed_target_rate=0.5,
        train_min_created_date="2024-01-01T12:00:00+00:00",
        train_max_created_date="2024-03-02T12:00:00+00:00",
        validation_min_created_date="2024-04-01T12:00:00+00:00",
        validation_max_created_date="2024-06-02T12:00:00+00:00",
        test_min_created_date="2024-07-01T12:00:00+00:00",
        test_max_created_date="2024-09-02T12:00:00+00:00",
        train_month_count=3, validation_month_count=3, test_month_count=3,
        unassigned_row_count=0, multiply_assigned_row_count=0,
        train_validation_identifier_overlap=0, train_test_identifier_overlap=0,
        validation_test_identifier_overlap=0, config_hash="config-hash",
        output_paths=path_map,
        output_hashes={"train": "a", "validation": "b", "test": "c", "rules_snapshot": "d"},
        warnings=[], python_version="3.13.7", package_versions={"pandas": "test"},
    )


def test_metadata_records_provenance_scope_boundaries_counts_rates_and_utc() -> None:
    value = metadata()
    value.validate()
    assert value.source_cleaning_run_id == "cleaning-fixture"
    assert value.source_raw_sha256 == "raw-hash"
    assert value.scope_agency == "DSNY"
    assert value.input_row_count == value.train_row_count + value.validation_row_count + value.test_row_count
    assert value.train_on_time_count + value.train_missed_count == value.train_row_count
    assert value.interval_convention == "left_closed_right_open"


def test_metadata_json_round_trip_and_no_secret_keys(tmp_path: Path) -> None:
    path = tmp_path / "split_metadata.json"
    write_split_metadata(metadata(), path)
    assert read_split_metadata(path) == metadata()
    assert "token" not in path.read_text().casefold()


@pytest.mark.parametrize("changed, message", [
    ({"train_row_count": 7}, "Input rows"),
    ({"train_missed_target_rate": 0.2}, "target rate"),
    ({"unassigned_row_count": 1}, "assignment failures"),
    ({"completion_status": "failure"}, "completion_status"),
])
def test_metadata_rejects_inconsistent_or_incomplete_success(changed, message) -> None:
    with pytest.raises(ValueError, match=message):
        replace(metadata(), **changed).validate()
