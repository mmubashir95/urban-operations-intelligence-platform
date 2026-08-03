"""Tests for cleaning metadata provenance, reconciliation, and serialization."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from urban_ops.cleaning.metadata import (
    CleaningMetadata, read_cleaning_metadata, write_cleaning_metadata,
)


def metadata() -> CleaningMetadata:
    """Return a fully reconciled successful metadata fixture."""
    return CleaningMetadata(
        source_raw_run_id="raw-1", source_raw_parquet_path="data/raw/file.parquet",
        source_raw_sha256="a" * 64, validation_report_root="reports/08_data_validation",
        validation_raw_run_id="raw-1", validation_evidence_status="ERROR",
        validation_critical_count=0, cleaning_started_utc="2026-01-01T00:00:00+00:00",
        cleaning_completed_utc="2026-01-01T00:01:00+00:00",
        cleaning_config_hash="b" * 64, cleaning_run_id="clean-1",
        input_row_count=10, cleaned_row_count=9, eligible_row_count=7,
        excluded_row_count=2, removed_exact_duplicate_copies=1,
        conflicting_duplicate_groups=0, timestamp_parse_failures=0,
        due_before_created_exclusions=0, closed_before_created_exclusions=1,
        missing_due_date_exclusions=1, missing_closed_date_exclusions=1,
        on_time_target_count=4, missed_target_count=3, missed_target_rate=3 / 7,
        zero_variance_columns=["open_data_channel_type"],
        all_null_columns=["descriptor_2"],
        output_paths={
            "cleaned": "data/processed/cleaned.parquet",
            "eligible": "data/processed/eligible.parquet",
            "excluded": "data/processed/excluded.parquet",
            "metadata": "data/processed/metadata.json",
            "rules_snapshot": "data/processed/rules.yaml",
        },
        output_hashes={"cleaned": "c" * 64}, warnings=[], completion_status="success",
        python_version="3.13.7", package_versions={"pandas": "2"}, schema_version="1.0",
    )


def test_metadata_records_lineage_counts_and_hashes() -> None:
    value = metadata()
    value.validate()
    assert value.source_raw_run_id == value.validation_raw_run_id
    assert value.source_raw_sha256 and value.cleaning_config_hash
    assert value.output_paths and value.output_hashes
    assert value.input_row_count == value.cleaned_row_count + value.removed_exact_duplicate_copies
    assert value.eligible_row_count == value.on_time_target_count + value.missed_target_count


@pytest.mark.parametrize("changes", [
    {"validation_critical_count": 1},
    {"cleaned_row_count": 8},
    {"eligible_row_count": 6},
    {"completion_status": "failure"},
    {"cleaning_started_utc": "2026-01-01"},
    {"output_hashes": {}},
])
def test_invalid_metadata_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(metadata(), **changes).validate()


def test_json_round_trip_is_stable_and_secret_free(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_cleaning_metadata(metadata(), path)
    assert read_cleaning_metadata(path) == metadata()
    text = path.read_text().casefold()
    assert "authorization" not in text and "x-app-token" not in text


def test_unknown_secret_field_is_rejected(tmp_path: Path) -> None:
    payload = metadata().to_dict()
    payload["token"] = "secret"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read"):
        read_cleaning_metadata(path)
