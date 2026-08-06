"""Tests for Step 7 source discovery and fail-closed eligible verification."""

import json
from pathlib import Path

import pandas as pd
import pytest

from urban_ops.splitting.source import (
    SplitSourceError, load_verified_source, resolve_cleaning_run,
    validate_eligible_frame,
)
from tests.unit.splitting.conftest import build_split_fixture


def test_latest_and_explicit_cleaning_runs_resolve(tmp_path: Path, eligible_frame) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    assert resolve_cleaning_run(
        processed_root=fixture.processed_root, latest_pointer=fixture.latest_pointer
    ) == fixture.cleaning_run.resolve()
    assert resolve_cleaning_run(
        processed_root=fixture.processed_root, latest_pointer=tmp_path / "missing",
        override=fixture.cleaning_run,
    ) == fixture.cleaning_run.resolve()


def test_missing_or_invalid_latest_pointer_fails(tmp_path: Path) -> None:
    with pytest.raises(SplitSourceError, match="missing"):
        resolve_cleaning_run(processed_root=tmp_path, latest_pointer=tmp_path / "latest.json")
    pointer = tmp_path / "latest.json"
    pointer.write_text("{}")
    with pytest.raises(SplitSourceError, match="invalid"):
        resolve_cleaning_run(processed_root=tmp_path, latest_pointer=pointer)


def test_missing_run_metadata_or_eligible_file_fails(
    tmp_path: Path, eligible_frame, selected_scope
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    (fixture.cleaning_run / "cleaning_metadata.json").unlink()
    with pytest.raises(SplitSourceError, match="metadata is missing"):
        load_verified_source(
            run_path=fixture.cleaning_run, dataset_name="eligible_service_requests.parquet",
            required_completion_status="success", scope=selected_scope,
            timestamp_column="created_date", identifier_column="unique_key",
            target_column="missed_resolution_target",
        )


def test_hash_and_row_count_mismatch_fail(tmp_path: Path, eligible_frame, selected_scope) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    metadata_path = fixture.cleaning_run / "cleaning_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["output_hashes"]["eligible"] = "wrong"
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(SplitSourceError, match="SHA-256"):
        load_verified_source(
            run_path=fixture.cleaning_run, dataset_name=fixture.eligible.name,
            required_completion_status="success", scope=selected_scope,
            timestamp_column="created_date", identifier_column="unique_key",
            target_column="missed_resolution_target",
        )


@pytest.mark.parametrize("mutation, message", [
    (lambda frame: frame.assign(missed_resolution_target=pd.NA), "null"),
    (lambda frame: frame.assign(missed_resolution_target=2), "binary"),
    (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "unique"),
    (lambda frame: frame.assign(target_eligible=False), "target-ineligible"),
])
def test_invalid_target_identifier_or_eligibility_fails(
    mutation, message, eligible_frame, selected_scope
) -> None:
    changed = mutation(eligible_frame.copy())
    with pytest.raises(SplitSourceError, match=message):
        validate_eligible_frame(
            changed, scope=selected_scope, timestamp_column="created_date",
            identifier_column="unique_key", target_column="missed_resolution_target",
        )


def test_missing_or_naive_created_date_and_scope_mismatch_fail(
    eligible_frame, selected_scope
) -> None:
    with pytest.raises(SplitSourceError, match="required"):
        validate_eligible_frame(
            eligible_frame.drop(columns="created_date"), scope=selected_scope,
            timestamp_column="created_date", identifier_column="unique_key",
            target_column="missed_resolution_target",
        )
    naive = eligible_frame.copy()
    naive["created_date"] = naive["created_date"].dt.tz_localize(None)
    with pytest.raises(SplitSourceError, match="timezone-aware"):
        validate_eligible_frame(
            naive, scope=selected_scope, timestamp_column="created_date",
            identifier_column="unique_key", target_column="missed_resolution_target",
        )
    wrong = eligible_frame.assign(agency="DOT")
    with pytest.raises(SplitSourceError, match="agency"):
        validate_eligible_frame(
            wrong, scope=selected_scope, timestamp_column="created_date",
            identifier_column="unique_key", target_column="missed_resolution_target",
        )


def test_source_validation_does_not_mutate_input(eligible_frame, selected_scope) -> None:
    before = eligible_frame.copy(deep=True)
    validate_eligible_frame(
        eligible_frame, scope=selected_scope, timestamp_column="created_date",
        identifier_column="unique_key", target_column="missed_resolution_target",
    )
    pd.testing.assert_frame_equal(eligible_frame, before)
