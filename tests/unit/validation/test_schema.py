"""Tests for raw structure and semantic type profiling."""

from dataclasses import replace

import pandas as pd
import pytest

from urban_ops.validation.schema import validate_schema


def validate(frame: object, metadata: object, required: list[str]) -> tuple:
    return validate_schema(
        frame, metadata=metadata, required_columns=required, require_column_order=True
    )


def test_valid_schema_passes_without_mutation(raw_frame, extraction_metadata) -> None:
    original = raw_frame.copy(deep=True)
    schema, profile, checks = validate(raw_frame, extraction_metadata, list(raw_frame.columns))
    assert len(schema) == len(profile) == 18
    assert all(check.status.value == "PASS" for check in checks)
    pd.testing.assert_frame_equal(raw_frame, original)


def test_missing_required_column_fails(raw_frame, extraction_metadata) -> None:
    frame = raw_frame.drop(columns="status")
    _, _, checks = validate(frame, extraction_metadata, list(raw_frame.columns))
    check = next(item for item in checks if item.check_id == "schema.required_columns")
    assert check.status.value == "FAIL" and "status" in str(check.observed_value)


def test_duplicate_column_names_fail(raw_frame, extraction_metadata) -> None:
    frame = raw_frame.copy()
    frame.columns = [*frame.columns[:-1], "unique_key"]
    _, _, checks = validate(frame, extraction_metadata, list(raw_frame.columns))
    assert next(c for c in checks if c.check_id == "schema.unique_column_names").status.value == "FAIL"


def test_unexpected_column_is_reported(raw_frame, extraction_metadata) -> None:
    frame = raw_frame.assign(new_source_field="x")
    _, _, checks = validate(frame, extraction_metadata, list(raw_frame.columns))
    check = next(c for c in checks if c.check_id == "schema.unexpected_columns")
    assert check.status.value == "WARN" and check.affected_rows == 1


def test_metadata_row_count_mismatch_is_critical(raw_frame, extraction_metadata) -> None:
    metadata = replace(extraction_metadata, retrieved_row_count=3, page_row_counts=[3])
    _, _, checks = validate(raw_frame, metadata, list(raw_frame.columns))
    check = next(c for c in checks if c.check_id == "schema.metadata_row_count")
    assert check.severity.value == "CRITICAL" and check.status.value == "FAIL"


def test_metadata_column_mismatch_fails(raw_frame, extraction_metadata) -> None:
    metadata = replace(extraction_metadata, selected_source_columns=["unique_key"])
    _, _, checks = validate(raw_frame, metadata, list(raw_frame.columns))
    assert next(c for c in checks if c.check_id == "schema.metadata_columns").status.value == "FAIL"


def test_empty_dataset_fails(raw_frame, extraction_metadata) -> None:
    frame = raw_frame.iloc[0:0]
    metadata = replace(extraction_metadata, retrieved_row_count=0, page_row_counts=[0])
    _, _, checks = validate(frame, metadata, list(frame.columns))
    assert next(c for c in checks if c.check_id == "schema.non_empty").status.value == "FAIL"


def test_unsupported_object_type_fails(extraction_metadata) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate([], extraction_metadata, ["unique_key"])


def test_sparse_nullable_column_is_accepted(raw_frame, extraction_metadata) -> None:
    _, profile, checks = validate(raw_frame, extraction_metadata, list(raw_frame.columns))
    assert profile.set_index("column_name").loc["descriptor_2", "null_count"] == 2
    assert next(c for c in checks if c.check_id == "schema.required_columns").status.value == "PASS"


def test_column_order_policy_is_enforced(raw_frame, extraction_metadata) -> None:
    frame = raw_frame[list(reversed(raw_frame.columns))]
    _, _, checks = validate(frame, extraction_metadata, list(raw_frame.columns))
    assert next(c for c in checks if c.check_id == "schema.column_order").status.value == "FAIL"


def test_numeric_identifier_representation_is_rejected(raw_frame, extraction_metadata) -> None:
    frame = raw_frame.copy()
    frame["unique_key"] = [1, 2]
    _, _, checks = validate(frame, extraction_metadata, list(raw_frame.columns))
    check = next(c for c in checks if c.check_id == "schema.identifier_representation")
    assert check.status.value == "FAIL" and check.severity.value == "ERROR"
