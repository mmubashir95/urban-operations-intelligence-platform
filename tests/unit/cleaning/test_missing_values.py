"""Tests for role-aware missing-value preservation and quality flags."""

import pandas as pd

from urban_ops.cleaning.missing_values import apply_missing_value_policy


def test_target_inputs_and_scope_values_are_never_imputed(raw_frame) -> None:
    frame = raw_frame.copy()
    for column in ("unique_key", "created_date", "agency", "complaint_type", "due_date", "closed_date"):
        frame.loc[0, column] = None
    original = frame.copy(deep=True)
    result, actions, _ = apply_missing_value_policy(frame)
    assert result.loc[0, list(original.columns[:1])].isna().all()
    for column in ("created_date", "agency", "complaint_type", "due_date", "closed_date"):
        assert pd.isna(result.loc[0, column])
    assert actions["imputed_count"].eq(0).all()
    pd.testing.assert_frame_equal(frame, original)


def test_unknown_unspecified_and_missing_geography_are_preserved(raw_frame) -> None:
    frame = raw_frame.iloc[[0]].copy()
    frame.loc[0, "open_data_channel_type"] = "UNKNOWN"
    frame.loc[0, "borough"] = "Unspecified"
    frame.loc[0, ["latitude", "longitude"]] = None
    result, _, _ = apply_missing_value_policy(frame)
    assert result.loc[0, "open_data_channel_type"] == "UNKNOWN"
    assert result.loc[0, "borough"] == "Unspecified"
    assert pd.isna(result.loc[0, "latitude"]) and pd.isna(result.loc[0, "longitude"])
    assert not result.loc[0, "has_coordinates"]


def test_zip_remains_string_and_preserves_leading_zero(raw_frame) -> None:
    result, _, _ = apply_missing_value_policy(raw_frame.iloc[[0]])
    assert str(result["incident_zip"].dtype) == "string"
    assert result.loc[0, "incident_zip"] == "01234"


def test_quality_flags_are_audit_fields(raw_frame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "descriptor"] = None
    result, _, _ = apply_missing_value_policy(frame)
    assert {"has_coordinates", "coordinates_valid", "has_borough", "has_descriptor"}.issubset(result)
    assert not result.loc[0, "has_descriptor"]
    assert result.loc[1, "coordinates_valid"]
