"""Tests for explicitly configured category transformations only."""

import pandas as pd

from urban_ops.cleaning.categories import clean_categories


def clean(frame: pd.DataFrame, **overrides: object):
    options = {
        "trim_columns": ["descriptor", "status", "borough", "open_data_channel_type"],
        "blank_to_null_columns": ["descriptor"],
        "collapse_repeated_spaces_columns": [],
        "approved_mappings": {},
    }
    options.update(overrides)
    return clean_categories(frame, **options)


def test_approved_trim_and_blank_to_null_are_audited() -> None:
    frame = pd.DataFrame({
        "descriptor": ["  value  ", "   "], "status": [" Closed ", "Closed"],
        "borough": ["Unspecified", "QUEENS"],
        "open_data_channel_type": ["UNKNOWN", "UNKNOWN"],
    })
    original = frame.copy(deep=True)
    result, audit, _ = clean(frame)
    assert result["descriptor"].tolist()[0] == "value"
    assert pd.isna(result.loc[1, "descriptor"])
    assert set(audit["mapping_type"]) == {"trim", "blank_to_null"}
    pd.testing.assert_frame_equal(frame, original)


def test_unknown_channel_and_unspecified_borough_remain_explicit() -> None:
    frame = pd.DataFrame({
        "descriptor": ["x"], "status": ["Closed"], "borough": ["Unspecified"],
        "open_data_channel_type": ["UNKNOWN"],
    })
    result, audit, _ = clean(frame)
    assert result.loc[0, "open_data_channel_type"] == "UNKNOWN"
    assert result.loc[0, "borough"] == "Unspecified"
    assert audit.empty


def test_repeated_spaces_change_only_when_approved() -> None:
    frame = pd.DataFrame({
        "descriptor": ["many   spaces"], "status": ["Closed"], "borough": ["QUEENS"],
        "open_data_channel_type": ["UNKNOWN"],
    })
    unchanged, _, _ = clean(frame)
    changed, audit, _ = clean(frame, collapse_repeated_spaces_columns=["descriptor"])
    assert unchanged.loc[0, "descriptor"] == "many   spaces"
    assert changed.loc[0, "descriptor"] == "many spaces"
    assert "collapse_repeated_spaces" in set(audit["mapping_type"])


def test_case_mapping_requires_explicit_approval() -> None:
    frame = pd.DataFrame({
        "descriptor": ["x"], "status": ["closed"], "borough": ["QUEENS"],
        "open_data_channel_type": ["UNKNOWN"],
    })
    unchanged, _, _ = clean(frame)
    mapped, audit, _ = clean(frame, approved_mappings={"status": {"closed": "Closed"}})
    assert unchanged.loc[0, "status"] == "closed"
    assert mapped.loc[0, "status"] == "Closed"
    assert audit.iloc[-1].mapping_type == "approved_mapping"


def test_blank_to_null_only_applies_to_configured_fields() -> None:
    frame = pd.DataFrame({
        "descriptor": [""], "status": [""], "borough": ["QUEENS"],
        "open_data_channel_type": ["UNKNOWN"],
    })
    result, _, _ = clean(frame)
    assert pd.isna(result.loc[0, "descriptor"])
    assert result.loc[0, "status"] == ""


def test_category_cleaning_is_deterministic() -> None:
    frame = pd.DataFrame({
        "descriptor": [" x ", " y "], "status": ["Closed", "Closed"],
        "borough": ["QUEENS", "QUEENS"], "open_data_channel_type": ["UNKNOWN", "UNKNOWN"],
    })
    first, first_audit, _ = clean(frame)
    second, second_audit, _ = clean(frame)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_audit, second_audit)
