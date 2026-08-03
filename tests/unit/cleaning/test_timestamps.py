"""Tests for UTC timestamp cleaning without guessing or raw mutation."""

import pandas as pd

from urban_ops.cleaning.timestamps import clean_timestamps


def test_valid_naive_utc_and_offset_values_normalize_to_utc() -> None:
    frame = pd.DataFrame({
        "created_date": [
            "2024-01-01T00:00:00Z", "2024-01-01 00:00:00",
            "2024-01-01T01:00:00+01:00",
        ]
    })
    result, summary, _ = clean_timestamps(frame, columns=["created_date"])
    assert isinstance(result["created_date"].dtype, pd.DatetimeTZDtype)
    assert str(result["created_date"].dt.tz) == "UTC"
    assert result["created_date"].nunique() == 1
    assert summary.iloc[0].parsed_count == 3


def test_null_remains_null_and_invalid_is_flagged_not_guessed() -> None:
    frame = pd.DataFrame({"due_date": [None, "invalid"]})
    original = frame.copy(deep=True)
    result, summary, _ = clean_timestamps(frame, columns=["due_date"])
    assert result["due_date"].isna().all()
    assert result["due_date_parse_failed"].tolist() == [False, True]
    assert summary.iloc[0].null_count == 1
    assert summary.iloc[0].parse_failure_count == 1
    assert summary.iloc[0].imputed_count == 0
    pd.testing.assert_frame_equal(frame, original)


def test_chronology_is_not_repaired() -> None:
    frame = pd.DataFrame({
        "created_date": ["2024-02-01"], "due_date": ["2024-01-01"],
        "closed_date": ["2023-12-01"],
    })
    result, _, _ = clean_timestamps(
        frame, columns=["created_date", "due_date", "closed_date"]
    )
    assert result.loc[0, "due_date"] < result.loc[0, "created_date"]
    assert result.loc[0, "closed_date"] < result.loc[0, "created_date"]


def test_summary_counts_reconcile() -> None:
    frame = pd.DataFrame({"closed_date": ["2024-01-01", None, "bad"]})
    _, summary, actions = clean_timestamps(frame, columns=["closed_date"])
    row = summary.iloc[0]
    assert row.input_non_null_count == row.parsed_count + row.parse_failure_count
    assert len(actions) == 1 and actions[0].action_type == "PARSE_UTC"


def test_missing_configured_timestamp_fails() -> None:
    try:
        clean_timestamps(pd.DataFrame({"created_date": []}), columns=["due_date"])
    except ValueError as error:
        assert "due_date" in str(error)
    else:
        raise AssertionError("Missing timestamp column should fail")
