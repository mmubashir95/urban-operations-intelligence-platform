"""Tests for timestamp parseability, timezone evidence, and raw preservation."""

import pandas as pd

from urban_ops.validation.timestamps import analyze_timestamps


def test_mixed_timestamp_formats_parse_consistently() -> None:
    frame = pd.DataFrame({
        "unique_key": ["utc", "offset", "naive", "invalid", "null"],
        "created_date": [
            "2024-01-01T00:00:00Z", "2024-01-01T01:00:00+01:00",
            "2024-01-02 00:00:00", "not-a-date", None,
        ],
    })
    original = frame.copy(deep=True)
    result = analyze_timestamps(frame, ["created_date"])
    row = result.summary.iloc[0]
    assert row.non_null_count == 4
    assert row.parse_success_count == 3
    assert row.parse_failure_count == 1
    assert row.parse_success_rate == 0.75
    assert row.timezone_aware_count == 2
    assert row.timezone_naive_count == 1
    assert row.minimum_timestamp == "2024-01-01T00:00:00+00:00"
    assert row.maximum_timestamp == "2024-01-02T00:00:00+00:00"
    assert result.invalid_examples.iloc[0].unique_key == "invalid"
    pd.testing.assert_frame_equal(frame, original)


def test_empty_timestamp_column_is_valid_profile() -> None:
    result = analyze_timestamps(
        pd.DataFrame({"unique_key": ["1"], "due_date": [None]}), ["due_date"]
    ).summary.iloc[0]
    assert result.non_null_count == result.parse_failure_count == 0
    assert result.parse_success_rate == 1.0
    assert pd.isna(result.minimum_timestamp)


def test_invalid_examples_are_bounded_and_identified() -> None:
    frame = pd.DataFrame({
        "unique_key": [str(i) for i in range(20)],
        "closed_date": ["invalid"] * 20,
    })
    result = analyze_timestamps(frame, ["closed_date"])
    assert len(result.invalid_examples) == 10
    assert result.invalid_examples["unique_key"].tolist() == [str(i) for i in range(10)]


def test_missing_configured_column_is_skipped() -> None:
    result = analyze_timestamps(pd.DataFrame({"unique_key": ["1"]}), ["due_date"])
    assert result.summary.empty and result.parsed == {}
