"""Tests for missing, empty, whitespace-only, and null-like profiling."""

import pandas as pd

from urban_ops.validation.missingness import profile_missingness


NULL_LIKE = ("", "null", "none", "nan", "n/a", "na", "unknown")


def test_missingness_counts_distinct_value_classes() -> None:
    frame = pd.DataFrame({"descriptor": [None, "", "   ", "NULL", "real"]})
    result = profile_missingness(frame, null_like_strings=NULL_LIKE).iloc[0]
    assert result.null_count == 1
    assert result.empty_string_count == 1
    assert result.whitespace_only_count == 1
    assert result.null_like_string_count == 1
    assert result.null_rate == 0.2


def test_null_like_strings_are_not_converted() -> None:
    frame = pd.DataFrame({"descriptor": ["none", "value"]})
    original = frame.copy(deep=True)
    profile_missingness(frame, null_like_strings=NULL_LIKE)
    pd.testing.assert_frame_equal(frame, original)


def test_target_input_missingness_is_material() -> None:
    result = profile_missingness(
        pd.DataFrame({"due_date": [None], "closed_date": [None]}),
        null_like_strings=NULL_LIKE,
    ).set_index("column_name")
    assert result.loc["due_date", "missingness_severity"] == "ERROR"
    assert result.loc["closed_date", "missingness_severity"] == "WARNING"


def test_optional_feature_missingness_is_warning() -> None:
    result = profile_missingness(
        pd.DataFrame({"latitude": [None]}), null_like_strings=NULL_LIKE
    ).iloc[0]
    assert result.column_role == "CONDITIONAL_FEATURE"
    assert result.missingness_severity == "WARNING"


def test_all_null_and_no_null_columns_reconcile() -> None:
    result = profile_missingness(
        pd.DataFrame({"descriptor_2": [None, None], "agency": ["DSNY", "DSNY"]}),
        null_like_strings=NULL_LIKE,
    ).set_index("column_name")
    assert result.loc["descriptor_2", "null_rate"] == 1.0
    assert result.loc["agency", "null_rate"] == 0.0


def test_critical_identifier_missingness() -> None:
    result = profile_missingness(
        pd.DataFrame({"unique_key": [None]}), null_like_strings=NULL_LIKE
    ).iloc[0]
    assert result.column_role == "IDENTIFIER"
    assert result.missingness_severity == "CRITICAL"


def test_unknown_column_uses_documented_policy() -> None:
    result = profile_missingness(
        pd.DataFrame({"future_new_field": [None]}), null_like_strings=NULL_LIKE
    ).iloc[0]
    assert result.column_role == "UNKNOWN"
    assert result.missingness_severity == "INFO"
