"""Tests for comparison-only category profiling and status governance."""

import pandas as pd

from urban_ops.validation.categories import profile_categories, validate_statuses
from urban_ops.validation.timestamps import analyze_timestamps


def test_formatting_and_null_like_values_are_profiled_without_mutation() -> None:
    frame = pd.DataFrame({
        "status": ["Closed", " closed", "CLOSED ", "CLOSED  ", "", "   ", "unknown", None]
    })
    original = frame.copy(deep=True)
    profile, variants = profile_categories(
        frame, columns=["status"], null_like_strings=["", "unknown"],
        rare_category_max_rows=1, high_cardinality_threshold=3,
        maximum_profile_values_per_column=100,
    )
    indexed = profile.set_index("raw_value")
    assert bool(indexed.loc[" closed", "has_leading_whitespace"])
    assert bool(indexed.loc["CLOSED ", "has_trailing_whitespace"])
    assert bool(indexed.loc["", "is_empty"])
    assert bool(indexed.loc["   ", "is_whitespace_only"])
    assert bool(indexed.loc["unknown", "is_null_like_string"])
    assert {"", "closed"}.issubset(set(variants["normalised_comparison_value"]))
    pd.testing.assert_frame_equal(frame, original)


def test_case_only_variants_are_grouped() -> None:
    _, variants = profile_categories(
        pd.DataFrame({"borough": ["Queens", "QUEENS"]}), columns=["borough"],
        null_like_strings=[], rare_category_max_rows=1, high_cardinality_threshold=10,
        maximum_profile_values_per_column=10,
    )
    assert variants.iloc[0].variant_type == "case"


def test_profile_is_bounded_for_high_cardinality() -> None:
    frame = pd.DataFrame({"descriptor": [f"value-{i}" for i in range(20)]})
    profile, _ = profile_categories(
        frame, columns=["descriptor"], null_like_strings=[], rare_category_max_rows=1,
        high_cardinality_threshold=5, maximum_profile_values_per_column=4,
    )
    assert len(profile) == 4 and profile["is_high_cardinality"].all()


def test_empty_categorical_column_has_stable_schema() -> None:
    profile, variants = profile_categories(
        pd.DataFrame({"descriptor": pd.Series(dtype="string")}), columns=["descriptor"],
        null_like_strings=[], rare_category_max_rows=1, high_cardinality_threshold=10,
        maximum_profile_values_per_column=10,
    )
    assert profile.empty and variants.empty
    assert "normalised_comparison_value" in profile.columns


def test_status_policy_classification_and_unexpected_value(raw_frame) -> None:
    frame = raw_frame.iloc[:2].copy()
    frame.loc[0, "status"] = "Closed"
    frame.loc[1, "status"] = "Novel Status"
    timestamps = analyze_timestamps(
        frame, ["created_date", "due_date", "closed_date"]
    )
    result = validate_statuses(frame, parsed_timestamps=timestamps.parsed).set_index("status_raw")
    assert result.loc["Closed", "step_4_policy"] == "ALLOWED"
    assert result.loc["Novel Status", "step_4_policy"] == "UNEXPECTED"
    assert not bool(result.loc["Novel Status", "is_expected"])


def test_excluded_status_is_expected(raw_frame) -> None:
    frame = raw_frame.iloc[[0]].assign(status="Open", closed_date=None)
    timestamps = analyze_timestamps(frame, ["created_date", "due_date", "closed_date"])
    result = validate_statuses(frame, parsed_timestamps=timestamps.parsed).iloc[0]
    assert result.step_4_policy == "EXCLUDED" and bool(result.is_expected)
