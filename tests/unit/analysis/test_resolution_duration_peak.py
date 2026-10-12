"""Focused synthetic tests for the Phase 7.2 900-hour investigation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_ops.analysis.resolution_duration_peak import (
    build_complaint_type_analysis,
    build_data_quality_checks,
    build_interval_comparison,
    build_peak_summary,
    build_timing_patterns,
    select_peak_interval,
)


def _eligible() -> pd.DataFrame:
    """Return boundary-rich eligible durations with stable timing patterns."""
    durations = [650.0, 749.0, 750.0, 849.0, 850.0, 900.0, 950.0, 950.5, 1050.0, 1149.0]
    created = pd.to_datetime(["2024-01-01T00:00:00Z"] * len(durations), utc=True)
    closed = created + pd.to_timedelta(durations, unit="h")
    closed = pd.Series(closed)
    closed.iloc[5] = closed.iloc[4]  # Synthetic repeated timestamp evidence.
    return pd.DataFrame({
        "unique_key": [f"id-{index}" for index in range(len(durations))],
        "complaint_type": ["A", "A", "A", "B", "A", "A", "B", "B", "C", "small"],
        "created_date": created,
        "closed_date": closed,
        "resolution_duration_hours": durations,
    })


def test_primary_interval_is_inclusive_and_summary_is_correct() -> None:
    eligible = _eligible()
    peak = select_peak_interval(eligible)
    summary = dict(build_peak_summary(eligible, peak).itertuples(index=False, name=None))
    assert peak["resolution_duration_hours"].tolist() == [850.0, 900.0, 950.0]
    assert summary["peak_complaint_count"] == 3
    assert summary["peak_percentage_of_eligible"] == pytest.approx(30.0)
    assert summary["peak_minimum_hours"] == 850.0
    assert summary["peak_median_hours"] == 900.0
    assert summary["peak_maximum_hours"] == 950.0
    assert summary["peak_unique_identifier_count"] == 3


def test_neighbor_intervals_are_half_open_without_double_counting() -> None:
    intervals = build_interval_comparison(_eligible())
    neighbors = intervals.loc[intervals["analysis_type"].eq("neighbor_interval")]
    assert neighbors["complaint_count"].tolist() == [2, 2, 2, 2, 2]
    assert neighbors["complaint_count"].sum() == 10
    assert neighbors["percentage_of_eligible"].tolist() == [20.0] * 5
    primary = intervals.loc[
        intervals["analysis_type"].eq("primary_inclusive_interval")
    ].iloc[0]
    assert primary["complaint_count"] == 3
    sensitivity = intervals.loc[intervals["analysis_type"].eq("histogram_sensitivity")]
    assert set(sensitivity["bin_width_hours"]) == {10.0, 25.0, 50.0}


def test_complaint_type_shares_and_representation_are_distinct() -> None:
    eligible = _eligible()
    result = build_complaint_type_analysis(eligible, select_peak_interval(eligible))
    indexed = result.set_index("complaint_type")
    assert indexed.loc["A", "eligible_count"] == 5
    assert indexed.loc["A", "peak_count"] == 2
    assert indexed.loc["A", "percentage_of_peak"] == pytest.approx(200 / 3)
    assert indexed.loc["A", "within_type_peak_percentage"] == 40.0
    assert indexed.loc["A", "representation_ratio"] == pytest.approx(4 / 3)
    assert indexed.loc["small", "eligible_count"] == 1
    assert indexed.loc["small", "peak_count"] == 0


def test_timing_patterns_use_utc_and_find_repeated_closures() -> None:
    peak = select_peak_interval(_eligible())
    patterns = build_timing_patterns(peak)
    repeated = patterns.loc[
        patterns["pattern_type"].eq("identical_closed_timestamp")
    ]
    assert len(repeated) == 1
    assert repeated.iloc[0]["count"] == 2
    hours = patterns.loc[patterns["pattern_type"].eq("closed_hour_utc"), "pattern_value"]
    assert set(hours) == {"10", "14"}
    day_bands = patterns.loc[patterns["pattern_type"].eq("duration_day_band")]
    assert int(day_bands["count"].sum()) == len(peak)


def test_data_quality_detects_duplicates_invalid_values_and_recalculation() -> None:
    eligible = _eligible()
    eligible.loc[1, "unique_key"] = eligible.loc[0, "unique_key"]
    eligible.loc[2, "resolution_duration_hours"] = -1.0
    eligible.loc[3, "resolution_duration_hours"] = np.inf
    checks = build_data_quality_checks(eligible).set_index("check")["value"]
    assert checks["duplicate_identifier_rows"] == 2
    assert checks["negative_duration_rows"] == 1
    assert checks["non_finite_duration_rows"] == 1
    assert checks["maximum_identical_closure_timestamp_repetition"] == 2


def test_analysis_functions_do_not_mutate_input_and_are_deterministic() -> None:
    eligible = _eligible()
    original = eligible.copy(deep=True)
    first = build_interval_comparison(eligible)
    second = build_interval_comparison(eligible)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(eligible, original)
