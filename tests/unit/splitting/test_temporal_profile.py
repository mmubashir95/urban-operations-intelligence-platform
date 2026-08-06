"""Tests for complete calendar-month target profiling."""

import pandas as pd
import pytest

from urban_ops.splitting.temporal_profile import build_monthly_profile


def test_monthly_counts_rates_cumulative_values_and_order(eligible_frame) -> None:
    profile = build_monthly_profile(
        eligible_frame, timestamp_column="created_date",
        target_column="missed_resolution_target",
    )
    assert profile["month"].tolist() == sorted(profile["month"])
    assert profile["row_count"].eq(2).all()
    assert profile["on_time_count"].eq(1).all()
    assert profile["missed_count"].eq(1).all()
    assert profile["missed_target_rate"].eq(0.5).all()
    assert profile["cumulative_row_count"].iloc[-1] == len(eligible_frame)
    assert profile["cumulative_row_share"].iloc[-1] == 1.0
    assert profile["has_both_classes"].all()


def test_missing_month_and_one_class_month_are_explicit(eligible_frame) -> None:
    frame = eligible_frame.loc[~eligible_frame["created_date"].dt.month.eq(5)].copy()
    frame = frame.loc[~(
        frame["created_date"].dt.month.eq(6)
        & frame["missed_resolution_target"].eq(1)
    )]
    profile = build_monthly_profile(frame, timestamp_column="created_date", target_column="missed_resolution_target").set_index("month")
    assert profile.loc["2024-05", "row_count"] == 0
    assert pd.isna(profile.loc["2024-05", "missed_target_rate"])
    assert not bool(profile.loc["2024-06", "has_both_classes"])


def test_minimum_maximum_timestamps_and_input_immutability(eligible_frame) -> None:
    before = eligible_frame.copy(deep=True)
    profile = build_monthly_profile(eligible_frame, timestamp_column="created_date", target_column="missed_resolution_target")
    assert profile["minimum_created_date"].min() == eligible_frame["created_date"].min()
    assert profile["maximum_created_date"].max() == eligible_frame["created_date"].max()
    pd.testing.assert_frame_equal(eligible_frame, before)


def test_empty_profile_is_rejected(eligible_frame) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_monthly_profile(eligible_frame.iloc[0:0], timestamp_column="created_date", target_column="missed_resolution_target")
