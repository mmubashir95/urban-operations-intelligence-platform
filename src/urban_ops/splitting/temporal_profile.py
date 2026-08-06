"""Build complete calendar-month volume and target profiles from creation time."""

from __future__ import annotations

import pandas as pd


MONTHLY_COLUMNS = [
    "month", "row_count", "on_time_count", "missed_count", "missed_target_rate",
    "cumulative_row_count", "cumulative_row_share", "minimum_created_date",
    "maximum_created_date", "has_both_classes",
]


def build_monthly_profile(
    frame: pd.DataFrame, *, timestamp_column: str, target_column: str
) -> pd.DataFrame:
    """Return a chronological month profile, explicitly including zero-row months."""
    if frame.empty:
        raise ValueError("Cannot profile an empty eligible dataset.")
    timestamps = frame[timestamp_column]
    if timestamps.isna().any() or not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Monthly profiling requires non-null timezone-aware timestamps.")
    local = pd.DataFrame({
        "timestamp": timestamps,
        "target": frame[target_column].astype(int),
    })
    local["month"] = local["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    months = pd.period_range(local["month"].min(), local["month"].max(), freq="M")
    grouped = local.groupby("month", observed=False).agg(
        row_count=("target", "size"),
        missed_count=("target", "sum"),
        minimum_created_date=("timestamp", "min"),
        maximum_created_date=("timestamp", "max"),
    ).reindex(months)
    grouped["row_count"] = grouped["row_count"].fillna(0).astype(int)
    grouped["missed_count"] = grouped["missed_count"].fillna(0).astype(int)
    grouped["on_time_count"] = grouped["row_count"] - grouped["missed_count"]
    grouped["missed_target_rate"] = grouped["missed_count"].div(
        grouped["row_count"].replace(0, pd.NA)
    )
    grouped["cumulative_row_count"] = grouped["row_count"].cumsum()
    grouped["cumulative_row_share"] = grouped["cumulative_row_count"] / len(frame)
    grouped["has_both_classes"] = (
        grouped["on_time_count"].gt(0) & grouped["missed_count"].gt(0)
    )
    grouped.index.name = "month"
    result = grouped.reset_index()
    result["month"] = result["month"].astype(str)
    return result[MONTHLY_COLUMNS]
