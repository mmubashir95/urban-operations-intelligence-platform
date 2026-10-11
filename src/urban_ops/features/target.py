"""Construct classification and resolution-time targets for NYC 311 complaints.

Both targets are outcome-derived and must remain outside creation-time model
features. The regression builder is intentionally independent of the existing
missed-resolution classification contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_NAME = "missed_resolution_target"
TARGET_DTYPE = "Int8"
RESOLUTION_TIME_TARGET_NAME = "resolution_duration_hours"
RESOLUTION_TIME_ELIGIBILITY_NAME = "resolution_time_eligible"
RESOLUTION_TIME_REASON_NAME = "resolution_time_exclusion_reason"
RESOLUTION_TIME_CATEGORY_NAME = "resolution_time_exclusion_category"

RESOLUTION_TIME_EXCLUSION_PRIORITY = (
    "missing_created_date",
    "invalid_created_date",
    "missing_closed_date",
    "invalid_closed_date",
    "invalid_timestamp_order",
    "cancelled_complaint",
    "open_complaint",
    "ambiguous_completion_status",
    "resolution_not_observed_by_cutoff",
    "invalid_calculated_duration",
)


def build_missed_resolution_target(
    df: pd.DataFrame, *, eligibility_column: str = "target_eligible"
) -> pd.Series:
    """Return 1 for late closure, 0 for on/before due, and NA when ineligible."""
    required = {eligibility_column, "closed_date", "due_date"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Target construction is missing required columns: {missing}")
    eligible = df[eligibility_column].fillna(False)
    if not eligible.isin([True, False]).all():
        raise ValueError(f"{eligibility_column} must contain only boolean values.")
    eligible = eligible.astype(bool)
    closed = pd.to_datetime(
        df["closed_date"], errors="coerce", utc=True, format="mixed"
    )
    due = pd.to_datetime(df["due_date"], errors="coerce", utc=True, format="mixed")
    invalid = eligible & (closed.isna() | due.isna())
    if invalid.any():
        raise ValueError(
            "Eligible rows require parseable non-null closed_date and due_date; "
            f"invalid row indices: {invalid.index[invalid].tolist()[:10]}"
        )
    target = pd.Series(pd.NA, index=df.index, dtype=TARGET_DTYPE, name=TARGET_NAME)
    target.loc[eligible] = closed.loc[eligible].gt(due.loc[eligible]).astype("int8")
    if not set(target.dropna().astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("Target contains values outside the binary contract.")
    return target


def _observation_cutoff(value: object) -> pd.Timestamp:
    """Return a validated UTC observation cutoff."""
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("observation_cutoff must be a valid timestamp.") from error
    if pd.isna(cutoff):
        raise ValueError("observation_cutoff must be a valid timestamp.")
    return (
        cutoff.tz_localize("UTC")
        if cutoff.tzinfo is None
        else cutoff.tz_convert("UTC")
    )


def build_resolution_time_target(
    frame: pd.DataFrame,
    *,
    observation_cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return aligned resolution-time labels and deterministic eligibility.

    A label is eligible only for a verifiably closed complaint with parseable
    UTC timestamps, nonnegative finite duration, and a closure observable at
    ``observation_cutoff`` when supplied. Missing/invalid records are retained
    with one primary reason; the input frame is never mutated.

    A naive cutoff is interpreted as UTC, matching the cleaning timestamp
    convention. Closure exactly at the cutoff is observable.
    """
    required = {"created_date", "closed_date", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Resolution-time target construction is missing required columns: {missing}"
        )

    created_raw = frame["created_date"]
    closed_raw = frame["closed_date"]
    created = pd.to_datetime(created_raw, errors="coerce", utc=True, format="mixed")
    closed = pd.to_datetime(closed_raw, errors="coerce", utc=True, format="mixed")
    status = frame["status"].astype("string").str.strip().str.casefold()
    duration = (closed - created).dt.total_seconds() / 3600.0
    finite_duration = pd.Series(
        np.isfinite(duration.to_numpy(dtype="float64", na_value=np.nan)),
        index=frame.index,
    )

    conditions: dict[str, pd.Series] = {
        "missing_created_date": created_raw.isna(),
        "invalid_created_date": created_raw.notna() & created.isna(),
        "missing_closed_date": closed_raw.isna(),
        "invalid_closed_date": closed_raw.notna() & closed.isna(),
        "invalid_timestamp_order": (
            created.notna() & closed.notna() & closed.lt(created)
        ),
        "cancelled_complaint": status.eq("cancelled").fillna(False),
        "open_complaint": status.eq("open").fillna(False),
        "ambiguous_completion_status": ~status.eq("closed").fillna(False)
        & ~status.isin({"cancelled", "open"}),
        "resolution_not_observed_by_cutoff": pd.Series(False, index=frame.index),
        "invalid_calculated_duration": duration.notna()
        & (~finite_duration | duration.lt(0)),
    }
    if observation_cutoff is not None:
        cutoff = _observation_cutoff(observation_cutoff)
        conditions["resolution_not_observed_by_cutoff"] = (
            closed.notna() & closed.gt(cutoff)
        )

    reasons = pd.Series("eligible", index=frame.index, dtype="string")
    available = pd.Series(True, index=frame.index)
    for reason in RESOLUTION_TIME_EXCLUSION_PRIORITY:
        selected = available & conditions[reason].fillna(False)
        reasons.loc[selected] = reason
        available.loc[selected] = False

    eligible = reasons.eq("eligible")
    target = pd.Series(
        pd.NA, index=frame.index, dtype="Float64", name=RESOLUTION_TIME_TARGET_NAME
    )
    target.loc[eligible] = duration.loc[eligible].astype("float64")
    categories = pd.Series("invalid", index=frame.index, dtype="string")
    categories.loc[eligible] = "eligible"
    categories.loc[
        reasons.isin(
            {
                "missing_closed_date",
                "open_complaint",
                "resolution_not_observed_by_cutoff",
            }
        )
    ] = "not_observed"

    result = pd.DataFrame(index=frame.index)
    result[RESOLUTION_TIME_ELIGIBILITY_NAME] = eligible.astype(bool)
    result[RESOLUTION_TIME_TARGET_NAME] = target
    result[RESOLUTION_TIME_REASON_NAME] = reasons
    result[RESOLUTION_TIME_CATEGORY_NAME] = categories
    return result
