"""Construct the nullable missed-resolution target for eligible complaints."""

from __future__ import annotations

import pandas as pd

TARGET_NAME = "missed_resolution_target"
TARGET_DTYPE = "Int8"


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
