"""Evaluate governed eligibility for the missed-resolution target.

Rows are classified without discarding or mutating source data. Timestamps are
compared in UTC and the selected end date is an inclusive calendar boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd


REQUIRED_COLUMNS: Final = frozenset(
    {"unique_key", "created_date", "due_date", "closed_date", "agency",
     "complaint_type", "status"}
)
DEFAULT_ALLOWED_STATUSES: Final = frozenset({"closed"})
DEFAULT_EXCLUDED_STATUSES: Final = frozenset(
    {"assigned", "cancelled", "duplicate", "open", "pending", "started"}
)
DUPLICATE_RELEVANT_COLUMNS: Final = (
    "created_date", "due_date", "closed_date", "agency", "complaint_type", "status"
)
EXCLUSION_PRIORITY: Final = (
    "outside_selected_scope", "missing_created_date", "missing_due_date",
    "missing_closed_date", "due_before_created", "closed_before_created",
    "excluded_status", "conflicting_duplicate_unique_key",
    "exact_duplicate_unique_key", "outcome_not_mature", "eligible",
)


def _boundary(value: object, name: str) -> pd.Timestamp:
    """Validate and normalize a calendar boundary."""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid date.") from error
    if pd.isna(result):
        raise ValueError(f"{name} must be a valid date.")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def _extraction_timestamp(value: object) -> pd.Timestamp:
    """Validate and normalize the outcome observation timestamp to UTC."""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("extraction_timestamp must be a valid timestamp.") from error
    if pd.isna(result):
        raise ValueError("extraction_timestamp must be a valid timestamp.")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _statuses(values: Iterable[str]) -> set[str]:
    """Normalize configured status values."""
    return {str(value).strip().casefold() for value in values}


def _duplicate_flags(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return redundant-exact and conflicting-group flags."""
    exact = pd.Series(False, index=frame.index)
    conflict = pd.Series(False, index=frame.index)
    members = frame["unique_key"].notna() & frame["unique_key"].duplicated(keep=False)
    if not members.any():
        return exact, conflict
    signatures = (
        frame.loc[members, list(DUPLICATE_RELEVANT_COLUMNS)]
        .astype("string").fillna("<NA>").agg("\x1f".join, axis=1)
    )
    work = pd.DataFrame({
        "key": frame.loc[members, "unique_key"].astype("string"),
        "signature": signatures,
        "stable_index": [f"{type(i).__name__}:{i!s}" for i in signatures.index],
    }, index=signatures.index)
    conflicting = work.groupby("key")["signature"].transform("nunique").gt(1)
    conflict.loc[work.index[conflicting]] = True
    candidates = work.loc[~conflicting].sort_values(
        ["key", "signature", "stable_index"], kind="stable"
    )
    redundant = candidates.duplicated(["key", "signature"], keep="first")
    exact.loc[candidates.index[redundant]] = True
    return exact, conflict


def evaluate_target_eligibility(
    df: pd.DataFrame,
    *,
    selected_agency: str,
    selected_complaint_type: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    extraction_timestamp: pd.Timestamp | str,
    allowed_statuses: set[str] | None = None,
    excluded_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Return a copy with deterministic eligibility and exclusion columns.

    Raises:
        ValueError: If inputs, boundaries, or status policies are invalid.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Target eligibility is missing required columns: {missing}")
    if not str(selected_agency).strip() or not str(selected_complaint_type).strip():
        raise ValueError("Selected agency and complaint type must be present.")
    start, end = _boundary(start_date, "start_date"), _boundary(end_date, "end_date")
    if start >= end:
        raise ValueError("start_date must be before end_date.")
    extracted_at = _extraction_timestamp(extraction_timestamp)
    allowed = _statuses(allowed_statuses or DEFAULT_ALLOWED_STATUSES)
    excluded = _statuses(excluded_statuses or DEFAULT_EXCLUDED_STATUSES)
    if allowed & excluded:
        raise ValueError("Statuses cannot be both allowed and excluded.")

    result = df.copy(deep=True)
    for column in ("created_date", "due_date", "closed_date"):
        result[column] = pd.to_datetime(
            result[column], errors="coerce", utc=True, format="mixed"
        )
    created_naive = result["created_date"].dt.tz_localize(None)
    agency_match = result["agency"].astype("string").eq(str(selected_agency))
    type_match = result["complaint_type"].astype("string").eq(str(selected_complaint_type))
    date_match = created_naive.ge(start) & created_naive.lt(end + pd.Timedelta(days=1))
    result["within_selected_scope"] = agency_match & type_match & date_match
    for column, flag in (
        ("created_date", "has_created_date"), ("due_date", "has_due_date"),
        ("closed_date", "has_closed_date"),
    ):
        result[flag] = result[column].notna()
    result["valid_due_chronology"] = (
        ~result["has_due_date"] | ~result["has_created_date"]
        | result["due_date"].ge(result["created_date"])
    )
    result["valid_closed_chronology"] = (
        ~result["has_closed_date"] | ~result["has_created_date"]
        | result["closed_date"].ge(result["created_date"])
    )
    status = result["status"].astype("string").str.strip().str.casefold()
    unrecognized = sorted(set(status.dropna().unique()) - (allowed | excluded))
    if unrecognized:
        raise ValueError(
            "Unrecognized status values require explicit governance "
            f"(add to allowed_statuses or excluded_statuses): {unrecognized}"
        )
    result["status_normalized"] = status
    result["status_allowed"] = status.isin(allowed)
    result["is_cancelled"] = status.eq("cancelled").fillna(False)
    # Broad "not verifiably closed" flag: true for any non-closed status (including
    # e.g. pending) and for a closed status inconsistently missing a closed_date.
    # Not the same as literal status == "open"; see status_normalized for that.
    result["is_open"] = ~result["has_closed_date"] | ~status.eq("closed").fillna(False)
    result["status_closed_date_inconsistent"] = (
        status.eq("closed").fillna(False) ^ result["has_closed_date"]
    )
    result["outcome_mature"] = (
        result["has_due_date"] & result["due_date"].le(extracted_at)
    )
    exact, conflict = _duplicate_flags(result)
    result["is_exact_duplicate"] = exact
    result["is_conflicting_duplicate"] = conflict
    result["target_eligible"] = (
        result["within_selected_scope"] & result["has_created_date"]
        & result["has_due_date"] & result["has_closed_date"]
        & result["valid_due_chronology"] & result["valid_closed_chronology"]
        & result["status_allowed"] & ~exact & ~conflict & result["outcome_mature"]
    )

    outside = ~result["within_selected_scope"] & (
        ~agency_match | ~type_match | result["has_created_date"]
    )
    condition_by_reason = {
        "outside_selected_scope": outside,
        "missing_created_date": ~result["has_created_date"],
        "missing_due_date": ~result["has_due_date"],
        "missing_closed_date": ~result["has_closed_date"],
        "due_before_created": ~result["valid_due_chronology"],
        "closed_before_created": ~result["valid_closed_chronology"],
        "excluded_status": ~result["status_allowed"],
        "conflicting_duplicate_unique_key": conflict,
        "exact_duplicate_unique_key": exact,
        "outcome_not_mature": ~result["outcome_mature"],
    }
    if set(condition_by_reason) != set(EXCLUSION_PRIORITY) - {"eligible"}:
        raise RuntimeError("EXCLUSION_PRIORITY and computed conditions have diverged.")
    reasons = pd.Series("eligible", index=result.index, dtype="string")
    available = pd.Series(True, index=result.index)
    # EXCLUSION_PRIORITY is the single source of truth for precedence order; the
    # mapping above only supplies each reason's condition, never its rank.
    for reason in EXCLUSION_PRIORITY:
        if reason == "eligible":
            continue
        selected = available & condition_by_reason[reason].fillna(False)
        reasons.loc[selected] = reason
        available.loc[selected] = False
    result["primary_exclusion_reason"] = reasons
    if (result["target_eligible"] ^ reasons.eq("eligible")).any():
        raise RuntimeError("Eligibility flags and exclusion reasons are inconsistent.")
    eligible_ids = result.loc[result["target_eligible"], "unique_key"]
    if eligible_ids.isna().any() or eligible_ids.duplicated().any():
        raise RuntimeError("Final eligible records must have unique complaint IDs.")
    return result
