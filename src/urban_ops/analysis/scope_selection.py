"""Build and document deterministic NYC 311 modelling-scope evidence.

The functions in this module operate on already acquired records. API access and
notebook presentation remain separate concerns. Target construction deliberately
matches Notebook 03: a record is eligible when creation, closure, and due
timestamps are present; status-only substitutions and timestamp imputation are
not permitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_NAME = "missed_resolution_target"
TARGET_REQUIRED_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "due_date",
    "agency",
    "agency_name",
    "complaint_type",
    "status",
]
SCORE_WEIGHTS = {
    "due_date_score": 0.30,
    "closed_date_score": 0.20,
    "eligible_volume_score": 0.15,
    "target_balance_score": 0.15,
    "data_consistency_score": 0.10,
    "operational_relevance_score": 0.10,
}


def safe_ratio(numerator: pd.Series | float | int, denominator: pd.Series | float | int) -> Any:
    """Divide safely, returning zero for missing or zero denominators."""
    if isinstance(denominator, pd.Series):
        result = pd.Series(0.0, index=denominator.index, dtype="float64")
        valid = denominator.notna() & denominator.ne(0)
        result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
        return result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if denominator is None or pd.isna(denominator) or float(denominator) == 0:
        return 0.0
    result = float(numerator) / float(denominator)
    return result if np.isfinite(result) else 0.0


def validate_scope_input(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate required columns, parse target timestamps, and return diagnostics.

    Raises:
        ValueError: If the input is empty or a required column is absent.
    """
    if frame.empty:
        raise ValueError("Scope-selection input is empty.")
    missing_columns = sorted(set(TARGET_REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Scope-selection input is missing columns: {missing_columns}")

    validated = frame.copy()
    parse_failures: dict[str, int] = {}
    for column in ("created_date", "closed_date", "due_date"):
        raw_present = validated[column].notna() & validated[column].astype("string").str.strip().ne("")
        parsed = pd.to_datetime(validated[column], errors="coerce", utc=True)
        parse_failures[column] = int((raw_present & parsed.isna()).sum())
        validated[column] = parsed

    duplicate_mask = validated["unique_key"].notna() & validated["unique_key"].duplicated(keep=False)
    conflict_columns = [column for column in TARGET_REQUIRED_COLUMNS if column != "unique_key"]
    conflicting_groups = 0
    if duplicate_mask.any():
        conflicting_groups = int(
            validated.loc[duplicate_mask]
            .groupby("unique_key", dropna=False)[conflict_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
            .sum()
        )

    diagnostics = {
        "total_records": len(validated),
        "missing_unique_key_count": int(validated["unique_key"].isna().sum()),
        "duplicate_unique_key_count": int(validated["unique_key"].duplicated().sum()),
        "conflicting_duplicate_unique_key_count": conflicting_groups,
        "created_date_parsing_failure_count": parse_failures["created_date"],
        "closed_date_parsing_failure_count": parse_failures["closed_date"],
        "due_date_parsing_failure_count": parse_failures["due_date"],
        "invalid_due_sequence_count": int(
            (validated["due_date"].notna() & validated["created_date"].notna() & (validated["due_date"] < validated["created_date"])).sum()
        ),
        "invalid_closed_sequence_count": int(
            (validated["closed_date"].notna() & validated["created_date"].notna() & (validated["closed_date"] < validated["created_date"])).sum()
        ),
    }
    return validated, diagnostics


def apply_target_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Notebook 03 eligibility and create a nullable binary target."""
    required_dates = ["created_date", "closed_date", "due_date"]
    missing_columns = sorted(set(required_dates).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Target eligibility is missing columns: {missing_columns}")

    result = frame.copy()
    eligible = result[required_dates].notna().all(axis=1)
    reason = pd.Series("eligible", index=result.index, dtype="string")
    missing_created = result["created_date"].isna()
    missing_closed = result["closed_date"].isna()
    missing_due = result["due_date"].isna()
    reason.loc[~missing_created & ~missing_closed & missing_due] = "has_closed_date_missing_due_date"
    reason.loc[~missing_created & missing_closed & ~missing_due] = "missing_closed_date_only"
    reason.loc[~missing_created & missing_closed & missing_due] = "missing_closed_and_due_date"
    reason.loc[missing_created] = "missing_created_date"

    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    if eligible.any():
        target.loc[eligible] = (
            result.loc[eligible, "closed_date"]
            > result.loc[eligible, "due_date"]
        ).astype("int8")
    result["eligible_for_target"] = eligible
    result["target_exclusion_reason"] = reason
    result[TARGET_NAME] = target
    return result


def _active_month_count(series: pd.Series) -> int:
    """Count distinct UTC calendar months represented by timestamps."""
    timestamps = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    return int(timestamps.dt.tz_localize(None).dt.to_period("M").nunique())


def _count_timestamp_before(
    earlier_candidate: pd.Series,
    reference: pd.Series,
) -> int:
    """Count chronology violations after normalizing both inputs to UTC."""
    candidate = pd.to_datetime(earlier_candidate, errors="coerce", utc=True)
    baseline = pd.to_datetime(reference, errors="coerce", utc=True)
    evaluable = candidate.notna() & baseline.notna()
    if not evaluable.any():
        return 0
    return int((candidate.loc[evaluable] < baseline.loc[evaluable]).sum())


def build_agency_scope_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable agency-level coverage, target, chronology, and volume metrics."""
    required = set(TARGET_REQUIRED_COLUMNS + ["eligible_for_target", TARGET_NAME])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Agency summary is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (agency, agency_name), group in frame.groupby(["agency", "agency_name"], dropna=False, sort=True):
        total = len(group)
        eligible_count = int(group["eligible_for_target"].sum())
        missed_count = int(group[TARGET_NAME].eq(1).sum())
        duplicate_count = int(group["unique_key"].duplicated().sum())
        rows.append(
            {
                "agency": agency,
                "agency_name": agency_name,
                "total_records": total,
                "created_date_present_count": int(group["created_date"].notna().sum()),
                "created_date_coverage": safe_ratio(group["created_date"].notna().sum(), total),
                "due_date_present_count": int(group["due_date"].notna().sum()),
                "due_date_coverage": safe_ratio(group["due_date"].notna().sum(), total),
                "closed_date_present_count": int(group["closed_date"].notna().sum()),
                "closed_date_coverage": safe_ratio(group["closed_date"].notna().sum(), total),
                "eligible_target_count": eligible_count,
                "eligible_target_rate": safe_ratio(eligible_count, total),
                "missed_target_count": missed_count,
                "on_time_count": eligible_count - missed_count,
                "missed_target_rate": safe_ratio(missed_count, eligible_count),
                "first_created_date": group["created_date"].min(),
                "last_created_date": group["created_date"].max(),
                "active_months": _active_month_count(group["created_date"]),
                "complaint_type_count": int(group["complaint_type"].nunique(dropna=True)),
                "duplicate_unique_key_count": duplicate_count,
                "invalid_due_sequence_count": _count_timestamp_before(group["due_date"], group["created_date"]),
                "invalid_closed_sequence_count": _count_timestamp_before(group["closed_date"], group["created_date"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["eligible_target_count", "total_records", "agency"], ascending=[False, False, True]).reset_index(drop=True)


def build_complaint_type_scope_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return complaint-type coverage, target, and monthly-volume metrics."""
    required = set(TARGET_REQUIRED_COLUMNS + ["eligible_for_target", TARGET_NAME])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Complaint-type summary is missing columns: {missing}")
    work = frame.copy()
    work["created_month"] = work["created_date"].dt.tz_localize(None).dt.to_period("M")
    rows: list[dict[str, Any]] = []
    keys = ["agency", "agency_name", "complaint_type"]
    for values, group in work.groupby(keys, dropna=False, sort=True):
        total = len(group)
        eligible_count = int(group["eligible_for_target"].sum())
        missed_count = int(group[TARGET_NAME].eq(1).sum())
        monthly = group.groupby("created_month", observed=True).size()
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "total_records": total,
                "due_date_present_count": int(group["due_date"].notna().sum()),
                "due_date_coverage": safe_ratio(group["due_date"].notna().sum(), total),
                "closed_date_present_count": int(group["closed_date"].notna().sum()),
                "closed_date_coverage": safe_ratio(group["closed_date"].notna().sum(), total),
                "eligible_target_count": eligible_count,
                "eligible_target_rate": safe_ratio(eligible_count, total),
                "missed_target_count": missed_count,
                "on_time_count": eligible_count - missed_count,
                "missed_target_rate": safe_ratio(missed_count, eligible_count),
                "first_created_date": group["created_date"].min(),
                "last_created_date": group["created_date"].max(),
                "active_months": int(monthly.size),
                "median_monthly_volume": float(monthly.median()) if not monthly.empty else 0.0,
                "minimum_monthly_volume": int(monthly.min()) if not monthly.empty else 0,
                "maximum_monthly_volume": int(monthly.max()) if not monthly.empty else 0,
                "invalid_timestamp_count": (
                    _count_timestamp_before(group["due_date"], group["created_date"])
                    + _count_timestamp_before(group["closed_date"], group["created_date"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["eligible_target_count", "total_records", "complaint_type"], ascending=[False, False, True]).reset_index(drop=True)


def calculate_target_balance_score(target_rate: pd.Series | float) -> pd.Series | float:
    """Score binary-target balance from 0 at the extremes to 1 at 50 percent."""
    if isinstance(target_rate, pd.Series):
        return (1.0 - 2.0 * (target_rate.astype("float64") - 0.5).abs()).clip(0.0, 1.0).fillna(0.0)
    if pd.isna(target_rate):
        return 0.0
    return float(np.clip(1.0 - 2.0 * abs(float(target_rate) - 0.5), 0.0, 1.0))


def summarize_scope_date_range(
    frame: pd.DataFrame,
    candidate_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Summarize an inclusive calendar-date candidate, including empty periods."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("Date-range start must not be after its end.")
    created = pd.to_datetime(frame["created_date"], errors="coerce", utc=True).dt.tz_localize(None)
    mask = created.between(start, end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1), inclusive="both")
    scoped = frame.loc[mask].copy()
    scoped_created = created.loc[mask]
    total = len(scoped)
    eligible_count = int(scoped.get("eligible_for_target", pd.Series(False, index=scoped.index)).sum())
    missed_count = int(scoped.get(TARGET_NAME, pd.Series(pd.NA, index=scoped.index)).eq(1).sum())
    monthly_eligible = scoped.assign(_month=scoped_created.dt.to_period("M")).groupby("_month", observed=True)["eligible_for_target"].sum() if total else pd.Series(dtype="int64")
    expected_months = len(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"))
    return {
        "candidate_name": candidate_name,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "total_records": total,
        "eligible_target_count": eligible_count,
        "eligible_target_rate": safe_ratio(eligible_count, total),
        "due_date_coverage": safe_ratio(scoped["due_date"].notna().sum(), total) if "due_date" in scoped else 0.0,
        "closed_date_coverage": safe_ratio(scoped["closed_date"].notna().sum(), total) if "closed_date" in scoped else 0.0,
        "missed_target_count": missed_count,
        "on_time_count": eligible_count - missed_count,
        "missed_target_rate": safe_ratio(missed_count, eligible_count),
        "active_months": int(monthly_eligible.size),
        "median_monthly_eligible_records": float(monthly_eligible.median()) if not monthly_eligible.empty else 0.0,
        "minimum_monthly_eligible_records": int(monthly_eligible.min()) if not monthly_eligible.empty else 0,
        "complaint_type_count": int(scoped["complaint_type"].nunique(dropna=True)) if "complaint_type" in scoped else 0,
        "missing_month_count": max(expected_months - int(monthly_eligible.size), 0),
    }


def summarize_monthly_stability(
    monthly: pd.DataFrame,
    *,
    expected_month_count: int,
    rate_column: str = "missed_target_rate",
    volume_column: str = "eligible_target_count",
    due_date_coverage_column: str = "due_date_coverage",
    closed_date_coverage_column: str = "closed_date_coverage",
    volatility_flag_threshold: float = 0.30,
) -> dict[str, Any]:
    """Measure month-to-month volatility, distinct from monthly continuity.

    A candidate can have zero missing months (continuity) and still show
    unstable target behaviour month to month. ``target_rate_volatility_flag``
    surfaces that separately so a complete monthly sequence is never read as
    evidence of stable target behaviour. The threshold is a configurable
    Month-1 decision rule, not a universal constant.
    """
    volume = monthly[volume_column].astype(float) if volume_column in monthly else pd.Series(dtype="float64")
    rate = monthly[rate_column].astype(float) if rate_column in monthly else pd.Series(dtype="float64")
    due_coverage = monthly[due_date_coverage_column].astype(float) if due_date_coverage_column in monthly else pd.Series(dtype="float64")
    closed_coverage = monthly[closed_date_coverage_column].astype(float) if closed_date_coverage_column in monthly else pd.Series(dtype="float64")

    target_rate_range = float(rate.max() - rate.min()) if not rate.empty else 0.0
    largest_change = float(rate.diff().abs().max()) if len(rate) > 1 else 0.0
    if pd.isna(largest_change):
        largest_change = 0.0

    return {
        "active_months": int(len(monthly)),
        "missing_month_count": max(expected_month_count - len(monthly), 0),
        "monthly_volume_coefficient_of_variation": (
            safe_ratio(volume.std(ddof=0), volume.mean()) if not volume.empty else 0.0
        ),
        "minimum_monthly_eligible_volume": int(volume.min()) if not volume.empty else 0,
        "maximum_monthly_eligible_volume": int(volume.max()) if not volume.empty else 0,
        "target_rate_range": target_rate_range,
        "largest_month_to_month_target_rate_change": largest_change,
        "due_date_coverage_range": (
            float(due_coverage.max() - due_coverage.min()) if not due_coverage.empty else 0.0
        ),
        "closed_date_coverage_range": (
            float(closed_coverage.max() - closed_coverage.min()) if not closed_coverage.empty else 0.0
        ),
        "target_rate_volatility_flag": bool(target_rate_range >= volatility_flag_threshold),
    }


def build_rejection_reasons(row: Mapping[str, Any]) -> str:
    """Build stable, pipe-delimited agency rejection reasons from rule flags."""
    ordered_rules = [
        ("eligible_target_count", lambda value: int(value or 0) == 0, "No target-eligible records"),
        ("passes_eligible_volume", lambda value: not bool(value), "Insufficient eligible records"),
        ("passes_due_date_coverage", lambda value: not bool(value), "Low due-date coverage"),
        ("passes_closed_date_coverage", lambda value: not bool(value), "Low closed-date coverage"),
        ("passes_target_balance", lambda value: not bool(value), "Target rate outside configured range"),
        ("passes_temporal_coverage", lambda value: not bool(value), "Insufficient active months"),
        ("passes_data_consistency", lambda value: not bool(value), "Excessive invalid timestamp sequences"),
        ("passes_recent_activity", lambda value: not bool(value), "No recent activity"),
    ]
    reasons = [label for key, failed, label in ordered_rules if key in row and failed(row[key])]
    return " | ".join(dict.fromkeys(reasons))


def apply_agency_discovery_rules(
    agencies: pd.DataFrame,
    *,
    minimum_eligible_records: int,
    minimum_due_date_coverage: float,
    minimum_closed_date_coverage: float,
    minimum_active_months: int,
    maximum_invalid_timestamp_rate: float,
) -> pd.DataFrame:
    """Separate agency-wide diagnostics from eligibility for deeper review.

    An agency proceeds when it contains sufficient target evidence, temporal
    history, and consistent timestamps. Agency-wide coverage percentages remain
    visible diagnostics but do not veto a viable complaint-type subpopulation.
    """
    result = agencies.copy()
    result["passes_eligible_volume"] = result["eligible_target_count"].ge(
        minimum_eligible_records
    )
    result["passes_due_date_coverage"] = result["due_date_coverage"].ge(
        minimum_due_date_coverage
    )
    result["passes_closed_date_coverage"] = result["closed_date_coverage"].ge(
        minimum_closed_date_coverage
    )
    result["passes_temporal_coverage"] = result["active_months"].ge(
        minimum_active_months
    )
    result["passes_data_consistency"] = result["invalid_timestamp_rate"].le(
        maximum_invalid_timestamp_rate
    )
    result["passes_agency_wide_coverage"] = (
        result["passes_due_date_coverage"]
        & result["passes_closed_date_coverage"]
    )
    # Currently identical to passes_eligible_volume; kept as a distinct named
    # signal because "enough absolute target evidence" and "enough eligible
    # volume" are conceptually separate questions that may diverge later
    # (e.g. a minimum evaluable-pairs rule).
    result["has_target_evidence"] = result["passes_eligible_volume"]
    result["proceed_to_complaint_type_review"] = (
        result["has_target_evidence"]
        & result["passes_temporal_coverage"]
        & result["passes_data_consistency"]
    )
    return result


def apply_complaint_type_decision_rules(
    complaint_types: pd.DataFrame,
    *,
    minimum_eligible_records: int,
    minimum_due_date_coverage: float,
    minimum_closed_date_coverage: float,
    minimum_target_rate: float,
    maximum_target_rate: float,
    minimum_active_months: int,
    minimum_median_monthly_volume: float,
    maximum_invalid_timestamp_rate: float,
) -> pd.DataFrame:
    """Classify complaint types independently from parent-agency percentages."""
    result = complaint_types.copy()
    result["passes_eligible_volume"] = result["eligible_target_count"].ge(
        minimum_eligible_records
    )
    result["passes_due_date_coverage"] = result["due_date_coverage"].ge(
        minimum_due_date_coverage
    )
    result["passes_closed_date_coverage"] = result["closed_date_coverage"].ge(
        minimum_closed_date_coverage
    )
    result["passes_target_balance"] = (
        result["eligible_target_count"].gt(0)
        & result["missed_target_rate"].between(
            minimum_target_rate,
            maximum_target_rate,
        )
    )
    result["passes_temporal_coverage"] = result["active_months"].ge(
        minimum_active_months
    )
    result["passes_monthly_volume"] = result["median_monthly_volume"].ge(
        minimum_median_monthly_volume
    )
    result["passes_recent_activity"] = result["recent_period_record_count"].gt(0)
    result["invalid_timestamp_rate"] = safe_ratio(
        result["invalid_timestamp_count"],
        result["total_records"],
    )
    result["passes_data_consistency"] = result["invalid_timestamp_rate"].le(
        maximum_invalid_timestamp_rate
    )
    rule_columns = [
        "passes_eligible_volume",
        "passes_due_date_coverage",
        "passes_closed_date_coverage",
        "passes_target_balance",
        "passes_temporal_coverage",
        "passes_monthly_volume",
        "passes_recent_activity",
        "passes_data_consistency",
    ]
    result["scope_status"] = np.select(
        [result[rule_columns].all(axis=1), result["eligible_target_count"].gt(0)],
        ["Include", "Review"],
        default="Exclude",
    )
    return result


def evaluate_candidate_critical_gates(
    candidates: pd.DataFrame,
    *,
    minimum_eligible_records: int,
    minimum_due_date_coverage: float,
    minimum_closed_date_coverage: float,
    minimum_target_rate: float,
    maximum_target_rate: float,
    minimum_active_months: int,
    maximum_invalid_timestamp_rate: float,
) -> pd.DataFrame:
    """Evaluate final candidates using only their filtered-population metrics."""
    result = candidates.copy()
    result["passes_eligible_volume"] = result["eligible_target_count"].ge(
        minimum_eligible_records
    )
    result["passes_due_date_coverage"] = result["due_date_coverage"].ge(
        minimum_due_date_coverage
    )
    result["passes_closed_date_coverage"] = result["closed_date_coverage"].ge(
        minimum_closed_date_coverage
    )
    result["passes_target_balance"] = result["missed_target_rate"].between(
        minimum_target_rate,
        maximum_target_rate,
    )
    result["passes_temporal_coverage"] = result["active_months"].ge(
        minimum_active_months
    )
    result["has_included_complaint_types"] = result["complaint_type_count"].gt(0)
    result["passes_month_continuity"] = result["missing_month_count"].eq(0)
    result["passes_data_consistency"] = result["invalid_timestamp_rate"].le(
        maximum_invalid_timestamp_rate
    )
    gate_columns = [
        "passes_eligible_volume",
        "passes_due_date_coverage",
        "passes_closed_date_coverage",
        "passes_target_balance",
        "passes_temporal_coverage",
        "has_included_complaint_types",
        "passes_month_continuity",
        "passes_data_consistency",
    ]
    result["passes_critical_gates"] = result[gate_columns].all(axis=1)
    return result


def build_candidate_rejection_reasons(row: Mapping[str, Any]) -> str:
    """Build stable rejection reasons from candidate-level gate evidence."""
    ordered_rules = [
        ("passes_eligible_volume", "Insufficient candidate eligible records"),
        ("passes_due_date_coverage", "Candidate due-date coverage below threshold"),
        (
            "passes_closed_date_coverage",
            "Candidate closed-date coverage below threshold",
        ),
        ("passes_target_balance", "Candidate target rate outside configured range"),
        ("passes_temporal_coverage", "Insufficient candidate active months"),
        ("has_included_complaint_types", "Candidate contains no included complaint types"),
        ("passes_month_continuity", "Candidate has missing months"),
        (
            "passes_data_consistency",
            "Candidate has an excessive invalid timestamp rate",
        ),
    ]
    reasons = [label for column, label in ordered_rules if not bool(row.get(column, False))]
    return " | ".join(reasons)


def determine_scope_decision_status(
    passes_data_gates: bool,
    *,
    semantic_limitations_remain: bool,
) -> str:
    """Distinguish statistical feasibility from target-governance approval."""
    if not passes_data_gates:
        return "Requires review"
    if semantic_limitations_remain:
        return "Approved with limitations"
    return "Approved"


def validate_selected_scope_consistency(
    selected_scope: Mapping[str, Any],
    selected_complaint_types: Sequence[str],
) -> None:
    """Reject contradictory selected-scope counts and candidate statistics."""
    declared_count = int(selected_scope.get("selected_complaint_type_count", 0))
    if declared_count != len(selected_complaint_types):
        raise ValueError(
            "Selected complaint-type count does not match the selected list."
        )
    if int(selected_scope.get("total_records", 0)) > 0 and declared_count == 0:
        raise ValueError(
            "A non-empty selected scope must contain a selected complaint type."
        )


def rebuild_scoped_sensitivity_candidate(
    monthly: pd.DataFrame,
    *,
    agency: str,
    agency_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    minimum_candidate_eligible_records: int,
    minimum_complaint_type_eligible_records: int,
    minimum_due_date_coverage: float,
    minimum_closed_date_coverage: float,
    minimum_target_rate: float,
    maximum_target_rate: float,
    minimum_active_months: int,
    minimum_median_monthly_volume: float,
    maximum_invalid_timestamp_rate: float,
) -> dict[str, Any]:
    """Rebuild one sensitivity scenario from complaint-type monthly aggregates."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    months = pd.to_datetime(monthly["created_month"], errors="coerce", utc=True)
    scoped = monthly.loc[
        monthly["agency"].eq(agency)
        & months.dt.tz_localize(None).between(
            start.to_period("M").start_time,
            end.to_period("M").end_time,
        )
    ].copy()
    if scoped.empty:
        included_types: list[str] = []
        candidate = {
            "agency": agency,
            "agency_name": agency_name,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "included_complaint_types": "",
            "complaint_type_count": 0,
            "total_records": 0,
            "eligible_target_count": 0,
            "eligible_target_rate": 0.0,
            "due_date_coverage": 0.0,
            "closed_date_coverage": 0.0,
            "missed_target_count": 0,
            "on_time_count": 0,
            "missed_target_rate": 0.0,
            "active_months": 0,
            "missing_month_count": len(
                pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
            ),
            "invalid_timestamp_rate": 0.0,
        }
    else:
        type_keys = ["agency", "agency_name", "complaint_type"]
        type_summary = (
            scoped.groupby(type_keys, dropna=False)
            .agg(
                total_records=("total_records", "sum"),
                due_date_present_count=("due_date_present_count", "sum"),
                closed_date_present_count=("closed_date_present_count", "sum"),
                eligible_target_count=("eligible_target_count", "sum"),
                missed_target_count=("missed_target_count", "sum"),
                invalid_due_sequence_count=("invalid_due_sequence_count", "sum"),
                invalid_closed_sequence_count=("invalid_closed_sequence_count", "sum"),
                active_months=("created_month", "nunique"),
                median_monthly_volume=("total_records", "median"),
            )
            .reset_index()
        )
        recent_cutoff = start.to_period("M").start_time
        if end.to_period("M").ordinal - start.to_period("M").ordinal >= 11:
            recent_cutoff = (
                end.to_period("M").start_time - pd.DateOffset(months=11)
            )
        scoped_months = pd.to_datetime(
            scoped["created_month"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        recent_counts = (
            scoped.loc[scoped_months.ge(recent_cutoff)]
            .groupby(type_keys, dropna=False)["total_records"]
            .sum()
            .rename("recent_period_record_count")
            .reset_index()
        )
        type_summary = type_summary.merge(
            recent_counts,
            on=type_keys,
            how="left",
        ).fillna({"recent_period_record_count": 0})
        type_summary["due_date_coverage"] = safe_ratio(
            type_summary["due_date_present_count"], type_summary["total_records"]
        )
        type_summary["closed_date_coverage"] = safe_ratio(
            type_summary["closed_date_present_count"], type_summary["total_records"]
        )
        type_summary["missed_target_rate"] = safe_ratio(
            type_summary["missed_target_count"],
            type_summary["eligible_target_count"],
        )
        type_summary["invalid_timestamp_count"] = (
            type_summary["invalid_due_sequence_count"]
            + type_summary["invalid_closed_sequence_count"]
        )
        type_summary = apply_complaint_type_decision_rules(
            type_summary,
            minimum_eligible_records=minimum_complaint_type_eligible_records,
            minimum_due_date_coverage=minimum_due_date_coverage,
            minimum_closed_date_coverage=minimum_closed_date_coverage,
            minimum_target_rate=minimum_target_rate,
            maximum_target_rate=maximum_target_rate,
            minimum_active_months=minimum_active_months,
            minimum_median_monthly_volume=minimum_median_monthly_volume,
            maximum_invalid_timestamp_rate=maximum_invalid_timestamp_rate,
        )
        included_types = (
            type_summary.loc[type_summary["scope_status"].eq("Include"), "complaint_type"]
            .astype(str)
            .sort_values()
            .tolist()
        )
        included = scoped.loc[scoped["complaint_type"].astype(str).isin(included_types)]
        total = int(included["total_records"].sum())
        eligible = int(included["eligible_target_count"].sum())
        missed = int(included["missed_target_count"].sum())
        active_months = int(included["created_month"].nunique()) if total else 0
        expected_months = len(
            pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
        )
        invalid_count = int(
            (
                included["invalid_due_sequence_count"]
                + included["invalid_closed_sequence_count"]
            ).sum()
        )
        candidate = {
            "agency": agency,
            "agency_name": agency_name,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "included_complaint_types": " | ".join(included_types),
            "complaint_type_count": len(included_types),
            "total_records": total,
            "eligible_target_count": eligible,
            "eligible_target_rate": safe_ratio(eligible, total),
            "due_date_coverage": safe_ratio(
                included["due_date_present_count"].sum(), total
            ),
            "closed_date_coverage": safe_ratio(
                included["closed_date_present_count"].sum(), total
            ),
            "missed_target_count": missed,
            "on_time_count": eligible - missed,
            "missed_target_rate": safe_ratio(missed, eligible),
            "active_months": active_months,
            "missing_month_count": max(expected_months - active_months, 0),
            "invalid_timestamp_rate": safe_ratio(invalid_count, total),
        }
    evaluated = evaluate_candidate_critical_gates(
        pd.DataFrame([candidate]),
        minimum_eligible_records=minimum_candidate_eligible_records,
        minimum_due_date_coverage=minimum_due_date_coverage,
        minimum_closed_date_coverage=minimum_closed_date_coverage,
        minimum_target_rate=minimum_target_rate,
        maximum_target_rate=maximum_target_rate,
        minimum_active_months=minimum_active_months,
        maximum_invalid_timestamp_rate=maximum_invalid_timestamp_rate,
    ).iloc[0]
    return evaluated.to_dict()


def score_scope_candidates(
    candidates: pd.DataFrame,
    *,
    minimum_eligible_records: int = 10_000,
    operational_relevance_score: float = 0.5,
) -> pd.DataFrame:
    """Add bounded, weighted score components to complete scope candidates."""
    if not 0 <= operational_relevance_score <= 1:
        raise ValueError("Operational relevance score must be between 0 and 1.")
    scored = candidates.copy()
    scored["due_date_score"] = scored["due_date_coverage"].astype(float).clip(0, 1)
    scored["closed_date_score"] = scored["closed_date_coverage"].astype(float).clip(0, 1)
    denominator = np.log1p(max(minimum_eligible_records, 1) * 10)
    scored["eligible_volume_score"] = (np.log1p(scored["eligible_target_count"].clip(lower=0)) / denominator).clip(0, 1)
    scored["target_balance_score"] = calculate_target_balance_score(scored["missed_target_rate"])
    invalid_rate = scored.get("invalid_timestamp_rate", pd.Series(0.0, index=scored.index)).astype(float).clip(0, 1)
    missing_rate = safe_ratio(scored.get("missing_month_count", pd.Series(0, index=scored.index)), scored.get("active_months", pd.Series(0, index=scored.index)) + scored.get("missing_month_count", pd.Series(0, index=scored.index)))
    scored["data_consistency_score"] = (1.0 - 0.7 * invalid_rate - 0.3 * missing_rate).clip(0, 1)
    scored["operational_relevance_score"] = float(operational_relevance_score)
    scored["scope_score"] = sum(scored[column] * weight for column, weight in SCORE_WEIGHTS.items()).clip(0, 1)
    return scored


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional tabulation packages."""
    if frame.empty:
        return "_No rows._"
    display_frame = frame.fillna("").astype(str).map(
        lambda value: value.replace("|", "\\|").replace("\n", " ")
    )
    header = "| " + " | ".join(display_frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display_frame.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display_frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_scope_decision(
    output_path: Path,
    *,
    decision: Mapping[str, Any],
    selected_complaint_types: Sequence[str],
    rejected_candidates: pd.DataFrame,
    extraction: Mapping[str, Any],
    stability: Mapping[str, Any],
    limitations: Sequence[str],
) -> None:
    """Write the required modelling-scope decision document with computed values."""
    status = str(decision.get("decision_status", "Requires review"))
    agency = str(decision.get("selected_agency", "None"))
    agency_name = str(decision.get("selected_agency_name", "None"))
    complaint_text = ", ".join(selected_complaint_types) or "None approved"
    agency_due_date_coverage = float(
        decision.get("agency_wide_due_date_coverage", 0)
    )
    evidence_start = decision.get("complaint_type_evidence_start_date", "the full analysis start date")
    evidence_end = decision.get("complaint_type_evidence_end_date", "the latest complete month")
    complaint_type_evidence_note = (
        f"Complaint-type inclusion evidence covers the full analysis history "
        f"({evidence_start} through {evidence_end}), which is wider than the selected "
        "date range below; the final modelling population is limited further to that "
        "narrower date range."
    )

    target_rate_range = float(stability.get("target_rate_range", 0.0))
    largest_change = float(stability.get("largest_month_to_month_target_rate_change", 0.0))
    volatility_flag = bool(stability.get("target_rate_volatility_flag", False))
    if volatility_flag:
        stability_note = (
            f"The monthly missed-target rate varies substantially across the selected "
            f"window (range {target_rate_range:.2%}, largest single-month change "
            f"{largest_change:.2%}). Zero missing months establishes temporal "
            "continuity only; it does not establish stable target behaviour. A "
            "time-aware validation design and an explicit drift check are required "
            "before modelling, and whether the full date range belongs in one "
            "modelling population should be revisited in light of this variation."
        )
    else:
        stability_note = (
            "Monthly missed-target rate, coverage, and volume remain within a stable "
            "range across the selected window."
        )
    limitations_effective = list(limitations)
    if volatility_flag:
        limitations_effective.append(
            f"Monthly missed-target rate varies by {target_rate_range:.0%} across the "
            "selected window (temporal continuity does not imply temporal stability); "
            "a time-aware validation design and drift check are required before modelling."
        )

    rejected_view = rejected_candidates.head(10).copy()
    percentage_columns = ["due_date_coverage", "closed_date_coverage", "missed_target_rate", "scope_score"]
    for column in percentage_columns:
        if column in rejected_view.columns:
            rejected_view[column] = pd.to_numeric(rejected_view[column], errors="coerce").map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
    content = f"""# Modelling Scope Decision

## Decision Status

**{status}**

## Executive Summary

The API-backed comparison produced status **{status}** for the narrow selected
population. Data-scope feasibility is established separately from final target
governance.

## Business Problem

Predict whether a newly created NYC 311 complaint will miss its expected resolution target.

## Prediction Target

`missed_resolution_target`: 1 when `closed_date > due_date`, otherwise 0, only for target-eligible historical records.

## Target Definition Reference

The eligibility and label rules match `notebooks/03_target_feasibility.ipynb` and `docs/target_definition_draft.md`.

## Data Source

Official NYC Open Data dataset `{extraction.get('dataset_id', 'erm2-nwe9')}` at `{extraction.get('source', '')}`.

## Extraction Summary

- Extraction timestamp: {extraction.get('extraction_timestamp', '')}
- Requested period: {extraction.get('requested_start_date', '')} through {extraction.get('requested_end_date', '')}
- Aggregated records represented: {int(extraction.get('rows_represented', 0)):,}
- Latest available date: {extraction.get('latest_available_date', '')}
- Latest complete month: {extraction.get('latest_complete_month', '')}

## Selection Criteria

Critical rules cover eligible volume, due-date and closure coverage, class balance, temporal continuity, and measurable timestamp consistency. Score weights are 30%, 20%, 15%, 15%, 10%, and 10% respectively; scoring does not override failed critical gates.

## Selected Agency

{agency} — {agency_name}

## Selected Scope

- Agency: {agency}
- Agency name: {agency_name}
- Complaint types: {complaint_text}
- Date range: {decision.get('selected_start_date', 'None')} through {decision.get('selected_end_date', 'None')}

## Selected Date Range

{decision.get('selected_start_date', 'None')} through {decision.get('selected_end_date', 'None')}.

## Selected Complaint Types

{complaint_text}

{complaint_type_evidence_note}

## Why This Scope Was Selected

{agency} has only {agency_due_date_coverage:.2%} agency-wide due-date coverage
because most of its complaint types do not use this field. The selected
complaint-type population is sufficiently large, temporally continuous, and
target-feasible, with strong due-date and closure coverage. The initial model
is therefore complaint-type-specific and is not intended to generalise to all
{agency} service requests.

## Eligible Population

Records with non-null parseable `created_date`, `closed_date`, and `due_date`, matching Notebook 03. Ineligible records retain a null target.

## Target Distribution

- Eligible records: {int(decision.get('eligible_target_records', 0)):,}
- Missed: {int(decision.get('missed_target_count', 0)):,}
- On time: {int(decision.get('on_time_count', 0)):,}
- Missed-target rate: {float(decision.get('missed_target_rate', 0)):.2%}

## Data Quality Summary

- Due-date coverage: {float(decision.get('due_date_coverage', 0)):.2%}
- Closed-date coverage: {float(decision.get('closed_date_coverage', 0)):.2%}
- Eligibility rate: {float(decision.get('eligibility_rate', 0)):.2%}

## Temporal Stability

The current incomplete month was excluded from recommendation and monthly comparisons. The selected candidate covers {int(decision.get('active_months', 0))} active months with {int(decision.get('missing_month_count', 0))} missing months.

- Monthly eligible-volume coefficient of variation: {float(stability.get('monthly_volume_coefficient_of_variation', 0)):.2f}
- Monthly eligible volume range: {int(stability.get('minimum_monthly_eligible_volume', 0)):,} to {int(stability.get('maximum_monthly_eligible_volume', 0)):,}
- Monthly missed-target rate range: {target_rate_range:.2%} (largest single-month change: {largest_change:.2%})
- Monthly due-date coverage range: {float(stability.get('due_date_coverage_range', 0)):.2%}
- Monthly closed-date coverage range: {float(stability.get('closed_date_coverage_range', 0)):.2%}

{stability_note}

## Sensitivity Analysis

The focused threshold analysis rebuilds the included complaint types and final
candidate for each threshold or date-range scenario. Agency-wide due-date
coverage remains diagnostic and does not veto a qualifying scoped population.

## Included Records

{int(decision.get('total_records', 0)):,} records are represented by the selected scope; {int(decision.get('eligible_target_records', 0)):,} can receive the target.

## Exclusion Rules

Rows without any required target timestamp are target-ineligible. The incomplete current month is excluded from stability and final date-range decisions. No status-only proxy target is used.

## Rejected Alternatives

{markdown_table(rejected_view)}

## Known Limitations

{chr(10).join(f'- {item}' for item in limitations_effective)}

## Downstream Implications

Before modelling, retrieve the selected agency, complaint types, and date range
at row level. Validate identifier uniqueness, conflicting duplicates, timestamp
parsing, creation-to-due and creation-to-closure chronology, target
construction, status distribution, cancelled and duplicate treatment,
prediction-time availability of `due_date`, and whether `due_date` changes
after creation. Because this scope contains one complaint type,
`complaint_type` is constant and is not useful as a varying initial-model
feature.

## Reproducibility Information

Generated by `notebooks/04_scope_selection.ipynb` from deterministic server-side API aggregations ordered by stable grouping keys.

## Approval Status

{status}. No model is trained and no train/validation/test split is created in this analysis.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
