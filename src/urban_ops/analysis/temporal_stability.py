"""Evaluate temporal stability and outcome maturity for a modelling scope.

The functions operate on already acquired NYC 311 records. Network access,
report rendering, and scope-specific business narration remain in notebooks.
Target labels continue to require creation, due, and closure timestamps; an
additional maturity rule prevents records with future due dates from entering
scope evidence before their outcome-observation window has elapsed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


APPROVED = "APPROVED"
APPROVED_WITH_LIMITATIONS = "APPROVED_WITH_LIMITATIONS"
REQUIRES_NARROWER_RANGE = "REQUIRES_NARROWER_RANGE"
NOT_FEASIBLE = "NOT_FEASIBLE"

REQUIRED_COLUMNS = {
    "created_date",
    "closed_date",
    "due_date",
}


@dataclass(frozen=True)
class TemporalStabilityThresholds:
    """Named Month-1 rules used to evaluate date-range candidates.

    The values are governance thresholds for this use case, not universal
    statistical constants. They require enough observations for useful binary
    modelling, reasonable field coverage, both outcome classes, two complete
    calendar years, and mature outcomes. Volatility has warning and severe
    levels so continuity cannot produce unconditional approval by itself.
    """

    minimum_eligible_records: int = 10_000
    minimum_due_date_coverage: float = 0.70
    minimum_closed_date_coverage: float = 0.80
    minimum_target_rate: float = 0.05
    maximum_target_rate: float = 0.95
    minimum_active_months: int = 24
    minimum_monthly_eligible_volume: int = 250
    minimum_complete_calendar_years: int = 2
    minimum_outcome_maturity_rate: float = 0.98
    maximum_invalid_timestamp_rate: float = 0.05
    warning_target_rate_range: float = 0.30
    warning_target_rate_standard_deviation: float = 0.10
    warning_month_to_month_change: float = 0.20
    severe_target_rate_range: float = 0.50
    severe_target_rate_standard_deviation: float = 0.15
    severe_month_to_month_change: float = 0.25
    severe_monthly_volume_coefficient_of_variation: float = 0.60
    better_candidate_minimum_eligible_fraction: float = 0.50

    def to_dict(self) -> dict[str, int | float]:
        """Return thresholds in a report-friendly mapping."""
        return asdict(self)


DEFAULT_THRESHOLDS = TemporalStabilityThresholds()


def _ratio(numerator: float | int, denominator: float | int) -> float:
    """Return a finite ratio or zero when the denominator is zero."""
    if not denominator:
        return 0.0
    value = float(numerator) / float(denominator)
    return value if np.isfinite(value) else 0.0


def _as_utc(series: pd.Series) -> pd.Series:
    """Parse a timestamp series as UTC without mutating its caller."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def add_outcome_maturity(
    frame: pd.DataFrame,
    extraction_timestamp: pd.Timestamp | str,
) -> pd.DataFrame:
    """Add target eligibility, maturity, and nullable target columns.

    ``outcome_mature`` means a usable due date is no later than the extraction
    timestamp. Open records remain target-ineligible under the established
    closed-date target contract; they are measured separately rather than
    silently labelled on time or late.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Temporal analysis is missing columns: {missing}")

    result = frame.copy()
    for column in REQUIRED_COLUMNS:
        result[column] = _as_utc(result[column])
    extracted_at = pd.Timestamp(extraction_timestamp)
    if extracted_at.tzinfo is None:
        extracted_at = extracted_at.tz_localize("UTC")
    else:
        extracted_at = extracted_at.tz_convert("UTC")

    result["outcome_mature"] = (
        result["due_date"].notna() & result["due_date"].le(extracted_at)
    )
    result["eligible_for_target"] = result[
        ["created_date", "closed_date", "due_date"]
    ].notna().all(axis=1)
    result["analysis_eligible"] = (
        result["eligible_for_target"] & result["outcome_mature"]
    )
    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    target.loc[result["analysis_eligible"]] = (
        result.loc[result["analysis_eligible"], "closed_date"]
        > result.loc[result["analysis_eligible"], "due_date"]
    ).astype("int8")
    result["missed_resolution_target"] = target
    return result


def build_monthly_scope_metrics(
    frame: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    extraction_timestamp: pd.Timestamp | str,
) -> pd.DataFrame:
    """Build a complete monthly index with coverage, maturity, and target metrics."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("Candidate start date must not be after its end date.")

    prepared = add_outcome_maturity(frame, extraction_timestamp)
    created_naive = prepared["created_date"].dt.tz_localize(None)
    inclusive_end = end + pd.Timedelta(days=1)
    scoped = prepared.loc[
        created_naive.ge(start) & created_naive.lt(inclusive_end)
    ].copy()
    scoped["month"] = scoped["created_date"].dt.tz_localize(None).dt.to_period("M")

    expected = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    rows: list[dict[str, Any]] = []
    for month in expected:
        group = scoped.loc[scoped["month"].eq(month)]
        total = len(group)
        due_count = int(group["due_date"].notna().sum())
        closed_count = int(group["closed_date"].notna().sum())
        mature_count = int(group["outcome_mature"].sum())
        eligible_count = int(group["analysis_eligible"].sum())
        missed_count = int(group["missed_resolution_target"].eq(1).sum())
        rows.append(
            {
                "month": month.to_timestamp(),
                "total_records": total,
                "due_date_present_count": due_count,
                "closed_date_present_count": closed_count,
                "outcome_mature_count": mature_count,
                "outcome_immature_count": due_count - mature_count,
                "mature_open_count": int(
                    (group["outcome_mature"] & group["closed_date"].isna()).sum()
                ),
                "invalid_timestamp_count": int(
                    (
                        group["due_date"].notna()
                        & group["created_date"].notna()
                        & group["due_date"].lt(group["created_date"])
                    ).sum()
                    + (
                        group["closed_date"].notna()
                        & group["created_date"].notna()
                        & group["closed_date"].lt(group["created_date"])
                    ).sum()
                ),
                "eligible_records": eligible_count,
                "eligible_rate": _ratio(eligible_count, total),
                "due_date_coverage": _ratio(due_count, total),
                "closed_date_coverage": _ratio(closed_count, total),
                "outcome_maturity_rate": _ratio(mature_count, due_count),
                "missed_count": missed_count,
                "on_time_count": eligible_count - missed_count,
                "missed_target_rate": (
                    _ratio(missed_count, eligible_count)
                    if eligible_count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_scope_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly evidence into yearly volume, coverage, and target metrics."""
    required = {
        "month",
        "total_records",
        "due_date_present_count",
        "closed_date_present_count",
        "eligible_records",
        "missed_count",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise ValueError(f"Monthly metrics are missing columns: {missing}")
    work = monthly.copy()
    work["year"] = pd.to_datetime(work["month"]).dt.year
    yearly = work.groupby("year", sort=True).agg(
        total_records=("total_records", "sum"),
        yearly_eligible_volume=("eligible_records", "sum"),
        due_date_present_count=("due_date_present_count", "sum"),
        closed_date_present_count=("closed_date_present_count", "sum"),
        missed_count=("missed_count", "sum"),
    ).reset_index()
    yearly["yearly_due_date_coverage"] = yearly.apply(
        lambda row: _ratio(row["due_date_present_count"], row["total_records"]),
        axis=1,
    )
    yearly["yearly_closed_date_coverage"] = yearly.apply(
        lambda row: _ratio(row["closed_date_present_count"], row["total_records"]),
        axis=1,
    )
    yearly["yearly_missed_target_rate"] = yearly.apply(
        lambda row: _ratio(row["missed_count"], row["yearly_eligible_volume"]),
        axis=1,
    )
    return yearly


def calculate_temporal_stability_metrics(monthly: pd.DataFrame) -> dict[str, Any]:
    """Calculate continuity, volume, coverage, and target-rate stability metrics."""
    active = monthly.loc[monthly["eligible_records"].gt(0)].copy()
    volumes = active["eligible_records"].astype(float)
    rates = active["missed_target_rate"].dropna().astype(float)
    total = int(monthly["total_records"].sum())
    due_count = int(monthly["due_date_present_count"].sum())
    closed_count = int(monthly["closed_date_present_count"].sum())
    mature_count = int(monthly["outcome_mature_count"].sum())
    eligible = int(monthly["eligible_records"].sum())
    missed = int(monthly["missed_count"].sum())
    rate_std = float(rates.std(ddof=0)) if not rates.empty else 0.0
    volume_std = float(volumes.std(ddof=0)) if not volumes.empty else 0.0
    return {
        "total_records": total,
        "eligible_records": eligible,
        "eligible_rate": _ratio(eligible, total),
        "due_date_coverage": _ratio(due_count, total),
        "closed_date_coverage": _ratio(closed_count, total),
        "outcome_maturity_rate": _ratio(mature_count, due_count),
        "outcome_immature_count": int(monthly["outcome_immature_count"].sum()),
        "mature_open_count": int(monthly["mature_open_count"].sum()),
        "missed_count": missed,
        "on_time_count": eligible - missed,
        "missed_target_rate": _ratio(missed, eligible),
        "active_months": len(active),
        "expected_months": len(monthly),
        "missing_months": len(monthly) - len(active),
        "minimum_monthly_eligible_volume": int(volumes.min()) if not volumes.empty else 0,
        "maximum_monthly_eligible_volume": int(volumes.max()) if not volumes.empty else 0,
        "mean_monthly_eligible_volume": float(volumes.mean()) if not volumes.empty else 0.0,
        "monthly_volume_standard_deviation": volume_std,
        "monthly_volume_coefficient_of_variation": _ratio(
            volume_std,
            float(volumes.mean()) if not volumes.empty else 0.0,
        ),
        "minimum_monthly_missed_target_rate": float(rates.min()) if not rates.empty else 0.0,
        "maximum_monthly_missed_target_rate": float(rates.max()) if not rates.empty else 0.0,
        "monthly_target_rate_range": (
            float(rates.max() - rates.min()) if not rates.empty else 0.0
        ),
        "monthly_target_rate_standard_deviation": rate_std,
        "largest_absolute_month_to_month_target_rate_change": (
            float(rates.diff().abs().max()) if len(rates) > 1 else 0.0
        ),
    }


def _complete_calendar_years(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count whole January-through-December years inside inclusive boundaries."""
    first_year = start.year if (start.month, start.day) == (1, 1) else start.year + 1
    last_year = end.year if (end.month, end.day) == (12, 31) else end.year - 1
    return max(last_year - first_year + 1, 0)


def evaluate_scope_candidate(
    monthly: pd.DataFrame,
    *,
    candidate_name: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    thresholds: TemporalStabilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate one candidate against feasibility, maturity, and volatility rules."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    metrics = calculate_temporal_stability_metrics(monthly)
    invalid_count = 0
    if "invalid_timestamp_count" in monthly:
        invalid_count = int(monthly["invalid_timestamp_count"].sum())
    invalid_rate = _ratio(invalid_count, metrics["total_records"])
    complete_years = _complete_calendar_years(start, end)

    gates = {
        "passes_eligible_record_count": (
            metrics["eligible_records"] >= thresholds.minimum_eligible_records
        ),
        "passes_due_date_coverage": (
            metrics["due_date_coverage"] >= thresholds.minimum_due_date_coverage
        ),
        "passes_closed_date_coverage": (
            metrics["closed_date_coverage"]
            >= thresholds.minimum_closed_date_coverage
        ),
        "passes_target_class_balance": (
            thresholds.minimum_target_rate
            <= metrics["missed_target_rate"]
            <= thresholds.maximum_target_rate
        ),
        "passes_month_continuity": metrics["missing_months"] == 0,
        "passes_monthly_eligible_volume": (
            metrics["minimum_monthly_eligible_volume"]
            >= thresholds.minimum_monthly_eligible_volume
        ),
        "passes_active_months": metrics["active_months"] >= thresholds.minimum_active_months,
        "passes_outcome_maturity": (
            metrics["outcome_maturity_rate"]
            >= thresholds.minimum_outcome_maturity_rate
        ),
        "passes_timestamp_consistency": invalid_rate <= thresholds.maximum_invalid_timestamp_rate,
        "passes_complete_calendar_years": (
            complete_years >= thresholds.minimum_complete_calendar_years
        ),
    }
    severe = bool(
        metrics["monthly_target_rate_range"] >= thresholds.severe_target_rate_range
        or metrics["monthly_target_rate_standard_deviation"]
        >= thresholds.severe_target_rate_standard_deviation
        or metrics["largest_absolute_month_to_month_target_rate_change"]
        >= thresholds.severe_month_to_month_change
        or metrics["monthly_volume_coefficient_of_variation"]
        >= thresholds.severe_monthly_volume_coefficient_of_variation
    )
    warning = bool(
        severe
        or metrics["monthly_target_rate_range"] >= thresholds.warning_target_rate_range
        or metrics["monthly_target_rate_standard_deviation"]
        >= thresholds.warning_target_rate_standard_deviation
        or metrics["largest_absolute_month_to_month_target_rate_change"]
        >= thresholds.warning_month_to_month_change
    )
    core_feasible = all(gates.values())
    status = NOT_FEASIBLE if not core_feasible else (
        APPROVED_WITH_LIMITATIONS if warning else APPROVED
    )
    failed_gates = [name for name, passed in gates.items() if not passed]
    return {
        "candidate_name": candidate_name,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        **metrics,
        "complete_calendar_years": complete_years,
        "invalid_timestamp_rate": invalid_rate,
        **gates,
        "core_data_feasibility": core_feasible,
        "temporal_volatility_warning": warning,
        "severe_temporal_volatility": severe,
        "decision_status": status,
        "decision_reasons": " | ".join(failed_gates),
    }


def compare_scope_candidates(
    candidates: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    thresholds: TemporalStabilityThresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Compare candidates and reject severe wider ranges when a stable subset exists."""
    result = pd.DataFrame(candidates).copy()
    if result.empty:
        return result
    result["start_date"] = pd.to_datetime(result["start_date"])
    result["end_date"] = pd.to_datetime(result["end_date"])
    result["better_stable_candidate_exists"] = False

    for index, row in result.iterrows():
        if not bool(row["severe_temporal_volatility"]):
            continue
        contained = result.loc[
            result["core_data_feasibility"].astype(bool)
            & ~result["severe_temporal_volatility"].astype(bool)
            & result["start_date"].ge(row["start_date"])
            & result["end_date"].le(row["end_date"])
            & result["eligible_records"].ge(
                float(row["eligible_records"])
                * thresholds.better_candidate_minimum_eligible_fraction
            )
        ]
        if not contained.empty:
            result.at[index, "better_stable_candidate_exists"] = True
            result.at[index, "decision_status"] = REQUIRES_NARROWER_RANGE
            result.at[index, "decision_reasons"] = (
                "Severe temporal volatility and a feasible, more stable contained candidate exists"
            )

    # A transparent selection score balances stability, recent complete-period
    # relevance, and usable volume. It ranks only after feasibility gates.
    maximum_volume = max(float(result["eligible_records"].max()), 1.0)
    latest_complete_end = result.loc[
        result["end_date"].dt.month.eq(12) & result["end_date"].dt.day.eq(31),
        "end_date",
    ].max()
    if pd.isna(latest_complete_end):
        latest_complete_end = result["end_date"].max()
    earliest_start = result["start_date"].min()
    latest_start = result["start_date"].max()
    start_span_days = max((latest_start - earliest_start).days, 1)
    result["recent_period_relevance_score"] = (
        (result["start_date"] - earliest_start).dt.days / start_span_days
    )
    result["selection_score"] = (
        0.30 * (1.0 - result["monthly_target_rate_range"].clip(0, 1))
        + 0.20 * (1.0 - result["monthly_target_rate_standard_deviation"].clip(0, 1))
        + 0.15 * (1.0 - result["monthly_volume_coefficient_of_variation"].clip(0, 1))
        + 0.15 * (result["eligible_records"] / maximum_volume)
        + 0.15 * result["end_date"].eq(latest_complete_end).astype(float)
        + 0.05 * result["recent_period_relevance_score"]
    )
    result["selected"] = False
    selectable = result.loc[
        result["core_data_feasibility"].astype(bool)
        & ~result["severe_temporal_volatility"].astype(bool)
        & result["end_date"].eq(latest_complete_end)
    ]
    if selectable.empty:
        selectable = result.loc[result["core_data_feasibility"].astype(bool)]
    if not selectable.empty:
        selected_index = selectable.sort_values(
            ["selection_score", "eligible_records"], ascending=False
        ).index[0]
        result.at[selected_index, "selected"] = True
    result["start_date"] = result["start_date"].dt.date.astype(str)
    result["end_date"] = result["end_date"].dt.date.astype(str)
    return result.sort_values(
        ["selected", "selection_score", "eligible_records"], ascending=False
    ).reset_index(drop=True)


def evaluate_candidate_periods(
    frame: pd.DataFrame,
    periods: Iterable[tuple[str, str, str]],
    *,
    extraction_timestamp: pd.Timestamp | str,
    thresholds: TemporalStabilityThresholds = DEFAULT_THRESHOLDS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build monthly/yearly evidence and compare all configured candidate periods."""
    rows: list[dict[str, Any]] = []
    monthly_outputs: dict[str, pd.DataFrame] = {}
    yearly_outputs: dict[str, pd.DataFrame] = {}
    for name, start, end in periods:
        monthly = build_monthly_scope_metrics(
            frame,
            start,
            end,
            extraction_timestamp,
        )
        yearly = build_yearly_scope_metrics(monthly)
        rows.append(
            evaluate_scope_candidate(
                monthly,
                candidate_name=name,
                start_date=start,
                end_date=end,
                thresholds=thresholds,
            )
        )
        monthly_outputs[name] = monthly
        yearly_outputs[name] = yearly
    return compare_scope_candidates(rows, thresholds=thresholds), monthly_outputs, yearly_outputs
