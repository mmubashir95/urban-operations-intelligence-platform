"""Focused evidence for the Phase 7.2 concentration near 900 hours.

The functions operate on the existing historical population and its aligned
Phase 7.1 targets. They describe the concentration without changing records,
target values, or eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from urban_ops.analysis.resolution_duration import DurationAnalysisPopulation
from urban_ops.features.target import (
    RESOLUTION_TIME_ELIGIBILITY_NAME,
    RESOLUTION_TIME_TARGET_NAME,
)


PEAK_LOWER_HOURS = 850.0
PEAK_UPPER_HOURS = 950.0
COMPARISON_EDGES = (650.0, 750.0, 850.0, 950.0, 1050.0, 1150.0)
HISTOGRAM_WIDTHS = (10, 25, 50)
DETAIL_LOWER_HOURS = 600.0
DETAIL_UPPER_HOURS = 1200.0


@dataclass(frozen=True)
class PeakInvestigationEvidence:
    """All deterministic tables needed by the focused report."""

    eligible: pd.DataFrame
    peak: pd.DataFrame
    summary: pd.DataFrame
    intervals: pd.DataFrame
    complaint_types: pd.DataFrame
    timing_patterns: pd.DataFrame
    data_quality: pd.DataFrame


def build_eligible_duration_frame(
    population: DurationAnalysisPopulation,
    *,
    identifier_column: str = "unique_key",
    complaint_type_column: str = "complaint_type",
) -> pd.DataFrame:
    """Join eligible targets to source fields while preserving source indexes."""
    required = {
        identifier_column, complaint_type_column, "created_date", "closed_date"
    }
    missing = sorted(required.difference(population.historical.columns))
    if missing:
        raise ValueError(f"Peak investigation is missing required columns: {missing}")
    mask = population.targets[RESOLUTION_TIME_ELIGIBILITY_NAME]
    eligible = population.historical.loc[mask].copy()
    eligible[RESOLUTION_TIME_TARGET_NAME] = population.targets.loc[
        mask, RESOLUTION_TIME_TARGET_NAME
    ].astype("float64")
    return eligible


def select_peak_interval(eligible: pd.DataFrame) -> pd.DataFrame:
    """Return complaints in the primary inclusive 850–950-hour interval."""
    duration = eligible[RESOLUTION_TIME_TARGET_NAME]
    return eligible.loc[
        duration.ge(PEAK_LOWER_HOURS) & duration.le(PEAK_UPPER_HOURS)
    ].copy()


def build_peak_summary(
    eligible: pd.DataFrame,
    peak: pd.DataFrame,
    *,
    identifier_column: str = "unique_key",
) -> pd.DataFrame:
    """Summarize primary inclusive interval membership and identifier coverage."""
    count = len(peak)
    duration = peak[RESOLUTION_TIME_TARGET_NAME]
    values = {
        "eligible_complaint_count": float(len(eligible)),
        "peak_complaint_count": float(count),
        "peak_percentage_of_eligible": count / len(eligible) * 100.0,
        "peak_minimum_hours": float(duration.min()),
        "peak_maximum_hours": float(duration.max()),
        "peak_median_hours": float(duration.median()),
        "peak_unique_identifier_count": float(peak[identifier_column].nunique(dropna=True)),
    }
    return pd.DataFrame(values.items(), columns=("metric", "value"))


def build_interval_comparison(eligible: pd.DataFrame) -> pd.DataFrame:
    """Compare non-overlapping half-open 100-hour bins and bin-width sensitivity."""
    duration = eligible[RESOLUTION_TIME_TARGET_NAME]
    total = len(eligible)
    primary_count = int(
        (duration.ge(PEAK_LOWER_HOURS) & duration.le(PEAK_UPPER_HOURS)).sum()
    )
    rows: list[dict[str, object]] = [{
        "analysis_type": "primary_inclusive_interval",
        "label": "[850, 950]",
        "lower_hours": PEAK_LOWER_HOURS,
        "upper_hours": PEAK_UPPER_HOURS,
        "bin_width_hours": PEAK_UPPER_HOURS - PEAK_LOWER_HOURS,
        "complaint_count": primary_count,
        "percentage_of_eligible": primary_count / total * 100.0,
    }]
    for lower, upper in zip(COMPARISON_EDGES[:-1], COMPARISON_EDGES[1:], strict=True):
        count = int((duration.ge(lower) & duration.lt(upper)).sum())
        rows.append({
            "analysis_type": "neighbor_interval",
            "label": f"[{lower:g}, {upper:g})",
            "lower_hours": lower,
            "upper_hours": upper,
            "bin_width_hours": upper - lower,
            "complaint_count": count,
            "percentage_of_eligible": count / total * 100.0,
        })
    # Fixed-origin bins make changes across widths reproducible and reveal
    # whether one plotting choice alone creates the apparent peak.
    for width in HISTOGRAM_WIDTHS:
        for lower in np.arange(DETAIL_LOWER_HOURS, DETAIL_UPPER_HOURS, width):
            upper = lower + width
            count = int((duration.ge(lower) & duration.lt(upper)).sum())
            rows.append({
                "analysis_type": "histogram_sensitivity",
                "label": f"[{lower:g}, {upper:g})",
                "lower_hours": float(lower),
                "upper_hours": float(upper),
                "bin_width_hours": float(width),
                "complaint_count": count,
                "percentage_of_eligible": count / total * 100.0,
            })
    return pd.DataFrame(rows)


def build_complaint_type_analysis(
    eligible: pd.DataFrame,
    peak: pd.DataFrame,
    *,
    complaint_type_column: str = "complaint_type",
) -> pd.DataFrame:
    """Compare peak composition with each complaint type's eligible prevalence."""
    total = len(eligible)
    peak_total = len(peak)
    overall = eligible[complaint_type_column].astype("string").fillna("<MISSING>")
    peak_types = peak[complaint_type_column].astype("string").fillna("<MISSING>")
    overall_counts = overall.value_counts(sort=False)
    peak_counts = peak_types.value_counts(sort=False)
    rows = []
    for complaint_type in sorted(overall_counts.index):
        eligible_count = int(overall_counts[complaint_type])
        peak_count = int(peak_counts.get(complaint_type, 0))
        overall_share = eligible_count / total * 100.0
        peak_share = peak_count / peak_total * 100.0 if peak_total else 0.0
        rows.append({
            "complaint_type": str(complaint_type),
            "eligible_count": eligible_count,
            "peak_count": peak_count,
            "percentage_of_peak": peak_share,
            "within_type_peak_percentage": peak_count / eligible_count * 100.0,
            "percentage_of_eligible_population": overall_share,
            "representation_ratio": peak_share / overall_share if overall_share else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["peak_count", "complaint_type"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def build_timing_patterns(peak: pd.DataFrame) -> pd.DataFrame:
    """Profile peak creation, closure, repeated timestamps, and 35–40-day bands."""
    created = pd.to_datetime(peak["created_date"], utc=True, errors="coerce")
    closed = pd.to_datetime(peak["closed_date"], utc=True, errors="coerce")
    total = len(peak)
    tables: list[pd.DataFrame] = []

    def add(pattern_type: str, values: pd.Series, *, minimum_count: int = 1) -> None:
        counts = values.value_counts(dropna=False).sort_index()
        counts = counts.loc[counts.ge(minimum_count)]
        table = counts.rename_axis("pattern_value").reset_index(name="count")
        table.insert(0, "pattern_type", pattern_type)
        table["percentage_of_peak"] = table["count"] / total * 100.0 if total else 0.0
        tables.append(table)

    add("created_month", created.dt.strftime("%Y-%m").fillna("<INVALID>"))
    add("created_day_of_week", created.dt.day_name().fillna("<INVALID>"))
    add("closed_month", closed.dt.strftime("%Y-%m").fillna("<INVALID>"))
    add("closed_hour_utc", closed.dt.hour.astype("Int64").astype("string").fillna("<INVALID>"))
    add("closed_date", closed.dt.strftime("%Y-%m-%d").fillna("<INVALID>"))
    add("identical_closed_timestamp", closed.astype("string"), minimum_count=2)

    duration_days = peak[RESOLUTION_TIME_TARGET_NAME] / 24.0
    day_band = pd.Series("outside_35_to_40_days", index=peak.index, dtype="string")
    for day in range(35, 41):
        mask = duration_days.ge(day) & duration_days.lt(day + 1)
        day_band.loc[mask] = f"[{day}, {day + 1}) days"
    add("duration_day_band", day_band)
    return pd.concat(tables, ignore_index=True)


def build_data_quality_checks(
    eligible: pd.DataFrame,
    *,
    identifier_column: str = "unique_key",
) -> pd.DataFrame:
    """Measure focused integrity and repetition indicators without altering rows."""
    created = pd.to_datetime(eligible["created_date"], utc=True, errors="coerce")
    closed = pd.to_datetime(eligible["closed_date"], utc=True, errors="coerce")
    duration = eligible[RESOLUTION_TIME_TARGET_NAME].astype("float64")
    recalculated = (closed - created).dt.total_seconds() / 3600.0
    finite = np.isfinite(duration.to_numpy())
    exact_duration_counts = duration.value_counts()
    closure_counts = closed.value_counts()
    checks: list[tuple[str, float, str]] = [
        ("duplicate_identifier_rows", float(eligible[identifier_column].duplicated(keep=False).sum()), "count"),
        ("exact_duplicate_rows", float(eligible.duplicated(keep=False).sum()), "count"),
        ("missing_or_invalid_created_timestamps", float(created.isna().sum()), "count"),
        ("missing_or_invalid_closed_timestamps", float(closed.isna().sum()), "count"),
        ("invalid_timestamp_order_rows", float(closed.lt(created).sum()), "count"),
        ("non_finite_duration_rows", float((~finite).sum()), "count"),
        ("negative_duration_rows", float(duration.lt(0).sum()), "count"),
        ("unique_duration_value_count", float(duration.nunique()), "count"),
        ("maximum_exact_duration_repetition", float(exact_duration_counts.max()), "count"),
        ("maximum_identical_closure_timestamp_repetition", float(closure_counts.max()), "count"),
        ("integer_hour_duration_count", float(np.isclose(duration, np.round(duration), rtol=0.0, atol=1e-12).sum()), "count"),
        ("maximum_recalculation_difference_hours", float((duration - recalculated).abs().max()), "hours"),
    ]
    return pd.DataFrame(checks, columns=("check", "value", "unit"))


def investigate_900_hour_concentration(
    population: DurationAnalysisPopulation,
    *,
    identifier_column: str = "unique_key",
    complaint_type_column: str = "complaint_type",
) -> PeakInvestigationEvidence:
    """Build all focused 900-hour investigation evidence from Phase 7.2 data."""
    eligible = build_eligible_duration_frame(
        population,
        identifier_column=identifier_column,
        complaint_type_column=complaint_type_column,
    )
    if eligible.empty:
        raise ValueError("The 900-hour investigation requires eligible durations.")
    original = eligible.copy(deep=True)
    peak = select_peak_interval(eligible)
    if peak.empty:
        raise ValueError("No eligible complaints fall in the inclusive 850–950-hour interval.")
    eligible_quality = build_data_quality_checks(
        eligible, identifier_column=identifier_column
    )
    peak_quality = build_data_quality_checks(
        peak, identifier_column=identifier_column
    )
    peak_quality["check"] = "peak_" + peak_quality["check"]
    evidence = PeakInvestigationEvidence(
        eligible=eligible,
        peak=peak,
        summary=build_peak_summary(eligible, peak, identifier_column=identifier_column),
        intervals=build_interval_comparison(eligible),
        complaint_types=build_complaint_type_analysis(
            eligible, peak, complaint_type_column=complaint_type_column
        ),
        timing_patterns=build_timing_patterns(peak),
        data_quality=pd.concat([eligible_quality, peak_quality], ignore_index=True),
    )
    if not eligible.equals(original):
        raise RuntimeError("Peak investigation mutated the eligible source frame.")
    return evidence
