"""Analyze Phase 7.1 resolution durations within the frozen training window.

This module selects complaints by creation time, applies the existing
resolution-time target at the training observation cutoff, and calculates
descriptive evidence. It never changes, caps, or imputes target values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from urban_ops.features.target import (
    RESOLUTION_TIME_CATEGORY_NAME,
    RESOLUTION_TIME_ELIGIBILITY_NAME,
    RESOLUTION_TIME_REASON_NAME,
    RESOLUTION_TIME_TARGET_NAME,
    build_resolution_time_target,
)
from urban_ops.splitting.models import SplitBoundaries


STATISTICS_SCHEMA = ("metric", "value", "unit")
ELIGIBILITY_SCHEMA = ("summary_type", "name", "count", "percentage")
PERCENTILES = {
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


@dataclass(frozen=True)
class DurationAnalysisPopulation:
    """Selected historical rows, aligned targets, and source exclusion counts."""

    source_row_count: int
    before_training_count: int
    after_training_count: int
    invalid_created_count: int
    historical: pd.DataFrame
    targets: pd.DataFrame

    @property
    def eligible_durations(self) -> pd.Series:
        """Return eligible, non-null durations without changing their values."""
        mask = self.targets[RESOLUTION_TIME_ELIGIBILITY_NAME]
        return self.targets.loc[mask, RESOLUTION_TIME_TARGET_NAME].astype("float64")


def select_training_duration_population(
    frame: pd.DataFrame,
    *,
    boundaries: SplitBoundaries,
    timestamp_column: str = "created_date",
) -> DurationAnalysisPopulation:
    """Select the half-open training window and apply Phase 7.1 at its cutoff.

    Selection uses creation time only. A closure after ``train_end_exclusive``
    is therefore retained in the historical population but marked unobserved
    by the target builder, preventing future outcome leakage.
    """
    required = {timestamp_column, "closed_date", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Duration analysis is missing required columns: {missing}")

    created = pd.to_datetime(
        frame[timestamp_column], errors="coerce", utc=True, format="mixed"
    )
    valid = created.notna()
    training_mask = (
        valid
        & created.ge(boundaries.train_start)
        & created.lt(boundaries.train_end_exclusive)
    )
    historical = frame.loc[training_mask].copy()
    # Normalize the selection timestamp so the target builder receives the
    # same UTC representation used by the governed chronological splitter.
    historical[timestamp_column] = created.loc[training_mask]
    targets = build_resolution_time_target(
        historical, observation_cutoff=boundaries.train_end_exclusive
    )
    before = int((valid & created.lt(boundaries.train_start)).sum())
    after = int((valid & created.ge(boundaries.train_end_exclusive)).sum())
    invalid = int((~valid).sum())
    if len(frame) != len(historical) + before + after + invalid:
        raise RuntimeError("Historical population selection counts do not reconcile.")
    return DurationAnalysisPopulation(
        source_row_count=len(frame),
        before_training_count=before,
        after_training_count=after,
        invalid_created_count=invalid,
        historical=historical,
        targets=targets,
    )


def calculate_duration_statistics(durations: pd.Series) -> pd.DataFrame:
    """Calculate deterministic descriptive statistics in canonical hours.

    Raises:
        ValueError: If there are no values or any value is null, non-numeric,
            non-finite, or negative.
    """
    numeric = pd.to_numeric(durations, errors="coerce")
    if numeric.empty:
        raise ValueError("Eligible resolution durations are empty.")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if np.isnan(values).any():
        raise ValueError("Eligible resolution durations must be numeric and non-null.")
    if not np.isfinite(values).all():
        raise ValueError("Eligible resolution durations must be finite.")
    if (values < 0).any():
        raise ValueError("Eligible resolution durations must be nonnegative.")

    metrics: list[tuple[str, float, str]] = [
        ("count", float(values.size), "complaints"),
        ("mean", float(np.mean(values)), "hours"),
        ("median", float(np.median(values)), "hours"),
        ("standard_deviation", float(pd.Series(values).std(ddof=1)), "hours"),
        ("minimum", float(np.min(values)), "hours"),
        ("maximum", float(np.max(values)), "hours"),
    ]
    metrics.extend(
        (name, float(np.quantile(values, quantile, method="linear")), "hours")
        for name, quantile in PERCENTILES.items()
    )
    metrics.extend([
        ("zero_duration_count", float(np.count_nonzero(values == 0.0)), "complaints"),
        ("skewness", float(pd.Series(values).skew()), "unitless"),
    ])
    return pd.DataFrame(metrics, columns=STATISTICS_SCHEMA)


def build_eligibility_summary(
    population: DurationAnalysisPopulation,
) -> pd.DataFrame:
    """Return reconciled population, reason, and category counts."""
    targets = population.targets
    total = len(population.historical)
    eligible = int(targets[RESOLUTION_TIME_ELIGIBILITY_NAME].sum())
    rows: list[tuple[str, str, int, float]] = []

    def append(
        summary_type: str, name: str, count: int, *, denominator: int = total
    ) -> None:
        percentage = count / denominator * 100.0 if denominator else 0.0
        rows.append((summary_type, name, int(count), percentage))

    append("population", "total_historical_complaints", total)
    append("population", "eligible_complaints", eligible)
    append("population", "ineligible_complaints", total - eligible)
    for reason, count in targets[RESOLUTION_TIME_REASON_NAME].value_counts().sort_index().items():
        append("exclusion_reason", str(reason), int(count))
    for category, count in targets[RESOLUTION_TIME_CATEGORY_NAME].value_counts().sort_index().items():
        append("exclusion_category", str(category), int(count))
    append(
        "source_exclusion", "created_before_training_window",
        population.before_training_count, denominator=population.source_row_count,
    )
    append(
        "source_exclusion", "created_at_or_after_training_cutoff",
        population.after_training_count, denominator=population.source_row_count,
    )
    append(
        "source_exclusion", "invalid_created_timestamp",
        population.invalid_created_count, denominator=population.source_row_count,
    )
    summary = pd.DataFrame(rows, columns=ELIGIBILITY_SCHEMA)
    if eligible + (total - eligible) != total:
        raise RuntimeError("Eligibility counts do not reconcile.")
    category_total = int(
        summary.loc[summary["summary_type"].eq("exclusion_category"), "count"].sum()
    )
    if category_total != total:
        raise RuntimeError("Eligibility category counts do not reconcile.")
    return summary


def most_common_durations(durations: pd.Series, *, limit: int = 10) -> pd.DataFrame:
    """Return repeated exact-hour values for descriptive peak inspection."""
    if limit < 1:
        raise ValueError("limit must be positive.")
    counts = durations.value_counts().rename_axis("duration_hours").reset_index(name="count")
    counts = counts.loc[counts["count"].gt(1)].sort_values(
        ["count", "duration_hours"], ascending=[False, True], kind="mergesort"
    )
    return counts.head(limit).reset_index(drop=True)
