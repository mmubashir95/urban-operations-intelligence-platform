"""Focused tests for Phase 7.2 population and duration statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_ops.analysis.resolution_duration import (
    build_eligibility_summary,
    calculate_duration_statistics,
    most_common_durations,
    select_training_duration_population,
)
from urban_ops.splitting.models import SplitBoundaries


def _boundaries() -> SplitBoundaries:
    """Return compact frozen-style half-open split boundaries."""
    return SplitBoundaries(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    )


def _metrics(table: pd.DataFrame) -> dict[str, float]:
    return dict(zip(table["metric"], table["value"], strict=True))


def test_statistics_include_required_metrics_and_are_deterministic() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 9.0])
    first = calculate_duration_statistics(values)
    second = calculate_duration_statistics(values)
    metrics = _metrics(first)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == ("metric", "value", "unit")
    assert metrics["count"] == 4
    assert metrics["mean"] == 3
    assert metrics["median"] == 1.5
    assert metrics["standard_deviation"] == pytest.approx(values.std())
    assert metrics["minimum"] == 0
    assert metrics["maximum"] == 9
    for name, quantile in {
        "p25": 0.25, "p50": 0.5, "p75": 0.75,
        "p90": 0.9, "p95": 0.95, "p99": 0.99,
    }.items():
        assert metrics[name] == pytest.approx(values.quantile(quantile))
    assert metrics["zero_duration_count"] == 1
    assert metrics["skewness"] == pytest.approx(values.skew())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "empty"),
        ([1.0, np.nan], "numeric and non-null"),
        ([1.0, np.inf], "finite"),
        ([1.0, -0.1], "nonnegative"),
        ([1.0, "invalid"], "numeric and non-null"),
    ],
)
def test_statistics_reject_empty_or_invalid_values(
    values: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_duration_statistics(pd.Series(values, dtype="object"))


def test_historical_population_obeys_boundaries_and_observation_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "unique_key": ["before", "start", "late-close", "zero", "cutoff", "invalid"],
            "created_date": [
                "2023-12-31T23:59:59Z",
                "2024-01-01T00:00:00Z",
                "2024-03-20T00:00:00Z",
                "2024-03-31T12:00:00Z",
                "2024-04-01T00:00:00Z",
                "bad-date",
            ],
            "closed_date": [
                "2024-01-01T01:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-04-15T00:00:00Z",
                "2024-03-31T12:00:00Z",
                "2024-04-01T01:00:00Z",
                None,
            ],
            "status": ["Closed"] * 6,
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    original = frame.copy(deep=True)

    population = select_training_duration_population(frame, boundaries=_boundaries())

    assert population.historical["unique_key"].tolist() == ["start", "late-close", "zero"]
    assert population.targets.index.tolist() == [11, 12, 13]
    assert population.eligible_durations.tolist() == [1.0, 0.0]
    assert population.targets.loc[12, "resolution_time_exclusion_reason"] == (
        "resolution_not_observed_by_cutoff"
    )
    assert population.before_training_count == 1
    assert population.after_training_count == 1
    assert population.invalid_created_count == 1
    pd.testing.assert_frame_equal(frame, original)

    summary = build_eligibility_summary(population)
    population_counts = dict(
        summary.loc[summary["summary_type"].eq("population"), ["name", "count"]]
        .itertuples(index=False, name=None)
    )
    assert population_counts == {
        "total_historical_complaints": 3,
        "eligible_complaints": 2,
        "ineligible_complaints": 1,
    }


def test_only_eligible_rows_contribute_to_statistics() -> None:
    frame = pd.DataFrame({
        "created_date": ["2024-02-01T00:00:00Z"] * 3,
        "closed_date": ["2024-02-01T02:00:00Z", None, "2024-02-01T08:00:00Z"],
        "status": ["Closed", "Open", "Cancelled"],
    })
    population = select_training_duration_population(frame, boundaries=_boundaries())
    metrics = _metrics(calculate_duration_statistics(population.eligible_durations))
    assert metrics["count"] == 1
    assert metrics["mean"] == 2.0


def test_common_duration_ranking_is_stable() -> None:
    result = most_common_durations(pd.Series([2.0, 1.0, 2.0, 1.0, 3.0]))
    assert result.to_dict("records") == [
        {"duration_hours": 1.0, "count": 2},
        {"duration_hours": 2.0, "count": 2},
    ]
