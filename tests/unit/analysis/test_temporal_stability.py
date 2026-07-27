"""Unit tests for deterministic temporal-stability scope decisions."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_ops.analysis.temporal_stability import (
    APPROVED,
    APPROVED_WITH_LIMITATIONS,
    NOT_FEASIBLE,
    REQUIRES_NARROWER_RANGE,
    TemporalStabilityThresholds,
    add_outcome_maturity,
    build_monthly_scope_metrics,
    calculate_temporal_stability_metrics,
    compare_scope_candidates,
    evaluate_scope_candidate,
)


EXTRACTED_AT = pd.Timestamp("2024-05-15T00:00:00Z")
TEST_THRESHOLDS = TemporalStabilityThresholds(
    minimum_eligible_records=4,
    minimum_due_date_coverage=0.70,
    minimum_closed_date_coverage=0.80,
    minimum_active_months=2,
    minimum_monthly_eligible_volume=1,
    minimum_complete_calendar_years=0,
)


def records_for_rates(rates: list[float], *, records_per_month: int = 10) -> pd.DataFrame:
    """Create deterministic monthly records with exact binary-target rates."""
    rows = []
    for month_number, rate in enumerate(rates, start=1):
        created = pd.Timestamp(2024, month_number, 1, tz="UTC")
        due = created + pd.Timedelta(days=5)
        missed = round(rate * records_per_month)
        for index in range(records_per_month):
            closed = due + pd.Timedelta(days=1) if index < missed else due
            rows.append(
                {
                    "unique_key": f"{month_number}-{index}",
                    "created_date": created,
                    "due_date": due,
                    "closed_date": closed,
                }
            )
    return pd.DataFrame(rows)


def candidate_for_rates(rates: list[float]) -> dict[str, object]:
    """Evaluate one test candidate using shared thresholds."""
    frame = records_for_rates(rates)
    monthly = build_monthly_scope_metrics(
        frame,
        "2024-01-01",
        f"2024-{len(rates):02d}-28",
        EXTRACTED_AT,
    )
    return evaluate_scope_candidate(
        monthly,
        candidate_name="test",
        start_date="2024-01-01",
        end_date=f"2024-{len(rates):02d}-28",
        thresholds=TEST_THRESHOLDS,
    )


def test_zero_missing_months_can_have_unstable_target_rate() -> None:
    candidate = candidate_for_rates([0.0, 1.0])

    assert candidate["missing_months"] == 0
    assert candidate["severe_temporal_volatility"]
    assert candidate["decision_status"] == APPROVED_WITH_LIMITATIONS


def test_stable_monthly_target_rate_is_approved() -> None:
    candidate = candidate_for_rates([0.4, 0.4])

    assert candidate["monthly_target_rate_range"] == pytest.approx(0.0)
    assert candidate["decision_status"] == APPROVED


def test_candidate_with_insufficient_eligible_volume_is_not_feasible() -> None:
    thresholds = TemporalStabilityThresholds(
        minimum_eligible_records=100,
        minimum_active_months=2,
        minimum_monthly_eligible_volume=1,
        minimum_complete_calendar_years=0,
    )
    monthly = build_monthly_scope_metrics(
        records_for_rates([0.5, 0.5]),
        "2024-01-01",
        "2024-02-29",
        EXTRACTED_AT,
    )

    candidate = evaluate_scope_candidate(
        monthly,
        candidate_name="small",
        start_date="2024-01-01",
        end_date="2024-02-29",
        thresholds=thresholds,
    )

    assert not candidate["passes_eligible_record_count"]
    assert candidate["decision_status"] == NOT_FEASIBLE


def test_candidate_with_inadequate_due_date_coverage_is_not_feasible() -> None:
    frame = records_for_rates([0.5, 0.5])
    frame.loc[:9, "due_date"] = pd.NaT
    monthly = build_monthly_scope_metrics(
        frame, "2024-01-01", "2024-02-29", EXTRACTED_AT
    )

    candidate = evaluate_scope_candidate(
        monthly,
        candidate_name="low coverage",
        start_date="2024-01-01",
        end_date="2024-02-29",
        thresholds=TEST_THRESHOLDS,
    )

    assert not candidate["passes_due_date_coverage"]
    assert candidate["decision_status"] == NOT_FEASIBLE


def test_candidate_with_one_target_class_is_not_feasible() -> None:
    candidate = candidate_for_rates([1.0, 1.0])

    assert not candidate["passes_target_class_balance"]
    assert candidate["decision_status"] == NOT_FEASIBLE


def test_incomplete_month_detection_adds_zero_volume_month() -> None:
    frame = records_for_rates([0.5, 0.5])
    frame = frame.loc[pd.to_datetime(frame["created_date"]).dt.month.eq(1)]

    monthly = build_monthly_scope_metrics(
        frame, "2024-01-01", "2024-02-29", EXTRACTED_AT
    )
    metrics = calculate_temporal_stability_metrics(monthly)

    assert len(monthly) == 2
    assert metrics["active_months"] == 1
    assert metrics["missing_months"] == 1


def test_outcome_immature_record_is_detected_and_not_labelled() -> None:
    frame = pd.DataFrame(
        [{
            "created_date": "2024-05-01T00:00:00Z",
            "due_date": "2024-06-01T00:00:00Z",
            "closed_date": None,
        }]
    )

    prepared = add_outcome_maturity(frame, EXTRACTED_AT)

    assert not prepared.loc[0, "outcome_mature"]
    assert not prepared.loc[0, "analysis_eligible"]
    assert pd.isna(prepared.loc[0, "missed_resolution_target"])


def test_month_to_month_target_rate_change_is_absolute_maximum() -> None:
    candidate = candidate_for_rates([0.2, 0.8, 0.5])

    assert candidate["largest_absolute_month_to_month_target_rate_change"] == pytest.approx(0.6)


def test_monthly_volume_coefficient_of_variation_uses_population_std() -> None:
    frame = pd.concat(
        [
            records_for_rates([0.5], records_per_month=10),
            records_for_rates([0.5], records_per_month=20).assign(
                created_date=pd.Timestamp("2024-02-01T00:00:00Z"),
                due_date=pd.Timestamp("2024-02-06T00:00:00Z"),
                closed_date=pd.Timestamp("2024-02-06T00:00:00Z"),
            ),
        ],
        ignore_index=True,
    )
    monthly = build_monthly_scope_metrics(
        frame, "2024-01-01", "2024-02-29", EXTRACTED_AT
    )

    metrics = calculate_temporal_stability_metrics(monthly)

    assert metrics["monthly_volume_coefficient_of_variation"] == pytest.approx(1 / 3)


def test_severe_volatility_changes_unconditional_approval_status() -> None:
    stable = candidate_for_rates([0.4, 0.4])
    unstable = candidate_for_rates([0.1, 0.9])

    assert stable["decision_status"] == APPROVED
    assert unstable["decision_status"] == APPROVED_WITH_LIMITATIONS


def test_better_stable_candidate_requires_wider_candidate_to_narrow() -> None:
    wide = candidate_for_rates([0.1, 0.9])
    wide.update({"candidate_name": "wide", "start_date": "2024-01-01", "end_date": "2024-02-29"})
    narrow = candidate_for_rates([0.4, 0.4])
    narrow.update(
        {
            "candidate_name": "narrow",
            "start_date": "2024-01-01",
            "end_date": "2024-02-29",
        }
    )

    comparison = compare_scope_candidates([wide, narrow], thresholds=TEST_THRESHOLDS)
    wide_result = comparison.loc[comparison["candidate_name"].eq("wide")].iloc[0]

    assert wide_result["better_stable_candidate_exists"]
    assert wide_result["decision_status"] == REQUIRES_NARROWER_RANGE


def test_final_selected_scope_uses_one_consistent_date_range() -> None:
    stable = candidate_for_rates([0.4, 0.4])
    stable.update(
        {
            "candidate_name": "stable",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }
    )
    unstable = candidate_for_rates([0.1, 0.9])
    unstable.update(
        {
            "candidate_name": "unstable",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
        }
    )

    comparison = compare_scope_candidates([unstable, stable], thresholds=TEST_THRESHOLDS)
    selected = comparison.loc[comparison["selected"]]

    assert len(selected) == 1
    assert selected.iloc[0][["start_date", "end_date"]].tolist() == [
        "2024-01-01",
        "2024-12-31",
    ]
