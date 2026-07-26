"""Unit tests for deterministic scope-selection analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_ops.analysis.scope_selection import (  # noqa: E402
    TARGET_NAME,
    apply_agency_discovery_rules,
    apply_complaint_type_decision_rules,
    apply_target_eligibility,
    build_agency_scope_summary,
    build_candidate_rejection_reasons,
    build_complaint_type_scope_summary,
    build_rejection_reasons,
    calculate_target_balance_score,
    determine_scope_decision_status,
    evaluate_candidate_critical_gates,
    rebuild_scoped_sensitivity_candidate,
    score_scope_candidates,
    summarize_monthly_stability,
    summarize_scope_date_range,
    validate_selected_scope_consistency,
    validate_scope_input,
    write_scope_decision,
)


def make_scope_frame() -> pd.DataFrame:
    """Return a small fixture spanning two months and complaint types."""
    return pd.DataFrame(
        {
            "unique_key": ["1", "2", "3", "4", "5"],
            "created_date": [
                "2024-01-01T00:00:00",
                "2024-01-15T00:00:00",
                "2024-02-01T00:00:00",
                "2024-02-15T00:00:00",
                "2024-02-20T00:00:00",
            ],
            "closed_date": [
                "2024-01-03T00:00:00",
                "2024-01-16T00:00:00",
                "2024-02-05T00:00:00",
                None,
                "2024-02-22T00:00:00",
            ],
            "due_date": [
                "2024-01-02T00:00:00",
                "2024-01-17T00:00:00",
                "2024-02-04T00:00:00",
                "2024-02-20T00:00:00",
                None,
            ],
            "agency": ["A"] * 5,
            "agency_name": ["Agency A"] * 5,
            "complaint_type": ["Noise", "Noise", "Heat", "Heat", "Heat"],
            "status": ["Closed", "Closed", "Closed", "Open", "Closed"],
        }
    )


def prepared_frame() -> pd.DataFrame:
    """Return the fixture after validation and target construction."""
    validated, _ = validate_scope_input(make_scope_frame())
    return apply_target_eligibility(validated)


def test_agency_summary_calculates_coverage_target_and_active_months() -> None:
    """Agency metrics use safe denominators and the authoritative target."""
    summary = build_agency_scope_summary(prepared_frame()).iloc[0]

    assert summary["total_records"] == 5
    assert summary["due_date_coverage"] == pytest.approx(0.8)
    assert summary["closed_date_coverage"] == pytest.approx(0.8)
    assert summary["eligible_target_count"] == 3
    assert summary["missed_target_count"] == 2
    assert summary["on_time_count"] == 1
    assert summary["missed_target_rate"] == pytest.approx(2 / 3)
    assert summary["active_months"] == 2


def test_agency_summary_handles_zero_target_denominator() -> None:
    """An agency without eligible records receives finite zero target rates."""
    frame = prepared_frame()
    frame["due_date"] = pd.NaT
    frame = apply_target_eligibility(frame)

    summary = build_agency_scope_summary(frame).iloc[0]

    assert summary["eligible_target_count"] == 0
    assert summary["missed_target_rate"] == 0.0


def test_complaint_type_summary_groups_and_calculates_monthly_volume() -> None:
    """Complaint summaries preserve grouping and monthly volume statistics."""
    summary = build_complaint_type_scope_summary(prepared_frame())
    heat = summary.loc[summary["complaint_type"] == "Heat"].iloc[0]

    assert len(summary) == 2
    assert heat["total_records"] == 3
    assert heat["eligible_target_count"] == 1
    assert heat["missed_target_count"] == 1
    assert heat["median_monthly_volume"] == 3


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0), (0.25, 0.5)],
)
def test_target_balance_score_is_bounded_and_centered(rate: float, expected: float) -> None:
    """Balance scoring peaks at 50 percent and is zero at both extremes."""
    score = calculate_target_balance_score(rate)

    assert score == pytest.approx(expected)
    assert 0 <= score <= 1


def test_date_range_summary_respects_inclusive_boundaries_and_missing_months() -> None:
    """Date candidates include both boundary dates and report absent months."""
    summary = summarize_scope_date_range(
        prepared_frame(),
        "first quarter",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-03-31"),
    )

    assert summary["total_records"] == 5
    assert summary["active_months"] == 2
    assert summary["missing_month_count"] == 1


def test_date_range_summary_handles_empty_period() -> None:
    """An empty date candidate returns schema-complete zero metrics."""
    summary = summarize_scope_date_range(
        prepared_frame(),
        "empty",
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-31"),
    )

    assert summary["total_records"] == 0
    assert summary["eligible_target_count"] == 0
    assert summary["active_months"] == 0
    assert summary["missing_month_count"] == 1


def test_rejection_reasons_are_stable_for_one_multiple_and_no_failures() -> None:
    """Rejection strings preserve documented ordering and avoid duplicates."""
    one = build_rejection_reasons({"eligible_target_count": 10, "passes_due_date_coverage": False})
    multiple = build_rejection_reasons(
        {
            "eligible_target_count": 0,
            "passes_eligible_volume": False,
            "passes_due_date_coverage": False,
        }
    )
    none = build_rejection_reasons({"eligible_target_count": 10, "passes_due_date_coverage": True})

    assert one == "Low due-date coverage"
    assert multiple == "No target-eligible records | Insufficient eligible records | Low due-date coverage"
    assert none == ""


def test_scope_score_applies_weights_and_handles_neutral_qualitative_input() -> None:
    """Candidate scores are bounded and use an explicit neutral relevance input."""
    candidates = pd.DataFrame(
        [
            {
                "due_date_coverage": 1.0,
                "closed_date_coverage": 1.0,
                "eligible_target_count": 100_000,
                "missed_target_rate": 0.5,
                "invalid_timestamp_rate": 0.0,
                "missing_month_count": 0,
                "active_months": 12,
            }
        ]
    )

    scored = score_scope_candidates(candidates, operational_relevance_score=0.5).iloc[0]

    assert scored["operational_relevance_score"] == 0.5
    assert scored["scope_score"] == pytest.approx(0.95)
    assert 0 <= scored["scope_score"] <= 1


def test_markdown_writer_includes_required_sections_and_values(tmp_path: Path) -> None:
    """The generated decision document contains every required heading and evidence."""
    output_path = tmp_path / "scope_decision.md"
    write_scope_decision(
        output_path,
        decision={
            "decision_status": "Requires review",
            "selected_agency": "DSNY",
            "selected_agency_name": "Department of Sanitation",
            "selected_start_date": "2021-01-01",
            "selected_end_date": "2024-12-31",
            "eligible_target_records": 1200,
            "missed_target_count": 300,
            "on_time_count": 900,
            "missed_target_rate": 0.25,
            "due_date_coverage": 0.1,
            "closed_date_coverage": 0.9,
            "complaint_type_evidence_start_date": "2020-01-01",
            "complaint_type_evidence_end_date": "2024-12-31",
        },
        selected_complaint_types=[],
        rejected_candidates=pd.DataFrame(
            [{"candidate_id": "C1", "due_date_coverage": 0.65, "rejection_reasons": "Low due-date coverage"}]
        ),
        extraction={"dataset_id": "erm2-nwe9", "rows_represented": 5000},
        stability={
            "target_rate_range": 0.05,
            "largest_month_to_month_target_rate_change": 0.02,
            "target_rate_volatility_flag": False,
        },
        limitations=["Expected-deadline coverage is insufficient."],
    )

    content = output_path.read_text(encoding="utf-8")
    required_sections = [
        "Decision Status",
        "Executive Summary",
        "Business Problem",
        "Prediction Target",
        "Target Definition Reference",
        "Data Source",
        "Extraction Summary",
        "Selection Criteria",
        "Selected Agency",
        "Selected Date Range",
        "Selected Complaint Types",
        "Eligible Population",
        "Target Distribution",
        "Data Quality Summary",
        "Temporal Stability",
        "Sensitivity Analysis",
        "Included Records",
        "Exclusion Rules",
        "Rejected Alternatives",
        "Known Limitations",
        "Downstream Implications",
        "Reproducibility Information",
        "Approval Status",
    ]
    assert all(f"## {section}" in content for section in required_sections)
    assert "DSNY" in content
    assert "1,200" in content
    assert "65.00%" in content
    assert "stable range" in content


def test_markdown_writer_surfaces_volatility_as_a_limitation(tmp_path: Path) -> None:
    """A volatile monthly target rate is disclosed in stability text and limitations."""
    output_path = tmp_path / "scope_decision.md"
    write_scope_decision(
        output_path,
        decision={
            "decision_status": "Approved with limitations",
            "selected_agency": "DSNY",
            "selected_agency_name": "Department of Sanitation",
            "selected_start_date": "2022-01-01",
            "selected_end_date": "2026-06-30",
            "eligible_target_records": 66_340,
            "missed_target_count": 34_271,
            "on_time_count": 32_069,
            "missed_target_rate": 0.5166,
            "due_date_coverage": 0.912,
            "closed_date_coverage": 0.986,
            "complaint_type_evidence_start_date": "2020-01-01",
            "complaint_type_evidence_end_date": "2026-06-30",
        },
        selected_complaint_types=["Graffiti"],
        rejected_candidates=pd.DataFrame(),
        extraction={"dataset_id": "erm2-nwe9", "rows_represented": 5000},
        stability={
            "target_rate_range": 0.8559,
            "largest_month_to_month_target_rate_change": 0.3582,
            "monthly_volume_coefficient_of_variation": 0.4669,
            "due_date_coverage_range": 0.1951,
            "closed_date_coverage_range": 0.1518,
            "minimum_monthly_eligible_volume": 325,
            "maximum_monthly_eligible_volume": 3692,
            "target_rate_volatility_flag": True,
        },
        limitations=["Expected-deadline coverage is insufficient."],
    )

    content = output_path.read_text(encoding="utf-8")
    assert "varies substantially" in content
    assert "85.59%" in content
    assert "time-aware validation design" in content
    assert "Monthly missed-target rate varies by 86%" in content


def test_validation_reports_parse_duplicates_and_sequence_failures() -> None:
    """Prerequisite diagnostics distinguish parsing, duplication, and chronology."""
    frame = make_scope_frame()
    frame.loc[1, "unique_key"] = "1"
    frame.loc[2, "due_date"] = "invalid"
    frame.loc[4, "closed_date"] = "2024-02-19T00:00:00"

    _, diagnostics = validate_scope_input(frame)

    assert diagnostics["duplicate_unique_key_count"] == 1
    assert diagnostics["conflicting_duplicate_unique_key_count"] == 1
    assert diagnostics["due_date_parsing_failure_count"] == 1
    assert diagnostics["invalid_closed_sequence_count"] == 1


def test_low_coverage_agency_with_target_evidence_proceeds_to_type_review() -> None:
    """Absolute target evidence permits discovery despite low broad coverage."""
    agencies = pd.DataFrame(
        [
            {
                "agency": "A",
                "eligible_target_count": 75_000,
                "due_date_coverage": 0.05,
                "closed_date_coverage": 0.99,
                "active_months": 24,
                "invalid_timestamp_rate": 0.001,
            }
        ]
    )

    discovered = apply_agency_discovery_rules(
        agencies,
        minimum_eligible_records=10_000,
        minimum_due_date_coverage=0.70,
        minimum_closed_date_coverage=0.80,
        minimum_active_months=12,
        maximum_invalid_timestamp_rate=0.05,
    ).iloc[0]

    assert not discovered["passes_agency_wide_coverage"]
    assert discovered["has_target_evidence"]
    assert discovered["proceed_to_complaint_type_review"]


def test_parent_agency_does_not_demote_included_complaint_type() -> None:
    """Complaint inclusion depends only on complaint-population evidence."""
    complaint_types = pd.DataFrame(
        [
            {
                "complaint_type": "Viable",
                "total_records": 11_000,
                "eligible_target_count": 10_000,
                "due_date_coverage": 0.95,
                "closed_date_coverage": 0.98,
                "missed_target_rate": 0.45,
                "active_months": 18,
                "median_monthly_volume": 500,
                "recent_period_record_count": 5_000,
                "invalid_timestamp_count": 1,
            }
        ]
    )

    classified = apply_complaint_type_decision_rules(
        complaint_types,
        minimum_eligible_records=1_000,
        minimum_due_date_coverage=0.70,
        minimum_closed_date_coverage=0.80,
        minimum_target_rate=0.05,
        maximum_target_rate=0.95,
        minimum_active_months=12,
        minimum_median_monthly_volume=25,
        maximum_invalid_timestamp_rate=0.05,
    ).iloc[0]

    assert classified["scope_status"] == "Include"


def test_candidate_gates_use_candidate_metrics_not_agency_coverage() -> None:
    """A 90% scoped coverage candidate passes despite 5% parent coverage."""
    candidate = pd.DataFrame(
        [
            {
                "agency_wide_due_date_coverage": 0.05,
                "eligible_target_count": 20_000,
                "due_date_coverage": 0.90,
                "closed_date_coverage": 0.95,
                "missed_target_rate": 0.40,
                "active_months": 24,
                "complaint_type_count": 1,
                "missing_month_count": 0,
                "invalid_timestamp_rate": 0.001,
            }
        ]
    )

    evaluated = evaluate_candidate_critical_gates(
        candidate,
        minimum_eligible_records=10_000,
        minimum_due_date_coverage=0.70,
        minimum_closed_date_coverage=0.80,
        minimum_target_rate=0.05,
        maximum_target_rate=0.95,
        minimum_active_months=12,
        maximum_invalid_timestamp_rate=0.05,
    ).iloc[0]

    assert evaluated["passes_critical_gates"]


def test_selected_scope_rejects_count_and_statistics_mismatch() -> None:
    """Serialization cannot describe non-empty statistics with zero types."""
    with pytest.raises(ValueError, match="count does not match"):
        validate_selected_scope_consistency(
            {"selected_complaint_type_count": 0, "total_records": 100},
            ["Viable"],
        )

    with pytest.raises(ValueError, match="non-empty selected scope"):
        validate_selected_scope_consistency(
            {"selected_complaint_type_count": 0, "total_records": 100},
            [],
        )


def test_candidate_rejection_reasons_use_candidate_evidence() -> None:
    """Passing scoped coverage never produces a due-date rejection reason."""
    reasons = build_candidate_rejection_reasons(
        {
            "passes_eligible_volume": False,
            "passes_due_date_coverage": True,
            "passes_closed_date_coverage": True,
            "passes_target_balance": True,
            "passes_temporal_coverage": True,
            "has_included_complaint_types": True,
            "passes_month_continuity": True,
            "passes_data_consistency": True,
        }
    )

    assert reasons == "Insufficient candidate eligible records"
    assert "due-date" not in reasons


def test_sensitivity_rebuilds_included_types_and_candidate_feasibility() -> None:
    """Changing complaint volume genuinely rebuilds the scoped candidate."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS", tz="UTC")
    monthly = pd.DataFrame(
        [
            {
                "agency": "A",
                "agency_name": "Agency A",
                "complaint_type": complaint_type,
                "created_month": month,
                "total_records": total,
                "due_date_present_count": due,
                "closed_date_present_count": closed,
                "eligible_target_count": eligible,
                "missed_target_count": missed,
                "invalid_due_sequence_count": 0,
                "invalid_closed_sequence_count": 0,
            }
            for month in months
            for complaint_type, total, due, closed, eligible, missed in [
                ("Viable", 1_000, 950, 980, 940, 470),
                ("Unusable", 2_000, 0, 1_990, 0, 0),
            ]
        ]
    )
    common = {
        "monthly": monthly,
        "agency": "A",
        "agency_name": "Agency A",
        "start_date": pd.Timestamp("2024-01-01"),
        "end_date": pd.Timestamp("2024-12-31"),
        "minimum_candidate_eligible_records": 5_000,
        "minimum_due_date_coverage": 0.70,
        "minimum_closed_date_coverage": 0.80,
        "minimum_target_rate": 0.05,
        "maximum_target_rate": 0.95,
        "minimum_active_months": 12,
        "minimum_median_monthly_volume": 25,
        "maximum_invalid_timestamp_rate": 0.05,
    }

    feasible = rebuild_scoped_sensitivity_candidate(
        **common,
        minimum_complaint_type_eligible_records=5_000,
    )
    infeasible = rebuild_scoped_sensitivity_candidate(
        **common,
        minimum_complaint_type_eligible_records=20_000,
    )

    assert feasible["included_complaint_types"] == "Viable"
    assert feasible["passes_critical_gates"]
    assert infeasible["complaint_type_count"] == 0
    assert not infeasible["passes_critical_gates"]


def test_monthly_stability_flags_volatile_target_rate_despite_full_continuity() -> None:
    """Zero missing months does not suppress a large monthly target-rate swing."""
    monthly = pd.DataFrame(
        {
            "eligible_target_count": [1000, 1000, 1000, 1000],
            "missed_target_rate": [0.95, 0.90, 0.20, 0.15],
            "due_date_coverage": [0.95, 0.80, 0.90, 0.85],
            "closed_date_coverage": [0.99, 0.98, 0.97, 0.99],
        }
    )

    stability = summarize_monthly_stability(monthly, expected_month_count=4)

    assert stability["missing_month_count"] == 0
    assert stability["target_rate_range"] == pytest.approx(0.80)
    assert stability["largest_month_to_month_target_rate_change"] == pytest.approx(0.70)
    assert stability["target_rate_volatility_flag"] is True


def test_monthly_stability_is_stable_for_flat_rates() -> None:
    """A near-constant monthly rate does not trip the volatility flag."""
    monthly = pd.DataFrame(
        {
            "eligible_target_count": [500, 520, 480, 510],
            "missed_target_rate": [0.50, 0.52, 0.49, 0.51],
            "due_date_coverage": [0.90, 0.91, 0.90, 0.92],
            "closed_date_coverage": [0.95, 0.96, 0.95, 0.96],
        }
    )

    stability = summarize_monthly_stability(monthly, expected_month_count=4)

    assert stability["target_rate_volatility_flag"] is False
    assert 0 <= stability["target_rate_range"] <= 1


def test_monthly_stability_handles_empty_and_single_row_input() -> None:
    """An empty or single-month frame returns finite zero-volatility metrics."""
    expected_columns = [
        "eligible_target_count", "missed_target_rate", "due_date_coverage", "closed_date_coverage",
    ]
    empty = summarize_monthly_stability(pd.DataFrame(columns=expected_columns), expected_month_count=3)
    assert empty["missing_month_count"] == 3
    assert empty["target_rate_range"] == 0.0
    assert empty["monthly_volume_coefficient_of_variation"] == 0.0
    assert empty["target_rate_volatility_flag"] is False

    single_row = pd.DataFrame(
        {
            "eligible_target_count": [100],
            "missed_target_rate": [0.4],
            "due_date_coverage": [0.9],
            "closed_date_coverage": [0.95],
        }
    )
    single = summarize_monthly_stability(single_row, expected_month_count=1)
    assert single["missing_month_count"] == 0
    assert single["largest_month_to_month_target_rate_change"] == 0.0


def test_data_gates_with_semantic_limitations_are_conditionally_approved() -> None:
    """Passing data gates preserve unresolved target governance in status."""
    assert (
        determine_scope_decision_status(
            True,
            semantic_limitations_remain=True,
        )
        == "Approved with limitations"
    )
