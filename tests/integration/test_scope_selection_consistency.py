"""Regression tests for Notebook 04 and Notebook 05 scope reconciliation."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_ops.analysis.final_scope_selection import (
    reconcile_final_scope,
    validate_selected_scope_outputs,
)


def general_candidates() -> pd.DataFrame:
    """Return deterministic general feasibility candidates."""
    return pd.DataFrame(
        [
            {
                "agency": "DSNY",
                "start_date": "2022-01-01",
                "end_date": "2025-12-31",
                "total_records": 160,
                "eligible_target_count": 140,
                "missed_target_rate": 0.60,
                "due_date_coverage": 0.90,
                "closed_date_coverage": 0.99,
            },
            {
                "agency": "DSNY",
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "total_records": 100,
                "eligible_target_count": 90,
                "missed_target_rate": 0.44,
                "due_date_coverage": 0.91,
                "closed_date_coverage": 0.995,
            },
        ]
    )


def temporal_candidates() -> pd.DataFrame:
    """Return deterministic temporal candidates with one selected row."""
    return pd.DataFrame(
        [
            {
                "start_date": "2022-01-01",
                "end_date": "2025-12-31",
                "total_records": 160,
                "eligible_records": 140,
                "missed_target_rate": 0.60,
                "due_date_coverage": 0.90,
                "closed_date_coverage": 0.99,
                "selected": False,
            },
            {
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "total_records": 100,
                "eligible_records": 90,
                "missed_target_rate": 0.44,
                "due_date_coverage": 0.91,
                "closed_date_coverage": 0.995,
                "selected": True,
            },
        ]
    )


def selected_summary() -> pd.DataFrame:
    """Return the deterministic selected-scope artifact contract."""
    return pd.DataFrame(
        [
            {
                "selected_agency": "DSNY",
                "selected_complaint_type": "Graffiti",
                "selected_start_date": "2024-01-01",
                "selected_end_date": "2025-12-31",
                "total_records": 100,
                "eligible_records": 90,
                "missed_target_rate": 0.44,
                "due_date_coverage": 0.91,
                "closed_date_coverage": 0.995,
                "extraction_timestamp": "2026-07-27T16:11:48+00:00",
            }
        ]
    )


def test_temporal_selection_reconciles_to_one_general_candidate() -> None:
    candidate, summary = reconcile_final_scope(
        general_candidates(),
        temporal_candidates(),
        selected_summary(),
        selected_complaint_types=["Graffiti"],
    )

    assert candidate["start_date"] == summary["selected_start_date"]
    assert candidate["end_date"] == summary["selected_end_date"]


def test_multiple_temporal_selected_rows_are_rejected() -> None:
    candidates = temporal_candidates()
    candidates["selected"] = True

    with pytest.raises(ValueError, match="exactly one selected row"):
        reconcile_final_scope(
            general_candidates(),
            candidates,
            selected_summary(),
            selected_complaint_types=["Graffiti"],
        )


def test_temporal_dates_must_match_exactly_one_general_candidate() -> None:
    summary = selected_summary()
    summary.loc[0, "selected_start_date"] = "2023-01-01"
    candidates = temporal_candidates()
    candidates.loc[candidates["selected"], "start_date"] = "2023-01-01"

    with pytest.raises(ValueError, match="exactly one candidate"):
        reconcile_final_scope(
            general_candidates(),
            candidates,
            summary,
            selected_complaint_types=["Graffiti"],
        )


def test_scope_identifiers_must_agree() -> None:
    with pytest.raises(ValueError, match="complaint type disagrees"):
        reconcile_final_scope(
            general_candidates(),
            temporal_candidates(),
            selected_summary(),
            selected_complaint_types=["Missed Collection"],
        )


def test_temporal_candidate_metrics_must_match_selected_summary() -> None:
    candidates = temporal_candidates()
    candidates.loc[candidates["selected"], "eligible_records"] = 89

    with pytest.raises(ValueError, match="disagrees with its summary"):
        reconcile_final_scope(
            general_candidates(),
            candidates,
            selected_summary(),
            selected_complaint_types=["Graffiti"],
        )


def test_report_outputs_compare_counts_rates_and_snapshot() -> None:
    report_04 = selected_summary()
    report_05 = selected_summary()
    report_04.loc[0, "missed_target_rate"] += 1e-12

    validate_selected_scope_outputs(report_04, report_05)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("selected_end_date", "2026-06-30"),
        ("total_records", 101),
        ("eligible_records", 89),
        ("due_date_coverage", 0.80),
        ("closed_date_coverage", 0.90),
        ("extraction_timestamp", "2026-07-28T00:00:00+00:00"),
    ],
)
def test_report_output_mismatch_is_rejected(column: str, replacement: object) -> None:
    report_04 = selected_summary()
    report_05 = selected_summary()
    report_04.loc[0, column] = replacement

    with pytest.raises(ValueError, match="mismatch"):
        validate_selected_scope_outputs(report_04, report_05)
