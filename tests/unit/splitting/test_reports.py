"""Tests for complete, deterministic, reconciled Step 8 report artifacts."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from urban_ops.splitting.pipeline import run_time_based_split
from urban_ops.splitting.reports import REQUIRED_REPORT_TABLES
from tests.unit.splitting.conftest import build_split_fixture


def test_all_required_reports_written_and_summary_references_source_and_split(
    tmp_path: Path, eligible_frame
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    result = run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert all((fixture.reports / "tables" / name).is_file() for name in REQUIRED_REPORT_TABLES)
    summary = (fixture.reports / "split_summary.md").read_text()
    assert result.source.metadata.cleaning_run_id in summary
    assert result.metadata.split_id in result.paths.metadata.read_text()


def test_report_counts_targets_dates_overlaps_and_reconciliation(
    tmp_path: Path, eligible_frame
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    result = run_time_based_split(config_path=fixture.config, dry_run=True)
    tables = result.tables
    assert tables["monthly_target_distribution.csv"]["row_count"].sum() == len(eligible_frame)
    assert tables["split_row_counts.csv"]["row_count"].sum() == len(eligible_frame)
    assert tables["split_target_distribution.csv"]["total_count"].sum() == len(eligible_frame)
    assert tables["identifier_overlap_checks.csv"]["overlap_count"].eq(0).all()
    assert tables["output_reconciliation.csv"]["status"].eq("PASS").all()
    assert tables["selected_split_boundaries.csv"].shape[0] == 3


def test_candidate_and_drift_reports_are_deterministic(tmp_path: Path, eligible_frame) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    first = run_time_based_split(config_path=fixture.config, dry_run=True)
    second = run_time_based_split(config_path=fixture.config, dry_run=True)
    pd.testing.assert_frame_equal(
        first.tables["candidate_split_boundaries.csv"],
        second.tables["candidate_split_boundaries.csv"],
    )
    pd.testing.assert_frame_equal(
        first.tables["temporal_drift_summary.csv"],
        second.tables["temporal_drift_summary.csv"],
    )
