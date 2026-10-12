"""Synthetic integration coverage for the complete Phase 7.2 report workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from urban_ops.analysis import resolution_duration_workflow as workflow
from urban_ops.analysis.resolution_duration_reporting import (
    ELIGIBILITY_FILENAME,
    HISTOGRAM_FILENAME,
    LOG_HISTOGRAM_FILENAME,
    REPORT_FILENAME,
    STATISTICS_FILENAME,
)
from urban_ops.analysis.resolution_duration_peak_reporting import (
    PEAK_FIGURE_FILENAME,
    PEAK_INTERVALS_FILENAME,
    PEAK_PATTERNS_FILENAME,
    PEAK_REPORT_FILENAME,
    PEAK_TYPES_FILENAME,
)
from urban_ops.splitting.models import SplitBoundaries


def test_workflow_writes_deterministic_reports_from_synthetic_source(
    tmp_path, monkeypatch
) -> None:
    boundaries = SplitBoundaries(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    )
    config = SimpleNamespace(
        processed_root=tmp_path / "processed",
        latest_pointer=tmp_path / "latest.json",
        timestamp_column="created_date",
        identifier_column="unique_key",
        boundaries=boundaries,
    )
    frame = pd.DataFrame({
        "unique_key": ["a", "b", "c", "peak", "d", "later"],
        "complaint_type": ["Type A", "Type A", "Type B", "Type B", "Type A", "Type A"],
        "created_date": [
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z", "2024-01-10T00:00:00Z",
            "2024-03-20T00:00:00Z",
            "2024-04-01T00:00:00Z",
        ],
        "closed_date": [
            "2024-01-01T00:00:00Z", "2024-01-02T00:20:00Z",
            "2024-01-05T00:00:00Z", "2024-02-16T12:00:00Z",
            "2024-04-15T00:00:00Z",
            "2024-04-01T02:00:00Z",
        ],
        "status": ["Closed"] * 6,
    })
    source_path = tmp_path / "cleaned_service_requests.parquet"
    monkeypatch.setattr(workflow, "load_split_config", lambda _: config)
    monkeypatch.setattr(workflow, "resolve_cleaning_run", lambda **_: tmp_path)
    monkeypatch.setattr(
        workflow, "load_verified_cleaned_source", lambda _: (frame, source_path)
    )
    report_root = tmp_path / "reports" / "month_2"

    result = workflow.run_resolution_duration_analysis(
        split_config_path=tmp_path / "splits.yaml", report_root=report_root
    )

    expected = {
        report_root / REPORT_FILENAME,
        report_root / STATISTICS_FILENAME,
        report_root / ELIGIBILITY_FILENAME,
        report_root / "figures" / HISTOGRAM_FILENAME,
        report_root / "figures" / LOG_HISTOGRAM_FILENAME,
    }
    assert expected == {
        result.report_path, result.statistics_path, result.eligibility_path,
        *result.figure_paths,
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
    statistics_text = result.statistics_path.read_text(encoding="utf-8")
    assert statistics_text.startswith("metric,value,unit\n")
    # Fractional hours survive CSV serialization; Markdown alone is rounded.
    assert "24.166666666666668" in statistics_text
    statistics = pd.read_csv(result.statistics_path)
    assert tuple(statistics.columns) == ("metric", "value", "unit")
    eligibility = pd.read_csv(result.eligibility_path)
    assert tuple(eligibility.columns) == (
        "summary_type", "name", "count", "percentage"
    )
    report = result.report_path.read_text(encoding="utf-8")
    assert "[2024-01-01T00:00:00+00:00, 2024-04-01T00:00:00+00:00)" in report
    assert "resolution_not_observed_by_cutoff" in report
    assert "Survival modelling remains a Phase 7.6\ndecision" in report
    assert report.count("# Phase 7.2") == 1

    original_statistics = result.statistics.copy(deep=True)
    peak_root = tmp_path / "peak_reports"
    peak_result = workflow.run_900_hour_investigation(
        split_config_path=tmp_path / "splits.yaml", report_root=peak_root
    )
    expected_peak = {
        peak_root / PEAK_REPORT_FILENAME,
        peak_root / PEAK_INTERVALS_FILENAME,
        peak_root / PEAK_TYPES_FILENAME,
        peak_root / PEAK_PATTERNS_FILENAME,
        peak_root / "figures" / PEAK_FIGURE_FILENAME,
    }
    assert expected_peak == {
        peak_result.report_path,
        peak_result.intervals_path,
        peak_result.complaint_types_path,
        peak_result.closure_patterns_path,
        peak_result.figure_path,
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_peak)
    pd.testing.assert_frame_equal(result.statistics, original_statistics)
    assert len(peak_result.evidence.eligible) == 4
    assert peak_result.evidence.peak["unique_key"].tolist() == ["peak"]
    peak_report = peak_result.report_path.read_text(encoding="utf-8")
    assert "## Observed findings" in peak_report
    assert "## Possible explanations requiring more evidence" in peak_report
