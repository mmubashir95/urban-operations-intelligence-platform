"""Tests for numeric summaries, percentiles, and target segmentation."""

from urban_ops.eda.pipeline import run_split_aware_eda


def test_numeric_summaries_include_configured_statistics(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    summary = tables["numeric_feature_summary.csv"]
    assert {"p01", "p05", "p25", "median", "p75", "p95", "p99", "iqr"} <= set(summary.columns)
    assert set(summary["feature_name"]) == {"latitude", "longitude", "created_day_of_month", "created_week_of_year"}
    assert set(tables["numeric_summary_by_target.csv"]["target_value"]) == {0, 1}
