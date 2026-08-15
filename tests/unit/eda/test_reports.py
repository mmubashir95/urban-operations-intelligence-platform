"""Tests for stable EDA report inventory, validation, and deterministic content."""

import pandas as pd
import pytest

from urban_ops.eda.pipeline import run_split_aware_eda
from urban_ops.eda.reports import REQUIRED_TABLES, validate_eda_report


def test_all_tables_summary_and_source_identity_are_published(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config)
    assert all((eda_fixture.reports / "tables" / name).is_file() for name in REQUIRED_TABLES)
    summary = (eda_fixture.reports / "eda_summary.md").read_text()
    assert result.source.split_id in summary
    assert result.source.artifacts["train"].sha256 in summary
    validate_eda_report(eda_fixture.reports, result.source)


def test_report_validation_rejects_corrupt_recommendation_schema(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config)
    path = eda_fixture.reports / "tables" / "baseline_feature_recommendation.csv"
    table = pd.read_csv(path).drop(columns="baseline_decision")
    table.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="schema"):
        validate_eda_report(eda_fixture.reports, result.source)


def test_repeated_dry_runs_have_deterministic_analysis_tables(eda_fixture) -> None:
    first = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    second = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    for name in REQUIRED_TABLES:
        if name not in {"source_split_verification.csv"}:
            pd.testing.assert_frame_equal(first.tables[name], second.tables[name])
