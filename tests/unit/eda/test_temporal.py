"""Tests for creation-time derivation and ordered temporal evidence."""

import pandas as pd

from urban_ops.eda.analysis import derive_temporal_features
from urban_ops.eda.pipeline import run_split_aware_eda


def test_temporal_derivation_domains_and_no_input_mutation(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    before = result.source.train.copy(deep=True)
    derived = derive_temporal_features(result.source.train, "created_date")
    pd.testing.assert_frame_equal(result.source.train, before)
    assert derived["created_hour"].between(0, 23).all()
    assert derived["created_week_of_year"].between(1, 53).all()
    assert set(derived["is_weekend"].unique()) <= {True, False}


def test_temporal_tables_include_missing_period_domains_and_extrema(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    assert tables["monthly_volume.csv"]["month"].tolist() == ["2024-01", "2024-02", "2024-03"]
    assert tables["hour_of_day_target_rate.csv"]["created_hour"].tolist() == list(range(24))
    assert tables["day_of_week_target_rate.csv"]["created_day_name"].tolist()[0] == "Monday"
    assert len(tables["temporal_outlier_analysis.csv"]) == 6
