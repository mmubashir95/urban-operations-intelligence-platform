"""Tests for train target balance and non-ML baseline evidence."""

import pytest

from urban_ops.eda.pipeline import run_split_aware_eda


def test_train_target_counts_rates_and_baselines_reconcile(eda_fixture) -> None:
    table = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "train_target_distribution.csv"
    ].iloc[0]
    assert table["on_time_count"] == 3
    assert table["missed_count"] == 3
    assert table["missed_target_rate"] == pytest.approx(0.5)
    assert table["majority_class_accuracy"] == pytest.approx(0.5)
    assert table["constant_probability_baseline"] == pytest.approx(0.5)
    assert "NOT_AUTOMATIC" in table["class_weighting_recommendation"]
