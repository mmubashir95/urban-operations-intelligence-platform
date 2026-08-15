"""Tests for descriptive missingness without imputation or mutation."""

from urban_ops.eda.analysis import missingness_band
from urban_ops.eda.pipeline import load_eda_config, run_split_aware_eda


def test_missingness_bands_and_all_null_detection(eda_fixture) -> None:
    config = load_eda_config(eda_fixture.config)
    assert missingness_band(0.0, config.missingness_bands) == "COMPLETE"
    assert missingness_band(1.0, config.missingness_bands) == "ALL_NULL"
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    missing = tables["train_missingness.csv"].set_index("column_name")
    assert missing.loc["descriptor_2", "missingness_band"] == "ALL_NULL"
    assert set(tables["missingness_by_target.csv"]["target_value"]) == {0, 1}
    assert set(tables["missingness_by_month.csv"]["month"]) == {"2024-01", "2024-02", "2024-03"}


def test_split_missingness_is_disclosure_not_policy(eda_fixture) -> None:
    table = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "split_missingness_comparison.csv"
    ]
    assert table["interpretation"].str.contains("train governs policy", case=False).all()
