"""Tests for restricted structural drift disclosure."""

from urban_ops.eda.pipeline import run_split_aware_eda


def test_target_missingness_unseen_and_numeric_drift_are_reported(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    assert set(tables["split_target_comparison.csv"]["split_name"]) == {"train", "validation", "test"}
    assert set(tables["numeric_range_drift.csv"]["split_name"]) == {"train", "validation", "test"}
    summary = tables["structural_drift_summary.csv"]
    assert {"target", "missingness", "unseen_category", "numeric_range"} <= set(summary["area"])
    assert not summary["warning_status"].eq("FAIL").any()
    assert summary.loc[summary["area"].eq("numeric_range"), "interpretation"].str.contains(
        "without a hard rejection threshold"
    ).all()


def test_test_evidence_does_not_change_train_feature_decisions(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    decisions = result.tables["baseline_feature_recommendation.csv"].set_index("feature_name")
    assert decisions.loc["created_hour", "baseline_decision"] == "INCLUDE"
    # Later unseen values cannot promote a train-zero-variance feature.
    assert decisions.loc["location_type", "baseline_decision"] == "EXCLUDE_ZERO_VARIANCE"
