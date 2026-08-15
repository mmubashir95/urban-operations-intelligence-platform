"""Tests for deterministic feature and train-only transformation priorities."""

from urban_ops.eda.pipeline import run_split_aware_eda
from urban_ops.eda.recommendations import assign_temporal_eda_status


EXPECTED_TEMPORAL_EDA_STATUS = {
    "created_date": "REVIEW",
    "created_hour": "CANDIDATE",
    "created_day_of_week": "CANDIDATE",
    "created_day_name": "ALTERNATIVE_REPRESENTATION",
    "created_day_of_month": "REVIEW",
    "created_week_of_year": "REVIEW_REDUNDANCY",
    "created_month": "CANDIDATE",
    "created_month_name": "ALTERNATIVE_REPRESENTATION",
    "created_quarter": "REVIEW_REDUNDANCY",
    "created_year": "CONDITIONAL",
    "is_weekend": "CANDIDATE",
}


def test_recommendation_priority_blocks_leakage_all_null_and_zero_variance(eda_fixture) -> None:
    table = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "baseline_feature_recommendation.csv"
    ].set_index("feature_name")
    assert table.loc["closed_date", "baseline_decision"] == "EXCLUDE_LEAKAGE"
    assert table.loc["descriptor_2", "baseline_decision"] == "EXCLUDE_ALL_NULL"
    assert table.loc["agency", "baseline_decision"] == "EXCLUDE_ZERO_VARIANCE"
    assert table.loc["created_hour", "baseline_decision"] == "INCLUDE"
    assert table.loc["incident_zip", "baseline_decision"] == "CONDITIONAL"


def test_transformations_are_recommendations_fit_on_train_only(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    transformations = tables["transformation_recommendation.csv"]
    assert transformations["fit_on"].eq("train").all()
    assert transformations["apply_to"].eq("validation,test").all()
    assert transformations["recommended_unknown_category_policy"].isin({"MAP_TO___UNKNOWN__", "NOT_APPLICABLE"}).all()
    outliers = tables["outlier_recommendation.csv"]
    assert not outliers["recommended_action"].str.contains("REMOVE|DELETE|CLIP$", regex=True).any()


def test_temporal_eda_status_framework_is_deterministic(eda_fixture) -> None:
    table = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "baseline_feature_recommendation.csv"
    ].set_index("feature_name")
    for feature_name, expected_status in EXPECTED_TEMPORAL_EDA_STATUS.items():
        row = table.loc[feature_name]
        if bool(row["zero_variance"]) or bool(row["all_null"]):
            assert row["eda_status"] == "EXCLUDE"
        else:
            assert row["eda_status"] == expected_status
    # Preserve existing handoff semantics for preferred temporal candidates.
    assert table.loc["created_hour", "baseline_decision"] == "INCLUDE"
    assert table.loc["created_date", "baseline_decision"] == "REVIEW"
    if not bool(table.loc["created_year", "zero_variance"]):
        assert table.loc["created_year", "eda_status"] == "CONDITIONAL"
    if not bool(table.loc["created_day_of_month", "zero_variance"]):
        assert table.loc["created_day_of_month", "eda_status"] == "REVIEW"
    candidate_features = {
        feature_name
        for feature_name, expected_status in EXPECTED_TEMPORAL_EDA_STATUS.items()
        if expected_status == "CANDIDATE"
        and not bool(table.loc[feature_name, "zero_variance"])
        and not bool(table.loc[feature_name, "all_null"])
    }
    observed_candidates = set(
        table.loc[
            table["eda_status"].eq("CANDIDATE")
            & table.index.isin(EXPECTED_TEMPORAL_EDA_STATUS),
            :,
        ].index
    )
    assert observed_candidates == candidate_features


def test_assign_temporal_eda_status_priority_excludes_before_candidate() -> None:
    excluded = assign_temporal_eda_status(
        "created_hour",
        prediction_time_available=True,
        leakage_status="SAFE",
        all_null=False,
        zero_variance=True,
        is_temporal_feature=True,
    )
    assert excluded["eda_status"] == "EXCLUDE"
    alternative = assign_temporal_eda_status(
        "created_day_name",
        prediction_time_available=True,
        leakage_status="SAFE",
        all_null=False,
        zero_variance=False,
        is_temporal_feature=True,
    )
    assert alternative["eda_status"] == "ALTERNATIVE_REPRESENTATION"
    redundant = assign_temporal_eda_status(
        "created_week_of_year",
        prediction_time_available=True,
        leakage_status="SAFE",
        all_null=False,
        zero_variance=False,
        is_temporal_feature=True,
    )
    assert redundant["eda_status"] == "REVIEW_REDUNDANCY"
    year = assign_temporal_eda_status(
        "created_year",
        prediction_time_available=True,
        leakage_status="SAFE",
        all_null=False,
        zero_variance=False,
        is_temporal_feature=True,
    )
    assert year["eda_status"] == "CONDITIONAL"
