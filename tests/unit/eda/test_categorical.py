"""Tests for cardinality, rare groups, ZIP strings, and later unknowns."""

from urban_ops.eda.pipeline import run_split_aware_eda


def test_cardinality_profiles_preserve_unspecified_and_zip_as_string(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    profiles = result.tables["categorical_feature_profile.csv"]
    borough = profiles.loc[profiles["feature_name"].eq("borough"), "category_value"]
    assert "Unspecified" in set(borough)
    inventory = result.tables["column_inventory.csv"].set_index("column_name")
    assert "string" in inventory.loc["incident_zip", "source_dtype"]


def test_rare_thresholds_and_unseen_categories_are_deterministic(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    rare = tables["rare_category_analysis.csv"]
    assert set(rare["threshold_type"]) == {"count", "share"}
    unseen = tables["unseen_categories.csv"].set_index(["feature_name", "comparison_split"])
    assert unseen.loc[("location_type", "validation"), "rows_affected"] > 0
    assert unseen.loc[("location_type", "test"), "rows_affected"] > 0
    assert unseen["recommended_unknown_policy"].eq("TRAIN_FITTED_UNKNOWN_SAFE_ENCODING").all()
