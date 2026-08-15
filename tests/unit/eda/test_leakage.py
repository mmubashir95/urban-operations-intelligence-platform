"""Tests proving Step 9A reuses the Step 4 prediction-time authority."""

from urban_ops.eda.pipeline import run_split_aware_eda


def test_target_outcome_status_and_eligibility_fields_are_blocked(eda_fixture) -> None:
    audit = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "leakage_audit.csv"
    ].set_index("column_name")
    for feature in (
        "missed_resolution_target", "closed_date", "resolution_description",
        "resolution_action_updated_date", "status", "target_eligible",
        "primary_exclusion_reason", "outcome_mature", "valid_closed_chronology",
    ):
        assert audit.loc[feature, "leakage_status"] == "BLOCKED"


def test_due_date_and_geography_remain_conditional_while_temporal_is_safe(eda_fixture) -> None:
    audit = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables[
        "leakage_audit.csv"
    ].set_index("column_name")
    assert audit.loc["due_date", "leakage_status"] == "BLOCKED"
    for feature in ("location_type", "incident_zip", "latitude", "longitude"):
        assert audit.loc[feature, "leakage_status"] == "CONDITIONAL"
    assert audit.loc["created_hour", "leakage_status"] == "SAFE"
