"""Tests for creation-time leakage enforcement."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.features.feature_roles import FIELD_ROLES, FeatureRole
from urban_ops.features.leakage import FeaturePolicyError, build_leakage_audit, validate_feature_columns


@pytest.mark.parametrize("column", [
    "closed_date", "missed_resolution_target", "target_eligible",
    "primary_exclusion_reason", "resolution_description",
    "resolution_action_updated_date", "due_date", "full_period_target_rate",
    "unique_key", "unknown_feature",
])
def test_blocked_fields_are_rejected(column: str) -> None:
    with pytest.raises(FeaturePolicyError, match=column):
        validate_feature_columns([column])


def test_safe_and_conditional_features() -> None:
    validate_feature_columns(["created_date", "agency", "open_data_channel_type"])
    with pytest.raises(FeaturePolicyError, match="explicit approval"):
        validate_feature_columns(["borough"])
    validate_feature_columns(["borough"], approved_conditional_features={"borough"})


def test_mixed_list_fails_and_due_cannot_be_conditionally_approved() -> None:
    with pytest.raises(FeaturePolicyError, match="closed_date"):
        validate_feature_columns(["agency", "closed_date"])
    with pytest.raises(FeaturePolicyError, match="TARGET_INPUT"):
        validate_feature_columns(["due_date"], approved_conditional_features={"due_date"})


def test_audit_covers_inventory_and_no_blocked_field_is_safe() -> None:
    audit = build_leakage_audit()
    assert set(audit["column_name"]) == set(FIELD_ROLES)
    blocked = {
        FeatureRole.IDENTIFIER.value, FeatureRole.TARGET_INPUT.value,
        FeatureRole.POST_CREATION_FIELD.value, FeatureRole.TARGET_DERIVED.value,
        FeatureRole.EXCLUDED.value,
    }
    assert not audit.loc[audit["role"].isin(blocked), "allowed_for_baseline"].any()
