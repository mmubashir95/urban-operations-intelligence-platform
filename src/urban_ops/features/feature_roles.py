"""Central creation-time role inventory for NYC 311 fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final

import pandas as pd

from urban_ops.data.nyc_311_config import SELECTED_COLUMNS


class FeatureRole(StrEnum):
    """Governed roles used to decide whether a field can enter a model."""

    IDENTIFIER = "IDENTIFIER"
    SAFE_FEATURE = "SAFE_FEATURE"
    CONDITIONAL_FEATURE = "CONDITIONAL_FEATURE"
    TARGET_INPUT = "TARGET_INPUT"
    POST_CREATION_FIELD = "POST_CREATION_FIELD"
    TARGET_DERIVED = "TARGET_DERIVED"
    METADATA = "METADATA"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class FieldRoleDefinition:
    """Prediction-time governance for one field."""

    source_column: str
    internal_feature_name: str
    data_type: str
    role: FeatureRole
    available_at_creation: bool | None
    mutable_after_creation: bool | None
    allowed_for_baseline: bool
    null_policy: str
    leakage_reason: str
    decision_status: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a report-friendly mapping."""
        result = asdict(self)
        result["role"] = self.role.value
        return result


SAFE = {"agency", "agency_name", "complaint_type", "open_data_channel_type"}
POST_CREATION = {
    "closed_date", "status", "final_status", "resolution_description",
    "resolution_action_updated_date", "actual_resolution_duration",
    "actual_resolution_duration_hours",
}
TARGET_DERIVED = {
    "missed_resolution_target", "target_eligible", "outcome_mature",
    "primary_exclusion_reason", "is_exact_duplicate", "is_conflicting_duplicate",
}
FUTURE_AGGREGATES = {
    "full_period_target_rate", "future_complaint_volume",
    "future_category_miss_rate", "post_period_aggregate",
    "validation_target_rate", "test_target_rate",
}


def _definition(name: str) -> FieldRoleDefinition:
    """Build the governed definition for a configured field."""
    common = {
        "source_column": name,
        "internal_feature_name": name,
        "data_type": "datetime64[ns, UTC]" if "date" in name else "string",
        "null_policy": "preserve and handle explicitly",
        "notes": "",
    }
    if name == "unique_key":
        return FieldRoleDefinition(**common, role=FeatureRole.IDENTIFIER,
            available_at_creation=True, mutable_after_creation=False,
            allowed_for_baseline=False, decision_status="BLOCKED",
            leakage_reason="Identifier invites memorisation and is not a model feature.")
    if name == "created_date":
        return FieldRoleDefinition(**common, role=FeatureRole.SAFE_FEATURE,
            available_at_creation=True, mutable_after_creation=False,
            allowed_for_baseline=True, decision_status="APPROVED",
            leakage_reason="Available at creation; prefer derived calendar features.")
    if name in {"due_date", "closed_date"}:
        return FieldRoleDefinition(**common, role=FeatureRole.TARGET_INPUT,
            available_at_creation=None if name == "due_date" else False,
            mutable_after_creation=None if name == "due_date" else True,
            allowed_for_baseline=False, decision_status="NOT_APPROVED",
            leakage_reason=(
                "Target input whose creation-time timing and mutability are unproven."
                if name == "due_date"
                else "Closure outcome is unavailable at complaint creation."
            ))
    if name in SAFE:
        return FieldRoleDefinition(**common, role=FeatureRole.SAFE_FEATURE,
            available_at_creation=True, mutable_after_creation=False,
            allowed_for_baseline=True, decision_status="APPROVED",
            leakage_reason="Available at complaint creation.")
    if name in POST_CREATION:
        return FieldRoleDefinition(**common, role=FeatureRole.POST_CREATION_FIELD,
            available_at_creation=False, mutable_after_creation=True,
            allowed_for_baseline=False, decision_status="BLOCKED",
            leakage_reason="Final outcome or post-creation operational update.")
    if name in TARGET_DERIVED:
        return FieldRoleDefinition(**common, role=FeatureRole.TARGET_DERIVED,
            available_at_creation=False, mutable_after_creation=False,
            allowed_for_baseline=False, decision_status="BLOCKED",
            leakage_reason="Label or outcome-derived governance field.")
    if name in FUTURE_AGGREGATES:
        return FieldRoleDefinition(**common, role=FeatureRole.EXCLUDED,
            available_at_creation=False, mutable_after_creation=False,
            allowed_for_baseline=False, decision_status="BLOCKED",
            leakage_reason="Uses future, full-period, validation, or test outcomes.")
    return FieldRoleDefinition(**common, role=FeatureRole.CONDITIONAL_FEATURE,
        available_at_creation=None, mutable_after_creation=None,
        allowed_for_baseline=False, decision_status="CONDITIONAL",
        leakage_reason="Intake-time availability or later correction is unproven.")


INVENTORY_COLUMNS: Final = tuple(dict.fromkeys([
    *SELECTED_COLUMNS, "final_status", "actual_resolution_duration",
    "actual_resolution_duration_hours", *sorted(TARGET_DERIVED),
    *sorted(FUTURE_AGGREGATES),
]))
FIELD_ROLE_INVENTORY: Final = tuple(_definition(name) for name in INVENTORY_COLUMNS)
FIELD_ROLES: Final = {item.source_column: item for item in FIELD_ROLE_INVENTORY}


def feature_role_inventory() -> pd.DataFrame:
    """Return a copy of the central field-role inventory."""
    result = pd.DataFrame(item.to_dict() for item in FIELD_ROLE_INVENTORY)
    result["creation_time_available"] = result["available_at_creation"]
    return result
