"""Enforce the complaint-creation feature leakage boundary."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from urban_ops.features.feature_roles import (
    FIELD_ROLES, FeatureRole, feature_role_inventory,
)


class FeaturePolicyError(ValueError):
    """Raised when requested model fields violate feature governance."""


def validate_feature_columns(
    feature_columns: Sequence[str],
    *,
    approved_conditional_features: set[str] | None = None,
) -> None:
    """Reject unknown, blocked, or unapproved conditional model fields."""
    approved = approved_conditional_features or set()
    violations: list[str] = []
    for column in feature_columns:
        item = FIELD_ROLES.get(column)
        if item is None:
            violations.append(f"{column}: unknown field")
        elif item.role is FeatureRole.SAFE_FEATURE and item.allowed_for_baseline:
            continue
        elif item.role is FeatureRole.CONDITIONAL_FEATURE and column in approved:
            continue
        elif item.role is FeatureRole.CONDITIONAL_FEATURE:
            violations.append(f"{column}: conditional feature requires explicit approval")
        else:
            violations.append(f"{column}: blocked {item.role.value} field")
    if violations:
        raise FeaturePolicyError("Feature policy violations: " + "; ".join(violations))


def build_leakage_audit(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return configured decisions, representing unknown proposals explicitly."""
    inventory = feature_role_inventory()
    inventory["column_name"] = inventory["source_column"]
    inventory["leakage_status"] = inventory["decision_status"].map(
        {"APPROVED": "SAFE", "CONDITIONAL": "CONDITIONAL"}
    ).fillna("BLOCKED")
    inventory["decision_reason"] = inventory["leakage_reason"]
    indexed = inventory.set_index("column_name", drop=False)
    requested = list(columns) if columns is not None else inventory["column_name"].tolist()
    rows = []
    for column in requested:
        if column in indexed.index:
            rows.append(indexed.loc[column].to_dict())
        else:
            rows.append({
                "column_name": column, "source_column": column, "role": "UNKNOWN",
                "available_at_creation": None, "mutable_after_creation": None,
                "allowed_for_baseline": False, "leakage_status": "BLOCKED",
                "decision_reason": "Unknown fields are rejected pending governance.",
                "decision_status": "UNKNOWN",
            })
    return pd.DataFrame(rows).reset_index(drop=True)
