"""Build and enforce a baseline-candidate audit using Step 4 feature governance."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from urban_ops.features.feature_roles import FeatureRole
from urban_ops.features.leakage import build_leakage_audit, validate_feature_columns


LEAKAGE_COLUMNS = [
    "column_name", "feature_role", "cleaning_status", "missingness_status",
    "variance_status", "baseline_allowed", "leakage_status", "decision_reason",
]


def build_cleaning_leakage_audit(
    frame: pd.DataFrame,
    *,
    configured_all_null_columns: Collection[str],
    configured_zero_variance_columns: Collection[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return enforced baseline decisions plus observed unusable field lists."""
    all_null = sorted(column for column in frame if frame[column].isna().all())
    zero_variance = sorted(
        column for column in frame
        if frame[column].nunique(dropna=False) <= 1 and column not in all_null
    )
    missing_configured_all_null = sorted(set(configured_all_null_columns) - set(all_null))
    missing_configured_zero = sorted(set(configured_zero_variance_columns) - set(zero_variance))
    if missing_configured_all_null:
        raise ValueError(
            f"Configured all-null columns are no longer all-null: {missing_configured_all_null}"
        )
    if missing_configured_zero:
        raise ValueError(
            "Configured zero-variance columns now vary: "
            f"{missing_configured_zero}"
        )
    governed = build_leakage_audit(list(frame.columns)).set_index("column_name")
    timestamp_columns = {
        "created_date", "closed_date", "due_date", "resolution_action_updated_date"
    }
    rows: list[dict[str, object]] = []
    candidates: list[str] = []
    for column in frame.columns:
        item = governed.loc[column]
        role = str(item["role"])
        leakage_status = str(item["leakage_status"])
        unusable = column in all_null or column in zero_variance
        baseline_allowed = (
            role == FeatureRole.SAFE_FEATURE.value
            and leakage_status == "SAFE"
            and not unusable
        )
        if baseline_allowed:
            candidates.append(column)
        reason = str(item["decision_reason"])
        if column in all_null:
            reason = "All-null after approved cleaning; unusable for the baseline."
        elif column in zero_variance:
            reason = "Zero-variance after approved cleaning; unusable for the baseline."
        rows.append({
            "column_name": column,
            "feature_role": role,
            "cleaning_status": (
                "typed_utc_timestamp" if column in timestamp_columns
                else "derived_audit_field" if role == "UNKNOWN" or role == FeatureRole.TARGET_DERIVED.value
                else "preserved_or_approved_category_cleaning"
            ),
            "missingness_status": "HAS_MISSING" if frame[column].isna().any() else "COMPLETE",
            "variance_status": (
                "ALL_NULL" if column in all_null else "ZERO_VARIANCE"
                if column in zero_variance else "VARIES"
            ),
            "baseline_allowed": baseline_allowed,
            "leakage_status": leakage_status,
            "decision_reason": reason,
        })
    validate_feature_columns(candidates)
    table = pd.DataFrame(rows, columns=LEAKAGE_COLUMNS)
    if table.loc[table["leakage_status"].eq("BLOCKED"), "baseline_allowed"].any():
        raise RuntimeError("A leakage-blocked field was marked baseline-allowed.")
    return table, all_null, zero_variance
