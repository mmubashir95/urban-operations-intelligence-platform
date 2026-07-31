"""Profile missing, blank, whitespace-only, and null-like raw values."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from urban_ops.features.feature_roles import FIELD_ROLES, FeatureRole


def _role(column: str) -> str:
    """Return the Step 4 governed role or a conservative unknown role."""
    definition = FIELD_ROLES.get(column)
    return definition.role.value if definition else "UNKNOWN"


def _missingness_policy(column: str, role: str, null_count: int) -> tuple[str, str]:
    """Return consistent severity and Step 7 recommendation for one field."""
    if null_count == 0:
        return "INFO", "No missing-value action required."
    if column in {"unique_key", "created_date"}:
        return "CRITICAL", "Quarantine affected rows; do not synthesize identifiers or creation time."
    if column == "due_date":
        return "ERROR", "Preserve rows and mark target-ineligible; do not impute due dates."
    if column == "closed_date":
        return "WARNING", "Preserve rows and mark target-ineligible; do not impute closure time."
    if column in {"latitude", "longitude", "incident_zip"}:
        return "WARNING", "Preserve complaint; handle the geographic feature as missing."
    if role == "SCOPE_FIELD":
        return "CRITICAL", "Quarantine rows whose scope cannot be verified."
    if role in {FeatureRole.POST_CREATION_FIELD.value, FeatureRole.CONDITIONAL_FEATURE.value}:
        return "WARNING", "Preserve nulls until an explicit Step 7 field policy is approved."
    return "INFO", "Preserve source nulls unless Step 7 approves a deterministic rule."


def profile_missingness(
    frame: pd.DataFrame, *, null_like_strings: Collection[str]
) -> pd.DataFrame:
    """Return per-column missingness evidence without changing the input."""
    total = len(frame)
    null_like = {str(value).casefold() for value in null_like_strings if str(value) != ""}
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        values = frame[column]
        as_string = values.astype("string")
        non_null_strings = as_string[values.notna()]
        empty = int(non_null_strings.eq("").sum())
        whitespace = int((non_null_strings.ne("") & non_null_strings.str.strip().eq("")).sum())
        null_like_count = int(
            (
                non_null_strings.str.strip().str.casefold().isin(null_like)
                & ~non_null_strings.str.strip().eq("")
            ).sum()
        )
        null_count = int(values.isna().sum())
        role = _role(column)
        severity, action = _missingness_policy(column, role, null_count)
        rows.append({
            "column_name": column,
            "column_role": role,
            "row_count": total,
            "non_null_count": total - null_count,
            "null_count": null_count,
            "null_rate": null_count / total if total else 0.0,
            "empty_string_count": empty,
            "whitespace_only_count": whitespace,
            "null_like_string_count": null_like_count,
            "missingness_severity": severity,
            "recommended_step_7_action": action,
        })
    return pd.DataFrame(rows)
