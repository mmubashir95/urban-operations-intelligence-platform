"""Preserve governed missing values and add non-model quality audit flags."""

from __future__ import annotations

import pandas as pd

from urban_ops.cleaning.models import CleaningAction


MISSING_ACTION_COLUMNS = [
    "column_name", "missing_count", "imputed_count", "row_policy",
    "target_policy", "quality_flag",
]


def apply_missing_value_policy(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[CleaningAction, ...]]:
    """Return a copy with preservation flags and no value imputation."""
    result = frame.copy(deep=True)
    tracked = [
        "unique_key", "created_date", "agency", "complaint_type", "due_date",
        "closed_date", "descriptor", "descriptor_2", "borough", "incident_zip",
        "latitude", "longitude", "location_type",
    ]
    rows: list[dict[str, object]] = []
    actions: list[CleaningAction] = []
    for column in tracked:
        if column not in result:
            continue
        count = int(result[column].isna().sum())
        target_input = column in {"due_date", "closed_date"}
        scope_or_identity = column in {"unique_key", "created_date", "agency", "complaint_type"}
        rows.append({
            "column_name": column,
            "missing_count": count,
            "imputed_count": 0,
            "row_policy": "preserve",
            "target_policy": (
                "target_ineligible" if target_input or scope_or_identity else "not_target_determinative"
            ),
            "quality_flag": f"has_{column}" if column in {"borough", "descriptor"} else "",
        })
        actions.append(CleaningAction(
            action_id=f"missing.preserve.{column}",
            cleaning_area="missing_values",
            source_column=column,
            action_type="PRESERVE_NULL",
            input_value="null",
            output_value="null",
            affected_rows=count,
            reason="Approved missing-value policy prohibits imputation.",
            source_rule="docs/cleaning_policy.md#missing-value-policy",
            governance_status="APPROVED",
            applied=count > 0,
        ))
    lat = pd.to_numeric(result.get("latitude"), errors="coerce")
    lon = pd.to_numeric(result.get("longitude"), errors="coerce")
    result["has_coordinates"] = lat.notna() & lon.notna()
    result["coordinates_valid"] = (
        result["has_coordinates"] & lat.between(-90, 90) & lon.between(-180, 180)
    )
    result["has_borough"] = result.get("borough", pd.Series(pd.NA, index=result.index)).notna()
    result["has_descriptor"] = result.get(
        "descriptor", pd.Series(pd.NA, index=result.index)
    ).notna()
    if "incident_zip" in result:
        result["incident_zip"] = result["incident_zip"].astype("string")
    return result, pd.DataFrame(rows, columns=MISSING_ACTION_COLUMNS), tuple(actions)
