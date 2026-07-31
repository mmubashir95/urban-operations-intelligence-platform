"""Profile raw categorical values and comparison-only formatting variants."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import pandas as pd

from urban_ops.features.eligibility import (
    DEFAULT_ALLOWED_STATUSES,
    DEFAULT_EXCLUDED_STATUSES,
)


CATEGORY_PROFILE_COLUMNS = [
    "column_name", "raw_value", "normalised_comparison_value", "row_count",
    "row_share", "has_leading_whitespace", "has_trailing_whitespace", "is_empty",
    "is_whitespace_only", "is_null_like_string", "case_variant_group",
    "is_rare_category", "column_distinct_count", "is_high_cardinality",
]
CATEGORY_VARIANT_COLUMNS = [
    "column_name", "normalised_comparison_value", "raw_variant_count",
    "raw_variants", "row_count", "variant_type",
]


def normalize_for_comparison(values: pd.Series) -> pd.Series:
    """Return stripped, case-folded values with repeated whitespace collapsed."""
    strings = values.astype("string")
    return strings.str.strip().str.split().str.join(" ").str.casefold()


def profile_categories(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    null_like_strings: Collection[str],
    rare_category_max_rows: int,
    high_cardinality_threshold: int,
    maximum_profile_values_per_column: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bounded raw category profiles and variant-group evidence."""
    null_like = {str(value).strip().casefold() for value in null_like_strings}
    profile_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    total = len(frame)
    for column in columns:
        if column not in frame:
            continue
        raw = frame[column].astype("string")
        normalized = normalize_for_comparison(raw)
        distinct = int(raw.nunique(dropna=True))
        counts = raw.value_counts(dropna=False, sort=True).head(maximum_profile_values_per_column)
        for value, count in counts.items():
            is_null = pd.isna(value)
            text = "<NULL>" if is_null else str(value)
            norm = "<NULL>" if is_null else " ".join(text.strip().split()).casefold()
            profile_rows.append({
                "column_name": column,
                "raw_value": text,
                "normalised_comparison_value": norm,
                "row_count": int(count),
                "row_share": count / total if total else 0.0,
                "has_leading_whitespace": False if is_null else text != text.lstrip(),
                "has_trailing_whitespace": False if is_null else text != text.rstrip(),
                "is_empty": False if is_null else text == "",
                "is_whitespace_only": False if is_null else text != "" and not text.strip(),
                "is_null_like_string": False if is_null else norm in null_like,
                "case_variant_group": norm,
                "is_rare_category": count <= rare_category_max_rows,
                "column_distinct_count": distinct,
                "is_high_cardinality": distinct > high_cardinality_threshold,
            })
        work = pd.DataFrame({"raw": raw, "normalized": normalized}).dropna()
        for norm, group in work.groupby("normalized", sort=True):
            variants = sorted(group["raw"].astype(str).unique())
            if len(variants) < 2:
                continue
            stripped = {value.strip() for value in variants}
            collapsed = {" ".join(value.strip().split()) for value in variants}
            if len(stripped) == 1:
                variant_type = "whitespace"
            elif len(collapsed) == 1:
                variant_type = "repeated_whitespace"
            elif len({value.casefold() for value in collapsed}) == 1:
                variant_type = "case"
            else:
                variant_type = "combined_formatting"
            variant_rows.append({
                "column_name": column,
                "normalised_comparison_value": norm,
                "raw_variant_count": len(variants),
                "raw_variants": " | ".join(variants),
                "row_count": len(group),
                "variant_type": variant_type,
            })
    return (
        pd.DataFrame(profile_rows, columns=CATEGORY_PROFILE_COLUMNS),
        pd.DataFrame(variant_rows, columns=CATEGORY_VARIANT_COLUMNS),
    )


def validate_statuses(
    frame: pd.DataFrame, *, parsed_timestamps: dict[str, pd.Series]
) -> pd.DataFrame:
    """Profile status coverage and classify values against Step 4 policy."""
    if "status" not in frame:
        return pd.DataFrame(columns=[
            "status_raw", "status_comparison_normalised", "row_count", "row_share",
            "due_date_coverage", "closed_date_coverage", "created_date_coverage",
            "chronology_violation_count", "step_4_policy", "is_expected",
            "recommended_step_7_action",
        ])
    normalized = normalize_for_comparison(frame["status"])
    created = parsed_timestamps.get("created_date", pd.Series(pd.NaT, index=frame.index))
    due = parsed_timestamps.get("due_date", pd.Series(pd.NaT, index=frame.index))
    closed = parsed_timestamps.get("closed_date", pd.Series(pd.NaT, index=frame.index))
    chronology = (
        (created.notna() & due.notna() & due.lt(created))
        | (created.notna() & closed.notna() & closed.lt(created))
    )
    rows: list[dict[str, object]] = []
    work = frame.assign(_status_normalized=normalized)
    for raw_value, group in work.groupby("status", dropna=False, sort=True):
        indices = group.index
        norm_values = normalized.loc[indices].dropna().unique()
        norm = str(norm_values[0]) if len(norm_values) else "<MISSING>"
        if norm in DEFAULT_ALLOWED_STATUSES:
            policy, expected = "ALLOWED", True
            action = "Preserve; apply governed status normalization only in Step 7."
        elif norm in DEFAULT_EXCLUDED_STATUSES:
            policy, expected = "EXCLUDED", True
            action = "Preserve row and keep target-ineligible under Step 4 policy."
        else:
            policy, expected = "UNEXPECTED", False
            action = "Obtain governance decision before mapping or target construction."
        rows.append({
            "status_raw": "<NULL>" if pd.isna(raw_value) else str(raw_value),
            "status_comparison_normalised": norm,
            "row_count": len(group),
            "row_share": len(group) / len(frame) if len(frame) else 0.0,
            "due_date_coverage": float(due.loc[indices].notna().mean()),
            "closed_date_coverage": float(closed.loc[indices].notna().mean()),
            "created_date_coverage": float(created.loc[indices].notna().mean()),
            "chronology_violation_count": int(chronology.loc[indices].sum()),
            "step_4_policy": policy,
            "is_expected": expected,
            "recommended_step_7_action": action,
        })
    return pd.DataFrame(rows).sort_values(
        ["status_comparison_normalised", "status_raw"], kind="stable"
    ).reset_index(drop=True)
