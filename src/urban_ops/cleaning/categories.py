"""Apply only configured category transformations and produce mapping evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from urban_ops.cleaning.models import CleaningAction


CATEGORY_MAPPING_COLUMNS = [
    "source_column", "original_value", "cleaned_value", "affected_rows",
    "mapping_type", "mapping_reason", "approved_rule",
]


def clean_categories(
    frame: pd.DataFrame,
    *,
    trim_columns: Sequence[str],
    blank_to_null_columns: Sequence[str],
    collapse_repeated_spaces_columns: Sequence[str],
    approved_mappings: Mapping[str, Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[CleaningAction, ...]]:
    """Return a copy with explicitly approved category transformations."""
    result = frame.copy(deep=True)
    audit_rows: list[dict[str, object]] = []
    actions: list[CleaningAction] = []
    trim_set = set(trim_columns)
    blank_set = set(blank_to_null_columns)
    collapse_set = set(collapse_repeated_spaces_columns)
    configured = trim_set | blank_set | collapse_set | set(approved_mappings)
    missing = sorted(configured.difference(result.columns))
    if missing:
        raise ValueError(f"Configured category columns are missing: {missing}")
    for column in sorted(configured):
        source = result[column].astype("string")
        cleaned = source.copy()
        if column in trim_set:
            transformed = cleaned.str.strip()
            _append_changes(audit_rows, column, cleaned, transformed, "trim", "Approved whitespace trimming")
            cleaned = transformed
        if column in collapse_set:
            transformed = cleaned.str.split().str.join(" ")
            _append_changes(
                audit_rows, column, cleaned, transformed, "collapse_repeated_spaces",
                "Explicitly approved repeated-space normalization",
            )
            cleaned = transformed
        if column in blank_set:
            transformed = cleaned.mask(cleaned.eq(""), pd.NA)
            _append_changes(
                audit_rows, column, cleaned, transformed, "blank_to_null",
                "Configured true blanks are represented as null",
            )
            cleaned = transformed
        mapping = approved_mappings.get(column, {})
        if mapping:
            transformed = cleaned.replace(dict(mapping))
            _append_changes(
                audit_rows, column, cleaned, transformed, "approved_mapping",
                "Explicit verified category mapping",
            )
            cleaned = transformed
        result[column] = cleaned
    audit = pd.DataFrame(audit_rows, columns=CATEGORY_MAPPING_COLUMNS)
    for row in audit.itertuples(index=False):
        actions.append(CleaningAction(
            action_id=f"category.{row.mapping_type}.{row.source_column}.{len(actions):04d}",
            cleaning_area="category",
            source_column=row.source_column,
            action_type=str(row.mapping_type).upper(),
            input_value=row.original_value,
            output_value=row.cleaned_value,
            affected_rows=int(row.affected_rows),
            reason=row.mapping_reason,
            source_rule=row.approved_rule,
            governance_status="APPROVED",
            applied=True,
        ))
    return result, audit, tuple(actions)


def _append_changes(
    rows: list[dict[str, object]],
    column: str,
    before: pd.Series,
    after: pd.Series,
    mapping_type: str,
    reason: str,
) -> None:
    """Append grouped before/after values for one transformation stage."""
    changed = ~(before.fillna("<NA>").eq(after.fillna("<NA>")))
    if not changed.any():
        return
    pairs = pd.DataFrame({"before": before[changed], "after": after[changed]})
    counts = pairs.value_counts(dropna=False, sort=True)
    for (original, cleaned), count in counts.items():
        rows.append({
            "source_column": column,
            "original_value": "<NULL>" if pd.isna(original) else str(original),
            "cleaned_value": "<NULL>" if pd.isna(cleaned) else str(cleaned),
            "affected_rows": int(count),
            "mapping_type": mapping_type,
            "mapping_reason": reason,
            "approved_rule": "configs/data/cleaning_rules.yaml",
        })
