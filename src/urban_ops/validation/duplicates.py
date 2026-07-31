"""Detect exact and conflicting complaint duplicates without removing records."""

from __future__ import annotations

import pandas as pd

from urban_ops.features.eligibility import DUPLICATE_RELEVANT_COLUMNS
from urban_ops.validation.models import DuplicateAnalysis


MATERIAL_COLUMNS = tuple(dict.fromkeys([
    *DUPLICATE_RELEVANT_COLUMNS,
    "descriptor", "resolution_description", "resolution_action_updated_date",
]))
EXACT_COLUMNS = ["unique_key", "duplicate_group_size", "canonical_row", "source_row_index"]
CONFLICT_COLUMNS = [
    "unique_key", "duplicate_group_size", "conflicting_fields", "source_row_index",
    *MATERIAL_COLUMNS,
]


def analyze_duplicates(frame: pd.DataFrame) -> DuplicateAnalysis:
    """Classify full-row, exact-complaint, and conflicting-key duplicates."""
    exact = pd.Series(False, index=frame.index, dtype=bool)
    conflict = pd.Series(False, index=frame.index, dtype=bool)
    exact_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []
    full_duplicate_rows = int(frame.duplicated(keep=False).sum())
    missing_keys = int(frame["unique_key"].isna().sum()) if "unique_key" in frame else len(frame)
    if "unique_key" in frame:
        members = frame["unique_key"].notna() & frame["unique_key"].duplicated(keep=False)
    else:
        members = pd.Series(False, index=frame.index)
    relevant = [column for column in MATERIAL_COLUMNS if column in frame]
    for key, group in frame.loc[members].groupby("unique_key", sort=True, dropna=False):
        signatures = group[relevant].astype("string").fillna("<NA>")
        changed = [column for column in relevant if signatures[column].nunique() > 1]
        if changed:
            conflict.loc[group.index] = True
            for index, row in group.iterrows():
                evidence = {
                    "unique_key": key,
                    "duplicate_group_size": len(group),
                    "conflicting_fields": "|".join(changed),
                    "source_row_index": str(index),
                }
                evidence.update({column: row.get(column) for column in MATERIAL_COLUMNS})
                conflict_rows.append(evidence)
        else:
            stable = sorted(group.index, key=lambda value: f"{type(value).__name__}:{value}")
            for position, index in enumerate(stable):
                if position:
                    exact.loc[index] = True
                exact_rows.append({
                    "unique_key": key,
                    "duplicate_group_size": len(group),
                    "canonical_row": position == 0,
                    "source_row_index": str(index),
                })
    duplicate_key_groups = int(frame.loc[members, "unique_key"].nunique()) if members.any() else 0
    summary = pd.DataFrame([
        {"metric": "duplicate_full_rows", "row_count": full_duplicate_rows},
        {"metric": "duplicate_unique_key_groups", "row_count": duplicate_key_groups},
        {"metric": "rows_in_duplicate_key_groups", "row_count": int(members.sum())},
        {"metric": "redundant_exact_duplicate_rows", "row_count": int(exact.sum())},
        {"metric": "conflicting_duplicate_keys", "row_count": int(
            frame.loc[conflict, "unique_key"].nunique() if conflict.any() else 0
        )},
        {"metric": "rows_in_conflicting_groups", "row_count": int(conflict.sum())},
        {"metric": "missing_unique_key_rows", "row_count": missing_keys},
    ])
    return DuplicateAnalysis(
        exact_flags=exact,
        conflicting_flags=conflict,
        summary=summary,
        exact_records=pd.DataFrame(exact_rows, columns=EXACT_COLUMNS),
        conflicting_records=pd.DataFrame(conflict_rows, columns=CONFLICT_COLUMNS),
    )
