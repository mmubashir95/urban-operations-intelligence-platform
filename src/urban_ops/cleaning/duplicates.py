"""Apply deterministic exact-copy retention and preserve conflicting groups."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pandas as pd

from urban_ops.cleaning.models import CleaningAction
from urban_ops.validation.duplicates import MATERIAL_COLUMNS


DUPLICATE_ACTION_COLUMNS = [
    "unique_key", "duplicate_type", "action", "group_size", "source_fingerprint",
    "canonical_fingerprint", "conflicting_fields",
]


@dataclass(frozen=True)
class DuplicateCleaningResult:
    """Deduplicated frame, conflict keys, and aggregate action evidence."""

    frame: pd.DataFrame
    conflicting_keys: frozenset[str]
    removed_exact_copies: int
    conflicting_group_count: int
    audit: pd.DataFrame
    actions: tuple[CleaningAction, ...]


def row_fingerprint(row: pd.Series, columns: list[str]) -> str:
    """Return a stable SHA-256 over typed, null-aware source values."""
    values: list[object] = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            values.append(None)
        elif isinstance(value, pd.Timestamp):
            values.append(value.isoformat())
        else:
            values.append(str(value))
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_duplicates(frame: pd.DataFrame, *, identifier: str) -> DuplicateCleaningResult:
    """Remove redundant exact copies and flag every conflicting-key member."""
    if identifier not in frame:
        raise ValueError(f"Duplicate identifier column is missing: {identifier}")
    source = frame.copy(deep=True)
    source_columns = list(source.columns)
    relevant = [column for column in MATERIAL_COLUMNS if column in source]
    fingerprints = source.apply(row_fingerprint, axis=1, columns=source_columns)
    source["source_row_fingerprint"] = fingerprints
    remove = pd.Series(False, index=source.index)
    conflict_keys: set[str] = set()
    audit_rows: list[dict[str, object]] = []
    members = source[identifier].notna() & source[identifier].duplicated(keep=False)
    for key, group in source.loc[members].groupby(identifier, sort=True, dropna=False):
        signatures = group[relevant].astype("string").fillna("<NA>")
        changed = [column for column in relevant if signatures[column].nunique() > 1]
        key_text = str(key)
        if changed:
            conflict_keys.add(key_text)
            for index, row in group.iterrows():
                audit_rows.append({
                    "unique_key": key_text,
                    "duplicate_type": "conflicting",
                    "action": "preserve_and_exclude",
                    "group_size": len(group),
                    "source_fingerprint": row["source_row_fingerprint"],
                    "canonical_fingerprint": "",
                    "conflicting_fields": "|".join(changed),
                })
            continue
        ordered = group.sort_values("source_row_fingerprint", kind="stable")
        canonical_index = ordered.index[0]
        canonical_hash = str(source.at[canonical_index, "source_row_fingerprint"])
        redundant_indices = list(ordered.index[1:])
        remove.loc[redundant_indices] = True
        for index, row in group.iterrows():
            audit_rows.append({
                "unique_key": key_text,
                "duplicate_type": "exact",
                "action": "retain_canonical" if index == canonical_index else "remove_copy",
                "group_size": len(group),
                "source_fingerprint": row["source_row_fingerprint"],
                "canonical_fingerprint": canonical_hash,
                "conflicting_fields": "",
            })
    cleaned = source.loc[~remove].copy().reset_index(drop=True)
    audit = pd.DataFrame(audit_rows, columns=DUPLICATE_ACTION_COLUMNS)
    removed_count = int(remove.sum())
    actions = (
        CleaningAction(
            action_id="duplicates.remove_exact_copies", cleaning_area="duplicates",
            source_column=identifier, action_type="KEEP_DETERMINISTIC",
            input_value="redundant exact copy", output_value="removed from processed copy",
            affected_rows=removed_count,
            reason="One stable canonical record is retained per exact duplicate group.",
            source_rule="urban_ops.features.eligibility.DUPLICATE_RELEVANT_COLUMNS",
            governance_status="APPROVED", applied=removed_count > 0,
        ),
        CleaningAction(
            action_id="duplicates.exclude_conflicting_groups", cleaning_area="duplicates",
            source_column=identifier, action_type="EXCLUDE_ENTIRE_GROUP",
            input_value="conflicting duplicate group", output_value="preserved but target-ineligible",
            affected_rows=int((audit["duplicate_type"] == "conflicting").sum()) if len(audit) else 0,
            reason="No conflicting record is selected as a winner.",
            source_rule="urban_ops.features.eligibility.DUPLICATE_RELEVANT_COLUMNS",
            governance_status="APPROVED", applied=bool(conflict_keys),
        ),
    )
    return DuplicateCleaningResult(
        frame=cleaned,
        conflicting_keys=frozenset(conflict_keys),
        removed_exact_copies=removed_count,
        conflicting_group_count=len(conflict_keys),
        audit=audit,
        actions=actions,
    )
