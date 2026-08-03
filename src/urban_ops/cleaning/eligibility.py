"""Reuse Step 4 eligibility and target constructors for cleaned records."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from urban_ops.data.selected_scope import SelectedScope
from urban_ops.features.eligibility import evaluate_target_eligibility
from urban_ops.features.target import TARGET_NAME, build_missed_resolution_target
from urban_ops.cleaning.models import CleaningFrames


def apply_governed_target(
    frame: pd.DataFrame,
    *,
    scope: SelectedScope,
    extraction_timestamp: pd.Timestamp | str,
    conflicting_keys: Collection[str],
) -> CleaningFrames:
    """Apply the authoritative Step 4 eligibility and nullable target contracts."""
    governed = evaluate_target_eligibility(
        frame,
        selected_agency=scope.agency,
        selected_complaint_type=scope.complaint_type,
        start_date=scope.start_date,
        end_date=scope.end_date,
        extraction_timestamp=extraction_timestamp,
    )
    if conflicting_keys:
        conflicts = governed["unique_key"].astype("string").isin(set(conflicting_keys))
        governed.loc[conflicts, "is_conflicting_duplicate"] = True
        governed.loc[conflicts, "target_eligible"] = False
        governed.loc[conflicts, "primary_exclusion_reason"] = (
            "conflicting_duplicate_unique_key"
        )
    governed[TARGET_NAME] = build_missed_resolution_target(governed)
    eligible_mask = governed["target_eligible"].fillna(False).astype(bool)
    eligible = governed.loc[eligible_mask].copy().reset_index(drop=True)
    excluded = governed.loc[~eligible_mask].copy().reset_index(drop=True)
    cleaned = governed.reset_index(drop=True)
    _assert_target_contract(cleaned, eligible, excluded)
    return CleaningFrames(cleaned=cleaned, eligible=eligible, excluded=excluded)


def _assert_target_contract(
    cleaned: pd.DataFrame, eligible: pd.DataFrame, excluded: pd.DataFrame
) -> None:
    """Enforce binary eligible targets, nullable exclusions, and unique IDs."""
    if len(cleaned) != len(eligible) + len(excluded):
        raise RuntimeError("Cleaned rows do not reconcile to eligible and excluded rows.")
    values = set(eligible[TARGET_NAME].dropna().astype(int).unique())
    if not values.issubset({0, 1}) or eligible[TARGET_NAME].isna().any():
        raise RuntimeError("Eligible rows must have only binary non-null target values.")
    if excluded[TARGET_NAME].notna().any():
        raise RuntimeError("Excluded rows must have nullable target values only.")
    if eligible["unique_key"].isna().any() or eligible["unique_key"].duplicated().any():
        raise RuntimeError("Eligible complaint identifiers must be non-null and unique.")
    if excluded["primary_exclusion_reason"].isna().any():
        raise RuntimeError("Every excluded record must have an explicit reason.")
