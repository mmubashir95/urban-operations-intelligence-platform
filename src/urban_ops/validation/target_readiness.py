"""Measure provisional target readiness by reusing Step 4 eligibility rules."""

from __future__ import annotations

import pandas as pd

from urban_ops.data.selected_scope import SelectedScope
from urban_ops.features.eligibility import (
    DEFAULT_ALLOWED_STATUSES,
    DEFAULT_EXCLUDED_STATUSES,
    evaluate_target_eligibility,
)
from urban_ops.validation.models import DuplicateAnalysis, TargetReadinessAnalysis


READINESS_FLAGS = [
    "has_created_date", "has_due_date", "has_closed_date", "valid_due_chronology",
    "valid_closed_chronology", "status_allowed", "outcome_mature",
    "is_conflicting_duplicate", "candidate_target_eligible",
]


def analyze_target_readiness(
    frame: pd.DataFrame,
    *,
    scope: SelectedScope,
    extraction_timestamp: pd.Timestamp | str,
    duplicates: DuplicateAnalysis,
) -> TargetReadinessAnalysis:
    """Return provisional readiness flags without constructing the target label.

    Unexpected statuses are passed to Step 4 as provisionally excluded only so
    validation can finish; the status report still requires a governance decision.
    """
    normalized = frame["status"].astype("string").str.strip().str.casefold()
    unexpected = set(normalized.dropna().unique()) - (
        set(DEFAULT_ALLOWED_STATUSES) | set(DEFAULT_EXCLUDED_STATUSES)
    )
    governed = evaluate_target_eligibility(
        frame,
        selected_agency=scope.agency,
        selected_complaint_type=scope.complaint_type,
        start_date=scope.start_date,
        end_date=scope.end_date,
        extraction_timestamp=extraction_timestamp,
        allowed_statuses=set(DEFAULT_ALLOWED_STATUSES),
        excluded_statuses=set(DEFAULT_EXCLUDED_STATUSES) | unexpected,
    )
    # Step 6's material-field duplicate report is intentionally broader than
    # Step 4's target-input signature. A newly found conflict cannot be target-ready.
    governed["is_conflicting_duplicate"] = (
        governed["is_conflicting_duplicate"] | duplicates.conflicting_flags
    )
    governed["candidate_target_eligible"] = (
        governed["target_eligible"] & ~duplicates.conflicting_flags
    )
    reasons = governed["primary_exclusion_reason"].copy()
    newly_conflicting = duplicates.conflicting_flags & governed["target_eligible"]
    reasons.loc[newly_conflicting] = "conflicting_duplicate_unique_key"
    flags = governed.reindex(columns=READINESS_FLAGS).copy()
    summary_rows = []
    for column in READINESS_FLAGS:
        values = flags[column].fillna(False).astype(bool)
        passes = ~values if column.startswith("is_") else values
        summary_rows.append({
            "readiness_rule": column,
            "true_count": int(values.sum()),
            "false_count": int((~values).sum()),
            "pass_count": int(passes.sum()),
            "fail_count": int((~passes).sum()),
            "pass_rate": float(passes.mean()) if len(flags) else 0.0,
            "provisional": True,
        })
    summary = pd.DataFrame(summary_rows)
    exclusions = (
        reasons.value_counts(dropna=False)
        .rename_axis("candidate_exclusion_reason")
        .rename("row_count")
        .reset_index()
    )
    exclusions["row_share"] = exclusions["row_count"] / len(frame) if len(frame) else 0.0
    exclusions["final_target_construction_stage"] = "Step 7 after approved cleaning"
    return TargetReadinessAnalysis(flags=flags, summary=summary, exclusions=exclusions)
