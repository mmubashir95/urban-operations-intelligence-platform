"""Build Step 4 row-level governance and reproducible report artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from urban_ops.data.selected_scope import SelectedScope
from urban_ops.features.eligibility import (
    DEFAULT_ALLOWED_STATUSES, DEFAULT_EXCLUDED_STATUSES,
    DUPLICATE_RELEVANT_COLUMNS, EXCLUSION_PRIORITY,
    evaluate_target_eligibility,
)
from urban_ops.features.feature_roles import feature_role_inventory
from urban_ops.features.leakage import build_leakage_audit
from urban_ops.features.target import TARGET_NAME, build_missed_resolution_target


FLAGS: Final = (
    "within_selected_scope", "has_created_date", "has_due_date", "has_closed_date",
    "valid_due_chronology", "valid_closed_chronology", "status_allowed",
    "is_exact_duplicate", "is_conflicting_duplicate", "outcome_mature",
    "target_eligible",
)
REQUIRED_REPORT_TABLES: Final = (
    "target_summary.csv", "target_distribution.csv", "eligibility_summary.csv",
    "exclusion_reason_summary.csv", "status_analysis.csv", "duplicate_summary.csv",
    "exact_duplicate_summary.csv", "conflicting_duplicate_summary.csv",
    "conflicting_duplicate_records.csv", "timestamp_violation_summary.csv",
    "outcome_maturity_summary.csv", "leakage_audit.csv",
    "feature_role_inventory.csv", "extraction_metadata.csv",
)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _duplicate_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    members = frame["unique_key"].notna() & frame["unique_key"].duplicated(keep=False)
    exact = frame.loc[frame["is_exact_duplicate"]]
    conflicts = frame.loc[frame["is_conflicting_duplicate"]]
    overview = pd.DataFrame([
        {"metric": "duplicate_unique_key_groups",
         "row_count": int(frame.loc[members, "unique_key"].nunique())},
        {"metric": "rows_in_duplicate_groups", "row_count": int(members.sum())},
        {"metric": "redundant_exact_duplicate_rows", "row_count": len(exact)},
        {"metric": "conflicting_duplicate_groups",
         "row_count": int(conflicts["unique_key"].nunique())},
        {"metric": "rows_in_conflicting_groups", "row_count": len(conflicts)},
    ])
    exact_summary = (
        exact.groupby("unique_key", dropna=False).size()
        .rename("redundant_row_count").reset_index()
        .reindex(columns=["unique_key", "redundant_row_count"])
    )
    conflict_summary = (
        conflicts.groupby("unique_key", dropna=False).size()
        .rename("row_count").reset_index()
    )
    conflict_summary["conflicting_fields"] = ""
    relevant = [column for column in DUPLICATE_RELEVANT_COLUMNS if column in conflicts]
    for key, group in conflicts.groupby("unique_key", dropna=False):
        changed = [
            column for column in relevant
            if group[column].astype("string").fillna("<NA>").nunique() > 1
        ]
        conflict_summary.loc[
            conflict_summary["unique_key"].astype("string").eq(str(key)),
            "conflicting_fields",
        ] = "|".join(changed)
    conflict_summary = conflict_summary.reindex(
        columns=["unique_key", "row_count", "conflicting_fields"]
    )
    records = conflicts.reindex(
        columns=["unique_key", *relevant, "primary_exclusion_reason"]
    )
    return {
        "duplicate_summary.csv": overview,
        "exact_duplicate_summary.csv": exact_summary,
        "conflicting_duplicate_summary.csv": conflict_summary,
        "conflicting_duplicate_records.csv": records,
    }


def build_step4_tables(
    source: pd.DataFrame,
    *,
    scope: SelectedScope,
    extraction_timestamp: pd.Timestamp | str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply Step 4 rules and return governed rows plus all required tables."""
    governed = evaluate_target_eligibility(
        source, selected_agency=scope.agency,
        selected_complaint_type=scope.complaint_type,
        start_date=scope.start_date, end_date=scope.end_date,
        extraction_timestamp=extraction_timestamp,
    )
    governed[TARGET_NAME] = build_missed_resolution_target(governed)
    total, eligible = len(governed), int(governed["target_eligible"].sum())
    if not eligible:
        raise ValueError("Step 4 produced no eligible rows for the selected scope.")
    missed, on_time = (
        int(governed[TARGET_NAME].eq(1).sum()),
        int(governed[TARGET_NAME].eq(0).sum()),
    )
    target_summary = pd.DataFrame([{
        "selected_agency": scope.agency,
        "selected_complaint_type": scope.complaint_type,
        "selected_start_date": scope.start_date.date().isoformat(),
        "selected_end_date": scope.end_date.date().isoformat(),
        "extraction_timestamp": pd.Timestamp(extraction_timestamp).isoformat(),
        "total_selected_rows": total, "target_eligible_rows": eligible,
        "target_ineligible_rows": total - eligible,
        "eligibility_rate": _ratio(eligible, total),
        "missed_count": missed, "on_time_count": on_time,
        "missed_target_rate": _ratio(missed, eligible),
        "open_count": int(governed["is_open"].sum()),
        "cancelled_count": int(governed["is_cancelled"].sum()),
        "exact_duplicate_count": int(governed["is_exact_duplicate"].sum()),
        "conflicting_duplicate_count": int(governed["is_conflicting_duplicate"].sum()),
        "invalid_due_chronology_count": int((~governed["valid_due_chronology"]).sum()),
        "invalid_closed_chronology_count": int((~governed["valid_closed_chronology"]).sum()),
        "outcome_immature_count": int(
            (governed["has_due_date"] & ~governed["outcome_mature"]).sum()
        ),
    }])
    distribution = pd.DataFrame([
        {"target_value": value, "target_label": label, "row_count": count,
         "row_share": _ratio(count, eligible)}
        for value, label, count in ((0, "on_time", on_time), (1, "missed", missed))
    ])
    rule_rows = []
    for flag in FLAGS:
        values = governed[flag].fillna(False).astype(bool)
        passes = ~values if flag.startswith("is_") else values
        count = int(passes.sum())
        rule_rows.append({"rule_name": flag, "pass_count": count,
                          "fail_count": total - count, "pass_rate": _ratio(count, total)})
    exclusions = (
        governed["primary_exclusion_reason"].value_counts()
        .rename_axis("primary_exclusion_reason").rename("row_count").reset_index()
    )
    exclusions["row_share"] = exclusions["row_count"] / total
    status_rows = []
    for status, group in governed.groupby("status_normalized", dropna=False, sort=True):
        label = "<MISSING>" if pd.isna(status) else str(status)
        allowed = label in DEFAULT_ALLOWED_STATUSES
        known_excluded = label in DEFAULT_EXCLUDED_STATUSES
        treatment = (
            "include when all other rules pass" if allowed else
            "exclude" if known_excluded else "exclude pending governance review"
        )
        status_rows.append({
            "status": label, "row_count": len(group), "row_share": len(group) / total,
            "created_date_coverage": group["has_created_date"].mean(),
            "due_date_coverage": group["has_due_date"].mean(),
            "closed_date_coverage": group["has_closed_date"].mean(),
            "outcome_mature_count": int(group["outcome_mature"].sum()),
            "target_eligible_count": int(group["target_eligible"].sum()),
            "proposed_treatment": treatment, "final_treatment": treatment,
            "decision_reason": (
                "Closed is the only approved completed status." if allowed
                else "Not an approved completed operational outcome."
            ),
        })
    timestamps = pd.DataFrame([
        {"violation": "due_before_created",
         "row_count": int((~governed["valid_due_chronology"]).sum())},
        {"violation": "closed_before_created",
         "row_count": int((~governed["valid_closed_chronology"]).sum())},
        {"violation": "status_closed_date_inconsistent",
         "row_count": int(governed["status_closed_date_inconsistent"].sum())},
    ])
    maturity = pd.DataFrame([
        {"outcome_maturity": "mature", "row_count": int(governed["outcome_mature"].sum())},
        {"outcome_maturity": "not_mature",
         "row_count": int((governed["has_due_date"] & ~governed["outcome_mature"]).sum())},
        {"outcome_maturity": "missing_due_date",
         "row_count": int((~governed["has_due_date"]).sum())},
        {"outcome_maturity": "mature_open",
         "row_count": int((governed["outcome_mature"] & governed["is_open"]).sum())},
    ])
    maturity["row_share"] = maturity["row_count"] / total
    tables = {
        "target_summary.csv": target_summary,
        "target_distribution.csv": distribution,
        "eligibility_summary.csv": pd.DataFrame(rule_rows),
        "exclusion_reason_summary.csv": exclusions,
        "status_analysis.csv": pd.DataFrame(status_rows),
        "timestamp_violation_summary.csv": timestamps,
        "outcome_maturity_summary.csv": maturity,
        "leakage_audit.csv": build_leakage_audit(),
        "feature_role_inventory.csv": feature_role_inventory(),
        **_duplicate_tables(governed),
    }
    return governed, tables


def write_step4_reports(
    tables: dict[str, pd.DataFrame],
    extraction_metadata: pd.DataFrame,
    *,
    report_directory: Path,
) -> None:
    """Write required tables and the machine-derived decision narrative."""
    table_dir = report_directory / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(table_dir / filename, index=False)
    extraction_metadata.to_csv(table_dir / "extraction_metadata.csv", index=False)
    row = tables["target_summary.csv"].iloc[0]
    text = f"""# Target and Leakage Decision

Step 4 is approved for `{row.selected_agency}` / `{row.selected_complaint_type}`,
`{row.selected_start_date}` through `{row.selected_end_date}` inclusive.

`missed_resolution_target = closed_date > due_date`: late is `1`; on or before
due is `0`; ineligible rows are `NA` with dtype `Int8`.

Open and cancelled records are excluded. Exact duplicate rows retain one stable
canonical record; every conflicting duplicate-group member is excluded.
Exclusion precedence: `{", ".join(EXCLUSION_PRIORITY)}`.

Prediction occurs immediately after complaint creation. Post-creation,
target-input, target-derived, identifier, unknown, and future-aggregate fields
are blocked. Conditional features require approval. `due_date` remains blocked
as a feature because its timing and mutability are unproven.

Snapshot: {int(row.total_selected_rows):,} selected, {int(row.target_eligible_rows):,}
eligible, {float(row.missed_target_rate):.2%} missed; extracted
`{row.extraction_timestamp}`.
"""
    (report_directory / "target_and_leakage_decision.md").write_text(text, encoding="utf-8")
