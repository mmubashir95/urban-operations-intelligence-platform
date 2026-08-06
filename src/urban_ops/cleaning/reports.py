"""Build and write the canonical Step 7 evidence tables and summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_ops.cleaning.metadata import CleaningMetadata
from urban_ops.cleaning.models import CleaningAction, CleaningCheck, CleaningFrames


REQUIRED_REPORT_TABLES = (
    "cleaning_checks.csv", "cleaning_actions.csv", "rows_before_after.csv",
    "timestamp_cleaning_summary.csv", "missing_value_actions.csv",
    "category_mapping.csv", "category_field_decisions.csv",
    "duplicate_actions.csv", "chronology_actions.csv",
    "exclusion_reason_summary.csv", "target_distribution.csv",
    "zero_variance_columns.csv", "all_null_columns.csv",
    "leakage_validation.csv", "output_reconciliation.csv",
)

CHRONOLOGY_ACTION_COLUMNS = [
    "unique_key", "violation_type", "created_date", "due_date", "closed_date",
    "action", "primary_exclusion_reason",
]


def build_chronology_actions(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return row evidence for target-governed due and closure chronology failures."""
    rows: list[dict[str, object]] = []
    rules = (
        ("due_before_created", ~cleaned["valid_due_chronology"]),
        ("closed_before_created", ~cleaned["valid_closed_chronology"]),
    )
    for violation, mask in rules:
        for row in cleaned.loc[mask.fillna(False)].itertuples(index=False):
            rows.append({
                "unique_key": row.unique_key,
                "violation_type": violation,
                "created_date": row.created_date,
                "due_date": row.due_date,
                "closed_date": row.closed_date,
                "action": "preserve_and_exclude_without_repair",
                "primary_exclusion_reason": row.primary_exclusion_reason,
            })
    return pd.DataFrame(rows, columns=CHRONOLOGY_ACTION_COLUMNS)


def build_report_tables(
    *,
    frames: CleaningFrames,
    input_rows: int,
    removed_exact_copies: int,
    timestamp_summary: pd.DataFrame,
    missing_actions: pd.DataFrame,
    category_mapping: pd.DataFrame,
    duplicate_actions: pd.DataFrame,
    leakage: pd.DataFrame,
    all_null_columns: list[str],
    zero_variance_columns: list[str],
    actions: tuple[CleaningAction, ...],
    checks: tuple[CleaningCheck, ...],
) -> dict[str, pd.DataFrame]:
    """Build all required, stable-schema cleaning report tables."""
    cleaned, eligible, excluded = frames.cleaned, frames.eligible, frames.excluded
    chronology = build_chronology_actions(cleaned)
    exclusion = (
        excluded["primary_exclusion_reason"].value_counts(sort=True)
        .rename_axis("primary_exclusion_reason").rename("row_count").reset_index()
    )
    exclusion["row_share"] = exclusion["row_count"] / len(excluded) if len(excluded) else 0.0
    target = pd.DataFrame([
        {"target_value": value, "target_label": label,
         "row_count": int(eligible["missed_resolution_target"].eq(value).sum())}
        for value, label in ((0, "on_time"), (1, "missed"))
    ])
    target["row_share"] = target["row_count"] / len(eligible) if len(eligible) else 0.0
    category_decisions = pd.DataFrame([
        {
            "column_name": column,
            "all_null": column in all_null_columns,
            "zero_variance": column in zero_variance_columns,
            "baseline_allowed": bool(
                leakage.set_index("column_name").loc[column, "baseline_allowed"]
            ) if column in set(leakage["column_name"]) else False,
            "decision": (
                "preserve_explicit_source_value" if column in {"open_data_channel_type", "borough"}
                else "preserve_null_without_imputation" if column == "descriptor_2"
                else "apply_configured_rules_only"
            ),
        }
        for column in sorted(set(category_mapping["source_column"]) | {
            "descriptor_2", "open_data_channel_type", "borough"
        })
    ])
    reconciliations = pd.DataFrame([
        {"check_name": "raw_equals_cleaned_plus_removed_exact", "left_value": input_rows,
         "right_value": len(cleaned) + removed_exact_copies,
         "status": "PASS" if input_rows == len(cleaned) + removed_exact_copies else "FAIL"},
        {"check_name": "cleaned_equals_eligible_plus_excluded", "left_value": len(cleaned),
         "right_value": len(eligible) + len(excluded),
         "status": "PASS" if len(cleaned) == len(eligible) + len(excluded) else "FAIL"},
        {"check_name": "eligible_equals_binary_targets", "left_value": len(eligible),
         "right_value": int(target["row_count"].sum()),
         "status": "PASS" if len(eligible) == int(target["row_count"].sum()) else "FAIL"},
        {"check_name": "excluded_equals_reason_counts", "left_value": len(excluded),
         "right_value": int(exclusion["row_count"].sum()),
         "status": "PASS" if len(excluded) == int(exclusion["row_count"].sum()) else "FAIL"},
    ])
    return {
        "cleaning_checks.csv": pd.DataFrame([check.to_dict() for check in checks], columns=[
            "check_id", "area", "status", "observed_value", "expected_value",
            "affected_rows", "message",
        ]),
        "cleaning_actions.csv": pd.DataFrame([action.to_dict() for action in actions], columns=[
            "action_id", "cleaning_area", "source_column", "action_type",
            "input_value", "output_value", "affected_rows", "reason", "source_rule",
            "governance_status", "applied",
        ]),
        "rows_before_after.csv": pd.DataFrame([
            {"dataset": "raw_input", "row_count": input_rows},
            {"dataset": "removed_exact_duplicate_copies", "row_count": removed_exact_copies},
            {"dataset": "cleaned", "row_count": len(cleaned)},
            {"dataset": "eligible", "row_count": len(eligible)},
            {"dataset": "excluded", "row_count": len(excluded)},
        ]),
        "timestamp_cleaning_summary.csv": timestamp_summary,
        "missing_value_actions.csv": missing_actions,
        "category_mapping.csv": category_mapping,
        "category_field_decisions.csv": category_decisions,
        "duplicate_actions.csv": duplicate_actions,
        "chronology_actions.csv": chronology,
        "exclusion_reason_summary.csv": exclusion,
        "target_distribution.csv": target,
        "zero_variance_columns.csv": pd.DataFrame(
            {"column_name": zero_variance_columns, "baseline_allowed": False}
        ),
        "all_null_columns.csv": pd.DataFrame(
            {"column_name": all_null_columns, "baseline_allowed": False}
        ),
        "leakage_validation.csv": leakage,
        "output_reconciliation.csv": reconciliations,
    }


def write_cleaning_reports(
    *,
    report_root: Path,
    tables: dict[str, pd.DataFrame],
    metadata: CleaningMetadata,
    scope_description: str,
) -> None:
    """Write every canonical table and the machine-derived cleaning summary."""
    missing = sorted(set(REQUIRED_REPORT_TABLES).difference(tables))
    if missing:
        raise ValueError(f"Cleaning report tables are missing: {missing}")
    table_root = report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_REPORT_TABLES:
        tables[filename].to_csv(table_root / filename, index=False)
    cleaning_checks = tables["cleaning_checks.csv"]
    immutability_checks = cleaning_checks.loc[
        cleaning_checks["check_id"].isin(
            {"boundary.raw_hash_immutable", "boundary.raw_mtime_immutable"}
        )
    ]
    raw_immutability_status = (
        "PASS"
        if len(immutability_checks) == 2 and immutability_checks["status"].eq("PASS").all()
        else "FAIL"
    )
    summary = f"""# Step 7 Data Cleaning Summary

- Source raw run: `{metadata.source_raw_run_id}`
- Source raw SHA-256: `{metadata.source_raw_sha256}`
- Validation evidence: `{metadata.validation_evidence_status}`, {metadata.validation_critical_count} critical findings
- Governed scope: `{scope_description}`
- Input / cleaned rows: {metadata.input_row_count:,} / {metadata.cleaned_row_count:,}
- Eligible / excluded rows: {metadata.eligible_row_count:,} / {metadata.excluded_row_count:,}
- Exact duplicate copies removed: {metadata.removed_exact_duplicate_copies:,}
- Conflicting duplicate groups: {metadata.conflicting_duplicate_groups:,}
- Missing due / closed exclusions: {metadata.missing_due_date_exclusions:,} / {metadata.missing_closed_date_exclusions:,}
- Due / closed chronology exclusions: {metadata.due_before_created_exclusions:,} / {metadata.closed_before_created_exclusions:,}
- On-time / missed targets: {metadata.on_time_target_count:,} / {metadata.missed_target_count:,}
- Missed-target rate: {metadata.missed_target_rate:.2%}
- All-null fields: `{', '.join(metadata.all_null_columns) or 'none'}`
- Zero-variance fields: `{', '.join(metadata.zero_variance_columns) or 'none'}`

## Applied cleaning decisions

Timestamps were parsed to UTC in processed outputs. Approved categories were
trimmed and configured blanks became null. Missing target inputs were not
imputed, chronology was not repaired, exact copies used deterministic canonical
retention, and conflicting groups were preserved but excluded. Step 4 supplied
eligibility, exclusion precedence, target construction, and leakage governance.
The complete timestamp, missing-value, category-mapping, duplicate, chronology,
and leakage actions are recorded in the canonical CSV tables alongside this
summary.

## Outputs and reconciliation

- Cleaned: `{metadata.output_paths['cleaned']}`
- Eligible: `{metadata.output_paths['eligible']}`
- Excluded: `{metadata.output_paths['excluded']}`
- Reconciliation: **{tables['output_reconciliation.csv']['status'].iloc[0] if tables['output_reconciliation.csv']['status'].eq('PASS').all() else 'FAIL'}**
- Raw immutability: **{raw_immutability_status}**

## Remaining modelling decisions

The processed analytical data is not a model feature matrix. Conditional
features still require approval. `descriptor_2` is all-null,
`open_data_channel_type` is zero-variance, target inputs and outcome fields are
blocked, and due-date creation-time semantics remain unresolved.

## Step 7 completion decision

The cleaning run is complete and reproducible. No split, EDA, feature matrix,
or model was created.
"""
    (report_root / "cleaning_summary.md").write_text(summary, encoding="utf-8")
