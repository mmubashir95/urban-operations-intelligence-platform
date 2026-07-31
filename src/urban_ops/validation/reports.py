"""Write the canonical Step 6 CSV evidence set and Markdown decision summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_ops.data.metadata import ExtractionMetadata
from urban_ops.data.selected_scope import SelectedScope
from urban_ops.validation.models import ValidationCheck
from urban_ops.validation.severity import severity_counts


REQUIRED_REPORT_TABLES = (
    "validation_checks.csv",
    "schema_validation.csv",
    "scope_validation.csv",
    "column_profile.csv",
    "missingness_summary.csv",
    "timestamp_validation_summary.csv",
    "chronology_violations.csv",
    "duplicate_summary.csv",
    "exact_duplicate_records.csv",
    "conflicting_duplicate_records.csv",
    "category_profile.csv",
    "category_variants.csv",
    "status_validation.csv",
    "geographic_validation.csv",
    "geographic_outliers.csv",
    "target_readiness_summary.csv",
    "candidate_exclusion_reason_summary.csv",
    "proposed_cleaning_actions.csv",
)


def proposed_cleaning_actions(checks: tuple[ValidationCheck, ...]) -> pd.DataFrame:
    """Convert non-pass findings into recommendations, never transformations."""
    columns = [
        "issue_id", "validation_area", "source_column", "issue_description",
        "affected_rows", "severity", "proposed_action", "action_type",
        "requires_governance_approval", "source_rule", "notes",
    ]
    rows = []
    for check in checks:
        if check.status.value == "PASS" or check.affected_rows == 0:
            continue
        parts = check.check_id.split(".")
        if check.area in {"missingness", "timestamp"} or check.check_id.startswith("null_like."):
            source_column = parts[-1]
        elif check.area == "geography" and len(parts) > 2:
            source_column = parts[1]
        elif check.area == "status":
            source_column = "status"
        elif check.area == "duplicate":
            source_column = "unique_key"
        else:
            source_column = ""
        rows.append({
            "issue_id": check.check_id,
            "validation_area": check.area,
            "source_column": source_column,
            "issue_description": check.message,
            "affected_rows": check.affected_rows,
            "severity": check.severity.value,
            "proposed_action": check.recommended_step_7_action,
            "action_type": "GOVERNANCE_REVIEW" if check.area in {"category", "status"} else "STEP_7_RULE",
            "requires_governance_approval": check.area in {"category", "status"},
            "source_rule": check.check_name,
            "notes": "Recommendation only; Step 6 did not modify any record.",
        })
    return pd.DataFrame(rows, columns=columns)


def write_validation_reports(
    *,
    report_root: Path,
    tables: dict[str, pd.DataFrame],
    checks: tuple[ValidationCheck, ...],
    metadata: ExtractionMetadata,
    scope: SelectedScope,
    raw_run_path: Path,
    raw_file: Path,
    overall_status: str,
) -> None:
    """Write all canonical tables and the machine-derived validation narrative."""
    table_root = report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    missing = sorted(set(REQUIRED_REPORT_TABLES).difference(tables))
    if missing:
        raise ValueError(f"Validation report tables are missing: {missing}")
    for filename in REQUIRED_REPORT_TABLES:
        tables[filename].to_csv(table_root / filename, index=False)
    counts = severity_counts(checks)
    missingness = tables["missingness_summary.csv"].set_index("column_name")
    chronology = tables["chronology_violations.csv"]
    duplicates = tables["duplicate_summary.csv"].set_index("metric")["row_count"]
    geography = tables["geographic_validation.csv"].set_index(["field", "metric"])["row_count"]
    readiness = tables["target_readiness_summary.csv"].set_index("readiness_rule")
    candidate_ready = int(readiness.loc["candidate_target_eligible", "pass_count"])
    unexpected = int((~tables["status_validation.csv"]["is_expected"]).sum())
    finding_sections = []
    for severity in ("CRITICAL", "ERROR", "WARNING", "INFO"):
        findings = [
            check for check in checks
            if check.severity.value == severity and check.status.value != "PASS"
        ]
        lines = "\n".join(
            f"- `{check.check_id}`: {check.affected_rows:,} affected — {check.message}"
            for check in findings
        ) or "- None."
        finding_sections.append(f"### {severity.title()} findings\n\n{lines}")
    findings_text = "\n\n".join(finding_sections)
    text = f"""# Step 6 Data Validation Summary

## Validated raw input

- Raw run ID: `{metadata.run_id}`
- Raw run path: `{raw_run_path}`
- Raw Parquet: `{raw_file}`
- Dataset: `{metadata.dataset_id}`
- Shape: {metadata.retrieved_row_count:,} rows × {len(metadata.selected_source_columns)} columns
- Scope: `{scope.agency}` / `{scope.complaint_type}`, `{scope.start_date.date()}` through `{scope.end_date.date()}` inclusive
- Scope authority: `{scope.authority_path}`

Metadata, query, source identity, scope, row count, and selected-column contracts
were reconciled before validation. The raw Parquet was read only and was not
rewritten.

## Severity decision

- Overall validation status: **{overall_status}**
- Critical findings: {counts['CRITICAL']}
- Error findings: {counts['ERROR']}
- Warning findings: {counts['WARNING']}
- Informational findings: {counts['INFO']}

Default CLI execution fails only for critical findings. `--fail-on-error` also
fails for errors; `--fail-on-warning` fails for errors or warnings.

## Findings by severity

{findings_text}

## Key findings

- Missing `due_date`: {int(missingness.loc['due_date', 'null_count']):,}
- Missing `closed_date`: {int(missingness.loc['closed_date', 'null_count']):,}
- Chronology evidence rows: {len(chronology):,}
- Duplicate unique-key groups: {int(duplicates.get('duplicate_unique_key_groups', 0)):,}
- Conflicting duplicate keys: {int(duplicates.get('conflicting_duplicate_keys', 0)):,}
- Unexpected raw status values: {unexpected:,}
- Invalid latitude values: {int(geography.get(('latitude', 'invalid_numeric'), 0) + geography.get(('latitude', 'outside_world_range'), 0)):,}
- Invalid longitude values: {int(geography.get(('longitude', 'invalid_numeric'), 0) + geography.get(('longitude', 'outside_world_range'), 0)):,}
- Coordinates outside broad NYC bounds: {int(geography.get(('coordinate_pair', 'outside_nyc_bounding_box'), 0)):,}
- Candidate target-ready rows: {candidate_ready:,}
- Candidate target-ineligible rows: {metadata.retrieved_row_count - candidate_ready:,}

## Step 7 recommendations

Every material finding is represented in `tables/proposed_cleaning_actions.csv`.
Missing target inputs should not be imputed; chronology violations should not be
silently repaired; duplicate conflicts and unexpected statuses require explicit
handling; and invalid geographic values should not automatically exclude the
complaint itself.

## Boundary and limitations

Step 6 detects and measures issues. It does not fill values, standardize saved
categories, remove rows, resolve duplicates, construct
`missed_resolution_target`, assign final target eligibility, or write a cleaned
dataset. Comparison normalization and timestamp parsing exist only in memory.
Final cleaning and target construction occur after approved Step 7 rules.
"""
    (report_root / "validation_summary.md").write_text(text, encoding="utf-8")
