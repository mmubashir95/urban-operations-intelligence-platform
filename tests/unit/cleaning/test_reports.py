"""Tests for stable cleaning tables, leakage evidence, and report writing."""

from pathlib import Path

import pandas as pd

from urban_ops.cleaning.eligibility import apply_governed_target
from urban_ops.cleaning.leakage import build_cleaning_leakage_audit
from urban_ops.cleaning.models import CleaningAction, CleaningCheck
from urban_ops.cleaning.reports import (
    REQUIRED_REPORT_TABLES, build_report_tables, write_cleaning_reports,
)
from tests.unit.cleaning.test_metadata import metadata


def build(raw_frame, selected_scope):
    frames = apply_governed_target(
        raw_frame, scope=selected_scope,
        extraction_timestamp="2026-12-31T00:00:00Z", conflicting_keys=set(),
    )
    leakage, all_null, zero = build_cleaning_leakage_audit(
        frames.cleaned, configured_all_null_columns=["descriptor_2"],
        configured_zero_variance_columns=["open_data_channel_type"],
    )
    timestamp = pd.DataFrame([{
        "column_name": "created_date", "input_non_null_count": len(raw_frame),
        "parsed_count": len(raw_frame), "null_count": 0, "parse_failure_count": 0,
        "timezone": "UTC", "invalid_policy": "preserve", "imputed_count": 0,
    }])
    missing = pd.DataFrame([{
        "column_name": "due_date", "missing_count": 0, "imputed_count": 0,
        "row_policy": "preserve", "target_policy": "target_ineligible", "quality_flag": "",
    }])
    category = pd.DataFrame(columns=[
        "source_column", "original_value", "cleaned_value", "affected_rows",
        "mapping_type", "mapping_reason", "approved_rule",
    ])
    duplicate = pd.DataFrame(columns=[
        "unique_key", "duplicate_type", "action", "group_size",
        "source_fingerprint", "canonical_fingerprint", "conflicting_fields",
    ])
    action = CleaningAction("a", "test", "", "PRESERVE", "x", "x", 0, "reason", "rule", "APPROVED", False)
    check = CleaningCheck("c", "test", "PASS", 0, 0, 0, "message")
    tables = build_report_tables(
        frames=frames, input_rows=len(raw_frame), removed_exact_copies=0,
        timestamp_summary=timestamp, missing_actions=missing,
        category_mapping=category, duplicate_actions=duplicate, leakage=leakage,
        all_null_columns=all_null, zero_variance_columns=zero,
        actions=(action,), checks=(check,),
    )
    return frames, tables


def test_all_required_tables_have_stable_empty_schemas(raw_frame, selected_scope) -> None:
    _, tables = build(raw_frame, selected_scope)
    assert set(tables) == set(REQUIRED_REPORT_TABLES)
    assert list(tables["duplicate_actions.csv"].columns)
    assert list(tables["category_mapping.csv"].columns)
    assert tables["output_reconciliation.csv"]["status"].eq("PASS").all()


def test_target_exclusion_and_field_reports_reconcile(raw_frame, selected_scope) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "due_date"] = None
    frames, tables = build(frame, selected_scope)
    assert tables["target_distribution.csv"]["row_count"].sum() == len(frames.eligible)
    assert tables["exclusion_reason_summary.csv"]["row_count"].sum() == len(frames.excluded)
    assert "descriptor_2" in set(tables["all_null_columns.csv"]["column_name"])
    assert "open_data_channel_type" in set(tables["zero_variance_columns.csv"]["column_name"])


def test_leakage_audit_blocks_outcomes_and_conditional_fields(raw_frame, selected_scope) -> None:
    _, tables = build(raw_frame, selected_scope)
    leakage = tables["leakage_validation.csv"].set_index("column_name")
    for column in ("unique_key", "closed_date", "status", "target_eligible", "missed_resolution_target"):
        assert not bool(leakage.loc[column, "baseline_allowed"])
    assert not bool(leakage.loc["descriptor", "baseline_allowed"])


def test_reports_write_all_tables_and_correct_run_id(
    tmp_path: Path, raw_frame, selected_scope
) -> None:
    _, tables = build(raw_frame, selected_scope)
    boundary_checks = pd.DataFrame([
        {
            "check_id": check_id, "area": "boundary", "status": "PASS",
            "observed_value": "same", "expected_value": "same",
            "affected_rows": 0, "message": "unchanged",
        }
        for check_id in (
            "boundary.raw_hash_immutable", "boundary.raw_mtime_immutable",
        )
    ])
    tables["cleaning_checks.csv"] = pd.concat(
        [tables["cleaning_checks.csv"], boundary_checks], ignore_index=True
    )
    write_cleaning_reports(
        report_root=tmp_path,
        tables=tables,
        metadata=metadata(),
        scope_description="agency=HPD; complaint_type=HEAT/HOT WATER",
    )
    assert all((tmp_path / "tables" / name).is_file() for name in REQUIRED_REPORT_TABLES)
    summary = (tmp_path / "cleaning_summary.md").read_text()
    assert "raw-1" in summary
    assert "agency=HPD; complaint_type=HEAT/HOT WATER" in summary
    assert "Raw immutability: **PASS**" in summary


def test_summary_derives_raw_immutability_failure_from_checks(
    tmp_path: Path, raw_frame, selected_scope
) -> None:
    _, tables = build(raw_frame, selected_scope)
    tables["cleaning_checks.csv"] = pd.DataFrame([
        {
            "check_id": "boundary.raw_hash_immutable", "area": "boundary",
            "status": "PASS", "observed_value": "same", "expected_value": "same",
            "affected_rows": 0, "message": "unchanged",
        },
        {
            "check_id": "boundary.raw_mtime_immutable", "area": "boundary",
            "status": "FAIL", "observed_value": 2, "expected_value": 1,
            "affected_rows": 1, "message": "changed",
        },
    ])
    write_cleaning_reports(
        report_root=tmp_path,
        tables=tables,
        metadata=metadata(),
        scope_description="fixture",
    )
    assert "Raw immutability: **FAIL**" in (
        tmp_path / "cleaning_summary.md"
    ).read_text()
