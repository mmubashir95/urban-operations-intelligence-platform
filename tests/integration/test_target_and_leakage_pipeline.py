"""Integration tests for the complete local Step 4 transformation."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_ops.analysis.target_and_leakage import build_step4_tables
from urban_ops.data.selected_scope import load_selected_scope_authority
from urban_ops.features.leakage import validate_feature_columns


def test_scope_loading_pipeline_and_metrics_reconcile(tmp_path: Path) -> None:
    authority_path = tmp_path / "selected_scope_summary.csv"
    metadata_path = tmp_path / "extraction_metadata.csv"
    pd.DataFrame([{
        "decision_status": "APPROVED_WITH_LIMITATIONS",
        "selected_agency": "DSNY", "selected_complaint_type": "Graffiti",
        "selected_start_date": "2024-01-01", "selected_end_date": "2025-12-31",
        "extraction_timestamp": "2026-01-31T00:00:00Z",
    }]).to_csv(authority_path, index=False)
    pd.DataFrame([{
        "source": "test", "dataset_identifier": "fixture",
        "extraction_timestamp": "2026-01-31T00:00:00Z",
        "agency": "DSNY", "complaint_type": "Graffiti",
    }]).to_csv(metadata_path, index=False)
    scope = load_selected_scope_authority(authority_path, metadata_path)
    source = pd.DataFrame([
        {"unique_key": "on-time", "created_date": "2024-01-01",
         "due_date": "2024-01-10", "closed_date": "2024-01-10",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
        {"unique_key": "late", "created_date": "2024-01-02",
         "due_date": "2024-01-10", "closed_date": "2024-01-11",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
        {"unique_key": "open", "created_date": "2024-01-03",
         "due_date": "2024-01-10", "closed_date": None,
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Open"},
        {"unique_key": "exact-dup", "created_date": "2024-01-04",
         "due_date": "2024-01-10", "closed_date": "2024-01-09",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
        {"unique_key": "exact-dup", "created_date": "2024-01-04",
         "due_date": "2024-01-10", "closed_date": "2024-01-09",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
        {"unique_key": "conflict-dup", "created_date": "2024-01-05",
         "due_date": "2024-01-10", "closed_date": "2024-01-09",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
        {"unique_key": "conflict-dup", "created_date": "2024-01-05",
         "due_date": "2024-01-10", "closed_date": "2024-01-11",
         "agency": "DSNY", "complaint_type": "Graffiti", "status": "Closed"},
    ])
    governed, tables = build_step4_tables(
        source, scope=scope, extraction_timestamp="2026-01-31T00:00:00Z"
    )
    summary = tables["target_summary.csv"].iloc[0]
    assert summary["total_selected_rows"] == len(governed)
    assert summary["target_eligible_rows"] == int(governed["target_eligible"].sum()) == 3
    assert summary["missed_count"] == int(governed["missed_resolution_target"].eq(1).sum()) == 1
    assert not governed.loc[governed["target_eligible"], "unique_key"].duplicated().any()
    assert "missing_closed_date" in set(
        tables["exclusion_reason_summary.csv"]["primary_exclusion_reason"]
    )

    exact_summary = tables["exact_duplicate_summary.csv"]
    assert exact_summary.set_index("unique_key").loc[
        "exact-dup", "redundant_row_count"
    ] == 1
    assert governed.loc[
        governed["unique_key"] == "exact-dup", "target_eligible"
    ].sum() == 1

    conflict_summary = tables["conflicting_duplicate_summary.csv"]
    conflict_row = conflict_summary.set_index("unique_key").loc["conflict-dup"]
    assert conflict_row["row_count"] == 2
    assert conflict_row["conflicting_fields"] == "closed_date"

    conflict_records = tables["conflicting_duplicate_records.csv"]
    conflict_records = conflict_records.loc[
        conflict_records["unique_key"] == "conflict-dup"
    ]
    assert len(conflict_records) == 2
    assert (
        conflict_records["primary_exclusion_reason"] == "conflicting_duplicate_unique_key"
    ).all()
    assert not governed.loc[
        governed["unique_key"] == "conflict-dup", "target_eligible"
    ].any()

    validate_feature_columns(["created_date", "agency"])
