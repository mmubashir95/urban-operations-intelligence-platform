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
    ])
    governed, tables = build_step4_tables(
        source, scope=scope, extraction_timestamp="2026-01-31T00:00:00Z"
    )
    summary = tables["target_summary.csv"].iloc[0]
    assert summary["total_selected_rows"] == len(governed)
    assert summary["target_eligible_rows"] == int(governed["target_eligible"].sum()) == 2
    assert summary["missed_count"] == int(governed["missed_resolution_target"].eq(1).sum()) == 1
    assert not governed.loc[governed["target_eligible"], "unique_key"].duplicated().any()
    assert "missing_closed_date" in set(
        tables["exclusion_reason_summary.csv"]["primary_exclusion_reason"]
    )
    validate_feature_columns(["created_date", "agency"])
