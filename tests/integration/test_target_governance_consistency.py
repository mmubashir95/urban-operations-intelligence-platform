"""Cross-artifact regression checks for the Step 4 governance contract."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from urban_ops.data.selected_scope import load_selected_scope_authority
from urban_ops.features.eligibility import (
    DEFAULT_ALLOWED_STATUSES, DEFAULT_EXCLUDED_STATUSES, EXCLUSION_PRIORITY,
)
from urban_ops.features.feature_roles import FIELD_ROLES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_governance_documents_match_central_rules() -> None:
    target = (PROJECT_ROOT / "docs/target_definition.md").read_text(encoding="utf-8")
    leakage = (PROJECT_ROOT / "docs/leakage_policy.md").read_text(encoding="utf-8")
    assert "closed_date > due_date" in target
    assert "closed_date == due_date" in target
    assert "Int8" in target
    assert ", ".join(EXCLUSION_PRIORITY) in target
    assert f"`{next(iter(DEFAULT_ALLOWED_STATUSES))}`" in target
    for status in DEFAULT_EXCLUDED_STATUSES:
        assert status in target.casefold()
    assert "`due_date`" in leakage and "not approved" in leakage.casefold()
    for blocked in ("closed_date", "missed_resolution_target", "target_eligible"):
        assert blocked in leakage
        assert blocked in FIELD_ROLES


def test_generated_scope_docs_and_summary_agree_when_reports_exist() -> None:
    summary_path = (
        PROJECT_ROOT / "reports/06_target_and_leakage/tables/target_summary.csv"
    )
    if not summary_path.is_file():
        pytest.skip("Generated Step 4 reports are absent; execute Notebook 06.")
    scope = load_selected_scope_authority()
    summary = pd.read_csv(summary_path)
    target = (PROJECT_ROOT / "docs/target_definition.md").read_text(encoding="utf-8")
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["selected_agency"] == scope.agency
    assert row["selected_complaint_type"] == scope.complaint_type
    assert row["selected_start_date"] == scope.start_date.date().isoformat()
    assert row["selected_end_date"] == scope.end_date.date().isoformat()
    for value in (
        scope.agency, scope.complaint_type,
        scope.start_date.date().isoformat(), scope.end_date.date().isoformat(),
    ):
        assert value in target
