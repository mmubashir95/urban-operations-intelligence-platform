"""End-to-end verified Step 8 split to governed Step 9A evidence publication."""

from pathlib import Path

import pytest

from urban_ops.eda.figures import REQUIRED_FIGURES
from urban_ops.eda.pipeline import run_split_aware_eda
from urban_ops.eda.reports import REQUIRED_TABLES
from tests.unit.eda.conftest import build_eda_fixture


def test_complete_split_aware_eda_flow_is_governed_atomic_and_non_mutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_eda_fixture(tmp_path)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir() if path.is_file()
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network forbidden")),
    )
    result = run_split_aware_eda(config_path=fixture.config)
    assert all((fixture.reports / "tables" / name).is_file() for name in REQUIRED_TABLES)
    assert all((fixture.reports / "figures" / name).is_file() for name in REQUIRED_FIGURES)
    assert result.tables["output_reconciliation.csv"]["status"].eq("PASS").all()
    assert result.tables["eda_integrity_checks.csv"]["status"].eq("PASS").all()
    assert result.tables["baseline_feature_recommendation.csv"].loc[
        lambda frame: frame["feature_name"].eq("missed_resolution_target"),
        "baseline_decision",
    ].iloc[0] == "EXCLUDE_LEAKAGE"
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir() if path.is_file()
    }
    assert before == after
