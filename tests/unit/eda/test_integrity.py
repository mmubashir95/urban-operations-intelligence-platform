"""Tests for governed EDA integrity and temporal-domain checks."""

from dataclasses import replace

import pytest

from urban_ops.eda.analysis import derive_temporal_features
from urban_ops.eda.integrity import build_integrity_checks, require_integrity
from urban_ops.eda.pipeline import load_eda_config, run_split_aware_eda


def test_integrity_checks_cover_schema_target_reconciliation_and_domains(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    check_ids = {check.check_id for check in result.checks}
    assert {"source.rows_reconcile", "source.schemas_equal", "source.target_binary_non_null"} <= check_ids
    assert all(check.status == "PASS" for check in result.checks)


def test_invalid_temporal_domain_fails_integrity(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    config = load_eda_config(eda_fixture.config)
    train = derive_temporal_features(result.source.train, config.timestamp_column)
    train.loc[train.index[0], "created_hour"] = 24
    checks = build_integrity_checks(result.source, train, config)
    with pytest.raises(RuntimeError, match="created_hour"):
        require_integrity(checks)
