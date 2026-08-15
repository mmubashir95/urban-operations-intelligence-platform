"""Integration coverage for governed categorical missing-value handling."""

from pathlib import Path

import pandas as pd

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run
from urban_ops.features.categorical_missing import (
    active_categorical_feature_names,
    build_categorical_missing_evidence,
    build_categorical_reconciliation_table,
    load_categorical_missing_config,
    replace_split_categorical_missing,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.temporal import derive_split_temporal_features
from tests.unit.eda.conftest import build_eda_fixture


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
MISSING_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_missing.yaml"
)
TEMPORAL_CODE_PATH = Path("src/urban_ops/features/temporal.py")


def test_split_wide_categorical_handling_preserves_governance_and_sources(
    tmp_path: Path,
) -> None:
    fixture = build_eda_fixture(tmp_path)
    config = load_eda_config(fixture.config)
    run_path = resolve_split_run(
        split_root=config.split_root,
        latest_pointer=config.latest_pointer,
    )
    source = load_verified_split(
        run_path=run_path,
        latest_pointer=config.latest_pointer,
        required_completion_status=config.required_completion_status,
        identifier_column=config.identifier_column,
        target_column=config.target_column,
        timestamp_column=config.timestamp_column,
    )
    source_frames = {
        "train": source.train,
        "validation": source.validation,
        "test": source.test,
    }
    frame_snapshots = {
        split: frame.copy(deep=True) for split, frame in source_frames.items()
    }
    artifact_snapshots = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    protected_code = TEMPORAL_CODE_PATH.read_bytes()
    policy_bytes = POLICY_PATH.read_bytes()
    policy = load_feature_policy(POLICY_PATH)
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)

    derived = derive_split_temporal_features(source_frames, policy=policy)
    transformed = replace_split_categorical_missing(
        derived, policy=policy, config=missing_config
    )
    repeated = replace_split_categorical_missing(
        transformed, policy=policy, config=missing_config
    )
    evidence = build_categorical_missing_evidence(
        derived, transformed, policy=policy, config=missing_config
    )
    reconciliation = build_categorical_reconciliation_table(
        derived,
        transformed,
        identifier_column=config.identifier_column,
        target_column=config.target_column,
    )

    assert active_categorical_feature_names(policy, missing_config) == ()
    assert evidence["active"].eq(False).all()
    assert evidence["source_token_collision"].eq(False).all()
    assert evidence["status"].eq("PASS").all()
    assert reconciliation[
        [
            "row_count_preserved",
            "index_preserved",
            "row_order_preserved",
            "unique_key_preserved",
            "target_preserved",
        ]
    ].all().all()
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
        pd.testing.assert_frame_equal(derived[split], transformed[split])
        pd.testing.assert_frame_equal(transformed[split], repeated[split])
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert POLICY_PATH.read_bytes() == policy_bytes
    assert TEMPORAL_CODE_PATH.read_bytes() == protected_code
