"""Integration coverage for governed split-wide temporal feature creation."""

from pathlib import Path

import pandas as pd

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.temporal import (
    build_feature_reconciliation_table,
    build_temporal_validation_table,
    derive_split_temporal_features,
)
from tests.unit.eda.conftest import build_eda_fixture


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")


def test_split_wide_feature_creation_is_reconciled_repeatable_and_non_mutating(
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
    policy_bytes = POLICY_PATH.read_bytes()
    policy = load_feature_policy(POLICY_PATH)

    derived = derive_split_temporal_features(source_frames, policy=policy)
    repeated = derive_split_temporal_features(source_frames, policy=policy)
    reconciliation = build_feature_reconciliation_table(
        source_frames,
        derived,
        identifier_column=config.identifier_column,
        target_column=config.target_column,
    )
    validation = build_temporal_validation_table(
        source_frames, derived, policy=policy
    )

    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
        pd.testing.assert_frame_equal(derived[split], repeated[split])
    assert reconciliation[
        [
            "row_count_preserved",
            "index_preserved",
            "row_order_preserved",
            "unique_key_preserved",
            "target_preserved",
            "source_values_preserved",
        ]
    ].all().all()
    assert validation["creation_status"].eq("COMPLETE").all()
    assert validation["domain_valid"].all()
    assert validation["deterministic"].all()
    assert validation.loc[
        validation["feature_name"].eq("is_weekend"),
        "weekend_relationship_valid",
    ].eq(True).all()
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert POLICY_PATH.read_bytes() == policy_bytes
