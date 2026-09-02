"""Integration coverage for split-safe rare and unseen category handling."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run
from urban_ops.features.categorical_missing import (
    build_categorical_reconciliation_table,
    load_categorical_missing_config,
    replace_split_categorical_missing,
)
from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.features.rare_unseen import (
    build_rare_unseen_evidence,
    fit_rare_unseen_handler,
    load_rare_unseen_config,
    transform_split_rare_unseen,
)
from urban_ops.features.temporal import derive_split_temporal_features
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
MISSING_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_missing.yaml"
)
CARDINALITY_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_cardinality.yaml"
)


def _synthetic_location_policy():
    """Return an in-memory policy that explicitly approves location_type."""
    policy = load_feature_policy(POLICY_PATH)
    features = tuple(
        replace(
            entry,
            policy_status=PolicyStatus.APPROVED_CANDIDATE,
            prediction_time_status="AVAILABLE",
            leakage_status="SAFE",
            eda_status="CANDIDATE",
            phase_2_allowed=True,
        )
        if entry.feature_name == "location_type"
        else entry
        for entry in policy.features
    )
    return replace(policy, features=features)


def test_split_fit_transform_is_training_only_reconciled_and_immutable(
    tmp_path: Path,
) -> None:
    raw = make_eda_frame()
    raw.loc[raw["unique_key"].eq("eda-001"), "location_type"] = "Alley"
    fixture = build_eda_fixture(tmp_path, raw)
    eda_config = load_eda_config(fixture.config)
    run_path = resolve_split_run(
        split_root=eda_config.split_root,
        latest_pointer=eda_config.latest_pointer,
    )
    source = load_verified_split(
        run_path=run_path,
        latest_pointer=eda_config.latest_pointer,
        required_completion_status=eda_config.required_completion_status,
        identifier_column=eda_config.identifier_column,
        target_column=eda_config.target_column,
        timestamp_column=eda_config.timestamp_column,
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
    governed_bytes = {
        path: path.read_bytes()
        for path in (POLICY_PATH, MISSING_CONFIG_PATH, CARDINALITY_CONFIG_PATH)
    }
    policy = load_feature_policy(POLICY_PATH)
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality_config = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH, missing_config=missing_config
    )
    derived = derive_split_temporal_features(source_frames, policy=policy)

    production_fitted = fit_rare_unseen_handler(
        derived["train"], policy=policy, config=cardinality_config
    )
    production_transformed = transform_split_rare_unseen(
        derived, fitted=production_fitted, config=cardinality_config
    )
    production_evidence = build_rare_unseen_evidence(
        derived,
        production_transformed,
        fitted=production_fitted,
        policy=policy,
        config=cardinality_config,
    )

    assert production_fitted.active_columns == ()
    assert production_evidence["active"].eq(False).all()
    assert production_evidence["transformation_applied"].eq(False).all()
    production_by_feature = production_evidence.set_index("feature_name")
    assert production_by_feature.loc[
        "location_type", "validation_unseen_value_count"
    ] == 1
    assert production_by_feature.loc[
        "location_type", "validation_rows_mapped_unknown"
    ] == 0
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(derived[split], production_transformed[split])

    synthetic_policy = _synthetic_location_policy()
    synthetic_config = replace(
        cardinality_config,
        min_count=2,
        production_decision="ACTIVE",
        production_reason="Synthetic integration-test activation.",
    )
    phase_3_frames = replace_split_categorical_missing(
        derived, policy=synthetic_policy, config=missing_config
    )
    fitted = fit_rare_unseen_handler(
        phase_3_frames["train"],
        policy=synthetic_policy,
        config=synthetic_config,
    )
    fitted_snapshot = fitted.to_dict()
    fitted_fingerprint = fitted.fingerprint
    transformed = transform_split_rare_unseen(
        phase_3_frames, fitted=fitted, config=synthetic_config
    )
    repeated = transform_split_rare_unseen(
        transformed, fitted=fitted, config=synthetic_config
    )
    evidence = build_rare_unseen_evidence(
        phase_3_frames,
        transformed,
        fitted=fitted,
        policy=synthetic_policy,
        config=synthetic_config,
    ).set_index("feature_name")
    reconciliation = build_categorical_reconciliation_table(
        phase_3_frames,
        transformed,
        identifier_column=eda_config.identifier_column,
        target_column=eda_config.target_column,
    )

    state = fitted.by_name["location_type"]
    assert state.rare_categories == ("Alley",)
    assert "Street" in state.retained_categories
    assert transformed["train"]["location_type"].eq("__RARE__").sum() == 1
    assert transformed["validation"]["location_type"].eq("__UNKNOWN__").all()
    assert transformed["test"]["location_type"].eq("__UNKNOWN__").all()
    assert evidence.loc["location_type", "validation_unseen_value_count"] == 1
    assert evidence.loc["location_type", "test_unseen_value_count"] == 1
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
    assert fitted.to_dict() == fitted_snapshot
    assert fitted.fingerprint == fitted_fingerprint
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(transformed[split], repeated[split])
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())
