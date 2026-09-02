"""Integration coverage for split-safe categorical one-hot encoding."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
from scipy import sparse

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run
from urban_ops.features.categorical_encoding import (
    build_categorical_encoding_evidence,
    fit_categorical_encoder,
    load_categorical_encoding_config,
    transform_split_categorical_encoder,
)
from urban_ops.features.categorical_missing import (
    build_categorical_reconciliation_table,
    load_categorical_missing_config,
    replace_split_categorical_missing,
)
from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.features.rare_unseen import (
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
ENCODING_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_encoding.yaml"
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


def test_split_categorical_encoding_is_training_only_reconciled_and_immutable(
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
        for path in (
            POLICY_PATH,
            MISSING_CONFIG_PATH,
            CARDINALITY_CONFIG_PATH,
            ENCODING_CONFIG_PATH,
        )
    }
    policy = load_feature_policy(POLICY_PATH)
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality_config = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH, missing_config=missing_config
    )
    encoding_config = load_categorical_encoding_config(
        ENCODING_CONFIG_PATH, cardinality_config=cardinality_config
    )
    derived = derive_split_temporal_features(source_frames, policy=policy)

    production_cardinality = fit_rare_unseen_handler(
        derived["train"], policy=policy, config=cardinality_config
    )
    production_encoder = fit_categorical_encoder(
        derived["train"],
        policy=policy,
        config=encoding_config,
        fitted_cardinality=production_cardinality,
    )
    production_matrices = transform_split_categorical_encoder(
        derived, fitted=production_encoder, config=encoding_config
    )
    production_evidence = build_categorical_encoding_evidence(
        derived,
        production_matrices,
        fitted=production_encoder,
        policy=policy,
        config=encoding_config,
    )

    assert production_encoder.active_columns == ()
    assert production_encoder.encoded_feature_count == 0
    assert production_evidence["active"].eq(False).all()
    assert production_evidence["transformation_applied"].eq(False).all()
    for split, matrix in production_matrices.items():
        assert sparse.issparse(matrix)
        assert matrix.shape == (len(derived[split]), 0)

    synthetic_policy = _synthetic_location_policy()
    synthetic_cardinality_config = replace(
        cardinality_config,
        min_count=2,
        production_decision="ACTIVE",
        production_reason="Synthetic integration-test activation.",
    )
    synthetic_encoding_config = replace(
        encoding_config,
        production_decision="ACTIVE",
        production_reason="Synthetic integration-test activation.",
    )
    phase_3_frames = replace_split_categorical_missing(
        derived, policy=synthetic_policy, config=missing_config
    )
    fitted_cardinality = fit_rare_unseen_handler(
        phase_3_frames["train"],
        policy=synthetic_policy,
        config=synthetic_cardinality_config,
    )
    phase_4_frames = transform_split_rare_unseen(
        phase_3_frames,
        fitted=fitted_cardinality,
        config=synthetic_cardinality_config,
    )
    fitted_encoder = fit_categorical_encoder(
        phase_4_frames["train"],
        policy=synthetic_policy,
        config=synthetic_encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    fitted_snapshot = fitted_encoder.to_dict()
    fitted_fingerprint = fitted_encoder.fingerprint
    matrices = transform_split_categorical_encoder(
        phase_4_frames,
        fitted=fitted_encoder,
        config=synthetic_encoding_config,
    )
    evidence = build_categorical_encoding_evidence(
        phase_4_frames,
        matrices,
        fitted=fitted_encoder,
        policy=synthetic_policy,
        config=synthetic_encoding_config,
    ).set_index("feature_name")
    reconciliation = build_categorical_reconciliation_table(
        phase_3_frames,
        phase_4_frames,
        identifier_column=eda_config.identifier_column,
        target_column=eda_config.target_column,
    )

    state = fitted_encoder.by_name["location_type"]
    assert state.encoder_categories == (
        "Street",
        "__MISSING__",
        "__RARE__",
        "__UNKNOWN__",
    )
    assert set(state.encoded_feature_names) == {
        "location_type_Street",
        "location_type___MISSING__",
        "location_type___RARE__",
        "location_type___UNKNOWN__",
    }
    assert {matrix.shape[1] for matrix in matrices.values()} == {
        fitted_encoder.encoded_feature_count
    }
    assert matrices["validation"].nnz == len(phase_4_frames["validation"])
    assert matrices["test"].nnz == len(phase_4_frames["test"])
    assert evidence.loc["location_type", "active"]
    assert evidence.loc["location_type", "reserved_token_columns_present"]
    assert evidence.loc["location_type", "feature_schema_identical"]
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
    assert fitted_encoder.to_dict() == fitted_snapshot
    assert fitted_encoder.fingerprint == fitted_fingerprint
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())
