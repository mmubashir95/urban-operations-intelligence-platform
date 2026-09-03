"""Integration coverage for Phase 7 preprocessing matrix composition."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
from scipy import sparse

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run
from urban_ops.features.categorical_encoding import (
    fit_categorical_encoder,
    load_categorical_encoding_config,
    transform_split_categorical_encoder,
)
from urban_ops.features.categorical_missing import (
    load_categorical_missing_config,
    replace_split_categorical_missing,
)
from urban_ops.features.numeric_preprocessing import (
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_split_numeric_preprocessor,
)
from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.features.preprocessing_composition import (
    build_preprocessing_composition,
    build_preprocessing_composition_evidence,
    compose_split_preprocessing_blocks,
    load_preprocessing_composition_config,
)
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
NUMERIC_CONFIG_PATH = Path(
    "configs/features/resolution_risk_numeric_preprocessing.yaml"
)
COMPOSITION_CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_composition.yaml"
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


def _load_verified_fixture(tmp_path: Path):
    """Create and load small verified split fixtures."""
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
    frames = {
        "train": source.train,
        "validation": source.validation,
        "test": source.test,
    }
    return fixture, frames


def _run_upstream_preprocessing(frames, *, policy, activate_categorical: bool = False):
    """Run Phases 2 through 6 explicitly before Phase 7 composition."""
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality_config = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH, missing_config=missing_config
    )
    encoding_config = load_categorical_encoding_config(
        ENCODING_CONFIG_PATH, cardinality_config=cardinality_config
    )
    if activate_categorical:
        cardinality_config = replace(
            cardinality_config,
            min_count=2,
            production_decision="ACTIVE",
            production_reason="Synthetic integration-test activation.",
        )
        encoding_config = replace(
            encoding_config,
            production_decision="ACTIVE",
            production_reason="Synthetic integration-test activation.",
        )
    numeric_config = load_numeric_preprocessing_config(NUMERIC_CONFIG_PATH)
    temporal_policy = load_feature_policy(POLICY_PATH)
    derived = derive_split_temporal_features(frames, policy=temporal_policy)
    phase_3_frames = replace_split_categorical_missing(
        derived, policy=policy, config=missing_config
    )
    fitted_cardinality = fit_rare_unseen_handler(
        phase_3_frames["train"], policy=policy, config=cardinality_config
    )
    phase_4_frames = transform_split_rare_unseen(
        phase_3_frames, fitted=fitted_cardinality, config=cardinality_config
    )
    fitted_encoder = fit_categorical_encoder(
        phase_4_frames["train"],
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    categorical_matrices = transform_split_categorical_encoder(
        phase_4_frames, fitted=fitted_encoder, config=encoding_config
    )
    fitted_numeric = fit_numeric_preprocessor(
        phase_4_frames["train"], policy=policy, config=numeric_config
    )
    numeric_matrices = transform_split_numeric_preprocessor(
        phase_4_frames, fitted=fitted_numeric, config=numeric_config
    )
    return phase_4_frames, fitted_encoder, categorical_matrices, fitted_numeric, numeric_matrices


def test_real_production_preprocessing_composition_is_sparse_numeric_only_and_immutable(
    tmp_path: Path,
) -> None:
    fixture, source_frames = _load_verified_fixture(tmp_path)
    frame_snapshots = {
        split: frame.copy(deep=True) for split, frame in source_frames.items()
    }
    artifact_snapshots = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    governed_paths = (
        POLICY_PATH,
        MISSING_CONFIG_PATH,
        CARDINALITY_CONFIG_PATH,
        ENCODING_CONFIG_PATH,
        NUMERIC_CONFIG_PATH,
        COMPOSITION_CONFIG_PATH,
    )
    governed_bytes = {path: path.read_bytes() for path in governed_paths}

    policy = load_feature_policy(POLICY_PATH)
    (
        phase_4_frames,
        fitted_encoder,
        categorical_matrices,
        fitted_numeric,
        numeric_matrices,
    ) = _run_upstream_preprocessing(source_frames, policy=policy)
    composition_config = load_preprocessing_composition_config(COMPOSITION_CONFIG_PATH)
    fitted_composition = build_preprocessing_composition(
        fitted_categorical=fitted_encoder,
        fitted_numeric=fitted_numeric,
        policy=policy,
        config=composition_config,
    )
    categorical_snapshot = fitted_encoder.to_dict()
    numeric_snapshot = fitted_numeric.to_dict()
    composition_snapshot = fitted_composition.to_dict()
    composition_fingerprint = fitted_composition.fingerprint
    combined_matrices = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted_composition,
    )
    evidence = build_preprocessing_composition_evidence(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        combined_matrices=combined_matrices,
        fitted=fitted_composition,
    )

    assert fitted_encoder.encoded_feature_count == 0
    assert fitted_numeric.output_feature_count == 4
    assert fitted_composition.categorical_feature_count == 0
    assert fitted_composition.numeric_feature_count == 4
    assert fitted_composition.combined_feature_count == 4
    assert fitted_composition.combined_feature_names == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert evidence["status"].eq("PASS").all()
    assert evidence["schema_identical"].all()
    for split, matrix in combined_matrices.items():
        assert sparse.isspmatrix_csr(matrix)
        assert str(matrix.dtype) == "float64"
        assert matrix.shape == (len(phase_4_frames[split]), 4)
        assert matrix.shape[0] == categorical_matrices[split].shape[0]
        assert matrix.shape[0] == numeric_matrices[split].shape[0]
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                matrix.toarray(),
                columns=fitted_composition.combined_feature_names,
                index=phase_4_frames[split].index,
            ),
            phase_4_frames[split]
            .loc[:, fitted_numeric.output_feature_names]
            .astype("float64"),
            check_dtype=False,
        )
    assert "missed_resolution_target" not in fitted_composition.combined_feature_names
    assert "unique_key" not in fitted_composition.combined_feature_names
    assert "latitude" not in fitted_composition.combined_feature_names
    assert "longitude" not in fitted_composition.combined_feature_names
    assert fitted_encoder.to_dict() == categorical_snapshot
    assert fitted_numeric.to_dict() == numeric_snapshot
    assert fitted_composition.to_dict() == composition_snapshot
    assert fitted_composition.fingerprint == composition_fingerprint
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())


def test_synthetic_active_categorical_composition_keeps_categorical_features_first(
    tmp_path: Path,
) -> None:
    _, source_frames = _load_verified_fixture(tmp_path)
    synthetic_policy = _synthetic_location_policy()
    (
        _phase_4_frames,
        fitted_encoder,
        categorical_matrices,
        fitted_numeric,
        numeric_matrices,
    ) = _run_upstream_preprocessing(
        source_frames, policy=synthetic_policy, activate_categorical=True
    )
    composition_config = load_preprocessing_composition_config(COMPOSITION_CONFIG_PATH)
    fitted_composition = build_preprocessing_composition(
        fitted_categorical=fitted_encoder,
        fitted_numeric=fitted_numeric,
        policy=synthetic_policy,
        config=composition_config,
    )
    combined_matrices = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted_composition,
    )

    assert fitted_encoder.encoded_feature_count > 0
    assert fitted_composition.combined_feature_names == (
        *fitted_encoder.encoded_feature_names,
        *fitted_numeric.output_feature_names,
    )
    assert fitted_composition.categorical_feature_names == tuple(
        fitted_composition.combined_feature_names[
            : fitted_encoder.encoded_feature_count
        ]
    )
    assert fitted_composition.numeric_feature_names == tuple(
        fitted_composition.combined_feature_names[
            fitted_encoder.encoded_feature_count :
        ]
    )
    for split, combined in combined_matrices.items():
        expected = sparse.hstack(
            [categorical_matrices[split], numeric_matrices[split]],
            format="csr",
            dtype="float64",
        )
        assert combined.shape[1] == (
            fitted_encoder.encoded_feature_count
            + fitted_numeric.output_feature_count
        )
        assert (combined != expected).nnz == 0
