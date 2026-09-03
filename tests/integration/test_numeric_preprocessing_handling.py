"""Integration coverage for split-safe numeric preprocessing."""

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
    build_numeric_preprocessing_evidence,
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_split_numeric_preprocessor,
)
from urban_ops.features.policy import load_feature_policy
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


def test_split_numeric_preprocessing_is_pass_through_reconciled_and_immutable(
    tmp_path: Path,
) -> None:
    raw = make_eda_frame()
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
    source_snapshots = {
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
            NUMERIC_CONFIG_PATH,
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
    numeric_config = load_numeric_preprocessing_config(NUMERIC_CONFIG_PATH)

    derived = derive_split_temporal_features(source_frames, policy=policy)
    phase_3_frames = replace_split_categorical_missing(
        derived, policy=policy, config=missing_config
    )
    fitted_cardinality = fit_rare_unseen_handler(
        phase_3_frames["train"],
        policy=policy,
        config=cardinality_config,
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
    numeric_snapshot = fitted_numeric.to_dict()
    numeric_fingerprint = fitted_numeric.fingerprint
    numeric_matrices = transform_split_numeric_preprocessor(
        phase_4_frames, fitted=fitted_numeric, config=numeric_config
    )
    evidence = build_numeric_preprocessing_evidence(
        phase_4_frames,
        numeric_matrices,
        fitted=fitted_numeric,
        policy=policy,
        config=numeric_config,
    ).set_index("feature_name")

    assert fitted_encoder.active_columns == ()
    assert {matrix.shape[1] for matrix in categorical_matrices.values()} == {0}
    assert fitted_numeric.active_columns == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert fitted_numeric.learned_statistics == "none"
    assert fitted_numeric.output_feature_names == fitted_numeric.active_columns
    assert fitted_numeric.matrix_type == "csr_matrix"
    assert {matrix.shape[1] for matrix in numeric_matrices.values()} == {4}
    for split, matrix in numeric_matrices.items():
        assert sparse.issparse(matrix)
        assert matrix.shape[0] == len(phase_4_frames[split])
        expected = phase_4_frames[split].loc[:, fitted_numeric.active_columns].astype(
            "float64"
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                matrix.toarray(),
                columns=fitted_numeric.output_feature_names,
                index=phase_4_frames[split].index,
            ),
            expected,
            check_dtype=False,
        )

    for feature in fitted_numeric.active_columns:
        assert evidence.loc[feature, "active"]
        assert evidence.loc[feature, "preprocessing_strategy"] == "pass_through"
        assert evidence.loc[feature, "learned_statistics"] == "none"
        assert evidence.loc[feature, "transformation_applied"]
        assert evidence.loc[feature, "status"] == "PASS"
    assert not evidence.loc["latitude", "active"]
    assert not evidence.loc["longitude", "active"]
    assert evidence.loc["latitude", "preprocessing_strategy"] == "deferred"
    assert evidence.loc["longitude", "preprocessing_strategy"] == "deferred"
    assert "missed_resolution_target" not in fitted_numeric.output_feature_names
    assert "unique_key" not in fitted_numeric.output_feature_names
    assert "latitude" not in fitted_numeric.output_feature_names
    assert "longitude" not in fitted_numeric.output_feature_names
    assert fitted_numeric.to_dict() == numeric_snapshot
    assert fitted_numeric.fingerprint == numeric_fingerprint

    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], source_snapshots[split])
    after_artifacts = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in fixture.split_run.iterdir()
        if path.is_file()
    }
    assert after_artifacts == artifact_snapshots
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())
