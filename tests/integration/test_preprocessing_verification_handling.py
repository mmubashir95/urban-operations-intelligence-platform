"""End-to-end coverage for the Phase 2-8 preprocessing verification path."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import (
    load_verified_split,
    resolve_split_run,
    verify_source_unchanged,
)
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
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_composition import (
    build_preprocessing_composition,
    compose_split_preprocessing_blocks,
    load_preprocessing_composition_config,
)
from urban_ops.features.preprocessing_verification import (
    build_final_feature_schema_evidence,
    build_preprocessing_verification_evidence,
    build_training_feature_variance_evidence,
    load_preprocessing_verification_config,
    verify_preprocessing_contract,
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
VERIFICATION_CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_verification.yaml"
)


def test_actual_phase_2_to_8_path_freezes_model_ready_contract(tmp_path: Path) -> None:
    """Verify the actual governed chain without mutating its inputs or artifacts."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
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
    governed_paths = (
        POLICY_PATH,
        MISSING_CONFIG_PATH,
        CARDINALITY_CONFIG_PATH,
        ENCODING_CONFIG_PATH,
        NUMERIC_CONFIG_PATH,
        COMPOSITION_CONFIG_PATH,
        VERIFICATION_CONFIG_PATH,
    )
    governed_bytes = {path: path.read_bytes() for path in governed_paths}

    policy = load_feature_policy(POLICY_PATH)
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality_config = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH, missing_config=missing_config
    )
    encoding_config = load_categorical_encoding_config(
        ENCODING_CONFIG_PATH, cardinality_config=cardinality_config
    )
    numeric_config = load_numeric_preprocessing_config(NUMERIC_CONFIG_PATH)
    composition_config = load_preprocessing_composition_config(
        COMPOSITION_CONFIG_PATH
    )
    verification_config = load_preprocessing_verification_config(
        VERIFICATION_CONFIG_PATH
    )

    derived_frames = derive_split_temporal_features(source_frames, policy=policy)
    missing_frames = replace_split_categorical_missing(
        derived_frames, policy=policy, config=missing_config
    )
    fitted_cardinality = fit_rare_unseen_handler(
        missing_frames["train"], policy=policy, config=cardinality_config
    )
    cardinality_frames = transform_split_rare_unseen(
        missing_frames, fitted=fitted_cardinality, config=cardinality_config
    )
    fitted_categorical = fit_categorical_encoder(
        cardinality_frames["train"],
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    categorical_matrices = transform_split_categorical_encoder(
        cardinality_frames, fitted=fitted_categorical, config=encoding_config
    )
    fitted_numeric = fit_numeric_preprocessor(
        cardinality_frames["train"], policy=policy, config=numeric_config
    )
    numeric_matrices = transform_split_numeric_preprocessor(
        cardinality_frames, fitted=fitted_numeric, config=numeric_config
    )
    fitted_composition = build_preprocessing_composition(
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        policy=policy,
        config=composition_config,
    )
    matrices = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted_composition,
    )
    upstream_snapshots = {
        "cardinality": fitted_cardinality.to_dict(),
        "categorical": fitted_categorical.to_dict(),
        "numeric": fitted_numeric.to_dict(),
        "composition": fitted_composition.to_dict(),
    }
    upstream_fingerprints = {
        "cardinality": fitted_cardinality.fingerprint,
        "categorical": fitted_categorical.fingerprint,
        "numeric": fitted_numeric.fingerprint,
        "composition": fitted_composition.fingerprint,
    }
    targets = {
        split: frame[eda_config.target_column]
        for split, frame in source_frames.items()
    }
    identifiers = {
        split: frame[eda_config.identifier_column]
        for split, frame in source_frames.items()
    }
    timestamps = {
        split: frame[eda_config.timestamp_column]
        for split, frame in source_frames.items()
    }
    schemas = {
        split: fitted_composition.combined_feature_names
        for split in ("train", "validation", "test")
    }

    contract = verify_preprocessing_contract(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names_by_split=schemas,
        target_name=eda_config.target_column,
        identifier_name=eda_config.identifier_column,
        timestamp_name=eda_config.timestamp_column,
        policy=policy,
        fitted_cardinality=fitted_cardinality,
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
        config=verification_config,
    )
    repeated_matrices = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted_composition,
    )
    repeated_contract = verify_preprocessing_contract(
        matrices=repeated_matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names_by_split=schemas,
        target_name=eda_config.target_column,
        identifier_name=eda_config.identifier_column,
        timestamp_name=eda_config.timestamp_column,
        policy=policy,
        fitted_cardinality=fitted_cardinality,
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
        config=verification_config,
    )
    split_evidence = build_preprocessing_verification_evidence(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        contract=contract,
    )
    schema_evidence = build_final_feature_schema_evidence(
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
    )
    variance_evidence = build_training_feature_variance_evidence(
        matrix=matrices["train"],
        feature_names=fitted_composition.combined_feature_names,
    )

    expected_names = (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert contract.combined_feature_names == expected_names
    assert contract.combined_feature_count == 4
    assert contract.verification_status == "FROZEN_MODEL_READY"
    assert contract == repeated_contract
    assert contract.fingerprint == repeated_contract.fingerprint
    assert split_evidence["status"].eq("PASS").all()
    assert split_evidence["non_finite_count"].eq(0).all()
    assert not variance_evidence["zero_variance"].any()
    assert schema_evidence["feature_name"].tolist() == list(expected_names)
    assert schema_evidence["source_branch"].eq("numeric").all()
    assert not {
        eda_config.target_column,
        eda_config.identifier_column,
        "borough",
        "location_type",
        "incident_zip",
        "latitude",
        "longitude",
        "closed_date",
        "due_date",
        "status",
    }.intersection(contract.combined_feature_names)
    for split, matrix in matrices.items():
        assert sparse.isspmatrix_csr(matrix)
        assert str(matrix.dtype) == "float64"
        assert matrix.shape == (len(source_frames[split]), 4)
        assert np.isfinite(matrix.data).all()
        assert (matrix != repeated_matrices[split]).nnz == 0
        assert len(targets[split]) == len(identifiers[split]) == matrix.shape[0]
        assert targets[split].index.equals(identifiers[split].index)
    assert fitted_cardinality.to_dict() == upstream_snapshots["cardinality"]
    assert fitted_categorical.to_dict() == upstream_snapshots["categorical"]
    assert fitted_numeric.to_dict() == upstream_snapshots["numeric"]
    assert fitted_composition.to_dict() == upstream_snapshots["composition"]
    assert fitted_cardinality.fingerprint == upstream_fingerprints["cardinality"]
    assert fitted_categorical.fingerprint == upstream_fingerprints["categorical"]
    assert fitted_numeric.fingerprint == upstream_fingerprints["numeric"]
    assert fitted_composition.fingerprint == upstream_fingerprints["composition"]
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
    verify_source_unchanged(source)
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())
