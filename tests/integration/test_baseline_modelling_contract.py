"""Integration coverage for Phase 8 outputs entering the Phase 9 contract."""

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
    load_preprocessing_verification_config,
    verify_preprocessing_contract,
)
from urban_ops.features.rare_unseen import (
    fit_rare_unseen_handler,
    load_rare_unseen_config,
    transform_split_rare_unseen,
)
from urban_ops.features.temporal import derive_split_temporal_features
from urban_ops.models.baseline_contract import (
    load_baseline_modelling_contract_config,
    verify_baseline_modelling_contract,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
MISSING_CONFIG_PATH = Path("configs/features/resolution_risk_categorical_missing.yaml")
CARDINALITY_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_cardinality.yaml"
)
ENCODING_CONFIG_PATH = Path("configs/features/resolution_risk_categorical_encoding.yaml")
NUMERIC_CONFIG_PATH = Path("configs/features/resolution_risk_numeric_preprocessing.yaml")
COMPOSITION_CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_composition.yaml"
)
VERIFICATION_CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_verification.yaml"
)
MODELLING_CONFIG_PATH = Path(
    "configs/models/resolution_risk_baseline_modelling_contract.yaml"
)


def test_phase_8_outputs_enter_phase_9_without_training_or_preprocessing_changes(
    tmp_path: Path,
) -> None:
    """Verify the actual local Phase 2-9 path freezes metadata only."""
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
        MODELLING_CONFIG_PATH,
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
    composition_config = load_preprocessing_composition_config(COMPOSITION_CONFIG_PATH)
    verification_config = load_preprocessing_verification_config(
        VERIFICATION_CONFIG_PATH
    )
    modelling_config = load_baseline_modelling_contract_config(MODELLING_CONFIG_PATH)

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
    targets = {
        split: frame[eda_config.target_column] for split, frame in source_frames.items()
    }
    identifiers = {
        split: frame[eda_config.identifier_column]
        for split, frame in source_frames.items()
    }
    timestamps = {
        split: frame[eda_config.timestamp_column] for split, frame in source_frames.items()
    }
    schemas = {
        split: fitted_composition.combined_feature_names
        for split in ("train", "validation", "test")
    }

    phase_8_contract = verify_preprocessing_contract(
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
    phase_9_contract = verify_baseline_modelling_contract(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names=fitted_composition.combined_feature_names,
        phase_8_contract=phase_8_contract,
        config=modelling_config,
        expected_phase_8_fingerprint=phase_8_contract.fingerprint,
        expected_schema_fingerprint=phase_8_contract.schema_fingerprint,
        policy=policy,
    )
    repeated_contract = verify_baseline_modelling_contract(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names=fitted_composition.combined_feature_names,
        phase_8_contract=phase_8_contract,
        config=modelling_config,
        expected_phase_8_fingerprint=phase_8_contract.fingerprint,
        expected_schema_fingerprint=phase_8_contract.schema_fingerprint,
        policy=policy,
    )

    assert phase_9_contract.status == "MODEL_INPUTS_VERIFIED"
    assert phase_9_contract.phase_8_contract_fingerprint == phase_8_contract.fingerprint
    assert phase_9_contract.preprocessing_schema_fingerprint == (
        phase_8_contract.schema_fingerprint
    )
    assert phase_9_contract.ordered_feature_names == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert phase_9_contract.feature_count == 4
    assert phase_9_contract.fingerprint == repeated_contract.fingerprint
    assert phase_9_contract.train_positive_class_count == int(targets["train"].eq(1).sum())
    assert phase_9_contract.train_negative_class_count == int(targets["train"].eq(0).sum())
    assert not {
        eda_config.target_column,
        eda_config.identifier_column,
        eda_config.timestamp_column,
        "borough",
        "incident_zip",
        "latitude",
        "longitude",
        "closed_date",
        "due_date",
        "status",
    }.intersection(phase_9_contract.ordered_feature_names)
    for split, matrix in matrices.items():
        assert sparse.isspmatrix_csr(matrix)
        assert str(matrix.dtype) == "float64"
        assert np.isfinite(matrix.data).all()
        assert phase_9_contract.row_counts_by_split[split] == matrix.shape[0]
        assert len(targets[split]) == len(identifiers[split]) == len(timestamps[split])
        assert targets[split].index.equals(identifiers[split].index)
        assert targets[split].index.equals(timestamps[split].index)
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(source_frames[split], frame_snapshots[split])
    verify_source_unchanged(source)
    assert all(path.read_bytes() == before for path, before in governed_bytes.items())
