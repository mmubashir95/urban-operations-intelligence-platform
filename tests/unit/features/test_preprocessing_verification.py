"""Tests for the Phase 8 final preprocessing verification contract."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from urban_ops.features.categorical_encoding import FittedCategoricalEncoder
from urban_ops.features.numeric_preprocessing import (
    FittedNumericPreprocessor,
    NumericFeatureState,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_composition import (
    FittedPreprocessingComposition,
)
from urban_ops.features.preprocessing_verification import (
    PreprocessingVerificationError,
    VerifiedPreprocessingContract,
    build_final_feature_schema_evidence,
    build_preprocessing_verification_evidence,
    build_training_feature_variance_evidence,
    load_preprocessing_verification_config,
    verify_preprocessing_contract,
)
from urban_ops.features.rare_unseen import FittedRareUnseenState


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_verification.yaml"
)
FEATURE_NAMES = (
    "created_hour",
    "created_day_of_week",
    "created_month",
    "is_weekend",
)


def _states(
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    policy_version: int = 1,
):
    cardinality = FittedRareUnseenState(
        config_version=1,
        policy_version=policy_version,
        strategy="minimum_count",
        min_count=50,
        missing_token="__MISSING__",
        rare_token="__RARE__",
        unknown_token="__UNKNOWN__",
        active_columns=(),
        columns=(),
    )
    categorical = FittedCategoricalEncoder(
        config_version=1,
        policy_version=policy_version,
        cardinality_fingerprint=cardinality.fingerprint,
        strategy="one_hot",
        active_columns=(),
        missing_token="__MISSING__",
        rare_token="__RARE__",
        unknown_token="__UNKNOWN__",
        handle_unknown="ignore",
        drop=None,
        sparse_output=True,
        columns=(),
        encoder=None,
    )
    numeric = FittedNumericPreprocessor(
        config_version=1,
        policy_version=policy_version,
        active_columns=feature_names,
        output_feature_names=feature_names,
        matrix_type="csr_matrix",
        dtype="float64",
        learned_statistics="none",
        columns=tuple(
            NumericFeatureState(
                feature_name=name,
                preprocessing_strategy="pass_through",
                value_type="integer",
                minimum=0,
                maximum=23,
                output_feature_name=name,
            )
            for name in feature_names
        ),
    )
    composition = FittedPreprocessingComposition(
        config_version=1,
        policy_version=policy_version,
        categorical_encoder_fingerprint=categorical.fingerprint,
        numeric_preprocessor_fingerprint=numeric.fingerprint,
        categorical_feature_names=(),
        numeric_feature_names=feature_names,
        combined_feature_names=feature_names,
        categorical_feature_count=0,
        numeric_feature_count=len(feature_names),
        combined_feature_count=len(feature_names),
        matrix_type="csr_matrix",
        output_dtype="float64",
        composition_order=("categorical", "numeric"),
        learned_statistics="none",
    )
    return cardinality, categorical, numeric, composition


def _valid_inputs(*, feature_names: tuple[str, ...] = FEATURE_NAMES):
    width = len(feature_names)
    base = np.arange(1, 1 + 4 * width, dtype=np.float64).reshape(4, width)
    matrices = {
        split: sparse.csr_matrix(base + offset, dtype=np.float64)
        for split, offset in zip(("train", "validation", "test"), (0, 20, 40))
    }
    targets = {
        split: pd.Series([0, 1, 0, 1])
        for split in ("train", "validation", "test")
    }
    identifiers = {
        split: pd.Series([f"{split}-{index}" for index in range(4)])
        for split in ("train", "validation", "test")
    }
    timestamps = {
        "train": pd.Series(pd.date_range("2022-01-01", periods=4, tz="UTC")),
        "validation": pd.Series(pd.date_range("2022-02-01", periods=4, tz="UTC")),
        "test": pd.Series(pd.date_range("2022-03-01", periods=4, tz="UTC")),
    }
    schemas = {
        split: feature_names for split in ("train", "validation", "test")
    }
    return matrices, targets, identifiers, timestamps, schemas


def _verify(
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    matrices=None,
    targets=None,
    identifiers=None,
    timestamps=None,
    schemas=None,
    states=None,
    config=None,
):
    defaults = _valid_inputs(feature_names=feature_names)
    cardinality, categorical, numeric, composition = states or _states(
        feature_names=feature_names
    )
    return verify_preprocessing_contract(
        matrices=matrices or defaults[0],
        targets=targets or defaults[1],
        identifiers=identifiers or defaults[2],
        timestamps=timestamps or defaults[3],
        feature_names_by_split=schemas or defaults[4],
        target_name="missed_resolution_target",
        identifier_name="unique_key",
        timestamp_name="created_date",
        policy=load_feature_policy(POLICY_PATH),
        fitted_cardinality=cardinality,
        fitted_categorical=categorical,
        fitted_numeric=numeric,
        fitted_composition=composition,
        config=config or load_preprocessing_verification_config(CONFIG_PATH),
    )


def test_valid_contract_passes_and_round_trips_for_persistence() -> None:
    contract = _verify()
    restored = VerifiedPreprocessingContract.from_dict(
        json.loads(json.dumps(contract.to_dict()))
    )

    assert restored == contract
    assert restored.fingerprint == contract.fingerprint
    assert contract.verification_status == "FROZEN_MODEL_READY"
    assert contract.combined_feature_names == FEATURE_NAMES


@pytest.mark.parametrize(
    ("input_name", "message"),
    [
        ("targets", "target count"),
        ("identifiers", "identifier count"),
    ],
)
def test_row_count_mismatch_fails(input_name: str, message: str) -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    values = targets if input_name == "targets" else identifiers
    values["validation"] = values["validation"].iloc[:-1]

    with pytest.raises(PreprocessingVerificationError, match=message):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_feature_width_mismatch_fails() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    matrices["test"] = sparse.csr_matrix((4, 3), dtype=np.float64)

    with pytest.raises(PreprocessingVerificationError, match="matrix width"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_split_schema_name_mismatch_fails() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    schemas["validation"] = tuple(reversed(FEATURE_NAMES))

    with pytest.raises(PreprocessingVerificationError, match="feature schema differs"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_non_finite_sparse_value_fails(non_finite: float) -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    matrices["test"].data[0] = non_finite

    with pytest.raises(PreprocessingVerificationError, match="non-finite"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_duplicate_feature_names_fail() -> None:
    names = ("created_hour", "created_hour")

    with pytest.raises(PreprocessingVerificationError, match="unique"):
        _verify(feature_names=names)


def test_zero_feature_count_fails() -> None:
    matrices = {
        split: sparse.csr_matrix((4, 0), dtype=np.float64)
        for split in ("train", "validation", "test")
    }

    with pytest.raises(PreprocessingVerificationError, match="greater than zero"):
        _verify(feature_names=(), matrices=matrices)


def test_target_null_and_invalid_domain_fail() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    targets["validation"] = pd.Series([0, 1, None, 1])
    with pytest.raises(PreprocessingVerificationError, match="null"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )

    targets["validation"] = pd.Series([0, 1, 2, 1])
    with pytest.raises(PreprocessingVerificationError, match="domain"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_single_training_class_fails() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    targets["train"] = pd.Series([0, 0, 0, 0])

    with pytest.raises(PreprocessingVerificationError, match="both classes"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


@pytest.mark.parametrize(
    "blocked_name", ["missed_resolution_target", "unique_key", "status"]
)
def test_target_identifier_or_leakage_lineage_fails(blocked_name: str) -> None:
    states = _states(feature_names=(blocked_name,))

    with pytest.raises(
        PreprocessingVerificationError,
        match="Target or identifier|unapproved",
    ):
        _verify(feature_names=(blocked_name,), states=states)


def test_row_alignment_uses_unchanged_positions_and_indexes() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    target_before = targets["train"].copy(deep=True)
    identifier_before = identifiers["train"].copy(deep=True)
    _verify(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        schemas=schemas,
    )
    pd.testing.assert_series_equal(targets["train"], target_before)
    pd.testing.assert_series_equal(identifiers["train"], identifier_before)

    identifiers["train"].index = [4, 5, 6, 7]
    with pytest.raises(PreprocessingVerificationError, match="indexes are not aligned"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_repeated_verification_is_deterministic_and_inputs_are_unchanged() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    matrix_copies = {split: matrix.copy() for split, matrix in matrices.items()}
    first = _verify(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        schemas=schemas,
    )
    second = _verify(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        schemas=schemas,
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert all((matrices[split] != matrix_copies[split]).nnz == 0 for split in matrices)


def test_upstream_fingerprint_and_config_mismatch_fail() -> None:
    states = list(_states())
    states[3] = replace(states[3], categorical_encoder_fingerprint="wrong")
    with pytest.raises(PreprocessingVerificationError, match="Phase 5/6"):
        _verify(states=tuple(states))

    config = replace(
        load_preprocessing_verification_config(CONFIG_PATH), output_dtype="float32"
    )
    with pytest.raises(PreprocessingVerificationError, match="differs"):
        _verify(config=config)


def test_zero_variance_train_feature_is_reported_and_fails_gate() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    matrices["train"] = sparse.csr_matrix(
        np.column_stack(
            [np.ones(4), np.arange(4), np.arange(4) + 1, [0, 1, 0, 1]]
        ),
        dtype=np.float64,
    )
    evidence = build_training_feature_variance_evidence(
        matrix=matrices["train"], feature_names=FEATURE_NAMES
    )
    assert evidence.loc[0, "zero_variance"]

    with pytest.raises(PreprocessingVerificationError, match="zero-variance"):
        _verify(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            schemas=schemas,
        )


def test_evidence_tables_are_stable_and_reconciled() -> None:
    matrices, targets, identifiers, timestamps, schemas = _valid_inputs()
    states = _states()
    contract = _verify(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        schemas=schemas,
        states=states,
    )
    split_evidence = build_preprocessing_verification_evidence(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        contract=contract,
    )
    schema_evidence = build_final_feature_schema_evidence(
        fitted_categorical=states[1],
        fitted_numeric=states[2],
        fitted_composition=states[3],
    )

    assert split_evidence["status"].eq("PASS").all()
    assert split_evidence["non_finite_count"].eq(0).all()
    assert schema_evidence["feature_name"].tolist() == list(FEATURE_NAMES)
    assert schema_evidence["source_branch"].eq("numeric").all()
