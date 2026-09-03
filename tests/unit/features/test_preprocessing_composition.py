"""Tests for Phase 7 preprocessing block composition."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from urban_ops.features.categorical_encoding import (
    EncodedCategoryState,
    FittedCategoricalEncoder,
)
from urban_ops.features.numeric_preprocessing import (
    FittedNumericPreprocessor,
    NumericFeatureState,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_composition import (
    PreprocessingCompositionError,
    build_preprocessing_composition,
    build_preprocessing_composition_evidence,
    compose_preprocessing_blocks,
    compose_split_preprocessing_blocks,
    load_preprocessing_composition_config,
)


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
COMPOSITION_CONFIG_PATH = Path(
    "configs/features/resolution_risk_preprocessing_composition.yaml"
)


def _repository_config():
    return load_preprocessing_composition_config(COMPOSITION_CONFIG_PATH)


def _categorical_state(
    *,
    names: tuple[str, ...] = ("cat_A", "cat_B"),
    policy_version: int = 1,
) -> FittedCategoricalEncoder:
    return FittedCategoricalEncoder(
        config_version=1,
        policy_version=policy_version,
        cardinality_fingerprint="cardinality-fingerprint",
        strategy="one_hot",
        active_columns=("synthetic_category",) if names else (),
        missing_token="__MISSING__",
        rare_token="__RARE__",
        unknown_token="__UNKNOWN__",
        handle_unknown="ignore",
        drop=None,
        sparse_output=True,
        columns=tuple(
            [
                EncodedCategoryState(
                    feature_name="synthetic_category",
                    source_category_count=len(names),
                    encoder_categories=names,
                    encoded_feature_names=names,
                )
            ]
            if names
            else []
        ),
        encoder=None,
    )


def _numeric_state(
    *,
    names: tuple[str, ...] = ("created_hour", "is_weekend"),
    policy_version: int = 1,
) -> FittedNumericPreprocessor:
    return FittedNumericPreprocessor(
        config_version=1,
        policy_version=policy_version,
        active_columns=names,
        output_feature_names=names,
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
            for name in names
        ),
    )


def _composition(
    *,
    categorical_names: tuple[str, ...] = ("cat_A", "cat_B"),
    numeric_names: tuple[str, ...] = ("created_hour", "is_weekend"),
):
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    return build_preprocessing_composition(
        fitted_categorical=_categorical_state(names=categorical_names),
        fitted_numeric=_numeric_state(names=numeric_names),
        policy=policy,
        config=config,
    )


def test_composition_order_places_categorical_names_before_numeric_names() -> None:
    fitted = _composition()

    assert fitted.categorical_feature_names == ("cat_A", "cat_B")
    assert fitted.numeric_feature_names == ("created_hour", "is_weekend")
    assert fitted.combined_feature_names == (
        "cat_A",
        "cat_B",
        "created_hour",
        "is_weekend",
    )
    assert fitted.composition_order == ("categorical", "numeric")


def test_matrix_value_order_is_exact_and_float64_csr() -> None:
    fitted = _composition()
    categorical = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.int8)
    numeric = sparse.csr_matrix([[15, 1], [3, 0]], dtype=np.float64)

    combined = compose_preprocessing_blocks(
        categorical_matrix=categorical, numeric_matrix=numeric, fitted=fitted
    )

    assert sparse.isspmatrix_csr(combined)
    assert combined.dtype == np.float64
    np.testing.assert_array_equal(
        combined.toarray(),
        np.array([[1, 0, 15, 1], [0, 1, 3, 0]], dtype=np.float64),
    )


def test_zero_width_categorical_block_composes_with_numeric_only() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    fitted = build_preprocessing_composition(
        fitted_categorical=_categorical_state(names=()),
        fitted_numeric=_numeric_state(
            names=("created_hour", "created_day_of_week", "created_month", "is_weekend")
        ),
        policy=policy,
        config=config,
    )
    categorical = sparse.csr_matrix((3, 0), dtype=np.int8)
    numeric = sparse.csr_matrix(
        [[0, 0, 1, 0], [12, 2, 8, 0], [23, 6, 12, 1]],
        dtype=np.float64,
    )

    combined = compose_preprocessing_blocks(
        categorical_matrix=categorical, numeric_matrix=numeric, fitted=fitted
    )

    assert combined.shape == (3, 4)
    assert fitted.combined_feature_names == fitted.numeric_feature_names
    np.testing.assert_array_equal(combined.toarray(), numeric.toarray())


def test_zero_width_numeric_block_composes_with_categorical_only() -> None:
    fitted = _composition(numeric_names=())
    categorical = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.int8)
    numeric = sparse.csr_matrix((2, 0), dtype=np.float64)

    combined = compose_preprocessing_blocks(
        categorical_matrix=categorical, numeric_matrix=numeric, fitted=fitted
    )

    assert combined.shape == (2, 2)
    assert fitted.combined_feature_names == fitted.categorical_feature_names
    np.testing.assert_array_equal(combined.toarray(), categorical.toarray())


def test_row_count_mismatch_fails_loudly() -> None:
    fitted = _composition()

    with pytest.raises(PreprocessingCompositionError, match="different row counts"):
        compose_preprocessing_blocks(
            categorical_matrix=sparse.csr_matrix((2, 2), dtype=np.int8),
            numeric_matrix=sparse.csr_matrix((3, 2), dtype=np.float64),
            fitted=fitted,
        )


@pytest.mark.parametrize(
    ("categorical_width", "numeric_width", "message"),
    [
        (1, 2, "Categorical matrix width"),
        (2, 1, "Numeric matrix width"),
    ],
)
def test_name_width_mismatch_fails_loudly(
    categorical_width: int, numeric_width: int, message: str
) -> None:
    fitted = _composition()

    with pytest.raises(PreprocessingCompositionError, match=message):
        compose_preprocessing_blocks(
            categorical_matrix=sparse.csr_matrix((2, categorical_width), dtype=np.int8),
            numeric_matrix=sparse.csr_matrix((2, numeric_width), dtype=np.float64),
            fitted=fitted,
        )


def test_duplicate_feature_names_fail_loudly() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()

    with pytest.raises(PreprocessingCompositionError, match="duplicates"):
        build_preprocessing_composition(
            fitted_categorical=_categorical_state(names=("created_hour",)),
            fitted_numeric=_numeric_state(names=("created_hour",)),
            policy=policy,
            config=config,
        )


def test_target_identifier_and_leakage_name_guard_fails_loudly() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()

    with pytest.raises(PreprocessingCompositionError, match="non-approved"):
        build_preprocessing_composition(
            fitted_categorical=_categorical_state(names=()),
            fitted_numeric=_numeric_state(names=("missed_resolution_target",)),
            policy=policy,
            config=config,
        )


def test_composition_is_deterministic_and_does_not_mutate_sparse_inputs() -> None:
    fitted = _composition()
    categorical = sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.int8)
    numeric = sparse.csr_matrix([[15, 1], [3, 0]], dtype=np.float64)
    categorical_before = categorical.copy()
    numeric_before = numeric.copy()

    first = compose_preprocessing_blocks(
        categorical_matrix=categorical, numeric_matrix=numeric, fitted=fitted
    )
    second = compose_preprocessing_blocks(
        categorical_matrix=categorical, numeric_matrix=numeric, fitted=fitted
    )

    assert fitted.fingerprint == _composition().fingerprint
    assert (first != second).nnz == 0
    assert (categorical != categorical_before).nnz == 0
    assert (numeric != numeric_before).nnz == 0


def test_split_composition_and_evidence_reconcile() -> None:
    fitted = _composition()
    categorical_matrices = {
        "train": sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.int8),
        "validation": sparse.csr_matrix([[0, 1]], dtype=np.int8),
        "test": sparse.csr_matrix([[1, 0], [0, 1]], dtype=np.int8),
    }
    numeric_matrices = {
        "train": sparse.csr_matrix([[15, 1], [3, 0]], dtype=np.float64),
        "validation": sparse.csr_matrix([[9, 0]], dtype=np.float64),
        "test": sparse.csr_matrix([[0, 1], [23, 0]], dtype=np.float64),
    }

    combined = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted,
    )
    evidence = build_preprocessing_composition_evidence(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        combined_matrices=combined,
        fitted=fitted,
    )

    assert evidence["status"].eq("PASS").all()
    assert evidence["schema_identical"].all()
    assert evidence["row_alignment_valid"].all()
    assert evidence["combined_column_count"].eq(4).all()
    assert evidence["feature_name_count"].eq(4).all()


def test_upstream_policy_version_mismatch_fails_loudly() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()

    with pytest.raises(PreprocessingCompositionError, match="policy version"):
        build_preprocessing_composition(
            fitted_categorical=_categorical_state(policy_version=999),
            fitted_numeric=_numeric_state(),
            policy=policy,
            config=config,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"config_version": 2}, "config_version"),
        ({"composition_order": ["numeric", "categorical"]}, "composition_order"),
        ({"composition_order": ["categorical", "categorical"]}, "duplicates"),
        ({"matrix_type": "dense"}, "csr_matrix"),
        ({"output_dtype": "int8"}, "float64"),
        ({"learned_statistics": "some"}, "learned statistics"),
        (
            {"required_upstream_phases": {"categorical_encoding": "phase_5"}},
            "required_upstream_phases",
        ),
    ],
)
def test_invalid_composition_config_fails_safely(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    payload = {
        "config_version": 1,
        "composition_order": ["categorical", "numeric"],
        "matrix_type": "csr_matrix",
        "output_dtype": "float64",
        "learned_statistics": "none",
        "required_upstream_phases": {
            "categorical_encoding": "phase_5",
            "numeric_preprocessing": "phase_6",
        },
        "row_alignment": "Matrix row i corresponds to source frame row i.",
        "decision_reason": "Compose existing blocks.",
    }
    payload.update(changes)
    path = tmp_path / "composition.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(PreprocessingCompositionError, match=message):
        load_preprocessing_composition_config(path)
