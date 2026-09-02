"""Tests for governed one-hot categorical encoding."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml
from scipy import sparse

from urban_ops.features.categorical_encoding import (
    CategoricalEncodingError,
    active_encoding_feature_names,
    build_categorical_encoding_evidence,
    fit_categorical_encoder,
    load_categorical_encoding_config,
    transform_categorical_encoder,
    transform_split_categorical_encoder,
)
from urban_ops.features.categorical_missing import load_categorical_missing_config
from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.features.rare_unseen import (
    fit_rare_unseen_handler,
    load_rare_unseen_config,
    transform_rare_unseen,
)


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


def _repository_configs():
    """Load reconciled Phase 3, 4, and 5 repository configs."""
    missing = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH, missing_config=missing
    )
    encoding = load_categorical_encoding_config(
        ENCODING_CONFIG_PATH, cardinality_config=cardinality
    )
    return missing, cardinality, encoding


def _synthetic_policy(feature_name: str = "location_type"):
    """Return an in-memory policy with one categorical field explicitly active."""
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
        if entry.feature_name == feature_name
        else entry
        for entry in policy.features
    )
    return replace(policy, features=features)


def _frame(location_values: list[str]) -> pd.DataFrame:
    """Build an identity-bearing frame with all governed categorical fields."""
    size = len(location_values)
    frame = pd.DataFrame(
        {
            "unique_key": pd.Series(
                [f"id-{index}" for index in range(size)], dtype="string"
            ),
            "missed_resolution_target": pd.Series(
                [index % 2 for index in range(size)], dtype="Int8"
            ),
            "borough": pd.Series(["BROOKLYN"] * size, dtype="string"),
            "location_type": pd.Series(location_values, dtype="string"),
            "incident_zip": pd.Series(["10001"] * size, dtype="string"),
        }
    )
    frame.index = pd.Index(range(100, 100 + size), name="source_row")
    return frame


def _active_configs(min_count: int = 2):
    """Return Phase 4 and 5 configs activated only for synthetic tests."""
    _, cardinality, encoding = _repository_configs()
    return (
        replace(
            cardinality,
            min_count=min_count,
            production_decision="ACTIVE",
            production_reason="Synthetic test-only activation.",
        ),
        replace(
            encoding,
            production_decision="ACTIVE",
            production_reason="Synthetic test-only activation.",
        ),
    )


def _fit_synthetic_encoder(train_values: list[str]):
    """Fit Phase 4 and Phase 5 artifacts for one active synthetic feature."""
    policy = _synthetic_policy()
    cardinality_config, encoding_config = _active_configs()
    train = _frame(train_values)
    fitted_cardinality = fit_rare_unseen_handler(
        train, policy=policy, config=cardinality_config
    )
    phase_4_train = transform_rare_unseen(
        train, fitted=fitted_cardinality, config=cardinality_config
    )
    fitted_encoder = fit_categorical_encoder(
        phase_4_train,
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    return (
        policy,
        cardinality_config,
        encoding_config,
        fitted_cardinality,
        phase_4_train,
        fitted_encoder,
    )


def _active_indices(fitted, encoded_row) -> list[str]:
    """Return active encoded feature names for one sparse row."""
    names = fitted.encoded_feature_names
    dense = encoded_row.toarray().ravel()
    return [name for name, value in zip(names, dense, strict=True) if value == 1]


def test_repository_config_selects_explicit_sparse_one_hot_policy() -> None:
    _, cardinality_config, encoding_config = _repository_configs()

    assert encoding_config.config_version == 1
    assert encoding_config.strategy == "one_hot"
    assert encoding_config.supported_columns == cardinality_config.supported_columns
    assert encoding_config.handle_unknown == "ignore"
    assert encoding_config.drop is None
    assert encoding_config.sparse_output is True
    assert encoding_config.include_missing is True
    assert encoding_config.include_rare is True
    assert encoding_config.include_unknown is True
    assert encoding_config.production_decision == "GOVERNED_NO_OP"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("strategy",), "ordinal", "Only strategy"),
        (("supported_columns",), ["borough", "borough"], "duplicates"),
        (("reserved_token_policy", "include_unknown"), False, "explicit missing"),
        (("one_hot", "handle_unknown"), "error", "Unsupported handle_unknown"),
        (("one_hot", "drop"), "first", "drop: null"),
        (("one_hot", "sparse_output"), "yes", "sparse_output"),
        (("production_decision",), "ENCODE", "Unsupported production_decision"),
    ],
)
def test_invalid_encoding_config_fails_safely(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    payload = yaml.safe_load(ENCODING_CONFIG_PATH.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    config_path = tmp_path / "encoding.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _, cardinality_config, _ = _repository_configs()

    with pytest.raises(CategoricalEncodingError, match=message):
        load_categorical_encoding_config(
            config_path, cardinality_config=cardinality_config
        )


def test_real_production_policy_is_governed_zero_column_no_op() -> None:
    policy = load_feature_policy(POLICY_PATH)
    _, cardinality_config, encoding_config = _repository_configs()
    source = _frame(["Street", "__MISSING__", "Bridge"])
    snapshot = source.copy(deep=True)
    fitted_cardinality = fit_rare_unseen_handler(
        source, policy=policy, config=cardinality_config
    )

    fitted_encoder = fit_categorical_encoder(
        source,
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    matrix = transform_categorical_encoder(
        source, fitted=fitted_encoder, config=encoding_config
    )

    assert active_encoding_feature_names(policy, encoding_config) == ()
    assert fitted_encoder.active_columns == ()
    assert fitted_encoder.encoder is None
    assert fitted_encoder.encoded_feature_names == ()
    assert matrix.shape == (3, 0)
    assert sparse.issparse(matrix)
    pd.testing.assert_frame_equal(source, snapshot)


def test_train_only_fit_includes_reserved_tokens_even_absent_from_train_rows() -> None:
    _, _, encoding_config, _, phase_4_train, fitted = _fit_synthetic_encoder(
        ["A", "A", "B", "__MISSING__"]
    )
    state = fitted.by_name["location_type"]

    assert state.encoder_categories == ("A", "__MISSING__", "__RARE__", "__UNKNOWN__")
    assert "location_type___UNKNOWN__" in state.encoded_feature_names
    matrix = transform_categorical_encoder(
        phase_4_train, fitted=fitted, config=encoding_config
    )
    assert matrix.shape == (4, 4)
    assert matrix.nnz == 4


def test_retained_missing_rare_and_unknown_get_dedicated_columns() -> None:
    _, cardinality_config, encoding_config, fitted_cardinality, _, fitted = (
        _fit_synthetic_encoder(["A", "A", "B", "__MISSING__"])
    )
    later = transform_rare_unseen(
        _frame(["A", "B", "C", "__MISSING__"]),
        fitted=fitted_cardinality,
        config=cardinality_config,
    )
    matrix = transform_categorical_encoder(later, fitted=fitted, config=encoding_config)

    assert later["location_type"].tolist() == [
        "A",
        "__RARE__",
        "__UNKNOWN__",
        "__MISSING__",
    ]
    assert _active_indices(fitted, matrix[0]) == ["location_type_A"]
    assert _active_indices(fitted, matrix[1]) == [
        "location_type___RARE__"
    ]
    assert _active_indices(fitted, matrix[2]) == [
        "location_type___UNKNOWN__"
    ]
    assert _active_indices(fitted, matrix[3]) == [
        "location_type___MISSING__"
    ]


def test_validation_frequency_and_values_cannot_change_encoder_state_or_schema() -> None:
    _, cardinality_config, encoding_config, fitted_cardinality, _, fitted = (
        _fit_synthetic_encoder(["A", "A", "B"])
    )
    before = fitted.to_dict()
    before_fingerprint = fitted.fingerprint
    validation = transform_rare_unseen(
        _frame(["B", "B", "B", "C", "C"]),
        fitted=fitted_cardinality,
        config=cardinality_config,
    )

    matrix = transform_categorical_encoder(
        validation, fitted=fitted, config=encoding_config
    )

    assert matrix.shape[1] == fitted.encoded_feature_count
    assert fitted.to_dict() == before
    assert fitted.fingerprint == before_fingerprint
    assert fitted.by_name["location_type"].encoder_categories == (
        "A",
        "__MISSING__",
        "__RARE__",
        "__UNKNOWN__",
    )


def test_split_transform_uses_one_schema_and_preserves_row_alignment() -> None:
    _, cardinality_config, encoding_config, fitted_cardinality, phase_4_train, fitted = (
        _fit_synthetic_encoder(["A", "A", "B", "__MISSING__"])
    )
    frames = {
        "train": phase_4_train,
        "validation": transform_rare_unseen(
            _frame(["A", "C"]), fitted=fitted_cardinality, config=cardinality_config
        ),
        "test": transform_rare_unseen(
            _frame(["B", "__MISSING__"]),
            fitted=fitted_cardinality,
            config=cardinality_config,
        ),
    }
    snapshots = {split: frame.copy(deep=True) for split, frame in frames.items()}

    matrices = transform_split_categorical_encoder(
        frames, fitted=fitted, config=encoding_config
    )
    evidence = build_categorical_encoding_evidence(
        frames,
        matrices,
        fitted=fitted,
        policy=_synthetic_policy(),
        config=encoding_config,
    ).set_index("feature_name")

    assert {matrix.shape[1] for matrix in matrices.values()} == {
        fitted.encoded_feature_count
    }
    assert [matrix.shape[0] for matrix in matrices.values()] == [4, 2, 2]
    assert evidence.loc["location_type", "active"]
    assert evidence.loc["location_type", "reserved_token_columns_present"]
    assert evidence.loc["location_type", "feature_schema_identical"]
    assert evidence["status"].eq("PASS").all()
    for split, frame in frames.items():
        pd.testing.assert_frame_equal(frame, snapshots[split])


def test_defensive_handle_unknown_ignore_does_not_mutate_schema() -> None:
    _, _, encoding_config, _, _, fitted = _fit_synthetic_encoder(["A", "A", "B"])
    unexpected = _frame(["UNMAPPED_RAW_VALUE"])
    before = fitted.fingerprint

    matrix = transform_categorical_encoder(
        unexpected, fitted=fitted, config=encoding_config
    )
    evidence = build_categorical_encoding_evidence(
        {"train": unexpected, "validation": unexpected, "test": unexpected},
        {"train": matrix, "validation": matrix, "test": matrix},
        fitted=fitted,
        policy=_synthetic_policy(),
        config=encoding_config,
    ).set_index("feature_name")

    assert matrix.shape == (1, fitted.encoded_feature_count)
    assert matrix.nnz == 0
    assert fitted.fingerprint == before
    assert evidence.loc[
        "location_type", "validation_defensive_unknown_fallback_count"
    ] == 1


def test_feature_names_and_fitted_state_are_deterministic() -> None:
    *_, first = _fit_synthetic_encoder(["B", "A", "A", "__MISSING__"])
    *_, second = _fit_synthetic_encoder(["B", "A", "A", "__MISSING__"])

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert first.encoded_feature_names == second.encoded_feature_names


def test_target_values_do_not_affect_encoder_state() -> None:
    policy = _synthetic_policy()
    cardinality_config, encoding_config = _active_configs()
    train = _frame(["A", "A", "B", "__MISSING__"])
    fitted_cardinality = fit_rare_unseen_handler(
        train, policy=policy, config=cardinality_config
    )
    phase_4_train = transform_rare_unseen(
        train, fitted=fitted_cardinality, config=cardinality_config
    )
    changed_targets = phase_4_train.copy(deep=True)
    changed_targets["missed_resolution_target"] = (
        1 - changed_targets["missed_resolution_target"]
    ).astype("Int8")

    first = fit_categorical_encoder(
        phase_4_train,
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    second = fit_categorical_encoder(
        changed_targets,
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint


def test_fit_requires_phase_4_transformed_training_values() -> None:
    policy = _synthetic_policy()
    cardinality_config, encoding_config = _active_configs()
    raw_train = _frame(["A", "A", "B"])
    fitted_cardinality = fit_rare_unseen_handler(
        raw_train, policy=policy, config=cardinality_config
    )

    with pytest.raises(CategoricalEncodingError, match="apply Phase 4"):
        fit_categorical_encoder(
            raw_train,
            policy=policy,
            config=encoding_config,
            fitted_cardinality=fitted_cardinality,
        )


def test_config_mismatch_fails_loudly() -> None:
    _, _, encoding_config, _, phase_4_train, fitted = _fit_synthetic_encoder(
        ["A", "A", "B"]
    )

    with pytest.raises(CategoricalEncodingError, match="does not match"):
        transform_categorical_encoder(
            phase_4_train,
            fitted=fitted,
            config=replace(encoding_config, sparse_output=False),
        )


def test_real_policy_file_is_not_modified_by_synthetic_activation() -> None:
    before = POLICY_PATH.read_bytes()
    _fit_synthetic_encoder(["A", "A", "B"])

    assert POLICY_PATH.read_bytes() == before
