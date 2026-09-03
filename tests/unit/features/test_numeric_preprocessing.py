"""Tests for governed numeric pass-through preprocessing."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from scipy import sparse

from urban_ops.features.numeric_preprocessing import (
    NumericPreprocessingError,
    active_numeric_feature_names,
    build_numeric_preprocessing_evidence,
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_numeric_preprocessor,
    transform_split_numeric_preprocessor,
)
from urban_ops.features.policy import load_feature_policy


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
NUMERIC_CONFIG_PATH = Path(
    "configs/features/resolution_risk_numeric_preprocessing.yaml"
)


def _repository_config():
    """Load the repository Phase 6 numeric configuration."""
    return load_numeric_preprocessing_config(NUMERIC_CONFIG_PATH)


def _frame() -> pd.DataFrame:
    """Build an identity-bearing frame with governed numeric fields."""
    frame = pd.DataFrame(
        {
            "unique_key": pd.Series(["id-1", "id-2", "id-3"], dtype="string"),
            "missed_resolution_target": pd.Series([0, 1, 0], dtype="Int8"),
            "created_hour": pd.Series([15, 0, 23], dtype="Int8"),
            "created_day_of_week": pd.Series([2, 5, 6], dtype="Int8"),
            "created_month": pd.Series([8, 1, 12], dtype="Int8"),
            "is_weekend": pd.Series([False, True, True], dtype="bool"),
            "latitude": pd.Series([40.7, 40.8, pd.NA], dtype="Float64"),
            "longitude": pd.Series([-73.9, -74.0, pd.NA], dtype="Float64"),
            "closed_date": pd.Series(["later", "later", "later"], dtype="string"),
        }
    )
    frame.index = pd.Index([10, 11, 12], name="source_row")
    return frame


def _fit():
    """Fit the deterministic numeric preprocessor for repository policy/config."""
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    frame = _frame()
    fitted = fit_numeric_preprocessor(frame, policy=policy, config=config)
    return policy, config, frame, fitted


def test_real_policy_returns_only_approved_temporal_numeric_features() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()

    assert active_numeric_feature_names(policy, config) == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert policy.by_name["latitude"].policy_status.value == "CONDITIONAL"
    assert policy.by_name["longitude"].policy_status.value == "CONDITIONAL"
    assert config.by_name["latitude"].strategy == "deferred"
    assert config.by_name["longitude"].strategy == "deferred"


def test_pass_through_values_use_deterministic_feature_order() -> None:
    _, config, frame, fitted = _fit()

    matrix = transform_numeric_preprocessor(frame, fitted=fitted, config=config)

    assert fitted.active_columns == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert fitted.output_feature_names == fitted.active_columns
    assert fitted.learned_statistics == "none"
    assert sparse.issparse(matrix)
    assert matrix.shape == (3, 4)
    assert matrix.toarray().tolist() == [
        [15.0, 2.0, 8.0, 0.0],
        [0.0, 5.0, 1.0, 1.0],
        [23.0, 6.0, 12.0, 1.0],
    ]


@pytest.mark.parametrize(
    ("feature_name", "value", "message"),
    [
        ("created_hour", 24, "above 23"),
        ("created_hour", -1, "below 0"),
        ("created_day_of_week", 7, "above 6"),
        ("created_day_of_week", -1, "below 0"),
        ("created_month", 0, "below 1"),
        ("created_month", 13, "above 12"),
        ("is_weekend", 2, "binary"),
    ],
)
def test_invalid_active_temporal_ranges_fail_loudly(
    feature_name: str, value: object, message: str
) -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    frame = _frame()
    if feature_name == "is_weekend":
        frame[feature_name] = frame[feature_name].astype("Int64")
    frame.loc[10, feature_name] = value

    with pytest.raises(NumericPreprocessingError, match=message):
        fit_numeric_preprocessor(frame, policy=policy, config=config)


def test_null_active_temporal_feature_fails_instead_of_imputing() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    frame = _frame()
    frame.loc[10, "created_hour"] = pd.NA

    with pytest.raises(NumericPreprocessingError, match="instead of imputing"):
        fit_numeric_preprocessor(frame, policy=policy, config=config)


def test_target_identifier_leakage_and_deferred_coordinates_are_excluded() -> None:
    _, config, frame, fitted = _fit()

    matrix = transform_numeric_preprocessor(frame, fitted=fitted, config=config)

    assert "missed_resolution_target" not in fitted.output_feature_names
    assert "unique_key" not in fitted.output_feature_names
    assert "closed_date" not in fitted.output_feature_names
    assert "latitude" not in fitted.output_feature_names
    assert "longitude" not in fitted.output_feature_names
    assert matrix.shape[1] == 4


def test_target_values_do_not_affect_numeric_state_or_output() -> None:
    policy, config, frame, fitted = _fit()
    changed = frame.copy(deep=True)
    changed["missed_resolution_target"] = (
        1 - changed["missed_resolution_target"]
    ).astype("Int8")

    changed_fitted = fit_numeric_preprocessor(
        changed, policy=policy, config=config
    )
    first = transform_numeric_preprocessor(frame, fitted=fitted, config=config)
    second = transform_numeric_preprocessor(
        changed, fitted=changed_fitted, config=config
    )

    assert changed_fitted.to_dict() == fitted.to_dict()
    assert changed_fitted.fingerprint == fitted.fingerprint
    assert (first != second).nnz == 0


def test_transform_is_non_mutating_deterministic_and_row_aligned() -> None:
    _, config, frame, fitted = _fit()
    snapshot = frame.copy(deep=True)

    first = transform_numeric_preprocessor(frame, fitted=fitted, config=config)
    second = transform_numeric_preprocessor(frame, fitted=fitted, config=config)
    repeated_fit = fit_numeric_preprocessor(
        frame, policy=load_feature_policy(POLICY_PATH), config=config
    )

    assert (first != second).nnz == 0
    assert repeated_fit.to_dict() == fitted.to_dict()
    assert repeated_fit.fingerprint == fitted.fingerprint
    assert first.shape[0] == len(frame)
    assert first.toarray()[0].tolist() == [15.0, 2.0, 8.0, 0.0]
    pd.testing.assert_frame_equal(frame, snapshot)


def test_split_transform_and_evidence_reconcile() -> None:
    policy, config, train, fitted = _fit()
    frames = {
        "train": train,
        "validation": train.assign(created_hour=pd.Series([1, 2, 3], index=train.index)),
        "test": train.assign(created_month=pd.Series([2, 3, 4], index=train.index)),
    }

    matrices = transform_split_numeric_preprocessor(
        frames, fitted=fitted, config=config
    )
    evidence = build_numeric_preprocessing_evidence(
        frames, matrices, fitted=fitted, policy=policy, config=config
    ).set_index("feature_name")

    assert {matrix.shape for matrix in matrices.values()} == {(3, 4)}
    assert evidence.loc["created_hour", "status"] == "PASS"
    assert evidence.loc["created_day_of_week", "status"] == "PASS"
    assert evidence.loc["created_month", "status"] == "PASS"
    assert evidence.loc["is_weekend", "status"] == "PASS"
    assert evidence.loc["latitude", "status"] == "DEFERRED"
    assert evidence.loc["longitude", "status"] == "DEFERRED"
    assert evidence["feature_schema_identical"].all()


def test_transform_config_mismatch_fails_loudly() -> None:
    _, config, frame, fitted = _fit()

    with pytest.raises(NumericPreprocessingError, match="does not match"):
        transform_numeric_preprocessor(
            frame, fitted=fitted, config=dataclass_replace(config, dtype="float32")
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("config_version",), 2, "config_version 1"),
        (("supported_columns",), ["created_hour", "created_hour"], "duplicates"),
        (("production_decision",), "GOVERNED_NO_OP", "Unsupported"),
        (("feature_rules", "created_hour", "strategy"), "scale", "Unsupported"),
        (("feature_rules", "created_hour", "minimum"), 24, "minimum cannot exceed"),
        (("output", "matrix_type"), "ndarray", "csr_matrix"),
        (("output", "dtype"), "float32", "float64"),
        (("learned_statistics",), "mean_scale", "must not declare"),
    ],
)
def test_invalid_numeric_config_fails_safely(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    payload = yaml.safe_load(NUMERIC_CONFIG_PATH.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    config_path = tmp_path / "numeric.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(NumericPreprocessingError, match=message):
        load_numeric_preprocessing_config(config_path)


def dataclass_replace(config, **changes):
    """Small local wrapper to avoid exposing dataclasses in test assertions."""
    from dataclasses import replace

    return replace(config, **changes)
