"""Tests for training-fitted rare and unseen categorical handling."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from urban_ops.features.categorical_missing import load_categorical_missing_config
from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.features.rare_unseen import (
    RareUnseenError,
    active_rare_unseen_feature_names,
    build_rare_unseen_evidence,
    fit_rare_unseen_handler,
    load_rare_unseen_config,
    transform_rare_unseen,
    transform_split_rare_unseen,
)


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
MISSING_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_missing.yaml"
)
CARDINALITY_CONFIG_PATH = Path(
    "configs/features/resolution_risk_categorical_cardinality.yaml"
)


def _repository_config():
    """Load the reconciled repository Phase 4 configuration."""
    missing = load_categorical_missing_config(MISSING_CONFIG_PATH)
    return load_rare_unseen_config(CARDINALITY_CONFIG_PATH, missing_config=missing)


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


def _active_config(min_count: int = 2):
    """Return the repository config with a small synthetic-test threshold."""
    return replace(
        _repository_config(),
        min_count=min_count,
        production_decision="ACTIVE",
        production_reason="Synthetic test-only activation.",
    )


def test_repository_config_reconciles_tokens_and_training_evidence() -> None:
    config = _repository_config()

    assert config.config_version == 1
    assert config.strategy == "minimum_count"
    assert config.min_count == 10
    assert config.missing_token == "__MISSING__"
    assert config.rare_token == "__RARE__"
    assert config.unknown_token == "__UNKNOWN__"
    assert config.evaluated_count_thresholds == (10, 25, 50)
    assert config.evaluated_frequency_thresholds == (0.001, 0.005, 0.01)
    assert config.production_decision == "GOVERNED_NO_OP"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "relative_frequency", "Only strategy"),
        ("min_count", 0, "positive integer"),
        ("rare_token", "__UNKNOWN__", "must be distinct"),
        ("supported_columns", ["borough", "borough"], "duplicates"),
        ("config_version", 2, "config_version 1"),
    ],
)
def test_invalid_config_fails_safely(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = yaml.safe_load(CARDINALITY_CONFIG_PATH.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "cardinality.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    missing = load_categorical_missing_config(MISSING_CONFIG_PATH)

    with pytest.raises(RareUnseenError, match=message):
        load_rare_unseen_config(path, missing_config=missing)


def test_production_conditional_fields_are_a_governed_no_op() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = _repository_config()
    source = _frame(["A", "B", "__MISSING__"])
    snapshot = source.copy(deep=True)

    fitted = fit_rare_unseen_handler(source, policy=policy, config=config)
    transformed = transform_rare_unseen(source, fitted=fitted, config=config)

    assert active_rare_unseen_feature_names(policy, config) == ()
    assert fitted.active_columns == ()
    assert fitted.columns == ()
    pd.testing.assert_frame_equal(transformed, snapshot)
    pd.testing.assert_frame_equal(source, snapshot)


def test_fit_learns_rare_categories_from_training_only() -> None:
    train = _frame(["A", "A", "A", "A", "B", "C"])
    fitted = fit_rare_unseen_handler(
        train, policy=_synthetic_policy(), config=_active_config()
    )
    state = fitted.by_name["location_type"]

    assert state.category_counts == (("A", 4), ("B", 1), ("C", 1))
    assert state.training_vocabulary == ("A", "B", "C")
    assert state.retained_categories == ("A",)
    assert state.rare_categories == ("B", "C")


def test_target_values_do_not_affect_fitted_category_state() -> None:
    config = _active_config()
    policy = _synthetic_policy()
    train = _frame(["A", "A", "A", "B"])
    changed_targets = train.copy(deep=True)
    changed_targets["missed_resolution_target"] = (
        1 - changed_targets["missed_resolution_target"]
    ).astype("Int8")

    original_state = fit_rare_unseen_handler(
        train, policy=policy, config=config
    )
    changed_target_state = fit_rare_unseen_handler(
        changed_targets, policy=policy, config=config
    )

    assert original_state == changed_target_state


def test_validation_frequency_cannot_change_training_rarity_or_state() -> None:
    config = _active_config()
    policy = _synthetic_policy()
    train = _frame(["A", "A", "A", "B"])
    validation = _frame(["B"] * 6)
    fitted = fit_rare_unseen_handler(train, policy=policy, config=config)
    before = fitted.to_dict()
    before_fingerprint = fitted.fingerprint

    transformed = transform_rare_unseen(
        validation, fitted=fitted, config=config
    )

    assert transformed["location_type"].eq("__RARE__").all()
    assert fitted.by_name["location_type"].rare_categories == ("B",)
    assert fitted.to_dict() == before
    assert fitted.fingerprint == before_fingerprint


def test_transform_distinguishes_retained_rare_unknown_and_missing() -> None:
    config = _active_config()
    policy = _synthetic_policy()
    train = _frame(["A", "A", "A", "B", "C", "__MISSING__"])
    later = _frame(["A", "B", "C", "D", "__MISSING__"])
    fitted = fit_rare_unseen_handler(train, policy=policy, config=config)

    transformed = transform_rare_unseen(later, fitted=fitted, config=config)

    assert transformed["location_type"].tolist() == [
        "A",
        "__RARE__",
        "__RARE__",
        "__UNKNOWN__",
        "__MISSING__",
    ]


@pytest.mark.parametrize("reserved", ["__RARE__", "__UNKNOWN__"])
def test_raw_reserved_token_collision_fails(reserved: str) -> None:
    with pytest.raises(RareUnseenError, match="reserved rare/unknown token"):
        fit_rare_unseen_handler(
            _frame(["A", reserved]),
            policy=_synthetic_policy(),
            config=_active_config(),
        )


def test_null_input_requires_phase_3_handling_first() -> None:
    source = _frame(["A", "B"])
    source.loc[100, "location_type"] = pd.NA

    with pytest.raises(RareUnseenError, match="Phase 3"):
        fit_rare_unseen_handler(
            source, policy=_synthetic_policy(), config=_active_config()
        )


def test_fit_and_transform_are_deterministic_non_mutating_and_idempotent() -> None:
    config = _active_config()
    policy = _synthetic_policy()
    train = _frame(["B", "A", "A", "C"])
    source = _frame(["D", "A", "B", "__MISSING__"])
    snapshot = source.copy(deep=True)

    first_fit = fit_rare_unseen_handler(train, policy=policy, config=config)
    second_fit = fit_rare_unseen_handler(train, policy=policy, config=config)
    first = transform_rare_unseen(source, fitted=first_fit, config=config)
    repeated = transform_rare_unseen(first, fitted=first_fit, config=config)
    second = transform_rare_unseen(source, fitted=second_fit, config=config)

    assert first_fit == second_fit
    assert first_fit.fingerprint == second_fit.fingerprint
    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(source, snapshot)
    assert first.index.equals(source.index)
    assert first["unique_key"].equals(source["unique_key"])
    assert first["missed_resolution_target"].equals(
        source["missed_resolution_target"]
    )
    assert first["borough"].equals(source["borough"])
    assert first["incident_zip"].equals(source["incident_zip"])


def test_split_transform_and_evidence_reconcile_training_only_mapping() -> None:
    config = _active_config()
    policy = _synthetic_policy()
    frames = {
        "train": _frame(["A", "A", "A", "A", "B", "C"]),
        "validation": _frame(["B", "B", "B", "D", "__MISSING__"]),
        "test": _frame(["C", "A", "E"]),
    }
    fitted = fit_rare_unseen_handler(
        frames["train"], policy=policy, config=config
    )
    before = fitted.fingerprint

    transformed = transform_split_rare_unseen(
        frames, fitted=fitted, config=config
    )
    evidence = build_rare_unseen_evidence(
        frames,
        transformed,
        fitted=fitted,
        policy=policy,
        config=config,
    ).set_index("feature_name")

    row = evidence.loc["location_type"]
    assert row["active"]
    assert row["training_distinct_count"] == 3
    assert row["retained_category_count"] == 1
    assert row["rare_category_count"] == 2
    assert row["train_rows_mapped_rare"] == 2
    assert row["train_rare_fraction"] == pytest.approx(2 / 6)
    assert row["validation_unseen_value_count"] == 1
    assert row["validation_rows_mapped_unknown"] == 1
    assert row["test_unseen_value_count"] == 1
    assert row["test_rows_mapped_unknown"] == 1
    assert row["reconciled"]
    assert row["status"] == "PASS"
    assert not bool(evidence.loc["borough", "active"])
    assert not bool(evidence.loc["borough", "transformation_applied"])
    assert fitted.fingerprint == before
