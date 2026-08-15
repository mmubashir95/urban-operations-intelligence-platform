"""Tests for deterministic, governed categorical missing-value handling."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from urban_ops.features.categorical_missing import (
    CategoricalMissingConfig,
    CategoricalMissingError,
    active_categorical_feature_names,
    build_categorical_missing_evidence,
    build_categorical_reconciliation_table,
    load_categorical_missing_config,
    replace_categorical_missing,
    replace_policy_approved_categorical_missing,
    replace_split_categorical_missing,
)
from urban_ops.features.policy import PolicyStatus, load_feature_policy


CONFIG_PATH = Path("configs/features/resolution_risk_categorical_missing.yaml")
POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")


def _config_payload() -> dict[str, object]:
    """Return a mutable copy of the repository missing-value configuration."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one intentionally modified config for validation tests."""
    path = tmp_path / "categorical_missing.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _identity_frame(values: list[object]) -> pd.DataFrame:
    """Build a small categorical frame with identity and target columns."""
    frame = pd.DataFrame(
        {
            "unique_key": [f"id-{index}" for index in range(len(values))],
            "missed_resolution_target": [index % 2 == 0 for index in range(len(values))],
            "location_type": values,
        },
        index=pd.Index(range(10, 10 + len(values)), name="source_row"),
    )
    frame["location_type"] = frame["location_type"].astype("string")
    return frame


def test_repository_config_loads_one_constant_missing_token() -> None:
    config = load_categorical_missing_config(CONFIG_PATH)

    assert config == CategoricalMissingConfig(
        config_version=1,
        strategy="constant",
        token="__MISSING__",
        supported_columns=("borough", "location_type", "incident_zip"),
        reserved_tokens=("__RARE__", "__UNKNOWN__"),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "most_frequent", "Only strategy"),
        ("token", "", "non-empty"),
        ("token", "__RARE__", "must differ"),
    ],
)
def test_unsafe_configuration_fails(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _config_payload()
    payload[field] = value

    with pytest.raises(CategoricalMissingError, match=message):
        load_categorical_missing_config(_write_config(tmp_path, payload))


def test_missing_value_is_replaced_and_real_categories_are_preserved() -> None:
    source = _identity_frame(["Residential", pd.NA, "Commercial"])

    result = replace_categorical_missing(
        source, columns=["location_type"], missing_token="__MISSING__"
    )

    assert result["location_type"].tolist() == [
        "Residential",
        "__MISSING__",
        "Commercial",
    ]
    assert str(result["location_type"].dtype) == "string"


def test_no_missing_values_remain_semantically_unchanged() -> None:
    source = _identity_frame(["A", "B", "C"])

    result = replace_categorical_missing(
        source, columns=["location_type"], missing_token="__MISSING__"
    )

    assert result["location_type"].tolist() == ["A", "B", "C"]
    assert result["location_type"].isna().sum() == 0


def test_all_missing_values_receive_the_same_token() -> None:
    source = _identity_frame([pd.NA, pd.NA, "Known", pd.NA])

    result = replace_categorical_missing(
        source, columns=["location_type"], missing_token="__MISSING__"
    )

    assert result["location_type"].eq("__MISSING__").sum() == 3
    assert result["location_type"].isna().sum() == 0


def test_incident_zip_remains_string_and_is_never_numerically_coerced() -> None:
    source = pd.DataFrame(
        {"incident_zip": pd.Series(["00123", pd.NA, "10099"], dtype="string")}
    )

    result = replace_categorical_missing(
        source, columns=["incident_zip"], missing_token="__MISSING__"
    )

    assert result["incident_zip"].tolist() == ["00123", "__MISSING__", "10099"]
    assert str(result["incident_zip"].dtype) == "string"


def test_numeric_category_input_fails_instead_of_being_silently_stringified() -> None:
    source = pd.DataFrame({"incident_zip": [10001.0, float("nan")]})

    with pytest.raises(CategoricalMissingError, match="categorical or string"):
        replace_categorical_missing(
            source, columns=["incident_zip"], missing_token="__MISSING__"
        )


def test_real_missing_token_category_collision_fails() -> None:
    source = _identity_frame(["Residential", "__MISSING__", pd.NA])

    with pytest.raises(CategoricalMissingError, match="source collision"):
        replace_categorical_missing(
            source, columns=["location_type"], missing_token="__MISSING__"
        )


def test_repeat_application_is_value_identical() -> None:
    source = _identity_frame(["Residential", pd.NA, "Commercial"])
    once = replace_categorical_missing(
        source, columns=["location_type"], missing_token="__MISSING__"
    )

    twice = replace_categorical_missing(
        once, columns=["location_type"], missing_token="__MISSING__"
    )

    pd.testing.assert_frame_equal(once, twice)


def test_handler_is_non_mutating_and_preserves_identity_order_and_target() -> None:
    source = _identity_frame(["Residential", pd.NA, "Commercial"])
    snapshot = source.copy(deep=True)

    result = replace_categorical_missing(
        source, columns=["location_type"], missing_token="__MISSING__"
    )

    pd.testing.assert_frame_equal(source, snapshot)
    assert result is not source
    assert result.index.equals(source.index)
    assert result["unique_key"].equals(source["unique_key"])
    assert result["missed_resolution_target"].equals(
        source["missed_resolution_target"]
    )


def test_missing_column_fails_clearly() -> None:
    with pytest.raises(CategoricalMissingError, match="source columns are missing"):
        replace_categorical_missing(
            pd.DataFrame({"borough": ["QUEENS"]}),
            columns=["location_type"],
            missing_token="__MISSING__",
        )


def test_deterministic_temporal_null_is_not_hidden_as_categorical_missingness() -> None:
    source = pd.DataFrame(
        {"created_month": pd.Series([1, pd.NA], dtype="Int8")}
    )

    with pytest.raises(CategoricalMissingError, match="temporal features"):
        replace_categorical_missing(
            source, columns=["created_month"], missing_token="__MISSING__"
        )


def test_conditional_categorical_fields_are_supported_but_not_activated() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = load_categorical_missing_config(CONFIG_PATH)

    assert active_categorical_feature_names(policy, config) == ()
    assert all(
        policy.by_name[name].policy_status is PolicyStatus.CONDITIONAL
        and policy.by_name[name].phase_2_allowed is False
        for name in config.supported_columns
    )


def test_governed_handler_leaves_conditional_and_excluded_values_unchanged() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = load_categorical_missing_config(CONFIG_PATH)
    source = pd.DataFrame(
        {
            "borough": pd.Series(["QUEENS", pd.NA], dtype="string"),
            "location_type": pd.Series([pd.NA, "Residential"], dtype="string"),
            "incident_zip": pd.Series(["10001", pd.NA], dtype="string"),
            "descriptor_2": pd.Series([pd.NA, pd.NA], dtype="string"),
        }
    )
    snapshot = source.copy(deep=True)

    result = replace_policy_approved_categorical_missing(
        source, policy=policy, config=config
    )

    pd.testing.assert_frame_equal(result, snapshot)
    pd.testing.assert_frame_equal(source, snapshot)


def test_split_evidence_and_reconciliation_report_inactive_policy_truthfully() -> None:
    policy = load_feature_policy(POLICY_PATH)
    config = load_categorical_missing_config(CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "unique_key": pd.Series(["a", "b"], dtype="string"),
            "missed_resolution_target": [False, True],
            "borough": pd.Series(["QUEENS", pd.NA], dtype="string"),
            "location_type": pd.Series([pd.NA, "Residential"], dtype="string"),
            "incident_zip": pd.Series(["10001", pd.NA], dtype="string"),
        }
    )
    source = {name: frame.copy(deep=True) for name in ("train", "validation", "test")}

    transformed = replace_split_categorical_missing(
        source, policy=policy, config=config
    )
    evidence = build_categorical_missing_evidence(
        source, transformed, policy=policy, config=config
    )
    reconciliation = build_categorical_reconciliation_table(
        source,
        transformed,
        identifier_column="unique_key",
        target_column="missed_resolution_target",
    )

    assert evidence["active"].eq(False).all()
    assert evidence["transformation_applied"].eq(False).all()
    assert evidence["status"].eq("PASS").all()
    assert evidence["null_count_after"].equals(evidence["missing_count_before"])
    assert reconciliation[
        [
            "row_count_preserved",
            "index_preserved",
            "row_order_preserved",
            "unique_key_preserved",
            "target_preserved",
        ]
    ].all().all()
