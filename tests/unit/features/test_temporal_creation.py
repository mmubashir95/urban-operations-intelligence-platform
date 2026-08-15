"""Tests for deterministic policy-driven creation-time features."""

from pathlib import Path

import pandas as pd
import pytest

from urban_ops.features.policy import load_feature_policy
from urban_ops.features.temporal import (
    TemporalFeatureError,
    approved_temporal_feature_names,
    derive_approved_temporal_features,
)


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
APPROVED_FEATURES = (
    "created_hour",
    "created_day_of_week",
    "created_month",
    "is_weekend",
)
NON_APPROVED_TEMPORAL_FEATURES = {
    "created_day_name",
    "created_month_name",
    "created_quarter",
    "created_week_of_year",
    "created_day_of_month",
    "created_year",
}


def _frame(values: list[str]) -> pd.DataFrame:
    """Return a UTC-aware frame with stable identity and target columns."""
    frame = pd.DataFrame(
        {
            "unique_key": [f"id-{index}" for index in range(len(values))],
            "created_date": pd.to_datetime(values, utc=True),
            "missed_resolution_target": [index % 2 for index in range(len(values))],
        },
        index=pd.Index(range(100, 100 + len(values)), name="source_index"),
    )
    frame["missed_resolution_target"] = frame["missed_resolution_target"].astype("Int8")
    return frame


def test_hour_derivation_handles_start_middle_and_end_of_day() -> None:
    frame = _frame(
        ["2024-01-01 00:00", "2024-07-15 14:37", "2024-12-31 23:59"]
    )

    result = derive_approved_temporal_features(
        frame, policy=load_feature_policy(POLICY_PATH)
    )

    assert result["created_hour"].tolist() == [0, 14, 23]
    assert str(result["created_hour"].dtype) == "Int8"


def test_weekday_derivation_uses_monday_zero_pandas_convention() -> None:
    frame = _frame(
        ["2024-07-15", "2024-07-19", "2024-07-20", "2024-07-21"]
    )

    result = derive_approved_temporal_features(
        frame, policy=load_feature_policy(POLICY_PATH)
    )

    assert result["created_day_of_week"].tolist() == [0, 4, 5, 6]
    assert result["is_weekend"].tolist() == [False, False, True, True]
    assert str(result["created_day_of_week"].dtype) == "Int8"
    assert str(result["is_weekend"].dtype) == "bool"


def test_month_derivation_uses_calendar_month_numbers() -> None:
    frame = _frame(["2024-01-15", "2024-07-15", "2024-12-15"])

    result = derive_approved_temporal_features(
        frame, policy=load_feature_policy(POLICY_PATH)
    )

    assert result["created_month"].tolist() == [1, 7, 12]
    assert str(result["created_month"].dtype) == "Int8"


def test_builder_uses_only_the_frozen_policy_allow_list() -> None:
    policy = load_feature_policy(POLICY_PATH)
    frame = _frame(["2024-07-15 14:37"])
    frame["borough"] = "BROOKLYN"

    result = derive_approved_temporal_features(frame, policy=policy)
    added = tuple(column for column in result.columns if column not in frame.columns)

    assert approved_temporal_feature_names(policy) == APPROVED_FEATURES
    assert added == APPROVED_FEATURES
    assert NON_APPROVED_TEMPORAL_FEATURES.isdisjoint(result.columns)
    assert result["borough"].equals(frame["borough"])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"other": [1]}), "created_date"),
        (pd.DataFrame({"created_date": ["2024-07-15"]}), "timezone-aware"),
        (
            pd.DataFrame({"created_date": pd.to_datetime(["2024-07-15"])}),
            "timezone-aware",
        ),
        (
            pd.DataFrame(
                {
                    "created_date": pd.Series(
                        [pd.Timestamp("2024-07-15", tz="UTC"), pd.NaT],
                        dtype="datetime64[ns, UTC]",
                    )
                }
            ),
            "non-null",
        ),
        (
            pd.DataFrame(
                {
                    "created_date": pd.Series(
                        [pd.Timestamp("2024-07-15", tz="America/New_York")]
                    )
                }
            ),
            "must use UTC",
        ),
    ],
)
def test_invalid_creation_timestamp_contract_fails_clearly(
    frame: pd.DataFrame, message: str
) -> None:
    with pytest.raises(TemporalFeatureError, match=message):
        derive_approved_temporal_features(
            frame, policy=load_feature_policy(POLICY_PATH)
        )


def test_builder_preserves_input_values_index_order_and_identity() -> None:
    frame = _frame(["2024-07-21 23:59", "2024-07-15 00:00"])
    before = frame.copy(deep=True)

    result = derive_approved_temporal_features(
        frame, policy=load_feature_policy(POLICY_PATH)
    )

    pd.testing.assert_frame_equal(frame, before)
    assert result.index.equals(before.index)
    assert result["unique_key"].equals(before["unique_key"])
    assert result["missed_resolution_target"].equals(
        before["missed_resolution_target"]
    )
    assert len(result) == len(before)


def test_repeat_derivation_is_exact_and_idempotent() -> None:
    frame = _frame(["2024-07-15 14:37", "2024-07-21 23:59"])
    policy = load_feature_policy(POLICY_PATH)

    first = derive_approved_temporal_features(frame, policy=policy)
    second = derive_approved_temporal_features(frame, policy=policy)
    idempotent = derive_approved_temporal_features(first, policy=policy)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, idempotent)


def test_incorrect_preexisting_derived_value_fails() -> None:
    frame = _frame(["2024-07-15 14:37"])
    frame["created_hour"] = pd.Series([3], index=frame.index, dtype="Int8")

    with pytest.raises(TemporalFeatureError, match="created_hour"):
        derive_approved_temporal_features(
            frame, policy=load_feature_policy(POLICY_PATH)
        )
