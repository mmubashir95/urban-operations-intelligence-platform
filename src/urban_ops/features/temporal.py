"""Create deterministic complaint-time features under the frozen policy.

The public builder is stateless, learns no data-dependent values, and returns a
copy. It requires non-null UTC-aware timestamps and never repairs malformed
source data, imputes values, encodes categories, or persists feature matrices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import pandas as pd

from urban_ops.features.policy import FeaturePolicy, PolicyStatus


CREATION_TIMESTAMP_COLUMN: Final = "created_date"
DAY_NAMES: Final = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_NAMES: Final = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
SUPPORTED_TEMPORAL_FEATURES: Final = frozenset(
    {
        "created_hour",
        "created_day_of_week",
        "created_day_name",
        "created_day_of_month",
        "created_week_of_year",
        "created_month",
        "created_month_name",
        "created_quarter",
        "created_year",
        "is_weekend",
    }
)
REQUIRED_SPLITS: Final = ("train", "validation", "test")


class TemporalFeatureError(ValueError):
    """Raised when timestamp input or governed derivation is invalid."""


def validate_creation_timestamp(
    frame: pd.DataFrame,
    *,
    source_column: str = CREATION_TIMESTAMP_COLUMN,
) -> pd.Series:
    """Return a valid non-null UTC-aware creation timestamp series.

    Raises:
        TemporalFeatureError: If the source column is missing, null, not a
            timezone-aware datetime, or not represented in UTC.
    """
    if source_column not in frame.columns:
        raise TemporalFeatureError(
            f"Feature creation requires source column {source_column!r}."
        )
    created = frame[source_column]
    if not isinstance(created.dtype, pd.DatetimeTZDtype):
        raise TemporalFeatureError(
            f"{source_column} must be a timezone-aware datetime series; "
            f"received dtype {created.dtype}."
        )
    if str(created.dt.tz) != "UTC":
        raise TemporalFeatureError(
            f"{source_column} must use UTC without implicit timezone conversion; "
            f"received {created.dt.tz}."
        )
    null_count = int(created.isna().sum())
    if null_count:
        raise TemporalFeatureError(
            f"{source_column} must be non-null; found {null_count} null values."
        )
    return created


def _derive_feature(created: pd.Series, feature_name: str) -> pd.Series:
    """Calculate one supported calendar field from a validated timestamp."""
    day_of_week = created.dt.dayofweek
    if feature_name == "created_hour":
        return created.dt.hour.astype("Int8").rename(feature_name)
    if feature_name == "created_day_of_week":
        return day_of_week.astype("Int8").rename(feature_name)
    if feature_name == "created_day_name":
        return (
            day_of_week.map(dict(enumerate(DAY_NAMES)))
            .astype("string")
            .rename(feature_name)
        )
    if feature_name == "created_day_of_month":
        return created.dt.day.astype("Int8").rename(feature_name)
    if feature_name == "created_week_of_year":
        return created.dt.isocalendar().week.astype("Int8").rename(feature_name)
    if feature_name == "created_month":
        return created.dt.month.astype("Int8").rename(feature_name)
    if feature_name == "created_month_name":
        month_index = created.dt.month.sub(1)
        return (
            month_index.map(dict(enumerate(MONTH_NAMES)))
            .astype("string")
            .rename(feature_name)
        )
    if feature_name == "created_quarter":
        return created.dt.quarter.astype("Int8").rename(feature_name)
    if feature_name == "created_year":
        return created.dt.year.astype("Int16").rename(feature_name)
    if feature_name == "is_weekend":
        return day_of_week.ge(5).rename(feature_name)
    raise TemporalFeatureError(f"Unsupported temporal feature: {feature_name!r}.")


def derive_temporal_features(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    source_column: str = CREATION_TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    """Return a copy with exactly the requested deterministic temporal fields.

    This low-level function centralizes formulas shared by EDA and governed
    feature creation. Eligibility is enforced by
    :func:`derive_approved_temporal_features`.

    Existing derived columns are accepted only when both their values and dtype
    exactly match the canonical derivation. Incorrect values fail clearly.
    """
    requested = tuple(feature_names)
    if len(requested) != len(set(requested)):
        raise TemporalFeatureError("Requested temporal feature names must be unique.")
    unsupported = sorted(set(requested).difference(SUPPORTED_TEMPORAL_FEATURES))
    if unsupported:
        raise TemporalFeatureError(f"Unsupported temporal features: {unsupported}")
    created = validate_creation_timestamp(frame, source_column=source_column)
    expected = {
        feature_name: _derive_feature(created, feature_name)
        for feature_name in requested
    }
    for feature_name, values in expected.items():
        if feature_name in frame.columns and not frame[feature_name].equals(values):
            raise TemporalFeatureError(
                f"Existing derived column {feature_name!r} differs from its "
                "canonical values or dtype."
            )
    result = frame.copy(deep=True)
    for feature_name, values in expected.items():
        result[feature_name] = values
    return result


def approved_temporal_feature_names(policy: FeaturePolicy) -> tuple[str, ...]:
    """Return policy-approved temporal derivations in frozen policy order."""
    source = policy.by_name.get(CREATION_TIMESTAMP_COLUMN)
    if source is None or source.policy_status is not PolicyStatus.SOURCE_ONLY:
        raise TemporalFeatureError(
            "The frozen policy must classify created_date as SOURCE_ONLY."
        )
    approved = tuple(
        entry
        for entry in policy.features
        if entry.policy_status is PolicyStatus.APPROVED_CANDIDATE
        and entry.phase_2_allowed
    )
    invalid_sources = sorted(
        entry.feature_name
        for entry in approved
        if entry.source_column != CREATION_TIMESTAMP_COLUMN
    )
    if invalid_sources:
        raise TemporalFeatureError(
            "Approved temporal features must derive from created_date; "
            f"invalid features: {invalid_sources}."
        )
    unsupported = sorted(
        entry.feature_name
        for entry in approved
        if entry.feature_name not in SUPPORTED_TEMPORAL_FEATURES
    )
    if unsupported:
        raise TemporalFeatureError(
            f"Approved features lack deterministic temporal rules: {unsupported}."
        )
    if not approved:
        raise TemporalFeatureError("The frozen policy approves no temporal features.")
    return tuple(entry.feature_name for entry in approved)


def derive_approved_temporal_features(
    frame: pd.DataFrame,
    *,
    policy: FeaturePolicy,
) -> pd.DataFrame:
    """Return a copy containing only policy-approved temporal derivations."""
    return derive_temporal_features(
        frame,
        feature_names=approved_temporal_feature_names(policy),
        source_column=CREATION_TIMESTAMP_COLUMN,
    )


def derive_split_temporal_features(
    frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeaturePolicy,
) -> dict[str, pd.DataFrame]:
    """Apply identical policy-driven derivation to train, validation, and test."""
    missing = [split for split in REQUIRED_SPLITS if split not in frames]
    if missing:
        raise TemporalFeatureError(f"Missing required split frames: {missing}")
    return {
        split: derive_approved_temporal_features(frames[split], policy=policy)
        for split in REQUIRED_SPLITS
    }


def _domain_valid(frame: pd.DataFrame, feature_name: str) -> bool:
    """Return whether one derived column satisfies its canonical domain."""
    if feature_name not in frame or frame[feature_name].isna().any():
        return False
    values = frame[feature_name]
    if feature_name == "created_hour":
        return bool(values.between(0, 23).all())
    if feature_name == "created_day_of_week":
        return bool(values.between(0, 6).all())
    if feature_name == "created_month":
        return bool(values.between(1, 12).all())
    if feature_name == "is_weekend":
        return str(values.dtype) == "bool" and bool(values.isin([False, True]).all())
    return False


def build_temporal_validation_table(
    source_frames: Mapping[str, pd.DataFrame],
    derived_frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeaturePolicy,
) -> pd.DataFrame:
    """Build domain and repeatability evidence for approved temporal features."""
    feature_names = approved_temporal_feature_names(policy)
    repeated = derive_split_temporal_features(source_frames, policy=policy)
    rows: list[dict[str, object]] = []
    for feature_name in feature_names:
        entry = policy.by_name[feature_name]
        created = all(feature_name in derived_frames[split] for split in REQUIRED_SPLITS)
        split_valid = {
            split: created and _domain_valid(derived_frames[split], feature_name)
            for split in REQUIRED_SPLITS
        }
        deterministic = all(
            created
            and derived_frames[split][feature_name].equals(
                repeated[split][feature_name]
            )
            for split in REQUIRED_SPLITS
        )
        weekend_relationship = None
        if feature_name == "is_weekend" and created:
            weekend_relationship = all(
                derived_frames[split]["is_weekend"].equals(
                    derived_frames[split]["created_day_of_week"]
                    .isin([5, 6])
                    .astype(bool)
                    .rename("is_weekend")
                )
                for split in REQUIRED_SPLITS
            )
        train_values = derived_frames["train"][feature_name]
        complete = created and all(split_valid.values()) and deterministic
        rows.append(
            {
                "feature_name": feature_name,
                "source_column": entry.source_column,
                "policy_status": entry.policy_status.value,
                "created": created,
                "dtype": str(train_values.dtype),
                "non_null_count": int(train_values.notna().sum()),
                "unique_count": int(train_values.nunique(dropna=True)),
                "minimum": train_values.min(),
                "maximum": train_values.max(),
                "domain_valid": all(split_valid.values()),
                "train_valid": split_valid["train"],
                "validation_valid": split_valid["validation"],
                "test_valid": split_valid["test"],
                "weekend_relationship_valid": weekend_relationship,
                "deterministic": deterministic,
                "creation_status": "COMPLETE" if complete else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_feature_reconciliation_table(
    source_frames: Mapping[str, pd.DataFrame],
    derived_frames: Mapping[str, pd.DataFrame],
    *,
    identifier_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Reconcile row identity, source values, and target after derivation."""
    rows: list[dict[str, object]] = []
    for split in REQUIRED_SPLITS:
        source = source_frames[split]
        derived = derived_frames[split]
        rows.append(
            {
                "split_name": split,
                "rows_before": len(source),
                "rows_after": len(derived),
                "row_count_preserved": len(source) == len(derived),
                "index_preserved": source.index.equals(derived.index),
                "row_order_preserved": source[identifier_column]
                .reset_index(drop=True)
                .equals(derived[identifier_column].reset_index(drop=True)),
                "unique_key_preserved": source[identifier_column].equals(
                    derived[identifier_column]
                ),
                "target_preserved": source[target_column].equals(derived[target_column]),
                "source_values_preserved": source.equals(derived.loc[:, source.columns]),
            }
        )
    return pd.DataFrame(rows)
