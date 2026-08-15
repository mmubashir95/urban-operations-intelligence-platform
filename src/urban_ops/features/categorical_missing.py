"""Represent governed categorical nulls with one explicit constant token.

This module provides stateless, non-mutating categorical missing-value handling.
It neither activates conditional features nor learns statistics, groups rare or
unseen categories, processes numeric values, encodes data, or persists outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import yaml
from pandas.api.types import is_object_dtype, is_string_dtype

from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.temporal import REQUIRED_SPLITS, SUPPORTED_TEMPORAL_FEATURES


MISSING_PROVENANCE_ATTR: Final = "urban_ops_categorical_missing"
SUPPORTED_STRATEGY: Final = "constant"


class CategoricalMissingError(ValueError):
    """Raised when categorical missing-value configuration or input is unsafe."""


@dataclass(frozen=True)
class CategoricalMissingConfig:
    """Validated constant-token configuration for categorical null handling."""

    config_version: int
    strategy: str
    token: str
    supported_columns: tuple[str, ...]
    reserved_tokens: tuple[str, ...]


def _required_string_list(value: object, field: str) -> tuple[str, ...]:
    """Return a unique, non-empty string tuple from configuration input."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise CategoricalMissingError(f"{field} must be a non-empty string list.")
    values = tuple(item.strip() for item in value)
    if len(values) != len(set(values)):
        raise CategoricalMissingError(f"{field} must not contain duplicates.")
    return values


def load_categorical_missing_config(path: Path | str) -> CategoricalMissingConfig:
    """Load and validate the categorical constant-token configuration.

    Raises:
        CategoricalMissingError: If the YAML is missing, malformed, ambiguous,
            or requests a strategy other than constant replacement.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise CategoricalMissingError(
            f"Categorical missing-value config does not exist: {config_path}"
        )
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CategoricalMissingError(
            f"Categorical missing-value YAML is invalid: {config_path}"
        ) from error
    if not isinstance(payload, dict):
        raise CategoricalMissingError("Categorical missing-value config must be a mapping.")
    required = {
        "config_version",
        "strategy",
        "token",
        "supported_columns",
        "reserved_tokens",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CategoricalMissingError(f"Configuration is missing fields: {missing}")
    if payload["config_version"] != 1:
        raise CategoricalMissingError("Only config_version 1 is supported.")
    if payload["strategy"] != SUPPORTED_STRATEGY:
        raise CategoricalMissingError(
            f"Only strategy {SUPPORTED_STRATEGY!r} is supported."
        )
    token = payload["token"]
    if not isinstance(token, str) or not token.strip():
        raise CategoricalMissingError("token must be non-empty text.")
    token = token.strip()
    supported_columns = _required_string_list(
        payload["supported_columns"], "supported_columns"
    )
    reserved_tokens = _required_string_list(payload["reserved_tokens"], "reserved_tokens")
    if token in reserved_tokens:
        raise CategoricalMissingError(
            "The missing token must differ from reserved rare and unknown tokens."
        )
    return CategoricalMissingConfig(
        config_version=1,
        strategy=SUPPORTED_STRATEGY,
        token=token,
        supported_columns=supported_columns,
        reserved_tokens=reserved_tokens,
    )


def active_categorical_feature_names(
    policy: FeaturePolicy,
    config: CategoricalMissingConfig,
) -> tuple[str, ...]:
    """Return supported categorical fields currently approved by policy.

    Configuration declares handler support only. Eligibility still requires an
    ``APPROVED_CANDIDATE`` policy status and the frozen creation allow flag.
    """
    unknown = sorted(set(config.supported_columns).difference(policy.by_name))
    if unknown:
        raise FeaturePolicyError(
            f"Categorical missing-value config references unknown features: {unknown}"
        )
    return tuple(
        feature_name
        for feature_name in config.supported_columns
        if policy.by_name[feature_name].policy_status
        is PolicyStatus.APPROVED_CANDIDATE
        and policy.by_name[feature_name].phase_2_allowed
    )


def _is_categorical_dtype(dtype: object) -> bool:
    """Return whether a dtype can safely represent literal categories."""
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def replace_categorical_missing(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    missing_token: str,
) -> pd.DataFrame:
    """Return a copy with nulls in selected categorical columns made explicit.

    Existing real categories are converted to pandas' nullable string dtype and
    otherwise preserved. A source value equal to ``missing_token`` is rejected
    because it would be indistinguishable from an inserted token. The returned
    frame records in-memory provenance so applying the function again is
    idempotent without accepting an unmarked source collision.

    Raises:
        CategoricalMissingError: If columns, dtypes, token, or source categories
            violate the categorical missing-value contract.
    """
    if not isinstance(missing_token, str) or not missing_token.strip():
        raise CategoricalMissingError("missing_token must be non-empty text.")
    missing_token = missing_token.strip()
    requested = tuple(columns)
    if len(requested) != len(set(requested)):
        raise CategoricalMissingError("Categorical columns must be unique.")
    missing_columns = sorted(set(requested).difference(frame.columns))
    if missing_columns:
        raise CategoricalMissingError(
            f"Categorical source columns are missing: {missing_columns}"
        )
    temporal_columns = sorted(set(requested).intersection(SUPPORTED_TEMPORAL_FEATURES))
    if temporal_columns:
        raise CategoricalMissingError(
            "Deterministic temporal features must satisfy their source contract; "
            f"categorical filling is forbidden for {temporal_columns}."
        )

    provenance = frame.attrs.get(MISSING_PROVENANCE_ATTR, {})
    if not isinstance(provenance, dict):
        provenance = {}
    for column in requested:
        if not _is_categorical_dtype(frame[column].dtype):
            raise CategoricalMissingError(
                f"Column {column!r} must be categorical or string; "
                f"received dtype {frame[column].dtype}."
            )
        values = frame[column].astype("string")
        contains_token = bool(values.eq(missing_token).fillna(False).any())
        previously_generated = provenance.get(column) == missing_token
        if contains_token and not previously_generated:
            raise CategoricalMissingError(
                f"Column {column!r} already contains reserved missing token "
                f"{missing_token!r}; source collision is ambiguous."
            )

    result = frame.copy(deep=True)
    result_provenance = dict(provenance)
    for column in requested:
        result[column] = result[column].astype("string").fillna(missing_token)
        result_provenance[column] = missing_token
    if requested:
        result.attrs[MISSING_PROVENANCE_ATTR] = result_provenance
    return result


def replace_policy_approved_categorical_missing(
    frame: pd.DataFrame,
    *,
    policy: FeaturePolicy,
    config: CategoricalMissingConfig,
) -> pd.DataFrame:
    """Apply the constant rule only to policy-approved categorical fields."""
    return replace_categorical_missing(
        frame,
        columns=active_categorical_feature_names(policy, config),
        missing_token=config.token,
    )


def replace_split_categorical_missing(
    frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeaturePolicy,
    config: CategoricalMissingConfig,
) -> dict[str, pd.DataFrame]:
    """Apply the identical governed categorical rule to every required split."""
    missing = [split for split in REQUIRED_SPLITS if split not in frames]
    if missing:
        raise CategoricalMissingError(f"Missing required split frames: {missing}")
    return {
        split: replace_policy_approved_categorical_missing(
            frames[split], policy=policy, config=config
        )
        for split in REQUIRED_SPLITS
    }


def build_categorical_missing_evidence(
    source_frames: Mapping[str, pd.DataFrame],
    transformed_frames: Mapping[str, pd.DataFrame],
    *,
    policy: FeaturePolicy,
    config: CategoricalMissingConfig,
) -> pd.DataFrame:
    """Build split-level missingness, collision, and governance evidence."""
    active = set(active_categorical_feature_names(policy, config))
    rows: list[dict[str, object]] = []
    for split in REQUIRED_SPLITS:
        source = source_frames[split]
        transformed = transformed_frames[split]
        for feature_name in config.supported_columns:
            if feature_name not in source or feature_name not in transformed:
                raise CategoricalMissingError(
                    f"Split {split!r} lacks reviewed categorical field {feature_name!r}."
                )
            before = source[feature_name].astype("string")
            after = transformed[feature_name].astype("string")
            missing_before = int(before.isna().sum())
            source_token_count = int(before.eq(config.token).fillna(False).sum())
            token_after = int(after.eq(config.token).fillna(False).sum())
            null_after = int(after.isna().sum())
            applied = feature_name in active
            reconciled = (
                source_token_count == 0
                and (
                    (applied and token_after == missing_before and null_after == 0)
                    or (
                        not applied
                        and token_after == 0
                        and null_after == missing_before
                        and source[feature_name].equals(transformed[feature_name])
                    )
                )
            )
            rows.append(
                {
                    "split_name": split,
                    "feature_name": feature_name,
                    "policy_status": policy.by_name[feature_name].policy_status.value,
                    "active": applied,
                    "missing_count_before": missing_before,
                    "missing_rate_before": missing_before / len(source),
                    "source_token_collision": source_token_count > 0,
                    "missing_token_count_after": token_after,
                    "null_count_after": null_after,
                    "row_count": len(transformed),
                    "dtype_before": str(source[feature_name].dtype),
                    "dtype_after": str(transformed[feature_name].dtype),
                    "categorical_dtype_preserved": _is_categorical_dtype(
                        transformed[feature_name].dtype
                    ),
                    "transformation_applied": applied,
                    "reconciled": reconciled,
                    "status": "PASS" if reconciled else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def build_categorical_reconciliation_table(
    source_frames: Mapping[str, pd.DataFrame],
    transformed_frames: Mapping[str, pd.DataFrame],
    *,
    identifier_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Reconcile split identity and ordering after categorical replacement."""
    rows: list[dict[str, object]] = []
    for split in REQUIRED_SPLITS:
        source = source_frames[split]
        transformed = transformed_frames[split]
        rows.append(
            {
                "split_name": split,
                "rows_before": len(source),
                "rows_after": len(transformed),
                "row_count_preserved": len(source) == len(transformed),
                "index_preserved": source.index.equals(transformed.index),
                "row_order_preserved": source[identifier_column]
                .reset_index(drop=True)
                .equals(transformed[identifier_column].reset_index(drop=True)),
                "unique_key_preserved": source[identifier_column].equals(
                    transformed[identifier_column]
                ),
                "target_preserved": source[target_column].equals(
                    transformed[target_column]
                ),
            }
        )
    return pd.DataFrame(rows)
