"""Fit and apply governed rare and unseen categorical mappings.

The fitted vocabulary is learned from training data only and stored explicitly
in immutable dataclasses. Policy support never activates conditional fields,
and this module does not encode categories, persist artifacts, or train models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

import pandas as pd
import yaml
from pandas.api.types import is_object_dtype, is_string_dtype

from urban_ops.features.categorical_missing import CategoricalMissingConfig
from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_STRATEGY: Final = "minimum_count"
CARDINALITY_PROVENANCE_ATTR: Final = "urban_ops_categorical_cardinality"
SUPPORTED_PRODUCTION_DECISIONS: Final = frozenset({"GOVERNED_NO_OP", "ACTIVE"})


class RareUnseenError(ValueError):
    """Raised when rare/unseen configuration, fitting, or input is unsafe."""


@dataclass(frozen=True)
class RareUnseenConfig:
    """Validated policy for training-fitted rare and unseen categories."""

    config_version: int
    strategy: str
    min_count: int
    supported_columns: tuple[str, ...]
    token_authority: str
    missing_token: str
    rare_token: str
    unknown_token: str
    evaluated_count_thresholds: tuple[int, ...]
    evaluated_frequency_thresholds: tuple[float, ...]
    evidence_source: str
    production_decision: str
    production_reason: str


@dataclass(frozen=True)
class FittedCategoryState:
    """Immutable training-only vocabulary and mapping for one active feature."""

    feature_name: str
    training_row_count: int
    training_distinct_count: int
    category_counts: tuple[tuple[str, int], ...]
    training_vocabulary: tuple[str, ...]
    retained_categories: tuple[str, ...]
    rare_categories: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "feature_name": self.feature_name,
            "training_row_count": self.training_row_count,
            "training_distinct_count": self.training_distinct_count,
            "category_counts": [list(item) for item in self.category_counts],
            "training_vocabulary": list(self.training_vocabulary),
            "retained_categories": list(self.retained_categories),
            "rare_categories": list(self.rare_categories),
        }


@dataclass(frozen=True)
class FittedRareUnseenState:
    """Explicit immutable state fitted exclusively from one training frame."""

    config_version: int
    policy_version: int
    strategy: str
    min_count: int
    missing_token: str
    rare_token: str
    unknown_token: str
    active_columns: tuple[str, ...]
    columns: tuple[FittedCategoryState, ...]

    @property
    def by_name(self) -> dict[str, FittedCategoryState]:
        """Return fitted column states indexed by feature name."""
        return {column.feature_name: column for column in self.columns}

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "config_version": self.config_version,
            "policy_version": self.policy_version,
            "strategy": self.strategy,
            "min_count": self.min_count,
            "missing_token": self.missing_token,
            "rare_token": self.rare_token,
            "unknown_token": self.unknown_token,
            "active_columns": list(self.active_columns),
            "columns": [column.to_dict() for column in self.columns],
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the fitted state."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable configuration error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RareUnseenError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty text from configuration input."""
    if not isinstance(value, str) or not value.strip():
        raise RareUnseenError(f"{field} must be non-empty text.")
    return value.strip()


def _unique_text_list(value: object, field: str) -> tuple[str, ...]:
    """Return a deterministic non-empty tuple from a configured string list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise RareUnseenError(f"{field} must be a non-empty string list.")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise RareUnseenError(f"{field} must not contain duplicates.")
    return result


def _integer_thresholds(value: object, field: str) -> tuple[int, ...]:
    """Validate positive, unique integer diagnostic thresholds."""
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
    ):
        raise RareUnseenError(f"{field} must contain positive integers.")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise RareUnseenError(f"{field} must not contain duplicates.")
    return result


def _frequency_thresholds(value: object, field: str) -> tuple[float, ...]:
    """Validate unique relative-frequency diagnostics in the open unit interval."""
    if not isinstance(value, list) or not value:
        raise RareUnseenError(f"{field} must be a non-empty numeric list.")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RareUnseenError(f"{field} must contain numeric values.")
        number = float(item)
        if not 0 < number < 1:
            raise RareUnseenError(f"{field} values must be between zero and one.")
        result.append(number)
    if len(result) != len(set(result)):
        raise RareUnseenError(f"{field} must not contain duplicates.")
    return tuple(result)


def load_rare_unseen_config(
    path: Path | str,
    *,
    missing_config: CategoricalMissingConfig,
) -> RareUnseenConfig:
    """Load and reconcile cardinality configuration with Phase 3 token authority."""
    config_path = Path(path)
    if not config_path.is_file():
        raise RareUnseenError(f"Rare/unseen config does not exist: {config_path}")
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise RareUnseenError(f"Rare/unseen YAML is invalid: {config_path}") from error
    required = {
        "config_version",
        "strategy",
        "min_count",
        "supported_columns",
        "token_authority",
        "rare_token",
        "unknown_token",
        "threshold_evidence",
        "production_decision",
        "production_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise RareUnseenError(f"Configuration is missing fields: {missing}")
    if root["config_version"] != 1:
        raise RareUnseenError("Only config_version 1 is supported.")
    if root["strategy"] != SUPPORTED_STRATEGY:
        raise RareUnseenError(f"Only strategy {SUPPORTED_STRATEGY!r} is supported.")
    min_count = root["min_count"]
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise RareUnseenError("min_count must be a positive integer.")
    supported_columns = _unique_text_list(root["supported_columns"], "supported_columns")
    if supported_columns != missing_config.supported_columns:
        raise RareUnseenError(
            "supported_columns must match the Phase 3 categorical token authority."
        )
    rare_token = _text(root["rare_token"], "rare_token")
    unknown_token = _text(root["unknown_token"], "unknown_token")
    tokens = (missing_config.token, rare_token, unknown_token)
    if len(set(tokens)) != len(tokens):
        raise RareUnseenError("Missing, rare, and unknown tokens must be distinct.")
    if (rare_token, unknown_token) != missing_config.reserved_tokens:
        raise RareUnseenError(
            "Rare and unknown tokens must match the Phase 3 reserved tokens."
        )
    threshold_evidence = _mapping(root["threshold_evidence"], "threshold_evidence")
    evidence_required = {
        "evaluated_count_thresholds",
        "evaluated_frequency_thresholds",
        "evidence_source",
    }
    evidence_missing = sorted(evidence_required.difference(threshold_evidence))
    if evidence_missing:
        raise RareUnseenError(
            f"threshold_evidence is missing fields: {evidence_missing}"
        )
    evaluated_counts = _integer_thresholds(
        threshold_evidence["evaluated_count_thresholds"],
        "evaluated_count_thresholds",
    )
    if min_count not in evaluated_counts:
        raise RareUnseenError("min_count must be one of the evaluated count thresholds.")
    decision = _text(root["production_decision"], "production_decision")
    if decision not in SUPPORTED_PRODUCTION_DECISIONS:
        raise RareUnseenError(f"Unsupported production_decision {decision!r}.")
    return RareUnseenConfig(
        config_version=1,
        strategy=SUPPORTED_STRATEGY,
        min_count=min_count,
        supported_columns=supported_columns,
        token_authority=_text(root["token_authority"], "token_authority"),
        missing_token=missing_config.token,
        rare_token=rare_token,
        unknown_token=unknown_token,
        evaluated_count_thresholds=evaluated_counts,
        evaluated_frequency_thresholds=_frequency_thresholds(
            threshold_evidence["evaluated_frequency_thresholds"],
            "evaluated_frequency_thresholds",
        ),
        evidence_source=_text(threshold_evidence["evidence_source"], "evidence_source"),
        production_decision=decision,
        production_reason=_text(root["production_reason"], "production_reason"),
    )


def active_rare_unseen_feature_names(
    policy: FeaturePolicy, config: RareUnseenConfig
) -> tuple[str, ...]:
    """Return fields enabled by both the cardinality decision and frozen policy."""
    unknown = sorted(set(config.supported_columns).difference(policy.by_name))
    if unknown:
        raise FeaturePolicyError(
            f"Rare/unseen config references unknown features: {unknown}"
        )
    approved = tuple(
        feature_name
        for feature_name in config.supported_columns
        if policy.by_name[feature_name].policy_status
        is PolicyStatus.APPROVED_CANDIDATE
        and policy.by_name[feature_name].phase_2_allowed
    )
    return approved if config.production_decision == "ACTIVE" else ()


def _is_categorical_dtype(dtype: object) -> bool:
    """Return whether a dtype safely represents literal category values."""
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _validated_values(
    frame: pd.DataFrame,
    column: str,
    *,
    config: RareUnseenConfig,
    allow_generated_tokens_for_fingerprint: str | None = None,
) -> pd.Series:
    """Validate one active input column and return nullable string values."""
    if column not in frame:
        raise RareUnseenError(f"Categorical source column is missing: {column!r}.")
    if not _is_categorical_dtype(frame[column].dtype):
        raise RareUnseenError(
            f"Column {column!r} must be categorical or string; "
            f"received dtype {frame[column].dtype}."
        )
    values = frame[column].astype("string")
    if values.isna().any():
        raise RareUnseenError(
            f"Column {column!r} contains nulls; apply Phase 3 missing handling first."
        )
    generated = {config.rare_token, config.unknown_token}
    contains_generated = bool(values.isin(generated).any())
    provenance = frame.attrs.get(CARDINALITY_PROVENANCE_ATTR, {})
    marked = (
        allow_generated_tokens_for_fingerprint is not None
        and isinstance(provenance, dict)
        and provenance.get(column) == allow_generated_tokens_for_fingerprint
    )
    if contains_generated and not marked:
        raise RareUnseenError(
            f"Column {column!r} contains a reserved rare/unknown token without "
            "matching fitted-state provenance."
        )
    return values


def fit_rare_unseen_handler(
    train: pd.DataFrame,
    *,
    policy: FeaturePolicy,
    config: RareUnseenConfig,
) -> FittedRareUnseenState:
    """Fit deterministic vocabularies from the training frame only.

    Target values are never read. Conditional fields remain inactive even when
    listed as supported by configuration.
    """
    active_columns = active_rare_unseen_feature_names(policy, config)
    fitted_columns: list[FittedCategoryState] = []
    for column in active_columns:
        values = _validated_values(train, column, config=config)
        genuine = values.loc[values.ne(config.missing_token)]
        counts = tuple(
            sorted(
                (
                    (str(category), int(count))
                    for category, count in genuine.value_counts(sort=False).items()
                ),
                key=lambda item: item[0],
            )
        )
        rare = tuple(category for category, count in counts if count < config.min_count)
        retained = tuple(
            category for category, count in counts if count >= config.min_count
        )
        vocabulary = tuple(category for category, _ in counts)
        fitted_columns.append(
            FittedCategoryState(
                feature_name=column,
                training_row_count=len(train),
                training_distinct_count=len(vocabulary),
                category_counts=counts,
                training_vocabulary=vocabulary,
                retained_categories=retained,
                rare_categories=rare,
            )
        )
    return FittedRareUnseenState(
        config_version=config.config_version,
        policy_version=policy.policy_version,
        strategy=config.strategy,
        min_count=config.min_count,
        missing_token=config.missing_token,
        rare_token=config.rare_token,
        unknown_token=config.unknown_token,
        active_columns=active_columns,
        columns=tuple(fitted_columns),
    )


def _map_values(
    values: pd.Series,
    *,
    state: FittedCategoryState,
    fitted: FittedRareUnseenState,
) -> pd.Series:
    """Map values using one frozen training vocabulary without learning state."""
    rare = set(state.rare_categories)
    retained = set(state.retained_categories)

    def map_value(value: str) -> str:
        if value == fitted.missing_token:
            return value
        if value in {fitted.rare_token, fitted.unknown_token}:
            return value
        if value in rare:
            return fitted.rare_token
        if value in retained:
            return value
        return fitted.unknown_token

    return values.map(map_value).astype("string")


def transform_rare_unseen(
    frame: pd.DataFrame,
    *,
    fitted: FittedRareUnseenState,
    config: RareUnseenConfig,
) -> pd.DataFrame:
    """Return a copy transformed only by the supplied immutable fitted state."""
    if (
        fitted.config_version != config.config_version
        or fitted.strategy != config.strategy
        or fitted.min_count != config.min_count
        or fitted.missing_token != config.missing_token
        or fitted.rare_token != config.rare_token
        or fitted.unknown_token != config.unknown_token
    ):
        raise RareUnseenError("Fitted state does not match the rare/unseen config.")
    result = frame.copy(deep=True)
    fingerprint = fitted.fingerprint
    for column in fitted.active_columns:
        values = _validated_values(
            frame,
            column,
            config=config,
            allow_generated_tokens_for_fingerprint=fingerprint,
        )
        result[column] = _map_values(
            values, state=fitted.by_name[column], fitted=fitted
        )
    if fitted.active_columns:
        provenance = result.attrs.get(CARDINALITY_PROVENANCE_ATTR, {})
        if not isinstance(provenance, dict):
            provenance = {}
        updated = dict(provenance)
        updated.update({column: fingerprint for column in fitted.active_columns})
        result.attrs[CARDINALITY_PROVENANCE_ATTR] = updated
    return result


def transform_split_rare_unseen(
    frames: Mapping[str, pd.DataFrame],
    *,
    fitted: FittedRareUnseenState,
    config: RareUnseenConfig,
) -> dict[str, pd.DataFrame]:
    """Transform required splits using one unchanged training-fitted state."""
    missing = [split for split in REQUIRED_SPLITS if split not in frames]
    if missing:
        raise RareUnseenError(f"Missing required split frames: {missing}")
    return {
        split: transform_rare_unseen(frames[split], fitted=fitted, config=config)
        for split in REQUIRED_SPLITS
    }


def _genuine_values(values: pd.Series, config: RareUnseenConfig) -> pd.Series:
    """Exclude only the Phase 3 missing sentinel from categorical evidence."""
    return values.loc[values.notna() & values.ne(config.missing_token)]


def build_rare_unseen_evidence(
    source_frames: Mapping[str, pd.DataFrame],
    transformed_frames: Mapping[str, pd.DataFrame],
    *,
    fitted: FittedRareUnseenState,
    policy: FeaturePolicy,
    config: RareUnseenConfig,
) -> pd.DataFrame:
    """Build one governance, cardinality, mapping, and reconciliation row per feature."""
    for split in REQUIRED_SPLITS:
        if split not in source_frames or split not in transformed_frames:
            raise RareUnseenError(f"Missing required split for evidence: {split!r}.")
    active = set(active_rare_unseen_feature_names(policy, config))
    if tuple(name for name in config.supported_columns if name in active) != fitted.active_columns:
        raise RareUnseenError("Fitted active columns do not reconcile with policy.")
    rows: list[dict[str, object]] = []
    for feature_name in config.supported_columns:
        for split in REQUIRED_SPLITS:
            if (
                feature_name not in source_frames[split]
                or feature_name not in transformed_frames[split]
            ):
                raise RareUnseenError(
                    f"Split {split!r} lacks governed categorical field {feature_name!r}."
                )
        train_values = source_frames["train"][feature_name].astype("string")
        train_genuine = _genuine_values(train_values, config)
        state = fitted.by_name.get(feature_name)
        if state is None:
            retained: set[str] = set()
            rare: set[str] = set()
            vocabulary: set[str] = set()
        else:
            retained = set(state.retained_categories)
            rare = set(state.rare_categories)
            vocabulary = set(state.training_vocabulary)
        evidence_vocabulary = vocabulary if state else set(train_genuine.dropna().tolist())

        metrics: dict[str, int] = {}
        reconciled = True
        for split in REQUIRED_SPLITS:
            before = source_frames[split][feature_name].astype("string")
            after = transformed_frames[split][feature_name].astype("string")
            genuine = _genuine_values(before, config)
            unseen_values = set(genuine.dropna().tolist()).difference(
                evidence_vocabulary
            )
            metrics[f"{split}_unseen_value_count"] = (
                len(unseen_values) if split != "train" else 0
            )
            metrics[f"{split}_rows_mapped_unknown"] = (
                int(after.eq(config.unknown_token).sum())
                if feature_name in active and split != "train"
                else 0
            )
            if feature_name in active and state is not None:
                expected = _map_values(before, state=state, fitted=fitted)
                reconciled = reconciled and expected.equals(after)
            else:
                reconciled = reconciled and source_frames[split][feature_name].equals(
                    transformed_frames[split][feature_name]
                )

        train_rare_rows = int(train_genuine.isin(rare).sum()) if state else 0
        rows.append(
            {
                "feature_name": feature_name,
                "policy_status": policy.by_name[feature_name].policy_status.value,
                "active": feature_name in active,
                "training_row_count": len(source_frames["train"]),
                "training_distinct_count": int(train_genuine.nunique()),
                "retained_category_count": len(retained),
                "rare_category_count": len(rare),
                "train_rows_mapped_rare": train_rare_rows,
                "train_rare_fraction": train_rare_rows / len(source_frames["train"]),
                "validation_unseen_value_count": metrics[
                    "validation_unseen_value_count"
                ],
                "validation_rows_mapped_unknown": metrics[
                    "validation_rows_mapped_unknown"
                ],
                "test_unseen_value_count": metrics["test_unseen_value_count"],
                "test_rows_mapped_unknown": metrics["test_rows_mapped_unknown"],
                "transformation_applied": feature_name in active,
                "reconciled": reconciled,
                "status": "PASS" if reconciled else "FAIL",
            }
        )
    return pd.DataFrame(rows)
