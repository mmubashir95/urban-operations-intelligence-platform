"""Fit and apply governed one-hot categorical encoding.

This module consumes Phase 4 categorical values and produces a separate sparse
numeric block. It does not learn rare/unseen mappings, mutate source frames,
build combined preprocessing matrices, process numeric features, or train
models.
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
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.rare_unseen import FittedRareUnseenState, RareUnseenConfig
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_STRATEGY: Final = "one_hot"
SUPPORTED_PRODUCTION_DECISIONS: Final = frozenset({"GOVERNED_NO_OP", "ACTIVE"})
SUPPORTED_HANDLE_UNKNOWN: Final = frozenset({"ignore"})
SUPPORTED_DROP: Final = frozenset({None})


class CategoricalEncodingError(ValueError):
    """Raised when categorical encoding configuration or input is unsafe."""


@dataclass(frozen=True)
class CategoricalEncodingConfig:
    """Validated one-hot encoding policy for governed categorical features."""

    config_version: int
    strategy: str
    supported_columns: tuple[str, ...]
    cardinality_authority: str
    token_authority: str
    include_missing: bool
    include_rare: bool
    include_unknown: bool
    reserved_token_reason: str
    handle_unknown: str
    drop: None
    sparse_output: bool
    production_decision: str
    production_reason: str


@dataclass(frozen=True)
class EncodedCategoryState:
    """Immutable one-hot vocabulary for one active categorical feature."""

    feature_name: str
    source_category_count: int
    encoder_categories: tuple[str, ...]
    encoded_feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "feature_name": self.feature_name,
            "source_category_count": self.source_category_count,
            "encoder_categories": list(self.encoder_categories),
            "encoded_feature_names": list(self.encoded_feature_names),
        }


@dataclass(frozen=True)
class FittedCategoricalEncoder:
    """Explicit one-hot encoder state fitted from training-derived categories."""

    config_version: int
    policy_version: int
    cardinality_fingerprint: str
    strategy: str
    active_columns: tuple[str, ...]
    missing_token: str
    rare_token: str
    unknown_token: str
    handle_unknown: str
    drop: None
    sparse_output: bool
    columns: tuple[EncodedCategoryState, ...]
    encoder: OneHotEncoder | None

    @property
    def by_name(self) -> dict[str, EncodedCategoryState]:
        """Return fitted encoded column states indexed by feature name."""
        return {column.feature_name: column for column in self.columns}

    @property
    def encoded_feature_names(self) -> tuple[str, ...]:
        """Return the deterministic full encoded categorical feature schema."""
        return tuple(
            name for column in self.columns for name in column.encoded_feature_names
        )

    @property
    def encoded_feature_count(self) -> int:
        """Return the number of encoded categorical output columns."""
        return len(self.encoded_feature_names)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic representation excluding the sklearn object."""
        return {
            "config_version": self.config_version,
            "policy_version": self.policy_version,
            "cardinality_fingerprint": self.cardinality_fingerprint,
            "strategy": self.strategy,
            "active_columns": list(self.active_columns),
            "missing_token": self.missing_token,
            "rare_token": self.rare_token,
            "unknown_token": self.unknown_token,
            "handle_unknown": self.handle_unknown,
            "drop": self.drop,
            "sparse_output": self.sparse_output,
            "encoded_feature_count": self.encoded_feature_count,
            "encoded_feature_names": list(self.encoded_feature_names),
            "columns": [column.to_dict() for column in self.columns],
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the fitted encoder state."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable configuration error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CategoricalEncodingError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty text from configuration input."""
    if not isinstance(value, str) or not value.strip():
        raise CategoricalEncodingError(f"{field} must be non-empty text.")
    return value.strip()


def _unique_text_list(value: object, field: str) -> tuple[str, ...]:
    """Return a unique, non-empty tuple from a configured string list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise CategoricalEncodingError(f"{field} must be a non-empty string list.")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise CategoricalEncodingError(f"{field} must not contain duplicates.")
    return result


def load_categorical_encoding_config(
    path: Path | str,
    *,
    cardinality_config: RareUnseenConfig,
) -> CategoricalEncodingConfig:
    """Load and reconcile Phase 5 encoding configuration with Phase 4 authority."""
    config_path = Path(path)
    if not config_path.is_file():
        raise CategoricalEncodingError(
            f"Categorical encoding config does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise CategoricalEncodingError(
            f"Categorical encoding YAML is invalid: {config_path}"
        ) from error
    required = {
        "config_version",
        "strategy",
        "supported_columns",
        "cardinality_authority",
        "token_authority",
        "reserved_token_policy",
        "one_hot",
        "production_decision",
        "production_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise CategoricalEncodingError(f"Configuration is missing fields: {missing}")
    if root["config_version"] != 1:
        raise CategoricalEncodingError("Only config_version 1 is supported.")
    if root["strategy"] != SUPPORTED_STRATEGY:
        raise CategoricalEncodingError(
            f"Only strategy {SUPPORTED_STRATEGY!r} is supported."
        )
    supported_columns = _unique_text_list(root["supported_columns"], "supported_columns")
    if supported_columns != cardinality_config.supported_columns:
        raise CategoricalEncodingError(
            "supported_columns must match the Phase 4 cardinality authority."
        )
    reserved = _mapping(root["reserved_token_policy"], "reserved_token_policy")
    one_hot = _mapping(root["one_hot"], "one_hot")
    handle_unknown = _text(one_hot.get("handle_unknown"), "handle_unknown")
    if handle_unknown not in SUPPORTED_HANDLE_UNKNOWN:
        raise CategoricalEncodingError(f"Unsupported handle_unknown {handle_unknown!r}.")
    drop = one_hot.get("drop")
    if drop not in SUPPORTED_DROP:
        raise CategoricalEncodingError("Only drop: null is supported.")
    sparse_output = one_hot.get("sparse_output")
    if not isinstance(sparse_output, bool):
        raise CategoricalEncodingError("sparse_output must be boolean.")
    token_flags = {
        "include_missing": reserved.get("include_missing"),
        "include_rare": reserved.get("include_rare"),
        "include_unknown": reserved.get("include_unknown"),
    }
    if token_flags != {
        "include_missing": True,
        "include_rare": True,
        "include_unknown": True,
    }:
        raise CategoricalEncodingError(
            "Phase 5 requires explicit missing, rare, and unknown token columns."
        )
    decision = _text(root["production_decision"], "production_decision")
    if decision not in SUPPORTED_PRODUCTION_DECISIONS:
        raise CategoricalEncodingError(f"Unsupported production_decision {decision!r}.")
    return CategoricalEncodingConfig(
        config_version=1,
        strategy=SUPPORTED_STRATEGY,
        supported_columns=supported_columns,
        cardinality_authority=_text(
            root["cardinality_authority"], "cardinality_authority"
        ),
        token_authority=_text(root["token_authority"], "token_authority"),
        include_missing=True,
        include_rare=True,
        include_unknown=True,
        reserved_token_reason=_text(reserved.get("reason"), "reserved_token_reason"),
        handle_unknown=handle_unknown,
        drop=None,
        sparse_output=sparse_output,
        production_decision=decision,
        production_reason=_text(root["production_reason"], "production_reason"),
    )


def active_encoding_feature_names(
    policy: FeaturePolicy, config: CategoricalEncodingConfig
) -> tuple[str, ...]:
    """Return categorical fields enabled by config decision and frozen policy."""
    unknown = sorted(set(config.supported_columns).difference(policy.by_name))
    if unknown:
        raise FeaturePolicyError(
            f"Categorical encoding config references unknown features: {unknown}"
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
    """Return whether a dtype safely represents categorical tokens."""
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _validate_config_matches_fitted(
    fitted: FittedCategoricalEncoder, config: CategoricalEncodingConfig
) -> None:
    """Raise if an encoder is transformed under a different config."""
    if (
        fitted.config_version != config.config_version
        or fitted.strategy != config.strategy
        or fitted.handle_unknown != config.handle_unknown
        or fitted.drop != config.drop
        or fitted.sparse_output != config.sparse_output
    ):
        raise CategoricalEncodingError(
            "Fitted encoder does not match the categorical encoding config."
        )


def _frame_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Validate and return one categorical column as nullable string values."""
    if column not in frame:
        raise CategoricalEncodingError(
            f"Categorical source column is missing: {column!r}."
        )
    if not _is_categorical_dtype(frame[column].dtype):
        raise CategoricalEncodingError(
            f"Column {column!r} must be categorical or string; "
            f"received dtype {frame[column].dtype}."
        )
    values = frame[column].astype("string")
    if values.isna().any():
        raise CategoricalEncodingError(
            f"Column {column!r} contains nulls; apply Phase 3 and Phase 4 first."
        )
    return values


def _encoder_vocabulary(
    fitted_cardinality: FittedRareUnseenState,
    feature_name: str,
) -> tuple[str, ...]:
    """Build deterministic one-hot categories from the training-fitted Phase 4 state."""
    state = fitted_cardinality.by_name[feature_name]
    categories = (
        *state.retained_categories,
        fitted_cardinality.missing_token,
        fitted_cardinality.rare_token,
        fitted_cardinality.unknown_token,
    )
    if len(categories) != len(set(categories)):
        raise CategoricalEncodingError(
            f"Encoder categories for {feature_name!r} contain duplicates."
        )
    return categories


def _fit_one_hot_encoder(
    train: pd.DataFrame,
    *,
    active_columns: tuple[str, ...],
    categories: tuple[tuple[str, ...], ...],
    config: CategoricalEncodingConfig,
) -> OneHotEncoder:
    """Fit a sklearn OneHotEncoder with an explicit training-derived vocabulary."""
    encoder = OneHotEncoder(
        categories=[list(values) for values in categories],
        handle_unknown=config.handle_unknown,
        drop=config.drop,
        sparse_output=config.sparse_output,
        dtype="int8",
    )
    encoder.fit(train.loc[:, list(active_columns)].astype("string"))
    return encoder


def fit_categorical_encoder(
    train: pd.DataFrame,
    *,
    policy: FeaturePolicy,
    config: CategoricalEncodingConfig,
    fitted_cardinality: FittedRareUnseenState,
) -> FittedCategoricalEncoder:
    """Fit one-hot encoding state from the training-authoritative Phase 4 artifact."""
    active_columns = active_encoding_feature_names(policy, config)
    if active_columns != fitted_cardinality.active_columns:
        raise CategoricalEncodingError(
            "Categorical encoder active columns must match the Phase 4 fitted state."
        )
    if (
        fitted_cardinality.policy_version != policy.policy_version
        or fitted_cardinality.missing_token == fitted_cardinality.rare_token
        or fitted_cardinality.missing_token == fitted_cardinality.unknown_token
        or fitted_cardinality.rare_token == fitted_cardinality.unknown_token
    ):
        raise CategoricalEncodingError(
            "Phase 4 fitted state does not reconcile with the encoding policy."
        )
    if not active_columns:
        return FittedCategoricalEncoder(
            config_version=config.config_version,
            policy_version=policy.policy_version,
            cardinality_fingerprint=fitted_cardinality.fingerprint,
            strategy=config.strategy,
            active_columns=(),
            missing_token=fitted_cardinality.missing_token,
            rare_token=fitted_cardinality.rare_token,
            unknown_token=fitted_cardinality.unknown_token,
            handle_unknown=config.handle_unknown,
            drop=config.drop,
            sparse_output=config.sparse_output,
            columns=(),
            encoder=None,
        )

    categories = tuple(
        _encoder_vocabulary(fitted_cardinality, feature_name)
        for feature_name in active_columns
    )
    allowed_by_feature = dict(zip(active_columns, categories, strict=True))
    for feature_name in active_columns:
        values = _frame_values(train, feature_name)
        unexpected = sorted(
            set(values.tolist()).difference(allowed_by_feature[feature_name])
        )
        if unexpected:
            raise CategoricalEncodingError(
                f"Column {feature_name!r} contains values outside the Phase 4 "
                f"encoding vocabulary; apply Phase 4 before fitting: {unexpected}"
            )

    encoder = _fit_one_hot_encoder(
        train, active_columns=active_columns, categories=categories, config=config
    )
    feature_names = tuple(str(name) for name in encoder.get_feature_names_out(active_columns))
    columns: list[EncodedCategoryState] = []
    offset = 0
    for feature_name, feature_categories in zip(active_columns, categories, strict=True):
        width = len(feature_categories)
        columns.append(
            EncodedCategoryState(
                feature_name=feature_name,
                source_category_count=len(
                    fitted_cardinality.by_name[feature_name].training_vocabulary
                ),
                encoder_categories=feature_categories,
                encoded_feature_names=feature_names[offset : offset + width],
            )
        )
        offset += width
    return FittedCategoricalEncoder(
        config_version=config.config_version,
        policy_version=policy.policy_version,
        cardinality_fingerprint=fitted_cardinality.fingerprint,
        strategy=config.strategy,
        active_columns=active_columns,
        missing_token=fitted_cardinality.missing_token,
        rare_token=fitted_cardinality.rare_token,
        unknown_token=fitted_cardinality.unknown_token,
        handle_unknown=config.handle_unknown,
        drop=config.drop,
        sparse_output=config.sparse_output,
        columns=tuple(columns),
        encoder=encoder,
    )


def transform_categorical_encoder(
    frame: pd.DataFrame,
    *,
    fitted: FittedCategoricalEncoder,
    config: CategoricalEncodingConfig,
) -> csr_matrix:
    """Transform one frame into the fitted sparse categorical block."""
    _validate_config_matches_fitted(fitted, config)
    if not fitted.active_columns:
        return sparse.csr_matrix((len(frame), 0), dtype="int8")
    if fitted.encoder is None:
        raise CategoricalEncodingError("Active fitted encoder is missing sklearn state.")
    for feature_name in fitted.active_columns:
        _frame_values(frame, feature_name)
    matrix = fitted.encoder.transform(
        frame.loc[:, list(fitted.active_columns)].astype("string")
    )
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def transform_split_categorical_encoder(
    frames: Mapping[str, pd.DataFrame],
    *,
    fitted: FittedCategoricalEncoder,
    config: CategoricalEncodingConfig,
) -> dict[str, csr_matrix]:
    """Transform required splits using one unchanged training-fitted encoder."""
    missing = [split for split in REQUIRED_SPLITS if split not in frames]
    if missing:
        raise CategoricalEncodingError(f"Missing required split frames: {missing}")
    return {
        split: transform_categorical_encoder(frames[split], fitted=fitted, config=config)
        for split in REQUIRED_SPLITS
    }


def _defensive_unknown_count(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    state: EncodedCategoryState | None,
) -> int:
    """Count values relying on OneHotEncoder's defensive unknown-ignore fallback."""
    if state is None or feature_name not in frame:
        return 0
    values = frame[feature_name].astype("string")
    return int(values.notna().mul(~values.isin(state.encoder_categories)).sum())


def build_categorical_encoding_evidence(
    source_frames: Mapping[str, pd.DataFrame],
    encoded_matrices: Mapping[str, csr_matrix],
    *,
    fitted: FittedCategoricalEncoder,
    policy: FeaturePolicy,
    config: CategoricalEncodingConfig,
) -> pd.DataFrame:
    """Build governance, sparse-shape, schema, and reconciliation evidence."""
    for split in REQUIRED_SPLITS:
        if split not in source_frames or split not in encoded_matrices:
            raise CategoricalEncodingError(
                f"Missing required split for evidence: {split!r}."
            )
    active = set(active_encoding_feature_names(policy, config))
    if tuple(name for name in config.supported_columns if name in active) != fitted.active_columns:
        raise CategoricalEncodingError(
            "Fitted encoder active columns do not reconcile with policy."
        )
    schema_widths = {split: encoded_matrices[split].shape[1] for split in REQUIRED_SPLITS}
    schema_identical = len(set(schema_widths.values())) == 1
    rows: list[dict[str, object]] = []
    for feature_name in config.supported_columns:
        state = fitted.by_name.get(feature_name)
        active_feature = feature_name in active
        reserved_present = (
            state is not None
            and fitted.missing_token in state.encoder_categories
            and fitted.rare_token in state.encoder_categories
            and fitted.unknown_token in state.encoder_categories
        )
        rows.append(
            {
                "feature_name": feature_name,
                "policy_status": policy.by_name[feature_name].policy_status.value,
                "active": active_feature,
                "source_category_count": state.source_category_count if state else 0,
                "encoder_category_count": len(state.encoder_categories) if state else 0,
                "encoded_output_column_count": len(state.encoded_feature_names) if state else 0,
                "reserved_token_columns_present": reserved_present,
                "train_row_count": len(source_frames["train"]),
                "validation_row_count": len(source_frames["validation"]),
                "test_row_count": len(source_frames["test"]),
                "train_encoded_shape": tuple(encoded_matrices["train"].shape),
                "validation_encoded_shape": tuple(encoded_matrices["validation"].shape),
                "test_encoded_shape": tuple(encoded_matrices["test"].shape),
                "feature_schema_identical": schema_identical,
                "train_defensive_unknown_fallback_count": _defensive_unknown_count(
                    source_frames["train"], feature_name=feature_name, state=state
                ),
                "validation_defensive_unknown_fallback_count": _defensive_unknown_count(
                    source_frames["validation"], feature_name=feature_name, state=state
                ),
                "test_defensive_unknown_fallback_count": _defensive_unknown_count(
                    source_frames["test"], feature_name=feature_name, state=state
                ),
                "sparse_output": config.sparse_output,
                "transformation_applied": active_feature,
                "reconciled": schema_identical
                and all(
                    encoded_matrices[split].shape[0] == len(source_frames[split])
                    for split in REQUIRED_SPLITS
                )
                and (
                    not active_feature
                    or (
                        state is not None
                        and reserved_present
                        and len(state.encoded_feature_names)
                        == len(state.encoder_categories)
                    )
                ),
                "status": "PASS",
            }
        )
    evidence = pd.DataFrame(rows)
    evidence.loc[~evidence["reconciled"], "status"] = "FAIL"
    return evidence
