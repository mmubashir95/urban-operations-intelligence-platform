"""Validate and materialize governed numeric pass-through features.

This module creates a separate numeric feature block for Phase 6. It does not
impute, scale, clip, transform cyclic values, combine with categorical blocks,
build a ColumnTransformer, or train models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse import csr_matrix

from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_PRODUCTION_DECISIONS: Final = frozenset({"ACTIVE"})
SUPPORTED_STRATEGIES: Final = frozenset({"pass_through", "deferred"})
SUPPORTED_VALUE_TYPES: Final = frozenset({"integer", "binary", "continuous"})
OUTPUT_MATRIX_TYPE: Final = "csr_matrix"
OUTPUT_DTYPE: Final = "float64"


class NumericPreprocessingError(ValueError):
    """Raised when numeric preprocessing configuration or input is unsafe."""


@dataclass(frozen=True)
class NumericFeatureRule:
    """Validated preprocessing rule for one governed numeric feature."""

    feature_name: str
    strategy: str
    value_type: str
    minimum: float | None
    maximum: float | None
    rationale: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "feature_name": self.feature_name,
            "strategy": self.strategy,
            "value_type": self.value_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class NumericPreprocessingConfig:
    """Validated Phase 6 numeric preprocessing policy."""

    config_version: int
    production_decision: str
    supported_columns: tuple[str, ...]
    rules: tuple[NumericFeatureRule, ...]
    matrix_type: str
    dtype: str
    row_alignment: str
    learned_statistics: str
    decision_reason: str

    @property
    def by_name(self) -> dict[str, NumericFeatureRule]:
        """Return feature rules indexed by feature name."""
        return {rule.feature_name: rule for rule in self.rules}


@dataclass(frozen=True)
class NumericFeatureState:
    """Immutable selected numeric feature metadata."""

    feature_name: str
    preprocessing_strategy: str
    value_type: str
    minimum: float | None
    maximum: float | None
    output_feature_name: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "feature_name": self.feature_name,
            "preprocessing_strategy": self.preprocessing_strategy,
            "value_type": self.value_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "output_feature_name": self.output_feature_name,
        }


@dataclass(frozen=True)
class FittedNumericPreprocessor:
    """Explicit Phase 6 state for deterministic numeric pass-through output."""

    config_version: int
    policy_version: int
    active_columns: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    matrix_type: str
    dtype: str
    learned_statistics: str
    columns: tuple[NumericFeatureState, ...]

    @property
    def by_name(self) -> dict[str, NumericFeatureState]:
        """Return fitted numeric column states indexed by feature name."""
        return {column.feature_name: column for column in self.columns}

    @property
    def output_feature_count(self) -> int:
        """Return the numeric output column count."""
        return len(self.output_feature_names)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "config_version": self.config_version,
            "policy_version": self.policy_version,
            "active_columns": list(self.active_columns),
            "output_feature_names": list(self.output_feature_names),
            "matrix_type": self.matrix_type,
            "dtype": self.dtype,
            "learned_statistics": self.learned_statistics,
            "columns": [column.to_dict() for column in self.columns],
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the fitted numeric state."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable configuration error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NumericPreprocessingError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty text from configuration input."""
    if not isinstance(value, str) or not value.strip():
        raise NumericPreprocessingError(f"{field} must be non-empty text.")
    return value.strip()


def _number_or_none(value: object, field: str, *, required: bool) -> float | None:
    """Return a finite numeric bound or None when allowed."""
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericPreprocessingError(f"{field} must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise NumericPreprocessingError(f"{field} must be finite.")
    return number


def _unique_text_list(value: object, field: str) -> tuple[str, ...]:
    """Return a unique, non-empty tuple from a configured string list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise NumericPreprocessingError(f"{field} must be a non-empty string list.")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise NumericPreprocessingError(f"{field} must not contain duplicates.")
    return result


def _parse_rule(feature_name: str, raw: object) -> NumericFeatureRule:
    """Parse and validate one numeric preprocessing rule."""
    item = _mapping(raw, f"feature_rules.{feature_name}")
    required = {"strategy", "value_type", "rationale"}
    missing = sorted(required.difference(item))
    if missing:
        raise NumericPreprocessingError(
            f"feature_rules.{feature_name} is missing fields: {missing}"
        )
    strategy = _text(item["strategy"], f"{feature_name}.strategy")
    if strategy not in SUPPORTED_STRATEGIES:
        raise NumericPreprocessingError(f"Unsupported numeric strategy {strategy!r}.")
    value_type = _text(item["value_type"], f"{feature_name}.value_type")
    if value_type not in SUPPORTED_VALUE_TYPES:
        raise NumericPreprocessingError(f"Unsupported value_type {value_type!r}.")
    bounds_required = strategy == "pass_through"
    minimum = _number_or_none(
        item.get("minimum"), f"{feature_name}.minimum", required=bounds_required
    )
    maximum = _number_or_none(
        item.get("maximum"), f"{feature_name}.maximum", required=bounds_required
    )
    if minimum is not None and maximum is not None and minimum > maximum:
        raise NumericPreprocessingError(
            f"{feature_name} minimum cannot exceed maximum."
        )
    if strategy == "deferred" and (minimum is not None or maximum is not None):
        raise NumericPreprocessingError(
            f"Deferred feature {feature_name!r} must not define active bounds."
        )
    if strategy == "pass_through" and value_type == "continuous":
        raise NumericPreprocessingError(
            f"Continuous feature {feature_name!r} cannot use pass_through in Phase 6."
        )
    return NumericFeatureRule(
        feature_name=feature_name,
        strategy=strategy,
        value_type=value_type,
        minimum=minimum,
        maximum=maximum,
        rationale=_text(item["rationale"], f"{feature_name}.rationale"),
    )


def load_numeric_preprocessing_config(
    path: Path | str,
) -> NumericPreprocessingConfig:
    """Load and validate the Phase 6 numeric preprocessing configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise NumericPreprocessingError(
            f"Numeric preprocessing config does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise NumericPreprocessingError(
            f"Numeric preprocessing YAML is invalid: {config_path}"
        ) from error
    required = {
        "config_version",
        "production_decision",
        "supported_columns",
        "feature_rules",
        "output",
        "learned_statistics",
        "decision_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise NumericPreprocessingError(f"Configuration is missing fields: {missing}")
    if root["config_version"] != 1:
        raise NumericPreprocessingError("Only config_version 1 is supported.")
    decision = _text(root["production_decision"], "production_decision")
    if decision not in SUPPORTED_PRODUCTION_DECISIONS:
        raise NumericPreprocessingError(
            f"Unsupported production_decision {decision!r}."
        )
    supported_columns = _unique_text_list(root["supported_columns"], "supported_columns")
    rule_payload = _mapping(root["feature_rules"], "feature_rules")
    if set(rule_payload) != set(supported_columns):
        raise NumericPreprocessingError(
            "feature_rules must exactly match supported_columns."
        )
    rules = tuple(_parse_rule(name, rule_payload[name]) for name in supported_columns)
    output = _mapping(root["output"], "output")
    matrix_type = _text(output.get("matrix_type"), "output.matrix_type")
    if matrix_type != OUTPUT_MATRIX_TYPE:
        raise NumericPreprocessingError("Only csr_matrix numeric output is supported.")
    dtype = _text(output.get("dtype"), "output.dtype")
    if dtype != OUTPUT_DTYPE:
        raise NumericPreprocessingError("Only float64 numeric output is supported.")
    learned_statistics = _text(root["learned_statistics"], "learned_statistics")
    if learned_statistics != "none":
        raise NumericPreprocessingError(
            "Phase 6 pass-through must not declare learned numeric statistics."
        )
    return NumericPreprocessingConfig(
        config_version=1,
        production_decision=decision,
        supported_columns=supported_columns,
        rules=rules,
        matrix_type=matrix_type,
        dtype=dtype,
        row_alignment=_text(output.get("row_alignment"), "output.row_alignment"),
        learned_statistics=learned_statistics,
        decision_reason=_text(root["decision_reason"], "decision_reason"),
    )


def active_numeric_feature_names(
    policy: FeaturePolicy, config: NumericPreprocessingConfig
) -> tuple[str, ...]:
    """Return numeric fields enabled by config strategy and frozen policy."""
    unknown = sorted(set(config.supported_columns).difference(policy.by_name))
    if unknown:
        raise FeaturePolicyError(
            f"Numeric preprocessing config references unknown features: {unknown}"
        )
    active: list[str] = []
    for feature_name in config.supported_columns:
        entry = policy.by_name[feature_name]
        rule = config.by_name[feature_name]
        approved = (
            entry.policy_status is PolicyStatus.APPROVED_CANDIDATE
            and entry.phase_2_allowed
        )
        if approved and rule.strategy == "deferred":
            raise NumericPreprocessingError(
                f"Approved numeric feature {feature_name!r} cannot be deferred."
            )
        if approved:
            active.append(feature_name)
    return tuple(active)


def _validate_config_matches_fitted(
    fitted: FittedNumericPreprocessor, config: NumericPreprocessingConfig
) -> None:
    """Raise if numeric transform uses a state/config mismatch."""
    if (
        fitted.config_version != config.config_version
        or fitted.matrix_type != config.matrix_type
        or fitted.dtype != config.dtype
        or fitted.learned_statistics != config.learned_statistics
    ):
        raise NumericPreprocessingError(
            "Fitted numeric state does not match the numeric preprocessing config."
        )


def _coerce_numeric_values(
    frame: pd.DataFrame, rule: NumericFeatureRule
) -> pd.Series:
    """Validate one active numeric feature and return float64 output values."""
    feature_name = rule.feature_name
    if feature_name not in frame:
        raise NumericPreprocessingError(
            f"Numeric source column is missing: {feature_name!r}."
        )
    source = frame[feature_name]
    if source.isna().any():
        raise NumericPreprocessingError(
            f"Active deterministic numeric feature {feature_name!r} contains nulls; "
            "rerun Phase 2 deterministic feature creation instead of imputing."
        )
    values = pd.to_numeric(source, errors="coerce")
    if values.isna().any():
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} contains non-numeric values."
        )
    array = values.to_numpy(dtype="float64")
    if not np.isfinite(array).all():
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} contains non-finite values."
        )
    if rule.value_type in {"integer", "binary"} and not np.equal(
        np.mod(array, 1), 0
    ).all():
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} must be integer-like."
        )
    if rule.value_type == "binary" and not set(array.tolist()).issubset({0.0, 1.0}):
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} must be binary 0/1."
        )
    if rule.minimum is not None and not (array >= rule.minimum).all():
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} is below {rule.minimum:g}."
        )
    if rule.maximum is not None and not (array <= rule.maximum).all():
        raise NumericPreprocessingError(
            f"Active numeric feature {feature_name!r} is above {rule.maximum:g}."
        )
    return pd.Series(array, index=frame.index, name=feature_name, dtype="float64")


def fit_numeric_preprocessor(
    train: pd.DataFrame,
    *,
    policy: FeaturePolicy,
    config: NumericPreprocessingConfig,
) -> FittedNumericPreprocessor:
    """Create explicit deterministic numeric state from the training frame.

    Phase 6 learns no statistics. The training frame is used only to validate
    that approved deterministic temporal features satisfy their domains before
    a stable pass-through schema is frozen.
    """
    active_columns = active_numeric_feature_names(policy, config)
    columns: list[NumericFeatureState] = []
    for feature_name in active_columns:
        rule = config.by_name[feature_name]
        if rule.strategy != "pass_through":
            raise NumericPreprocessingError(
                f"Active numeric feature {feature_name!r} must be pass_through."
            )
        _coerce_numeric_values(train, rule)
        columns.append(
            NumericFeatureState(
                feature_name=feature_name,
                preprocessing_strategy=rule.strategy,
                value_type=rule.value_type,
                minimum=rule.minimum,
                maximum=rule.maximum,
                output_feature_name=feature_name,
            )
        )
    return FittedNumericPreprocessor(
        config_version=config.config_version,
        policy_version=policy.policy_version,
        active_columns=active_columns,
        output_feature_names=tuple(column.output_feature_name for column in columns),
        matrix_type=config.matrix_type,
        dtype=config.dtype,
        learned_statistics=config.learned_statistics,
        columns=tuple(columns),
    )


def transform_numeric_preprocessor(
    frame: pd.DataFrame,
    *,
    fitted: FittedNumericPreprocessor,
    config: NumericPreprocessingConfig,
) -> csr_matrix:
    """Transform one frame into the fitted numeric CSR pass-through block."""
    _validate_config_matches_fitted(fitted, config)
    if not fitted.active_columns:
        return sparse.csr_matrix((len(frame), 0), dtype=np.float64)
    columns = [
        _coerce_numeric_values(frame, config.by_name[feature_name]).to_numpy()
        for feature_name in fitted.active_columns
    ]
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    return sparse.csr_matrix(matrix)


def transform_split_numeric_preprocessor(
    frames: Mapping[str, pd.DataFrame],
    *,
    fitted: FittedNumericPreprocessor,
    config: NumericPreprocessingConfig,
) -> dict[str, csr_matrix]:
    """Transform required splits with one unchanged numeric fitted state."""
    missing = [split for split in REQUIRED_SPLITS if split not in frames]
    if missing:
        raise NumericPreprocessingError(f"Missing required split frames: {missing}")
    return {
        split: transform_numeric_preprocessor(
            frames[split], fitted=fitted, config=config
        )
        for split in REQUIRED_SPLITS
    }


def _split_min_max(
    frame: pd.DataFrame, feature_name: str
) -> tuple[object, object, int, str]:
    """Return descriptive min/max/null/dtype evidence for one feature."""
    if feature_name not in frame:
        return pd.NA, pd.NA, 0, "MISSING_COLUMN"
    source = frame[feature_name]
    null_count = int(source.isna().sum())
    values = pd.to_numeric(source, errors="coerce")
    if values.notna().any():
        return values.min(), values.max(), null_count, str(source.dtype)
    return pd.NA, pd.NA, null_count, str(source.dtype)


def build_numeric_preprocessing_evidence(
    source_frames: Mapping[str, pd.DataFrame],
    numeric_matrices: Mapping[str, csr_matrix],
    *,
    fitted: FittedNumericPreprocessor,
    policy: FeaturePolicy,
    config: NumericPreprocessingConfig,
) -> pd.DataFrame:
    """Build governance, domain, sparse-shape, and reconciliation evidence."""
    for split in REQUIRED_SPLITS:
        if split not in source_frames or split not in numeric_matrices:
            raise NumericPreprocessingError(
                f"Missing required split for evidence: {split!r}."
            )
    active = set(active_numeric_feature_names(policy, config))
    if tuple(name for name in config.supported_columns if name in active) != fitted.active_columns:
        raise NumericPreprocessingError(
            "Fitted numeric active columns do not reconcile with policy."
        )
    schema_widths = {split: numeric_matrices[split].shape[1] for split in REQUIRED_SPLITS}
    schema_identical = len(set(schema_widths.values())) == 1
    rows: list[dict[str, object]] = []
    for feature_name in config.supported_columns:
        rule = config.by_name[feature_name]
        state = fitted.by_name.get(feature_name)
        row: dict[str, object] = {
            "feature_name": feature_name,
            "policy_status": policy.by_name[feature_name].policy_status.value,
            "active": feature_name in active,
            "preprocessing_strategy": rule.strategy,
            "output_dtype": config.dtype if state else pd.NA,
            "output_column_count": 1 if state else 0,
            "learned_statistics": config.learned_statistics if state else "none",
            "transformation_applied": state is not None,
            "train_encoded_shape": tuple(numeric_matrices["train"].shape),
            "validation_encoded_shape": tuple(numeric_matrices["validation"].shape),
            "test_encoded_shape": tuple(numeric_matrices["test"].shape),
            "feature_schema_identical": schema_identical,
        }
        for split in REQUIRED_SPLITS:
            minimum, maximum, null_count, dtype = _split_min_max(
                source_frames[split], feature_name
            )
            row[f"{split}_min"] = minimum
            row[f"{split}_max"] = maximum
            row[f"null_count_{split}"] = null_count
            if split == "train":
                row["input_dtype"] = dtype
        reconciled = (
            schema_identical
            and all(
                numeric_matrices[split].shape[0] == len(source_frames[split])
                for split in REQUIRED_SPLITS
            )
            and (
                state is None
                or all(row[f"null_count_{split}"] == 0 for split in REQUIRED_SPLITS)
            )
        )
        row["reconciled"] = reconciled
        row["status"] = "PASS" if reconciled and state is not None else (
            "DEFERRED" if rule.strategy == "deferred" else "FAIL"
        )
        rows.append(row)
    return pd.DataFrame(rows)
