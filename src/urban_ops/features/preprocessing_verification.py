"""Verify and freeze final model inputs produced by preprocessing Phases 2-7.

This module validates existing sparse matrices, targets, identifiers, chronology,
feature lineage, and fitted-state links. It does not transform values, refit
preprocessing, select features, persist matrices, or train or evaluate a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

from urban_ops.features.categorical_encoding import FittedCategoricalEncoder
from urban_ops.features.numeric_preprocessing import FittedNumericPreprocessor
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.preprocessing_composition import FittedPreprocessingComposition
from urban_ops.features.rare_unseen import FittedRareUnseenState
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_VERIFICATION_VERSION: Final = 1
SUPPORTED_MATRIX_TYPE: Final = "csr_matrix"
SUPPORTED_OUTPUT_DTYPE: Final = "float64"
SUPPORTED_TARGET_DOMAIN: Final = (0, 1)
SUPPORTED_ZERO_VARIANCE_ACTION: Final = "fail"
SUPPORTED_VERIFICATION_STATUS: Final = "FROZEN_MODEL_READY"
REQUIRED_UPSTREAM_PHASES: Final = {
    "rare_unseen": "phase_4",
    "categorical_encoding": "phase_5",
    "numeric_preprocessing": "phase_6",
    "preprocessing_composition": "phase_7",
}


class PreprocessingVerificationError(ValueError):
    """Raised when final preprocessed model inputs violate the frozen contract."""


@dataclass(frozen=True)
class PreprocessingVerificationConfig:
    """Validated rules for the Phase 8 model-input verification gate."""

    verification_version: int
    required_splits: tuple[str, ...]
    matrix_type: str
    output_dtype: str
    target_domain: tuple[int, ...]
    require_both_train_classes: bool
    zero_variance_action: str
    verification_status: str
    baseline_estimator_compatibility: str
    required_upstream_phases: dict[str, str]
    row_alignment: str
    decision_reason: str


@dataclass(frozen=True)
class FinalFeatureSchemaEntry:
    """One ordered final feature and its policy-approved upstream lineage."""

    column_index: int
    feature_name: str
    source_branch: str
    source_feature: str
    preprocessing_stage: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialization-ready schema row."""
        return {
            "column_index": self.column_index,
            "feature_name": self.feature_name,
            "source_branch": self.source_branch,
            "source_feature": self.source_feature,
            "preprocessing_stage": self.preprocessing_stage,
        }


@dataclass(frozen=True)
class VerifiedPreprocessingContract:
    """Immutable, serializable Phase 8 contract for baseline model inputs."""

    verification_version: int
    policy_version: int
    cardinality_fingerprint: str
    categorical_encoder_fingerprint: str
    numeric_preprocessor_fingerprint: str
    composition_fingerprint: str
    combined_feature_names: tuple[str, ...]
    combined_feature_count: int
    matrix_type: str
    output_dtype: str
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    target_name: str
    identifier_name: str
    timestamp_name: str
    split_names: tuple[str, ...]
    schema_fingerprint: str
    verification_status: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic representation suitable for JSON persistence."""
        return {
            "verification_version": self.verification_version,
            "policy_version": self.policy_version,
            "cardinality_fingerprint": self.cardinality_fingerprint,
            "categorical_encoder_fingerprint": self.categorical_encoder_fingerprint,
            "numeric_preprocessor_fingerprint": self.numeric_preprocessor_fingerprint,
            "composition_fingerprint": self.composition_fingerprint,
            "combined_feature_names": list(self.combined_feature_names),
            "combined_feature_count": self.combined_feature_count,
            "matrix_type": self.matrix_type,
            "output_dtype": self.output_dtype,
            "train_row_count": self.train_row_count,
            "validation_row_count": self.validation_row_count,
            "test_row_count": self.test_row_count,
            "target_name": self.target_name,
            "identifier_name": self.identifier_name,
            "timestamp_name": self.timestamp_name,
            "split_names": list(self.split_names),
            "schema_fingerprint": self.schema_fingerprint,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "VerifiedPreprocessingContract":
        """Rebuild a persisted contract and reject a malformed field shape."""
        try:
            values = dict(payload)
            values["combined_feature_names"] = tuple(values["combined_feature_names"])
            values["split_names"] = tuple(values["split_names"])
            contract = cls(**values)
        except (KeyError, TypeError) as error:
            raise PreprocessingVerificationError(
                "Persisted preprocessing contract is malformed."
            ) from error
        if contract.combined_feature_count != len(contract.combined_feature_names):
            raise PreprocessingVerificationError(
                "Persisted feature count does not match its feature-name schema."
            )
        return contract

    @property
    def fingerprint(self) -> str:
        """Return the deterministic SHA-256 fingerprint of the verified contract."""
        return _fingerprint(self.to_dict())


def _fingerprint(payload: object) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable value."""
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable config error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PreprocessingVerificationError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty configuration text."""
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingVerificationError(f"{field} must be non-empty text.")
    return value.strip()


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    """Return unique non-empty strings from a configured list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise PreprocessingVerificationError(
            f"{field} must be a non-empty string list."
        )
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise PreprocessingVerificationError(f"{field} must not contain duplicates.")
    return result


def load_preprocessing_verification_config(
    path: Path | str,
) -> PreprocessingVerificationConfig:
    """Load and strictly validate the Phase 8 verification configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise PreprocessingVerificationError(
            f"Preprocessing verification config does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise PreprocessingVerificationError(
            f"Preprocessing verification YAML is invalid: {config_path}"
        ) from error
    required = {
        "verification_version",
        "required_splits",
        "matrix_type",
        "output_dtype",
        "target_domain",
        "require_both_train_classes",
        "zero_variance_action",
        "verification_status",
        "baseline_estimator_compatibility",
        "required_upstream_phases",
        "row_alignment",
        "decision_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise PreprocessingVerificationError(
            f"Configuration is missing fields: {missing}"
        )
    if root["verification_version"] != SUPPORTED_VERIFICATION_VERSION:
        raise PreprocessingVerificationError(
            "Only verification_version 1 is supported."
        )
    required_splits = _text_tuple(root["required_splits"], "required_splits")
    if required_splits != REQUIRED_SPLITS:
        raise PreprocessingVerificationError(
            "required_splits must be train, validation, test in that order."
        )
    if root["target_domain"] != list(SUPPORTED_TARGET_DOMAIN):
        raise PreprocessingVerificationError("target_domain must be [0, 1].")
    if root["require_both_train_classes"] is not True:
        raise PreprocessingVerificationError(
            "require_both_train_classes must remain true."
        )
    matrix_type = _text(root["matrix_type"], "matrix_type")
    output_dtype = _text(root["output_dtype"], "output_dtype")
    zero_variance_action = _text(
        root["zero_variance_action"], "zero_variance_action"
    )
    verification_status = _text(root["verification_status"], "verification_status")
    upstream = _mapping(root["required_upstream_phases"], "required_upstream_phases")
    if matrix_type != SUPPORTED_MATRIX_TYPE or output_dtype != SUPPORTED_OUTPUT_DTYPE:
        raise PreprocessingVerificationError(
            "Phase 8 requires csr_matrix output with float64 dtype."
        )
    if zero_variance_action != SUPPORTED_ZERO_VARIANCE_ACTION:
        raise PreprocessingVerificationError("zero_variance_action must be fail.")
    if verification_status != SUPPORTED_VERIFICATION_STATUS:
        raise PreprocessingVerificationError(
            "verification_status must be FROZEN_MODEL_READY."
        )
    if upstream != REQUIRED_UPSTREAM_PHASES:
        raise PreprocessingVerificationError(
            "required_upstream_phases must reference Phases 4 through 7."
        )
    return PreprocessingVerificationConfig(
        verification_version=SUPPORTED_VERIFICATION_VERSION,
        required_splits=required_splits,
        matrix_type=matrix_type,
        output_dtype=output_dtype,
        target_domain=SUPPORTED_TARGET_DOMAIN,
        require_both_train_classes=True,
        zero_variance_action=zero_variance_action,
        verification_status=verification_status,
        baseline_estimator_compatibility=_text(
            root["baseline_estimator_compatibility"],
            "baseline_estimator_compatibility",
        ),
        required_upstream_phases=dict(upstream),
        row_alignment=_text(root["row_alignment"], "row_alignment"),
        decision_reason=_text(root["decision_reason"], "decision_reason"),
    )


def build_final_feature_schema(
    *,
    fitted_categorical: FittedCategoricalEncoder,
    fitted_numeric: FittedNumericPreprocessor,
    fitted_composition: FittedPreprocessingComposition,
) -> tuple[FinalFeatureSchemaEntry, ...]:
    """Build the exact final column order from explicit upstream state lineage."""
    rows: list[FinalFeatureSchemaEntry] = []
    for column in fitted_categorical.columns:
        for feature_name in column.encoded_feature_names:
            rows.append(
                FinalFeatureSchemaEntry(
                    column_index=len(rows),
                    feature_name=feature_name,
                    source_branch="categorical",
                    source_feature=column.feature_name,
                    preprocessing_stage="phase_5_one_hot",
                )
            )
    for column in fitted_numeric.columns:
        rows.append(
            FinalFeatureSchemaEntry(
                column_index=len(rows),
                feature_name=column.output_feature_name,
                source_branch="numeric",
                source_feature=column.feature_name,
                preprocessing_stage=f"phase_6_{column.preprocessing_strategy}",
            )
        )
    categorical_names = fitted_categorical.encoded_feature_names
    numeric_names = fitted_numeric.output_feature_names
    names = tuple(row.feature_name for row in rows)
    if (
        fitted_composition.categorical_feature_names != categorical_names
        or fitted_composition.numeric_feature_names != numeric_names
        or fitted_composition.categorical_feature_count != len(categorical_names)
        or fitted_composition.numeric_feature_count != len(numeric_names)
        or fitted_composition.combined_feature_count != len(names)
        or names != fitted_composition.combined_feature_names
    ):
        raise PreprocessingVerificationError(
            "Phase 4-6 feature lineage does not match the frozen Phase 7 schema."
        )
    return tuple(rows)


def _validate_upstream_states(
    *,
    policy: FeaturePolicy,
    fitted_cardinality: FittedRareUnseenState,
    fitted_categorical: FittedCategoricalEncoder,
    fitted_numeric: FittedNumericPreprocessor,
    fitted_composition: FittedPreprocessingComposition,
) -> None:
    """Verify policy versions and the complete Phase 4-7 fingerprint chain."""
    versions = {
        fitted_cardinality.policy_version,
        fitted_categorical.policy_version,
        fitted_numeric.policy_version,
        fitted_composition.policy_version,
    }
    if versions != {policy.policy_version}:
        raise PreprocessingVerificationError(
            "Upstream fitted states do not match the active policy version."
        )
    if fitted_categorical.cardinality_fingerprint != fitted_cardinality.fingerprint:
        raise PreprocessingVerificationError(
            "Phase 5 encoder does not reference the supplied Phase 4 fingerprint."
        )
    if (
        fitted_composition.categorical_encoder_fingerprint
        != fitted_categorical.fingerprint
        or fitted_composition.numeric_preprocessor_fingerprint
        != fitted_numeric.fingerprint
    ):
        raise PreprocessingVerificationError(
            "Phase 7 composition does not reference the supplied Phase 5/6 states."
        )


def _validate_schema_policy(
    schema: Sequence[FinalFeatureSchemaEntry],
    *,
    policy: FeaturePolicy,
    target_name: str,
    identifier_name: str,
) -> None:
    """Reject duplicate, blocked, untraceable, target, identifier, or leakage fields."""
    names = tuple(row.feature_name for row in schema)
    if not names:
        raise PreprocessingVerificationError(
            "Final feature count must be greater than zero."
        )
    if any(not name.strip() for name in names):
        raise PreprocessingVerificationError("Final feature names must be non-empty.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PreprocessingVerificationError(
            f"Final feature names must be unique; duplicates: {duplicates}"
        )
    direct_forbidden = sorted({target_name, identifier_name}.intersection(names))
    if direct_forbidden:
        raise PreprocessingVerificationError(
            f"Target or identifier appears in final feature names: {direct_forbidden}"
        )
    policy_by_name = policy.by_name
    blocked: list[str] = []
    for row in schema:
        entry = policy_by_name.get(row.source_feature)
        if (
            entry is None
            or entry.policy_status is not PolicyStatus.APPROVED_CANDIDATE
            or entry.prediction_time_status != "AVAILABLE"
            or entry.leakage_status != "SAFE"
            or not entry.phase_2_allowed
        ):
            blocked.append(row.source_feature)
    if blocked:
        raise PreprocessingVerificationError(
            "Final schema contains unapproved, conditional, or leakage feature "
            f"lineage: {sorted(set(blocked))}"
        )


def _as_series(value: object, *, split: str, label: str) -> pd.Series:
    """Return one-dimensional split data without changing its positional order."""
    if isinstance(value, pd.Series):
        return value
    array = np.asarray(value)
    if array.ndim != 1:
        raise PreprocessingVerificationError(
            f"{split} {label} must be one-dimensional."
        )
    return pd.Series(array)


def _validate_chronology(timestamps: Mapping[str, object]) -> dict[str, pd.Series]:
    """Verify ordered, UTC, non-overlapping train/validation/test timestamps."""
    prepared: dict[str, pd.Series] = {}
    for split in REQUIRED_SPLITS:
        values = _as_series(timestamps[split], split=split, label="timestamps")
        if values.isna().any() or not isinstance(values.dtype, pd.DatetimeTZDtype):
            raise PreprocessingVerificationError(
                f"{split} timestamps must be non-null timezone-aware datetimes."
            )
        if str(values.dt.tz) != "UTC":
            raise PreprocessingVerificationError(f"{split} timestamps must use UTC.")
        if not values.is_monotonic_increasing:
            raise PreprocessingVerificationError(
                f"{split} timestamps must retain chronological row order."
            )
        prepared[split] = values
    if not (
        prepared["train"].max() < prepared["validation"].min()
        and prepared["validation"].max() < prepared["test"].min()
    ):
        raise PreprocessingVerificationError(
            "Chronology must satisfy train before validation before test."
        )
    return prepared


def _column_values(matrix: csr_matrix, column_index: int) -> np.ndarray:
    """Return unique stored values plus implicit zero without densifying a column."""
    column = matrix.getcol(column_index)
    values = np.unique(column.data)
    if column.nnz < matrix.shape[0] and not np.any(values == 0):
        values = np.append(values, 0.0)
    return values


def build_training_feature_variance_evidence(
    *, matrix: csr_matrix, feature_names: Sequence[str]
) -> pd.DataFrame:
    """Describe train-only feature variation without densifying the matrix."""
    if not sparse.isspmatrix_csr(matrix):
        raise PreprocessingVerificationError("Training matrix must be CSR.")
    if matrix.shape[1] != len(feature_names):
        raise PreprocessingVerificationError(
            "Training matrix width does not match feature-name count."
        )
    rows: list[dict[str, object]] = []
    for index, feature_name in enumerate(feature_names):
        values = _column_values(matrix, index)
        rows.append(
            {
                "column_index": index,
                "feature_name": feature_name,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "unique_count": int(values.size),
                "all_zero": bool(values.size == 1 and values[0] == 0),
                "zero_variance": bool(values.size == 1),
            }
        )
    return pd.DataFrame(rows)


def verify_preprocessing_contract(
    *,
    matrices: Mapping[str, csr_matrix],
    targets: Mapping[str, object],
    identifiers: Mapping[str, object],
    timestamps: Mapping[str, object],
    feature_names_by_split: Mapping[str, Sequence[str]],
    target_name: str,
    identifier_name: str,
    timestamp_name: str,
    policy: FeaturePolicy,
    fitted_cardinality: FittedRareUnseenState,
    fitted_categorical: FittedCategoricalEncoder,
    fitted_numeric: FittedNumericPreprocessor,
    fitted_composition: FittedPreprocessingComposition,
    config: PreprocessingVerificationConfig,
) -> VerifiedPreprocessingContract:
    """Validate final X/y/identifier inputs and freeze their reusable contract.

    Raises:
        PreprocessingVerificationError: If matrix readiness, positional alignment,
            target validity, chronology, lineage, or upstream state linkage fails.
    """
    if (
        config.verification_version != SUPPORTED_VERIFICATION_VERSION
        or config.required_splits != REQUIRED_SPLITS
        or config.matrix_type != SUPPORTED_MATRIX_TYPE
        or config.output_dtype != SUPPORTED_OUTPUT_DTYPE
        or config.target_domain != SUPPORTED_TARGET_DOMAIN
        or not config.require_both_train_classes
        or config.zero_variance_action != SUPPORTED_ZERO_VARIANCE_ACTION
        or config.verification_status != SUPPORTED_VERIFICATION_STATUS
        or config.required_upstream_phases != REQUIRED_UPSTREAM_PHASES
    ):
        raise PreprocessingVerificationError(
            "Phase 8 runtime configuration differs from the supported contract."
        )
    collections = {
        "matrices": matrices,
        "targets": targets,
        "identifiers": identifiers,
        "timestamps": timestamps,
        "feature_names_by_split": feature_names_by_split,
    }
    for label, values in collections.items():
        missing = [split for split in config.required_splits if split not in values]
        if missing:
            raise PreprocessingVerificationError(
                f"Missing required {label} splits: {missing}"
            )
    _validate_upstream_states(
        policy=policy,
        fitted_cardinality=fitted_cardinality,
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
    )
    schema = build_final_feature_schema(
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
    )
    _validate_schema_policy(
        schema,
        policy=policy,
        target_name=target_name,
        identifier_name=identifier_name,
    )
    feature_names = fitted_composition.combined_feature_names
    if (
        fitted_composition.matrix_type != config.matrix_type
        or fitted_composition.output_dtype != config.output_dtype
    ):
        raise PreprocessingVerificationError(
            "Phase 7 matrix contract differs from Phase 8 configuration."
        )
    prepared_timestamps = _validate_chronology(timestamps)
    prepared_targets: dict[str, pd.Series] = {}
    identifier_sets: dict[str, set[object]] = {}
    for split in config.required_splits:
        matrix = matrices[split]
        if not sparse.isspmatrix_csr(matrix):
            raise PreprocessingVerificationError(f"{split} matrix must be CSR.")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise PreprocessingVerificationError(
                f"{split} matrix must be a non-empty two-dimensional matrix."
            )
        if str(matrix.dtype) != config.output_dtype:
            raise PreprocessingVerificationError(
                f"{split} matrix dtype {matrix.dtype} does not match "
                f"{config.output_dtype}."
            )
        split_names = tuple(feature_names_by_split[split])
        if split_names != feature_names:
            raise PreprocessingVerificationError(
                f"{split} feature schema differs from the frozen Phase 7 schema."
            )
        if matrix.shape[1] != len(split_names):
            raise PreprocessingVerificationError(
                f"{split} matrix width does not match its feature-name count."
            )
        non_finite_count = int((~np.isfinite(matrix.data)).sum())
        if non_finite_count:
            raise PreprocessingVerificationError(
                f"{split} matrix contains {non_finite_count} non-finite values."
            )
        target = _as_series(targets[split], split=split, label="target")
        identifier = _as_series(
            identifiers[split], split=split, label="identifiers"
        )
        timestamp = prepared_timestamps[split]
        if len(target) != matrix.shape[0]:
            raise PreprocessingVerificationError(
                f"{split} target count does not match matrix row count."
            )
        if len(identifier) != matrix.shape[0]:
            raise PreprocessingVerificationError(
                f"{split} identifier count does not match matrix row count."
            )
        if len(timestamp) != matrix.shape[0]:
            raise PreprocessingVerificationError(
                f"{split} timestamp count does not match matrix row count."
            )
        if not (
            target.index.equals(identifier.index)
            and target.index.equals(timestamp.index)
        ):
            raise PreprocessingVerificationError(
                f"{split} target, identifier, and timestamp indexes are not aligned."
            )
        if target.isna().any():
            raise PreprocessingVerificationError(
                f"{split} target contains null values."
            )
        target_values = set(target.unique())
        if not target_values.issubset(set(config.target_domain)):
            raise PreprocessingVerificationError(
                f"{split} target domain is not {config.target_domain}."
            )
        if identifier.isna().any() or identifier.duplicated().any():
            raise PreprocessingVerificationError(
                f"{split} identifiers must be non-null and unique."
            )
        prepared_targets[split] = target
        identifier_sets[split] = set(identifier.tolist())
    if (
        config.require_both_train_classes
        and set(prepared_targets["train"].unique()) != set(config.target_domain)
    ):
        raise PreprocessingVerificationError(
            "Training target must contain both classes."
        )
    if any(
        identifier_sets[left] & identifier_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise PreprocessingVerificationError(
            "Complaint identifiers must not overlap across splits."
        )
    variance = build_training_feature_variance_evidence(
        matrix=matrices["train"], feature_names=feature_names
    )
    zero_variance = variance.loc[
        variance["zero_variance"], "feature_name"
    ].tolist()
    if zero_variance and config.zero_variance_action == "fail":
        raise PreprocessingVerificationError(
            f"Training matrix contains zero-variance features: {zero_variance}"
        )
    schema_payload = [row.to_dict() for row in schema]
    return VerifiedPreprocessingContract(
        verification_version=config.verification_version,
        policy_version=policy.policy_version,
        cardinality_fingerprint=fitted_cardinality.fingerprint,
        categorical_encoder_fingerprint=fitted_categorical.fingerprint,
        numeric_preprocessor_fingerprint=fitted_numeric.fingerprint,
        composition_fingerprint=fitted_composition.fingerprint,
        combined_feature_names=feature_names,
        combined_feature_count=len(feature_names),
        matrix_type=config.matrix_type,
        output_dtype=config.output_dtype,
        train_row_count=matrices["train"].shape[0],
        validation_row_count=matrices["validation"].shape[0],
        test_row_count=matrices["test"].shape[0],
        target_name=target_name,
        identifier_name=identifier_name,
        timestamp_name=timestamp_name,
        split_names=config.required_splits,
        schema_fingerprint=_fingerprint(schema_payload),
        verification_status=config.verification_status,
    )


def build_final_feature_schema_evidence(
    *,
    fitted_categorical: FittedCategoricalEncoder,
    fitted_numeric: FittedNumericPreprocessor,
    fitted_composition: FittedPreprocessingComposition,
) -> pd.DataFrame:
    """Return the authoritative ordered final feature schema as a table."""
    schema = build_final_feature_schema(
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
    )
    return pd.DataFrame([row.to_dict() for row in schema])


def build_preprocessing_verification_evidence(
    *,
    matrices: Mapping[str, csr_matrix],
    targets: Mapping[str, object],
    identifiers: Mapping[str, object],
    contract: VerifiedPreprocessingContract,
) -> pd.DataFrame:
    """Build descriptive per-split evidence for an already verified contract."""
    rows: list[dict[str, object]] = []
    for split in contract.split_names:
        matrix = matrices[split]
        target = _as_series(targets[split], split=split, label="target")
        identifier = _as_series(
            identifiers[split], split=split, label="identifiers"
        )
        positive_count = int(target.eq(1).sum())
        negative_count = int(target.eq(0).sum())
        size = matrix.shape[0] * matrix.shape[1]
        rows.append(
            {
                "split": split,
                "row_count": matrix.shape[0],
                "feature_count": matrix.shape[1],
                "target_count": len(target),
                "identifier_count": len(identifier),
                "matrix_type": type(matrix).__name__,
                "dtype": str(matrix.dtype),
                "nnz": matrix.nnz,
                "density": float(matrix.nnz / size),
                "non_finite_count": int((~np.isfinite(matrix.data)).sum()),
                "schema_fingerprint": contract.schema_fingerprint,
                "target_null_count": int(target.isna().sum()),
                "target_positive_count": positive_count,
                "target_negative_count": negative_count,
                "target_positive_rate": float(positive_count / len(target)),
                "identifier_null_count": int(identifier.isna().sum()),
                "identifier_duplicate_count": int(identifier.duplicated().sum()),
                "row_alignment_valid": bool(
                    matrix.shape[0] == len(target) == len(identifier)
                ),
                "schema_valid": bool(
                    matrix.shape[1] == contract.combined_feature_count
                ),
                "leakage_free": True,
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)
