"""Verify and freeze metadata for baseline modelling inputs.

This module defines the Phase 9 boundary between the frozen Phase 8
preprocessing output and later baseline model phases. It validates supplied
X/y/identifier/timestamp objects and records metadata only. It does not fit,
predict, score, transform, sort, repair, or persist matrices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Final

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse import csr_matrix

from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.preprocessing_verification import (
    VerifiedPreprocessingContract,
    _fingerprint,
)
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_BASELINE_CONTRACT_VERSION: Final = 1
SUPPORTED_MODELLING_POLICY_VERSION: Final = 1
SUPPORTED_PHASE_9_STATUS: Final = "MODEL_INPUTS_VERIFIED"
SUPPORTED_MATRIX_TYPE: Final = "csr_matrix"
SUPPORTED_OUTPUT_DTYPE: Final = "float64"
SUPPORTED_TARGET_DOMAIN: Final = (0, 1)
SUPPORTED_PHASE_8_STATUS: Final = "FROZEN_MODEL_READY"


class BaselineModellingContractError(ValueError):
    """Raised when Phase 9 baseline modelling inputs violate the contract."""


@dataclass(frozen=True)
class BaselineModellingContractConfig:
    """Validated rules for the Phase 9 baseline modelling input gate."""

    contract_version: int
    modelling_policy_version: int
    required_splits: tuple[str, ...]
    phase_8_required_status: str
    matrix_type: str
    output_dtype: str
    target_domain: tuple[int, ...]
    require_both_train_classes: bool
    contract_status: str
    train_only_target_statistics: bool
    decision_reason: str


@dataclass(frozen=True)
class VerifiedBaselineModellingContract:
    """Metadata-only contract for inputs later baseline models may consume."""

    contract_version: int
    modelling_policy_version: int
    phase_8_contract_fingerprint: str
    ordered_feature_names: tuple[str, ...]
    feature_count: int
    matrix_type: str
    matrix_dtype: str
    target_name: str
    identifier_name: str
    chronology_name: str
    split_order: tuple[str, ...]
    row_counts_by_split: dict[str, int]
    train_row_count: int
    train_positive_class_count: int
    train_negative_class_count: int
    train_positive_class_prevalence: float
    train_majority_class: int
    train_target_classes: tuple[int, ...]
    preprocessing_schema_fingerprint: str
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe metadata representation."""
        return {
            "contract_version": self.contract_version,
            "modelling_policy_version": self.modelling_policy_version,
            "phase_8_contract_fingerprint": self.phase_8_contract_fingerprint,
            "ordered_feature_names": list(self.ordered_feature_names),
            "feature_count": self.feature_count,
            "matrix_type": self.matrix_type,
            "matrix_dtype": self.matrix_dtype,
            "target_name": self.target_name,
            "identifier_name": self.identifier_name,
            "chronology_name": self.chronology_name,
            "split_order": list(self.split_order),
            "row_counts_by_split": {
                split: self.row_counts_by_split[split] for split in self.split_order
            },
            "train_row_count": self.train_row_count,
            "train_positive_class_count": self.train_positive_class_count,
            "train_negative_class_count": self.train_negative_class_count,
            "train_positive_class_prevalence": self.train_positive_class_prevalence,
            "train_majority_class": self.train_majority_class,
            "train_target_classes": list(self.train_target_classes),
            "preprocessing_schema_fingerprint": self.preprocessing_schema_fingerprint,
            "status": self.status,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "VerifiedBaselineModellingContract":
        """Rebuild a persisted metadata contract and validate basic shape."""
        try:
            values = dict(payload)
            values["ordered_feature_names"] = tuple(values["ordered_feature_names"])
            values["split_order"] = tuple(values["split_order"])
            values["row_counts_by_split"] = dict(values["row_counts_by_split"])
            values["train_target_classes"] = tuple(values["train_target_classes"])
            contract = cls(**values)
        except (KeyError, TypeError) as error:
            raise BaselineModellingContractError(
                "Persisted baseline modelling contract is malformed."
            ) from error
        if contract.feature_count != len(contract.ordered_feature_names):
            raise BaselineModellingContractError(
                "Persisted feature count does not match ordered_feature_names."
            )
        if tuple(contract.row_counts_by_split) != contract.split_order:
            raise BaselineModellingContractError(
                "Persisted row_counts_by_split order does not match split_order."
            )
        if contract.train_row_count != contract.row_counts_by_split.get("train"):
            raise BaselineModellingContractError(
                "Persisted train row count does not match row_counts_by_split."
            )
        return contract

    @property
    def fingerprint(self) -> str:
        """Return the deterministic SHA-256 fingerprint of this metadata contract."""
        return _fingerprint(self.to_dict())


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable configuration error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BaselineModellingContractError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty configuration text."""
    if not isinstance(value, str) or not value.strip():
        raise BaselineModellingContractError(f"{field} must be non-empty text.")
    return value.strip()


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    """Return unique non-empty strings from a configured list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise BaselineModellingContractError(
            f"{field} must be a non-empty string list."
        )
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise BaselineModellingContractError(f"{field} must not contain duplicates.")
    return result


def load_baseline_modelling_contract_config(
    path: Path | str,
) -> BaselineModellingContractConfig:
    """Load and strictly validate the Phase 9 modelling-contract configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise BaselineModellingContractError(
            f"Baseline modelling contract config does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise BaselineModellingContractError(
            f"Baseline modelling contract YAML is invalid: {config_path}"
        ) from error
    required = {
        "contract_version",
        "modelling_policy_version",
        "required_splits",
        "phase_8_required_status",
        "matrix_type",
        "output_dtype",
        "target_domain",
        "require_both_train_classes",
        "contract_status",
        "train_only_target_statistics",
        "decision_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise BaselineModellingContractError(
            f"Configuration is missing fields: {missing}"
        )
    required_splits = _text_tuple(root["required_splits"], "required_splits")
    if root["contract_version"] != SUPPORTED_BASELINE_CONTRACT_VERSION:
        raise BaselineModellingContractError("Only contract_version 1 is supported.")
    if root["modelling_policy_version"] != SUPPORTED_MODELLING_POLICY_VERSION:
        raise BaselineModellingContractError(
            "Only modelling_policy_version 1 is supported."
        )
    if required_splits != REQUIRED_SPLITS:
        raise BaselineModellingContractError(
            "required_splits must be train, validation, test in that order."
        )
    if root["target_domain"] != list(SUPPORTED_TARGET_DOMAIN):
        raise BaselineModellingContractError("target_domain must be [0, 1].")
    if root["require_both_train_classes"] is not True:
        raise BaselineModellingContractError(
            "require_both_train_classes must remain true."
        )
    if root["train_only_target_statistics"] is not True:
        raise BaselineModellingContractError(
            "train_only_target_statistics must remain true."
        )
    phase_8_status = _text(root["phase_8_required_status"], "phase_8_required_status")
    matrix_type = _text(root["matrix_type"], "matrix_type")
    output_dtype = _text(root["output_dtype"], "output_dtype")
    contract_status = _text(root["contract_status"], "contract_status")
    if phase_8_status != SUPPORTED_PHASE_8_STATUS:
        raise BaselineModellingContractError(
            "phase_8_required_status must be FROZEN_MODEL_READY."
        )
    if matrix_type != SUPPORTED_MATRIX_TYPE or output_dtype != SUPPORTED_OUTPUT_DTYPE:
        raise BaselineModellingContractError(
            "Phase 9 requires csr_matrix output with float64 dtype."
        )
    if contract_status != SUPPORTED_PHASE_9_STATUS:
        raise BaselineModellingContractError(
            "contract_status must be MODEL_INPUTS_VERIFIED."
        )
    return BaselineModellingContractConfig(
        contract_version=SUPPORTED_BASELINE_CONTRACT_VERSION,
        modelling_policy_version=SUPPORTED_MODELLING_POLICY_VERSION,
        required_splits=required_splits,
        phase_8_required_status=phase_8_status,
        matrix_type=matrix_type,
        output_dtype=output_dtype,
        target_domain=SUPPORTED_TARGET_DOMAIN,
        require_both_train_classes=True,
        contract_status=contract_status,
        train_only_target_statistics=True,
        decision_reason=_text(root["decision_reason"], "decision_reason"),
    )


def _as_series(value: object, *, split: str, field: str) -> pd.Series:
    """Return one-dimensional data without changing its positional order."""
    if isinstance(value, pd.Series):
        return value
    array = np.asarray(value)
    if array.ndim != 1:
        raise BaselineModellingContractError(
            f"{split} {field} must be one-dimensional; observed {array.ndim}D."
        )
    return pd.Series(array)


def _validate_exact_splits(
    values: Mapping[str, object], *, field: str, split_order: Sequence[str]
) -> None:
    """Reject missing or unexpected split keys."""
    expected = tuple(split_order)
    observed = tuple(values.keys())
    missing = [split for split in expected if split not in values]
    unexpected = [split for split in observed if split not in expected]
    if missing or unexpected:
        raise BaselineModellingContractError(
            f"{field} splits must be exactly {expected}; missing={missing}, "
            f"unexpected={unexpected}."
        )


def _validate_phase_8_lineage(
    *,
    phase_8_contract: VerifiedPreprocessingContract | None,
    feature_names: Sequence[str],
    config: BaselineModellingContractConfig,
    expected_phase_8_fingerprint: str | None,
    expected_schema_fingerprint: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Verify Phase 8 is present, frozen, and still authoritative."""
    if phase_8_contract is None:
        raise BaselineModellingContractError("Phase 8 preprocessing contract is required.")
    if phase_8_contract.verification_status != config.phase_8_required_status:
        raise BaselineModellingContractError(
            "Phase 8 status must be FROZEN_MODEL_READY; observed "
            f"{phase_8_contract.verification_status!r}."
        )
    phase_8_fingerprint = phase_8_contract.fingerprint
    if (
        expected_phase_8_fingerprint is not None
        and expected_phase_8_fingerprint != phase_8_fingerprint
    ):
        raise BaselineModellingContractError(
            "Phase 8 fingerprint mismatch: expected "
            f"{expected_phase_8_fingerprint}, observed {phase_8_fingerprint}."
        )
    if (
        expected_schema_fingerprint is not None
        and expected_schema_fingerprint != phase_8_contract.schema_fingerprint
    ):
        raise BaselineModellingContractError(
            "Preprocessing schema fingerprint mismatch: expected "
            f"{expected_schema_fingerprint}, observed "
            f"{phase_8_contract.schema_fingerprint}."
        )
    if phase_8_contract.split_names != config.required_splits:
        raise BaselineModellingContractError(
            "Phase 8 split order does not match the Phase 9 required split order."
        )
    if (
        phase_8_contract.matrix_type != config.matrix_type
        or phase_8_contract.output_dtype != config.output_dtype
    ):
        raise BaselineModellingContractError(
            "Phase 8 matrix contract does not match the Phase 9 modelling contract."
        )
    ordered_names = tuple(feature_names)
    if ordered_names != phase_8_contract.combined_feature_names:
        raise BaselineModellingContractError(
            "Supplied feature names differ from the frozen Phase 8 feature schema."
        )
    if len(ordered_names) != phase_8_contract.combined_feature_count:
        raise BaselineModellingContractError(
            "Supplied feature count differs from the frozen Phase 8 feature count."
        )
    return phase_8_fingerprint, ordered_names


def _validate_feature_isolation(
    *,
    feature_names: Sequence[str],
    target_name: str,
    identifier_name: str,
    chronology_name: str,
    policy: FeaturePolicy | None,
) -> None:
    """Reject direct target, identifier, timestamp, leakage, or unapproved fields."""
    names = tuple(feature_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise BaselineModellingContractError(
            f"Feature names must be unique; duplicates={duplicates}."
        )
    forbidden = sorted({target_name, identifier_name, chronology_name}.intersection(names))
    if forbidden:
        raise BaselineModellingContractError(
            f"Forbidden modelling fields appear in feature names: {forbidden}."
        )
    if policy is None:
        return
    policy_by_name = policy.by_name
    violations: list[str] = []
    for name in names:
        entry = policy_by_name.get(name)
        if (
            entry is None
            or entry.policy_status is not PolicyStatus.APPROVED_CANDIDATE
            or entry.prediction_time_status != "AVAILABLE"
            or entry.leakage_status != "SAFE"
            or not entry.phase_2_allowed
        ):
            violations.append(name)
    if violations:
        raise BaselineModellingContractError(
            "Feature schema contains target, identifier, leakage, excluded, "
            f"conditional, or post-creation fields: {sorted(set(violations))}."
        )


def _validate_chronology(
    *,
    timestamps: Mapping[str, object],
    split_order: Sequence[str],
) -> dict[str, pd.Series]:
    """Verify valid UTC chronological order without sorting or repairing rows."""
    prepared: dict[str, pd.Series] = {}
    for split in split_order:
        values = _as_series(timestamps[split], split=split, field="created_date")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
            )
            parsed = pd.to_datetime(values, utc=True, errors="coerce")
        if parsed.isna().any():
            raise BaselineModellingContractError(
                f"{split} created_date contains null or invalid timestamps."
            )
        if isinstance(values, pd.Series) and not isinstance(
            values.dtype, pd.DatetimeTZDtype
        ):
            raise BaselineModellingContractError(
                f"{split} created_date must be timezone-aware UTC datetimes."
            )
        if str(parsed.dt.tz) != "UTC":
            raise BaselineModellingContractError(f"{split} created_date must use UTC.")
        if not parsed.is_monotonic_increasing:
            raise BaselineModellingContractError(
                f"{split} created_date must retain chronological row order."
            )
        prepared[split] = pd.Series(parsed.array, index=values.index)
    if not (
        prepared["train"].max() < prepared["validation"].min()
        and prepared["validation"].max() < prepared["test"].min()
    ):
        raise BaselineModellingContractError(
            "Chronology violation: train must end before validation and "
            "validation must end before test."
        )
    return prepared


def _validate_matrix(
    *,
    split: str,
    matrix: object,
    feature_count: int,
    config: BaselineModellingContractConfig,
) -> csr_matrix:
    """Validate the sparse matrix contract without densifying the full matrix."""
    if not sparse.isspmatrix_csr(matrix):
        raise BaselineModellingContractError(
            f"{split} X must be scipy.sparse csr_matrix; observed "
            f"{type(matrix).__name__}."
        )
    if matrix.ndim != 2:
        raise BaselineModellingContractError(
            f"{split} X must be two-dimensional; observed ndim={matrix.ndim}."
        )
    if matrix.shape[0] <= 0:
        raise BaselineModellingContractError(
            f"{split} X row count must be greater than zero; observed {matrix.shape[0]}."
        )
    if matrix.shape[1] != feature_count:
        raise BaselineModellingContractError(
            f"{split} X feature count mismatch: expected {feature_count}, "
            f"observed {matrix.shape[1]}."
        )
    if str(matrix.dtype) != config.output_dtype:
        raise BaselineModellingContractError(
            f"{split} X dtype mismatch: expected {config.output_dtype}, "
            f"observed {matrix.dtype}."
        )
    non_finite_count = int((~np.isfinite(matrix.data)).sum())
    if non_finite_count:
        raise BaselineModellingContractError(
            f"{split} X contains {non_finite_count} non-finite values."
        )
    return matrix


def _validate_target(
    *,
    split: str,
    target: object,
    row_count: int,
    config: BaselineModellingContractConfig,
) -> pd.Series:
    """Validate one split target vector as binary metadata input."""
    values = _as_series(target, split=split, field="y")
    if len(values) != row_count:
        raise BaselineModellingContractError(
            f"{split} y count mismatch: expected {row_count}, observed {len(values)}."
        )
    if values.isna().any():
        raise BaselineModellingContractError(f"{split} y contains null values.")
    observed = set(values.unique())
    if not observed.issubset(set(config.target_domain)):
        raise BaselineModellingContractError(
            f"{split} y contains values outside {config.target_domain}: "
            f"{sorted(observed)}."
        )
    return values


def _validate_identifier(*, split: str, identifiers: object, row_count: int) -> pd.Series:
    """Validate one split identifier vector without adding it to features."""
    values = _as_series(identifiers, split=split, field="unique_key")
    if len(values) != row_count:
        raise BaselineModellingContractError(
            f"{split} unique_key count mismatch: expected {row_count}, "
            f"observed {len(values)}."
        )
    if values.isna().any():
        raise BaselineModellingContractError(f"{split} unique_key contains null values.")
    duplicate_count = int(values.duplicated().sum())
    if duplicate_count:
        raise BaselineModellingContractError(
            f"{split} unique_key contains {duplicate_count} duplicate values."
        )
    return values


def _phase_8_row_counts(contract: VerifiedPreprocessingContract) -> dict[str, int]:
    """Return Phase 8 row counts keyed by the authoritative split names."""
    return {
        "train": contract.train_row_count,
        "validation": contract.validation_row_count,
        "test": contract.test_row_count,
    }


def verify_baseline_modelling_contract(
    *,
    matrices: Mapping[str, object],
    targets: Mapping[str, object],
    identifiers: Mapping[str, object],
    timestamps: Mapping[str, object],
    feature_names: Sequence[str],
    phase_8_contract: VerifiedPreprocessingContract | None,
    config: BaselineModellingContractConfig,
    expected_phase_8_fingerprint: str | None = None,
    expected_schema_fingerprint: str | None = None,
    policy: FeaturePolicy | None = None,
) -> VerifiedBaselineModellingContract:
    """Validate Phase 9 baseline modelling inputs and freeze metadata only.

    Raises:
        BaselineModellingContractError: If lineage, split presence, matrix shape,
            target domain, identifier uniqueness, chronology, row alignment, or
            feature isolation does not match the frozen Phase 8 boundary.
    """
    if (
        config.contract_version != SUPPORTED_BASELINE_CONTRACT_VERSION
        or config.modelling_policy_version != SUPPORTED_MODELLING_POLICY_VERSION
        or config.required_splits != REQUIRED_SPLITS
        or config.phase_8_required_status != SUPPORTED_PHASE_8_STATUS
        or config.matrix_type != SUPPORTED_MATRIX_TYPE
        or config.output_dtype != SUPPORTED_OUTPUT_DTYPE
        or config.target_domain != SUPPORTED_TARGET_DOMAIN
        or not config.require_both_train_classes
        or config.contract_status != SUPPORTED_PHASE_9_STATUS
        or not config.train_only_target_statistics
    ):
        raise BaselineModellingContractError(
            "Phase 9 runtime configuration differs from the supported contract."
        )
    phase_8_fingerprint, ordered_feature_names = _validate_phase_8_lineage(
        phase_8_contract=phase_8_contract,
        feature_names=feature_names,
        config=config,
        expected_phase_8_fingerprint=expected_phase_8_fingerprint,
        expected_schema_fingerprint=expected_schema_fingerprint,
    )
    _validate_feature_isolation(
        feature_names=ordered_feature_names,
        target_name=phase_8_contract.target_name,
        identifier_name=phase_8_contract.identifier_name,
        chronology_name=phase_8_contract.timestamp_name,
        policy=policy,
    )
    collections = {
        "matrices": matrices,
        "targets": targets,
        "identifiers": identifiers,
        "timestamps": timestamps,
    }
    for field, values in collections.items():
        _validate_exact_splits(values, field=field, split_order=config.required_splits)
    prepared_timestamps = _validate_chronology(
        timestamps=timestamps, split_order=config.required_splits
    )
    row_counts: dict[str, int] = {}
    prepared_targets: dict[str, pd.Series] = {}
    identifier_sets: dict[str, set[object]] = {}
    phase_8_counts = _phase_8_row_counts(phase_8_contract)
    for split in config.required_splits:
        matrix = _validate_matrix(
            split=split,
            matrix=matrices[split],
            feature_count=len(ordered_feature_names),
            config=config,
        )
        if matrix.shape[0] != phase_8_counts[split]:
            raise BaselineModellingContractError(
                f"{split} row count differs from Phase 8: expected "
                f"{phase_8_counts[split]}, observed {matrix.shape[0]}."
            )
        target = _validate_target(
            split=split,
            target=targets[split],
            row_count=matrix.shape[0],
            config=config,
        )
        identifier = _validate_identifier(
            split=split, identifiers=identifiers[split], row_count=matrix.shape[0]
        )
        timestamp = prepared_timestamps[split]
        if len(timestamp) != matrix.shape[0]:
            raise BaselineModellingContractError(
                f"{split} created_date count mismatch: expected {matrix.shape[0]}, "
                f"observed {len(timestamp)}."
            )
        if not (
            target.index.equals(identifier.index)
            and target.index.equals(timestamp.index)
        ):
            raise BaselineModellingContractError(
                f"{split} row/index alignment violation across X, y, unique_key, "
                "and created_date."
            )
        row_counts[split] = matrix.shape[0]
        prepared_targets[split] = target
        identifier_sets[split] = set(identifier.tolist())
    train_values = prepared_targets["train"]
    train_classes = tuple(sorted(int(value) for value in train_values.unique()))
    if config.require_both_train_classes and train_classes != config.target_domain:
        raise BaselineModellingContractError(
            "Training y must contain both target classes {0, 1}."
        )
    if any(
        identifier_sets[left] & identifier_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise BaselineModellingContractError(
            "Split overlap violation: unique_key values must be disjoint."
        )
    positive_count = int(train_values.eq(1).sum())
    negative_count = int(train_values.eq(0).sum())
    train_row_count = len(train_values)
    train_majority_class = 1 if positive_count > negative_count else 0
    return VerifiedBaselineModellingContract(
        contract_version=config.contract_version,
        modelling_policy_version=config.modelling_policy_version,
        phase_8_contract_fingerprint=phase_8_fingerprint,
        ordered_feature_names=ordered_feature_names,
        feature_count=len(ordered_feature_names),
        matrix_type=config.matrix_type,
        matrix_dtype=config.output_dtype,
        target_name=phase_8_contract.target_name,
        identifier_name=phase_8_contract.identifier_name,
        chronology_name=phase_8_contract.timestamp_name,
        split_order=config.required_splits,
        row_counts_by_split={split: row_counts[split] for split in config.required_splits},
        train_row_count=train_row_count,
        train_positive_class_count=positive_count,
        train_negative_class_count=negative_count,
        train_positive_class_prevalence=float(positive_count / train_row_count),
        train_majority_class=train_majority_class,
        train_target_classes=train_classes,
        preprocessing_schema_fingerprint=phase_8_contract.schema_fingerprint,
        status=config.contract_status,
    )


def build_baseline_modelling_contract_evidence(
    contract: VerifiedBaselineModellingContract,
) -> pd.DataFrame:
    """Return display-friendly evidence for a verified Phase 9 contract."""
    return pd.DataFrame(
        [
            {
                "split": split,
                "row_count": contract.row_counts_by_split[split],
                "feature_count": contract.feature_count,
                "matrix_type": contract.matrix_type,
                "matrix_dtype": contract.matrix_dtype,
                "target_name": contract.target_name,
                "identifier_name": contract.identifier_name,
                "chronology_name": contract.chronology_name,
                "preprocessing_schema_fingerprint": (
                    contract.preprocessing_schema_fingerprint
                ),
                "phase_8_contract_fingerprint": contract.phase_8_contract_fingerprint,
                "status": contract.status,
            }
            for split in contract.split_order
        ]
    )
