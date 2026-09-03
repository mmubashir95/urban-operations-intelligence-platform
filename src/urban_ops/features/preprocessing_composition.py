"""Compose governed categorical and numeric preprocessing blocks.

This module creates the Phase 7 model-ready sparse feature matrix from already
materialized Phase 5 categorical and Phase 6 numeric CSR blocks. It does not
refit upstream preprocessing, mutate inputs, train models, evaluate models, or
persist artifacts.
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
from scipy import sparse
from scipy.sparse import csr_matrix

from urban_ops.features.categorical_encoding import FittedCategoricalEncoder
from urban_ops.features.numeric_preprocessing import FittedNumericPreprocessor
from urban_ops.features.policy import FeaturePolicy, PolicyStatus
from urban_ops.features.temporal import REQUIRED_SPLITS


SUPPORTED_CONFIG_VERSION: Final = 1
SUPPORTED_COMPOSITION_ORDER: Final = ("categorical", "numeric")
SUPPORTED_MATRIX_TYPE: Final = "csr_matrix"
SUPPORTED_OUTPUT_DTYPE: Final = "float64"
SUPPORTED_LEARNED_STATISTICS: Final = "none"
REQUIRED_UPSTREAM_PHASES: Final = {
    "categorical_encoding": "phase_5",
    "numeric_preprocessing": "phase_6",
}


class PreprocessingCompositionError(ValueError):
    """Raised when preprocessing block composition is unsafe or inconsistent."""


@dataclass(frozen=True)
class PreprocessingCompositionConfig:
    """Validated Phase 7 sparse-block composition configuration."""

    config_version: int
    composition_order: tuple[str, ...]
    matrix_type: str
    output_dtype: str
    learned_statistics: str
    required_upstream_phases: dict[str, str]
    row_alignment: str
    decision_reason: str


@dataclass(frozen=True)
class FittedPreprocessingComposition:
    """Immutable Phase 7 schema and upstream-fingerprint composition state."""

    config_version: int
    policy_version: int
    categorical_encoder_fingerprint: str
    numeric_preprocessor_fingerprint: str
    categorical_feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    combined_feature_names: tuple[str, ...]
    categorical_feature_count: int
    numeric_feature_count: int
    combined_feature_count: int
    matrix_type: str
    output_dtype: str
    composition_order: tuple[str, ...]
    learned_statistics: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic serialization-ready representation."""
        return {
            "config_version": self.config_version,
            "policy_version": self.policy_version,
            "categorical_encoder_fingerprint": self.categorical_encoder_fingerprint,
            "numeric_preprocessor_fingerprint": self.numeric_preprocessor_fingerprint,
            "categorical_feature_names": list(self.categorical_feature_names),
            "numeric_feature_names": list(self.numeric_feature_names),
            "combined_feature_names": list(self.combined_feature_names),
            "categorical_feature_count": self.categorical_feature_count,
            "numeric_feature_count": self.numeric_feature_count,
            "combined_feature_count": self.combined_feature_count,
            "matrix_type": self.matrix_type,
            "output_dtype": self.output_dtype,
            "composition_order": list(self.composition_order),
            "learned_statistics": self.learned_statistics,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the composition state."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable configuration error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PreprocessingCompositionError(f"{field} must be a mapping.")
    return value


def _text(value: object, field: str) -> str:
    """Return stripped non-empty text from configuration input."""
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingCompositionError(f"{field} must be non-empty text.")
    return value.strip()


def _unique_text_list(value: object, field: str) -> tuple[str, ...]:
    """Return a unique, non-empty tuple from a configured string list."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise PreprocessingCompositionError(
            f"{field} must be a non-empty string list."
        )
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise PreprocessingCompositionError(f"{field} must not contain duplicates.")
    return result


def load_preprocessing_composition_config(
    path: Path | str,
) -> PreprocessingCompositionConfig:
    """Load and validate the Phase 7 preprocessing composition configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise PreprocessingCompositionError(
            f"Preprocessing composition config does not exist: {config_path}"
        )
    try:
        root = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise PreprocessingCompositionError(
            f"Preprocessing composition YAML is invalid: {config_path}"
        ) from error
    required = {
        "config_version",
        "composition_order",
        "matrix_type",
        "output_dtype",
        "learned_statistics",
        "required_upstream_phases",
        "row_alignment",
        "decision_reason",
    }
    missing = sorted(required.difference(root))
    if missing:
        raise PreprocessingCompositionError(
            f"Configuration is missing fields: {missing}"
        )
    if root["config_version"] != SUPPORTED_CONFIG_VERSION:
        raise PreprocessingCompositionError("Only config_version 1 is supported.")
    composition_order = _unique_text_list(
        root["composition_order"], "composition_order"
    )
    if composition_order != SUPPORTED_COMPOSITION_ORDER:
        raise PreprocessingCompositionError(
            "Phase 7 requires composition_order: categorical, numeric."
        )
    matrix_type = _text(root["matrix_type"], "matrix_type")
    if matrix_type != SUPPORTED_MATRIX_TYPE:
        raise PreprocessingCompositionError("Only csr_matrix output is supported.")
    output_dtype = _text(root["output_dtype"], "output_dtype")
    if output_dtype != SUPPORTED_OUTPUT_DTYPE:
        raise PreprocessingCompositionError("Only float64 output is supported.")
    learned_statistics = _text(root["learned_statistics"], "learned_statistics")
    if learned_statistics != SUPPORTED_LEARNED_STATISTICS:
        raise PreprocessingCompositionError(
            "Phase 7 composition must not declare learned statistics."
        )
    upstream = _mapping(root["required_upstream_phases"], "required_upstream_phases")
    if upstream != REQUIRED_UPSTREAM_PHASES:
        raise PreprocessingCompositionError(
            "required_upstream_phases must reference Phase 5 and Phase 6."
        )
    return PreprocessingCompositionConfig(
        config_version=SUPPORTED_CONFIG_VERSION,
        composition_order=composition_order,
        matrix_type=matrix_type,
        output_dtype=output_dtype,
        learned_statistics=learned_statistics,
        required_upstream_phases=dict(upstream),
        row_alignment=_text(root["row_alignment"], "row_alignment"),
        decision_reason=_text(root["decision_reason"], "decision_reason"),
    )


def _validate_feature_names(
    feature_names: tuple[str, ...], *, policy: FeaturePolicy
) -> None:
    """Reject duplicate, empty, or policy-blocked final feature names."""
    if any(not name.strip() for name in feature_names):
        raise PreprocessingCompositionError("Combined feature names must be non-empty.")
    duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
    if duplicates:
        raise PreprocessingCompositionError(
            f"Combined feature names must be unique; duplicates: {duplicates}"
        )
    by_name = policy.by_name
    blocked_exact = sorted(
        name
        for name in feature_names
        if name in by_name
        and by_name[name].policy_status is not PolicyStatus.APPROVED_CANDIDATE
    )
    if blocked_exact:
        raise PreprocessingCompositionError(
            f"Combined feature schema includes non-approved features: {blocked_exact}"
        )


def build_preprocessing_composition(
    *,
    fitted_categorical: FittedCategoricalEncoder,
    fitted_numeric: FittedNumericPreprocessor,
    policy: FeaturePolicy,
    config: PreprocessingCompositionConfig,
) -> FittedPreprocessingComposition:
    """Freeze the final schema from already-fitted categorical and numeric states."""
    if (
        fitted_categorical.policy_version != policy.policy_version
        or fitted_numeric.policy_version != policy.policy_version
    ):
        raise PreprocessingCompositionError(
            "Upstream fitted states must match the active policy version."
        )
    categorical_names = fitted_categorical.encoded_feature_names
    numeric_names = fitted_numeric.output_feature_names
    combined_names = (*categorical_names, *numeric_names)
    _validate_feature_names(combined_names, policy=policy)
    return FittedPreprocessingComposition(
        config_version=config.config_version,
        policy_version=policy.policy_version,
        categorical_encoder_fingerprint=fitted_categorical.fingerprint,
        numeric_preprocessor_fingerprint=fitted_numeric.fingerprint,
        categorical_feature_names=categorical_names,
        numeric_feature_names=numeric_names,
        combined_feature_names=combined_names,
        categorical_feature_count=len(categorical_names),
        numeric_feature_count=len(numeric_names),
        combined_feature_count=len(combined_names),
        matrix_type=config.matrix_type,
        output_dtype=config.output_dtype,
        composition_order=config.composition_order,
        learned_statistics=config.learned_statistics,
    )


def _validate_matrix_width(
    matrix: csr_matrix, expected_width: int, label: str
) -> None:
    """Raise when a branch matrix width no longer matches fitted state."""
    if matrix.shape[1] != expected_width:
        raise PreprocessingCompositionError(
            f"{label} matrix width {matrix.shape[1]} does not match "
            f"expected feature count {expected_width}."
        )


def compose_preprocessing_blocks(
    *,
    categorical_matrix: csr_matrix,
    numeric_matrix: csr_matrix,
    fitted: FittedPreprocessingComposition,
) -> csr_matrix:
    """Horizontally compose one categorical and numeric sparse block."""
    if categorical_matrix.shape[0] != numeric_matrix.shape[0]:
        raise PreprocessingCompositionError(
            "Cannot compose preprocessing blocks with different row counts: "
            f"categorical={categorical_matrix.shape[0]}, "
            f"numeric={numeric_matrix.shape[0]}."
        )
    _validate_matrix_width(
        categorical_matrix, fitted.categorical_feature_count, "Categorical"
    )
    _validate_matrix_width(numeric_matrix, fitted.numeric_feature_count, "Numeric")
    combined = sparse.hstack(
        [categorical_matrix, numeric_matrix],
        format="csr",
        dtype=fitted.output_dtype,
    )
    _validate_matrix_width(combined, fitted.combined_feature_count, "Combined")
    return combined


def compose_split_preprocessing_blocks(
    *,
    categorical_matrices: Mapping[str, csr_matrix],
    numeric_matrices: Mapping[str, csr_matrix],
    fitted: FittedPreprocessingComposition,
) -> dict[str, csr_matrix]:
    """Compose required split matrices using one frozen Phase 7 schema."""
    missing = [
        split
        for split in REQUIRED_SPLITS
        if split not in categorical_matrices or split not in numeric_matrices
    ]
    if missing:
        raise PreprocessingCompositionError(
            f"Missing required split matrices: {missing}"
        )
    return {
        split: compose_preprocessing_blocks(
            categorical_matrix=categorical_matrices[split],
            numeric_matrix=numeric_matrices[split],
            fitted=fitted,
        )
        for split in REQUIRED_SPLITS
    }


def build_preprocessing_composition_evidence(
    *,
    categorical_matrices: Mapping[str, csr_matrix],
    numeric_matrices: Mapping[str, csr_matrix],
    combined_matrices: Mapping[str, csr_matrix],
    fitted: FittedPreprocessingComposition,
) -> pd.DataFrame:
    """Build split-level reconciliation evidence for composed preprocessing output."""
    for split in REQUIRED_SPLITS:
        if (
            split not in categorical_matrices
            or split not in numeric_matrices
            or split not in combined_matrices
        ):
            raise PreprocessingCompositionError(
                f"Missing required split for composition evidence: {split!r}."
            )
    schema_widths = {
        split: combined_matrices[split].shape[1] for split in REQUIRED_SPLITS
    }
    schema_identical = len(set(schema_widths.values())) == 1
    duplicate_count = len(fitted.combined_feature_names) - len(
        set(fitted.combined_feature_names)
    )
    rows: list[dict[str, object]] = []
    for split in REQUIRED_SPLITS:
        categorical_matrix = categorical_matrices[split]
        numeric_matrix = numeric_matrices[split]
        combined_matrix = combined_matrices[split]
        row_alignment_valid = (
            categorical_matrix.shape[0]
            == numeric_matrix.shape[0]
            == combined_matrix.shape[0]
        )
        expected_count = (
            categorical_matrix.shape[1] + numeric_matrix.shape[1]
        )
        reconciled = (
            row_alignment_valid
            and combined_matrix.shape[1] == expected_count
            and expected_count == fitted.combined_feature_count
            and len(fitted.combined_feature_names) == combined_matrix.shape[1]
            and duplicate_count == 0
            and schema_identical
            and sparse.isspmatrix_csr(combined_matrix)
        )
        rows.append(
            {
                "split": split,
                "row_count": combined_matrix.shape[0],
                "categorical_column_count": categorical_matrix.shape[1],
                "numeric_column_count": numeric_matrix.shape[1],
                "combined_column_count": combined_matrix.shape[1],
                "expected_combined_count": expected_count,
                "matrix_type": type(combined_matrix).__name__,
                "output_dtype": str(combined_matrix.dtype),
                "feature_name_count": len(fitted.combined_feature_names),
                "duplicate_feature_name_count": duplicate_count,
                "categorical_fingerprint": (
                    fitted.categorical_encoder_fingerprint
                ),
                "numeric_fingerprint": fitted.numeric_preprocessor_fingerprint,
                "composition_fingerprint": fitted.fingerprint,
                "schema_identical": schema_identical,
                "row_alignment_valid": row_alignment_valid,
                "reconciled": reconciled,
                "status": "PASS" if reconciled else "FAIL",
            }
        )
    return pd.DataFrame(rows)
