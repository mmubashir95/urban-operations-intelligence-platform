"""Phase 2.1 safety gate for the frozen Month 1 modelling inputs.

This module does not build features, fit preprocessing, train a model, or
evaluate any split.  It loads the existing Month 1 inputs and independently
confirms that their frozen modelling contract is still valid for Month 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from urban_ops.features.policy import PolicyStatus, load_feature_policy
from urban_ops.models.baseline_contract import (
    BaselineModellingContractError,
    load_baseline_modelling_contract_config,
    verify_baseline_modelling_contract,
)
from urban_ops.models.baseline_workflow import (
    EDA_CONFIG_PATH,
    MODELLING_CONFIG_PATH,
    POLICY_PATH,
    FrozenBaselineInputs,
    load_frozen_baseline_inputs,
)


REQUIRED_SPLITS: Final = ("train", "validation", "test")
TARGET_DOMAIN: Final = frozenset({0, 1})


class GradientBoostingInputError(ValueError):
    """Raised when frozen Month 1 inputs are unsafe for Phase 2."""


def _require_splits(values: Mapping[str, object], *, label: str) -> None:
    """Require exactly the frozen train, validation, and test split keys."""
    observed = tuple(values)
    missing = [split for split in REQUIRED_SPLITS if split not in values]
    unexpected = [split for split in observed if split not in REQUIRED_SPLITS]
    if missing or unexpected:
        raise GradientBoostingInputError(
            f"Frozen {label} splits must be exactly {REQUIRED_SPLITS}; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _verify_feature_names(feature_names: Sequence[object]) -> tuple[str, ...]:
    """Validate the frozen names without sorting or otherwise changing order."""
    names = tuple(feature_names)
    if not names:
        raise GradientBoostingInputError("Frozen feature names must not be empty.")
    invalid = [name for name in names if not isinstance(name, str) or not name.strip()]
    if invalid:
        raise GradientBoostingInputError(
            "Frozen feature names contain null, non-text, or empty values."
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GradientBoostingInputError(
            f"Frozen feature contract contains duplicate feature names: {duplicates}."
        )
    return names  # type: ignore[return-value]


def _verify_shapes_and_targets(
    inputs: FrozenBaselineInputs, *, feature_count: int
) -> None:
    """Check split matrix shapes and binary targets without densifying matrices."""
    widths: dict[str, int] = {}
    for split in REQUIRED_SPLITS:
        matrix = inputs.matrices[split]
        shape = getattr(matrix, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise GradientBoostingInputError(
                f"{split.title()} matrix must be two-dimensional."
            )
        row_count, width = shape
        target = inputs.targets[split]
        if row_count != len(target):
            raise GradientBoostingInputError(
                f"{split.title()} matrix row count does not match "
                f"{split} target length: {row_count} != {len(target)}."
            )
        widths[split] = width

        values = pd.Series(target, copy=False)
        if values.isna().any():
            raise GradientBoostingInputError(
                f"{split.title()} target contains null values."
            )
        observed = set(values.unique().tolist())
        if not observed.issubset(TARGET_DOMAIN):
            raise GradientBoostingInputError(
                f"Unexpected target values found in {split}: {observed}."
            )

    if len(set(widths.values())) != 1:
        raise GradientBoostingInputError(
            f"Frozen matrices have different feature counts across splits: {widths}."
        )
    for split, width in widths.items():
        if width != feature_count:
            raise GradientBoostingInputError(
                f"{split.title()} feature count does not match frozen "
                f"feature-name count: {width} != {feature_count}."
            )
    train_classes = set(pd.Series(inputs.targets["train"], copy=False).unique().tolist())
    if train_classes != TARGET_DOMAIN:
        raise GradientBoostingInputError(
            "Training target must contain both target classes {0, 1}."
        )


def _verify_frozen_lineage_and_policy(
    inputs: FrozenBaselineInputs,
    *,
    feature_names: tuple[str, ...],
    policy_path: Path | str,
    modelling_config_path: Path | str,
) -> None:
    """Re-run the authoritative Month 1 chronology and leakage contract."""
    phase_8 = inputs.phase_8_contract
    phase_9 = inputs.phase_9_contract
    if feature_names != tuple(phase_8.combined_feature_names):
        raise GradientBoostingInputError(
            "Frozen feature names differ from the Phase 8 preprocessing contract."
        )
    if feature_names != tuple(phase_9.ordered_feature_names):
        raise GradientBoostingInputError(
            "Frozen feature names differ from the Phase 9 modelling contract."
        )

    policy = load_feature_policy(policy_path)
    violations = []
    for name in feature_names:
        entry = policy.by_name.get(name)
        if (
            entry is None
            or entry.policy_status is not PolicyStatus.APPROVED_CANDIDATE
            or entry.prediction_time_status != "AVAILABLE"
            or entry.leakage_status != "SAFE"
            or not entry.phase_2_allowed
        ):
            violations.append(name)
    if violations:
        raise GradientBoostingInputError(
            "Leakage-prohibited or unapproved feature detected in frozen model "
            f"matrix: {sorted(set(violations))}."
        )

    target_name = phase_9.target_name
    identifier_name = phase_9.identifier_name
    chronology_name = phase_9.chronology_name
    try:
        verified_contract = verify_baseline_modelling_contract(
            matrices=inputs.matrices,
            targets=inputs.targets,
            identifiers={
                split: inputs.frames[split][identifier_name]
                for split in REQUIRED_SPLITS
            },
            timestamps={
                split: inputs.frames[split][chronology_name]
                for split in REQUIRED_SPLITS
            },
            feature_names=feature_names,
            phase_8_contract=phase_8,
            config=load_baseline_modelling_contract_config(modelling_config_path),
            expected_phase_8_fingerprint=phase_9.phase_8_contract_fingerprint,
            expected_schema_fingerprint=phase_9.preprocessing_schema_fingerprint,
            policy=policy,
        )
    except (BaselineModellingContractError, KeyError) as error:
        raise GradientBoostingInputError(
            f"Frozen Month 1 modelling contract is invalid: {error}"
        ) from error
    if verified_contract != phase_9:
        raise GradientBoostingInputError(
            "Re-verified Month 1 inputs differ from the frozen Phase 9 contract."
        )


def verify_frozen_gradient_boosting_inputs(
    inputs: FrozenBaselineInputs,
    *,
    policy_path: Path | str = POLICY_PATH,
    modelling_config_path: Path | str = MODELLING_CONFIG_PATH,
) -> FrozenBaselineInputs:
    """Verify and return the same frozen input object for later Phase 2 work."""
    _require_splits(inputs.matrices, label="matrix")
    _require_splits(inputs.targets, label="target")
    _require_splits(inputs.frames, label="frame")
    feature_names = _verify_feature_names(inputs.feature_names)
    _verify_shapes_and_targets(inputs, feature_count=len(feature_names))
    _verify_frozen_lineage_and_policy(
        inputs,
        feature_names=feature_names,
        policy_path=policy_path,
        modelling_config_path=modelling_config_path,
    )
    return inputs


def load_and_verify_gradient_boosting_inputs(
    *,
    eda_config_path: Path | str = EDA_CONFIG_PATH,
    split_run_path: Path | None = None,
    policy_path: Path | str = POLICY_PATH,
    modelling_config_path: Path | str = MODELLING_CONFIG_PATH,
) -> FrozenBaselineInputs:
    """Load through Month 1 and apply the Phase 2.1 verification gate."""
    inputs = load_frozen_baseline_inputs(
        eda_config_path=eda_config_path,
        split_run_path=split_run_path,
    )
    return verify_frozen_gradient_boosting_inputs(
        inputs,
        policy_path=policy_path,
        modelling_config_path=modelling_config_path,
    )
