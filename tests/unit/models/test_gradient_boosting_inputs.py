"""Unit tests for the Phase 2.1 frozen-input safety gate."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.gradient_boosting_inputs import (
    GradientBoostingInputError,
    verify_frozen_gradient_boosting_inputs,
)
from tests.unit.models.test_baseline_contract import (
    FEATURE_NAMES,
    SPLITS,
    _phase_8_contract,
    _verify,
)


def _inputs() -> FrozenBaselineInputs:
    row_count = 4
    matrix = sparse.csr_matrix(
        np.arange(row_count * len(FEATURE_NAMES), dtype=np.float64).reshape(
            row_count, len(FEATURE_NAMES)
        )
    )
    matrices = {split: matrix.copy() for split in SPLITS}
    targets = {split: pd.Series([0, 1, 0, 1]) for split in SPLITS}
    timestamps = {
        "train": pd.date_range("2022-01-01", periods=row_count, tz="UTC"),
        "validation": pd.date_range("2022-02-01", periods=row_count, tz="UTC"),
        "test": pd.date_range("2022-03-01", periods=row_count, tz="UTC"),
    }
    frames = {
        split: pd.DataFrame(
            {
                "unique_key": [f"{split}-{index}" for index in range(row_count)],
                "created_date": timestamps[split],
            }
        )
        for split in SPLITS
    }
    phase_8 = _phase_8_contract()
    phase_9 = _verify(phase_8_contract=phase_8)
    return FrozenBaselineInputs(
        matrices=matrices,
        targets=targets,
        frames=frames,
        feature_names=FEATURE_NAMES,
        phase_8_contract=phase_8,
        phase_9_contract=phase_9,
        split_id="unit-test",
    )


def test_valid_frozen_inputs_pass_and_return_same_sparse_object() -> None:
    inputs = _inputs()
    train_matrix = inputs.matrices["train"]

    result = verify_frozen_gradient_boosting_inputs(inputs)

    assert result is inputs
    assert result.matrices["train"] is train_matrix
    assert sparse.isspmatrix_csr(result.matrices["train"])


@pytest.mark.parametrize("split", SPLITS)
def test_row_mismatch_fails_for_each_split(split: str) -> None:
    inputs = _inputs()
    inputs.targets[split] = inputs.targets[split].iloc[:-1]

    with pytest.raises(GradientBoostingInputError, match=f"{split.title()} matrix row"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_feature_count_mismatch_fails() -> None:
    inputs = _inputs()
    inputs.matrices["validation"] = sparse.csr_matrix((4, 3), dtype=np.float64)

    with pytest.raises(GradientBoostingInputError, match="feature count"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_different_feature_counts_across_splits_fail() -> None:
    inputs = _inputs()
    inputs.matrices["test"] = sparse.csr_matrix((4, 5), dtype=np.float64)

    with pytest.raises(GradientBoostingInputError, match="feature count"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_empty_feature_names_fail() -> None:
    inputs = replace(_inputs(), feature_names=())

    with pytest.raises(GradientBoostingInputError, match="must not be empty"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_duplicate_feature_names_fail() -> None:
    inputs = replace(
        _inputs(),
        feature_names=("created_hour", "created_hour", "created_month", "is_weekend"),
    )

    with pytest.raises(GradientBoostingInputError, match="duplicate feature names"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_unexpected_target_class_fails() -> None:
    inputs = _inputs()
    inputs.targets["validation"] = pd.Series([0, 1, 2, 1])

    with pytest.raises(GradientBoostingInputError, match="Unexpected target values"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_training_target_with_only_one_class_fails() -> None:
    inputs = _inputs()
    inputs.targets["train"] = pd.Series([0, 0, 0, 0])

    with pytest.raises(GradientBoostingInputError, match="both target classes"):
        verify_frozen_gradient_boosting_inputs(inputs)


def test_leakage_policy_violation_fails() -> None:
    inputs = _inputs()
    blocked_names = ("closed_date",)
    phase_8 = _phase_8_contract(feature_names=blocked_names)
    phase_9 = replace(
        inputs.phase_9_contract,
        ordered_feature_names=blocked_names,
        feature_count=1,
    )
    blocked = replace(
        inputs,
        matrices={
            split: sparse.csr_matrix((4, 1), dtype=np.float64) for split in SPLITS
        },
        feature_names=blocked_names,
        phase_8_contract=phase_8,
        phase_9_contract=phase_9,
    )

    with pytest.raises(GradientBoostingInputError, match="Leakage-prohibited"):
        verify_frozen_gradient_boosting_inputs(blocked)


def test_chronological_split_violation_fails() -> None:
    inputs = _inputs()
    inputs.frames["validation"]["created_date"] = pd.date_range(
        "2021-12-01", periods=4, tz="UTC"
    )

    with pytest.raises(GradientBoostingInputError, match="Chronology violation"):
        verify_frozen_gradient_boosting_inputs(inputs)
