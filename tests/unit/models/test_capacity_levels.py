"""Verify the predefined Phase 5 operational capacity levels."""

from dataclasses import FrozenInstanceError

import pytest

from urban_ops.models.evaluation import (
    TOP_K_CAPACITIES,
    TOP_K_FRACTIONS,
    CapacityLevel,
    EvaluationError,
    get_standard_capacity_levels,
)


def test_standard_capacities_are_canonical_and_ordered() -> None:
    """Phase 5 uses exactly 5%, 10%, and 20% in report order."""
    assert TOP_K_CAPACITIES == (0.05, 0.10, 0.20)
    assert TOP_K_FRACTIONS is TOP_K_CAPACITIES


@pytest.mark.parametrize(
    ("n_samples", "expected_k"),
    [
        (100, (5, 10, 20)),
        (5_499, (275, 550, 1_100)),
        (11, (1, 2, 3)),
    ],
)
def test_standard_capacity_levels_expand_with_phase_5_1_ceiling_rule(
    n_samples: int,
    expected_k: tuple[int, int, int],
) -> None:
    """Standard percentages delegate to the existing ceiling conversion."""
    levels = get_standard_capacity_levels(n_samples)

    assert tuple(level.capacity for level in levels) == TOP_K_CAPACITIES
    assert tuple(level.k for level in levels) == expected_k


def test_standard_capacity_level_order_is_deterministic() -> None:
    """Repeated expansion preserves the canonical 5%, 10%, 20% order."""
    first = get_standard_capacity_levels(100)

    assert get_standard_capacity_levels(100) == first
    assert first == (
        CapacityLevel(capacity=0.05, k=5),
        CapacityLevel(capacity=0.10, k=10),
        CapacityLevel(capacity=0.20, k=20),
    )


def test_capacity_levels_are_immutable_and_contain_no_threshold_field() -> None:
    """Capacity points carry workload counts, never classification cutoffs."""
    level = get_standard_capacity_levels(100)[0]

    assert set(level.to_dict()) == {"capacity", "k"}
    with pytest.raises(FrozenInstanceError):
        level.k = 6  # type: ignore[misc]


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_standard_capacity_levels_reuse_sample_count_validation(n_samples) -> None:
    """Invalid counts fail through Phase 5.1's capacity conversion boundary."""
    with pytest.raises(EvaluationError, match="positive integer"):
        get_standard_capacity_levels(n_samples)
