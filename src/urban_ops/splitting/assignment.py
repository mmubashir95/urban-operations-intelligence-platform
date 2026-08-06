"""Assign eligible complaints to deterministic UTC half-open time partitions."""

from __future__ import annotations

import pandas as pd

from urban_ops.splitting.models import SplitBoundaries, SplitFrames


class SplitAssignmentError(ValueError):
    """Raised when timestamp parsing or partition assignment is incomplete."""


def assign_time_splits(
    frame: pd.DataFrame,
    *,
    boundaries: SplitBoundaries,
    timestamp_column: str,
    identifier_column: str,
    require_all_assigned: bool = True,
) -> SplitFrames:
    """Assign and sort rows without randomization, target use, or input mutation."""
    if timestamp_column not in frame:
        raise SplitAssignmentError(f"Missing split timestamp: {timestamp_column}")
    timestamps = frame[timestamp_column]
    if timestamps.isna().any() or not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise SplitAssignmentError("Split timestamps must be valid timezone-aware values.")
    train_mask = timestamps.ge(boundaries.train_start) & timestamps.lt(
        boundaries.train_end_exclusive
    )
    validation_mask = timestamps.ge(boundaries.validation_start) & timestamps.lt(
        boundaries.validation_end_exclusive
    )
    test_mask = timestamps.ge(boundaries.test_start) & timestamps.lt(
        boundaries.test_end_exclusive
    )
    assignment_count = (
        train_mask.astype("int8") + validation_mask.astype("int8") + test_mask.astype("int8")
    )
    unassigned = int(assignment_count.eq(0).sum())
    multiply_assigned = int(assignment_count.gt(1).sum())
    if multiply_assigned:
        raise SplitAssignmentError(f"{multiply_assigned} rows match multiple splits.")
    if require_all_assigned and unassigned:
        raise SplitAssignmentError(f"{unassigned} rows are outside configured boundaries.")

    def selected(mask: pd.Series) -> pd.DataFrame:
        return frame.loc[mask].copy().sort_values(
            [timestamp_column, identifier_column], kind="mergesort"
        ).reset_index(drop=True)

    return SplitFrames(
        train=selected(train_mask), validation=selected(validation_mask),
        test=selected(test_mask), unassigned_row_count=unassigned,
        multiply_assigned_row_count=multiply_assigned,
    )
