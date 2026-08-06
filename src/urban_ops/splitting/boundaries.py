"""Validate, evaluate, rank, and select chronological split boundary candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from urban_ops.splitting.assignment import assign_time_splits
from urban_ops.splitting.models import CandidateSplitResult, SplitBoundaries, SplitFrames


SUPPORTED_INTERVAL = "left_closed_right_open"


def utc_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    """Parse a configured boundary, interpreting a timezone-naive value as UTC."""
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid split boundary {field_name}: {value!r}") from error
    if pd.isna(parsed):
        raise ValueError(f"Invalid split boundary {field_name}: {value!r}")
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def boundaries_from_mapping(
    payload: Mapping[str, Any], *, interval_convention: str
) -> SplitBoundaries:
    """Build typed UTC boundaries from either nested or flattened configuration."""
    if {"train", "validation", "test"}.issubset(payload):
        train = payload["train"]
        validation = payload["validation"]
        test = payload["test"]
        if not all(isinstance(item, dict) for item in (train, validation, test)):
            raise ValueError("Each configured split boundary must be a mapping.")
        values = {
            "train_start": train["start"],
            "train_end_exclusive": train["end_exclusive"],
            "validation_start": validation["start"],
            "validation_end_exclusive": validation["end_exclusive"],
            "test_start": test["start"],
            "test_end_exclusive": test["end_exclusive"],
        }
    else:
        values = {name: payload[name] for name in (
            "train_start", "train_end_exclusive", "validation_start",
            "validation_end_exclusive", "test_start", "test_end_exclusive",
        )}
    return SplitBoundaries(
        **{
            name: utc_timestamp(value, field_name=name)
            for name, value in values.items()
        },
        interval_convention=interval_convention,
    )


def validate_boundaries(
    boundaries: SplitBoundaries,
    *,
    source_min: pd.Timestamp | None = None,
    source_max: pd.Timestamp | None = None,
) -> None:
    """Require ordered contiguous half-open ranges that fully cover the source."""
    if boundaries.interval_convention != SUPPORTED_INTERVAL:
        raise ValueError(f"Unsupported interval convention: {boundaries.interval_convention}")
    ordered = (
        boundaries.train_start < boundaries.train_end_exclusive
        == boundaries.validation_start < boundaries.validation_end_exclusive
        == boundaries.test_start < boundaries.test_end_exclusive
    )
    if not ordered:
        raise ValueError("Split boundaries must be ordered, contiguous, and non-overlapping.")
    if source_min is not None and boundaries.train_start > source_min:
        raise ValueError("Split boundaries do not include the earliest source row.")
    if source_max is not None and boundaries.test_end_exclusive <= source_max:
        raise ValueError("Split boundaries do not include the latest source row.")


def _split_metrics(
    frames: SplitFrames,
    *,
    total_rows: int,
    timestamp_column: str,
    target_column: str,
    minimum_rows: int,
    minimum_class_rows: int,
) -> dict[str, Any]:
    """Calculate candidate counts, rates, coverage, and threshold evidence."""
    metrics: dict[str, Any] = {}
    rates: list[float] = []
    minimum_monthly_rows: list[int] = []
    both_classes = True
    minimum_rows_passed = True
    minimum_class_counts_passed = True
    for name in ("train", "validation", "test"):
        split = getattr(frames, name)
        target = split[target_column].astype(int)
        on_time, missed = int(target.eq(0).sum()), int(target.eq(1).sum())
        rate = float(target.mean()) if len(target) else float("nan")
        month_counts = (
            split[timestamp_column].dt.tz_localize(None).dt.to_period("M").value_counts()
            if len(split) else pd.Series(dtype="int64")
        )
        metrics.update({
            f"{name}_rows": len(split), f"{name}_share": len(split) / total_rows,
            f"{name}_months": int(len(month_counts)), f"{name}_on_time": on_time,
            f"{name}_missed": missed, f"{name}_missed_rate": rate,
            f"{name}_minimum_class_count": min(on_time, missed),
            f"{name}_minimum_monthly_row_count": int(month_counts.min()) if len(month_counts) else 0,
        })
        rates.append(rate)
        minimum_monthly_rows.append(metrics[f"{name}_minimum_monthly_row_count"])
        both_classes &= on_time > 0 and missed > 0
        minimum_rows_passed &= len(split) >= minimum_rows
        minimum_class_counts_passed &= min(on_time, missed) >= minimum_class_rows
    metrics.update({
        "minimum_monthly_row_count": min(minimum_monthly_rows),
        "date_coverage_gaps": frames.unassigned_row_count,
        "maximum_rate_difference": max(rates) - min(rates),
        "chronology_passed": True,
        "all_rows_assigned": frames.unassigned_row_count == 0,
        "all_splits_have_both_classes": both_classes,
        "minimum_rows_passed": minimum_rows_passed,
        "minimum_class_counts_passed": minimum_class_counts_passed,
    })
    return metrics


def evaluate_candidates(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[tuple[str, SplitBoundaries]],
    selected_candidate_id: str,
    timestamp_column: str,
    identifier_column: str,
    target_column: str,
    minimum_rows: int,
    minimum_class_rows: int,
    maximum_target_rate_difference: float | None,
    drift_is_warning_only: bool,
) -> tuple[CandidateSplitResult, ...]:
    """Evaluate multiple candidates and require the explicit selection to rank first."""
    if len(candidates) < 3:
        raise ValueError("At least three chronological boundary candidates are required.")
    if len({candidate_id for candidate_id, _ in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique.")
    provisional: list[tuple[str, SplitBoundaries, dict[str, Any], bool, str]] = []
    for candidate_id, boundaries in candidates:
        validate_boundaries(
            boundaries, source_min=frame[timestamp_column].min(),
            source_max=frame[timestamp_column].max(),
        )
        frames = assign_time_splits(
            frame, boundaries=boundaries, timestamp_column=timestamp_column,
            identifier_column=identifier_column, require_all_assigned=False,
        )
        metrics = _split_metrics(
            frames, total_rows=len(frame), timestamp_column=timestamp_column,
            target_column=target_column, minimum_rows=minimum_rows,
            minimum_class_rows=minimum_class_rows,
        )
        hard_drift_pass = (
            maximum_target_rate_difference is None
            or metrics["maximum_rate_difference"] <= maximum_target_rate_difference
            or drift_is_warning_only
        )
        qualified = all([
            metrics["all_rows_assigned"], metrics["chronology_passed"],
            metrics["all_splits_have_both_classes"], metrics["minimum_rows_passed"],
            metrics["minimum_class_counts_passed"], hard_drift_pass,
        ])
        failures = [
            label for label, passed in (
                ("full coverage", metrics["all_rows_assigned"]),
                ("both classes", metrics["all_splits_have_both_classes"]),
                ("minimum rows", metrics["minimum_rows_passed"]),
                ("minimum class counts", metrics["minimum_class_counts_passed"]),
                ("configured drift threshold", hard_drift_pass),
            ) if not passed
        ]
        reason = (
            "Passes all integrity and minimum-size gates."
            if qualified else "Fails: " + ", ".join(failures) + "."
        )
        provisional.append((candidate_id, boundaries, metrics, qualified, reason))
    ranked = sorted(
        provisional,
        key=lambda item: (
            not item[3], item[2]["maximum_rate_difference"],
            -min(item[2][f"{name}_rows"] for name in ("validation", "test")),
            item[0],
        ),
    )
    rank_by_id = {item[0]: rank for rank, item in enumerate(ranked, start=1)}
    if selected_candidate_id not in rank_by_id:
        raise ValueError(f"Selected candidate does not exist: {selected_candidate_id}")
    selected = next(item for item in provisional if item[0] == selected_candidate_id)
    if not selected[3]:
        raise ValueError("Explicitly selected candidate fails required integrity gates.")
    if rank_by_id[selected_candidate_id] != 1:
        raise ValueError("Explicitly selected candidate is not the deterministic top-ranked candidate.")
    results = []
    for candidate_id, boundaries, metrics, qualified, reason in provisional:
        recommended = candidate_id == selected_candidate_id
        if recommended:
            reason += (
                " Selected because it has the smallest split-level target-rate spread; "
                "ties favor larger holdouts and then candidate ID."
            )
        elif qualified:
            reason += " Qualified but ranked below the selected candidate."
        results.append(CandidateSplitResult(
            candidate_id=candidate_id, boundaries=boundaries, metrics=metrics,
            qualified=qualified, rank=rank_by_id[candidate_id], recommended=recommended,
            decision_reason=reason,
        ))
    return tuple(results)


def selected_boundaries(
    candidates: Sequence[CandidateSplitResult], selected_candidate_id: str
) -> SplitBoundaries:
    """Return the single explicitly recommended boundary contract."""
    selected = [
        item for item in candidates
        if item.candidate_id == selected_candidate_id and item.recommended
    ]
    if len(selected) != 1 or sum(item.recommended for item in candidates) != 1:
        raise ValueError("Exactly one selected candidate is required.")
    return selected[0].boundaries
