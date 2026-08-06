"""Build fail-closed split integrity evidence without altering split records."""

from __future__ import annotations

import pandas as pd

from urban_ops.splitting.models import (
    SplitBoundaries, SplitFrames, SplitIntegrityCheck,
)


def _check(
    check_id: str, area: str, passed: bool, observed: object, expected: object,
    affected_rows: int, message: str,
) -> SplitIntegrityCheck:
    """Create one normalized PASS or FAIL integrity result."""
    return SplitIntegrityCheck(
        check_id, area, "PASS" if passed else "FAIL", observed, expected,
        int(affected_rows), message,
    )


def build_integrity_checks(
    source: pd.DataFrame,
    frames: SplitFrames,
    *,
    boundaries: SplitBoundaries,
    timestamp_column: str,
    identifier_column: str,
    target_column: str,
    minimum_rows: int,
    minimum_class_rows: int,
    source_hash_before: str,
    source_hash_after: str,
    source_mtime_before: int,
    source_mtime_after: int,
) -> tuple[SplitIntegrityCheck, ...]:
    """Return source, assignment, chronology, ID, target, and immutability checks."""
    checks: list[SplitIntegrityCheck] = []
    checks.extend([
        _check("source.non_empty", "source", not source.empty, len(source), "> 0", 0, "Eligible source is non-empty."),
        _check("source.identifiers_non_null", "source", source[identifier_column].notna().all(), int(source[identifier_column].isna().sum()), 0, int(source[identifier_column].isna().sum()), "Identifiers are non-null."),
        _check("source.identifiers_unique", "source", source[identifier_column].is_unique, int(source[identifier_column].duplicated().sum()), 0, int(source[identifier_column].duplicated().sum()), "Identifiers are unique."),
        _check("source.timestamps_non_null", "source", source[timestamp_column].notna().all(), int(source[timestamp_column].isna().sum()), 0, int(source[timestamp_column].isna().sum()), "Split timestamps are non-null."),
        _check("source.targets_non_null", "target", source[target_column].notna().all(), int(source[target_column].isna().sum()), 0, int(source[target_column].isna().sum()), "Targets are non-null."),
        _check("source.targets_binary", "target", set(source[target_column].dropna().astype(int).unique()).issubset({0, 1}), "binary", "{0,1}", 0, "Targets are binary."),
    ])
    total = sum(len(getattr(frames, name)) for name in ("train", "validation", "test"))
    checks.extend([
        _check("assignment.all_rows", "assignment", total == len(source), total, len(source), abs(total - len(source)), "Every source row is assigned."),
        _check("assignment.unassigned", "assignment", frames.unassigned_row_count == 0, frames.unassigned_row_count, 0, frames.unassigned_row_count, "No source row is unassigned."),
        _check("assignment.multiple", "assignment", frames.multiply_assigned_row_count == 0, frames.multiply_assigned_row_count, 0, frames.multiply_assigned_row_count, "No source row is multiply assigned."),
    ])
    ranges = {
        "train": (boundaries.train_start, boundaries.train_end_exclusive),
        "validation": (boundaries.validation_start, boundaries.validation_end_exclusive),
        "test": (boundaries.test_start, boundaries.test_end_exclusive),
    }
    for name, (start, end) in ranges.items():
        split = getattr(frames, name)
        in_range = split[timestamp_column].ge(start) & split[timestamp_column].lt(end)
        checks.append(_check(
            f"chronology.{name}_range", "chronology", in_range.all(),
            int((~in_range).sum()), 0, int((~in_range).sum()),
            f"{name} rows satisfy their configured half-open range.",
        ))
        target = split[target_column].astype(int)
        class_min = min(int(target.eq(0).sum()), int(target.eq(1).sum()))
        checks.extend([
            _check(f"minimum.{name}_rows", "minimum", len(split) >= minimum_rows, len(split), minimum_rows, max(minimum_rows - len(split), 0), f"{name} meets the minimum row count."),
            _check(f"minimum.{name}_class", "minimum", class_min >= minimum_class_rows, class_min, minimum_class_rows, max(minimum_class_rows - class_min, 0), f"{name} meets the minimum per-class count."),
            _check(f"target.{name}_both_classes", "target", set(target.unique()) == {0, 1}, sorted(target.unique()), "[0, 1]", 0, f"{name} contains both target classes."),
        ])
    train_max, validation_min = frames.train[timestamp_column].max(), frames.validation[timestamp_column].min()
    validation_max, test_min = frames.validation[timestamp_column].max(), frames.test[timestamp_column].min()
    checks.extend([
        _check("chronology.train_before_validation", "chronology", train_max < validation_min, train_max, validation_min, 0 if train_max < validation_min else 1, "Train observations precede validation observations."),
        _check("chronology.validation_before_test", "chronology", validation_max < test_min, validation_max, test_min, 0 if validation_max < test_min else 1, "Validation observations precede test observations."),
    ])
    id_sets = {name: set(getattr(frames, name)[identifier_column]) for name in ("train", "validation", "test")}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = id_sets[left] & id_sets[right]
        checks.append(_check(
            f"identifier.{left}_{right}_overlap", "identifier", not overlap,
            len(overlap), 0, len(overlap), f"{left} and {right} identifiers do not overlap.",
        ))
    if source[identifier_column].is_unique and source[target_column].notna().all():
        source_targets = source.set_index(identifier_column)[target_column].astype(int)
        output_targets = pd.concat([
            getattr(frames, name)[[identifier_column, target_column]]
            for name in ("train", "validation", "test")
        ]).set_index(identifier_column)[target_column].astype(int)
        source_targets = source_targets.sort_index()
        output_targets = output_targets.sort_index()
        changed = (
            int((source_targets != output_targets).sum())
            if source_targets.index.equals(output_targets.index)
            else len(source)
        )
    else:
        changed = len(source)
    checks.extend([
        _check("target.values_unchanged", "target", changed == 0, changed, 0, changed, "Target values are unchanged by assignment."),
        _check("boundary.source_hash_immutable", "boundary", source_hash_before == source_hash_after, source_hash_after, source_hash_before, 0 if source_hash_before == source_hash_after else len(source), "Eligible source bytes remain unchanged."),
        _check("boundary.source_mtime_immutable", "boundary", source_mtime_before == source_mtime_after, source_mtime_after, source_mtime_before, 0 if source_mtime_before == source_mtime_after else 1, "Eligible source modification time remains unchanged."),
    ])
    return tuple(checks)


def require_integrity(checks: tuple[SplitIntegrityCheck, ...]) -> None:
    """Raise with actionable check IDs if any mandatory check failed."""
    failures = [check.check_id for check in checks if check.status == "FAIL"]
    if failures:
        raise ValueError("Time-split integrity checks failed: " + ", ".join(failures))
