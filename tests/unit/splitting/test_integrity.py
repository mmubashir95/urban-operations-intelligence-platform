"""Tests for chronology, identifiers, targets, minimums, and source immutability."""

import pandas as pd
import pytest

from urban_ops.splitting.assignment import assign_time_splits
from urban_ops.splitting.integrity import build_integrity_checks, require_integrity


def checks_for(eligible_frame, boundaries, *, minimum_rows=1, minimum_class=1, after_hash="same", after_mtime=1):
    """Build checks for a canonical assignment with configurable failure gates."""
    frames = assign_time_splits(
        eligible_frame, boundaries=boundaries, timestamp_column="created_date",
        identifier_column="unique_key",
    )
    return build_integrity_checks(
        eligible_frame, frames, boundaries=boundaries,
        timestamp_column="created_date", identifier_column="unique_key",
        target_column="missed_resolution_target", minimum_rows=minimum_rows,
        minimum_class_rows=minimum_class, source_hash_before="same",
        source_hash_after=after_hash, source_mtime_before=1,
        source_mtime_after=after_mtime,
    )


def test_valid_assignment_passes_chronology_ids_targets_and_reconciliation(
    eligible_frame, boundaries
) -> None:
    checks = checks_for(eligible_frame, boundaries)
    require_integrity(checks)
    status = {check.check_id: check.status for check in checks}
    assert status["chronology.train_before_validation"] == "PASS"
    assert status["chronology.validation_before_test"] == "PASS"
    assert status["identifier.train_validation_overlap"] == "PASS"
    assert status["identifier.train_test_overlap"] == "PASS"
    assert status["identifier.validation_test_overlap"] == "PASS"
    assert status["target.values_unchanged"] == "PASS"


def test_minimum_rows_and_class_counts_fail_closed(eligible_frame, boundaries) -> None:
    checks = checks_for(eligible_frame, boundaries, minimum_rows=100, minimum_class=100)
    failures = {check.check_id for check in checks if check.status == "FAIL"}
    assert "minimum.train_rows" in failures
    assert "minimum.test_class" in failures
    with pytest.raises(ValueError, match="minimum"):
        require_integrity(checks)


def test_source_hash_and_mtime_changes_fail(eligible_frame, boundaries) -> None:
    checks = checks_for(eligible_frame, boundaries, after_hash="changed", after_mtime=2)
    failures = {check.check_id for check in checks if check.status == "FAIL"}
    assert failures >= {
        "boundary.source_hash_immutable", "boundary.source_mtime_immutable",
    }


def test_duplicate_and_null_source_contracts_are_reported(eligible_frame, boundaries) -> None:
    frames = assign_time_splits(eligible_frame, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    changed = eligible_frame.copy()
    changed.loc[1, "unique_key"] = changed.loc[0, "unique_key"]
    checks = build_integrity_checks(
        changed, frames, boundaries=boundaries, timestamp_column="created_date",
        identifier_column="unique_key", target_column="missed_resolution_target",
        minimum_rows=1, minimum_class_rows=1, source_hash_before="x",
        source_hash_after="x", source_mtime_before=1, source_mtime_after=1,
    )
    assert next(check for check in checks if check.check_id == "source.identifiers_unique").status == "FAIL"
