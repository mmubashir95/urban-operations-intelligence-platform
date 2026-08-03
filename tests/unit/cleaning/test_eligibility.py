"""Tests proving Step 7 reuses Step 4 eligibility and target contracts."""

import pandas as pd
import pytest

import urban_ops.cleaning.eligibility as cleaning_eligibility
from urban_ops.cleaning.eligibility import apply_governed_target


def apply(frame: pd.DataFrame, selected_scope, conflicts: set[str] | None = None):
    return apply_governed_target(
        frame, scope=selected_scope, extraction_timestamp="2026-12-31T00:00:00Z",
        conflicting_keys=conflicts or set(),
    )


def test_step4_eligibility_and_target_functions_are_called(
    raw_frame, selected_scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    eligibility_calls = 0
    target_calls = 0
    real_eligibility = cleaning_eligibility.evaluate_target_eligibility
    real_target = cleaning_eligibility.build_missed_resolution_target

    def eligibility_spy(*args, **kwargs):
        nonlocal eligibility_calls
        eligibility_calls += 1
        return real_eligibility(*args, **kwargs)

    def target_spy(*args, **kwargs):
        nonlocal target_calls
        target_calls += 1
        return real_target(*args, **kwargs)

    monkeypatch.setattr(cleaning_eligibility, "evaluate_target_eligibility", eligibility_spy)
    monkeypatch.setattr(cleaning_eligibility, "build_missed_resolution_target", target_spy)
    apply(raw_frame, selected_scope)
    assert eligibility_calls == target_calls == 1


@pytest.mark.parametrize(("closed", "due", "expected"), [
    ("2024-01-09", "2024-01-10", 0),
    ("2024-01-10", "2024-01-10", 0),
    ("2024-01-11", "2024-01-10", 1),
])
def test_governed_target_values(
    raw_frame, selected_scope, closed: str, due: str, expected: int
) -> None:
    frame = raw_frame.iloc[[0]].assign(closed_date=closed, due_date=due)
    result = apply(frame, selected_scope)
    assert int(result.eligible.loc[0, "missed_resolution_target"]) == expected


@pytest.mark.parametrize(("changes", "reason"), [
    ({"due_date": None}, "missing_due_date"),
    ({"closed_date": None}, "missing_closed_date"),
    ({"closed_date": "2023-12-31"}, "closed_before_created"),
])
def test_ineligible_rows_keep_null_target_and_step4_reason(
    raw_frame, selected_scope, changes: dict[str, object], reason: str
) -> None:
    frame = raw_frame.iloc[[0]].copy()
    for column, value in changes.items():
        frame.loc[frame.index[0], column] = value
    result = apply(frame, selected_scope)
    assert result.eligible.empty
    assert pd.isna(result.excluded.loc[0, "missed_resolution_target"])
    assert result.excluded.loc[0, "primary_exclusion_reason"] == reason


def test_cleaning_conflict_override_excludes_entire_group(raw_frame, selected_scope) -> None:
    frame = raw_frame.copy()
    result = apply(frame, selected_scope, conflicts={"1"})
    assert set(result.eligible["unique_key"]) == {"2"}
    excluded = result.excluded.set_index("unique_key")
    assert excluded.loc["1", "primary_exclusion_reason"] == "conflicting_duplicate_unique_key"
    assert pd.isna(excluded.loc["1", "missed_resolution_target"])


def test_input_is_unchanged_and_outputs_reconcile(raw_frame, selected_scope) -> None:
    original = raw_frame.copy(deep=True)
    result = apply(raw_frame, selected_scope)
    assert len(result.cleaned) == len(result.eligible) + len(result.excluded)
    assert result.eligible["unique_key"].is_unique
    assert set(result.eligible["missed_resolution_target"].astype(int)).issubset({0, 1})
    pd.testing.assert_frame_equal(raw_frame, original)
