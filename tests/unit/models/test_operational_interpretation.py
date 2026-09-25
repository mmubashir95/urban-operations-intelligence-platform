"""Unit coverage for Phase 5.6 plain-language capacity interpretation."""

import pytest

from urban_ops.models.baseline_workflow import (
    build_logistic_validation_capacity_comparison_table,
    format_capacity_interpretation,
    format_capacity_step_interpretation,
)
from urban_ops.models.evaluation import compare_capacity_levels

# Words that would overclaim intervention impact or pick a capacity winner.
PROHIBITED_PHRASES = ("prevent", "saved", "avoided", "optimal", "best capacity")


def _flat(text: str) -> str:
    """Collapse Markdown line wrapping so assertions target wording, not layout."""
    return " ".join(text.split())


@pytest.fixture
def capacity_rows() -> list:
    """Rows with K = 3, 6, 12 of 60 and captured misses 3, 4, 7 of 20."""
    top_twelve = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    labels = top_twelve + [0] * 8 + [1] * 13 + [0] * 27
    scores = [1.0 - index / 60 for index in range(60)]
    table = build_logistic_validation_capacity_comparison_table(
        compare_capacity_levels(labels, scores)
    )
    return list(table.itertuples(index=False))


def test_capacity_rows_fixture_has_expected_source_values(capacity_rows) -> None:
    """Guard the fixture so the wording tests assert against known values."""
    assert [row.capacity_pct for row in capacity_rows] == [5.0, 10.0, 20.0]
    assert [row.selected_count for row in capacity_rows] == [3, 6, 12]
    assert [row.captured_positive_count for row in capacity_rows] == [3, 4, 7]
    assert [round(row.recall_at_k, 4) for row in capacity_rows] == [
        0.15,
        0.2,
        0.35,
    ]


def test_capacity_interpretation_formats_source_metrics(capacity_rows) -> None:
    """Each statement shows the row's capacity, counts, recall, and precision."""
    raw = format_capacity_interpretation(capacity_rows[1])
    text = _flat(raw)

    assert raw.startswith("### 10% capacity")
    assert "On the validation set, the highest-risk 10% of complaints" in text
    assert "reviews the 6 highest-risk complaints" in text
    assert "4 actual missed-target complaints fall within this review queue" in text
    assert "the queue contains 20.0% of all actual missed-target" in text
    assert "66.7% of reviewed complaints are actual missed-target cases" in text
    assert "roughly 67 out of every 100" in text


def test_capacity_interpretation_keeps_recall_and_precision_distinct(
    capacity_rows,
) -> None:
    """Recall is framed over all misses; precision over the reviewed queue."""
    text = _flat(format_capacity_interpretation(capacity_rows[2]))

    assert "Recall@K):** the queue contains 35.0% of all actual" in text
    assert "Precision@K):** 58.3% of reviewed complaints" in text


def test_capacity_interpretation_names_the_supplied_split(capacity_rows) -> None:
    """The split label is explicit rather than implied."""
    text = format_capacity_interpretation(capacity_rows[0], split_name="validation")

    assert text.count("validation set") == 2


def test_step_interpretation_uses_adjacent_increments(capacity_rows) -> None:
    """Steps report extra reviews, extra misses, and a percentage-point gain."""
    first_raw = format_capacity_step_interpretation(*capacity_rows[0:2])
    second_raw = format_capacity_step_interpretation(*capacity_rows[1:3])
    first_step, second_step = _flat(first_raw), _flat(second_raw)

    assert first_raw.startswith("### 5% → 10%")
    assert "requires reviewing 3 additional complaints" in first_step
    assert "adds 1 actual missed-target complaints" in first_step
    assert "increases by 5.00 percentage points (15.0% to 20.0%)" in first_step
    assert second_raw.startswith("### 10% → 20%")
    assert "requires reviewing 6 additional complaints" in second_step
    assert "adds 3 actual missed-target complaints" in second_step
    assert "increases by 15.00 percentage points (20.0% to 35.0%)" in second_step


def test_interpretations_do_not_overclaim_or_pick_a_winner(capacity_rows) -> None:
    """Formatter output never claims prevention or a preferred capacity."""
    texts = [format_capacity_interpretation(row) for row in capacity_rows]
    texts += [
        format_capacity_step_interpretation(previous, current)
        for previous, current in zip(capacity_rows, capacity_rows[1:])
    ]

    for text in texts:
        lowered = text.lower()
        for phrase in PROHIBITED_PHRASES:
            assert phrase not in lowered
        assert "relative" not in lowered


def test_interpretations_are_deterministic(capacity_rows) -> None:
    """Formatting the same rows twice gives identical text."""
    first = [format_capacity_interpretation(row) for row in capacity_rows]
    second = [format_capacity_interpretation(row) for row in capacity_rows]

    assert first == second
