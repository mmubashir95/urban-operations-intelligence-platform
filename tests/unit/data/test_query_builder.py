"""Unit tests for deterministic and safe Socrata query construction."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.data.query_builder import build_query_plan


COLUMNS = ["unique_key", "created_date", "agency", "complaint_type"]
ORDERING = ["created_date ASC", "unique_key ASC"]


def plan(**overrides: object):
    values = {
        "agency": "DSNY",
        "complaint_type": "Graffiti",
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "selected_columns": COLUMNS,
        "order_by": ORDERING,
        "page_size": 10_000,
    }
    values.update(overrides)
    return build_query_plan(**values)


def test_scope_boundaries_filters_and_count_query_match() -> None:
    result = plan()
    query = result.page_query(offset=0)
    assert "agency = 'DSNY'" in query
    assert "complaint_type = 'Graffiti'" in query
    assert "created_date >= '2024-01-01T00:00:00.000'" in query
    assert "created_date < '2026-01-01T00:00:00.000'" in query
    assert result.where_clause in result.count_query


def test_strings_are_escaped_and_credentials_never_appear() -> None:
    result = plan(agency="D'SNY")
    assert "D''SNY" in result.where_clause
    audit = result.audit_text()
    assert "authorization" not in audit.casefold()
    assert "x-app-token" not in audit.casefold()
    assert "secret-token" not in audit


def test_columns_order_limit_and_offset_are_deterministic() -> None:
    result = plan()
    query = result.page_query(offset=20_000)
    assert query.startswith("SELECT unique_key, created_date, agency, complaint_type")
    assert "ORDER BY created_date ASC, unique_key ASC" in query
    assert query.endswith("LIMIT 10000 OFFSET 20000")
    assert result.page_query(offset=20_000) == query


@pytest.mark.parametrize("column", ["closed_date; DROP TABLE x", "not_a_source_field"])
def test_invalid_columns_are_rejected(column: str) -> None:
    with pytest.raises(ValueError, match="Invalid or unsupported"):
        plan(selected_columns=[column])


@pytest.mark.parametrize("page_size", [0, -1, 50_001, 1.5])
def test_invalid_page_sizes_are_rejected(page_size: object) -> None:
    with pytest.raises(ValueError, match="Page size"):
        plan(page_size=page_size)


def test_negative_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Offset"):
        plan().page_query(offset=-1)


def test_start_after_end_is_rejected_but_single_day_is_valid() -> None:
    with pytest.raises(ValueError, match="must not be after"):
        plan(start_date="2025-01-02", end_date="2025-01-01")
    assert "2025-01-02T00:00:00.000" in plan(
        start_date="2025-01-01", end_date="2025-01-01"
    ).where_clause


def test_empty_and_duplicate_selected_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        plan(selected_columns=[])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        plan(selected_columns=["unique_key", "unique_key"])


def test_query_hash_is_stable_and_scope_sensitive() -> None:
    first, second = plan(), plan()
    changed = plan(complaint_type="Missed Collection")
    assert first.query_hash == second.query_hash
    assert first.page_query(offset=0) == second.page_query(offset=0)
    assert first.query_hash != changed.query_hash


def test_invalid_ordering_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid ordering"):
        plan(order_by=["created_date ASC; DROP"])
