"""Build deterministic, scope-safe Socrata queries for NYC 311 ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from urllib.parse import urlencode

import pandas as pd

from urban_ops.data.nyc_311_config import SELECTED_COLUMNS


VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_SOURCE_COLUMNS = frozenset(SELECTED_COLUMNS)
ALLOWED_ORDER_DIRECTIONS = frozenset({"ASC", "DESC"})


@dataclass(frozen=True)
class QueryPlan:
    """Deterministic query contract shared by preflight and page retrieval."""

    selected_columns: tuple[str, ...]
    where_clause: str
    order_by: tuple[str, ...]
    page_size: int
    count_query: str
    query_hash: str

    @property
    def ordering_clause(self) -> str:
        """Return the stable comma-separated ordering expression."""
        return ", ".join(self.order_by)

    def page_query(self, *, offset: int, limit: int | None = None) -> str:
        """Return one deterministic paginated SoQL query."""
        requested_limit = self.page_size if limit is None else limit
        _validate_page(requested_limit, offset)
        return (
            f"SELECT {', '.join(self.selected_columns)} "
            f"WHERE {self.where_clause} "
            f"ORDER BY {self.ordering_clause} "
            f"LIMIT {requested_limit} OFFSET {offset}"
        )

    def page_url(self, base_url: str, *, offset: int, limit: int | None = None) -> str:
        """Return a URL containing one encoded paginated query."""
        return f"{base_url}?{urlencode({'$query': self.page_query(offset=offset, limit=limit)})}"

    def count_url(self, base_url: str) -> str:
        """Return a URL containing the encoded count query."""
        return f"{base_url}?{urlencode({'$query': self.count_query})}"

    def audit_text(self) -> str:
        """Return a credential-free human-readable query audit artifact."""
        return (
            "-- Deterministic NYC 311 ingestion query\n"
            f"-- query_hash: {self.query_hash}\n"
            f"-- page_size: {self.page_size}\n\n"
            f"-- count query\n{self.count_query};\n\n"
            "-- paginated query template\n"
            f"SELECT {', '.join(self.selected_columns)}\n"
            f"WHERE {self.where_clause}\n"
            f"ORDER BY {self.ordering_clause}\n"
            "LIMIT :limit OFFSET :offset;\n"
        )


def escape_soql_string(value: str) -> str:
    """Escape a value for a single-quoted SoQL string literal."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SoQL string filter values must be non-empty strings.")
    return value.replace("'", "''")


def _validate_column(column: str) -> None:
    """Reject unrecognized or syntactically unsafe source identifiers."""
    if not VALID_IDENTIFIER.fullmatch(column) or column not in ALLOWED_SOURCE_COLUMNS:
        raise ValueError(f"Invalid or unsupported source column: {column!r}")


def _validate_ordering(order_by: tuple[str, ...]) -> None:
    """Validate simple configured column/direction ordering expressions."""
    if not order_by:
        raise ValueError("At least one deterministic ordering field is required.")
    for expression in order_by:
        parts = expression.split()
        if len(parts) != 2 or parts[1].upper() not in ALLOWED_ORDER_DIRECTIONS:
            raise ValueError(f"Invalid ordering expression: {expression!r}")
        _validate_column(parts[0])


def _validate_page(limit: int, offset: int) -> None:
    """Validate one page boundary."""
    if not isinstance(limit, int) or limit < 1 or limit > 50_000:
        raise ValueError("Page size must be an integer between 1 and 50000.")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("Offset must be a non-negative integer.")


def build_query_plan(
    *,
    agency: str,
    complaint_type: str,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    selected_columns: list[str] | tuple[str, ...],
    order_by: list[str] | tuple[str, ...],
    page_size: int,
) -> QueryPlan:
    """Build and hash a deterministic query plan for an inclusive date scope."""
    if not selected_columns:
        raise ValueError("Selected source columns must not be empty.")
    columns = tuple(selected_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Selected source columns must not contain duplicates.")
    for column in columns:
        _validate_column(column)
    raw_ordering = tuple(order_by)
    _validate_ordering(raw_ordering)
    ordering = tuple(
        f"{expression.split()[0]} {expression.split()[1].upper()}"
        for expression in raw_ordering
    )
    _validate_page(page_size, 0)
    try:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("Scope dates must be valid calendar dates.") from error
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Scope start date must not be after its end date.")
    end_exclusive = end + pd.Timedelta(days=1)
    where = (
        f"agency = '{escape_soql_string(agency)}' "
        f"AND complaint_type = '{escape_soql_string(complaint_type)}' "
        f"AND created_date >= '{start:%Y-%m-%d}T00:00:00.000' "
        f"AND created_date < '{end_exclusive:%Y-%m-%d}T00:00:00.000'"
    )
    count_query = f"SELECT count(*) AS record_count WHERE {where}"
    canonical = {
        "selected_columns": columns,
        "where_clause": where,
        "order_by": ordering,
        "page_size": page_size,
    }
    query_hash = sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return QueryPlan(columns, where, ordering, page_size, count_query, query_hash)
