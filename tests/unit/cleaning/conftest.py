"""Shared deterministic fixtures for Step 7 cleaning unit tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from urban_ops.data.selected_scope import SelectedScope


COLUMNS = (
    "unique_key", "created_date", "closed_date", "due_date", "agency",
    "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
    "borough", "incident_zip", "latitude", "longitude", "location_type",
    "open_data_channel_type", "resolution_description",
    "resolution_action_updated_date",
)


def make_row(key: str = "1", **changes: object) -> dict[str, object]:
    """Return one complete raw row suitable for cleaning tests."""
    row: dict[str, object] = {
        "unique_key": key, "created_date": "2024-01-01T12:00:00.000",
        "closed_date": "2024-01-09T12:00:00.000",
        "due_date": "2024-01-10T12:00:00.000", "agency": "DSNY",
        "agency_name": "Department of Sanitation", "complaint_type": "Graffiti",
        "descriptor": "Graffiti", "descriptor_2": None, "status": "Closed",
        "borough": "QUEENS", "incident_zip": "01234", "latitude": "40.75",
        "longitude": "-73.90", "location_type": "Residential",
        "open_data_channel_type": "UNKNOWN", "resolution_description": "Resolved",
        "resolution_action_updated_date": "2024-01-09T12:00:00.000",
    }
    row.update(changes)
    return row


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Return two valid raw rows with the production source schema."""
    return pd.DataFrame([make_row("1"), make_row("2")], columns=COLUMNS)


@pytest.fixture
def selected_scope(tmp_path: Path) -> SelectedScope:
    """Return the authoritative calendar scope used by cleaning tests."""
    return SelectedScope(
        agency="DSNY", complaint_type="Graffiti",
        start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2025-12-31"),
        authority_extraction_timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        decision_status="APPROVED_WITH_LIMITATIONS",
        authority_path=tmp_path / "scope.csv",
        extraction_metadata_path=tmp_path / "scope_metadata.csv",
    )
