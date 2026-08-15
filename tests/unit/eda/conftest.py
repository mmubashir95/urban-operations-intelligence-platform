"""Small governed Step 8 fixtures shared by Step 9A unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from urban_ops.splitting.pipeline import run_time_based_split
from tests.unit.splitting.conftest import build_split_fixture


@dataclass(frozen=True)
class EDAFixture:
    """Paths for one isolated successful Step 8 input and EDA configuration."""

    config: Path
    split_run: Path
    split_root: Path
    latest_pointer: Path
    reports: Path


def make_eda_frame() -> pd.DataFrame:
    """Return representative train, validation, and test rows for EDA tests."""
    rows = []
    key = 0
    for month in range(1, 10):
        for target, day, hour in ((0, 3, 9), (1, 10, 19)):
            key += 1
            split = "train" if month <= 3 else "validation" if month <= 6 else "test"
            location = "Street" if split == "train" else "Park" if split == "validation" else "Bridge"
            rows.append({
                "unique_key": f"eda-{key:03d}",
                "created_date": pd.Timestamp(2024, month, day, hour, tz="UTC"),
                "closed_date": pd.Timestamp(2024, month, day + 2, hour, tz="UTC"),
                "due_date": pd.Timestamp(2024, month, day + 1, hour, tz="UTC"),
                "agency": "DSNY", "agency_name": "Department of Sanitation",
                "complaint_type": "Graffiti", "descriptor": "Graffiti",
                "descriptor_2": pd.NA, "status": "Closed",
                "borough": "Unspecified" if key == 1 else "BROOKLYN",
                "incident_zip": pd.NA if key == 2 else f"112{key % 5:02d}",
                "latitude": pd.NA if key == 3 else str(40.45 + key * 0.01),
                "longitude": pd.NA if key == 3 else str(-74.20 + key * 0.01),
                "location_type": pd.NA if key == 4 else location,
                "open_data_channel_type": "ONLINE",
                "resolution_description": "resolved later",
                "resolution_action_updated_date": pd.Timestamp(2024, month, day + 2, hour, tz="UTC"),
                "target_eligible": True, "primary_exclusion_reason": "eligible",
                "outcome_mature": True, "valid_closed_chronology": True,
                "missed_resolution_target": target,
            })
    frame = pd.DataFrame(rows)
    frame["missed_resolution_target"] = frame["missed_resolution_target"].astype("Int8")
    for column in ("agency", "agency_name", "complaint_type", "descriptor", "descriptor_2", "status", "borough", "incident_zip", "location_type", "open_data_channel_type"):
        frame[column] = frame[column].astype("string")
    return frame


def build_eda_fixture(tmp_path: Path, frame: pd.DataFrame | None = None) -> EDAFixture:
    """Create a successful split run plus a local analysis-only configuration."""
    split_fixture = build_split_fixture(tmp_path, make_eda_frame() if frame is None else frame)
    result = run_time_based_split(
        config_path=split_fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    repository_config = Path(__file__).parents[3] / "configs" / "eda" / "resolution_risk.yaml"
    payload = yaml.safe_load(repository_config.read_text(encoding="utf-8"))
    payload["input"]["split_root"] = str(split_fixture.output)
    payload["input"]["latest_pointer"] = str(split_fixture.output / "latest.json")
    payload["categorical_analysis"]["minimum_support_for_target_rate"] = 2
    payload["output"]["report_root"] = str(tmp_path / "eda_reports")
    config = tmp_path / "eda.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return EDAFixture(
        config=config, split_run=result.paths.run_directory,
        split_root=split_fixture.output, latest_pointer=result.paths.latest_pointer,
        reports=tmp_path / "eda_reports",
    )


@pytest.fixture
def eda_fixture(tmp_path: Path) -> EDAFixture:
    """Provide one fresh successful Step 8 input per test."""
    return build_eda_fixture(tmp_path)
