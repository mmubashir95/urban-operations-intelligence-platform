"""End-to-end local Step 7 eligible source to immutable chronological split."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from urban_ops.cleaning.outputs import sha256_file
from urban_ops.splitting.pipeline import run_time_based_split
from tests.unit.splitting.conftest import build_split_fixture, make_eligible_frame


def test_complete_time_split_flow_is_atomic_governed_and_non_mutating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = make_eligible_frame()
    boundary_rows = pd.DataFrame([
        {"unique_key": "train-start", "created_date": pd.Timestamp("2024-01-01", tz="UTC"), "missed_resolution_target": 0, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
        {"unique_key": "validation-start", "created_date": pd.Timestamp("2024-04-01", tz="UTC"), "missed_resolution_target": 1, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
        {"unique_key": "test-start", "created_date": pd.Timestamp("2024-07-01", tz="UTC"), "missed_resolution_target": 0, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
        {"unique_key": "train-end-minus", "created_date": pd.Timestamp("2024-03-31T23:59:59.999999Z"), "missed_resolution_target": 1, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
        {"unique_key": "validation-end-minus", "created_date": pd.Timestamp("2024-06-30T23:59:59.999999Z"), "missed_resolution_target": 0, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
        {"unique_key": "test-end-minus", "created_date": pd.Timestamp("2024-12-31T23:59:59.999999Z"), "missed_resolution_target": 1, "target_eligible": True, "agency": "DSNY", "complaint_type": "Graffiti"},
    ])
    boundary_rows["missed_resolution_target"] = boundary_rows["missed_resolution_target"].astype("Int8")
    source_frame = pd.concat([frame, boundary_rows], ignore_index=True)
    fixture = build_split_fixture(tmp_path, source_frame)
    before_bytes = fixture.eligible.read_bytes()
    before_hash, before_mtime = sha256_file(fixture.eligible), fixture.eligible.stat().st_mtime_ns

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Step 8 must not call the network")
    monkeypatch.setattr("urllib.request.urlopen", network_forbidden)

    dry = run_time_based_split(config_path=fixture.config, dry_run=True)
    assert not fixture.output.exists() and not fixture.reports.exists()
    result = run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert dry.frames.train["unique_key"].tolist() == result.frames.train["unique_key"].tolist()
    memberships = {
        name: set(getattr(result.frames, name)["unique_key"])
        for name in ("train", "validation", "test")
    }
    assert {"train-start", "train-end-minus"} <= memberships["train"]
    assert {"validation-start", "validation-end-minus"} <= memberships["validation"]
    assert {"test-start", "test-end-minus"} <= memberships["test"]
    assert sum(len(values) for values in memberships.values()) == len(source_frame)
    assert not (memberships["train"] & memberships["validation"])
    assert not (memberships["train"] & memberships["test"])
    assert not (memberships["validation"] & memberships["test"])
    output = pd.concat([result.frames.train, result.frames.validation, result.frames.test]).set_index("unique_key")
    expected = source_frame.set_index("unique_key")
    pd.testing.assert_series_equal(
        output["missed_resolution_target"].sort_index(),
        expected["missed_resolution_target"].sort_index(),
    )
    assert fixture.eligible.read_bytes() == before_bytes
    assert sha256_file(fixture.eligible) == before_hash
    assert fixture.eligible.stat().st_mtime_ns == before_mtime
    assert result.paths.latest_pointer.is_file()
    assert result.metadata.unassigned_row_count == 0
    assert result.tables["output_reconciliation.csv"]["status"].eq("PASS").all()
    assert (fixture.reports / "split_summary.md").is_file()
