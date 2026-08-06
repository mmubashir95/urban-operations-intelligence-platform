"""Tests for Step 8 configuration, dry/full orchestration, failures, and CLI."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import urban_ops.splitting.pipeline as pipeline_module
from urban_ops.cleaning.outputs import sha256_file
from urban_ops.splitting.pipeline import (
    SplitConfigurationError, load_split_config, main, run_time_based_split,
)
from urban_ops.splitting.source import SplitSourceError
from tests.unit.splitting.conftest import build_split_fixture


def test_config_loads_selected_half_open_created_date_policy(
    tmp_path: Path, eligible_frame
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    config = load_split_config(fixture.config)
    assert config.timestamp_column == "created_date"
    assert config.interval_convention == "left_closed_right_open"
    assert len(config.candidates) == 3
    payload = yaml.safe_load(fixture.config.read_text())
    payload["split"]["timestamp_column"] = "closed_date"
    fixture.config.write_text(yaml.safe_dump(payload))
    with pytest.raises(SplitConfigurationError, match="created_date"):
        load_split_config(fixture.config)


def test_dry_run_writes_no_outputs_reports_or_latest(tmp_path: Path, eligible_frame) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    before_hash, before_mtime = sha256_file(fixture.eligible), fixture.eligible.stat().st_mtime_ns
    result = run_time_based_split(config_path=fixture.config, dry_run=True)
    assert result.dry_run and result.metadata is None
    assert not fixture.output.exists() and not fixture.reports.exists()
    assert not result.paths.latest_pointer.exists()
    assert sha256_file(fixture.eligible) == before_hash
    assert fixture.eligible.stat().st_mtime_ns == before_mtime


def test_full_run_writes_outputs_metadata_reports_snapshot_and_latest(
    tmp_path: Path, eligible_frame
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    result = run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert not result.dry_run and result.metadata is not None
    for path in (
        result.paths.train, result.paths.validation, result.paths.test,
        result.paths.metadata, result.paths.rules_snapshot, result.paths.latest_pointer,
    ):
        assert path.is_file()
    assert (fixture.reports / "split_summary.md").is_file()
    assert json.loads(result.paths.latest_pointer.read_text())["split_id"] == result.metadata.split_id
    assert set(result.metadata.output_hashes) == {"train", "validation", "test", "rules_snapshot"}


def test_duplicate_null_nonbinary_and_one_class_sources_fail(
    tmp_path: Path, eligible_frame
) -> None:
    mutations = [
        pd.concat([eligible_frame, eligible_frame.iloc[[0]]], ignore_index=True),
        eligible_frame.assign(missed_resolution_target=pd.NA),
        eligible_frame.assign(missed_resolution_target=2),
        eligible_frame.assign(missed_resolution_target=0),
    ]
    for index, frame in enumerate(mutations):
        fixture = build_split_fixture(tmp_path / str(index), frame)
        with pytest.raises((SplitSourceError, ValueError)):
            run_time_based_split(config_path=fixture.config, dry_run=True)


def test_hash_scope_and_invalid_boundary_mismatches_fail(tmp_path: Path, eligible_frame) -> None:
    hash_fixture = build_split_fixture(tmp_path / "hash", eligible_frame)
    with hash_fixture.eligible.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(SplitSourceError, match="SHA-256"):
        run_time_based_split(config_path=hash_fixture.config, dry_run=True)

    scope_fixture = build_split_fixture(tmp_path / "scope", eligible_frame)
    scope = pd.read_csv(scope_fixture.scope)
    scope.loc[0, "selected_agency"] = "DOT"
    scope.to_csv(scope_fixture.scope, index=False)
    with pytest.raises(ValueError, match="agency"):
        run_time_based_split(config_path=scope_fixture.config, dry_run=True)

    boundary_fixture = build_split_fixture(tmp_path / "boundary", eligible_frame)
    payload = yaml.safe_load(boundary_fixture.config.read_text())
    payload["boundaries"]["validation"]["start"] = "2024-05-01"
    boundary_fixture.config.write_text(yaml.safe_dump(payload))
    with pytest.raises((SplitConfigurationError, ValueError), match="contiguous|differ"):
        run_time_based_split(config_path=boundary_fixture.config, dry_run=True)


def test_repeated_runs_have_deterministic_membership_order_schema_and_hashes(
    tmp_path: Path, eligible_frame
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame.sample(frac=1, random_state=2))
    first = run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second = run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    for name in ("train", "validation", "test", "rules_snapshot"):
        assert first.metadata.output_hashes[name] == second.metadata.output_hashes[name]
    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(getattr(first.frames, name), getattr(second.frames, name))


def test_successful_run_is_immutable_and_partial_failure_cleans_temporary_output(
    tmp_path: Path, eligible_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_split_fixture(tmp_path / "immutable", eligible_frame)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run_time_based_split(config_path=fixture.config, run_started_at=started)
    with pytest.raises(FileExistsError, match="immutable"):
        run_time_based_split(config_path=fixture.config, run_started_at=started)

    failing = build_split_fixture(tmp_path / "failure", eligible_frame)
    def fail(*args, **kwargs):
        raise RuntimeError("forced metadata failure")
    monkeypatch.setattr(pipeline_module, "write_split_metadata", fail)
    with pytest.raises(RuntimeError, match="forced"):
        run_time_based_split(
            config_path=failing.config,
            run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    assert not list(failing.output.glob("split_id=*"))
    assert not list(failing.output.glob(".*.tmp-*"))
    assert not (failing.output / "latest.json").exists()


def test_cli_success_and_failure_codes(tmp_path: Path, eligible_frame) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    assert main(["--config", str(fixture.config), "--dry-run"]) == 0
    assert main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]) == 1
