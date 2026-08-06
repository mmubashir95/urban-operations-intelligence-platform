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
from tests.unit.splitting.conftest import SplitFixture, build_split_fixture


def _directory_bytes(root: Path) -> dict[str, bytes]:
    """Capture exact recursive file content for rollback assertions."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def _assert_publication_rolled_back(
    *, fixture: SplitFixture, previous_reports: dict[str, bytes], previous_latest: bytes,
    previous_runs: set[Path], source_bytes: bytes, source_mtime: int,
) -> None:
    """Assert every split publication surface matches its pre-run state."""
    assert _directory_bytes(fixture.reports) == previous_reports
    assert fixture.output.joinpath("latest.json").read_bytes() == previous_latest
    assert set(fixture.output.glob("split_id=*")) == previous_runs
    assert not list(fixture.output.glob(".*.tmp-*"))
    assert not list(fixture.output.glob("latest*.tmp"))
    assert not list(fixture.reports.parent.glob(f".{fixture.reports.name}.tmp-*"))
    assert not list(fixture.reports.parent.glob(f".{fixture.reports.name}.backup-*"))
    assert fixture.eligible.read_bytes() == source_bytes
    assert fixture.eligible.stat().st_mtime_ns == source_mtime


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
    assert result.metadata.split_id in (fixture.reports / "split_summary.md").read_text()
    assert set(result.metadata.output_hashes) == {"train", "validation", "test", "rules_snapshot"}
    assert not list(fixture.output.glob(".*.tmp-*"))
    assert not list(fixture.reports.parent.glob(f".{fixture.reports.name}.tmp-*"))
    assert not list(fixture.reports.parent.glob(f".{fixture.reports.name}.backup-*"))


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
    assert not list(failing.reports.parent.glob(f".{failing.reports.name}.tmp-*"))


def test_late_source_recheck_failure_preserves_reports_pointer_and_source(
    tmp_path: Path, eligible_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    previous_reports = _directory_bytes(fixture.reports)
    previous_latest = fixture.output.joinpath("latest.json").read_bytes()
    previous_runs = set(fixture.output.glob("split_id=*"))
    source_bytes = fixture.eligible.read_bytes()
    source_mtime = fixture.eligible.stat().st_mtime_ns
    original_sha256 = pipeline_module.sha256_file
    eligible_hash_calls = 0

    def fail_final_source_hash(path: Path) -> str:
        nonlocal eligible_hash_calls
        digest = original_sha256(path)
        if Path(path) == fixture.eligible:
            eligible_hash_calls += 1
            if eligible_hash_calls == 2:
                return "forced-changed-source-hash"
        return digest

    monkeypatch.setattr(pipeline_module, "sha256_file", fail_final_source_hash)
    with pytest.raises(RuntimeError, match="source changed"):
        run_time_based_split(
            config_path=fixture.config,
            run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    _assert_publication_rolled_back(
        fixture=fixture, previous_reports=previous_reports,
        previous_latest=previous_latest, previous_runs=previous_runs,
        source_bytes=source_bytes, source_mtime=source_mtime,
    )


def test_split_publication_failure_preserves_previous_publication(
    tmp_path: Path, eligible_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    previous_reports = _directory_bytes(fixture.reports)
    previous_latest = fixture.output.joinpath("latest.json").read_bytes()
    previous_runs = set(fixture.output.glob("split_id=*"))
    source_bytes = fixture.eligible.read_bytes()
    source_mtime = fixture.eligible.stat().st_mtime_ns

    def fail_split_publication(*args, **kwargs) -> None:
        raise OSError("forced split publication failure")

    monkeypatch.setattr(pipeline_module, "publish_split_run", fail_split_publication)
    with pytest.raises(OSError, match="split publication"):
        run_time_based_split(
            config_path=fixture.config,
            run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    _assert_publication_rolled_back(
        fixture=fixture, previous_reports=previous_reports,
        previous_latest=previous_latest, previous_runs=previous_runs,
        source_bytes=source_bytes, source_mtime=source_mtime,
    )


def test_report_publication_failure_removes_new_split_and_restores_reports(
    tmp_path: Path, eligible_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    previous_reports = _directory_bytes(fixture.reports)
    previous_latest = fixture.output.joinpath("latest.json").read_bytes()
    previous_runs = set(fixture.output.glob("split_id=*"))
    source_bytes = fixture.eligible.read_bytes()
    source_mtime = fixture.eligible.stat().st_mtime_ns
    original_replace = Path.replace

    def fail_staged_report_replace(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if (
            self.parent == fixture.reports.parent
            and self.name.startswith(f".{fixture.reports.name}.tmp-")
            and target_path == fixture.reports
        ):
            raise OSError("forced report publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_staged_report_replace)
    with pytest.raises(OSError, match="report publication"):
        run_time_based_split(
            config_path=fixture.config,
            run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    _assert_publication_rolled_back(
        fixture=fixture, previous_reports=previous_reports,
        previous_latest=previous_latest, previous_runs=previous_runs,
        source_bytes=source_bytes, source_mtime=source_mtime,
    )


def test_latest_pointer_failure_rolls_back_split_and_reports(
    tmp_path: Path, eligible_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    run_time_based_split(
        config_path=fixture.config,
        run_started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    previous_reports = _directory_bytes(fixture.reports)
    previous_latest = fixture.output.joinpath("latest.json").read_bytes()
    previous_runs = set(fixture.output.glob("split_id=*"))
    source_bytes = fixture.eligible.read_bytes()
    source_mtime = fixture.eligible.stat().st_mtime_ns
    original_update = pipeline_module.update_split_latest

    def fail_after_latest_update(*args, **kwargs) -> None:
        original_update(*args, **kwargs)
        raise OSError("forced latest pointer failure")

    monkeypatch.setattr(pipeline_module, "update_split_latest", fail_after_latest_update)
    with pytest.raises(OSError, match="latest pointer"):
        run_time_based_split(
            config_path=fixture.config,
            run_started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    _assert_publication_rolled_back(
        fixture=fixture, previous_reports=previous_reports,
        previous_latest=previous_latest, previous_runs=previous_runs,
        source_bytes=source_bytes, source_mtime=source_mtime,
    )


def test_cli_success_and_failure_codes(tmp_path: Path, eligible_frame) -> None:
    fixture = build_split_fixture(tmp_path, eligible_frame)
    assert main(["--config", str(fixture.config), "--dry-run"]) == 0
    assert main(["--config", str(tmp_path / "missing.yaml"), "--dry-run"]) == 1
