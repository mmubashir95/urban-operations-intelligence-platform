"""Tests for processed Parquet persistence, immutability, and latest pointers."""

from pathlib import Path

import pandas as pd
import pytest

from urban_ops.cleaning.metadata import write_cleaning_metadata
from urban_ops.cleaning.models import CleaningFrames
from urban_ops.cleaning.outputs import (
    ProcessedOutputExistsError, create_temporary_run, output_paths,
    prepare_run_directory, remove_temporary_run, sha256_file,
    update_latest_pointer, write_and_validate_parquets,
)
from tests.unit.cleaning.test_metadata import metadata


def frames() -> CleaningFrames:
    """Return reconciled, writable output frames."""
    cleaned = pd.DataFrame({
        "unique_key": ["1", "2"], "target_eligible": [True, False],
        "missed_resolution_target": pd.Series([1, pd.NA], dtype="Int8"),
        "primary_exclusion_reason": ["eligible", "missing_due_date"],
    })
    return CleaningFrames(
        cleaned=cleaned, eligible=cleaned.iloc[[0]].copy(),
        excluded=cleaned.iloc[[1]].copy(),
    )


def test_three_parquets_are_written_readable_and_hashed(tmp_path: Path) -> None:
    hashes = write_and_validate_parquets(tmp_path, frames())
    assert set(hashes) == {"cleaned", "eligible", "excluded"}
    for name in hashes:
        path = tmp_path / f"{name}_service_requests.parquet"
        assert pd.read_parquet(path).shape[0] in {1, 2}
        assert sha256_file(path) == hashes[name]


def test_output_paths_have_stable_contract(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run")
    assert paths.cleaned.name == "cleaned_service_requests.parquet"
    assert paths.rules_snapshot.name == "cleaning_rules_snapshot.yaml"
    assert paths.latest_pointer == tmp_path / "latest.json"


def test_successful_run_is_immutable(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run")
    paths.run_directory.mkdir(parents=True)
    write_cleaning_metadata(metadata(), paths.metadata)
    with pytest.raises(ProcessedOutputExistsError, match="immutable"):
        prepare_run_directory(paths, overwrite_incomplete=True)


def test_incomplete_run_requires_explicit_overwrite(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run")
    paths.run_directory.mkdir(parents=True)
    (paths.run_directory / "partial").write_text("x")
    with pytest.raises(ProcessedOutputExistsError, match="overwrite-incomplete"):
        prepare_run_directory(paths, overwrite_incomplete=False)
    prepare_run_directory(paths, overwrite_incomplete=True)
    assert not paths.run_directory.exists()


def test_temporary_run_is_sibling_and_removable(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run")
    temporary = create_temporary_run(paths)
    assert temporary.parent == paths.run_directory.parent
    remove_temporary_run(temporary)
    assert not temporary.exists()


def test_latest_pointer_updates_only_when_called(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "run")
    paths.run_directory.mkdir(parents=True)
    assert not paths.latest_pointer.exists()
    update_latest_pointer(paths, updated_utc="2026-01-01T00:00:00+00:00")
    assert '"run_id": "run"' in paths.latest_pointer.read_text()
