"""Tests for split Parquet persistence, immutability, cleanup, and latest pointers."""

from pathlib import Path

import pandas as pd
import pytest

from urban_ops.splitting.assignment import assign_time_splits
from urban_ops.splitting.metadata import write_split_metadata
from urban_ops.splitting.outputs import (
    SplitOutputExistsError, create_temporary_split, prepare_split_directory,
    remove_temporary_split, split_output_paths, update_split_latest,
    write_and_validate_split_parquets,
)
from tests.unit.splitting.test_metadata import metadata


def test_three_parquets_read_back_with_schema_order_and_hashes(
    tmp_path: Path, eligible_frame, boundaries
) -> None:
    frames = assign_time_splits(eligible_frame, boundaries=boundaries, timestamp_column="created_date", identifier_column="unique_key")
    hashes = write_and_validate_split_parquets(tmp_path, frames, source_columns=list(eligible_frame.columns))
    assert set(hashes) == {"train", "validation", "test"}
    for name in hashes:
        frame = pd.read_parquet(tmp_path / f"{name}.parquet")
        assert list(frame.columns) == list(eligible_frame.columns)
        assert frame.equals(frame.sort_values(["created_date", "unique_key"]).reset_index(drop=True))


def test_paths_and_latest_pointer_contract(tmp_path: Path) -> None:
    paths = split_output_paths(tmp_path, "split-fixture")
    assert paths.train.name == "train.parquet"
    assert paths.rules_snapshot.name == "split_rules_snapshot.yaml"
    paths.run_directory.mkdir(parents=True)
    update_split_latest(paths, updated_utc="2026-01-01T00:00:00+00:00")
    assert '"split_id": "split-fixture"' in paths.latest_pointer.read_text()


def test_successful_run_is_immutable(tmp_path: Path) -> None:
    paths = split_output_paths(tmp_path, "split-fixture")
    paths.run_directory.mkdir(parents=True)
    write_split_metadata(metadata(paths=paths), paths.metadata)
    with pytest.raises(SplitOutputExistsError, match="immutable"):
        prepare_split_directory(paths, overwrite_incomplete=True)


def test_incomplete_run_requires_explicit_overwrite(tmp_path: Path) -> None:
    paths = split_output_paths(tmp_path, "split-fixture")
    paths.run_directory.mkdir(parents=True)
    (paths.run_directory / "partial").write_text("x")
    with pytest.raises(SplitOutputExistsError, match="overwrite-incomplete"):
        prepare_split_directory(paths, overwrite_incomplete=False)
    prepare_split_directory(paths, overwrite_incomplete=True)
    assert not paths.run_directory.exists()


def test_temporary_split_is_sibling_and_removable(tmp_path: Path) -> None:
    paths = split_output_paths(tmp_path, "split-fixture")
    temporary = create_temporary_split(paths)
    assert temporary.parent == paths.run_directory.parent
    remove_temporary_split(temporary)
    assert not temporary.exists()
