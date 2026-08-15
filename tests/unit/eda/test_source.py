"""Tests for Step 8 authority resolution and fail-closed EDA source loading."""

import json
from pathlib import Path

import pandas as pd
import pytest

from urban_ops.cleaning.outputs import sha256_file
from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import EDASourceError, load_verified_split, resolve_split_run
from tests.unit.eda.conftest import EDAFixture, build_eda_fixture


def _load(fixture: EDAFixture):
    config = load_eda_config(fixture.config)
    run = resolve_split_run(split_root=config.split_root, latest_pointer=config.latest_pointer)
    return load_verified_split(
        run_path=run, latest_pointer=config.latest_pointer,
        required_completion_status="success", identifier_column="unique_key",
        target_column="missed_resolution_target", timestamp_column="created_date",
    )


def test_latest_and_explicit_override_resolve_same_successful_split(eda_fixture) -> None:
    config = load_eda_config(eda_fixture.config)
    latest = resolve_split_run(split_root=config.split_root, latest_pointer=config.latest_pointer)
    override = resolve_split_run(
        split_root=config.split_root, latest_pointer=config.latest_pointer,
        override=eda_fixture.split_run,
    )
    assert latest == override == eda_fixture.split_run.resolve()


def test_missing_pointer_run_and_required_files_fail(tmp_path: Path) -> None:
    with pytest.raises(EDASourceError, match="latest pointer"):
        resolve_split_run(split_root=tmp_path, latest_pointer=tmp_path / "missing.json")
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"run_path": "missing", "split_id": "missing"}))
    with pytest.raises(EDASourceError, match="does not exist"):
        resolve_split_run(split_root=tmp_path, latest_pointer=pointer)
    for filename in ("split_metadata.json", "train.parquet", "validation.parquet", "test.parquet"):
        fixture = build_eda_fixture(tmp_path / filename.replace(".", "_"))
        (fixture.split_run / filename).unlink()
        with pytest.raises(EDASourceError, match="missing"):
            _load(fixture)


def test_hash_row_schema_and_split_id_mismatches_fail(tmp_path: Path) -> None:
    hash_fixture = build_eda_fixture(tmp_path / "hash")
    with (hash_fixture.split_run / "train.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(EDASourceError, match="hash"):
        _load(hash_fixture)

    id_fixture = build_eda_fixture(tmp_path / "id")
    metadata_path = id_fixture.split_run / "split_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["split_id"] = "different"
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(EDASourceError, match="split ID"):
        _load(id_fixture)

    schema_fixture = build_eda_fixture(tmp_path / "schema")
    validation_path = schema_fixture.split_run / "validation.parquet"
    validation = pd.read_parquet(validation_path).drop(columns="borough")
    validation.to_parquet(validation_path, index=False)
    metadata_path = schema_fixture.split_run / "split_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["output_hashes"]["validation"] = sha256_file(validation_path)
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(EDASourceError, match="schemas differ"):
        _load(schema_fixture)


def test_loading_does_not_modify_source_frames_or_artifacts(eda_fixture) -> None:
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in eda_fixture.split_run.iterdir() if path.is_file()
    }
    source = _load(eda_fixture)
    assert len(source.train) + len(source.validation) + len(source.test) == 18
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in eda_fixture.split_run.iterdir() if path.is_file()
    }
    assert before == after


def test_null_target_and_cross_split_identifier_overlap_fail(tmp_path: Path) -> None:
    null_fixture = build_eda_fixture(tmp_path / "null_target")
    train_path = null_fixture.split_run / "train.parquet"
    train = pd.read_parquet(train_path)
    train.loc[train.index[0], "missed_resolution_target"] = pd.NA
    train.to_parquet(train_path, index=False)
    metadata_path = null_fixture.split_run / "split_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["output_hashes"]["train"] = sha256_file(train_path)
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(EDASourceError, match="target"):
        _load(null_fixture)

    overlap_fixture = build_eda_fixture(tmp_path / "overlap")
    train = pd.read_parquet(overlap_fixture.split_run / "train.parquet")
    validation_path = overlap_fixture.split_run / "validation.parquet"
    validation = pd.read_parquet(validation_path)
    validation.loc[validation.index[0], "unique_key"] = train.iloc[0]["unique_key"]
    validation.to_parquet(validation_path, index=False)
    metadata_path = overlap_fixture.split_run / "split_metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["output_hashes"]["validation"] = sha256_file(validation_path)
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(EDASourceError, match="overlap"):
        _load(overlap_fixture)
