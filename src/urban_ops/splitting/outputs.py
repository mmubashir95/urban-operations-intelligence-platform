"""Atomic and immutable filesystem operations for chronological split artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from urban_ops.cleaning.outputs import project_path, sha256_file
from urban_ops.splitting.metadata import read_split_metadata
from urban_ops.splitting.models import SplitFrames, SplitRunPaths


class SplitOutputExistsError(FileExistsError):
    """Raised when a split run would overwrite immutable or incomplete output."""


def split_output_paths(output_root: Path, split_id: str) -> SplitRunPaths:
    """Return canonical files for one immutable split run."""
    run = output_root / f"split_id={split_id}"
    return SplitRunPaths(
        run_directory=run, train=run / "train.parquet",
        validation=run / "validation.parquet", test=run / "test.parquet",
        metadata=run / "split_metadata.json",
        rules_snapshot=run / "split_rules_snapshot.yaml",
        latest_pointer=output_root / "latest.json",
    )


def prepare_split_directory(paths: SplitRunPaths, *, overwrite_incomplete: bool) -> None:
    """Protect successful split runs and optionally replace incomplete collisions."""
    if not paths.run_directory.exists():
        return
    successful = False
    if paths.metadata.is_file():
        try:
            successful = read_split_metadata(paths.metadata).completion_status == "success"
        except ValueError:
            successful = False
    if successful:
        raise SplitOutputExistsError(f"Successful split run is immutable: {paths.run_directory}")
    if not overwrite_incomplete:
        raise SplitOutputExistsError(
            "Incomplete split output exists; pass --overwrite-incomplete to replace it: "
            f"{paths.run_directory}"
        )
    shutil.rmtree(paths.run_directory)


def create_temporary_split(paths: SplitRunPaths) -> Path:
    """Create a sibling temporary directory for atomic finalization."""
    paths.run_directory.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f".{paths.run_directory.name}.tmp-", dir=paths.run_directory.parent
    ))


def publish_split_run(temporary_run: Path, paths: SplitRunPaths) -> None:
    """Atomically publish one validated split directory at its immutable path."""
    temporary_run.replace(paths.run_directory)


def write_and_validate_split_parquets(
    temporary_run: Path, frames: SplitFrames, *, source_columns: list[str]
) -> dict[str, str]:
    """Write, read back, and hash train, validation, and test Parquet files."""
    expected = {
        "train": frames.train, "validation": frames.validation, "test": frames.test,
    }
    hashes: dict[str, str] = {}
    for name, frame in expected.items():
        path = temporary_run / f"{name}.parquet"
        if list(frame.columns) != source_columns:
            raise RuntimeError(f"{name} schema differs from the eligible source schema.")
        frame.to_parquet(path, index=False, engine="pyarrow")
        read_back = pd.read_parquet(path, engine="pyarrow")
        if len(read_back) != len(frame) or list(read_back.columns) != source_columns:
            raise RuntimeError(f"{name} Parquet read-back validation failed.")
        if not read_back.equals(frame):
            raise RuntimeError(f"{name} Parquet content or ordering changed on write.")
        hashes[name] = sha256_file(path)
    return hashes


def update_split_latest(paths: SplitRunPaths, *, updated_utc: str) -> None:
    """Atomically update the latest pointer after successful finalization."""
    payload = {
        "split_id": paths.run_directory.name.removeprefix("split_id="),
        "run_path": project_path(paths.run_directory),
        "metadata_path": project_path(paths.metadata),
        "updated_utc": updated_utc,
    }
    temporary = paths.latest_pointer.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(paths.latest_pointer)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def read_split_latest_bytes(paths: SplitRunPaths) -> bytes | None:
    """Capture the current latest pointer for transaction rollback."""
    return paths.latest_pointer.read_bytes() if paths.latest_pointer.is_file() else None


def restore_split_latest(paths: SplitRunPaths, previous_content: bytes | None) -> None:
    """Restore the latest pointer to its exact pre-publication state."""
    update_temporary = paths.latest_pointer.with_suffix(".json.tmp")
    rollback_temporary = paths.latest_pointer.with_suffix(".json.rollback.tmp")
    update_temporary.unlink(missing_ok=True)
    if previous_content is None:
        paths.latest_pointer.unlink(missing_ok=True)
        rollback_temporary.unlink(missing_ok=True)
        return
    try:
        rollback_temporary.write_bytes(previous_content)
        rollback_temporary.replace(paths.latest_pointer)
    except OSError:
        rollback_temporary.unlink(missing_ok=True)
        raise


def remove_temporary_split(path: Path) -> None:
    """Remove a failed temporary run without changing finalized runs."""
    shutil.rmtree(path, ignore_errors=True)
