"""Atomic and immutable filesystem operations for processed cleaning outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from urban_ops.cleaning.metadata import read_cleaning_metadata
from urban_ops.cleaning.models import CleaningFrames, ProcessedOutputPaths
from urban_ops.utils.paths import PROJECT_ROOT


class ProcessedOutputExistsError(FileExistsError):
    """Raised when a cleaning run would replace immutable successful output."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def output_paths(output_root: Path, run_id: str) -> ProcessedOutputPaths:
    """Return canonical final paths for one processed run."""
    run = output_root / f"run_id={run_id}"
    return ProcessedOutputPaths(
        run_directory=run,
        cleaned=run / "cleaned_service_requests.parquet",
        eligible=run / "eligible_service_requests.parquet",
        excluded=run / "excluded_service_requests.parquet",
        metadata=run / "cleaning_metadata.json",
        rules_snapshot=run / "cleaning_rules_snapshot.yaml",
        latest_pointer=output_root / "latest.json",
    )


def prepare_run_directory(paths: ProcessedOutputPaths, *, overwrite_incomplete: bool) -> None:
    """Protect successful runs and optionally remove only incomplete collisions."""
    if not paths.run_directory.exists():
        return
    successful = False
    if paths.metadata.is_file():
        try:
            successful = read_cleaning_metadata(paths.metadata).completion_status == "success"
        except ValueError:
            successful = False
    if successful:
        raise ProcessedOutputExistsError(
            f"Successful processed run is immutable: {paths.run_directory}"
        )
    if not overwrite_incomplete:
        raise ProcessedOutputExistsError(
            "Incomplete processed output exists; pass --overwrite-incomplete to replace it: "
            f"{paths.run_directory}"
        )
    shutil.rmtree(paths.run_directory)


def create_temporary_run(paths: ProcessedOutputPaths) -> Path:
    """Create a temporary sibling directory suitable for atomic rename."""
    paths.run_directory.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f".{paths.run_directory.name}.tmp-", dir=paths.run_directory.parent
    ))


def write_and_validate_parquets(
    temporary_run: Path, frames: CleaningFrames
) -> dict[str, str]:
    """Write three Parquet files, read them back, and return their hashes."""
    files = {
        "cleaned": temporary_run / "cleaned_service_requests.parquet",
        "eligible": temporary_run / "eligible_service_requests.parquet",
        "excluded": temporary_run / "excluded_service_requests.parquet",
    }
    expected = {
        "cleaned": frames.cleaned,
        "eligible": frames.eligible,
        "excluded": frames.excluded,
    }
    for name, path in files.items():
        expected[name].to_parquet(path, index=False, engine="pyarrow")
        read_back = pd.read_parquet(path, engine="pyarrow")
        if len(read_back) != len(expected[name]):
            raise RuntimeError(f"Processed {name} Parquet row-count validation failed.")
        if list(read_back.columns) != list(expected[name].columns):
            raise RuntimeError(f"Processed {name} Parquet schema validation failed.")
    return {name: sha256_file(path) for name, path in files.items()}


def update_latest_pointer(paths: ProcessedOutputPaths, *, updated_utc: str) -> None:
    """Atomically point consumers to the newly finalized successful run."""
    payload = {
        "run_id": paths.run_directory.name.removeprefix("run_id="),
        "run_path": project_path(paths.run_directory),
        "metadata_path": project_path(paths.metadata),
        "updated_utc": updated_utc,
    }
    temporary = paths.latest_pointer.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(paths.latest_pointer)


def remove_temporary_run(path: Path) -> None:
    """Remove a failed temporary run without touching any finalized output."""
    shutil.rmtree(path, ignore_errors=True)
