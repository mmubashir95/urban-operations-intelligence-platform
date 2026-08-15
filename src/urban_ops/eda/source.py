"""Resolve and fail-closed verify immutable Step 8 split inputs for EDA."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from urban_ops.cleaning.outputs import sha256_file
from urban_ops.eda.models import SourceArtifactState, SourceSplitEvidence
from urban_ops.splitting.metadata import read_split_metadata
from urban_ops.utils.paths import PROJECT_ROOT


class EDASourceError(RuntimeError):
    """Raised when Step 8 authority or split content is unsafe for analysis."""


def resolve_split_run(
    *, split_root: Path, latest_pointer: Path, override: Path | None = None
) -> Path:
    """Resolve an explicit split or the run referenced by the latest pointer."""
    if override is not None:
        run_path = override
    else:
        if not latest_pointer.is_file():
            raise EDASourceError(f"Step 8 latest pointer is missing: {latest_pointer}")
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            referenced = payload["run_path"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise EDASourceError("Step 8 latest pointer is invalid.") from error
        candidate = Path(str(referenced))
        run_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if not run_path.exists():
            run_path = split_root / f"split_id={payload.get('split_id', '')}"
        pointer_split_id = str(payload.get("split_id", ""))
        if pointer_split_id and run_path.name != f"split_id={pointer_split_id}":
            raise EDASourceError("Step 8 latest pointer split ID differs from its run path.")
    if not run_path.is_dir():
        raise EDASourceError(f"Step 8 split run does not exist: {run_path}")
    return run_path.resolve()


def _validate_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    identifier_column: str,
    target_column: str,
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Validate one split's identifier, target, timestamp, and date boundary."""
    required = {identifier_column, target_column, timestamp_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EDASourceError(f"{split_name} is missing required columns: {missing}")
    if frame[identifier_column].isna().any() or frame[identifier_column].duplicated().any():
        raise EDASourceError(f"{split_name} identifiers must be non-null and unique.")
    target = frame[target_column]
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise EDASourceError(f"{split_name} target must be non-null and binary.")
    timestamps = frame[timestamp_column]
    if timestamps.isna().any() or not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise EDASourceError(f"{split_name} created_date must be timezone-aware.")
    if str(timestamps.dt.tz) != "UTC":
        raise EDASourceError(f"{split_name} created_date must use UTC.")
    if not timestamps.ge(start).all() or not timestamps.lt(end).all():
        raise EDASourceError(f"{split_name} timestamps violate Step 8 boundaries.")


def _artifact_state(path: Path) -> SourceArtifactState:
    """Capture hash and nanosecond modification time for one required artifact."""
    if not path.is_file():
        raise EDASourceError(f"Required Step 8 artifact is missing: {path}")
    return SourceArtifactState(path, sha256_file(path), path.stat().st_mtime_ns)


def load_verified_split(
    *,
    run_path: Path,
    latest_pointer: Path,
    required_completion_status: str,
    identifier_column: str,
    target_column: str,
    timestamp_column: str,
) -> SourceSplitEvidence:
    """Load and reconcile Step 8 metadata, files, hashes, schemas, IDs, and ranges."""
    metadata_path = run_path / "split_metadata.json"
    rules_path = run_path / "split_rules_snapshot.yaml"
    paths = {name: run_path / f"{name}.parquet" for name in ("train", "validation", "test")}
    artifacts = {
        **{name: _artifact_state(path) for name, path in paths.items()},
        "metadata": _artifact_state(metadata_path),
        "rules_snapshot": _artifact_state(rules_path),
        "latest_pointer": _artifact_state(latest_pointer),
    }
    try:
        metadata = read_split_metadata(metadata_path)
    except ValueError as error:
        raise EDASourceError("Step 8 split metadata is invalid.") from error
    split_id = run_path.name.removeprefix("split_id=")
    if metadata.completion_status != required_completion_status:
        raise EDASourceError("Step 8 split is not successful.")
    if metadata.split_id != split_id:
        raise EDASourceError("Step 8 split ID differs from its directory.")
    for name in ("train", "validation", "test"):
        if artifacts[name].sha256 != metadata.output_hashes.get(name):
            raise EDASourceError(f"{name} hash differs from Step 8 metadata.")
    if artifacts["rules_snapshot"].sha256 != metadata.output_hashes.get("rules_snapshot"):
        raise EDASourceError("Split rules snapshot hash differs from metadata.")

    frames = {name: pd.read_parquet(path, engine="pyarrow") for name, path in paths.items()}
    schemas = {name: tuple(frame.columns) for name, frame in frames.items()}
    if len(set(schemas.values())) != 1:
        raise EDASourceError("Train, validation, and test schemas differ.")
    boundaries = {
        "train": (metadata.train_start_inclusive, metadata.train_end_exclusive),
        "validation": (metadata.validation_start_inclusive, metadata.validation_end_exclusive),
        "test": (metadata.test_start_inclusive, metadata.test_end_exclusive),
    }
    for name, frame in frames.items():
        start, end = boundaries[name]
        _validate_frame(
            frame, split_name=name, identifier_column=identifier_column,
            target_column=target_column, timestamp_column=timestamp_column,
            start=pd.Timestamp(start), end=pd.Timestamp(end),
        )
        expected_rows = int(getattr(metadata, f"{name}_row_count"))
        if len(frame) != expected_rows:
            raise EDASourceError(f"{name} row count differs from Step 8 metadata.")
        expected_missed = int(getattr(metadata, f"{name}_missed_count"))
        if int(frame[target_column].astype(int).sum()) != expected_missed:
            raise EDASourceError(f"{name} target count differs from Step 8 metadata.")
    identifiers = {name: set(frame[identifier_column]) for name, frame in frames.items()}
    if any((
        identifiers["train"] & identifiers["validation"],
        identifiers["train"] & identifiers["test"],
        identifiers["validation"] & identifiers["test"],
    )):
        raise EDASourceError("Complaint identifiers overlap across Step 8 splits.")
    if sum(len(frame) for frame in frames.values()) != metadata.input_row_count:
        raise EDASourceError("Split row counts do not reconcile with metadata.")
    return SourceSplitEvidence(
        run_path=run_path, split_id=split_id, metadata=metadata,
        train=frames["train"], validation=frames["validation"], test=frames["test"],
        artifacts=artifacts,
    )


def verify_source_unchanged(source: SourceSplitEvidence) -> None:
    """Fail when any governed Step 8 artifact hash or mtime changed during EDA."""
    for name, before in source.artifacts.items():
        if not before.path.is_file():
            raise EDASourceError(f"Step 8 artifact disappeared during EDA: {name}")
        if sha256_file(before.path) != before.sha256:
            raise EDASourceError(f"Step 8 artifact hash changed during EDA: {name}")
        if before.path.stat().st_mtime_ns != before.mtime_ns:
            raise EDASourceError(f"Step 8 artifact mtime changed during EDA: {name}")
