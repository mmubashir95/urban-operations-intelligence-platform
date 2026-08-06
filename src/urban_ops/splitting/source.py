"""Locate and fail-closed verify the eligible output of a Step 7 cleaning run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from urban_ops.cleaning.metadata import CleaningMetadata, read_cleaning_metadata
from urban_ops.cleaning.outputs import sha256_file
from urban_ops.data.selected_scope import SelectedScope
from urban_ops.splitting.models import SourceEvidence
from urban_ops.utils.paths import PROJECT_ROOT


class SplitSourceError(RuntimeError):
    """Raised when Step 7 provenance or eligible data is unsafe to split."""


def resolve_cleaning_run(
    *, processed_root: Path, latest_pointer: Path, override: Path | None = None
) -> Path:
    """Resolve an explicit run or the run referenced by the Step 7 latest pointer."""
    if override is not None:
        run_path = override
    else:
        if not latest_pointer.is_file():
            raise SplitSourceError(f"Step 7 latest pointer is missing: {latest_pointer}")
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            referenced = payload["run_path"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise SplitSourceError("Step 7 latest pointer is invalid.") from error
        candidate = Path(str(referenced))
        run_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if not run_path.exists():
            fallback = processed_root / f"run_id={payload.get('run_id', '')}"
            run_path = fallback
    if not run_path.is_dir():
        raise SplitSourceError(f"Step 7 cleaning run does not exist: {run_path}")
    return run_path.resolve()


def validate_eligible_frame(
    frame: pd.DataFrame,
    *,
    scope: SelectedScope,
    timestamp_column: str,
    identifier_column: str,
    target_column: str,
) -> None:
    """Validate the eligible dataset contract without mutating its frame."""
    if frame.empty:
        raise SplitSourceError("Eligible Step 7 dataset is empty.")
    required = {timestamp_column, identifier_column, target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SplitSourceError(f"Eligible dataset is missing required columns: {missing}")
    if frame[identifier_column].isna().any():
        raise SplitSourceError("Eligible identifiers contain null values.")
    if frame[identifier_column].duplicated().any():
        raise SplitSourceError("Eligible identifiers are not unique.")
    if frame[target_column].isna().any():
        raise SplitSourceError("Eligible targets contain null values.")
    target_values = set(frame[target_column].astype(int).unique())
    if not target_values.issubset({0, 1}):
        raise SplitSourceError(f"Eligible target is not binary: {sorted(target_values)}")
    if "target_eligible" in frame and not frame["target_eligible"].eq(True).all():
        raise SplitSourceError("Eligible dataset contains target-ineligible rows.")
    timestamps = frame[timestamp_column]
    if timestamps.isna().any() or not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise SplitSourceError("created_date must be non-null timezone-aware timestamps.")
    if str(timestamps.dt.tz) != "UTC":
        raise SplitSourceError("created_date must use UTC.")
    end_exclusive = scope.end_date.tz_localize("UTC") + pd.Timedelta(days=1)
    start = scope.start_date.tz_localize("UTC")
    if not timestamps.ge(start).all() or not timestamps.lt(end_exclusive).all():
        raise SplitSourceError("Eligible created_date values violate Step 3 scope.")
    if "agency" in frame and not frame["agency"].eq(scope.agency).all():
        raise SplitSourceError("Eligible agency differs from Step 3 scope.")
    if "complaint_type" in frame and not frame["complaint_type"].eq(scope.complaint_type).all():
        raise SplitSourceError("Eligible complaint type differs from Step 3 scope.")


def load_verified_source(
    *,
    run_path: Path,
    dataset_name: str,
    required_completion_status: str,
    scope: SelectedScope,
    timestamp_column: str,
    identifier_column: str,
    target_column: str,
) -> SourceEvidence:
    """Read and reconcile Step 7 metadata, eligible bytes, shape, scope, and target."""
    metadata_path = run_path / "cleaning_metadata.json"
    eligible_path = run_path / dataset_name
    if not metadata_path.is_file():
        raise SplitSourceError(f"Cleaning metadata is missing: {metadata_path}")
    if not eligible_path.is_file():
        raise SplitSourceError(f"Eligible dataset is missing: {eligible_path}")
    try:
        metadata = read_cleaning_metadata(metadata_path)
    except ValueError as error:
        raise SplitSourceError(str(error)) from error
    if metadata.completion_status != required_completion_status:
        raise SplitSourceError("Step 7 cleaning run is not successful.")
    if metadata.cleaning_run_id != run_path.name.removeprefix("run_id="):
        raise SplitSourceError("Cleaning metadata run ID does not match its directory.")
    referenced_eligible = Path(metadata.output_paths.get("eligible", ""))
    referenced_eligible = (
        referenced_eligible if referenced_eligible.is_absolute()
        else PROJECT_ROOT / referenced_eligible
    )
    if not referenced_eligible.exists() or referenced_eligible.resolve() != eligible_path.resolve():
        raise SplitSourceError("Cleaning metadata does not reference this eligible dataset.")
    eligible_hash = sha256_file(eligible_path)
    if metadata.output_hashes.get("eligible") != eligible_hash:
        raise SplitSourceError("Eligible file SHA-256 does not match cleaning metadata.")
    frame = pd.read_parquet(eligible_path, engine="pyarrow")
    if len(frame) != metadata.eligible_row_count:
        raise SplitSourceError("Eligible row count does not match cleaning metadata.")
    validate_eligible_frame(
        frame, scope=scope, timestamp_column=timestamp_column,
        identifier_column=identifier_column, target_column=target_column,
    )
    return SourceEvidence(
        run_path=run_path, metadata_path=metadata_path, eligible_path=eligible_path,
        eligible_sha256=eligible_hash, eligible_mtime_ns=eligible_path.stat().st_mtime_ns,
        metadata=metadata, scope=scope, frame=frame,
    )
