"""Verify and load the governed cleaned source used by duration workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_ops.cleaning.metadata import read_cleaning_metadata
from urban_ops.cleaning.outputs import sha256_file
from urban_ops.utils.paths import PROJECT_ROOT


CLEANED_DATASET_NAME = "cleaned_service_requests.parquet"


class DurationAnalysisSourceError(RuntimeError):
    """Raised when the governed cleaned source cannot be verified."""


def load_verified_cleaned_source(run_path: Path) -> tuple[pd.DataFrame, Path]:
    """Load the full cleaned artifact after metadata, hash, and count checks."""
    metadata_path = run_path / "cleaning_metadata.json"
    cleaned_path = run_path / CLEANED_DATASET_NAME
    if not metadata_path.is_file() or not cleaned_path.is_file():
        raise DurationAnalysisSourceError(
            f"Cleaning run lacks metadata or {CLEANED_DATASET_NAME}: {run_path}"
        )
    try:
        metadata = read_cleaning_metadata(metadata_path)
    except ValueError as error:
        raise DurationAnalysisSourceError(str(error)) from error
    referenced = Path(metadata.output_paths.get("cleaned", ""))
    referenced = referenced if referenced.is_absolute() else PROJECT_ROOT / referenced
    if not referenced.exists() or referenced.resolve() != cleaned_path.resolve():
        raise DurationAnalysisSourceError(
            "Cleaning metadata does not reference the selected cleaned dataset."
        )
    if metadata.output_hashes.get("cleaned") != sha256_file(cleaned_path):
        raise DurationAnalysisSourceError("Cleaned dataset SHA-256 does not match metadata.")
    frame = pd.read_parquet(cleaned_path, engine="pyarrow")
    if len(frame) != metadata.cleaned_row_count:
        raise DurationAnalysisSourceError(
            "Cleaned dataset row count does not match cleaning metadata."
        )
    return frame, cleaned_path
