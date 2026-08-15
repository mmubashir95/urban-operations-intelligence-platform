"""Typed configuration, source, check, and result contracts for Step 9A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from urban_ops.splitting.metadata import SplitMetadata


@dataclass(frozen=True)
class EDAConfig:
    """Validated analysis-only settings for split-aware EDA."""

    split_root: Path
    latest_pointer: Path
    required_completion_status: str
    identifier_column: str
    target_column: str
    timestamp_column: str
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    derived_temporal_features: tuple[str, ...]
    missingness_bands: dict[str, float]
    top_n_categories: int
    minimum_support_for_target_rate: int
    rare_count_candidates: tuple[int, ...]
    rare_share_candidates: tuple[float, ...]
    percentiles: tuple[float, ...]
    iqr_multiplier: float
    latitude_range: tuple[float, float]
    longitude_range: tuple[float, float]
    nyc_bounds: tuple[float, float, float, float]
    report_root: Path


@dataclass(frozen=True)
class SourceArtifactState:
    """Immutable before-run bytes evidence for one governed source artifact."""

    path: Path
    sha256: str
    mtime_ns: int


@dataclass(frozen=True)
class SourceSplitEvidence:
    """Verified Step 8 authority, source frames, and immutable artifact states."""

    run_path: Path
    split_id: str
    metadata: SplitMetadata
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    artifacts: dict[str, SourceArtifactState]


@dataclass(frozen=True)
class EDAIntegrityCheck:
    """One source, governance, reconciliation, or immutability result."""

    check_id: str
    area: str
    status: str
    observed_value: object
    expected_value: object
    affected_rows: int
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable report row."""
        return {
            "check_id": self.check_id,
            "area": self.area,
            "status": self.status,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "affected_rows": self.affected_rows,
            "message": self.message,
        }


@dataclass(frozen=True)
class EDAResult:
    """Complete in-memory outcome and optional published Step 9A report."""

    dry_run: bool
    source: SourceSplitEvidence
    tables: dict[str, pd.DataFrame]
    checks: tuple[EDAIntegrityCheck, ...]
    report_root: Path
    figure_names: tuple[str, ...]
