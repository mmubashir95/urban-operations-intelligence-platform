"""CLI and orchestration for immutable, deterministic chronological dataset splits."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from urban_ops.cleaning.outputs import project_path, sha256_file
from urban_ops.data.selected_scope import load_selected_scope_authority
from urban_ops.splitting.assignment import assign_time_splits
from urban_ops.splitting.boundaries import (
    boundaries_from_mapping, evaluate_candidates, selected_boundaries,
    validate_boundaries,
)
from urban_ops.splitting.integrity import build_integrity_checks, require_integrity
from urban_ops.splitting.metadata import (
    SplitMetadata, read_split_metadata, write_split_metadata,
)
from urban_ops.splitting.models import (
    CandidateSplitResult, SplitBoundaries, SplitRunResult,
)
from urban_ops.splitting.outputs import (
    create_temporary_split, prepare_split_directory, remove_temporary_split,
    split_output_paths, update_split_latest, write_and_validate_split_parquets,
)
from urban_ops.splitting.reports import build_report_tables, write_split_reports
from urban_ops.splitting.source import load_verified_source, resolve_cleaning_run
from urban_ops.splitting.temporal_profile import build_monthly_profile
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)


class SplitConfigurationError(ValueError):
    """Raised when time-split configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class SplitConfig:
    """Validated settings needed for candidate analysis and final assignment."""

    processed_root: Path
    latest_pointer: Path
    dataset_name: str
    required_completion_status: str
    scope_authority_file: Path
    scope_extraction_metadata_file: Path
    timestamp_column: str
    identifier_column: str
    target_column: str
    interval_convention: str
    boundaries: SplitBoundaries
    candidates: tuple[tuple[str, SplitBoundaries], ...]
    selected_candidate_id: str
    minimum_rows: int
    minimum_class_rows: int
    maximum_target_rate_difference: float | None
    drift_is_warning_only: bool
    output_root: Path
    report_root: Path
    output_format: str
    schema_version: str


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require one configuration section to be a mapping."""
    if not isinstance(value, dict):
        raise SplitConfigurationError(f"Configuration section {name!r} is required.")
    return value


def _path(value: object, name: str) -> Path:
    """Resolve one configured path relative to the repository root."""
    if not isinstance(value, str) or not value.strip():
        raise SplitConfigurationError(f"Configuration path {name!r} is required.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_split_config(path: Path | str) -> SplitConfig:
    """Parse and fail-closed validate the complete Step 8 configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Split configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = _mapping(payload, "root")
        source = _mapping(root.get("input"), "input")
        split = _mapping(root.get("split"), "split")
        boundaries_payload = _mapping(root.get("boundaries"), "boundaries")
        minimums = _mapping(root.get("minimums"), "minimums")
        analysis = _mapping(root.get("candidate_analysis"), "candidate_analysis")
        output = _mapping(root.get("output"), "output")
        integrity = _mapping(root.get("integrity"), "integrity")
        interval = str(split["interval_convention"])
        candidates_payload = root.get("candidates")
        if not isinstance(candidates_payload, list):
            raise SplitConfigurationError("Configuration candidates must be a list.")
        candidates = tuple(
            (
                str(_mapping(item, "candidate")["candidate_id"]),
                boundaries_from_mapping(_mapping(item, "candidate"), interval_convention=interval),
            )
            for item in candidates_payload
        )
        maximum_drift = analysis.get("maximum_target_rate_difference")
        config = SplitConfig(
            processed_root=_path(source["processed_root"], "input.processed_root"),
            latest_pointer=_path(source["latest_pointer"], "input.latest_pointer"),
            dataset_name=str(source["dataset"]),
            required_completion_status=str(source["required_completion_status"]),
            scope_authority_file=_path(source["scope_authority_file"], "input.scope_authority_file"),
            scope_extraction_metadata_file=_path(source["scope_extraction_metadata_file"], "input.scope_extraction_metadata_file"),
            timestamp_column=str(split["timestamp_column"]),
            identifier_column=str(split["identifier_column"]),
            target_column=str(split["target_column"]),
            interval_convention=interval,
            boundaries=boundaries_from_mapping(boundaries_payload, interval_convention=interval),
            candidates=candidates,
            selected_candidate_id=str(root["selected_candidate"]),
            minimum_rows=int(minimums["rows_per_split"]),
            minimum_class_rows=int(minimums["rows_per_class_per_split"]),
            maximum_target_rate_difference=(
                None if maximum_drift is None else float(maximum_drift)
            ),
            drift_is_warning_only=bool(analysis["drift_is_warning_only"]),
            output_root=_path(output["root"], "output.root"),
            report_root=_path(output["report_root"], "output.report_root"),
            output_format=str(output["format"]).casefold(),
            schema_version=str(output["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        if isinstance(error, SplitConfigurationError):
            raise
        raise SplitConfigurationError(f"Invalid split configuration: {error}") from error
    if config.output_format != "parquet":
        raise SplitConfigurationError("Step 8 supports Parquet split outputs only.")
    if config.timestamp_column != "created_date":
        raise SplitConfigurationError("created_date is the only approved split timestamp.")
    if config.interval_convention != "left_closed_right_open":
        raise SplitConfigurationError("Step 8 requires left-closed, right-open intervals.")
    if config.minimum_rows < 1 or config.minimum_class_rows < 1:
        raise SplitConfigurationError("Split minimum counts must be positive.")
    if not bool(analysis.get("evaluate_multiple_candidates")) or len(config.candidates) < 3:
        raise SplitConfigurationError("Step 8 requires at least three candidate boundaries.")
    required_integrity = (
        "require_all_rows_assigned", "require_no_date_overlap",
        "require_no_identifier_overlap", "require_unique_identifiers",
        "require_binary_target", "require_non_null_target",
        "require_both_classes_per_split", "require_source_immutability",
    )
    if not all(bool(integrity.get(name)) for name in required_integrity):
        raise SplitConfigurationError("All governed split integrity requirements must be enabled.")
    validate_boundaries(config.boundaries)
    return config


def _config_hash(path: Path) -> str:
    """Return the SHA-256 of the exact split rule bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    """Return versions for packages materially involved in split persistence."""
    result: dict[str, str] = {}
    for distribution in ("pandas", "pyarrow", "PyYAML"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return result


def _month_count(frame: pd.DataFrame, timestamp_column: str) -> int:
    """Count distinct UTC calendar months in one non-empty split."""
    return int(frame[timestamp_column].dt.tz_localize(None).dt.to_period("M").nunique())


def _iso(value: pd.Timestamp) -> str:
    """Return a timezone-aware ISO timestamp."""
    return value.isoformat()


def _same_boundaries(left: SplitBoundaries, right: SplitBoundaries) -> bool:
    """Return whether two complete boundary contracts match exactly."""
    return left == right


def _build_metadata(
    *, split_id: str, created_at: datetime, config: SplitConfig, config_hash: str,
    source: Any, boundaries: SplitBoundaries, frames: Any, paths: Any,
    output_hashes: dict[str, str],
) -> SplitMetadata:
    """Construct reconciled metadata from source, assignment, and persisted hashes."""
    stats: dict[str, dict[str, Any]] = {}
    for name in ("train", "validation", "test"):
        frame = getattr(frames, name)
        target = frame[config.target_column].astype(int)
        stats[name] = {
            "rows": len(frame), "on_time": int(target.eq(0).sum()),
            "missed": int(target.eq(1).sum()), "rate": float(target.mean()),
            "min": _iso(frame[config.timestamp_column].min()),
            "max": _iso(frame[config.timestamp_column].max()),
            "months": _month_count(frame, config.timestamp_column),
        }
    ids = {
        name: set(getattr(frames, name)[config.identifier_column])
        for name in ("train", "validation", "test")
    }
    return SplitMetadata(
        schema_version=config.schema_version, split_id=split_id,
        completion_status="success", created_at_utc=created_at.isoformat(),
        source_cleaning_run_id=source.metadata.cleaning_run_id,
        source_cleaning_metadata_path=project_path(source.metadata_path),
        source_eligible_dataset_path=project_path(source.eligible_path),
        source_eligible_sha256=source.eligible_sha256,
        source_eligible_mtime=source.eligible_mtime_ns,
        source_raw_run_id=source.metadata.source_raw_run_id,
        source_raw_sha256=source.metadata.source_raw_sha256,
        scope_agency=source.scope.agency,
        scope_complaint_type=source.scope.complaint_type,
        scope_start_date=source.scope.start_date.date().isoformat(),
        scope_end_date=source.scope.end_date.date().isoformat(),
        timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column,
        target_column=config.target_column,
        interval_convention=config.interval_convention,
        train_start_inclusive=_iso(boundaries.train_start),
        train_end_exclusive=_iso(boundaries.train_end_exclusive),
        validation_start_inclusive=_iso(boundaries.validation_start),
        validation_end_exclusive=_iso(boundaries.validation_end_exclusive),
        test_start_inclusive=_iso(boundaries.test_start),
        test_end_exclusive=_iso(boundaries.test_end_exclusive),
        input_row_count=len(source.frame),
        train_row_count=stats["train"]["rows"],
        validation_row_count=stats["validation"]["rows"],
        test_row_count=stats["test"]["rows"],
        train_on_time_count=stats["train"]["on_time"],
        train_missed_count=stats["train"]["missed"],
        validation_on_time_count=stats["validation"]["on_time"],
        validation_missed_count=stats["validation"]["missed"],
        test_on_time_count=stats["test"]["on_time"],
        test_missed_count=stats["test"]["missed"],
        train_missed_target_rate=stats["train"]["rate"],
        validation_missed_target_rate=stats["validation"]["rate"],
        test_missed_target_rate=stats["test"]["rate"],
        train_min_created_date=stats["train"]["min"],
        train_max_created_date=stats["train"]["max"],
        validation_min_created_date=stats["validation"]["min"],
        validation_max_created_date=stats["validation"]["max"],
        test_min_created_date=stats["test"]["min"],
        test_max_created_date=stats["test"]["max"],
        train_month_count=stats["train"]["months"],
        validation_month_count=stats["validation"]["months"],
        test_month_count=stats["test"]["months"],
        unassigned_row_count=frames.unassigned_row_count,
        multiply_assigned_row_count=frames.multiply_assigned_row_count,
        train_validation_identifier_overlap=len(ids["train"] & ids["validation"]),
        train_test_identifier_overlap=len(ids["train"] & ids["test"]),
        validation_test_identifier_overlap=len(ids["validation"] & ids["test"]),
        config_hash=config_hash,
        output_paths={
            "train": project_path(paths.train), "validation": project_path(paths.validation),
            "test": project_path(paths.test), "metadata": project_path(paths.metadata),
            "rules_snapshot": project_path(paths.rules_snapshot),
        },
        output_hashes=output_hashes,
        warnings=[
            "Target prevalence varies over time; drift is evidence, not an automatic rejection.",
            "Test data is reserved for one final evaluation after development decisions.",
            "Split outputs are governed analytical partitions, not model feature matrices.",
        ],
        python_version=SplitMetadata.current_python_version(),
        package_versions=_package_versions(),
    )


def run_time_based_split(
    *,
    config_path: Path | str,
    cleaning_run_path: Path | None = None,
    output_root: Path | None = None,
    report_root: Path | None = None,
    dry_run: bool = False,
    overwrite_incomplete: bool = False,
    run_started_at: datetime | None = None,
) -> SplitRunResult:
    """Verify Step 7, evaluate candidates, assign rows, and optionally persist a split."""
    config_file = Path(config_path)
    config = load_split_config(config_file)
    scope = load_selected_scope_authority(
        config.scope_authority_file, config.scope_extraction_metadata_file
    )
    run_path = resolve_cleaning_run(
        processed_root=config.processed_root, latest_pointer=config.latest_pointer,
        override=cleaning_run_path,
    )
    source = load_verified_source(
        run_path=run_path, dataset_name=config.dataset_name,
        required_completion_status=config.required_completion_status, scope=scope,
        timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column, target_column=config.target_column,
    )
    candidates = evaluate_candidates(
        source.frame, candidates=config.candidates,
        selected_candidate_id=config.selected_candidate_id,
        timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column, target_column=config.target_column,
        minimum_rows=config.minimum_rows, minimum_class_rows=config.minimum_class_rows,
        maximum_target_rate_difference=config.maximum_target_rate_difference,
        drift_is_warning_only=config.drift_is_warning_only,
    )
    boundaries = selected_boundaries(candidates, config.selected_candidate_id)
    if not _same_boundaries(boundaries, config.boundaries):
        raise SplitConfigurationError(
            "Configured final boundaries differ from the selected candidate."
        )
    frames = assign_time_splits(
        source.frame, boundaries=boundaries,
        timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column,
    )
    monthly = build_monthly_profile(
        source.frame, timestamp_column=config.timestamp_column,
        target_column=config.target_column,
    )
    source_hash_after = sha256_file(source.eligible_path)
    source_mtime_after = source.eligible_path.stat().st_mtime_ns
    checks = build_integrity_checks(
        source.frame, frames, boundaries=boundaries,
        timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column, target_column=config.target_column,
        minimum_rows=config.minimum_rows, minimum_class_rows=config.minimum_class_rows,
        source_hash_before=source.eligible_sha256, source_hash_after=source_hash_after,
        source_mtime_before=source.eligible_mtime_ns, source_mtime_after=source_mtime_after,
    )
    require_integrity(checks)
    tables = build_report_tables(
        source=source.frame, frames=frames, boundaries=boundaries,
        candidates=candidates, selected_candidate_id=config.selected_candidate_id,
        monthly=monthly, checks=checks, timestamp_column=config.timestamp_column,
        identifier_column=config.identifier_column, target_column=config.target_column,
    )
    started = run_started_at or datetime.now(timezone.utc)
    started = started.replace(tzinfo=timezone.utc) if started.tzinfo is None else started.astimezone(timezone.utc)
    config_hash = _config_hash(config_file)
    split_id = f"{started:%Y%m%dT%H%M%SZ}_{config_hash[:16]}"
    paths = split_output_paths(output_root or config.output_root, split_id)
    selected = next(item for item in candidates if item.recommended)
    if dry_run:
        LOGGER.info(
            "Split dry run split_id=%s candidate=%s train=%s (%.2f%%) validation=%s (%.2f%%) test=%s (%.2f%%) max_rate_difference=%.2f%% proposed_output=%s",
            split_id, selected.candidate_id, len(frames.train),
            float(frames.train[config.target_column].mean()) * 100,
            len(frames.validation), float(frames.validation[config.target_column].mean()) * 100,
            len(frames.test), float(frames.test[config.target_column].mean()) * 100,
            selected.metrics["maximum_rate_difference"] * 100, paths.run_directory,
        )
        return SplitRunResult(
            True, source, boundaries, candidates, frames, checks, tables, paths,
            source_hash_after, source_mtime_after, None,
        )

    prepare_split_directory(paths, overwrite_incomplete=overwrite_incomplete)
    temporary_run = create_temporary_split(paths)
    metadata: SplitMetadata | None = None
    try:
        output_hashes = write_and_validate_split_parquets(
            temporary_run, frames, source_columns=list(source.frame.columns)
        )
        snapshot = temporary_run / "split_rules_snapshot.yaml"
        snapshot.write_bytes(config_file.read_bytes())
        output_hashes["rules_snapshot"] = sha256_file(snapshot)
        completed = datetime.now(timezone.utc)
        metadata = _build_metadata(
            split_id=split_id, created_at=completed, config=config,
            config_hash=config_hash, source=source, boundaries=boundaries,
            frames=frames, paths=paths, output_hashes=output_hashes,
        )
        temporary_metadata = temporary_run / "split_metadata.json"
        write_split_metadata(metadata, temporary_metadata)
        read_split_metadata(temporary_metadata)
        write_split_reports(
            report_root=report_root or config.report_root, tables=tables,
            metadata=metadata, selected_candidate_id=config.selected_candidate_id,
        )
        if (
            sha256_file(source.eligible_path) != source.eligible_sha256
            or source.eligible_path.stat().st_mtime_ns != source.eligible_mtime_ns
        ):
            raise RuntimeError("Step 7 eligible source changed before split finalization.")
        temporary_run.replace(paths.run_directory)
        try:
            update_split_latest(paths, updated_utc=completed.isoformat())
        except Exception:
            remove_temporary_split(paths.run_directory)
            raise
    except Exception:
        remove_temporary_split(temporary_run)
        LOGGER.exception("Time-based split failed; temporary split output was removed.")
        raise
    final_hash = sha256_file(source.eligible_path)
    final_mtime = source.eligible_path.stat().st_mtime_ns
    LOGGER.info(
        "Split complete split_id=%s train=%s validation=%s test=%s output=%s",
        split_id, len(frames.train), len(frames.validation), len(frames.test),
        paths.run_directory,
    )
    return SplitRunResult(
        False, source, boundaries, candidates, frames, checks, tables, paths,
        final_hash, final_mtime, metadata,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the documented Step 8 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cleaning-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split CLI and return non-zero for configuration or integrity failure."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        run_time_based_split(
            config_path=args.config, cleaning_run_path=args.cleaning_run,
            output_root=args.output_root, report_root=args.report_root,
            dry_run=args.dry_run, overwrite_incomplete=args.overwrite_incomplete,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Time-based split command failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
