"""CLI and orchestration for reproducible, immutable NYC 311 data cleaning."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml

from urban_ops.cleaning.categories import clean_categories
from urban_ops.cleaning.duplicates import clean_duplicates
from urban_ops.cleaning.eligibility import apply_governed_target
from urban_ops.cleaning.leakage import build_cleaning_leakage_audit
from urban_ops.cleaning.metadata import (
    CleaningMetadata, read_cleaning_metadata, write_cleaning_metadata,
)
from urban_ops.cleaning.missing_values import apply_missing_value_policy
from urban_ops.cleaning.models import (
    CleaningAction, CleaningCheck, CleaningRunResult, ProcessedOutputPaths,
    ValidationEvidence,
)
from urban_ops.cleaning.outputs import (
    create_temporary_run, output_paths, prepare_run_directory, project_path,
    remove_temporary_run, sha256_file, update_latest_pointer,
    write_and_validate_parquets,
)
from urban_ops.cleaning.reports import (
    build_chronology_actions, build_report_tables, write_cleaning_reports,
)
from urban_ops.cleaning.timestamps import clean_timestamps
from urban_ops.data.ingest import find_latest_successful_run
from urban_ops.data.metadata import ExtractionMetadata, read_metadata
from urban_ops.data.selected_scope import SelectedScope, load_selected_scope_authority
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
RUN_ID_PATTERN = re.compile(r"Raw run ID: `([^`]+)`")
STATUS_PATTERN = re.compile(r"Overall validation status: \*\*([^*]+)\*\*")


class CleaningConfigurationError(ValueError):
    """Raised when cleaning configuration is incomplete or unsafe."""


class CleaningInputError(RuntimeError):
    """Raised when raw input or Step 6 evidence cannot authorize cleaning."""


@dataclass(frozen=True)
class CleaningConfig:
    """Validated settings required for one cleaning run."""

    raw_root: Path
    validation_report_root: Path
    scope_authority_file: Path
    scope_extraction_metadata_file: Path
    require_validation_critical_count: int
    output_root: Path
    report_root: Path
    output_format: str
    schema_version: str
    timestamp_columns: tuple[str, ...]
    timezone: str
    invalid_timestamp_policy: str
    impute_timestamps: bool
    trim_columns: tuple[str, ...]
    blank_to_null_columns: tuple[str, ...]
    collapse_repeated_spaces_columns: tuple[str, ...]
    approved_mappings: dict[str, dict[str, str]]
    duplicate_identifier: str
    exact_duplicate_policy: str
    conflict_duplicate_policy: str
    configured_all_null_columns: tuple[str, ...]
    configured_zero_variance_columns: tuple[str, ...]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require a configuration section to be a mapping."""
    if not isinstance(value, dict):
        raise CleaningConfigurationError(f"Configuration section {name!r} is required.")
    return value


def _path(value: object, name: str) -> Path:
    """Resolve a configured path relative to the project root."""
    if not isinstance(value, str) or not value.strip():
        raise CleaningConfigurationError(f"Configuration path {name!r} is required.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _string_tuple(section: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """Read one required list of string configuration values."""
    value = section.get(key)
    if not isinstance(value, list):
        raise CleaningConfigurationError(f"Configuration value {key!r} must be a list.")
    return tuple(str(item) for item in value)


def load_cleaning_config(path: Path | str) -> CleaningConfig:
    """Load and validate the Step 7 cleaning configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Cleaning configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CleaningConfigurationError(f"Unable to parse {config_path}") from error
    root = _mapping(payload, "root")
    input_config = _mapping(root.get("input"), "input")
    output = _mapping(root.get("output"), "output")
    timestamps = _mapping(root.get("timestamps"), "timestamps")
    categories = _mapping(root.get("categories"), "categories")
    duplicates = _mapping(root.get("duplicates"), "duplicates")
    feature_decisions = _mapping(root.get("feature_decisions"), "feature_decisions")
    mappings_payload = _mapping(categories.get("approved_mappings", {}), "approved_mappings")
    approved_mappings: dict[str, dict[str, str]] = {}
    for column, mapping in mappings_payload.items():
        if not isinstance(mapping, dict):
            raise CleaningConfigurationError(
                f"Approved mapping for {column!r} must be a mapping."
            )
        approved_mappings[str(column)] = {
            str(source): str(target) for source, target in mapping.items()
        }
    try:
        config = CleaningConfig(
            raw_root=_path(input_config["raw_root"], "input.raw_root"),
            validation_report_root=_path(
                input_config["validation_report_root"], "input.validation_report_root"
            ),
            scope_authority_file=_path(
                input_config["scope_authority_file"], "input.scope_authority_file"
            ),
            scope_extraction_metadata_file=_path(
                input_config["scope_extraction_metadata_file"],
                "input.scope_extraction_metadata_file",
            ),
            require_validation_critical_count=int(
                input_config["require_validation_critical_count"]
            ),
            output_root=_path(output["root"], "output.root"),
            report_root=_path(output["report_root"], "output.report_root"),
            output_format=str(output["format"]).casefold(),
            schema_version=str(output["schema_version"]),
            timestamp_columns=_string_tuple(timestamps, "columns"),
            timezone=str(timestamps["timezone"]),
            invalid_timestamp_policy=str(timestamps["invalid_policy"]),
            impute_timestamps=bool(timestamps["impute"]),
            trim_columns=_string_tuple(categories, "trim_columns"),
            blank_to_null_columns=_string_tuple(categories, "blank_to_null_columns"),
            collapse_repeated_spaces_columns=_string_tuple(
                categories, "collapse_repeated_spaces_columns"
            ),
            approved_mappings=approved_mappings,
            duplicate_identifier=str(duplicates["identifier"]),
            exact_duplicate_policy=str(duplicates["exact_policy"]),
            conflict_duplicate_policy=str(duplicates["conflict_policy"]),
            configured_all_null_columns=_string_tuple(
                feature_decisions, "all_null_columns"
            ),
            configured_zero_variance_columns=_string_tuple(
                feature_decisions, "zero_variance_columns"
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, CleaningConfigurationError):
            raise
        raise CleaningConfigurationError(
            f"Cleaning configuration has a missing or invalid value: {error}"
        ) from error
    if config.output_format != "parquet":
        raise CleaningConfigurationError("Step 7 supports Parquet processed outputs only.")
    if config.timezone != "UTC" or config.impute_timestamps:
        raise CleaningConfigurationError("Timestamps must use UTC with imputation disabled.")
    if config.invalid_timestamp_policy != "preserve_and_exclude_from_target":
        raise CleaningConfigurationError("Invalid timestamp policy is not approved.")
    if config.exact_duplicate_policy != "keep_deterministic":
        raise CleaningConfigurationError("Exact duplicate policy is not approved.")
    if config.conflict_duplicate_policy != "exclude_entire_group":
        raise CleaningConfigurationError("Conflicting duplicate policy is not approved.")
    if config.require_validation_critical_count != 0:
        raise CleaningConfigurationError("Cleaning requires exactly zero critical findings.")
    return config


def _read_validation_evidence(report_root: Path) -> ValidationEvidence:
    """Load the machine-readable Step 6 authorization evidence."""
    summary_path = report_root / "validation_summary.md"
    table_root = report_root / "tables"
    required = {
        "summary": summary_path,
        "checks": table_root / "validation_checks.csv",
        "schema": table_root / "schema_validation.csv",
        "scope": table_root / "scope_validation.csv",
        "readiness": table_root / "target_readiness_summary.csv",
        "chronology": table_root / "chronology_violations.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise CleaningInputError(f"Step 6 validation evidence is missing: {missing}")
    summary = summary_path.read_text(encoding="utf-8")
    run_match, status_match = RUN_ID_PATTERN.search(summary), STATUS_PATTERN.search(summary)
    if not run_match or not status_match:
        raise CleaningInputError("Step 6 summary does not expose raw run ID and status.")
    checks = pd.read_csv(required["checks"], dtype={"observed_value": "string"})
    boundary = checks.loc[checks["check_id"].eq("boundary.raw_immutable")]
    if len(boundary) != 1 or boundary.iloc[0]["status"] != "PASS":
        raise CleaningInputError("Step 6 raw-immutability evidence is missing or failed.")
    critical = int(
        (checks["severity"].eq("CRITICAL") & ~checks["status"].eq("PASS")).sum()
    )
    metadata_scope = checks.loc[checks["check_id"].str.startswith("metadata_scope.")]
    scope_table = pd.read_csv(required["scope"])
    if metadata_scope.empty or not metadata_scope["status"].eq("PASS").all():
        raise CleaningInputError("Step 6 metadata/scope reconciliation did not pass.")
    if not scope_table["status"].eq("PASS").all():
        raise CleaningInputError("Step 6 row-level scope validation did not pass.")
    readiness = pd.read_csv(required["readiness"]).set_index("readiness_rule")
    if "candidate_target_eligible" not in readiness.index:
        raise CleaningInputError("Step 6 candidate readiness evidence is missing.")
    candidate = readiness.loc["candidate_target_eligible"]
    schema = pd.read_csv(required["schema"])
    return ValidationEvidence(
        report_root=report_root,
        raw_run_id=run_match.group(1),
        raw_sha256=str(boundary.iloc[0]["observed_value"]),
        row_count=int(candidate["pass_count"] + candidate["fail_count"]),
        column_count=len(schema),
        critical_finding_count=critical,
        overall_status=status_match.group(1),
        candidate_eligible_count=int(candidate["pass_count"]),
        candidate_ineligible_count=int(candidate["fail_count"]),
        chronology=pd.read_csv(required["chronology"], dtype={"unique_key": "string"}),
    )


def _load_raw_run(
    run_path: Path,
) -> tuple[Path, ExtractionMetadata, pd.DataFrame]:
    """Load a successful immutable raw run and reconcile its artifacts."""
    metadata_path = run_path / "metadata.json"
    raw_file = run_path / "service_requests.parquet"
    query_file = run_path / "query.sql"
    for path in (metadata_path, raw_file, query_file):
        if not path.is_file():
            raise CleaningInputError(f"Required raw-run artifact is missing: {path}")
    try:
        metadata = read_metadata(metadata_path)
    except ValueError as error:
        raise CleaningInputError(str(error)) from error
    if metadata.completion_status != "success":
        raise CleaningInputError("Cleaning requires a successful Step 5 raw run.")
    if metadata.raw_file_format.casefold() != "parquet":
        raise CleaningInputError(f"Unsupported raw format: {metadata.raw_file_format}")
    referenced = Path(metadata.raw_file_path)
    if not referenced.is_file() or referenced.resolve() != raw_file.resolve():
        raise CleaningInputError("Raw metadata does not reference this run's Parquet file.")
    frame = pd.read_parquet(raw_file, engine="pyarrow")
    if len(frame) != metadata.retrieved_row_count:
        raise CleaningInputError("Raw metadata and Parquet row counts disagree.")
    if list(frame.columns) != metadata.selected_source_columns:
        raise CleaningInputError("Raw metadata and Parquet columns disagree.")
    return raw_file, metadata, frame


def _verify_authorities(
    *,
    metadata: ExtractionMetadata,
    frame: pd.DataFrame,
    raw_hash: str,
    evidence: ValidationEvidence,
    scope: SelectedScope,
    required_critical_count: int,
) -> None:
    """Fail unless raw, validation, and selected-scope authorities reconcile."""
    failures: list[str] = []
    if evidence.raw_run_id != metadata.run_id:
        failures.append("validation evidence refers to another raw run")
    if evidence.raw_sha256 != raw_hash:
        failures.append("raw SHA-256 differs from Step 6 evidence")
    if evidence.critical_finding_count != required_critical_count:
        failures.append("critical validation finding count is not approved")
    if evidence.row_count != len(frame) or evidence.column_count != len(frame.columns):
        failures.append("validation shape differs from the raw Parquet")
    expected_scope = (
        scope.agency, scope.complaint_type, scope.start_date.date().isoformat(),
        scope.end_date.date().isoformat(),
    )
    metadata_scope = (
        metadata.selected_agency, metadata.selected_complaint_type,
        metadata.selected_start_date, metadata.selected_end_date,
    )
    if expected_scope != metadata_scope:
        failures.append("raw metadata differs from the authoritative Step 3 scope")
    if failures:
        raise CleaningInputError("Cleaning authorization failed: " + "; ".join(failures))


def _check(
    check_id: str, area: str, passed: bool, observed: object, expected: object,
    affected_rows: int, message: str,
) -> CleaningCheck:
    """Build one stable cleaning check."""
    return CleaningCheck(
        check_id=check_id, area=area, status="PASS" if passed else "FAIL",
        observed_value=observed, expected_value=expected,
        affected_rows=int(affected_rows), message=message,
    )


def _package_versions() -> dict[str, str]:
    """Return versions for packages materially used by cleaning."""
    result: dict[str, str] = {}
    for distribution in ("pandas", "pyarrow", "PyYAML"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return result


def _config_hash(path: Path) -> str:
    """Return the SHA-256 of the exact cleaning rule bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chronology_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Return comparable complaint/violation keys from one chronology table."""
    if frame.empty:
        return set()
    governed = frame.loc[frame["violation_type"].isin(
        {"due_before_created", "closed_before_created"}
    )]
    return set(zip(governed["unique_key"].astype("string"), governed["violation_type"]))


def run_cleaning(
    *,
    config_path: Path | str,
    raw_run_path: Path | None = None,
    validation_report_root: Path | None = None,
    output_root: Path | None = None,
    report_root: Path | None = None,
    dry_run: bool = False,
    overwrite_incomplete: bool = False,
    run_started_at: datetime | None = None,
) -> CleaningRunResult:
    """Execute the governed cleaning flow or a write-free dry run."""
    config_file = Path(config_path)
    config = load_cleaning_config(config_file)
    selected_run = raw_run_path or find_latest_successful_run(config.raw_root)
    raw_file, raw_metadata, raw_frame = _load_raw_run(selected_run)
    raw_hash_before = sha256_file(raw_file)
    raw_mtime_before = raw_file.stat().st_mtime_ns
    evidence = _read_validation_evidence(
        validation_report_root or config.validation_report_root
    )
    scope = load_selected_scope_authority(
        config.scope_authority_file, config.scope_extraction_metadata_file
    )
    _verify_authorities(
        metadata=raw_metadata, frame=raw_frame, raw_hash=raw_hash_before,
        evidence=evidence, scope=scope,
        required_critical_count=config.require_validation_critical_count,
    )
    started = run_started_at or datetime.now(timezone.utc)
    started = started.replace(tzinfo=timezone.utc) if started.tzinfo is None else started.astimezone(timezone.utc)
    config_hash = _config_hash(config_file)
    cleaning_hash = hashlib.sha256(
        f"{raw_hash_before}:{config_hash}:{config.schema_version}".encode("utf-8")
    ).hexdigest()[:16]
    run_id = f"{started:%Y%m%dT%H%M%SZ}_{cleaning_hash}"
    paths = output_paths(output_root or config.output_root, run_id)

    duplicate_result = clean_duplicates(raw_frame, identifier=config.duplicate_identifier)
    timestamp_frame, timestamp_summary, timestamp_actions = clean_timestamps(
        duplicate_result.frame, columns=config.timestamp_columns
    )
    category_frame, category_mapping, category_actions = clean_categories(
        timestamp_frame,
        trim_columns=config.trim_columns,
        blank_to_null_columns=config.blank_to_null_columns,
        collapse_repeated_spaces_columns=config.collapse_repeated_spaces_columns,
        approved_mappings=config.approved_mappings,
    )
    cleaned_input, missing_actions, missing_action_models = apply_missing_value_policy(
        category_frame
    )
    frames = apply_governed_target(
        cleaned_input, scope=scope,
        extraction_timestamp=raw_metadata.extraction_completion_utc,
        conflicting_keys=duplicate_result.conflicting_keys,
    )
    leakage, all_null_columns, zero_variance_columns = build_cleaning_leakage_audit(
        frames.cleaned,
        configured_all_null_columns=config.configured_all_null_columns,
        configured_zero_variance_columns=config.configured_zero_variance_columns,
    )
    actions = tuple([
        *timestamp_actions, *category_actions, *missing_action_models,
        *duplicate_result.actions,
    ])
    chronology_actions = build_chronology_actions(frames.cleaned)
    chronology_matches = _chronology_keys(chronology_actions) == _chronology_keys(
        evidence.chronology
    )
    raw_hash_after = sha256_file(raw_file)
    raw_mtime_after = raw_file.stat().st_mtime_ns
    candidate_matches = (
        len(frames.eligible) == evidence.candidate_eligible_count
        and len(frames.excluded) + duplicate_result.removed_exact_copies
        == evidence.candidate_ineligible_count
    )
    checks = (
        _check("input.validation_run", "input", evidence.raw_run_id == raw_metadata.run_id,
               evidence.raw_run_id, raw_metadata.run_id, 0, "Step 6 and raw run IDs reconcile."),
        _check("input.raw_hash", "input", evidence.raw_sha256 == raw_hash_before,
               raw_hash_before, evidence.raw_sha256, 0, "Step 6 and raw hashes reconcile."),
        _check("input.critical_findings", "input", evidence.critical_finding_count == 0,
               evidence.critical_finding_count, 0, evidence.critical_finding_count,
               "Cleaning is authorized only with zero critical validation findings."),
        _check("cleaning.candidate_reconciliation", "eligibility", candidate_matches,
               len(frames.eligible), evidence.candidate_eligible_count,
               abs(len(frames.eligible) - evidence.candidate_eligible_count),
               "Cleaned candidate eligibility reconciles to Step 6."),
        _check("cleaning.chronology_reconciliation", "chronology", chronology_matches,
               len(chronology_actions), len(evidence.chronology),
               0 if chronology_matches else len(chronology_actions),
               "Cleaning chronology actions reconcile exactly to Step 6 row evidence."),
        _check("boundary.raw_hash_immutable", "boundary", raw_hash_before == raw_hash_after,
               raw_hash_after, raw_hash_before, 0 if raw_hash_before == raw_hash_after else len(raw_frame),
               "Raw bytes remain unchanged."),
        _check("boundary.raw_mtime_immutable", "boundary", raw_mtime_before == raw_mtime_after,
               raw_mtime_after, raw_mtime_before, 0 if raw_mtime_before == raw_mtime_after else 1,
               "Raw modification time remains unchanged."),
        _check("output.row_reconciliation", "output",
               len(raw_frame) == len(frames.cleaned) + duplicate_result.removed_exact_copies,
               len(raw_frame), len(frames.cleaned) + duplicate_result.removed_exact_copies,
               0, "Raw rows reconcile after deterministic exact-copy removal."),
        _check("target.binary_eligible", "target",
               set(frames.eligible["missed_resolution_target"].astype(int).unique()).issubset({0, 1}),
               "binary", "binary", 0, "Eligible target values are binary."),
        _check("target.null_excluded", "target",
               frames.excluded["missed_resolution_target"].isna().all(),
               int(frames.excluded["missed_resolution_target"].notna().sum()), 0,
               int(frames.excluded["missed_resolution_target"].notna().sum()),
               "Excluded target values remain null."),
    )
    failures = [check.check_id for check in checks if check.status == "FAIL"]
    if failures:
        raise CleaningInputError("Cleaning checks failed: " + ", ".join(failures))
    tables = build_report_tables(
        frames=frames, input_rows=len(raw_frame),
        removed_exact_copies=duplicate_result.removed_exact_copies,
        timestamp_summary=timestamp_summary, missing_actions=missing_actions,
        category_mapping=category_mapping, duplicate_actions=duplicate_result.audit,
        leakage=leakage, all_null_columns=all_null_columns,
        zero_variance_columns=zero_variance_columns, actions=actions, checks=checks,
    )
    if dry_run:
        LOGGER.info(
            "Cleaning dry run run_id=%s input=%s cleaned=%s eligible=%s excluded=%s proposed_output=%s",
            run_id, len(raw_frame), len(frames.cleaned), len(frames.eligible),
            len(frames.excluded), paths.run_directory,
        )
        return CleaningRunResult(
            dry_run=True, raw_run_path=selected_run, raw_file=raw_file,
            raw_sha256_before=raw_hash_before, raw_sha256_after=raw_hash_after,
            validation_evidence=evidence, output_paths=paths,
            input_row_count=len(raw_frame), cleaned_row_count=len(frames.cleaned),
            eligible_row_count=len(frames.eligible), excluded_row_count=len(frames.excluded),
            removed_exact_duplicate_copies=duplicate_result.removed_exact_copies,
            cleaning_hash=cleaning_hash, actions=actions, checks=checks, tables=tables,
            metadata=None,
        )

    prepare_run_directory(paths, overwrite_incomplete=overwrite_incomplete)
    temporary_run = create_temporary_run(paths)
    metadata: CleaningMetadata | None = None
    try:
        output_hashes = write_and_validate_parquets(temporary_run, frames)
        snapshot_path = temporary_run / "cleaning_rules_snapshot.yaml"
        snapshot_path.write_bytes(config_file.read_bytes())
        output_hashes["rules_snapshot"] = sha256_file(snapshot_path)
        completed = datetime.now(timezone.utc)
        target = frames.eligible["missed_resolution_target"]
        output_path_map = {
            "cleaned": project_path(paths.cleaned),
            "eligible": project_path(paths.eligible),
            "excluded": project_path(paths.excluded),
            "metadata": project_path(paths.metadata),
            "rules_snapshot": project_path(paths.rules_snapshot),
        }
        metadata = CleaningMetadata(
            source_raw_run_id=raw_metadata.run_id,
            source_raw_parquet_path=project_path(raw_file),
            source_raw_sha256=raw_hash_before,
            validation_report_root=project_path(evidence.report_root),
            validation_raw_run_id=evidence.raw_run_id,
            validation_evidence_status=evidence.overall_status,
            validation_critical_count=evidence.critical_finding_count,
            cleaning_started_utc=started.isoformat(),
            cleaning_completed_utc=completed.isoformat(),
            cleaning_config_hash=config_hash,
            cleaning_run_id=run_id,
            input_row_count=len(raw_frame), cleaned_row_count=len(frames.cleaned),
            eligible_row_count=len(frames.eligible), excluded_row_count=len(frames.excluded),
            removed_exact_duplicate_copies=duplicate_result.removed_exact_copies,
            conflicting_duplicate_groups=duplicate_result.conflicting_group_count,
            timestamp_parse_failures=int(timestamp_summary["parse_failure_count"].sum()),
            due_before_created_exclusions=int((~frames.cleaned["valid_due_chronology"]).sum()),
            closed_before_created_exclusions=int((~frames.cleaned["valid_closed_chronology"]).sum()),
            missing_due_date_exclusions=int((~frames.cleaned["has_due_date"]).sum()),
            missing_closed_date_exclusions=int((~frames.cleaned["has_closed_date"]).sum()),
            on_time_target_count=int(target.eq(0).sum()),
            missed_target_count=int(target.eq(1).sum()),
            missed_target_rate=float(target.eq(1).mean()) if len(target) else 0.0,
            zero_variance_columns=zero_variance_columns,
            all_null_columns=all_null_columns,
            output_paths=output_path_map,
            output_hashes=output_hashes,
            warnings=[
                "Processed analytical fields are not automatically approved model features.",
                "Due-date creation-time semantics and mutability remain unresolved.",
            ],
            completion_status="success",
            python_version=CleaningMetadata.current_python_version(),
            package_versions=_package_versions(),
            schema_version=config.schema_version,
        )
        temporary_metadata = temporary_run / "cleaning_metadata.json"
        write_cleaning_metadata(metadata, temporary_metadata)
        read_cleaning_metadata(temporary_metadata)
        temporary_run.replace(paths.run_directory)
        write_cleaning_reports(
            report_root=report_root or config.report_root,
            tables=tables,
            metadata=metadata,
            scope_description=(
                f"agency={scope.agency}; complaint_type={scope.complaint_type}; "
                f"start={scope.start_date.date().isoformat()}; "
                f"end={scope.end_date.date().isoformat()}"
            ),
        )
        update_latest_pointer(paths, updated_utc=completed.isoformat())
    except Exception:
        remove_temporary_run(temporary_run)
        LOGGER.exception("Cleaning failed; temporary processed output was removed.")
        raise
    final_hash = sha256_file(raw_file)
    if final_hash != raw_hash_before or raw_file.stat().st_mtime_ns != raw_mtime_before:
        raise RuntimeError("Raw artifact changed during processed-output finalization.")
    LOGGER.info(
        "Cleaning complete run_id=%s cleaned=%s eligible=%s excluded=%s output=%s",
        run_id, len(frames.cleaned), len(frames.eligible), len(frames.excluded),
        paths.run_directory,
    )
    return CleaningRunResult(
        dry_run=False, raw_run_path=selected_run, raw_file=raw_file,
        raw_sha256_before=raw_hash_before, raw_sha256_after=final_hash,
        validation_evidence=evidence, output_paths=paths,
        input_row_count=len(raw_frame), cleaned_row_count=len(frames.cleaned),
        eligible_row_count=len(frames.eligible), excluded_row_count=len(frames.excluded),
        removed_exact_duplicate_copies=duplicate_result.removed_exact_copies,
        cleaning_hash=cleaning_hash, actions=actions, checks=checks, tables=tables,
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the documented cleaning CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-run", type=Path)
    parser.add_argument("--validation-report-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cleaning CLI and return zero only for a verified result."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        run_cleaning(
            config_path=args.config, raw_run_path=args.raw_run,
            validation_report_root=args.validation_report_root,
            output_root=args.output_root, report_root=args.report_root,
            dry_run=args.dry_run, overwrite_incomplete=args.overwrite_incomplete,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Cleaning command failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
