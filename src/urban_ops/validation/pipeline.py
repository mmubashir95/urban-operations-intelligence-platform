"""CLI and orchestration for non-mutating validation of the latest raw run."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from urban_ops.data.ingest import find_latest_successful_run
from urban_ops.data.metadata import ExtractionMetadata, read_metadata
from urban_ops.data.selected_scope import SelectedScope, load_selected_scope_authority
from urban_ops.utils.paths import PROJECT_ROOT
from urban_ops.validation.categories import profile_categories, validate_statuses
from urban_ops.validation.duplicates import analyze_duplicates
from urban_ops.validation.geography import BoundingBox, validate_geography
from urban_ops.validation.missingness import profile_missingness
from urban_ops.validation.models import Severity, ValidationCheck, ValidationRunResult
from urban_ops.validation.reports import proposed_cleaning_actions, write_validation_reports
from urban_ops.validation.schema import validate_schema
from urban_ops.validation.severity import checks_frame, make_check, overall_status, validation_exit_code
from urban_ops.validation.target_readiness import analyze_target_readiness
from urban_ops.validation.timestamps import analyze_timestamps, chronology_violations


LOGGER = logging.getLogger(__name__)


class ValidationConfigurationError(ValueError):
    """Raised when Step 6 configuration is missing or invalid."""


class ValidationInputError(RuntimeError):
    """Raised when a raw run cannot be safely inspected."""


@dataclass(frozen=True)
class ValidationConfig:
    """Validated settings needed by the Step 6 pipeline."""

    raw_root: Path
    expected_dataset_id: str
    supported_format: str
    scope_authority_file: Path
    scope_extraction_metadata_file: Path
    report_root: Path
    required_columns: tuple[str, ...]
    require_column_order: bool
    timestamp_columns: tuple[str, ...]
    category_columns: tuple[str, ...]
    rare_category_max_rows: int
    high_cardinality_threshold: int
    maximum_profile_values_per_column: int
    null_like_strings: tuple[str, ...]
    latitude_range: tuple[float, float]
    longitude_range: tuple[float, float]
    nyc_box: BoundingBox


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require a YAML configuration section to be a mapping."""
    if not isinstance(value, dict):
        raise ValidationConfigurationError(f"Configuration section {name!r} is required.")
    return value


def _path(value: object, name: str) -> Path:
    """Resolve a required configured path relative to the project root."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationConfigurationError(f"Configuration path {name!r} is required.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _float_pair(value: object, name: str) -> tuple[float, float]:
    """Parse an increasing two-value numeric range."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValidationConfigurationError(f"{name} must contain exactly two values.")
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise ValidationConfigurationError(f"{name} must be numeric.") from error
    if result[0] >= result[1]:
        raise ValidationConfigurationError(f"{name} must be increasing.")
    return result


def load_validation_config(path: Path | str) -> ValidationConfig:
    """Load and validate the Step 6 YAML configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Validation configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValidationConfigurationError(f"Unable to parse {config_path}") from error
    root = _mapping(payload, "root")
    input_config = _mapping(root.get("input"), "input")
    output = _mapping(root.get("output"), "output")
    schema = _mapping(root.get("schema"), "schema")
    timestamps = _mapping(root.get("timestamps"), "timestamps")
    categories = _mapping(root.get("categories"), "categories")
    missingness = _mapping(root.get("missingness"), "missingness")
    geography = _mapping(root.get("geography"), "geography")
    box = _mapping(geography.get("nyc_bounding_box"), "geography.nyc_bounding_box")
    try:
        config = ValidationConfig(
            raw_root=_path(input_config["raw_root"], "input.raw_root"),
            expected_dataset_id=str(input_config["expected_dataset_id"]),
            supported_format=str(input_config["supported_format"]).casefold(),
            scope_authority_file=_path(
                input_config["scope_authority_file"], "input.scope_authority_file"
            ),
            scope_extraction_metadata_file=_path(
                input_config["scope_extraction_metadata_file"],
                "input.scope_extraction_metadata_file",
            ),
            report_root=_path(output["report_root"], "output.report_root"),
            required_columns=tuple(str(value) for value in schema["required_columns"]),
            require_column_order=bool(schema["require_column_order"]),
            timestamp_columns=tuple(str(value) for value in timestamps["columns"]),
            category_columns=tuple(str(value) for value in categories["columns"]),
            rare_category_max_rows=int(categories["rare_category_max_rows"]),
            high_cardinality_threshold=int(categories["high_cardinality_threshold"]),
            maximum_profile_values_per_column=int(
                categories["maximum_profile_values_per_column"]
            ),
            null_like_strings=tuple(str(value) for value in missingness["null_like_strings"]),
            latitude_range=_float_pair(geography["latitude_valid_range"], "latitude range"),
            longitude_range=_float_pair(
                geography["longitude_valid_range"], "longitude range"
            ),
            nyc_box=BoundingBox(
                float(box["min_latitude"]), float(box["max_latitude"]),
                float(box["min_longitude"]), float(box["max_longitude"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValidationConfigurationError):
            raise
        raise ValidationConfigurationError(
            f"Validation configuration has a missing or invalid value: {error}"
        ) from error
    if not config.required_columns or not config.timestamp_columns:
        raise ValidationConfigurationError("Required and timestamp column lists cannot be empty.")
    if config.supported_format != "parquet":
        raise ValidationConfigurationError("Step 6 currently supports Parquet raw runs only.")
    return config


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable raw file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_run(raw_run_path: Path) -> tuple[Path, Path, Path, ExtractionMetadata, pd.DataFrame]:
    """Validate required run artifacts and load the immutable Parquet snapshot."""
    if not raw_run_path.is_dir():
        raise ValidationInputError(f"Raw run directory is missing: {raw_run_path}")
    metadata_path = raw_run_path / "metadata.json"
    query_path = raw_run_path / "query.sql"
    raw_file = raw_run_path / "service_requests.parquet"
    for path, label in (
        (metadata_path, "metadata"), (query_path, "query audit"), (raw_file, "raw Parquet")
    ):
        if not path.is_file():
            raise ValidationInputError(f"Raw run {label} artifact is missing: {path}")
    try:
        metadata = read_metadata(metadata_path)
    except ValueError as error:
        raise ValidationInputError(str(error)) from error
    if metadata.completion_status != "success":
        raise ValidationInputError("Step 6 requires a successful raw-run completion status.")
    if not metadata.query_hash:
        raise ValidationInputError("Raw-run metadata is missing query_hash.")
    if not metadata.scope_authority_path:
        raise ValidationInputError("Raw-run metadata is missing scope authority provenance.")
    if metadata.raw_file_format.casefold() != "parquet":
        raise ValidationInputError(f"Unsupported raw format: {metadata.raw_file_format}")
    referenced_raw_file = Path(metadata.raw_file_path)
    if not referenced_raw_file.is_file():
        raise ValidationInputError(
            f"Metadata-referenced raw Parquet is missing: {referenced_raw_file}"
        )
    if referenced_raw_file.resolve() != raw_file.resolve():
        raise ValidationInputError(
            "Metadata raw_file_path does not identify this immutable run's Parquet file."
        )
    try:
        frame = pd.read_parquet(raw_file)
    except (OSError, ValueError) as error:
        raise ValidationInputError(f"Unable to read raw Parquet: {raw_file}") from error
    return raw_file, metadata_path, query_path, metadata, frame


def _scope_table(
    frame: pd.DataFrame, scope: SelectedScope, parsed_created: pd.Series
) -> tuple[pd.DataFrame, list[ValidationCheck]]:
    """Validate every row against the authoritative selected scope."""
    total = len(frame)
    agency_bad = ~frame.get("agency", pd.Series(pd.NA, index=frame.index)).eq(scope.agency)
    type_bad = ~frame.get("complaint_type", pd.Series(pd.NA, index=frame.index)).eq(
        scope.complaint_type
    )
    start = scope.start_date.tz_localize("UTC")
    end_exclusive = (scope.end_date + pd.Timedelta(days=1)).tz_localize("UTC")
    unparseable = parsed_created.isna()
    before = parsed_created.notna() & parsed_created.lt(start)
    after = parsed_created.notna() & parsed_created.ge(end_exclusive)
    metrics = [
        ("out_of_scope_agency", int(agency_bad.sum()), scope.agency),
        ("out_of_scope_complaint_type", int(type_bad.sum()), scope.complaint_type),
        ("before_start", int(before.sum()), scope.start_date.date().isoformat()),
        ("after_end", int(after.sum()), scope.end_date.date().isoformat()),
        ("unparseable_created_date", int(unparseable.sum()), "0"),
    ]
    table = pd.DataFrame([
        {
            "scope_rule": name, "affected_row_count": count,
            "affected_row_rate": count / total if total else 0.0,
            "expected_value": expected,
            "status": "PASS" if count == 0 else "FAIL", "severity": "CRITICAL",
        }
        for name, count, expected in metrics
    ])
    checks = [
        make_check(
            check_id=f"scope.{name}", area="scope", check_name=name.replace("_", " ").title(),
            severity=Severity.CRITICAL, passed=count == 0, observed_value=count,
            expected_value=0, affected_rows=count, total_rows=total,
            message="Every raw record must match the authoritative selected scope.",
            recommended_action="Reject or re-ingest the raw run; do not clean across scope boundaries.",
        )
        for name, count, _ in metrics
    ]
    return table, checks


def _metadata_scope_checks(
    metadata: ExtractionMetadata, scope: SelectedScope, total_rows: int
) -> list[ValidationCheck]:
    """Reconcile raw metadata to the current Step 3 scope authority."""
    expected = {
        "agency": scope.agency,
        "complaint_type": scope.complaint_type,
        "start_date": scope.start_date.date().isoformat(),
        "end_date": scope.end_date.date().isoformat(),
        "authority_path": str(scope.authority_path.resolve()),
    }
    observed = {
        "agency": metadata.selected_agency,
        "complaint_type": metadata.selected_complaint_type,
        "start_date": metadata.selected_start_date,
        "end_date": metadata.selected_end_date,
        "authority_path": str(Path(metadata.scope_authority_path).resolve()),
    }
    return [
        make_check(
            check_id=f"metadata_scope.{name}", area="metadata",
            check_name=f"Metadata scope {name}", severity=Severity.CRITICAL,
            passed=observed[name] == expected[name], observed_value=observed[name],
            expected_value=expected[name], affected_rows=0 if observed[name] == expected[name] else total_rows,
            total_rows=total_rows,
            message="Raw-run metadata must agree with the current selected-scope authority.",
            recommended_action="Stop downstream work and create a new governed ingestion run.",
        )
        for name in expected
    ]


def _checks_from_evidence(
    *,
    frame: pd.DataFrame,
    missingness: pd.DataFrame,
    timestamp_summary: pd.DataFrame,
    chronology: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    variants: pd.DataFrame,
    statuses: pd.DataFrame,
    geography: pd.DataFrame,
) -> list[ValidationCheck]:
    """Translate module evidence into the central severity model."""
    total = len(frame)
    checks: list[ValidationCheck] = []
    for row in missingness.itertuples(index=False):
        severity = Severity(str(row.missingness_severity))
        checks.append(make_check(
            check_id=f"missingness.{row.column_name}", area="missingness",
            check_name=f"Missing {row.column_name}", severity=severity,
            passed=row.null_count == 0, observed_value=row.null_count, expected_value=0,
            affected_rows=row.null_count, total_rows=total,
            message=f"Raw nulls in {row.column_name} were measured and preserved.",
            recommended_action=row.recommended_step_7_action,
        ))
        text_issues = row.empty_string_count + row.whitespace_only_count + row.null_like_string_count
        checks.append(make_check(
            check_id=f"null_like.{row.column_name}", area="category",
            check_name=f"Blank or null-like {row.column_name}", severity=Severity.WARNING,
            passed=text_issues == 0, observed_value=text_issues, expected_value=0,
            affected_rows=text_issues, total_rows=total,
            message="String sentinels remain distinct from true null values in raw data.",
            recommended_action="Review and map only approved null-equivalent values in Step 7.",
        ))
    for row in timestamp_summary.itertuples(index=False):
        severity = Severity.CRITICAL if row.column_name == "created_date" else Severity.ERROR
        checks.append(make_check(
            check_id=f"timestamp.{row.column_name}", area="timestamp",
            check_name=f"Parseable {row.column_name}", severity=severity,
            passed=row.parse_failure_count == 0, observed_value=row.parse_failure_count,
            expected_value=0, affected_rows=row.parse_failure_count, total_rows=total,
            message="Non-null timestamps must be parseable under UTC-compatible logic.",
            recommended_action="Quarantine invalid timestamp rows; do not guess or overwrite values.",
        ))
    for violation, group in chronology.groupby("violation_type", sort=True):
        severity = Severity(str(group["severity"].iloc[0]))
        checks.append(make_check(
            check_id=f"chronology.{violation}", area="chronology",
            check_name=violation.replace("_", " ").title(), severity=severity,
            passed=False, observed_value=len(group), expected_value=0,
            affected_rows=int(group["unique_key"].nunique(dropna=False)), total_rows=total,
            message="Timestamp order conflicts with the governed chronology contract.",
            recommended_action=str(group["recommended_step_7_action"].iloc[0]),
        ))
    present_violations = set(chronology["violation_type"]) if len(chronology) else set()
    for violation, severity in (
        ("due_before_created", Severity.ERROR), ("closed_before_created", Severity.ERROR),
        ("resolution_action_before_created", Severity.WARNING),
    ):
        if violation not in present_violations:
            checks.append(make_check(
                check_id=f"chronology.{violation}", area="chronology",
                check_name=violation.replace("_", " ").title(), severity=severity,
                passed=True, observed_value=0, expected_value=0, affected_rows=0,
                total_rows=total, message="Chronology rule was evaluated.",
                recommended_action="No chronology action required.",
            ))
    duplicate_metrics = duplicate_summary.set_index("metric")["row_count"]
    for metric, severity in (
        ("missing_unique_key_rows", Severity.CRITICAL),
        ("redundant_exact_duplicate_rows", Severity.WARNING),
        ("conflicting_duplicate_keys", Severity.ERROR),
    ):
        count = int(duplicate_metrics.get(metric, 0))
        checks.append(make_check(
            check_id=f"duplicate.{metric}", area="duplicate",
            check_name=metric.replace("_", " ").title(), severity=severity,
            passed=count == 0, observed_value=count, expected_value=0,
            affected_rows=count, total_rows=total,
            message="Duplicate evidence is reported without removing or resolving rows.",
            recommended_action="Apply a deterministic approved duplicate rule in Step 7.",
        ))
    checks.append(make_check(
        check_id="category.formatting_variants", area="category",
        check_name="Formatting variant groups", severity=Severity.WARNING,
        passed=len(variants) == 0, observed_value=len(variants), expected_value=0,
        affected_rows=int(variants["row_count"].sum()) if len(variants) else 0,
        total_rows=total, message="Values differ only by case or whitespace formatting.",
        recommended_action="Normalize only verified equivalent variants in Step 7.",
    ))
    unexpected_status_rows = int(statuses.loc[~statuses["is_expected"], "row_count"].sum())
    checks.append(make_check(
        check_id="status.unexpected", area="status", check_name="Unexpected statuses",
        severity=Severity.ERROR, passed=unexpected_status_rows == 0,
        observed_value=unexpected_status_rows, expected_value=0,
        affected_rows=unexpected_status_rows, total_rows=total,
        message="Status is outside the Step 4 allowed and excluded policy.",
        recommended_action="Obtain a governance decision; do not silently map the status.",
    ))
    for row in geography.itertuples(index=False):
        if row.metric == "parseable":
            continue
        severity = Severity(str(row.severity))
        checks.append(make_check(
            check_id=f"geography.{row.field}.{row.metric}", area="geography",
            check_name=f"{row.field} {row.metric}", severity=severity,
            passed=row.row_count == 0, observed_value=row.row_count, expected_value=0,
            affected_rows=row.row_count, total_rows=total,
            message="Geographic quality was measured without correcting or excluding complaints.",
            recommended_action="Preserve complaint and apply an approved feature-level policy in Step 7.",
        ))
    return checks


def run_validation(
    *,
    config_path: Path | str,
    raw_run_path: Path | None = None,
    output_root: Path | None = None,
) -> ValidationRunResult:
    """Run the full validation suite and write reports without mutating raw data."""
    config = load_validation_config(config_path)
    selected_run = raw_run_path or find_latest_successful_run(config.raw_root)
    raw_file, metadata_file, query_file, metadata, frame = _load_raw_run(selected_run)
    raw_hash_before = _sha256(raw_file)
    scope = load_selected_scope_authority(
        config.scope_authority_file, config.scope_extraction_metadata_file
    )
    report_root = output_root or config.report_root
    checks: list[ValidationCheck] = []
    checks.extend(_metadata_scope_checks(metadata, scope, len(frame)))
    checks.append(make_check(
        check_id="metadata.dataset_id", area="metadata", check_name="Expected dataset ID",
        severity=Severity.CRITICAL, passed=metadata.dataset_id == config.expected_dataset_id,
        observed_value=metadata.dataset_id, expected_value=config.expected_dataset_id,
        affected_rows=0 if metadata.dataset_id == config.expected_dataset_id else len(frame),
        total_rows=len(frame), message="Raw provenance must identify the configured NYC 311 source.",
        recommended_action="Reject the raw run and re-ingest from the expected dataset.",
    ))
    schema_table, column_profile, schema_checks = validate_schema(
        frame, metadata=metadata, required_columns=config.required_columns,
        require_column_order=config.require_column_order,
    )
    checks.extend(schema_checks)
    timestamps = analyze_timestamps(frame, config.timestamp_columns)
    created = timestamps.parsed.get("created_date", pd.Series(pd.NaT, index=frame.index))
    scope_table, scope_checks = _scope_table(frame, scope, created)
    checks.extend(scope_checks)
    missingness = profile_missingness(frame, null_like_strings=config.null_like_strings)
    chronology = chronology_violations(frame, timestamps.parsed)
    duplicates = analyze_duplicates(frame)
    category_profile, category_variants = profile_categories(
        frame, columns=config.category_columns, null_like_strings=config.null_like_strings,
        rare_category_max_rows=config.rare_category_max_rows,
        high_cardinality_threshold=config.high_cardinality_threshold,
        maximum_profile_values_per_column=config.maximum_profile_values_per_column,
    )
    status_table = validate_statuses(frame, parsed_timestamps=timestamps.parsed)
    geographic_table, geographic_outliers = validate_geography(
        frame, latitude_range=config.latitude_range, longitude_range=config.longitude_range,
        nyc_box=config.nyc_box,
    )
    readiness = analyze_target_readiness(
        frame, scope=scope, extraction_timestamp=metadata.extraction_completion_utc,
        duplicates=duplicates,
    )
    checks.extend(_checks_from_evidence(
        frame=frame, missingness=missingness,
        timestamp_summary=timestamps.summary, chronology=chronology,
        duplicate_summary=duplicates.summary, variants=category_variants,
        statuses=status_table, geography=geographic_table,
    ))
    raw_hash_after = _sha256(raw_file)
    checks.append(make_check(
        check_id="boundary.raw_immutable", area="boundary", check_name="Raw file immutable",
        severity=Severity.CRITICAL, passed=raw_hash_before == raw_hash_after,
        observed_value=raw_hash_after, expected_value=raw_hash_before,
        affected_rows=0 if raw_hash_before == raw_hash_after else len(frame),
        total_rows=len(frame), message="Step 6 must not rewrite the raw Parquet artifact.",
        recommended_action="Restore or recreate the immutable raw run before proceeding.",
    ))
    check_tuple = tuple(checks)
    check_table = checks_frame(check_tuple)
    tables = {
        "validation_checks.csv": check_table,
        "schema_validation.csv": schema_table,
        "scope_validation.csv": scope_table,
        "column_profile.csv": column_profile,
        "missingness_summary.csv": missingness,
        "timestamp_validation_summary.csv": timestamps.summary,
        "chronology_violations.csv": chronology,
        "duplicate_summary.csv": duplicates.summary,
        "exact_duplicate_records.csv": duplicates.exact_records,
        "conflicting_duplicate_records.csv": duplicates.conflicting_records,
        "category_profile.csv": category_profile,
        "category_variants.csv": category_variants,
        "status_validation.csv": status_table,
        "geographic_validation.csv": geographic_table,
        "geographic_outliers.csv": geographic_outliers,
        "target_readiness_summary.csv": readiness.summary,
        "candidate_exclusion_reason_summary.csv": readiness.exclusions,
        "proposed_cleaning_actions.csv": proposed_cleaning_actions(check_tuple),
    }
    status = overall_status(check_tuple)
    write_validation_reports(
        report_root=report_root, tables=tables, checks=check_tuple, metadata=metadata,
        scope=scope, raw_run_path=selected_run, raw_file=raw_file, overall_status=status,
    )
    LOGGER.info(
        "Validation complete run_id=%s rows=%s status=%s reports=%s",
        metadata.run_id, len(frame), status, report_root,
    )
    return ValidationRunResult(
        raw_run_path=selected_run, raw_file=raw_file, metadata_file=metadata_file,
        query_file=query_file, report_root=report_root, checks=check_tuple, tables=tables,
        overall_status=status, row_count=len(frame), column_count=len(frame.columns),
        raw_sha256_before=raw_hash_before, raw_sha256_after=raw_hash_after,
        metadata=metadata,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser with explicit severity thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Step 6 and return a severity-aware process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        result = run_validation(
            config_path=args.config, raw_run_path=args.raw_run, output_root=args.output_root
        )
    except (FileNotFoundError, ValidationConfigurationError, ValidationInputError, ValueError) as error:
        LOGGER.error("Validation could not run: %s", error)
        return 2
    return validation_exit_code(
        result.checks,
        fail_on_error=args.fail_on_error,
        fail_on_warning=args.fail_on_warning,
    )


if __name__ == "__main__":
    raise SystemExit(main())
