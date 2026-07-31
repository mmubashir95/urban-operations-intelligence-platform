"""CLI and orchestration for immutable NYC 311 raw-data ingestion.

This module performs extraction-level integrity checks only. It deliberately
does not clean data, construct targets, assign eligibility, or create features.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from typing import Any

import pandas as pd
import yaml

from urban_ops.data.api_client import (
    APIRequestError,
    APIResponse,
    PageSummary,
    PaginationResult,
    fetch_json_response,
    fetch_paginated_records,
)
from urban_ops.data.metadata import ExtractionMetadata, read_metadata, write_metadata
from urban_ops.data.query_builder import QueryPlan, build_query_plan
from urban_ops.data.selected_scope import SelectedScope, load_selected_scope_authority
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
REQUIRED_RAW_COLUMNS = frozenset(
    {
        "unique_key", "created_date", "closed_date", "due_date", "agency",
        "agency_name", "complaint_type", "descriptor", "descriptor_2", "status",
        "borough", "incident_zip", "latitude", "longitude", "location_type",
        "open_data_channel_type", "resolution_description",
        "resolution_action_updated_date",
    }
)


class IngestionConfigurationError(ValueError):
    """Raised when ingestion configuration is missing or unsafe."""


class InvalidScopeError(ValueError):
    """Raised when the authoritative scope cannot be used for ingestion."""


class ExtractionValidationError(RuntimeError):
    """Raised when extracted records violate lightweight integrity checks."""


class OutputAlreadyExistsError(FileExistsError):
    """Raised when a run would overwrite an immutable successful extraction."""


@dataclass(frozen=True)
class SourceConfig:
    """Credential-free Socrata dataset identity."""

    dataset_id: str
    base_url: str


@dataclass(frozen=True)
class ScopeConfig:
    """Authoritative scope artifact paths and accepted decisions."""

    authority_file: Path
    extraction_metadata_file: Path
    acceptable_decision_statuses: tuple[str, ...]


@dataclass(frozen=True)
class QueryConfig:
    """Selected raw fields, ordering, and page size."""

    selected_columns: tuple[str, ...]
    order_by: tuple[str, ...]
    page_size: int


@dataclass(frozen=True)
class HttpConfig:
    """Bounded request timeout and retry behavior."""

    timeout_seconds: float
    max_retries: int
    initial_backoff_seconds: float
    maximum_backoff_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    """Immutable raw output root and storage contract."""

    root: Path
    format: str
    schema_version: str


@dataclass(frozen=True)
class IngestionConfig:
    """Validated configuration required by one ingestion run."""

    source: SourceConfig
    scope: ScopeConfig
    query: QueryConfig
    http: HttpConfig
    output: OutputConfig


@dataclass(frozen=True)
class DryRunResult:
    """Resolved dry-run contract; no source pages or artifacts were created."""

    scope: SelectedScope
    plan: QueryPlan
    expected_output_directory: Path


@dataclass(frozen=True)
class IngestionResult:
    """Locations and summary facts for a successful immutable extraction."""

    output_directory: Path
    raw_file: Path
    metadata_file: Path
    query_file: Path
    report_directory: Path
    metadata: ExtractionMetadata
    elapsed_seconds: float


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require one configuration section to be a mapping."""
    if not isinstance(value, dict):
        raise IngestionConfigurationError(f"Configuration section {name!r} is required.")
    return value


def _project_path(value: object, name: str) -> Path:
    """Resolve a configured path relative to the project root."""
    if not isinstance(value, str) or not value.strip():
        raise IngestionConfigurationError(f"Configuration path {name!r} is required.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_ingestion_config(path: Path | str) -> IngestionConfig:
    """Load and validate the YAML ingestion configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Ingestion configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise IngestionConfigurationError(
            f"Unable to parse ingestion configuration: {config_path}"
        ) from error
    root = _mapping(payload, "root")
    source = _mapping(root.get("source"), "source")
    scope = _mapping(root.get("scope"), "scope")
    query = _mapping(root.get("query"), "query")
    http = _mapping(root.get("http"), "http")
    output = _mapping(root.get("output"), "output")
    try:
        config = IngestionConfig(
            source=SourceConfig(str(source["dataset_id"]), str(source["base_url"])),
            scope=ScopeConfig(
                _project_path(scope["authority_file"], "authority_file"),
                _project_path(
                    scope["extraction_metadata_file"], "extraction_metadata_file"
                ),
                tuple(str(value) for value in scope["acceptable_decision_statuses"]),
            ),
            query=QueryConfig(
                tuple(str(value) for value in query["selected_columns"]),
                tuple(str(value) for value in query["order_by"]),
                int(query["page_size"]),
            ),
            http=HttpConfig(
                float(http["timeout_seconds"]),
                int(http["max_retries"]),
                float(http["initial_backoff_seconds"]),
                float(http["maximum_backoff_seconds"]),
            ),
            output=OutputConfig(
                _project_path(output["root"], "output.root"),
                str(output["format"]).casefold(),
                str(output["schema_version"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IngestionConfigurationError(
            f"Ingestion configuration has a missing or invalid value: {error}"
        ) from error
    _validate_config(config)
    return config


def _validate_config(config: IngestionConfig) -> None:
    """Validate bounded HTTP, source, query, scope, and output settings."""
    if not config.source.dataset_id or not config.source.base_url.startswith("https://"):
        raise IngestionConfigurationError("A credential-free HTTPS source URL is required.")
    if "?" in config.source.base_url or config.source.dataset_id not in config.source.base_url:
        raise IngestionConfigurationError("Source URL must identify the configured dataset.")
    if not REQUIRED_RAW_COLUMNS.issubset(config.query.selected_columns):
        missing = sorted(REQUIRED_RAW_COLUMNS.difference(config.query.selected_columns))
        raise IngestionConfigurationError(f"Required raw columns are missing: {missing}")
    if not config.scope.acceptable_decision_statuses:
        raise IngestionConfigurationError("At least one acceptable scope status is required.")
    if config.http.timeout_seconds <= 0 or config.http.max_retries < 0:
        raise IngestionConfigurationError("HTTP timeout and retry settings are invalid.")
    if (
        config.http.initial_backoff_seconds < 0
        or config.http.maximum_backoff_seconds < config.http.initial_backoff_seconds
    ):
        raise IngestionConfigurationError("HTTP backoff settings are invalid.")
    if config.output.format != "parquet":
        raise IngestionConfigurationError("Step 5 raw output format must be parquet.")


def load_ingestion_scope(config: IngestionConfig) -> SelectedScope:
    """Load the authority and enforce its downstream decision status."""
    try:
        scope = load_selected_scope_authority(
            config.scope.authority_file,
            config.scope.extraction_metadata_file,
        )
    except (FileNotFoundError, ValueError) as error:
        raise InvalidScopeError(str(error)) from error
    if scope.decision_status not in config.scope.acceptable_decision_statuses:
        raise InvalidScopeError(
            f"Scope decision status {scope.decision_status!r} is not approved for ingestion."
        )
    return scope


def build_ingestion_plan(config: IngestionConfig, scope: SelectedScope) -> QueryPlan:
    """Resolve a query plan from validated configuration and authority."""
    return build_query_plan(
        agency=scope.agency,
        complaint_type=scope.complaint_type,
        start_date=scope.start_date,
        end_date=scope.end_date,
        selected_columns=config.query.selected_columns,
        order_by=config.query.order_by,
        page_size=config.query.page_size,
    )


def _request_headers() -> dict[str, str]:
    """Return source headers with an optional environment-only application token."""
    headers = {"User-Agent": "urban-operations-intelligence-ingestion/1.0"}
    token = os.getenv("NYC_OPEN_DATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    return headers


def _source_count(payload: object) -> int:
    """Parse one Socrata count response."""
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise APIRequestError("Count query must return exactly one object.")
    value = payload[0].get("record_count")
    try:
        count = int(str(value))
    except (TypeError, ValueError) as error:
        raise APIRequestError("Count query did not return an integer record_count.") from error
    if count < 0:
        raise APIRequestError("Count query returned a negative record_count.")
    return count


def validate_extraction(
    frame: pd.DataFrame,
    *,
    scope: SelectedScope,
    selected_columns: Sequence[str],
    expected_count: int,
    final_source_count: int,
    query_hash: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Run lightweight extraction checks and return a validation table and warnings."""
    checks: list[dict[str, object]] = []
    warnings: list[str] = []

    def add(name: str, passed: bool, observed: object, expected: object, message: str) -> None:
        checks.append({
            "check_name": name,
            "status": "PASS" if passed else "FAIL",
            "observed_value": observed,
            "expected_value": expected,
            "message": message,
        })

    missing = sorted(set(selected_columns).difference(frame.columns))
    add("required_source_columns", not missing, "|".join(missing), "none missing",
        "Every configured raw source column must be represented.")
    add("non_empty_extract", len(frame) > 0, len(frame), "> 0",
        "The selected historical scope must return records.")
    def source_series(column: str) -> pd.Series:
        """Return an aligned null series when a required response field is absent."""
        return frame[column] if column in frame else pd.Series(pd.NA, index=frame.index)

    agency_bad = int((~source_series("agency").eq(scope.agency)).sum())
    type_bad = int((~source_series("complaint_type").eq(scope.complaint_type)).sum())
    add("agency_scope", agency_bad == 0, agency_bad, 0, "All rows must match selected agency.")
    add("complaint_type_scope", type_bad == 0, type_bad, 0,
        "All rows must match selected complaint type.")
    created = pd.to_datetime(
        source_series("created_date"), errors="coerce", utc=True, format="mixed"
    )
    start = scope.start_date.tz_localize("UTC")
    end_exclusive = (scope.end_date + pd.Timedelta(days=1)).tz_localize("UTC")
    invalid_created = int((created.isna() | created.lt(start) | created.ge(end_exclusive)).sum())
    add("created_date_scope", invalid_created == 0, invalid_created, 0,
        "Created timestamps must fall inside the inclusive selected calendar scope.")
    add("preflight_count", len(frame) == expected_count, len(frame), expected_count,
        "Retrieved rows must equal the preflight count.")
    add("source_stability", expected_count == final_source_count, final_source_count,
        expected_count, "Preflight and postflight source counts must match.")
    add("query_hash", bool(query_hash), query_hash, "non-empty",
        "The deterministic query must have a stable hash.")
    failures = [row["check_name"] for row in checks if row["status"] == "FAIL"]
    if failures:
        raise ExtractionValidationError(
            "Extraction integrity checks failed: " + ", ".join(failures)
        )

    duplicate_count = int(frame["unique_key"].duplicated().sum())
    due_missing = int(frame["due_date"].isna().sum())
    closed_missing = int(frame["closed_date"].isna().sum())
    due = pd.to_datetime(frame["due_date"], errors="coerce", utc=True, format="mixed")
    closed = pd.to_datetime(frame["closed_date"], errors="coerce", utc=True, format="mixed")
    chronology = int(((due.notna() & due.lt(created)) | (closed.notna() & closed.lt(created))).sum())
    for label, count in (
        ("duplicate unique-key rows", duplicate_count),
        ("missing due dates", due_missing),
        ("missing closed dates", closed_missing),
        ("invalid timestamp chronology rows", chronology),
    ):
        checks.append({
            "check_name": label.replace(" ", "_"),
            "status": "WARN" if count else "PASS",
            "observed_value": count,
            "expected_value": "reported, not modified",
            "message": "Observation is preserved for Step 6 validation.",
        })
        if count:
            warnings.append(f"{count} {label}; preserved for Step 6 validation.")
    statuses = sorted(frame["status"].dropna().astype(str).unique())
    checks.append({
        "check_name": "status_values_observed",
        "status": "PASS",
        "observed_value": "|".join(statuses),
        "expected_value": "reported, not normalized",
        "message": "Status vocabulary is preserved for Step 6 governance.",
    })
    return pd.DataFrame(checks), warnings


def _prepare_output_directory(
    final_directory: Path, *, overwrite: bool
) -> None:
    """Protect successful runs and replace only explicitly incomplete outputs."""
    if not final_directory.exists():
        return
    metadata_path = final_directory / "metadata.json"
    successful = False
    if metadata_path.is_file():
        try:
            successful = read_metadata(metadata_path).completion_status == "success"
        except ValueError:
            successful = False
    if successful:
        raise OutputAlreadyExistsError(
            f"Successful raw extraction is immutable: {final_directory}"
        )
    if not overwrite:
        raise OutputAlreadyExistsError(
            f"Incomplete output exists; pass --overwrite to replace it: {final_directory}"
        )
    shutil.rmtree(final_directory)


def _write_reports(
    *,
    report_directory: Path,
    metadata: ExtractionMetadata,
    pages: Sequence[PageSummary],
    validation: pd.DataFrame,
) -> None:
    """Write concise latest-run ingestion reports."""
    tables = report_directory / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(page) for page in pages).to_csv(
        tables / "page_summary.csv", index=False
    )
    validation.to_csv(tables / "extraction_validation.csv", index=False)
    warnings = "\n".join(f"- {warning}" for warning in metadata.warnings) or "- None"
    summary = f"""# API Ingestion Summary

- Completion status: **{metadata.completion_status}**
- Scope: `{metadata.selected_agency}` / `{metadata.selected_complaint_type}`, `{metadata.selected_start_date}` through `{metadata.selected_end_date}`
- Scope authority: `{metadata.scope_authority_path}`
- Dataset: `{metadata.dataset_id}` at `{metadata.base_url}`
- Query hash: `{metadata.query_hash}`
- Extraction: `{metadata.extraction_start_utc}` to `{metadata.extraction_completion_utc}`
- Expected/retrieved rows: {metadata.expected_source_count:,} / {metadata.retrieved_row_count:,}
- Pages / retries: {metadata.page_count} / {metadata.retry_count}
- Duplicate unique-key rows: {metadata.duplicate_unique_key_count}
- Created-date range: `{metadata.minimum_created_date}` to `{metadata.maximum_created_date}`
- Immutable raw output: `{metadata.raw_file_path}`

## Warnings

{warnings}

The API is live and may receive historical corrections. This saved raw run is
immutable and is the downstream source for this extraction snapshot. Full data
quality and business validation belongs to Step 6.
"""
    (report_directory / "ingestion_summary.md").write_text(summary, encoding="utf-8")


def find_latest_successful_run(output_root: Path) -> Path:
    """Return the newest completed immutable run for downstream consumers."""
    candidates: list[Path] = []
    if output_root.is_dir():
        for metadata_path in output_root.glob("extraction_date=*/run_id=*/metadata.json"):
            try:
                if read_metadata(metadata_path).completion_status == "success":
                    candidates.append(metadata_path.parent)
            except ValueError:
                continue
    if not candidates:
        raise FileNotFoundError(f"No successful raw extraction exists under {output_root}")
    return max(candidates, key=lambda path: path.name)


def run_ingestion(
    config: IngestionConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    report_directory: Path | None = None,
    run_started_at: datetime | None = None,
    request_json: Callable[..., APIResponse] = fetch_json_response,
) -> DryRunResult | IngestionResult:
    """Execute a dry run or one atomic immutable raw extraction."""
    started_at = run_started_at or datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    else:
        started_at = started_at.astimezone(timezone.utc)
    timer = monotonic()
    scope = load_ingestion_scope(config)
    plan = build_ingestion_plan(config, scope)
    run_id = f"{started_at:%Y%m%dT%H%M%SZ}_{plan.query_hash}"
    final_directory = (
        config.output.root
        / f"extraction_date={started_at:%Y-%m-%d}"
        / f"run_id={run_id}"
    )
    LOGGER.info(
        "Resolved ingestion scope agency=%s complaint_type=%s start=%s end=%s authority=%s query_hash=%s",
        scope.agency, scope.complaint_type, scope.start_date.date(), scope.end_date.date(),
        scope.authority_path, plan.query_hash,
    )
    if dry_run:
        LOGGER.info(
            "Dry run: columns=%s ordering=%s page_size=%s expected_output=%s",
            len(plan.selected_columns), plan.ordering_clause, plan.page_size, final_directory,
        )
        return DryRunResult(scope, plan, final_directory)

    _prepare_output_directory(final_directory, overwrite=overwrite)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=final_directory.parent)
    )
    headers = _request_headers()
    transport = {
        "timeout_seconds": config.http.timeout_seconds,
        "max_attempts": config.http.max_retries + 1,
        "headers": headers,
        "initial_backoff_seconds": config.http.initial_backoff_seconds,
        "maximum_backoff_seconds": config.http.maximum_backoff_seconds,
    }
    try:
        count_preflight_at = datetime.now(timezone.utc)
        preflight_response = request_json(
            plan.count_url(config.source.base_url), **transport
        )
        expected_count = _source_count(preflight_response.payload)
        LOGGER.info("Count preflight expected_rows=%s", expected_count)
        pagination: PaginationResult = fetch_paginated_records(
            plan,
            base_url=config.source.base_url,
            timeout_seconds=config.http.timeout_seconds,
            max_attempts=config.http.max_retries + 1,
            headers=headers,
            initial_backoff_seconds=config.http.initial_backoff_seconds,
            maximum_backoff_seconds=config.http.maximum_backoff_seconds,
            request_json=request_json,
        )
        count_postflight_at = datetime.now(timezone.utc)
        postflight_response = request_json(
            plan.count_url(config.source.base_url), **transport
        )
        final_source_count = _source_count(postflight_response.payload)
        sparse_response_frame = pd.DataFrame.from_records(pagination.records)
        sparse_json_columns = sorted(
            set(plan.selected_columns).difference(sparse_response_frame.columns)
        )
        # Socrata omits keys whose value is null. Reindexing restores the
        # explicitly requested raw schema without altering any returned value.
        frame = sparse_response_frame.reindex(columns=plan.selected_columns)
        validation, warnings = validate_extraction(
            frame,
            scope=scope,
            selected_columns=plan.selected_columns,
            expected_count=expected_count,
            final_source_count=final_source_count,
            query_hash=plan.query_hash,
        )
        if sparse_json_columns:
            warnings.append(
                "Configured columns absent from every sparse JSON object were "
                "materialized as raw null columns: " + ", ".join(sparse_json_columns)
            )
            validation.loc[len(validation)] = {
                "check_name": "sparse_json_columns",
                "status": "WARN",
                "observed_value": "|".join(sparse_json_columns),
                "expected_value": "materialized as raw null columns",
                "message": "Socrata omits keys that are null in every response object.",
            }
        raw_path = temporary_directory / "service_requests.parquet"
        query_path = temporary_directory / "query.sql"
        metadata_path = temporary_directory / "metadata.json"
        frame.to_parquet(raw_path, index=False, engine="pyarrow")
        query_path.write_text(plan.audit_text(), encoding="utf-8")
        read_back = pd.read_parquet(raw_path, engine="pyarrow")
        if len(read_back) != len(frame) or list(read_back.columns) != list(frame.columns):
            raise ExtractionValidationError("Raw Parquet read-back validation failed.")
        completed_at = datetime.now(timezone.utc)
        created = pd.to_datetime(frame["created_date"], utc=True, format="mixed")
        duplicate_count = int(frame["unique_key"].duplicated().sum())
        if pagination.duplicate_keys_crossing_pages and not duplicate_count:
            raise ExtractionValidationError("Cross-page duplicate accounting is inconsistent.")
        final_raw_path = final_directory / raw_path.name
        metadata = ExtractionMetadata(
            source_name="NYC Open Data — 311 Service Requests",
            dataset_id=config.source.dataset_id,
            base_url=config.source.base_url,
            scope_authority_path=str(scope.authority_path),
            scope_authority_extraction_timestamp=scope.authority_extraction_timestamp.isoformat(),
            selected_agency=scope.agency,
            selected_complaint_type=scope.complaint_type,
            selected_start_date=scope.start_date.date().isoformat(),
            selected_end_date=scope.end_date.date().isoformat(),
            extraction_start_utc=started_at.isoformat(),
            extraction_completion_utc=completed_at.isoformat(),
            count_preflight_utc=count_preflight_at.isoformat(),
            count_postflight_utc=count_postflight_at.isoformat(),
            selected_source_columns=list(plan.selected_columns),
            where_clause=plan.where_clause,
            ordering=list(plan.order_by),
            page_size=plan.page_size,
            timeout_seconds=config.http.timeout_seconds,
            maximum_retries=config.http.max_retries,
            expected_source_count=expected_count,
            retrieved_row_count=len(frame),
            page_count=len(pagination.pages),
            page_row_counts=[page.returned_rows for page in pagination.pages],
            retry_count=(
                preflight_response.retry_count + pagination.retry_count
                + postflight_response.retry_count
            ),
            minimum_created_date=created.min().isoformat(),
            maximum_created_date=created.max().isoformat(),
            unique_key_count=int(frame["unique_key"].nunique(dropna=True)),
            duplicate_unique_key_count=duplicate_count,
            raw_file_path=str(final_raw_path),
            raw_file_format=config.output.format,
            raw_file_size=raw_path.stat().st_size,
            schema_version=config.output.schema_version,
            query_hash=plan.query_hash,
            run_id=run_id,
            warnings=warnings,
            completion_status="success",
            python_version=ExtractionMetadata.current_python_version(),
            package_version=None,
        )
        write_metadata(metadata, metadata_path)
        if read_metadata(metadata_path).retrieved_row_count != len(read_back):
            raise ExtractionValidationError("Metadata and stored raw row counts disagree.")
        temporary_directory.replace(final_directory)
        report_root = report_directory or PROJECT_ROOT / "reports/07_api_ingestion"
        _write_reports(
            report_directory=report_root,
            metadata=metadata,
            pages=pagination.pages,
            validation=validation,
        )
        elapsed = monotonic() - timer
        LOGGER.info(
            "Ingestion complete rows=%s pages=%s retries=%s duplicates=%s output=%s elapsed_seconds=%.2f",
            len(frame), len(pagination.pages), metadata.retry_count, duplicate_count,
            final_directory, elapsed,
        )
        return IngestionResult(
            final_directory,
            final_directory / raw_path.name,
            final_directory / metadata_path.name,
            final_directory / query_path.name,
            report_root,
            metadata,
            elapsed,
        )
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        LOGGER.exception("Ingestion failed; temporary output was removed.")
        raise


def _parser() -> argparse.ArgumentParser:
    """Build the documented Step 5 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion CLI and return a process-compatible status code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        config = load_ingestion_config(args.config)
        if args.output_root:
            config = replace(config, output=replace(config.output, root=args.output_root))
        if args.page_size is not None:
            config = replace(config, query=replace(config.query, page_size=args.page_size))
        if args.timeout is not None:
            config = replace(config, http=replace(config.http, timeout_seconds=args.timeout))
        if args.max_retries is not None:
            config = replace(config, http=replace(config.http, max_retries=args.max_retries))
        _validate_config(config)
        run_ingestion(config, dry_run=args.dry_run, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("Ingestion command failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
