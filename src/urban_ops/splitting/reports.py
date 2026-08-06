"""Build and write canonical Step 8 split evidence tables and Markdown summary."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import pandas as pd

from urban_ops.splitting.metadata import SplitMetadata
from urban_ops.splitting.models import (
    CandidateSplitResult, SplitBoundaries, SplitFrames, SplitIntegrityCheck,
)


REQUIRED_REPORT_TABLES = (
    "monthly_target_distribution.csv", "candidate_split_boundaries.csv",
    "selected_split_boundaries.csv", "split_row_counts.csv",
    "split_target_distribution.csv", "split_date_ranges.csv",
    "split_month_coverage.csv", "split_integrity_checks.csv",
    "identifier_overlap_checks.csv", "temporal_drift_summary.csv",
    "output_reconciliation.csv",
)

REQUIRED_REPORT_COLUMNS = {
    "monthly_target_distribution.csv": (
        "month", "row_count", "on_time_count", "missed_count",
        "missed_target_rate", "cumulative_row_count", "cumulative_row_share",
        "minimum_created_date", "maximum_created_date", "has_both_classes",
    ),
    "candidate_split_boundaries.csv": (
        "candidate_id", "train_start", "train_end_exclusive",
        "validation_start", "validation_end_exclusive", "test_start",
        "test_end_exclusive", "interval_convention", "train_rows", "train_share",
        "train_months", "train_on_time", "train_missed", "train_missed_rate",
        "train_minimum_class_count", "train_minimum_monthly_row_count",
        "validation_rows", "validation_share", "validation_months",
        "validation_on_time", "validation_missed", "validation_missed_rate",
        "validation_minimum_class_count", "validation_minimum_monthly_row_count",
        "test_rows", "test_share", "test_months", "test_on_time", "test_missed",
        "test_missed_rate", "test_minimum_class_count",
        "test_minimum_monthly_row_count", "minimum_monthly_row_count",
        "date_coverage_gaps", "maximum_rate_difference", "chronology_passed",
        "all_rows_assigned", "all_splits_have_both_classes",
        "minimum_rows_passed", "minimum_class_counts_passed", "qualified", "rank",
        "recommended", "decision_reason",
    ),
    "selected_split_boundaries.csv": (
        "split_name", "start_inclusive", "end_exclusive", "row_count",
        "row_share", "on_time_count", "missed_count", "missed_target_rate",
        "minimum_created_date", "maximum_created_date", "selection_reason",
    ),
    "split_row_counts.csv": ("split_name", "row_count", "row_share"),
    "split_target_distribution.csv": (
        "split_name", "on_time_count", "missed_count", "total_count",
        "missed_target_rate",
    ),
    "split_date_ranges.csv": (
        "split_name", "configured_start_inclusive", "configured_end_exclusive",
        "observed_min_created_date", "observed_max_created_date",
    ),
    "split_month_coverage.csv": (
        "split_name", "month", "row_count", "on_time_count", "missed_count",
        "missed_target_rate",
    ),
    "split_integrity_checks.csv": (
        "check_id", "area", "status", "observed_value", "expected_value",
        "affected_rows", "message",
    ),
    "identifier_overlap_checks.csv": (
        "left_split", "right_split", "overlap_count", "status",
    ),
    "temporal_drift_summary.csv": (
        "comparison_type", "source_period", "destination_period",
        "source_target_rate", "destination_target_rate", "absolute_difference",
        "relative_difference", "source_rows", "destination_rows", "metric_value",
        "interpretation",
    ),
    "output_reconciliation.csv": (
        "check_name", "left_value", "right_value", "status",
    ),
}


def create_temporary_report_directory(report_root: Path) -> Path:
    """Create a sibling report directory that is invisible to final consumers."""
    report_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f".{report_root.name}.tmp-", dir=report_root.parent
    ))


def publish_report_directory(temporary_report: Path, report_root: Path) -> Path | None:
    """Atomically replace reports and return the retained previous-report backup."""
    backup = None
    if report_root.exists():
        backup = report_root.with_name(f".{report_root.name}.backup-{uuid4().hex}")
        report_root.replace(backup)
    try:
        temporary_report.replace(report_root)
    except OSError:
        if backup is not None and backup.exists():
            backup.replace(report_root)
        raise
    return backup


def restore_previous_report_directory(report_root: Path, backup: Path | None) -> None:
    """Remove newly published reports and restore the exact previous directory."""
    if report_root.exists():
        shutil.rmtree(report_root)
    if backup is not None and backup.exists():
        backup.replace(report_root)


def complete_report_publication(backup: Path | None) -> None:
    """Discard the previous-report backup after the latest pointer is durable."""
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def remove_temporary_report_directory(path: Path) -> None:
    """Remove a staged or backed-up report directory after failure."""
    shutil.rmtree(path, ignore_errors=True)


def _target_stats(frame: pd.DataFrame, target_column: str) -> tuple[int, int, float]:
    """Return on-time, missed, and missed-rate values for one split."""
    target = frame[target_column].astype(int)
    on_time, missed = int(target.eq(0).sum()), int(target.eq(1).sum())
    return on_time, missed, float(target.mean()) if len(target) else 0.0


def build_temporal_drift_summary(
    frames: SplitFrames, monthly: pd.DataFrame, *, target_column: str
) -> pd.DataFrame:
    """Compare split target rates and summarize month-level instability."""
    rows: list[dict[str, object]] = []
    stats = {
        name: (*_target_stats(getattr(frames, name), target_column), len(getattr(frames, name)))
        for name in ("train", "validation", "test")
    }
    for source, destination in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ):
        source_rate, destination_rate = stats[source][2], stats[destination][2]
        absolute = abs(destination_rate - source_rate)
        relative = absolute / source_rate if source_rate else pd.NA
        rows.append({
            "comparison_type": "split_pair", "source_period": source,
            "destination_period": destination, "source_target_rate": source_rate,
            "destination_target_rate": destination_rate,
            "absolute_difference": absolute, "relative_difference": relative,
            "source_rows": stats[source][3], "destination_rows": stats[destination][3],
            "metric_value": pd.NA,
            "interpretation": "Temporal target prevalence differs; retain as drift evidence.",
        })
    active = monthly.loc[monthly["row_count"].gt(0)].copy()
    rate = active["missed_target_rate"].astype(float)
    changes = rate.diff().abs()
    summaries = (
        ("minimum_monthly_target_rate", float(rate.min()), str(active.loc[rate.idxmin(), "month"])),
        ("maximum_monthly_target_rate", float(rate.max()), str(active.loc[rate.idxmax(), "month"])),
        ("monthly_target_rate_range", float(rate.max() - rate.min()), "max minus min"),
        ("largest_month_to_month_absolute_change", float(changes.max()), str(active.loc[changes.idxmax(), "month"])),
    )
    for metric, value, detail in summaries:
        rows.append({
            "comparison_type": "monthly_summary", "source_period": metric,
            "destination_period": detail, "source_target_rate": pd.NA,
            "destination_target_rate": pd.NA, "absolute_difference": pd.NA,
            "relative_difference": pd.NA, "source_rows": pd.NA,
            "destination_rows": pd.NA, "metric_value": value,
            "interpretation": "Observed monthly instability; not an automatic rejection rule.",
        })
    return pd.DataFrame(rows, columns=[
        "comparison_type", "source_period", "destination_period",
        "source_target_rate", "destination_target_rate", "absolute_difference",
        "relative_difference", "source_rows", "destination_rows", "metric_value",
        "interpretation",
    ])


def build_report_tables(
    *,
    source: pd.DataFrame,
    frames: SplitFrames,
    boundaries: SplitBoundaries,
    candidates: tuple[CandidateSplitResult, ...],
    selected_candidate_id: str,
    monthly: pd.DataFrame,
    checks: tuple[SplitIntegrityCheck, ...],
    timestamp_column: str,
    identifier_column: str,
    target_column: str,
) -> dict[str, pd.DataFrame]:
    """Build all stable-schema split reports from authoritative in-memory evidence."""
    row_counts: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    date_ranges: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    month_coverage: list[dict[str, object]] = []
    ranges = {
        "train": (boundaries.train_start, boundaries.train_end_exclusive),
        "validation": (boundaries.validation_start, boundaries.validation_end_exclusive),
        "test": (boundaries.test_start, boundaries.test_end_exclusive),
    }
    selected_reason = next(
        candidate.decision_reason for candidate in candidates
        if candidate.candidate_id == selected_candidate_id
    )
    for name, (start, end) in ranges.items():
        split = getattr(frames, name)
        on_time, missed, rate = _target_stats(split, target_column)
        row_counts.append({"split_name": name, "row_count": len(split), "row_share": len(split) / len(source)})
        targets.append({"split_name": name, "on_time_count": on_time, "missed_count": missed, "total_count": len(split), "missed_target_rate": rate})
        date_ranges.append({
            "split_name": name, "configured_start_inclusive": start.isoformat(),
            "configured_end_exclusive": end.isoformat(),
            "observed_min_created_date": split[timestamp_column].min(),
            "observed_max_created_date": split[timestamp_column].max(),
        })
        selected_rows.append({
            "split_name": name, "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(), "row_count": len(split),
            "row_share": len(split) / len(source), "on_time_count": on_time,
            "missed_count": missed, "missed_target_rate": rate,
            "minimum_created_date": split[timestamp_column].min(),
            "maximum_created_date": split[timestamp_column].max(),
            "selection_reason": selected_reason,
        })
        local = split.assign(
            _month=split[timestamp_column].dt.tz_localize(None).dt.to_period("M").astype(str)
        )
        grouped = local.groupby("_month")[target_column].agg(["size", "sum"])
        for month, record in grouped.iterrows():
            total, missed_month = int(record["size"]), int(record["sum"])
            month_coverage.append({
                "split_name": name, "month": month, "row_count": total,
                "on_time_count": total - missed_month, "missed_count": missed_month,
                "missed_target_rate": missed_month / total if total else pd.NA,
            })
    check_table = pd.DataFrame([check.to_dict() for check in checks], columns=[
        "check_id", "area", "status", "observed_value", "expected_value",
        "affected_rows", "message",
    ])
    overlap_rows = []
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(getattr(frames, left)[identifier_column]) & set(getattr(frames, right)[identifier_column])
        overlap_rows.append({"left_split": left, "right_split": right, "overlap_count": len(overlap), "status": "PASS" if not overlap else "FAIL"})
    check_status = check_table.set_index("check_id")["status"]
    reconciliations = pd.DataFrame([
        {"check_name": "input_equals_train_validation_test", "left_value": len(source), "right_value": sum(len(getattr(frames, name)) for name in ranges), "status": "PASS" if len(source) == sum(len(getattr(frames, name)) for name in ranges) else "FAIL"},
        {"check_name": "all_rows_assigned", "left_value": frames.unassigned_row_count, "right_value": 0, "status": "PASS" if frames.unassigned_row_count == 0 else "FAIL"},
        {"check_name": "no_rows_multiply_assigned", "left_value": frames.multiply_assigned_row_count, "right_value": 0, "status": "PASS" if frames.multiply_assigned_row_count == 0 else "FAIL"},
        {"check_name": "target_counts_reconcile", "left_value": sum(row["total_count"] for row in targets), "right_value": sum(row["on_time_count"] + row["missed_count"] for row in targets), "status": "PASS" if all(row["total_count"] == row["on_time_count"] + row["missed_count"] for row in targets) else "FAIL"},
        {"check_name": "eligible_source_hash_unchanged", "left_value": check_status.get("boundary.source_hash_immutable", "FAIL"), "right_value": "PASS", "status": check_status.get("boundary.source_hash_immutable", "FAIL")},
        {"check_name": "eligible_source_mtime_unchanged", "left_value": check_status.get("boundary.source_mtime_immutable", "FAIL"), "right_value": "PASS", "status": check_status.get("boundary.source_mtime_immutable", "FAIL")},
    ])
    return {
        "monthly_target_distribution.csv": monthly,
        "candidate_split_boundaries.csv": pd.DataFrame([item.to_dict() for item in candidates]),
        "selected_split_boundaries.csv": pd.DataFrame(selected_rows),
        "split_row_counts.csv": pd.DataFrame(row_counts),
        "split_target_distribution.csv": pd.DataFrame(targets),
        "split_date_ranges.csv": pd.DataFrame(date_ranges),
        "split_month_coverage.csv": pd.DataFrame(month_coverage, columns=["split_name", "month", "row_count", "on_time_count", "missed_count", "missed_target_rate"]),
        "split_integrity_checks.csv": check_table,
        "identifier_overlap_checks.csv": pd.DataFrame(overlap_rows),
        "temporal_drift_summary.csv": build_temporal_drift_summary(frames, monthly, target_column=target_column),
        "output_reconciliation.csv": reconciliations,
    }


def write_split_reports(
    *, report_root: Path, tables: dict[str, pd.DataFrame], metadata: SplitMetadata,
    selected_candidate_id: str,
) -> None:
    """Write all canonical CSV tables and a metadata-driven Step 8 summary."""
    missing = sorted(set(REQUIRED_REPORT_TABLES).difference(tables))
    if missing:
        raise ValueError(f"Split report tables are missing: {missing}")
    table_root = report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_REPORT_TABLES:
        tables[filename].to_csv(table_root / filename, index=False)
    targets = tables["split_target_distribution.csv"].set_index("split_name")
    drift = tables["temporal_drift_summary.csv"]
    pair = drift.loc[drift["comparison_type"].eq("split_pair")]
    monthly = drift.loc[drift["comparison_type"].eq("monthly_summary")].set_index("source_period")
    checks_pass = tables["split_integrity_checks.csv"]["status"].eq("PASS").all()
    reconciliation_pass = tables["output_reconciliation.csv"]["status"].eq("PASS").all()
    selected_reason = tables["selected_split_boundaries.csv"]["selection_reason"].iloc[0]
    summary = f"""# Step 8 Time-Based Splitting Summary

- Split run: `{metadata.split_id}`
- Source cleaning run: `{metadata.source_cleaning_run_id}`
- Source eligible dataset: `{metadata.source_eligible_dataset_path}`
- Source eligible SHA-256: `{metadata.source_eligible_sha256}`
- Scope: `{metadata.scope_agency} / {metadata.scope_complaint_type}, {metadata.scope_start_date} through {metadata.scope_end_date}`
- Input rows: {metadata.input_row_count:,}
- Split timestamp: `{metadata.timestamp_column}`
- Interval convention: `{metadata.interval_convention}` (`[start, end)`)
- Candidate boundaries evaluated: {len(tables['candidate_split_boundaries.csv'])}
- Selected candidate: `{selected_candidate_id}`
- Selection justification: {selected_reason}

## Selected partitions

- Train: `[{metadata.train_start_inclusive}, {metadata.train_end_exclusive})` — {metadata.train_row_count:,} rows, {targets.loc['train', 'missed_target_rate']:.2%} missed
- Validation: `[{metadata.validation_start_inclusive}, {metadata.validation_end_exclusive})` — {metadata.validation_row_count:,} rows, {targets.loc['validation', 'missed_target_rate']:.2%} missed
- Test: `[{metadata.test_start_inclusive}, {metadata.test_end_exclusive})` — {metadata.test_row_count:,} rows, {targets.loc['test', 'missed_target_rate']:.2%} missed

Every split contains both classes and exceeds the configured total-row and
per-class minimums. Identifier overlaps are zero. Train precedes validation,
and validation precedes test.

## Temporal drift

Target-rate differences are evidence, not automatic rejection criteria. The
largest split-level difference is {pair['absolute_difference'].max():.2%}.
Monthly rates range from {monthly.loc['minimum_monthly_target_rate', 'metric_value']:.2%}
to {monthly.loc['maximum_monthly_target_rate', 'metric_value']:.2%}; the largest
month-to-month change is {monthly.loc['largest_month_to_month_absolute_change', 'metric_value']:.2%}.
The population is temporally unstable and downstream evaluation must retain
that interpretation.

## Integrity and outputs

- Integrity checks: **{'PASS' if checks_pass else 'FAIL'}**
- Output reconciliation: **{'PASS' if reconciliation_pass else 'FAIL'}**
- Source immutability: **{'PASS' if metadata.source_eligible_sha256 and checks_pass else 'FAIL'}**
- Train: `{metadata.output_paths['train']}`
- Validation: `{metadata.output_paths['validation']}`
- Test: `{metadata.output_paths['test']}`

Split data and reports were prepared and validated in temporary sibling
directories. The source hash and modification time were rechecked before
publication. Publication across the split directory, report directory, and
latest pointer is rollback-safe: a failed publication restores the previous
reports and pointer and removes the new split run.

## Test-set governance

The test set must not be used for feature selection, preprocessing design,
category-policy decisions, threshold selection, hyperparameter tuning, or
model selection. Future preprocessing must be fit on train only and applied
unchanged to validation and test.

## Known limitations

The external source is a later-state snapshot, target prevalence changes over
time, and due-date creation-time semantics remain unresolved. These governed
partitions retain the eligible analytical schema; they are not model-ready
feature matrices.

## Step 8 completion decision

The deterministic chronological split is complete. No preprocessing, feature
engineering, model selection, evaluation, or training was performed.
"""
    (report_root / "split_summary.md").write_text(summary, encoding="utf-8")


def validate_split_reports(report_root: Path, metadata: SplitMetadata) -> None:
    """Read back staged reports and reconcile them with successful-run metadata."""
    summary_path = report_root / "split_summary.md"
    required_paths = [
        report_root / "tables" / filename for filename in REQUIRED_REPORT_TABLES
    ]
    missing = [str(path) for path in [summary_path, *required_paths] if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required split reports are missing: {missing}")

    summary = summary_path.read_text(encoding="utf-8")
    if not summary.strip():
        raise RuntimeError("Split summary is empty.")
    if metadata.split_id not in summary:
        raise RuntimeError("Split summary does not reference the expected split_id.")
    if metadata.source_cleaning_run_id not in summary:
        raise RuntimeError(
            "Split summary does not reference the expected source cleaning run."
        )

    tables: dict[str, pd.DataFrame] = {}
    for filename, required_columns in REQUIRED_REPORT_COLUMNS.items():
        table = pd.read_csv(report_root / "tables" / filename)
        if tuple(table.columns) != required_columns:
            raise RuntimeError(f"Split report has an invalid schema: {filename}")
        tables[filename] = table

    split_names = ("train", "validation", "test")
    selected = tables["selected_split_boundaries.csv"].set_index("split_name")
    rows = tables["split_row_counts.csv"].set_index("split_name")
    targets = tables["split_target_distribution.csv"].set_index("split_name")
    if set(selected.index) != set(split_names):
        raise RuntimeError("Selected-boundary reports must contain all three splits.")
    if set(rows.index) != set(split_names) or set(targets.index) != set(split_names):
        raise RuntimeError("Count reports must contain all three splits.")

    expected_boundaries = {
        "train": (metadata.train_start_inclusive, metadata.train_end_exclusive),
        "validation": (
            metadata.validation_start_inclusive,
            metadata.validation_end_exclusive,
        ),
        "test": (metadata.test_start_inclusive, metadata.test_end_exclusive),
    }
    for name in split_names:
        expected_start, expected_end = expected_boundaries[name]
        if (
            selected.loc[name, "start_inclusive"] != expected_start
            or selected.loc[name, "end_exclusive"] != expected_end
        ):
            raise RuntimeError(f"Reported {name} boundaries differ from split metadata.")
        expected_rows = int(getattr(metadata, f"{name}_row_count"))
        expected_on_time = int(getattr(metadata, f"{name}_on_time_count"))
        expected_missed = int(getattr(metadata, f"{name}_missed_count"))
        if int(selected.loc[name, "row_count"]) != expected_rows:
            raise RuntimeError(f"Selected-boundary {name} rows differ from metadata.")
        if (
            int(selected.loc[name, "on_time_count"]) != expected_on_time
            or int(selected.loc[name, "missed_count"]) != expected_missed
        ):
            raise RuntimeError(
                f"Selected-boundary {name} target counts differ from metadata."
            )
        if int(rows.loc[name, "row_count"]) != expected_rows:
            raise RuntimeError(f"Reported {name} rows differ from split metadata.")
        if (
            int(targets.loc[name, "on_time_count"]) != expected_on_time
            or int(targets.loc[name, "missed_count"]) != expected_missed
            or int(targets.loc[name, "total_count"]) != expected_rows
        ):
            raise RuntimeError(f"Reported {name} target counts differ from metadata.")

    reconciliation = tables["output_reconciliation.csv"]
    if reconciliation.empty or not reconciliation["status"].eq("PASS").all():
        raise RuntimeError("Split output reconciliation reports a failure.")
    integrity = tables["split_integrity_checks.csv"]
    if integrity.empty or integrity["status"].eq("FAIL").any():
        raise RuntimeError("Split integrity reports contain a failure.")
