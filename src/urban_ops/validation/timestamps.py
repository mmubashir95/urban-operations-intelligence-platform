"""Profile timestamp parseability and chronology using temporary UTC views."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from urban_ops.validation.models import TimestampAnalysis


TIMESTAMP_SUMMARY_COLUMNS = [
    "column_name", "non_null_count", "parse_success_count", "parse_failure_count",
    "parse_success_rate", "timezone_aware_count", "timezone_naive_count",
    "minimum_timestamp", "maximum_timestamp", "invalid_value_examples",
]
INVALID_EXAMPLE_COLUMNS = ["column_name", "unique_key", "raw_value"]
CHRONOLOGY_COLUMNS = [
    "unique_key", "created_date_raw", "due_date_raw", "closed_date_raw",
    "resolution_action_updated_date_raw", "violation_type", "severity",
    "recommended_step_7_action",
]


def _timezone_kind(value: object) -> str:
    """Classify one non-null parseable source value as aware or naive."""
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return "invalid"
    if pd.isna(parsed):
        return "invalid"
    return "aware" if parsed.tzinfo is not None else "naive"


def analyze_timestamps(frame: pd.DataFrame, columns: Sequence[str]) -> TimestampAnalysis:
    """Parse configured timestamp columns to temporary UTC Series and profile them."""
    parsed_views: dict[str, pd.Series] = {}
    summaries: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    identifiers = (
        frame["unique_key"].astype("string")
        if "unique_key" in frame
        else pd.Series(frame.index.astype(str), index=frame.index, dtype="string")
    )
    for column in columns:
        if column not in frame:
            continue
        raw = frame[column]
        non_null = raw.notna()
        parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        parsed_views[column] = parsed
        invalid = non_null & parsed.isna()
        kinds = raw.loc[non_null].map(_timezone_kind)
        for index in frame.index[invalid][:10]:
            invalid_rows.append({
                "column_name": column,
                "unique_key": identifiers.loc[index],
                "raw_value": str(raw.loc[index]),
            })
        valid = parsed.dropna()
        examples = [str(value) for value in raw.loc[invalid].astype("string").head(5)]
        non_null_count = int(non_null.sum())
        success = int((non_null & parsed.notna()).sum())
        summaries.append({
            "column_name": column,
            "non_null_count": non_null_count,
            "parse_success_count": success,
            "parse_failure_count": int(invalid.sum()),
            "parse_success_rate": success / non_null_count if non_null_count else 1.0,
            "timezone_aware_count": int(kinds.eq("aware").sum()),
            "timezone_naive_count": int(kinds.eq("naive").sum()),
            "minimum_timestamp": valid.min().isoformat() if len(valid) else None,
            "maximum_timestamp": valid.max().isoformat() if len(valid) else None,
            "invalid_value_examples": " | ".join(examples),
        })
    return TimestampAnalysis(
        parsed=parsed_views,
        summary=pd.DataFrame(summaries, columns=TIMESTAMP_SUMMARY_COLUMNS),
        invalid_examples=pd.DataFrame(invalid_rows, columns=INVALID_EXAMPLE_COLUMNS),
    )


def chronology_violations(
    frame: pd.DataFrame, parsed: dict[str, pd.Series]
) -> pd.DataFrame:
    """Return one evidence row per chronology rule violation."""
    created = parsed.get("created_date", pd.Series(pd.NaT, index=frame.index))
    rules: list[tuple[str, pd.Series, str, str]] = []
    due = parsed.get("due_date", pd.Series(pd.NaT, index=frame.index))
    closed = parsed.get("closed_date", pd.Series(pd.NaT, index=frame.index))
    updated = parsed.get(
        "resolution_action_updated_date", pd.Series(pd.NaT, index=frame.index)
    )
    rules.extend([
        ("due_before_created", due.notna() & created.notna() & due.lt(created), "ERROR",
         "Retain as excluded/quarantine evidence; do not repair automatically."),
        ("closed_before_created", closed.notna() & created.notna() & closed.lt(created), "ERROR",
         "Retain as excluded/quarantine evidence; do not repair automatically."),
        ("resolution_action_before_created",
         updated.notna() & created.notna() & updated.lt(created), "WARNING",
         "Review source timing; do not use the post-creation field as a feature."),
        ("closed_present_created_invalid", frame.get("closed_date", pd.Series(index=frame.index)).notna() & created.isna(),
         "ERROR", "Quarantine row because chronology cannot be established."),
        ("due_present_created_invalid", frame.get("due_date", pd.Series(index=frame.index)).notna() & created.isna(),
         "ERROR", "Quarantine row because chronology cannot be established."),
    ])
    rows: list[dict[str, object]] = []
    raw_names = [
        "created_date", "due_date", "closed_date", "resolution_action_updated_date"
    ]
    for violation_type, mask, severity, action in rules:
        for index in frame.index[mask.fillna(False)]:
            row: dict[str, object] = {
                "unique_key": frame.at[index, "unique_key"] if "unique_key" in frame else index,
                "violation_type": violation_type,
                "severity": severity,
                "recommended_step_7_action": action,
            }
            for name in raw_names:
                row[f"{name}_raw"] = frame.at[index, name] if name in frame else None
            rows.append(row)
    return pd.DataFrame(rows, columns=CHRONOLOGY_COLUMNS)
