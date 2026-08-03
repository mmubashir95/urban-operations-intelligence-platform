"""Create UTC-aware processed timestamps while preserving raw-source auditability."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from urban_ops.cleaning.models import CleaningAction


TIMESTAMP_SUMMARY_COLUMNS = [
    "column_name", "input_non_null_count", "parsed_count", "null_count",
    "parse_failure_count", "timezone", "invalid_policy", "imputed_count",
]


def clean_timestamps(
    frame: pd.DataFrame, *, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[CleaningAction, ...]]:
    """Return a copy with configured timestamp columns parsed as UTC.

    Invalid non-null strings become ``NaT`` only in the processed copy and are
    identified by a companion parse-failure audit flag. No value is imputed or
    adjusted for chronology.
    """
    result = frame.copy(deep=True)
    rows: list[dict[str, object]] = []
    actions: list[CleaningAction] = []
    for column in columns:
        if column not in result:
            raise ValueError(f"Configured timestamp column is missing: {column}")
        raw = result[column]
        parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        failures = raw.notna() & parsed.isna()
        result[column] = parsed
        result[f"{column}_parse_failed"] = failures.astype(bool)
        rows.append({
            "column_name": column,
            "input_non_null_count": int(raw.notna().sum()),
            "parsed_count": int(parsed.notna().sum()),
            "null_count": int(raw.isna().sum()),
            "parse_failure_count": int(failures.sum()),
            "timezone": "UTC",
            "invalid_policy": "preserve_in_raw_and_exclude_from_target",
            "imputed_count": 0,
        })
        actions.append(CleaningAction(
            action_id=f"timestamp.parse.{column}",
            cleaning_area="timestamp",
            source_column=column,
            action_type="PARSE_UTC",
            input_value="source string or null",
            output_value="datetime64[ns, UTC] or NaT",
            affected_rows=int(parsed.notna().sum()),
            reason="Use typed UTC timestamps in processed outputs without changing raw data.",
            source_rule="docs/cleaning_policy.md#timestamp-policy",
            governance_status="APPROVED",
            applied=True,
        ))
    return (
        result,
        pd.DataFrame(rows, columns=TIMESTAMP_SUMMARY_COLUMNS),
        tuple(actions),
    )
