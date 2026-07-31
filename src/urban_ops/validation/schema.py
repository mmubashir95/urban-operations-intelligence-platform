"""Validate raw DataFrame structure without coercing or changing source values."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pandas as pd

from urban_ops.data.metadata import ExtractionMetadata
from urban_ops.validation.models import Severity, ValidationCheck
from urban_ops.validation.severity import make_check


SEMANTIC_TYPES = {
    "unique_key": "identifier_string",
    "created_date": "timestamp",
    "closed_date": "timestamp",
    "due_date": "timestamp",
    "resolution_action_updated_date": "timestamp",
    "latitude": "numeric_coordinate",
    "longitude": "numeric_coordinate",
    "incident_zip": "postal_code_string",
}


def validate_schema(
    frame: object,
    *,
    metadata: ExtractionMetadata,
    required_columns: Sequence[str],
    require_column_order: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[ValidationCheck]]:
    """Return schema checks and a column profile for one raw DataFrame.

    Raises:
        TypeError: If ``frame`` is not a pandas DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Raw validation input must be a pandas DataFrame.")
    total = len(frame)
    actual = list(frame.columns)
    required = list(required_columns)
    selected = list(metadata.selected_source_columns)
    missing = [column for column in required if column not in actual]
    unexpected = [column for column in actual if column not in required]
    duplicate_names = sorted(
        column for column, count in Counter(actual).items() if count > 1
    )
    identifier_safe = False
    if "unique_key" in actual and actual.count("unique_key") == 1:
        identifier_dtype = frame["unique_key"].dtype
        identifier_safe = bool(
            pd.api.types.is_string_dtype(identifier_dtype)
            or pd.api.types.is_object_dtype(identifier_dtype)
        )
    checks = [
        make_check(
            check_id="schema.non_empty", area="schema", check_name="Non-empty dataset",
            severity=Severity.CRITICAL, passed=total > 0, observed_value=total,
            expected_value="> 0", affected_rows=0 if total else 1, total_rows=total,
            message="The raw extraction must contain at least one record.",
            recommended_action="Re-run or investigate Step 5 extraction.",
        ),
        make_check(
            check_id="schema.unique_column_names", area="schema",
            check_name="Unique column names", severity=Severity.CRITICAL,
            passed=not duplicate_names, observed_value="|".join(duplicate_names) or "none",
            expected_value="none", affected_rows=len(duplicate_names), total_rows=total,
            message="Raw column names must be unique.",
            recommended_action="Correct the ingestion schema before cleaning.",
        ),
        make_check(
            check_id="schema.required_columns", area="schema",
            check_name="Required source columns", severity=Severity.CRITICAL,
            passed=not missing, observed_value="|".join(missing) or "none",
            expected_value="none missing", affected_rows=len(missing), total_rows=total,
            message="All governed raw source fields must be present.",
            recommended_action="Reconcile source schema and ingestion selection.",
        ),
        make_check(
            check_id="schema.metadata_columns", area="schema",
            check_name="Stored columns match metadata", severity=Severity.CRITICAL,
            passed=set(actual) == set(selected),
            observed_value="|".join(actual), expected_value="|".join(selected),
            affected_rows=len(set(actual).symmetric_difference(selected)), total_rows=total,
            message="Metadata-selected fields must equal stored raw fields.",
            recommended_action="Treat the raw run as inconsistent and re-ingest.",
        ),
        make_check(
            check_id="schema.column_order", area="schema",
            check_name="Column order matches metadata", severity=Severity.ERROR,
            passed=(not require_column_order or actual == selected),
            observed_value="|".join(actual), expected_value="|".join(selected),
            affected_rows=0 if actual == selected else 1, total_rows=total,
            message="Column order is part of the configured raw contract.",
            recommended_action="Preserve configured order during Step 7 reads.",
        ),
        make_check(
            check_id="schema.unexpected_columns", area="schema",
            check_name="Unexpected source columns", severity=Severity.WARNING,
            passed=not unexpected, observed_value="|".join(unexpected) or "none",
            expected_value="none", affected_rows=len(unexpected), total_rows=total,
            message="Unexpected source fields may indicate schema drift.",
            recommended_action="Review new fields before any cleaning or feature use.",
        ),
        make_check(
            check_id="schema.metadata_row_count", area="schema",
            check_name="Row count matches metadata", severity=Severity.CRITICAL,
            passed=total == metadata.retrieved_row_count, observed_value=total,
            expected_value=metadata.retrieved_row_count,
            affected_rows=abs(total - metadata.retrieved_row_count), total_rows=total,
            message="Parquet row count must reconcile to extraction metadata.",
            recommended_action="Reject the inconsistent raw run and re-ingest.",
        ),
        make_check(
            check_id="schema.identifier_representation", area="schema",
            check_name="Identifier represented as string", severity=Severity.ERROR,
            passed=identifier_safe,
            observed_value=str(frame["unique_key"].dtype) if "unique_key" in actual and actual.count("unique_key") == 1 else "missing or duplicated",
            expected_value="string or object preserving source digits",
            affected_rows=0 if identifier_safe else total, total_rows=total,
            message="Complaint identifiers must not be stored as lossy numeric values.",
            recommended_action="Preserve the source identifier as a string in Step 7.",
        ),
    ]
    schema_rows = [
        {
            "column_name": column,
            "position": position,
            "raw_dtype": str(frame.dtypes.iloc[position]),
            "required": column in required,
            "selected_in_metadata": column in selected,
            "semantic_type": SEMANTIC_TYPES.get(column, "string_or_nullable_source_value"),
            "status": "PASS" if column in selected else "WARN",
        }
        for position, column in enumerate(actual)
    ]
    profile_rows = []
    timestamp_columns = {
        "created_date", "closed_date", "due_date", "resolution_action_updated_date"
    }
    numeric_columns = {"latitude", "longitude"}
    for position, column in enumerate(actual):
        # Select by position so duplicate-name validation can still finish and
        # report all physical columns instead of failing during profiling.
        values = frame.iloc[:, position]
        non_null = values.dropna()
        if column in timestamp_columns:
            parsed = pd.to_datetime(non_null, errors="coerce", utc=True, format="mixed")
            parseable = int(parsed.notna().sum())
        elif column in numeric_columns:
            parsed = pd.to_numeric(non_null, errors="coerce")
            parseable = int(parsed.notna().sum())
        else:
            parseable = len(non_null)
        samples = [str(value) for value in non_null.astype("string").drop_duplicates().head(3)]
        profile_rows.append({
            "column_name": column,
            "raw_dtype": str(values.dtype),
            "non_null_count": len(non_null),
            "null_count": int(values.isna().sum()),
            "distinct_count": int(non_null.nunique(dropna=True)),
            "sample_values": " | ".join(samples),
            "expected_semantic_type": SEMANTIC_TYPES.get(
                column, "string_or_nullable_source_value"
            ),
            "parseable_count": parseable,
            "parse_failure_count": len(non_null) - parseable,
            "validation_status": "PASS" if parseable == len(non_null) else "WARN",
        })
    return pd.DataFrame(schema_rows), pd.DataFrame(profile_rows), checks
