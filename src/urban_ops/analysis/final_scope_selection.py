"""Reconcile general scope feasibility with the authoritative temporal decision.

Notebook 04 produces general candidate feasibility evidence. Notebook 05 owns
the final date-range decision because it additionally evaluates volatility,
outcome maturity, complete calendar years, and recent-period relevance. This
module validates the handoff without reproducing either scoring algorithm.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


GENERAL_REQUIRED_COLUMNS = {
    "agency",
    "start_date",
    "end_date",
    "total_records",
    "eligible_target_count",
    "missed_target_rate",
    "due_date_coverage",
    "closed_date_coverage",
}
TEMPORAL_CANDIDATE_REQUIRED_COLUMNS = {
    "start_date",
    "end_date",
    "total_records",
    "eligible_records",
    "missed_target_rate",
    "due_date_coverage",
    "closed_date_coverage",
    "selected",
}
TEMPORAL_SUMMARY_REQUIRED_COLUMNS = {
    "selected_agency",
    "selected_complaint_type",
    "selected_start_date",
    "selected_end_date",
    "total_records",
    "eligible_records",
    "missed_target_rate",
    "due_date_coverage",
    "closed_date_coverage",
    "extraction_timestamp",
}
SCOPE_IDENTIFIER_COLUMNS = [
    "selected_agency",
    "selected_complaint_type",
    "selected_start_date",
    "selected_end_date",
    "extraction_timestamp",
]
SCOPE_COUNT_COLUMNS = ["total_records", "eligible_records"]
SCOPE_RATE_COLUMNS = [
    "missed_target_rate",
    "due_date_coverage",
    "closed_date_coverage",
]


def _validate_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    frame_name: str,
) -> None:
    """Raise a useful error when an input artifact is missing required columns."""
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _normalized_date(value: object) -> str:
    """Return an ISO calendar date for cross-artifact comparisons."""
    parsed = pd.to_datetime(value, errors="raise")
    return parsed.date().isoformat()


def _selected_mask(series: pd.Series) -> pd.Series:
    """Interpret CSV boolean values without treating non-empty strings as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    invalid = normalized.dropna().loc[~normalized.dropna().isin({"true", "false"})]
    if not invalid.empty:
        raise ValueError("Temporal candidate selected values must be true or false.")
    return normalized.eq("true").fillna(False)


def reconcile_final_scope(
    general_candidates: pd.DataFrame,
    temporal_candidates: pd.DataFrame,
    temporal_selected_scope: pd.DataFrame,
    *,
    selected_complaint_types: Sequence[str],
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> tuple[pd.Series, pd.Series]:
    """Return the one general candidate chosen by temporal analysis.

    The function does not rank candidates. It proves that Notebook 05 selected
    exactly one temporal candidate, that its summary is internally consistent,
    and that the decision matches exactly one Notebook 04 feasibility row.

    Raises:
        ValueError: If identifiers, dates, counts, rates, or selected-row
            cardinality disagree across the artifacts.
    """
    _validate_columns(
        general_candidates,
        GENERAL_REQUIRED_COLUMNS,
        frame_name="General candidate scorecard",
    )
    _validate_columns(
        temporal_candidates,
        TEMPORAL_CANDIDATE_REQUIRED_COLUMNS,
        frame_name="Temporal candidate comparison",
    )
    _validate_columns(
        temporal_selected_scope,
        TEMPORAL_SUMMARY_REQUIRED_COLUMNS,
        frame_name="Temporal selected-scope summary",
    )
    if len(temporal_selected_scope) != 1:
        raise ValueError("Expected exactly one temporal selected-scope summary row.")

    selected_temporal_rows = temporal_candidates.loc[
        _selected_mask(temporal_candidates["selected"])
    ]
    if len(selected_temporal_rows) != 1:
        raise ValueError("Temporal comparison must contain exactly one selected row.")

    temporal_candidate = selected_temporal_rows.iloc[0]
    temporal_summary = temporal_selected_scope.iloc[0]
    temporal_start = _normalized_date(temporal_summary["selected_start_date"])
    temporal_end = _normalized_date(temporal_summary["selected_end_date"])
    if temporal_start != _normalized_date(temporal_candidate["start_date"]):
        raise ValueError("Temporal summary start date disagrees with its selected candidate.")
    if temporal_end != _normalized_date(temporal_candidate["end_date"]):
        raise ValueError("Temporal summary end date disagrees with its selected candidate.")
    for column in SCOPE_COUNT_COLUMNS:
        if int(temporal_candidate[column]) != int(temporal_summary[column]):
            raise ValueError(
                f"Temporal selected candidate disagrees with its summary for {column}."
            )
    for column in SCOPE_RATE_COLUMNS:
        if not np.isclose(
            float(temporal_candidate[column]),
            float(temporal_summary[column]),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise ValueError(
                f"Temporal selected candidate disagrees with its summary for {column}."
            )

    selected_types = [str(value) for value in selected_complaint_types]
    temporal_complaint_type = str(temporal_summary["selected_complaint_type"])
    if selected_types != [temporal_complaint_type]:
        raise ValueError(
            "Temporal complaint type disagrees with the general feasibility selection."
        )

    general_start = general_candidates["start_date"].map(_normalized_date)
    general_end = general_candidates["end_date"].map(_normalized_date)
    matching_candidates = general_candidates.loc[
        general_start.eq(temporal_start)
        & general_end.eq(temporal_end)
        & general_candidates["agency"].astype(str).eq(
            str(temporal_summary["selected_agency"])
        )
    ]
    if len(matching_candidates) != 1:
        raise ValueError(
            "Temporal selection must match exactly one candidate in the general scorecard."
        )
    general_candidate = matching_candidates.iloc[0]

    count_pairs = {
        "total_records": "total_records",
        "eligible_target_count": "eligible_records",
    }
    for general_column, temporal_column in count_pairs.items():
        if int(general_candidate[general_column]) != int(
            temporal_summary[temporal_column]
        ):
            raise ValueError(
                f"Selected-scope count mismatch for {temporal_column}."
            )

    for rate_column in SCOPE_RATE_COLUMNS:
        if not np.isclose(
            float(general_candidate[rate_column]),
            float(temporal_summary[rate_column]),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise ValueError(f"Selected-scope rate mismatch for {rate_column}.")
    return general_candidate, temporal_summary


def validate_selected_scope_outputs(
    scope_selection_summary: pd.DataFrame,
    temporal_selected_scope: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-12,
) -> None:
    """Validate that report 04 and report 05 expose one identical final scope."""
    for frame, name in (
        (scope_selection_summary, "Scope-selection summary"),
        (temporal_selected_scope, "Temporal selected-scope summary"),
    ):
        _validate_columns(frame, TEMPORAL_SUMMARY_REQUIRED_COLUMNS, frame_name=name)
        if len(frame) != 1:
            raise ValueError(f"{name} must contain exactly one row.")

    scope_row = scope_selection_summary.iloc[0]
    temporal_row = temporal_selected_scope.iloc[0]
    for column in SCOPE_IDENTIFIER_COLUMNS:
        scope_value = str(scope_row[column])
        temporal_value = str(temporal_row[column])
        if column in {"selected_start_date", "selected_end_date"}:
            scope_value = _normalized_date(scope_value)
            temporal_value = _normalized_date(temporal_value)
        if scope_value != temporal_value:
            raise ValueError(f"Selected-scope identifier mismatch for {column}.")
    for column in SCOPE_COUNT_COLUMNS:
        if int(scope_row[column]) != int(temporal_row[column]):
            raise ValueError(f"Selected-scope count mismatch for {column}.")
    for column in SCOPE_RATE_COLUMNS:
        if not np.isclose(
            float(scope_row[column]),
            float(temporal_row[column]),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise ValueError(f"Selected-scope rate mismatch for {column}.")
