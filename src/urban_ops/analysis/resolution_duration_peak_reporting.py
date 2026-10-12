"""Persist and interpret the focused Phase 7.2 900-hour investigation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "urban_ops_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urban_ops.analysis.resolution_duration_peak import (
    DETAIL_LOWER_HOURS,
    DETAIL_UPPER_HOURS,
    PEAK_LOWER_HOURS,
    PEAK_UPPER_HOURS,
    PeakInvestigationEvidence,
)
from urban_ops.features.target import RESOLUTION_TIME_TARGET_NAME


PEAK_REPORT_FILENAME = "phase_7_2_900_hour_investigation.md"
PEAK_INTERVALS_FILENAME = "phase_7_2_900_hour_intervals.csv"
PEAK_TYPES_FILENAME = "phase_7_2_900_hour_complaint_types.csv"
PEAK_PATTERNS_FILENAME = "phase_7_2_900_hour_closure_patterns.csv"
PEAK_FIGURE_FILENAME = "phase_7_2_900_hour_detail.png"


def write_peak_detail_figure(figure_root: Path, eligible: pd.DataFrame) -> Path:
    """Write a deterministic 10-hour histogram over the focused 600–1,200 range."""
    figure_root.mkdir(parents=True, exist_ok=True)
    path = figure_root / PEAK_FIGURE_FILENAME
    values = eligible[RESOLUTION_TIME_TARGET_NAME]
    bins = np.arange(DETAIL_LOWER_HOURS, DETAIL_UPPER_HOURS + 10.0, 10.0)
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins)
    plt.axvspan(
        PEAK_LOWER_HOURS, PEAK_UPPER_HOURS, color="tab:orange", alpha=0.2,
        label="Inclusive investigation interval (850–950 hours)",
    )
    plt.title("Resolution-duration detail around 900 hours")
    plt.xlabel("Resolution duration (hours)")
    plt.ylabel("Complaint count")
    plt.xlim(DETAIL_LOWER_HOURS, DETAIL_UPPER_HOURS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def _value(table: pd.DataFrame, key_column: str, key: str) -> float:
    """Return one numeric value from a two-column evidence table."""
    selected = table.loc[table[key_column].eq(key), "value"]
    if len(selected) != 1:
        raise ValueError(f"Expected one value for {key!r}.")
    return float(selected.iloc[0])


def _markdown_rows(
    table: pd.DataFrame, columns: tuple[str, ...], *, limit: int | None = None
) -> str:
    """Render a compact deterministic Markdown table without extra dependencies."""
    selected = table.loc[:, list(columns)]
    if limit is not None:
        selected = selected.head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(
            f"{value:.3f}" if isinstance(value, float) else str(value)
            for value in row
        ) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_peak_investigation_report(
    *,
    evidence: PeakInvestigationEvidence,
    source_path: Path,
    train_start: pd.Timestamp,
    observation_cutoff: pd.Timestamp,
) -> str:
    """Build an evidence-backed report separating observations from hypotheses."""
    summary = evidence.summary
    eligible_count = int(_value(summary, "metric", "eligible_complaint_count"))
    peak_count = int(_value(summary, "metric", "peak_complaint_count"))
    peak_percentage = _value(summary, "metric", "peak_percentage_of_eligible")
    neighboring = evidence.intervals.loc[
        evidence.intervals["analysis_type"].eq("neighbor_interval")
    ]
    sensitivity = evidence.intervals.loc[
        evidence.intervals["analysis_type"].eq("histogram_sensitivity")
        & evidence.intervals["lower_hours"].ge(800)
        & evidence.intervals["upper_hours"].le(1000)
    ]
    peak_types = evidence.complaint_types.loc[evidence.complaint_types["peak_count"].gt(0)]
    timing = evidence.timing_patterns
    repeated_closures = timing.loc[timing["pattern_type"].eq("identical_closed_timestamp")]
    quality = evidence.data_quality
    duration_repetition = int(
        _value(quality, "check", "maximum_exact_duration_repetition")
    )
    closure_repetition = int(
        _value(quality, "check", "maximum_identical_closure_timestamp_repetition")
    )
    return f"""# Phase 7.2 — Investigation of the 900-Hour Concentration

## Objective and population

Investigate the apparent concentration near 900 hours without modifying the
Phase 7.1 target or training any model.

- Governed cleaned source: `{source_path}`
- Training creation window: `[{train_start.isoformat()}, {observation_cutoff.isoformat()})`
- Inclusive closure-observation cutoff: `{observation_cutoff.isoformat()}`
- Eligible training complaints: {eligible_count:,}

## Observed findings

### Primary 850–950-hour interval

- Complaints: {peak_count:,}
- Percentage of eligible complaints: {peak_percentage:.3f}%
- Minimum / median / maximum: {_value(summary, 'metric', 'peak_minimum_hours'):.3f} /
  {_value(summary, 'metric', 'peak_median_hours'):.3f} /
  {_value(summary, 'metric', 'peak_maximum_hours'):.3f} hours
- Unique complaint identifiers: {int(_value(summary, 'metric', 'peak_unique_identifier_count')):,}

The primary question uses the inclusive interval `[850, 950]`. Neighboring
comparisons use non-overlapping half-open bins, so an exact 950-hour duration is
in the primary summary and the separate `[950, 1050)` comparison bin.

### Neighboring 100-hour intervals

{_markdown_rows(neighboring, ('label', 'complaint_count', 'percentage_of_eligible'))}

### Histogram bin-width sensitivity

The intervals CSV contains all fixed-origin 10-, 25-, and 50-hour bins from
600 to 1,200 hours. The bins spanning 800–1,000 hours are:

{_markdown_rows(sensitivity, ('label', 'bin_width_hours', 'complaint_count', 'percentage_of_eligible'))}

Persistence across widths supports a genuine concentration; large changes
confined to one width indicate that plotting choices materially affect its
appearance. The detailed figure uses 10-hour bins.

### Complaint-type representation

{_markdown_rows(peak_types, ('complaint_type', 'eligible_count', 'peak_count', 'percentage_of_peak', 'within_type_peak_percentage', 'representation_ratio'), limit=15)}

`percentage_of_peak` describes composition. `within_type_peak_percentage`
describes concentration within a type. `representation_ratio` compares peak
share with overall eligible-population share. Small counts must not be treated
as stable evidence merely because their ratio is large.

### Creation and closure timing

The closure-pattern CSV contains peak creation month/day, closure month/hour/date,
repeated exact closure timestamps, and 35–40-day duration bands. Repeated exact
closure timestamps detected: {len(repeated_closures):,}; maximum complaints at
one identical closure timestamp: {closure_repetition:,}.

### Focused data-quality checks

{_markdown_rows(quality, ('check', 'value', 'unit'))}

The largest exact-duration repetition is {duration_repetition:,}. The target is
recalculated from UTC timestamps without rounding; the reported maximum
recalculation difference tests that contract directly.

## Possible explanations requiring more evidence

- Persistence across bin widths may indicate a real operational duration band.
- Complaint-type overrepresentation may indicate type-specific processing.
- Repeated closure dates or exact timestamps may be consistent with batch or
  administrative closure, but do not prove that mechanism.
- Concentration in 35–40-day bands may reflect scheduling or operational
  practice, but the later-state snapshot cannot distinguish administrative
  record closure from field resolution time.

These are hypotheses, not causal findings. NYC Open Data may contain later
status revisions that cannot reconstruct every historical state.

## Regression implications

Phase 7.3 should retain these valid observed durations in its median baseline.
Later regression work should review error by complaint type and near this band,
while MAE and RMSE should both be reported because the broader distribution is
right-skewed. No capping, transformation, feature engineering, or survival
modelling decision is made here.

## Outputs

- [{PEAK_INTERVALS_FILENAME}]({PEAK_INTERVALS_FILENAME})
- [{PEAK_TYPES_FILENAME}]({PEAK_TYPES_FILENAME})
- [{PEAK_PATTERNS_FILENAME}]({PEAK_PATTERNS_FILENAME})
- ![Detailed 900-hour histogram](figures/{PEAK_FIGURE_FILENAME})
"""


def write_peak_investigation_outputs(
    *,
    report_root: Path,
    evidence: PeakInvestigationEvidence,
    source_path: Path,
    train_start: pd.Timestamp,
    observation_cutoff: pd.Timestamp,
) -> tuple[Path, Path, Path, Path, Path]:
    """Write the three focused CSVs, Markdown report, and detailed figure."""
    report_root.mkdir(parents=True, exist_ok=True)
    intervals_path = report_root / PEAK_INTERVALS_FILENAME
    types_path = report_root / PEAK_TYPES_FILENAME
    patterns_path = report_root / PEAK_PATTERNS_FILENAME
    report_path = report_root / PEAK_REPORT_FILENAME
    evidence.intervals.to_csv(intervals_path, index=False, float_format="%.17g")
    evidence.complaint_types.to_csv(types_path, index=False, float_format="%.17g")
    evidence.timing_patterns.to_csv(patterns_path, index=False, float_format="%.17g")
    figure_path = write_peak_detail_figure(report_root / "figures", evidence.eligible)
    report_path.write_text(
        build_peak_investigation_report(
            evidence=evidence,
            source_path=source_path,
            train_start=train_start,
            observation_cutoff=observation_cutoff,
        ),
        encoding="utf-8",
    )
    return report_path, intervals_path, types_path, patterns_path, figure_path
