"""Write deterministic Phase 7.2 duration figures and evidence reports."""

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

from urban_ops.analysis.resolution_duration import DurationAnalysisPopulation
from urban_ops.features.target import RESOLUTION_TIME_CATEGORY_NAME


STATISTICS_FILENAME = "phase_7_2_duration_statistics.csv"
ELIGIBILITY_FILENAME = "phase_7_2_eligibility_summary.csv"
REPORT_FILENAME = "phase_7_2_duration_distribution.md"
HISTOGRAM_FILENAME = "phase_7_2_duration_histogram.png"
LOG_HISTOGRAM_FILENAME = "phase_7_2_log_duration_histogram.png"
HISTOGRAM_BINS = 50


def _save_figure(path: Path, *, title: str, xlabel: str) -> None:
    """Label, save, and close the active Matplotlib figure."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Complaint count")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def write_duration_figures(figure_root: Path, durations: pd.Series) -> tuple[Path, Path]:
    """Write raw-hour and exploratory log1p histograms with fixed bin counts."""
    values = durations.to_numpy(dtype="float64")
    if values.size == 0:
        raise ValueError("Cannot plot an empty duration distribution.")
    figure_root.mkdir(parents=True, exist_ok=True)
    raw_path = figure_root / HISTOGRAM_FILENAME
    log_path = figure_root / LOG_HISTOGRAM_FILENAME

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=HISTOGRAM_BINS)
    _save_figure(
        raw_path,
        title="Historical training resolution-duration distribution",
        xlabel="Resolution duration (hours)",
    )

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(values), bins=HISTOGRAM_BINS)
    _save_figure(
        log_path,
        title="Log-transformed historical resolution-duration distribution",
        xlabel="log1p(resolution duration in hours)",
    )
    return raw_path, log_path


def _metric(statistics: pd.DataFrame, name: str) -> float:
    """Return one named statistic from the long-form statistics table."""
    selected = statistics.loc[statistics["metric"].eq(name), "value"]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one statistic named {name!r}.")
    return float(selected.iloc[0])


def build_duration_report(
    *,
    population: DurationAnalysisPopulation,
    statistics: pd.DataFrame,
    eligibility: pd.DataFrame,
    common_durations: pd.DataFrame,
    source_path: Path,
    train_start: pd.Timestamp,
    observation_cutoff: pd.Timestamp,
) -> str:
    """Build a result-backed Markdown interpretation without causal claims."""
    total = len(population.historical)
    eligible_count = int(_metric(statistics, "count"))
    ineligible_count = total - eligible_count
    mean, median = _metric(statistics, "mean"), _metric(statistics, "median")
    skewness = _metric(statistics, "skewness")
    zero_count = int(_metric(statistics, "zero_duration_count"))
    invalid_count = int(
        (population.targets[RESOLUTION_TIME_CATEGORY_NAME] == "invalid").sum()
    )
    not_observed_count = int(
        (population.targets[RESOLUTION_TIME_CATEGORY_NAME] == "not_observed").sum()
    )
    reasons = eligibility.loc[
        eligibility["summary_type"].eq("exclusion_reason")
        & ~eligibility["name"].eq("eligible")
    ]
    reason_lines = "\n".join(
        f"- `{row.name}`: {int(row.count):,}"
        for row in reasons.itertuples(index=False)
    ) or "- None"
    common_lines = "\n".join(
        f"- {row.duration_hours:.17g} hours: {int(row.count):,} complaints"
        for row in common_durations.itertuples(index=False)
    ) or "- No duration value occurs more than once."
    shape = (
        "positive/right skew is present"
        if skewness > 0
        else "positive/right skew is not indicated by the sample skewness"
    )
    return f"""# Phase 7.2 — Resolution Duration Distribution

## Objective

Describe observed NYC 311 complaint resolution durations in the frozen training
population. This phase performs no model training and does not alter targets.

## Data source and temporal policy

- Governed cleaned dataset: `{source_path}`
- Training creation window: `[{train_start.isoformat()}, {observation_cutoff.isoformat()})`
- Inclusive outcome-observation cutoff: `{observation_cutoff.isoformat()}`
- Total complaints created in the training window: {total:,}
- Source rows created at or after the training cutoff (excluded before eligibility): {population.after_training_count:,}

Complaints are selected using creation time only. A complaint created before the
cutoff but closed after it is retained in the historical population and marked
`resolution_not_observed_by_cutoff`; it cannot contribute a completed duration.

## Eligibility summary

- Eligible completed durations: {eligible_count:,} ({eligible_count / total:.2%})
- Ineligible: {ineligible_count:,} ({ineligible_count / total:.2%})
- Not observed by the cutoff or lacking observable closure: {not_observed_count:,}
- Invalid records: {invalid_count:,}

### Exclusion breakdown

{reason_lines}

The complete reason/category counts and source-window exclusions are in
[{ELIGIBILITY_FILENAME}]({ELIGIBILITY_FILENAME}). Counts reconcile within the
historical population; later-created complaints are source exclusions, not
historical unresolved complaints.

## Descriptive statistics

| Metric | Value |
|---|---:|
| Count | {eligible_count:,} |
| Mean | {mean:.3f} hours |
| Median / P50 | {median:.3f} hours |
| Standard deviation | {_metric(statistics, 'standard_deviation'):.3f} hours |
| Minimum | {_metric(statistics, 'minimum'):.3f} hours |
| P25 | {_metric(statistics, 'p25'):.3f} hours |
| P75 | {_metric(statistics, 'p75'):.3f} hours |
| P90 | {_metric(statistics, 'p90'):.3f} hours |
| P95 | {_metric(statistics, 'p95'):.3f} hours |
| P99 | {_metric(statistics, 'p99'):.3f} hours |
| Maximum | {_metric(statistics, 'maximum'):.3f} hours |
| Zero-hour count | {zero_count:,} ({zero_count / eligible_count:.2%}) |
| Skewness | {skewness:.3f} |

The CSV preserves calculation precision; rounding above is presentation-only.
The mean-minus-median difference is {mean - median:.3f} hours, and {shape}.

## Repeated values and long-duration tail

Most common repeated exact durations (descriptive only):

{common_lines}

P90/P95/P99, the maximum, and the raw-versus-log histogram provide evidence
about the upper tail. No observations were removed, capped, winsorized, or
imputed, and no causal explanation is assigned to repeated values.

## Figures

![Duration histogram](figures/{HISTOGRAM_FILENAME})

![Log duration histogram](figures/{LOG_HISTOGRAM_FILENAME})

The second figure uses `log1p(hours)`, which retains zero-hour observations. It
is exploratory and does not overwrite the Phase 7.1 target.

## Data limitations

This analysis describes completed outcomes observable at the training cutoff.
A later-state NYC Open Data snapshot may not fully reconstruct historical
status revisions. Excluding unobserved outcomes can create completion/selection
bias: ordinary regression learns from complaints completed by the cutoff, not
all complaints created in the window. Survival modelling remains a Phase 7.6
decision.

## Implications for later phases

- The training median is a robust constant benchmark for Phase 7.3 because it
  minimizes absolute error on the observed training durations.
- The measured skewness and log histogram indicate whether a `log1p` target is
  worth evaluating in Phase 7.4; no transformation is applied here.
- Long durations can influence squared-error training and RMSE more strongly
  than typical observations. They remain valid targets unless a later governed
  policy says otherwise.
- Phase 7.5 should report MAE for typical absolute error and RMSE for sensitivity
  to large errors. Neither metric is calculated as model evaluation here.
"""


def write_duration_reports(
    *,
    report_root: Path,
    population: DurationAnalysisPopulation,
    statistics: pd.DataFrame,
    eligibility: pd.DataFrame,
    common_durations: pd.DataFrame,
    source_path: Path,
    train_start: pd.Timestamp,
    observation_cutoff: pd.Timestamp,
) -> tuple[Path, Path, Path, tuple[Path, Path]]:
    """Write precision-preserving CSVs, figures, and the Markdown report."""
    report_root.mkdir(parents=True, exist_ok=True)
    statistics_path = report_root / STATISTICS_FILENAME
    eligibility_path = report_root / ELIGIBILITY_FILENAME
    report_path = report_root / REPORT_FILENAME
    statistics.to_csv(statistics_path, index=False, float_format="%.17g")
    eligibility.to_csv(eligibility_path, index=False, float_format="%.17g")
    figures = write_duration_figures(report_root / "figures", population.eligible_durations)
    report_path.write_text(
        build_duration_report(
            population=population,
            statistics=statistics,
            eligibility=eligibility,
            common_durations=common_durations,
            source_path=source_path,
            train_start=train_start,
            observation_cutoff=observation_cutoff,
        ),
        encoding="utf-8",
    )
    return report_path, statistics_path, eligibility_path, figures
