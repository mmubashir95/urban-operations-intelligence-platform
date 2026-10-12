"""Write the concise Phase 7.3 median-duration baseline evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_ops.analysis.resolution_duration import DurationAnalysisPopulation
from urban_ops.models.duration_baseline import DurationRegressionMetrics


METRICS_FILENAME = "phase_7_3_median_baseline_metrics.csv"
REPORT_FILENAME = "phase_7_3_median_baseline.md"


def build_duration_baseline_metrics_table(
    *,
    training: DurationAnalysisPopulation,
    validation: DurationAnalysisPopulation,
    fitted_median_hours: float,
    validation_metrics: DurationRegressionMetrics,
) -> pd.DataFrame:
    """Return precision-preserving scalar baseline and evaluation evidence."""
    validation_median = float(validation.eligible_durations.median())
    rows = [
        ("training_total_count", float(len(training.historical)), "complaints"),
        ("training_eligible_count", float(len(training.eligible_durations)), "complaints"),
        ("training_ineligible_count", float(len(training.historical) - len(training.eligible_durations)), "complaints"),
        ("validation_total_count", float(len(validation.historical)), "complaints"),
        ("validation_eligible_count", float(len(validation.eligible_durations)), "complaints"),
        ("validation_ineligible_count", float(len(validation.historical) - len(validation.eligible_durations)), "complaints"),
        ("fitted_training_median_hours", fitted_median_hours, "hours"),
        ("validation_observed_median_hours", validation_median, "hours"),
        ("validation_minus_training_median_hours", validation_median - fitted_median_hours, "hours"),
        ("validation_mae_hours", validation_metrics.mae_hours, "hours"),
        ("validation_rmse_hours", validation_metrics.rmse_hours, "hours"),
    ]
    return pd.DataFrame(rows, columns=("metric", "value", "unit"))


def build_duration_baseline_report(
    *,
    source_path: Path,
    training: DurationAnalysisPopulation,
    validation: DurationAnalysisPopulation,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    fitted_median_hours: float,
    validation_metrics: DurationRegressionMetrics,
) -> str:
    """Build the human-readable Phase 7.3 baseline report from actual results."""
    training_eligible = len(training.eligible_durations)
    validation_eligible = len(validation.eligible_durations)
    validation_median = float(validation.eligible_durations.median())
    return f"""# Phase 7.3 — Median Duration Baseline

## Objective

Establish a deterministic, feature-free benchmark for predicting Graffiti
complaint resolution duration. The model learns one value—the eligible training
median—and predicts it for every eligible validation complaint.

## Source and temporal policy

- Governed cleaned source: `{source_path}`
- Training creation window: `[{training_start.isoformat()}, {training_end.isoformat()})`
- Training inclusive outcome-observation cutoff: `{training_end.isoformat()}`
- Validation creation window: `[{validation_start.isoformat()}, {validation_end.isoformat()})`
- Validation inclusive outcome-observation cutoff: `{validation_end.isoformat()}`
- Test data accessed: No

Split assignment uses `created_date`. Closures after each split-specific cutoff
are unobserved for that historical evaluation and cannot enter its supervised
target array. Validation labels are never used to fit or adjust the baseline.

## Eligibility

| Split | Total | Eligible | Ineligible |
|---|---:|---:|---:|
| Training | {len(training.historical):,} | {training_eligible:,} | {len(training.historical) - training_eligible:,} |
| Validation | {len(validation.historical):,} | {validation_eligible:,} | {len(validation.historical) - validation_eligible:,} |

The evaluation describes only completed outcomes observable at the applicable
cutoff. It does not measure performance on unresolved or otherwise ineligible
complaints.

## Fitted baseline and validation performance

- Fitted training median: {fitted_median_hours:.3f} hours ({fitted_median_hours / 24.0:.3f} days)
- Validation observed median (descriptive only): {validation_median:.3f} hours
- Validation minus training median: {validation_median - fitted_median_hours:.3f} hours
- Validation MAE: {validation_metrics.mae_hours:.3f} hours
- Validation RMSE: {validation_metrics.rmse_hours:.3f} hours

Every validation prediction equals the fitted training median. This is a
constant benchmark, not a feature-based regression model. MAE describes typical
absolute error; RMSE places more weight on large misses.

## Limitations and handoff

Phase 7.2 found a right-skewed training distribution and separately investigated
the concentration near 900 hours. This baseline retains zero, fractional, and
long positive durations without capping or log transformation.

Using only completed outcomes observable by each cutoff introduces selection
bias when unresolved complaints differ systematically from completed ones. A
later-state NYC Open Data snapshot also may not reconstruct historical status
revisions. Phase 7.4 models must use these same split-specific eligible targets,
fit only on training data, and compare validation MAE/RMSE directly against this
frozen benchmark. Survival modelling remains outside this phase.
"""


def write_duration_baseline_outputs(
    *,
    report_root: Path,
    metrics_table: pd.DataFrame,
    report: str,
) -> tuple[Path, Path]:
    """Write the precision-preserving CSV and concise Markdown report."""
    report_root.mkdir(parents=True, exist_ok=True)
    metrics_path = report_root / METRICS_FILENAME
    report_path = report_root / REPORT_FILENAME
    metrics_table.to_csv(metrics_path, index=False, float_format="%.17g")
    report_path.write_text(report, encoding="utf-8")
    return report_path, metrics_path
