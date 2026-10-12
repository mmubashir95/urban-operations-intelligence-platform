"""Run the Phase 7.3 median-duration baseline on frozen train and validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from urban_ops.analysis.resolution_duration import (
    DurationAnalysisPopulation,
    select_duration_population,
)
from urban_ops.analysis.resolution_duration_source import load_verified_cleaned_source
from urban_ops.models.duration_baseline import (
    DurationRegressionMetrics,
    MedianDurationBaseline,
    evaluate_duration_predictions,
)
from urban_ops.models.duration_baseline_reporting import (
    build_duration_baseline_metrics_table,
    build_duration_baseline_report,
    write_duration_baseline_outputs,
)
from urban_ops.splitting.pipeline import load_split_config
from urban_ops.splitting.source import resolve_cleaning_run
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "configs/data/splits.yaml"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/month_2"


@dataclass(frozen=True)
class DurationBaselineWorkflowResult:
    """Phase 7.3 fitted evidence, aligned validation values, and report paths."""

    training: DurationAnalysisPopulation
    validation: DurationAnalysisPopulation
    fitted_median_hours: float
    validation_targets: pd.Series
    validation_predictions: np.ndarray
    validation_metrics: DurationRegressionMetrics
    metrics_table: pd.DataFrame
    report_path: Path
    metrics_path: Path


def run_duration_baseline_workflow(
    *,
    split_config_path: Path | str = DEFAULT_SPLIT_CONFIG,
    cleaning_run_path: Path | None = None,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> DurationBaselineWorkflowResult:
    """Fit the train median, score eligible validation outcomes, and report."""
    config = load_split_config(split_config_path)
    run_path = resolve_cleaning_run(
        processed_root=config.processed_root,
        latest_pointer=config.latest_pointer,
        override=cleaning_run_path,
    )
    source, source_path = load_verified_cleaned_source(run_path)
    training = select_duration_population(
        source,
        creation_start=config.boundaries.train_start,
        creation_end_exclusive=config.boundaries.train_end_exclusive,
        observation_cutoff=config.boundaries.train_end_exclusive,
        timestamp_column=config.timestamp_column,
    )
    validation = select_duration_population(
        source,
        creation_start=config.boundaries.validation_start,
        creation_end_exclusive=config.boundaries.validation_end_exclusive,
        # The validation window end is the frozen historical knowledge boundary;
        # later closures would leak outcomes unavailable at validation time.
        observation_cutoff=config.boundaries.validation_end_exclusive,
        timestamp_column=config.timestamp_column,
    )
    baseline = MedianDurationBaseline().fit(training.eligible_durations)
    validation_targets = validation.eligible_durations
    validation_predictions = baseline.predict(len(validation_targets))
    validation_metrics = evaluate_duration_predictions(
        validation_targets, validation_predictions
    )
    metrics_table = build_duration_baseline_metrics_table(
        training=training,
        validation=validation,
        fitted_median_hours=baseline.median_duration_hours_,
        validation_metrics=validation_metrics,
    )
    report = build_duration_baseline_report(
        source_path=source_path,
        training=training,
        validation=validation,
        training_start=config.boundaries.train_start,
        training_end=config.boundaries.train_end_exclusive,
        validation_start=config.boundaries.validation_start,
        validation_end=config.boundaries.validation_end_exclusive,
        fitted_median_hours=baseline.median_duration_hours_,
        validation_metrics=validation_metrics,
    )
    report_path, metrics_path = write_duration_baseline_outputs(
        report_root=report_root, metrics_table=metrics_table, report=report
    )
    LOGGER.info(
        "Phase 7.3 median baseline complete train_eligible=%s validation_eligible=%s report=%s",
        len(training.eligible_durations), len(validation_targets), report_path,
    )
    return DurationBaselineWorkflowResult(
        training=training,
        validation=validation,
        fitted_median_hours=baseline.median_duration_hours_,
        validation_targets=validation_targets,
        validation_predictions=validation_predictions,
        validation_metrics=validation_metrics,
        metrics_table=metrics_table,
        report_path=report_path,
        metrics_path=metrics_path,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the Phase 7.3 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--cleaning-run", type=Path)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Phase 7.3 workflow from the command line."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_duration_baseline_workflow(
        split_config_path=arguments.split_config,
        cleaning_run_path=arguments.cleaning_run,
        report_root=arguments.report_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
