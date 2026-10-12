"""CLI orchestration for the Phase 7.2 resolution-duration analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

from urban_ops.analysis.resolution_duration import (
    DurationAnalysisPopulation,
    build_eligibility_summary,
    calculate_duration_statistics,
    most_common_durations,
    select_training_duration_population,
)
from urban_ops.analysis.resolution_duration_reporting import write_duration_reports
from urban_ops.analysis.resolution_duration_peak import (
    PeakInvestigationEvidence,
    investigate_900_hour_concentration,
)
from urban_ops.analysis.resolution_duration_peak_reporting import (
    write_peak_investigation_outputs,
)
from urban_ops.analysis.resolution_duration_source import load_verified_cleaned_source
from urban_ops.splitting.pipeline import load_split_config
from urban_ops.splitting.source import resolve_cleaning_run
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "configs/data/splits.yaml"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/month_2"
@dataclass(frozen=True)
class DurationAnalysisResult:
    """In-memory Phase 7.2 evidence and paths written by one run."""

    population: DurationAnalysisPopulation
    statistics: pd.DataFrame
    eligibility: pd.DataFrame
    report_path: Path
    statistics_path: Path
    eligibility_path: Path
    figure_paths: tuple[Path, Path]


@dataclass(frozen=True)
class PeakInvestigationResult:
    """In-memory 900-hour evidence and its five persisted output paths."""

    population: DurationAnalysisPopulation
    evidence: PeakInvestigationEvidence
    report_path: Path
    intervals_path: Path
    complaint_types_path: Path
    closure_patterns_path: Path
    figure_path: Path


def run_resolution_duration_analysis(
    *,
    split_config_path: Path | str = DEFAULT_SPLIT_CONFIG,
    cleaning_run_path: Path | None = None,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> DurationAnalysisResult:
    """Verify inputs, analyze frozen training durations, and write Phase 7.2 outputs."""
    config = load_split_config(split_config_path)
    run_path = resolve_cleaning_run(
        processed_root=config.processed_root,
        latest_pointer=config.latest_pointer,
        override=cleaning_run_path,
    )
    source, source_path = load_verified_cleaned_source(run_path)
    population = select_training_duration_population(
        source,
        boundaries=config.boundaries,
        timestamp_column=config.timestamp_column,
    )
    statistics = calculate_duration_statistics(population.eligible_durations)
    eligibility = build_eligibility_summary(population)
    common = most_common_durations(population.eligible_durations)
    report, statistics_path, eligibility_path, figures = write_duration_reports(
        report_root=report_root,
        population=population,
        statistics=statistics,
        eligibility=eligibility,
        common_durations=common,
        source_path=source_path,
        train_start=config.boundaries.train_start,
        observation_cutoff=config.boundaries.train_end_exclusive,
    )
    LOGGER.info(
        "Phase 7.2 complete historical=%s eligible=%s report=%s",
        len(population.historical),
        len(population.eligible_durations),
        report,
    )
    return DurationAnalysisResult(
        population=population,
        statistics=statistics,
        eligibility=eligibility,
        report_path=report,
        statistics_path=statistics_path,
        eligibility_path=eligibility_path,
        figure_paths=figures,
    )


def run_900_hour_investigation(
    *,
    split_config_path: Path | str = DEFAULT_SPLIT_CONFIG,
    cleaning_run_path: Path | None = None,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> PeakInvestigationResult:
    """Run the focused investigation using the existing Phase 7.2 population."""
    config = load_split_config(split_config_path)
    run_path = resolve_cleaning_run(
        processed_root=config.processed_root,
        latest_pointer=config.latest_pointer,
        override=cleaning_run_path,
    )
    source, source_path = load_verified_cleaned_source(run_path)
    population = select_training_duration_population(
        source,
        boundaries=config.boundaries,
        timestamp_column=config.timestamp_column,
    )
    evidence = investigate_900_hour_concentration(
        population,
        identifier_column=config.identifier_column,
        complaint_type_column="complaint_type",
    )
    report, intervals, types, patterns, figure = write_peak_investigation_outputs(
        report_root=report_root,
        evidence=evidence,
        source_path=source_path,
        train_start=config.boundaries.train_start,
        observation_cutoff=config.boundaries.train_end_exclusive,
    )
    LOGGER.info(
        "Phase 7.2 900-hour investigation complete eligible=%s peak=%s report=%s",
        len(evidence.eligible), len(evidence.peak), report,
    )
    return PeakInvestigationResult(
        population=population,
        evidence=evidence,
        report_path=report,
        intervals_path=intervals,
        complaint_types_path=types,
        closure_patterns_path=patterns,
        figure_path=figure,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the Phase 7.2 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        "--cleaning-run",
        type=Path,
        default=None,
        help="Optional explicit successful cleaning run; defaults to latest.json.",
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--investigate-900-hours",
        action="store_true",
        help="Write only the focused 850–950-hour investigation outputs.",
    )
    return parser


def main() -> None:
    """Run the command-line workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    arguments = _parser().parse_args()
    runner = (
        run_900_hour_investigation
        if arguments.investigate_900_hours
        else run_resolution_duration_analysis
    )
    runner(
        split_config_path=arguments.split_config,
        cleaning_run_path=arguments.cleaning_run,
        report_root=arguments.report_root,
    )


if __name__ == "__main__":
    main()
