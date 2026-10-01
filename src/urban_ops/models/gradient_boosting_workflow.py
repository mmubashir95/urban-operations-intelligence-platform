"""Orchestrate the Month 2 Gradient Boosting validation experiment.

The workflow reuses verified frozen Month 1 inputs, fits on training only,
evaluates one raw validation score vector, and leaves the test split untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH
from urban_ops.models.evaluation import (
    CALIBRATION_N_BINS,
    CalibrationEvaluation,
    CapacityComparisonRow,
    CapacityLevel,
    RankingEvaluation,
    build_calibration_table,
    compare_capacity_levels,
    evaluate_calibration,
    evaluate_ranking,
    get_standard_capacity_levels,
)
from urban_ops.models.gradient_boosting_capacity import (
    build_gradient_boosting_capacity_table,
)
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)
from urban_ops.models.gradient_boosting_reporting import (
    FROZEN_CAPACITY_RESULTS_PATH,
    FROZEN_VALIDATION_RESULTS_PATH,
    MONTH_2_REPORT_DIR,
    GradientBoostingReportArtifacts,
    FrozenLogisticRegressionValidationEvidence,
    write_gradient_boosting_reports,
)
from urban_ops.models.gradient_boosting_training import (
    GradientBoostingTrainingResult,
    train_gradient_boosted_risk_model,
)
from urban_ops.models.gradient_boosting_validation import (
    generate_gradient_boosting_validation_scores,
)


@dataclass(frozen=True)
class GradientBoostingWorkflowResult:
    """Structured outputs from the dedicated Month 2 validation workflow."""

    training: GradientBoostingTrainingResult
    validation_scores: np.ndarray
    ranking: RankingEvaluation
    calibration: CalibrationEvaluation
    calibration_table: pd.DataFrame
    capacity_levels: tuple[CapacityLevel, ...]
    capacity: tuple[CapacityComparisonRow, ...]
    capacity_table: pd.DataFrame
    feature_names: tuple[str, ...]
    split_id: str
    frozen_logistic_regression: FrozenLogisticRegressionValidationEvidence
    comparison: pd.DataFrame
    report_artifacts: GradientBoostingReportArtifacts


def run_gradient_boosting_workflow(
    *,
    eda_config_path: Path | str = EDA_CONFIG_PATH,
    split_run_path: Path | None = None,
    output_directory: Path | str = MONTH_2_REPORT_DIR,
    frozen_validation_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    frozen_capacity_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
) -> GradientBoostingWorkflowResult:
    """Run train-only fitting and validation-only shared evaluation."""
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=eda_config_path,
        split_run_path=split_run_path,
    )
    training = train_gradient_boosted_risk_model(inputs)
    validation_scores = generate_gradient_boosting_validation_scores(
        training.model,
        inputs,
    )
    y_validation = inputs.targets["validation"]
    ranking = evaluate_ranking(y_validation, validation_scores)
    calibration = evaluate_calibration(y_validation, validation_scores)
    calibration_table = build_calibration_table(
        y_validation,
        validation_scores,
        n_bins=CALIBRATION_N_BINS,
    )
    capacity_levels = get_standard_capacity_levels(len(y_validation))
    capacity = compare_capacity_levels(
        y_validation,
        validation_scores,
        capacities=tuple(level.capacity for level in capacity_levels),
    )
    capacity_table = build_gradient_boosting_capacity_table(capacity)
    report_artifacts = write_gradient_boosting_reports(
        ranking=ranking,
        calibration=calibration,
        calibration_table=calibration_table,
        capacity_table=capacity_table,
        training_row_count=training.metadata.training_row_count,
        feature_count=training.metadata.feature_count,
        split_id=inputs.split_id,
        model_implementation=training.metadata.implementation,
        model_class=training.metadata.model_class,
        configuration_version=training.metadata.configuration_version,
        model_configuration=training.model.config.model_parameters,
        output_directory=output_directory,
        frozen_validation_path=frozen_validation_path,
        frozen_capacity_path=frozen_capacity_path,
    )
    return GradientBoostingWorkflowResult(
        training=training,
        validation_scores=validation_scores,
        ranking=ranking,
        calibration=calibration,
        calibration_table=calibration_table,
        capacity_levels=capacity_levels,
        capacity=capacity,
        capacity_table=capacity_table,
        feature_names=inputs.feature_names,
        split_id=inputs.split_id,
        frozen_logistic_regression=(
            report_artifacts.frozen_logistic_regression
        ),
        comparison=report_artifacts.comparison,
        report_artifacts=report_artifacts,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the dedicated Month 2 workflow command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the Gradient Boosting validation workflow."
    )
    parser.add_argument("--split-run", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=MONTH_2_REPORT_DIR)
    return parser


def main() -> None:
    """Run the dedicated workflow from the command line."""
    args = _parser().parse_args()
    run_gradient_boosting_workflow(
        split_run_path=args.split_run,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
