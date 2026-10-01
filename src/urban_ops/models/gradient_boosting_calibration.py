"""Evaluate raw Gradient Boosting validation probabilities for calibration.

This module delegates all probability metrics and binning to the shared Month 1
evaluation layer. It does not fit a calibration model, choose a threshold, or
access the test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.evaluation import (
    CALIBRATION_N_BINS,
    CalibrationEvaluation,
    build_calibration_table,
    evaluate_calibration,
)


@dataclass(frozen=True)
class GradientBoostingCalibrationResult:
    """Shared calibration result and canonical fixed-bin reporting table."""

    calibration: CalibrationEvaluation
    table: pd.DataFrame


GRADIENT_BOOSTING_CALIBRATION_FILENAME = (
    "phase_2_gradient_boosting_calibration.csv"
)


def evaluate_gradient_boosting_validation_calibration(
    inputs: FrozenBaselineInputs,
    validation_scores: object,
) -> GradientBoostingCalibrationResult:
    """Evaluate unchanged raw scores against frozen validation labels only."""
    if "validation" not in inputs.targets:
        raise ValueError(
            "Frozen Gradient Boosting inputs are missing the validation target."
        )
    y_validation = inputs.targets["validation"]
    calibration = evaluate_calibration(y_validation, validation_scores)
    table = build_calibration_table(
        y_validation,
        validation_scores,
        n_bins=CALIBRATION_N_BINS,
    )
    return GradientBoostingCalibrationResult(
        calibration=calibration,
        table=table,
    )


def write_gradient_boosting_calibration_table(
    result: GradientBoostingCalibrationResult,
    output_directory: Path | str,
) -> Path:
    """Persist the canonical shared calibration table as a Month 2 CSV."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / GRADIENT_BOOSTING_CALIBRATION_FILENAME
    result.table.to_csv(output_path, index=False)
    return output_path
