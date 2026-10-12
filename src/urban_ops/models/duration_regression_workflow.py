"""Run the controlled Phase 7.4 Linear Regression and XGBoost experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from urban_ops.analysis.resolution_duration import (
    DurationAnalysisPopulation,
    select_duration_population,
)
from urban_ops.analysis.resolution_duration_source import load_verified_cleaned_source
from urban_ops.features.numeric_preprocessing import (
    FittedNumericPreprocessor,
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_numeric_preprocessor,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.temporal import derive_approved_temporal_features
from urban_ops.models.duration_baseline import evaluate_duration_predictions
from urban_ops.models.duration_regression import (
    DurationRegressionConfig,
    DurationPredictions,
    build_duration_regressors,
    comparison_improvement,
    load_duration_regression_config,
)
from urban_ops.splitting.pipeline import load_split_config
from urban_ops.splitting.models import SplitBoundaries
from urban_ops.splitting.source import resolve_cleaning_run
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "configs/data/splits.yaml"
POLICY_PATH = PROJECT_ROOT / "configs/features/resolution_risk_baseline.yaml"
NUMERIC_CONFIG_PATH = PROJECT_ROOT / "configs/features/resolution_risk_numeric_preprocessing.yaml"
BASELINE_METRICS_PATH = PROJECT_ROOT / "reports/month_2/phase_7_3_median_baseline_metrics.csv"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/month_2"
COMPARISON_FILENAME = "phase_7_4_regression_comparison.csv"
REPORT_FILENAME = "phase_7_4_regression_comparison.md"


@dataclass(frozen=True)
class RegressionMatrices:
    """Aligned train/validation matrices and frozen preprocessing state."""

    train: object
    validation: object
    feature_names: tuple[str, ...]
    fitted_preprocessor: FittedNumericPreprocessor


@dataclass(frozen=True)
class RegressionWorkflowResult:
    """Phase 7.4 populations, comparison evidence, and report paths."""

    training: DurationAnalysisPopulation
    validation: DurationAnalysisPopulation
    matrices: RegressionMatrices
    model_predictions: dict[str, DurationPredictions]
    comparison: pd.DataFrame
    report_path: Path
    comparison_path: Path


def _eligible_frame(population: DurationAnalysisPopulation) -> pd.DataFrame:
    """Return source rows aligned exactly to the eligible target series."""
    return population.historical.loc[population.eligible_durations.index].copy()


def prepare_regression_matrices(
    training: DurationAnalysisPopulation,
    validation: DurationAnalysisPopulation,
    *,
    policy_path: Path | str = POLICY_PATH,
    numeric_config_path: Path | str = NUMERIC_CONFIG_PATH,
) -> RegressionMatrices:
    """Derive governed features and fit preprocessing on eligible train rows only."""
    policy = load_feature_policy(policy_path)
    numeric_config = load_numeric_preprocessing_config(numeric_config_path)
    train_features = derive_approved_temporal_features(_eligible_frame(training), policy=policy)
    validation_features = derive_approved_temporal_features(_eligible_frame(validation), policy=policy)
    fitted = fit_numeric_preprocessor(train_features, policy=policy, config=numeric_config)
    train_matrix = transform_numeric_preprocessor(train_features, fitted=fitted, config=numeric_config)
    validation_matrix = transform_numeric_preprocessor(validation_features, fitted=fitted, config=numeric_config)
    if train_matrix.shape[0] != len(training.eligible_durations):
        raise RuntimeError("Training feature and target rows do not align.")
    if validation_matrix.shape[0] != len(validation.eligible_durations):
        raise RuntimeError("Validation feature and target rows do not align.")
    return RegressionMatrices(
        train=train_matrix,
        validation=validation_matrix,
        feature_names=fitted.output_feature_names,
        fitted_preprocessor=fitted,
    )


def load_frozen_baseline_metrics(
    path: Path | str,
    *,
    expected_training_count: int,
    expected_validation_count: int,
) -> tuple[float, float]:
    """Load exact Phase 7.3 errors and require matching eligible populations."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Frozen Phase 7.3 metrics are missing; run Phase 7.3 first: {artifact}"
        )
    table = pd.read_csv(artifact)
    if tuple(table.columns) != ("metric", "value", "unit") or table["metric"].duplicated().any():
        raise ValueError("Frozen Phase 7.3 metrics schema is invalid.")
    values = table.set_index("metric")["value"]
    required = {
        "training_eligible_count", "validation_eligible_count",
        "validation_mae_hours", "validation_rmse_hours",
    }
    if not required.issubset(values.index):
        raise ValueError("Frozen Phase 7.3 metrics are incomplete.")
    if int(values["training_eligible_count"]) != expected_training_count or int(values["validation_eligible_count"]) != expected_validation_count:
        raise ValueError("Phase 7.3 and Phase 7.4 eligible populations do not match.")
    return float(values["validation_mae_hours"]), float(values["validation_rmse_hours"])


def _comparison_row(
    model_name: str,
    *,
    mae: float,
    rmse: float,
    baseline_mae: float,
    baseline_rmse: float,
    sample_count: int,
    negative_raw_count: int,
    prediction_policy: str = "clip_at_zero",
) -> dict[str, object]:
    mae_absolute, mae_percent = comparison_improvement(baseline_mae, mae)
    rmse_absolute, rmse_percent = comparison_improvement(baseline_rmse, rmse)
    return {
        "model": model_name,
        "validation_sample_count": sample_count,
        "validation_mae_hours": mae,
        "validation_rmse_hours": rmse,
        "mae_improvement_hours": mae_absolute,
        "mae_improvement_percentage": mae_percent,
        "rmse_improvement_hours": rmse_absolute,
        "rmse_improvement_percentage": rmse_percent,
        "negative_raw_prediction_count": negative_raw_count,
        "nonnegative_prediction_policy": prediction_policy,
    }


def build_comparison_table(
    *,
    baseline_mae: float,
    baseline_rmse: float,
    validation_count: int,
    model_results: Sequence[tuple[str, object, DurationPredictions]],
) -> pd.DataFrame:
    """Build the frozen-baseline and newly evaluated model comparison."""
    rows = [_comparison_row(
        "Median Duration Baseline", mae=baseline_mae, rmse=baseline_rmse,
        baseline_mae=baseline_mae, baseline_rmse=baseline_rmse,
        sample_count=validation_count, negative_raw_count=0,
        prediction_policy="not_applicable_frozen_artifact",
    )]
    for name, metrics, predictions in model_results:
        rows.append(_comparison_row(
            name, mae=metrics.mae_hours, rmse=metrics.rmse_hours,
            baseline_mae=baseline_mae, baseline_rmse=baseline_rmse,
            sample_count=metrics.sample_count,
            negative_raw_count=predictions.negative_raw_count,
        ))
    return pd.DataFrame(rows)


def _report(
    *, source_path: Path, boundaries: SplitBoundaries,
    regression_config: DurationRegressionConfig,
    training: DurationAnalysisPopulation,
    validation: DurationAnalysisPopulation, matrices: RegressionMatrices,
    comparison: pd.DataFrame,
) -> str:
    rows = "\n".join(
        f"| {row.model} | {row.validation_mae_hours:.3f} | {row.validation_rmse_hours:.3f} | {row.mae_improvement_percentage:.2f}% | {row.rmse_improvement_percentage:.2f}% | {int(row.negative_raw_prediction_count)} |"
        for row in comparison.itertuples(index=False)
    )
    return f"""# Phase 7.4 — Regression Model Comparison

## Objective and governed data

Compare feature-based Linear Regression and XGBRegressor with the frozen Phase
7.3 median baseline. Source: `{source_path}`.

- Training: `[{boundaries.train_start.isoformat()}, {boundaries.train_end_exclusive.isoformat()})`; cutoff `{boundaries.train_end_exclusive.isoformat()}`
- Validation: `[{boundaries.validation_start.isoformat()}, {boundaries.validation_end_exclusive.isoformat()})`; cutoff `{boundaries.validation_end_exclusive.isoformat()}`
- Eligible train / validation: {len(training.eligible_durations):,} / {len(validation.eligible_durations):,}
- Test data or targets used: No

## Features and preprocessing

Active creation-time features, in matrix order: `{', '.join(matrices.feature_names)}`.
They are derived deterministically from UTC `created_date`. Existing governed
numeric preprocessing is fit/validated on eligible training rows and applied
unchanged to validation. It performs bounded float64 pass-through with no
imputation or scaling. Outcome, status, closure, target, and conditional
geographic fields are excluded by the central feature policy.

## Model configurations

- Linear Regression: `fit_intercept={regression_config.fit_intercept}`.
- XGBRegressor: `{regression_config.xgboost_parameters}`.
- Target: original resolution hours; no cap, outlier removal, or log transform.
- Output policy: raw predictions remain available for diagnostics; negative raw
  predictions are counted and clipped to zero before MAE/RMSE evaluation.

## Validation comparison

MAE is primary; RMSE is secondary. Positive improvement means lower error than
the frozen Phase 7.3 baseline.

| Model | MAE (hours) | RMSE (hours) | MAE improvement | RMSE improvement | Negative raw predictions |
|---|---:|---:|---:|---:|---:|
{rows}

This is a preliminary validation comparison, not final model selection. The
validation outcome distribution shifted substantially from training in Phase
7.3, and evaluation covers only completed outcomes observable by each cutoff.
Phase 7.5 should examine error distributions, temporal/segment behavior, and
the influence of the right tail and 900-hour concentration before selection.
"""


def run_duration_regression_workflow(
    *, split_config_path: Path | str = DEFAULT_SPLIT_CONFIG,
    cleaning_run_path: Path | None = None,
    baseline_metrics_path: Path | str = BASELINE_METRICS_PATH,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> RegressionWorkflowResult:
    """Train both regressors and compare validation errors with Phase 7.3."""
    split_config = load_split_config(split_config_path)
    run_path = resolve_cleaning_run(
        processed_root=split_config.processed_root,
        latest_pointer=split_config.latest_pointer, override=cleaning_run_path,
    )
    source, source_path = load_verified_cleaned_source(run_path)
    training = select_duration_population(
        source, creation_start=split_config.boundaries.train_start,
        creation_end_exclusive=split_config.boundaries.train_end_exclusive,
        observation_cutoff=split_config.boundaries.train_end_exclusive,
        timestamp_column=split_config.timestamp_column,
    )
    validation = select_duration_population(
        source, creation_start=split_config.boundaries.validation_start,
        creation_end_exclusive=split_config.boundaries.validation_end_exclusive,
        observation_cutoff=split_config.boundaries.validation_end_exclusive,
        timestamp_column=split_config.timestamp_column,
    )
    matrices = prepare_regression_matrices(training, validation)
    baseline_mae, baseline_rmse = load_frozen_baseline_metrics(
        baseline_metrics_path,
        expected_training_count=len(training.eligible_durations),
        expected_validation_count=len(validation.eligible_durations),
    )
    regression_config = load_duration_regression_config()
    linear, xgboost = build_duration_regressors(regression_config)
    results = []
    model_predictions: dict[str, DurationPredictions] = {}
    for model in (linear, xgboost):
        model.fit(matrices.train, training.eligible_durations, feature_names=matrices.feature_names)
        predictions = model.predict(matrices.validation)
        metrics = evaluate_duration_predictions(validation.eligible_durations, predictions.nonnegative)
        results.append((model.model_name, metrics, predictions))
        model_predictions[model.model_name] = predictions
    comparison = build_comparison_table(
        baseline_mae=baseline_mae, baseline_rmse=baseline_rmse,
        validation_count=len(validation.eligible_durations), model_results=results,
    )
    report = _report(
        source_path=source_path, boundaries=split_config.boundaries,
        regression_config=regression_config, training=training,
        validation=validation, matrices=matrices, comparison=comparison,
    )
    report_root.mkdir(parents=True, exist_ok=True)
    comparison_path = report_root / COMPARISON_FILENAME
    report_path = report_root / REPORT_FILENAME
    comparison.to_csv(comparison_path, index=False, float_format="%.17g")
    report_path.write_text(report, encoding="utf-8")
    LOGGER.info("Phase 7.4 complete report=%s", report_path)
    return RegressionWorkflowResult(
        training, validation, matrices, model_predictions, comparison,
        report_path, comparison_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--cleaning-run", type=Path)
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_METRICS_PATH)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_duration_regression_workflow(
        split_config_path=args.split_config, cleaning_run_path=args.cleaning_run,
        baseline_metrics_path=args.baseline_metrics, report_root=args.report_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
