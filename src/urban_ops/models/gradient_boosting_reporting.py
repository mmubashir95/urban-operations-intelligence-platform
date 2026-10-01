"""Build Month 2 Gradient Boosting reports from completed evaluations.

This module formats and persists validation-only evidence. It reads frozen
Month 1 Logistic Regression CSV artifacts and never trains or scores a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from urban_ops.models.evaluation import CalibrationEvaluation, RankingEvaluation
from urban_ops.utils.paths import PROJECT_ROOT


MONTH_2_REPORT_DIR: Final = PROJECT_ROOT / "reports/month_2"
FROZEN_VALIDATION_RESULTS_PATH: Final = (
    PROJECT_ROOT / "reports/tables/baseline_validation_results.csv"
)
FROZEN_CAPACITY_RESULTS_PATH: Final = (
    PROJECT_ROOT
    / "reports/tables/logistic_regression_validation_capacity_comparison.csv"
)
LOGISTIC_REGRESSION_MODEL_NAME: Final = "Logistic Regression"
GRADIENT_BOOSTING_MODEL_NAME: Final = "Gradient Boosting"


@dataclass(frozen=True)
class GradientBoostingReportArtifacts:
    """Paths and comparison evidence written by the Month 2 workflow."""

    validation_path: Path
    calibration_path: Path
    capacity_path: Path
    comparison_path: Path
    markdown_path: Path
    comparison: pd.DataFrame


def load_frozen_logistic_regression_evidence(
    *,
    validation_results_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    capacity_results_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
) -> tuple[pd.Series, pd.DataFrame]:
    """Load the frozen Month 1 validation row and capacity table from CSV."""
    validation = pd.read_csv(validation_results_path)
    capacity = pd.read_csv(capacity_results_path)
    row = validation.loc[
        validation["model"].eq(LOGISTIC_REGRESSION_MODEL_NAME)
        & validation["evaluated_split"].eq("validation")
    ]
    if len(row) != 1:
        raise ValueError(
            "Frozen validation evidence must contain exactly one Logistic "
            "Regression validation row."
        )
    if capacity["model"].tolist() != [LOGISTIC_REGRESSION_MODEL_NAME] * 3:
        raise ValueError("Frozen capacity evidence must contain Logistic Regression.")
    if capacity["evaluated_split"].tolist() != ["validation"] * 3:
        raise ValueError("Frozen capacity evidence must use validation only.")
    if capacity["capacity"].tolist() != [0.05, 0.10, 0.20]:
        raise ValueError("Frozen capacity evidence must use 5%, 10%, and 20%.")
    return row.iloc[0], capacity


def _validation_table(
    ranking: RankingEvaluation,
    calibration: CalibrationEvaluation,
) -> pd.DataFrame:
    """Build the established one-row Month 2 validation summary."""
    metrics = ranking.metrics
    if calibration.metrics.row_count != metrics.row_count:
        raise ValueError("Ranking and calibration row counts must match.")
    return pd.DataFrame(
        [
            {
                "model": GRADIENT_BOOSTING_MODEL_NAME,
                "evaluated_split": "validation",
                "probability_status": "raw_uncalibrated",
                "row_count": metrics.row_count,
                "positive_count": metrics.positive_count,
                "negative_count": metrics.negative_count,
                "positive_rate": metrics.positive_rate,
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "brier_score": calibration.metrics.brier_score,
            }
        ]
    )


def _comparison_table(
    *,
    ranking: RankingEvaluation,
    calibration: CalibrationEvaluation,
    capacity: pd.DataFrame,
    logistic_validation: pd.Series,
    logistic_capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Build frozen Logistic Regression versus Gradient Boosting evidence."""
    metric_values = [
        ("PR-AUC", logistic_validation["pr_auc"], ranking.metrics.pr_auc),
        ("ROC-AUC", logistic_validation["roc_auc"], ranking.metrics.roc_auc),
        (
            "Brier Score",
            logistic_validation["brier_score"],
            calibration.metrics.brier_score,
        ),
    ]
    for capacity_fraction in (0.05, 0.10, 0.20):
        label = f"{capacity_fraction:.0%}"
        logistic_row = logistic_capacity.loc[
            logistic_capacity["capacity"].eq(capacity_fraction)
        ].iloc[0]
        gradient_row = capacity.loc[
            capacity["capacity"].eq(capacity_fraction)
        ].iloc[0]
        metric_values.extend(
            [
                (
                    f"Precision@{label}",
                    logistic_row["precision_at_k"],
                    gradient_row["precision_at_k"],
                ),
                (
                    f"Recall@{label}",
                    logistic_row["recall_at_k"],
                    gradient_row["recall_at_k"],
                ),
            ]
        )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "logistic_regression": float(logistic_value),
                "gradient_boosting": float(gradient_value),
                "difference_gb_minus_lr": float(gradient_value - logistic_value),
            }
            for metric, logistic_value, gradient_value in metric_values
        ]
    )


def _markdown_report(
    *,
    validation: pd.DataFrame,
    calibration_table: pd.DataFrame,
    capacity: pd.DataFrame,
    comparison: pd.DataFrame,
    training_row_count: int,
    feature_count: int,
    split_id: str,
) -> str:
    """Render the concise validation-only Phase 2 workflow report."""
    metrics = validation.iloc[0]
    capacity_rows = list(capacity.itertuples(index=False))
    capacity_lines = "\n".join(
        f"| {row.capacity_pct:.0f}% | {int(row.selected_count):,} | "
        f"{int(row.captured_positive_count):,} | {row.precision_at_k:.4f} | "
        f"{row.recall_at_k:.4f} |"
        for row in capacity_rows
    )
    comparison_lines = "\n".join(
        f"| {row.metric} | {row.logistic_regression:.4f} | "
        f"{row.gradient_boosting:.4f} | {row.difference_gb_minus_lr:+.4f} |"
        for row in comparison.itertuples(index=False)
    )
    populated_bins = calibration_table.loc[calibration_table["row_count"].gt(0)]
    calibration_lines = "\n".join(
        f"| {row.lower_bound:.1f}–{row.upper_bound:.1f} | "
        f"{int(row.row_count):,} | {row.mean_predicted_risk:.4f} | "
        f"{row.observed_positive_rate:.4f} |"
        for row in populated_bins.itertuples(index=False)
    )
    return f"""# Phase 2 Gradient Boosting Validation Workflow

## Workflow boundary

The dedicated Month 2 workflow reused frozen Month 1 inputs (`{split_id}`),
fitted the deterministic XGBoost configuration on {training_row_count:,}
training rows and {feature_count:,} frozen features, then generated one raw
positive-class probability per validation complaint. Preprocessing was not
refitted. Test scores and labels were not accessed.

The same raw, uncalibrated validation score array feeds ranking, calibration,
and operational Top-K evaluation. No classification threshold is applied; in
particular, the frozen Logistic Regression threshold `0.49` is not transferred
to Gradient Boosting.

## Validation metrics

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | {metrics.pr_auc:.10f} |
| ROC-AUC | {metrics.roc_auc:.10f} |
| Brier Score | {metrics.brier_score:.10f} |

## Raw-probability calibration

The Brier Score and ten uniform calibration bins use raw validation
probabilities. Empty bins remain present in the CSV; populated bins are shown
below.

| Probability bin | Count | Mean predicted risk | Observed miss rate |
|---:|---:|---:|---:|
{calibration_lines}

Differences between predicted and observed rates show region-specific under-
or over-prediction. No probability calibration transformation or qualitative
pass/fail rule is applied.

## Operational capacity

Capacity is a percentage of the {int(metrics.row_count):,}-complaint validation
population. Counts use the shared `ceil(n × capacity)` rule.

| Capacity | Reviews | Actual misses captured | Precision@K | Recall@K |
|---:|---:|---:|---:|---:|
{capacity_lines}

Expanding the reviewed prefix increases captured misses and recall. These
figures describe validation evidence only; no capacity or staffing policy is
selected.

## Frozen Logistic Regression comparison

The comparison below loads frozen Month 1 CSV evidence. Logistic Regression is
not retrained by this workflow.

| Metric | Logistic Regression | Gradient Boosting | Difference (GB − LR) |
|---|---:|---:|---:|
{comparison_lines}

The comparison is evidence, not final model selection. No tuning, probability
calibration transformation, threshold selection, or test-set evaluation occurs
in this workflow.
"""


def write_gradient_boosting_reports(
    *,
    ranking: RankingEvaluation,
    calibration: CalibrationEvaluation,
    calibration_table: pd.DataFrame,
    capacity_table: pd.DataFrame,
    training_row_count: int,
    feature_count: int,
    split_id: str,
    output_directory: Path | str = MONTH_2_REPORT_DIR,
    frozen_validation_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    frozen_capacity_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
) -> GradientBoostingReportArtifacts:
    """Write all Phase 2 validation reports from completed workflow results."""
    logistic_validation, logistic_capacity = load_frozen_logistic_regression_evidence(
        validation_results_path=frozen_validation_path,
        capacity_results_path=frozen_capacity_path,
    )
    validation = _validation_table(ranking, calibration)
    comparison = _comparison_table(
        ranking=ranking,
        calibration=calibration,
        capacity=capacity_table,
        logistic_validation=logistic_validation,
        logistic_capacity=logistic_capacity,
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    validation_path = directory / "phase_2_gradient_boosting_validation.csv"
    calibration_path = directory / "phase_2_gradient_boosting_calibration.csv"
    capacity_path = directory / "phase_2_gradient_boosting_capacity.csv"
    comparison_path = directory / "phase_2_model_comparison.csv"
    markdown_path = directory / "phase_2_gradient_boosting.md"
    validation.to_csv(validation_path, index=False)
    calibration_table.to_csv(calibration_path, index=False)
    capacity_table.to_csv(capacity_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    markdown_path.write_text(
        _markdown_report(
            validation=validation,
            calibration_table=calibration_table,
            capacity=capacity_table,
            comparison=comparison,
            training_row_count=training_row_count,
            feature_count=feature_count,
            split_id=split_id,
        ),
        encoding="utf-8",
    )
    return GradientBoostingReportArtifacts(
        validation_path=validation_path,
        calibration_path=calibration_path,
        capacity_path=capacity_path,
        comparison_path=comparison_path,
        markdown_path=markdown_path,
        comparison=comparison,
    )
