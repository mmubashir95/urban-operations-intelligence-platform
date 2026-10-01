"""Build Month 2 Gradient Boosting reports from completed evaluations.

This module formats and persists validation-only evidence. It reads frozen
Month 1 Logistic Regression CSV artifacts and never trains or scores a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isclose
from pathlib import Path
from typing import Final, Mapping

import numpy as np
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
COMPARISON_METRICS: Final = (
    "PR-AUC",
    "ROC-AUC",
    "Brier Score",
    "Precision@5%",
    "Recall@5%",
    "Precision@10%",
    "Recall@10%",
    "Precision@20%",
    "Recall@20%",
)


@dataclass(frozen=True)
class FrozenLogisticRegressionValidationEvidence:
    """Validated and normalized frozen Month 1 validation metrics."""

    model_name: str
    evaluated_split: str
    frozen: bool
    validation_source: Path
    capacity_source: Path
    row_count: int
    positive_count: int
    metrics: tuple[tuple[str, float], ...]

    def metric_values(self) -> dict[str, float]:
        """Return canonical metric names mapped to full-precision values."""
        return dict(self.metrics)


@dataclass(frozen=True)
class GradientBoostingReportArtifacts:
    """Paths and comparison evidence written by the Month 2 workflow."""

    validation_path: Path
    calibration_path: Path
    capacity_path: Path
    comparison_path: Path
    markdown_path: Path
    frozen_logistic_regression: FrozenLogisticRegressionValidationEvidence
    comparison: pd.DataFrame


def _read_frozen_csv(path: Path | str, *, artifact_name: str) -> pd.DataFrame:
    """Read one required structured frozen artifact or fail clearly."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Frozen {artifact_name} artifact does not exist: {artifact_path}"
        )
    try:
        return pd.read_csv(artifact_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Frozen {artifact_name} artifact is not a readable CSV: {artifact_path}"
        ) from exc


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    artifact_name: str,
) -> None:
    """Require the frozen artifact's machine-readable schema."""
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(
            f"Frozen {artifact_name} artifact is missing required columns: {missing}."
        )


def _finite_unit_metric(value: object, *, metric: str) -> float:
    """Return one finite metric in the closed unit interval."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Frozen {metric} must be numeric.") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Frozen {metric} must be finite and within [0, 1].")
    return result


def _artifact_display_path(path: Path) -> str:
    """Return project-relative provenance when the artifact is in the project."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_frozen_logistic_regression_validation_metrics(
    *,
    validation_results_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    capacity_results_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
) -> FrozenLogisticRegressionValidationEvidence:
    """Load, validate, normalize, and retain provenance for Month 1 evidence."""
    validation_path = Path(validation_results_path)
    capacity_path = Path(capacity_results_path)
    validation = _read_frozen_csv(
        validation_path,
        artifact_name="validation results",
    )
    capacity = _read_frozen_csv(
        capacity_path,
        artifact_name="capacity results",
    )
    _require_columns(
        validation,
        (
            "model",
            "evaluated_split",
            "row_count",
            "positive_count",
            "pr_auc",
            "roc_auc",
            "brier_score",
        ),
        artifact_name="validation results",
    )
    _require_columns(
        capacity,
        (
            "model",
            "evaluated_split",
            "capacity",
            "selected_count",
            "captured_positive_count",
            "precision_at_k",
            "recall_at_k",
        ),
        artifact_name="capacity results",
    )
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
    validation_row = row.iloc[0]
    try:
        row_count = int(validation_row["row_count"])
        positive_count = int(validation_row["positive_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frozen validation population counts must be integers."
        ) from exc
    if row_count <= 0 or not 0 <= positive_count <= row_count:
        raise ValueError("Frozen validation population counts are invalid.")

    normalized = [
        ("PR-AUC", _finite_unit_metric(validation_row["pr_auc"], metric="PR-AUC")),
        (
            "ROC-AUC",
            _finite_unit_metric(validation_row["roc_auc"], metric="ROC-AUC"),
        ),
        (
            "Brier Score",
            _finite_unit_metric(validation_row["brier_score"], metric="Brier Score"),
        ),
    ]
    for capacity_row in capacity.itertuples(index=False):
        fraction = float(capacity_row.capacity)
        label = f"{fraction:.0%}"
        try:
            selected_count = int(capacity_row.selected_count)
            captured_count = int(capacity_row.captured_positive_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("Frozen capacity counts must be integers.") from exc
        if selected_count != ceil(row_count * fraction):
            raise ValueError(
                "Frozen capacity selected count does not match the validation "
                f"population at {label}."
            )
        if not 0 <= captured_count <= min(selected_count, positive_count):
            raise ValueError(f"Frozen captured-positive count is invalid at {label}.")
        precision = _finite_unit_metric(
            capacity_row.precision_at_k,
            metric=f"Precision@{label}",
        )
        recall = _finite_unit_metric(
            capacity_row.recall_at_k,
            metric=f"Recall@{label}",
        )
        if not isclose(
            precision,
            captured_count / selected_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Frozen Precision@{label} does not match its counts.")
        expected_recall = captured_count / positive_count if positive_count else 0.0
        if not isclose(recall, expected_recall, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Frozen Recall@{label} does not match its counts.")
        normalized.extend(
            [
                (f"Precision@{label}", precision),
                (f"Recall@{label}", recall),
            ]
        )
    if tuple(metric for metric, _ in normalized) != COMPARISON_METRICS:
        raise ValueError("Frozen evidence does not provide all comparison metrics.")
    return FrozenLogisticRegressionValidationEvidence(
        model_name=LOGISTIC_REGRESSION_MODEL_NAME,
        evaluated_split="validation",
        frozen=True,
        validation_source=validation_path,
        capacity_source=capacity_path,
        row_count=row_count,
        positive_count=positive_count,
        metrics=tuple(normalized),
    )


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


def build_model_comparison_table(
    *,
    ranking: RankingEvaluation,
    calibration: CalibrationEvaluation,
    capacity: pd.DataFrame,
    frozen_logistic_regression: FrozenLogisticRegressionValidationEvidence,
) -> pd.DataFrame:
    """Build canonical comparisons with difference always equal to GB minus LR."""
    if frozen_logistic_regression.evaluated_split != "validation":
        raise ValueError("Frozen Logistic Regression evidence must use validation.")
    if ranking.metrics.row_count != frozen_logistic_regression.row_count:
        raise ValueError("LR and GB validation row counts must match.")
    if ranking.metrics.positive_count != frozen_logistic_regression.positive_count:
        raise ValueError("LR and GB validation positive counts must match.")
    gradient_values = {
        "PR-AUC": ranking.metrics.pr_auc,
        "ROC-AUC": ranking.metrics.roc_auc,
        "Brier Score": calibration.metrics.brier_score,
    }
    for capacity_fraction in (0.05, 0.10, 0.20):
        label = f"{capacity_fraction:.0%}"
        gradient_row = capacity.loc[
            capacity["capacity"].eq(capacity_fraction)
        ].iloc[0]
        gradient_values[f"Precision@{label}"] = gradient_row["precision_at_k"]
        gradient_values[f"Recall@{label}"] = gradient_row["recall_at_k"]
    frozen_values = frozen_logistic_regression.metric_values()
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "logistic_regression": frozen_values[metric],
                "gradient_boosting": float(gradient_values[metric]),
                "difference_gb_minus_lr": float(
                    gradient_values[metric] - frozen_values[metric]
                ),
            }
            for metric in COMPARISON_METRICS
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
    model_implementation: str,
    model_class: str,
    configuration_version: int,
    model_configuration: Mapping[str, object],
    frozen_logistic_regression: FrozenLogisticRegressionValidationEvidence,
) -> str:
    """Render the complete human-readable Phase 2 reporting contract."""
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
    comparison_by_metric = comparison.set_index("metric")
    pr_difference = comparison_by_metric.loc["PR-AUC", "difference_gb_minus_lr"]
    roc_difference = comparison_by_metric.loc["ROC-AUC", "difference_gb_minus_lr"]
    brier_difference = comparison_by_metric.loc[
        "Brier Score", "difference_gb_minus_lr"
    ]
    ranking_interpretation = (
        "The initial Gradient Boosting configuration has lower validation "
        "PR-AUC and ROC-AUC than the frozen Logistic Regression benchmark."
        if pr_difference < 0.0 and roc_difference < 0.0
        else "Ranking differences are shown numerically in the table above."
    )
    brier_interpretation = (
        "Gradient Boosting also has a slightly higher Brier Score, indicating "
        "slightly higher validation probability error."
        if brier_difference > 0.0
        else "The Brier difference is shown without changing its GB - LR sign."
    )
    top_k_differences = comparison.loc[
        comparison["metric"].str.startswith(("Precision@", "Recall@")),
        "difference_gb_minus_lr",
    ]
    top_k_interpretation = (
        "Precision@K and Recall@K are also lower at every shared capacity."
        if top_k_differences.lt(0.0).all()
        else "The Top-K table provides the capacity-specific ranking evidence."
    )
    validation_source = _artifact_display_path(
        frozen_logistic_regression.validation_source
    )
    capacity_source = _artifact_display_path(
        frozen_logistic_regression.capacity_source
    )
    configuration_lines = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in model_configuration.items()
    )
    populated_bins = calibration_table.loc[calibration_table["row_count"].gt(0)]
    calibration_lines = "\n".join(
        f"| {row.lower_bound:.1f}–{row.upper_bound:.1f} | "
        f"{int(row.row_count):,} | {row.mean_predicted_risk:.4f} | "
        f"{row.observed_positive_rate:.4f} |"
        for row in populated_bins.itertuples(index=False)
    )
    return f"""# Phase 2 — Gradient Boosting Validation Report

## 1. Experiment Objective

This first Gradient Boosting experiment tests whether a deterministic boosted
tree model improves missed-target risk ranking over the frozen Month 1 Logistic
Regression benchmark. Phase 2 establishes reproducible validation evidence; it
does not select a production model or operational policy.

## 2. Model Trained

The workflow trained the existing `{model_implementation}` binary classifier
through the `{model_class}` project wrapper. Class `1` represents a complaint
that misses its expected resolution target.

## 3. Frozen Inputs Reused

The dedicated Month 2 workflow reused frozen Month 1 inputs (`{split_id}`),
including the chronological split membership, target definition, fitted
preprocessing outputs, sparse feature matrices, and ordered feature names.
The matrices contain {feature_count:,} features. Preprocessing was not refitted
and feature columns were not reordered.

Month 1 inputs remained frozen; only the model implementation changed.

## 4. Evaluation Boundary

- **TRAIN:** fitted the model on {training_row_count:,} rows.
- **VALIDATION:** evaluated {int(metrics.row_count):,} rows using ranking,
  raw-probability calibration, Top-K capacity, and frozen-model comparison.
- **TEST:** untouched; no test scores or labels were accessed.

The same raw, uncalibrated validation score array feeds ranking, calibration,
and operational Top-K evaluation. No classification threshold is applied; in
particular, the frozen Logistic Regression threshold `0.49` is not transferred
to Gradient Boosting.

## 5. Model Configuration

Configuration version: `{configuration_version}`. Only explicitly configured
parameters are shown.

| Parameter | Value |
|---|---|
{configuration_lines}

No hyperparameter tuning was performed in Phase 2.

## 6. Validation Metrics

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | {metrics.pr_auc:.10f} |
| ROC-AUC | {metrics.roc_auc:.10f} |
| Brier Score | {metrics.brier_score:.10f} |

Higher PR-AUC and ROC-AUC indicate stronger ranking. Lower Brier Score indicates
lower probability prediction error.

Machine-readable values are in
`reports/month_2/phase_2_gradient_boosting_validation.csv`.

## 7. Raw Probability Calibration

The Brier Score and ten uniform calibration bins use raw validation
probabilities. Empty bins remain present in the CSV; populated bins are shown
below.

| Probability bin | Count | Mean predicted risk | Observed miss rate |
|---:|---:|---:|---:|
{calibration_lines}

Differences between predicted and observed rates show region-specific under-
or over-prediction. No probability calibration transformation or qualitative
pass/fail rule is applied.

The evidence does not support a binary "well calibrated" label: the raw
probabilities show region-specific under- and over-prediction. No calibration
method was applied. Full-precision bins are in
`reports/month_2/phase_2_gradient_boosting_calibration.csv`.

## 8. Operational Top-K Evaluation

Capacity is a percentage of the {int(metrics.row_count):,}-complaint validation
population. Counts use the shared `ceil(n × capacity)` rule.

| Capacity | Reviews | Actual misses captured | Precision@K | Recall@K |
|---:|---:|---:|---:|---:|
{capacity_lines}

Expanding the reviewed prefix increases captured misses and recall. These
figures describe validation evidence only; no capacity or staffing policy is
selected.

Full-precision capacity evidence is in
`reports/month_2/phase_2_gradient_boosting_capacity.csv`.

## 9. Frozen Logistic Regression Comparison

The Logistic Regression values come from frozen Month 1 validation CSV
evidence. Logistic Regression is not retrained. Gradient Boosting values come
from the current Month 2 validation workflow, and every difference is
`Gradient Boosting - Logistic Regression`.

- Validation metrics source: `{validation_source}`
- Capacity metrics source: `{capacity_source}`
- Frozen model: `{frozen_logistic_regression.model_name}`
- Frozen split: `{frozen_logistic_regression.evaluated_split}`

| Metric | Logistic Regression | Gradient Boosting | Difference (GB − LR) |
|---|---:|---:|---:|
{comparison_lines}

Full-precision differences are in
`reports/month_2/phase_2_model_comparison.csv`.

## 10. Ranking Improvement Assessment

{ranking_interpretation} {top_k_interpretation} Therefore, the initial untuned
Gradient Boosting configuration
did not materially improve validation ranking over the frozen Logistic
Regression benchmark. This does not establish that Gradient Boosting is
universally worse or unusable.

{brier_interpretation}

## 11. Threshold Policy Status

Gradient Boosting operational threshold selection remains deferred. The frozen
Logistic Regression threshold `0.49` was not transferred to Gradient Boosting,
and no `0.5` or other threshold is selected by this report.

## 12. Test-Set Protection

No test labels were used, no test probabilities were generated, and no test
evaluation was performed. Phase 2 reporting contains validation evidence only.

## 13. Phase 3 Readiness

Yes—the experiment is technically ready to proceed to Phase 3 model
selection/tuning because the train-only fit, validation evaluation stack,
frozen comparison, and deterministic reports work end to end while the test set
remains protected. This does not mean the model is production-ready.

## 14. Conclusion

The initial deterministic Gradient Boosting pipeline is reproducible, but its
current validation ranking, Top-K performance, and Brier Score do not improve
on the frozen Logistic Regression evidence. Phase 2 changes no features,
preprocessing, threshold policy, calibration method, or test-set boundary.
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
    model_implementation: str,
    model_class: str,
    configuration_version: int,
    model_configuration: Mapping[str, object],
    output_directory: Path | str = MONTH_2_REPORT_DIR,
    frozen_validation_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    frozen_capacity_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
) -> GradientBoostingReportArtifacts:
    """Write all Phase 2 validation reports from completed workflow results."""
    frozen_logistic_regression = load_frozen_logistic_regression_validation_metrics(
        validation_results_path=frozen_validation_path,
        capacity_results_path=frozen_capacity_path,
    )
    validation = _validation_table(ranking, calibration)
    comparison = build_model_comparison_table(
        ranking=ranking,
        calibration=calibration,
        capacity=capacity_table,
        frozen_logistic_regression=frozen_logistic_regression,
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
            model_implementation=model_implementation,
            model_class=model_class,
            configuration_version=configuration_version,
            model_configuration=model_configuration,
            frozen_logistic_regression=frozen_logistic_regression,
        ),
        encoding="utf-8",
    )
    return GradientBoostingReportArtifacts(
        validation_path=validation_path,
        calibration_path=calibration_path,
        capacity_path=capacity_path,
        comparison_path=comparison_path,
        markdown_path=markdown_path,
        frozen_logistic_regression=frozen_logistic_regression,
        comparison=comparison,
    )
