"""Build Month 2 Gradient Boosting reports from completed evaluations.

This module formats and persists validation-only evidence. Frozen Month 1
Logistic Regression metrics come from a tracked, write-once JSON snapshot that
is bound to the frozen split and Phase 9 contract. The regenerable Month 1 CSV
artifacts are read only to create that snapshot. No model is trained or scored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH: Final = (
    PROJECT_ROOT / "configs/models/month1_logistic_regression_validation_evidence.json"
)
EVIDENCE_SNAPSHOT_VERSION: Final = 1
STANDARD_CAPACITIES: Final = (0.05, 0.10, 0.20)
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
    split_id: str | None = None
    phase_9_contract_fingerprint: str | None = None
    snapshot_source: Path | None = None

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
        # round_trip parsing keeps frozen floats identical to the written text.
        return pd.read_csv(artifact_path, float_precision="round_trip")
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
    row_count, positive_count, metrics = _normalize_frozen_evidence(
        row.iloc[0].to_dict(),
        capacity.to_dict(orient="records"),
    )
    return FrozenLogisticRegressionValidationEvidence(
        model_name=LOGISTIC_REGRESSION_MODEL_NAME,
        evaluated_split="validation",
        frozen=True,
        validation_source=validation_path,
        capacity_source=capacity_path,
        row_count=row_count,
        positive_count=positive_count,
        metrics=metrics,
    )


def _normalize_frozen_evidence(
    validation_row: Mapping[str, object],
    capacity_rows: Sequence[Mapping[str, object]],
) -> tuple[int, int, tuple[tuple[str, float], ...]]:
    """Validate frozen counts and metrics, returning canonical metric pairs."""
    capacities = [row.get("capacity") for row in capacity_rows]
    if capacities != list(STANDARD_CAPACITIES):
        raise ValueError("Frozen capacity evidence must use 5%, 10%, and 20%.")
    try:
        row_count = int(validation_row["row_count"])
        positive_count = int(validation_row["positive_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Frozen validation population counts must be integers."
        ) from exc
    if row_count <= 0 or not 0 <= positive_count <= row_count:
        raise ValueError("Frozen validation population counts are invalid.")

    normalized = [
        ("PR-AUC", _finite_unit_metric(validation_row.get("pr_auc"), metric="PR-AUC")),
        (
            "ROC-AUC",
            _finite_unit_metric(validation_row.get("roc_auc"), metric="ROC-AUC"),
        ),
        (
            "Brier Score",
            _finite_unit_metric(
                validation_row.get("brier_score"), metric="Brier Score"
            ),
        ),
    ]
    for capacity_row in capacity_rows:
        fraction = float(capacity_row["capacity"])
        label = f"{fraction:.0%}"
        try:
            selected_count = int(capacity_row["selected_count"])
            captured_count = int(capacity_row["captured_positive_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Frozen capacity counts must be integers.") from exc
        if selected_count != ceil(row_count * fraction):
            raise ValueError(
                "Frozen capacity selected count does not match the validation "
                f"population at {label}."
            )
        if not 0 <= captured_count <= min(selected_count, positive_count):
            raise ValueError(f"Frozen captured-positive count is invalid at {label}.")
        precision = _finite_unit_metric(
            capacity_row.get("precision_at_k"),
            metric=f"Precision@{label}",
        )
        recall = _finite_unit_metric(
            capacity_row.get("recall_at_k"),
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
    return row_count, positive_count, tuple(normalized)


def _snapshot_payload(
    *,
    validation_path: Path,
    capacity_path: Path,
    split_id: str,
    phase_9_contract_fingerprint: str,
) -> dict[str, object]:
    """Extract the validated Month 1 LR rows into the snapshot schema."""
    # Loading first applies every structural and count-consistency check.
    load_frozen_logistic_regression_validation_metrics(
        validation_results_path=validation_path,
        capacity_results_path=capacity_path,
    )
    validation = _read_frozen_csv(validation_path, artifact_name="validation results")
    capacity = _read_frozen_csv(capacity_path, artifact_name="capacity results")
    row = validation.loc[
        validation["model"].eq(LOGISTIC_REGRESSION_MODEL_NAME)
        & validation["evaluated_split"].eq("validation")
    ].iloc[0]
    return {
        "snapshot_version": EVIDENCE_SNAPSHOT_VERSION,
        "model": LOGISTIC_REGRESSION_MODEL_NAME,
        "evaluated_split": "validation",
        "split_id": split_id,
        "phase_9_contract_fingerprint": phase_9_contract_fingerprint,
        "derived_from": {
            "validation_results": _artifact_display_path(validation_path),
            "capacity_results": _artifact_display_path(capacity_path),
        },
        "validation": {
            "row_count": int(row["row_count"]),
            "positive_count": int(row["positive_count"]),
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "brier_score": float(row["brier_score"]),
        },
        "capacity": [
            {
                "capacity": float(capacity_row["capacity"]),
                "selected_count": int(capacity_row["selected_count"]),
                "captured_positive_count": int(
                    capacity_row["captured_positive_count"]
                ),
                "precision_at_k": float(capacity_row["precision_at_k"]),
                "recall_at_k": float(capacity_row["recall_at_k"]),
            }
            for capacity_row in capacity.to_dict(orient="records")
        ],
    }


def write_frozen_logistic_regression_evidence_snapshot(
    *,
    split_id: str,
    phase_9_contract_fingerprint: str,
    validation_results_path: Path | str = FROZEN_VALIDATION_RESULTS_PATH,
    capacity_results_path: Path | str = FROZEN_CAPACITY_RESULTS_PATH,
    output_path: Path | str = FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH,
) -> Path:
    """Freeze validated Month 1 LR evidence once, bound to its split and contract.

    The snapshot is write-once: re-running with identical evidence is a no-op,
    while different evidence raises instead of silently replacing the record.
    """
    if not split_id or not phase_9_contract_fingerprint:
        raise ValueError("Snapshot requires a split_id and Phase 9 fingerprint.")
    payload = _snapshot_payload(
        validation_path=Path(validation_results_path),
        capacity_path=Path(capacity_results_path),
        split_id=split_id,
        phase_9_contract_fingerprint=phase_9_contract_fingerprint,
    )
    text = json.dumps(payload, indent=2) + "\n"
    path = Path(output_path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(
                "Refusing to overwrite the frozen Logistic Regression evidence "
                f"snapshot with different content: {path}"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_frozen_logistic_regression_evidence_snapshot(
    path: Path | str = FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH,
    *,
    expected_split_id: str,
    expected_phase_9_contract_fingerprint: str,
) -> FrozenLogisticRegressionValidationEvidence:
    """Load the tracked LR snapshot and require it to match the current inputs."""
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            "Frozen Logistic Regression evidence snapshot does not exist: "
            f"{snapshot_path}"
        )
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Frozen Logistic Regression evidence snapshot is invalid: {snapshot_path}"
        ) from exc
    expected_keys = {
        "snapshot_version",
        "model",
        "evaluated_split",
        "split_id",
        "phase_9_contract_fingerprint",
        "derived_from",
        "validation",
        "capacity",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"Frozen evidence snapshot fields must be exactly {sorted(expected_keys)}."
        )
    if payload["snapshot_version"] != EVIDENCE_SNAPSHOT_VERSION:
        raise ValueError("Frozen evidence snapshot_version must be 1.")
    if payload["model"] != LOGISTIC_REGRESSION_MODEL_NAME:
        raise ValueError("Frozen evidence snapshot must describe Logistic Regression.")
    if payload["evaluated_split"] != "validation":
        raise ValueError("Frozen evidence snapshot must use validation only.")
    if payload["split_id"] != expected_split_id:
        raise ValueError(
            "Frozen Logistic Regression evidence was recorded for split "
            f"{payload['split_id']!r}, but the workflow loaded {expected_split_id!r}."
        )
    if payload["phase_9_contract_fingerprint"] != expected_phase_9_contract_fingerprint:
        raise ValueError(
            "Frozen Logistic Regression evidence does not match the current "
            "Phase 9 modelling contract fingerprint."
        )
    derived_from = payload["derived_from"]
    if (
        not isinstance(derived_from, dict)
        or not isinstance(payload["validation"], dict)
        or not isinstance(payload["capacity"], list)
        or not all(isinstance(row, dict) for row in payload["capacity"])
    ):
        raise ValueError("Frozen evidence snapshot structure is invalid.")
    row_count, positive_count, metrics = _normalize_frozen_evidence(
        payload["validation"],
        payload["capacity"],
    )
    return FrozenLogisticRegressionValidationEvidence(
        model_name=LOGISTIC_REGRESSION_MODEL_NAME,
        evaluated_split="validation",
        frozen=True,
        validation_source=Path(str(derived_from.get("validation_results", ""))),
        capacity_source=Path(str(derived_from.get("capacity_results", ""))),
        row_count=row_count,
        positive_count=positive_count,
        metrics=metrics,
        split_id=payload["split_id"],
        phase_9_contract_fingerprint=payload["phase_9_contract_fingerprint"],
        snapshot_source=snapshot_path,
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
    snapshot_source = (
        "not recorded"
        if frozen_logistic_regression.snapshot_source is None
        else _artifact_display_path(frozen_logistic_regression.snapshot_source)
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

The Logistic Regression values come from a tracked, write-once snapshot of the
frozen Month 1 validation evidence. The workflow verifies that the snapshot's
split ID and Phase 9 contract fingerprint match the inputs it loaded.
Logistic Regression is not retrained. Gradient Boosting values come from the
current Month 2 validation workflow, and every difference is
`Gradient Boosting - Logistic Regression`.

- Frozen evidence snapshot: `{snapshot_source}`
- Derived from Month 1 validation metrics: `{validation_source}`
- Derived from Month 1 capacity metrics: `{capacity_source}`
- Frozen split ID: `{frozen_logistic_regression.split_id}`
- Phase 9 contract fingerprint: `{frozen_logistic_regression.phase_9_contract_fingerprint}`
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
    phase_9_contract_fingerprint: str,
    model_implementation: str,
    model_class: str,
    configuration_version: int,
    model_configuration: Mapping[str, object],
    output_directory: Path | str = MONTH_2_REPORT_DIR,
    frozen_evidence_path: Path | str = FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH,
) -> GradientBoostingReportArtifacts:
    """Write all Phase 2 validation reports from completed workflow results."""
    frozen_logistic_regression = load_frozen_logistic_regression_evidence_snapshot(
        frozen_evidence_path,
        expected_split_id=split_id,
        expected_phase_9_contract_fingerprint=phase_9_contract_fingerprint,
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
