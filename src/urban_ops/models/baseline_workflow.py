"""Run Month 1 baseline modelling, evaluation, and report generation.

The workflow reconstructs the already-governed frozen preprocessing outputs
from Step 8 split artifacts, verifies Phase 8 and Phase 9 contracts, evaluates
Month 1 baselines, freezes one selected baseline, and only then evaluates the
untouched test split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Final, Sequence

import joblib
import pandas as pd

from urban_ops.eda.pipeline import load_eda_config
from urban_ops.eda.source import load_verified_split, resolve_split_run, verify_source_unchanged
from urban_ops.features.categorical_encoding import (
    fit_categorical_encoder,
    load_categorical_encoding_config,
    transform_split_categorical_encoder,
)
from urban_ops.features.categorical_missing import (
    load_categorical_missing_config,
    replace_split_categorical_missing,
)
from urban_ops.features.numeric_preprocessing import (
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_split_numeric_preprocessor,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_composition import (
    build_preprocessing_composition,
    compose_split_preprocessing_blocks,
    load_preprocessing_composition_config,
)
from urban_ops.features.preprocessing_verification import (
    load_preprocessing_verification_config,
    verify_preprocessing_contract,
)
from urban_ops.features.rare_unseen import (
    fit_rare_unseen_handler,
    load_rare_unseen_config,
    transform_split_rare_unseen,
)
from urban_ops.features.temporal import derive_split_temporal_features
from urban_ops.models.baseline_contract import (
    load_baseline_modelling_contract_config,
    verify_baseline_modelling_contract,
)
from urban_ops.models.baselines import (
    HistoricalRateBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    RuleBasedHistoricalRateBaseline,
    predict_from_scores,
)
from urban_ops.models.evaluation import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MANUAL_CLASSIFICATION_THRESHOLDS,
    PR_AUC_DEFINITION,
    SWEEP_CLASSIFICATION_THRESHOLDS,
    CalibrationEvaluation,
    ClassificationMetrics,
    RankingEvaluation,
    ThresholdMetrics,
    ThresholdPolicyResult,
    FrozenThresholdDecision,
    build_calibration_table,
    evaluate_binary_classifier,
    evaluate_calibration,
    evaluate_manual_thresholds,
    evaluate_ranking,
    evaluate_threshold,
    evaluate_threshold_selection_policies,
    evaluate_threshold_sweep,
    freeze_workload_limited_threshold,
    metrics_row,
)
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)

POLICY_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_baseline.yaml"
MISSING_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_categorical_missing.yaml"
CARDINALITY_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_categorical_cardinality.yaml"
ENCODING_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_categorical_encoding.yaml"
NUMERIC_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_numeric_preprocessing.yaml"
COMPOSITION_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_preprocessing_composition.yaml"
VERIFICATION_CONFIG_PATH: Final = PROJECT_ROOT / "configs/features/resolution_risk_preprocessing_verification.yaml"
MODELLING_CONFIG_PATH: Final = PROJECT_ROOT / "configs/models/resolution_risk_baseline_modelling_contract.yaml"
EDA_CONFIG_PATH: Final = PROJECT_ROOT / "configs/eda/resolution_risk.yaml"

BASELINE_REPORT_DIR: Final = PROJECT_ROOT / "reports/12_baseline_modelling"
TABLES_DIR: Final = BASELINE_REPORT_DIR / "tables"
FIGURES_DIR: Final = BASELINE_REPORT_DIR / "figures"
ROOT_TABLES_DIR: Final = PROJECT_ROOT / "reports/tables"
ROOT_FIGURES_DIR: Final = PROJECT_ROOT / "reports/figures"
MODEL_DIR: Final = PROJECT_ROOT / "models/baselines"
FROZEN_THRESHOLD_DECISION_PATH: Final = (
    PROJECT_ROOT / "configs/models/month1_logistic_regression_threshold_decision.json"
)

HISTORICAL_GROUP_COLUMN: Final = "created_month"
SUBGROUP_COLUMNS: Final = (
    "created_hour",
    "created_day_of_week",
    "created_month",
    "is_weekend",
)
SUBGROUP_MIN_COUNT: Final = 50
SELECTION_RULE: Final = (
    "Prefer models with non-constant validation risk scores; then maximize "
    "validation Recall@10%; break ties by PR-AUC, then lower Brier score, then "
    "simpler model order."
)


@dataclass(frozen=True)
class FrozenBaselineInputs:
    """Verified frozen modelling inputs consumed by Month 1 baselines."""

    matrices: dict[str, object]
    targets: dict[str, pd.Series]
    frames: dict[str, pd.DataFrame]
    feature_names: tuple[str, ...]
    phase_8_contract: object
    phase_9_contract: object
    split_id: str


@dataclass(frozen=True)
class BaselineWorkflowResult:
    """Key outputs from one completed baseline workflow run."""

    validation_results: pd.DataFrame
    test_results: pd.DataFrame
    subgroup_results: pd.DataFrame
    calibration_results: pd.DataFrame
    validation_calibration_results: pd.DataFrame
    logistic_validation_threshold_result: pd.DataFrame
    logistic_validation_manual_threshold_results: pd.DataFrame
    logistic_validation_sweep_results: pd.DataFrame
    logistic_validation_policy_results: pd.DataFrame
    frozen_threshold_decision: FrozenThresholdDecision
    frozen_threshold_decision_path: Path
    threshold_tradeoff_figure_paths: tuple[Path, ...]
    roc_curve_results: pd.DataFrame
    pr_curve_results: pd.DataFrame
    selected_model_name: str
    selected_threshold: float
    selected_artifact_path: Path


def load_frozen_baseline_inputs(
    *,
    eda_config_path: Path | str = EDA_CONFIG_PATH,
    split_run_path: Path | None = None,
) -> FrozenBaselineInputs:
    """Rebuild and verify the frozen Phase 8/9 model input contract."""
    eda_config = load_eda_config(eda_config_path)
    run_path = resolve_split_run(
        split_root=eda_config.split_root,
        latest_pointer=eda_config.latest_pointer,
        override=split_run_path,
    )
    source = load_verified_split(
        run_path=run_path,
        latest_pointer=eda_config.latest_pointer,
        required_completion_status=eda_config.required_completion_status,
        identifier_column=eda_config.identifier_column,
        target_column=eda_config.target_column,
        timestamp_column=eda_config.timestamp_column,
    )
    source_frames = {
        "train": source.train,
        "validation": source.validation,
        "test": source.test,
    }
    policy = load_feature_policy(POLICY_PATH)
    missing_config = load_categorical_missing_config(MISSING_CONFIG_PATH)
    cardinality_config = load_rare_unseen_config(
        CARDINALITY_CONFIG_PATH,
        missing_config=missing_config,
    )
    encoding_config = load_categorical_encoding_config(
        ENCODING_CONFIG_PATH,
        cardinality_config=cardinality_config,
    )
    numeric_config = load_numeric_preprocessing_config(NUMERIC_CONFIG_PATH)
    composition_config = load_preprocessing_composition_config(COMPOSITION_CONFIG_PATH)
    verification_config = load_preprocessing_verification_config(VERIFICATION_CONFIG_PATH)
    modelling_config = load_baseline_modelling_contract_config(MODELLING_CONFIG_PATH)

    derived_frames = derive_split_temporal_features(source_frames, policy=policy)
    missing_frames = replace_split_categorical_missing(
        derived_frames,
        policy=policy,
        config=missing_config,
    )
    fitted_cardinality = fit_rare_unseen_handler(
        missing_frames["train"],
        policy=policy,
        config=cardinality_config,
    )
    cardinality_frames = transform_split_rare_unseen(
        missing_frames,
        fitted=fitted_cardinality,
        config=cardinality_config,
    )
    fitted_categorical = fit_categorical_encoder(
        cardinality_frames["train"],
        policy=policy,
        config=encoding_config,
        fitted_cardinality=fitted_cardinality,
    )
    categorical_matrices = transform_split_categorical_encoder(
        cardinality_frames,
        fitted=fitted_categorical,
        config=encoding_config,
    )
    fitted_numeric = fit_numeric_preprocessor(
        cardinality_frames["train"],
        policy=policy,
        config=numeric_config,
    )
    numeric_matrices = transform_split_numeric_preprocessor(
        cardinality_frames,
        fitted=fitted_numeric,
        config=numeric_config,
    )
    fitted_composition = build_preprocessing_composition(
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        policy=policy,
        config=composition_config,
    )
    matrices = compose_split_preprocessing_blocks(
        categorical_matrices=categorical_matrices,
        numeric_matrices=numeric_matrices,
        fitted=fitted_composition,
    )
    targets = {
        split: frame[eda_config.target_column].astype(int)
        for split, frame in source_frames.items()
    }
    identifiers = {
        split: frame[eda_config.identifier_column]
        for split, frame in source_frames.items()
    }
    timestamps = {
        split: frame[eda_config.timestamp_column]
        for split, frame in source_frames.items()
    }
    feature_names = fitted_composition.combined_feature_names
    schemas = {split: feature_names for split in ("train", "validation", "test")}
    phase_8_contract = verify_preprocessing_contract(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names_by_split=schemas,
        target_name=eda_config.target_column,
        identifier_name=eda_config.identifier_column,
        timestamp_name=eda_config.timestamp_column,
        policy=policy,
        fitted_cardinality=fitted_cardinality,
        fitted_categorical=fitted_categorical,
        fitted_numeric=fitted_numeric,
        fitted_composition=fitted_composition,
        config=verification_config,
    )
    phase_9_contract = verify_baseline_modelling_contract(
        matrices=matrices,
        targets=targets,
        identifiers=identifiers,
        timestamps=timestamps,
        feature_names=feature_names,
        phase_8_contract=phase_8_contract,
        config=modelling_config,
        expected_phase_8_fingerprint=phase_8_contract.fingerprint,
        expected_schema_fingerprint=phase_8_contract.schema_fingerprint,
        policy=policy,
    )
    verify_source_unchanged(source)
    return FrozenBaselineInputs(
        matrices=matrices,
        targets=targets,
        frames=cardinality_frames,
        feature_names=feature_names,
        phase_8_contract=phase_8_contract,
        phase_9_contract=phase_9_contract,
        split_id=source.split_id,
    )


def _evaluate(
    model_name: str,
    y_true: pd.Series,
    y_pred: object,
    y_score: object,
) -> ClassificationMetrics:
    """Evaluate one model and log a compact trace line."""
    metrics = evaluate_binary_classifier(y_true, y_pred, y_score)
    LOGGER.info(
        "%s rows=%s f1=%.4f pr_auc=%.4f recall@10=%.4f",
        model_name,
        metrics.row_count,
        metrics.f1,
        metrics.pr_auc,
        metrics.recall_at_10_percent,
    )
    return metrics


def _fit_and_evaluate_validation(
    inputs: FrozenBaselineInputs,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    dict[str, RankingEvaluation],
    dict[str, CalibrationEvaluation],
    ThresholdMetrics,
    tuple[ThresholdMetrics, ...],
    tuple[ThresholdMetrics, ...],
]:
    """Fit train-only baselines and evaluate all models on validation."""
    X_train = inputs.matrices["train"]
    X_validation = inputs.matrices["validation"]
    y_train = inputs.targets["train"]
    y_validation = inputs.targets["validation"]
    validation_frame = inputs.frames["validation"]

    majority = MajorityClassBaseline().fit(X_train, y_train)
    majority_scores = majority.predict_proba(X_validation)[:, 1]
    majority_metrics = _evaluate(
        "Majority Class",
        y_validation,
        majority.predict(X_validation),
        majority_scores,
    )

    historical = HistoricalRateBaseline(HISTORICAL_GROUP_COLUMN).fit(
        inputs.frames["train"],
        y_train,
    )
    historical_scores = historical.predict_score(validation_frame)
    historical_threshold = float(y_train.mean())
    historical_metrics = _evaluate(
        "Historical Rate",
        y_validation,
        predict_from_scores(historical_scores, historical_threshold),
        historical_scores,
    )

    rule = RuleBasedHistoricalRateBaseline(historical)
    threshold_selection = rule.select_threshold(validation_frame, y_validation)
    rule_scores = rule.predict_score(validation_frame)
    rule_metrics = _evaluate(
        "Rule Based",
        y_validation,
        rule.predict(validation_frame),
        rule_scores,
    )

    logistic = LogisticRegressionBaseline().fit(
        X_train,
        y_train,
        feature_names=inputs.feature_names,
    )
    logistic_scores = logistic.predict_score(X_validation)
    logistic_metrics = _evaluate(
        "Logistic Regression",
        y_validation,
        logistic.predict(X_validation),
        logistic_scores,
    )

    rows = [
        {
            **metrics_row(
                "Majority Class", majority_metrics, evaluated_split="validation"
            ),
            "score_unique_count": int(pd.Series(majority_scores).nunique()),
        },
        {
            **metrics_row(
                "Historical Rate", historical_metrics, evaluated_split="validation"
            ),
            "score_unique_count": int(pd.Series(historical_scores).nunique()),
        },
        {
            **metrics_row("Rule Based", rule_metrics, evaluated_split="validation"),
            "score_unique_count": int(pd.Series(rule_scores).nunique()),
        },
        {
            **metrics_row(
                "Logistic Regression",
                logistic_metrics,
                evaluated_split="validation",
            ),
            "score_unique_count": int(pd.Series(logistic_scores).nunique()),
        },
    ]
    validation_results = pd.DataFrame(rows)
    ranking_evaluations = {
        "Majority Class": evaluate_ranking(y_validation, majority_scores),
        "Historical Rate": evaluate_ranking(y_validation, historical_scores),
        "Rule Based": evaluate_ranking(y_validation, rule_scores),
        "Logistic Regression": evaluate_ranking(y_validation, logistic_scores),
    }
    calibration_evaluations = {
        "Majority Class": evaluate_calibration(y_validation, majority_scores),
        "Historical Rate": evaluate_calibration(y_validation, historical_scores),
        "Rule Based": evaluate_calibration(y_validation, rule_scores),
        "Logistic Regression": evaluate_calibration(y_validation, logistic_scores),
    }
    logistic_threshold_metrics = evaluate_threshold(
        y_validation,
        logistic_scores,
        threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    logistic_manual_threshold_metrics = evaluate_manual_thresholds(
        y_validation,
        logistic_scores,
    )
    logistic_sweep_metrics = evaluate_threshold_sweep(
        y_validation,
        logistic_scores,
    )
    model_objects: dict[str, object] = {
        "Majority Class": majority,
        "Historical Rate": historical,
        "Rule Based": rule,
        "Logistic Regression": logistic,
        "threshold_selection": threshold_selection,
        "historical_threshold": historical_threshold,
    }
    subgroup = build_subgroup_analysis(
        frame=validation_frame,
        y_true=y_validation,
        y_pred=logistic.predict(X_validation),
        y_score=logistic_scores,
        split_name="validation",
        model_name="Logistic Regression",
    )
    return (
        validation_results,
        model_objects,
        subgroup,
        ranking_evaluations,
        calibration_evaluations,
        logistic_threshold_metrics,
        logistic_manual_threshold_metrics,
        logistic_sweep_metrics,
    )


def select_baseline(validation_results: pd.DataFrame) -> str:
    """Select the Month 1 baseline using the documented validation rule."""
    simplicity = {
        "Majority Class": 0,
        "Historical Rate": 1,
        "Rule Based": 2,
        "Logistic Regression": 3,
    }
    ranked = validation_results.assign(
        simplicity_rank=validation_results["model"].map(simplicity),
        ranking_signal=validation_results["score_unique_count"].gt(1),
    ).sort_values(
        by=[
            "ranking_signal",
            "recall_at_10_percent",
            "pr_auc",
            "brier_score",
            "simplicity_rank",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def _split_predictions(
    selected_model_name: str,
    model_objects: dict[str, object],
    inputs: FrozenBaselineInputs,
    *,
    split: str,
) -> tuple[object, object, float]:
    """Return split predictions, scores, and frozen binary threshold."""
    matrix = inputs.matrices[split]
    frame = inputs.frames[split]
    if selected_model_name == "Majority Class":
        model = model_objects[selected_model_name]
        return model.predict(matrix), model.predict_proba(matrix)[:, 1], 0.5
    if selected_model_name == "Historical Rate":
        model = model_objects[selected_model_name]
        scores = model.predict_score(frame)
        threshold = float(model_objects["historical_threshold"])
        return predict_from_scores(scores, threshold), scores, threshold
    if selected_model_name == "Rule Based":
        model = model_objects[selected_model_name]
        scores = model.predict_score(frame)
        threshold = model.threshold_selection_.selected_threshold
        return model.predict(frame), scores, threshold
    if selected_model_name == "Logistic Regression":
        model = model_objects[selected_model_name]
        return model.predict(matrix), model.predict_score(matrix), 0.5
    raise RuntimeError(f"Unsupported selected model: {selected_model_name}")


def build_subgroup_analysis(
    *,
    frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: object,
    y_score: object,
    split_name: str,
    model_name: str,
    minimum_count: int = SUBGROUP_MIN_COUNT,
) -> pd.DataFrame:
    """Calculate descriptive subgroup/error metrics for approved groups."""
    base = pd.DataFrame(
        {
            "y_true": y_true.astype(int).to_numpy(),
            "y_pred": pd.Series(y_pred).astype(int).to_numpy(),
            "y_score": pd.Series(y_score).astype(float).to_numpy(),
        },
        index=frame.index,
    )
    rows: list[dict[str, object]] = []
    for column in SUBGROUP_COLUMNS:
        if column not in frame:
            continue
        working = base.assign(group_value=frame[column].astype("string"))
        for group_value, group in working.groupby("group_value", sort=True):
            row_count = len(group)
            if row_count < minimum_count:
                continue
            true_positive = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
            false_positive = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
            false_negative = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
            predicted_positive = int(group["y_pred"].sum())
            actual_positive = int(group["y_true"].sum())
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "group_column": column,
                    "group_value": str(group_value),
                    "row_count": row_count,
                    "actual_positive_rate": float(group["y_true"].mean()),
                    "mean_predicted_risk": float(group["y_score"].mean()),
                    "precision": (
                        float(true_positive / predicted_positive)
                        if predicted_positive
                        else 0.0
                    ),
                    "recall": (
                        float(true_positive / actual_positive)
                        if actual_positive
                        else 0.0
                    ),
                    "false_positives": false_positive,
                    "false_negatives": false_negative,
                    "minimum_count": minimum_count,
                }
            )
    return pd.DataFrame(rows)


def build_ranking_curve_tables(
    evaluations: Mapping[str, RankingEvaluation],
    *,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build machine-readable ROC and PR curve tables for evaluated scores."""
    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    for model_name, evaluation in evaluations.items():
        curves = evaluation.curves
        roc_points = zip(
            curves.roc_false_positive_rate,
            curves.roc_true_positive_rate,
            curves.roc_thresholds,
        )
        for point_index, (
            false_positive_rate,
            true_positive_rate,
            threshold,
        ) in enumerate(roc_points):
            roc_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "point_index": point_index,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "threshold": threshold,
                    "roc_auc": evaluation.metrics.roc_auc,
                }
            )
        for point_index, (precision, recall) in enumerate(
            zip(curves.pr_precision, curves.pr_recall)
        ):
            threshold = (
                curves.pr_thresholds[point_index]
                if point_index < len(curves.pr_thresholds)
                else float("nan")
            )
            pr_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "point_index": point_index,
                    "recall": recall,
                    "precision": precision,
                    "threshold": threshold,
                    "positive_rate": evaluation.metrics.positive_rate,
                    "pr_auc": evaluation.metrics.pr_auc,
                    "pr_auc_definition": PR_AUC_DEFINITION,
                }
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def build_validation_calibration_table(
    evaluations: Mapping[str, CalibrationEvaluation],
) -> pd.DataFrame:
    """Build populated validation calibration points for all baselines."""
    rows: list[dict[str, object]] = []
    for model_name, evaluation in evaluations.items():
        for point_index, (
            mean_predicted_probability,
            observed_positive_rate,
        ) in enumerate(
            zip(
                evaluation.curve.mean_predicted_probability,
                evaluation.curve.observed_positive_rate,
            )
        ):
            rows.append(
                {
                    "split": "validation",
                    "model": model_name,
                    "point_index": point_index,
                    "mean_predicted_probability": mean_predicted_probability,
                    "observed_positive_rate": observed_positive_rate,
                    "requested_bin_count": evaluation.curve.requested_bin_count,
                    "strategy": evaluation.curve.strategy,
                    "positive_rate": evaluation.metrics.positive_rate,
                    "brier_score": evaluation.metrics.brier_score,
                }
            )
    return pd.DataFrame(rows)


def build_logistic_validation_threshold_table(
    metrics: ThresholdMetrics,
) -> pd.DataFrame:
    """Build the one-row Phase 4.1 validation operating-point table."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                **metrics.to_dict(),
            }
        ]
    )


def build_logistic_validation_manual_threshold_table(
    metrics: tuple[ThresholdMetrics, ...],
) -> pd.DataFrame:
    """Build the ordered five-row Phase 4.2 validation comparison."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                **result.to_dict(),
            }
            for result in metrics
        ]
    )


def _write_phase_4_1_outputs(results: pd.DataFrame) -> None:
    """Write Phase 4.1's validation-only CSV and operational summary."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_1_default_threshold.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    filename = "logistic_regression_validation_threshold_050.csv"
    for directory in (TABLES_DIR, ROOT_TABLES_DIR):
        results.to_csv(directory / filename, index=False)

    row = results.iloc[0]
    report = f"""# Phase 4.1 — Default Threshold 0.50

Positive class `1` means missed resolution target. This report evaluates the
existing Logistic Regression validation probabilities at one fixed operating
point. It does not select or optimize a threshold.

{_format_phase_4_1_table(results)}

At threshold {row["threshold"]:.2f}, {int(row["predicted_positive_count"]):,} of
{int(row["sample_count"]):,} validation complaints
({row["predicted_positive_rate"] * 100:.1f}%) are flagged as at risk. The flags
capture {int(row["true_positives"]):,} of
{int(row["actual_positive_count"]):,} complaints that actually miss their target
({row["recall"] * 100:.1f}% recall). Of the flagged complaints,
{int(row["true_positives"]):,} of {int(row["predicted_positive_count"]):,} actually
miss their target ({row["precision"] * 100:.1f}% precision).
These values describe the current operating point without judging it as good,
bad, optimal, or selected.
"""
    (BASELINE_REPORT_DIR / "phase_4_1_default_threshold.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")


def _format_phase_4_2_comparisons(results: pd.DataFrame) -> str:
    """Describe every manual threshold relative to the 0.50 reference row."""
    reference = results.loc[
        results["threshold"].eq(DEFAULT_CLASSIFICATION_THRESHOLD)
    ].iloc[0]
    lines: list[str] = []
    for _, row in results.iterrows():
        if row["threshold"] == DEFAULT_CLASSIFICATION_THRESHOLD:
            continue
        lines.append(
            "- At "
            f"{row['threshold']:.2f} versus 0.50: flagged "
            f"{int(row['predicted_positive_count']):,} "
            f"({int(row['predicted_positive_count'] - reference['predicted_positive_count']):+,} complaints); "
            f"TP {int(row['true_positives']):,} "
            f"({int(row['true_positives'] - reference['true_positives']):+,}); "
            f"FP {int(row['false_positives']):,} "
            f"({int(row['false_positives'] - reference['false_positives']):+,}); "
            f"TN {int(row['true_negatives']):,} "
            f"({int(row['true_negatives'] - reference['true_negatives']):+,}); "
            f"FN {int(row['false_negatives']):,} "
            f"({int(row['false_negatives'] - reference['false_negatives']):+,}); "
            f"precision {row['precision']:.4f} "
            f"({row['precision'] - reference['precision']:+.4f}); "
            f"recall {row['recall']:.4f} "
            f"({row['recall'] - reference['recall']:+.4f}); "
            f"F1 {row['f1']:.4f} ({row['f1'] - reference['f1']:+.4f}); "
            f"flagged rate {row['predicted_positive_rate']:.4f} "
            f"({row['predicted_positive_rate'] - reference['predicted_positive_rate']:+.4f})."
        )
    return "\n".join(lines)


def _format_phase_4_2_directional_changes(results: pd.DataFrame) -> str:
    """Summarize observed endpoint and non-monotonic metric sequences."""
    first = results.iloc[0]
    last = results.iloc[-1]
    precision_path = " -> ".join(f"{value:.4f}" for value in results["precision"])
    f1_path = " -> ".join(f"{value:.4f}" for value in results["f1"])
    return (
        f"Across the ordered comparison from {first['threshold']:.2f} to "
        f"{last['threshold']:.2f}, flagged complaints changed from "
        f"{int(first['predicted_positive_count']):,} to "
        f"{int(last['predicted_positive_count']):,}, true positives from "
        f"{int(first['true_positives']):,} to {int(last['true_positives']):,}, "
        f"false positives from {int(first['false_positives']):,} to "
        f"{int(last['false_positives']):,}, and recall from "
        f"{first['recall']:.4f} to {last['recall']:.4f}. True negatives "
        f"changed from {int(first['true_negatives']):,} to "
        f"{int(last['true_negatives']):,}, while false negatives changed from "
        f"{int(first['false_negatives']):,} to "
        f"{int(last['false_negatives']):,}. Observed precision followed "
        f"{precision_path}; observed F1 followed {f1_path}. Precision and F1 "
        "are reported as observations, not assumed to be monotonic."
    )


def _write_phase_4_2_outputs(results: pd.DataFrame) -> None:
    """Write Phase 4.2's validation-only manual comparison outputs."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_2_manual_threshold_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    filename = "logistic_regression_validation_manual_thresholds.csv"
    for directory in (TABLES_DIR, ROOT_TABLES_DIR):
        results.to_csv(directory / filename, index=False)

    report = f"""# Phase 4.2 — Manual Threshold Comparison

The same Logistic Regression validation probability vector is evaluated at the
five manually specified thresholds. Model probabilities, ranking, ROC-AUC, and
PR-AUC do not change; only the hard-classification operating point changes. No
threshold is ranked, optimized, recommended, or selected.

{_format_phase_4_2_table(results)}

## Observed Directional Changes

{_format_phase_4_2_directional_changes(results)}

## Comparisons With 0.50

{_format_phase_4_2_comparisons(results)}
"""
    (BASELINE_REPORT_DIR / "phase_4_2_manual_threshold_comparison.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")


def build_logistic_validation_sweep_table(
    metrics: tuple[ThresholdMetrics, ...],
) -> pd.DataFrame:
    """Build the ordered Phase 4.3 validation threshold sweep table."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                **result.to_dict(),
            }
            for result in metrics
        ]
    )


def build_logistic_validation_policy_table(
    results: tuple[ThresholdPolicyResult, ...],
) -> pd.DataFrame:
    """Build the ordered Phase 4.5 validation policy-candidate table."""
    return pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                **result.to_dict(),
            }
            for result in results
        ]
    )


def _phase_4_3_focus_region(results: pd.DataFrame) -> pd.DataFrame:
    """Return the 0.40-0.50 sweep rows used for transition reporting."""
    focus = results.loc[results["threshold"].between(0.40, 0.50)]
    return focus.reset_index(drop=True)


def _format_phase_4_3_checkpoint_table(results: pd.DataFrame) -> str:
    """Render the Phase 4.2 checkpoint thresholds as they appear in the sweep."""
    checkpoints = results.loc[
        results["threshold"].isin(MANUAL_CLASSIFICATION_THRESHOLDS)
    ]
    return _format_phase_4_1_table(checkpoints)


def _format_phase_4_3_focus_table(results: pd.DataFrame) -> str:
    """Render the focused 0.40-0.50 transition region."""
    return _format_phase_4_1_table(_phase_4_3_focus_region(results))


def _format_phase_4_3_transition_summary(results: pd.DataFrame) -> str:
    """Describe where the 0.40-0.50 flagged-rate reduction actually occurs."""
    focus = _phase_4_3_focus_region(results)
    first = focus.iloc[0]
    last = focus.iloc[-1]
    flagged_steps = focus["predicted_positive_count"].diff().iloc[1:]
    largest_step_position = int(flagged_steps.abs().idxmax())
    before = focus.iloc[largest_step_position - 1]
    after = focus.iloc[largest_step_position]
    return (
        f"From threshold {first['threshold']:.2f} to {last['threshold']:.2f}, "
        f"flagged complaints fall from {int(first['predicted_positive_count']):,} "
        f"({first['predicted_positive_rate'] * 100:.1f}%) to "
        f"{int(last['predicted_positive_count']):,} "
        f"({last['predicted_positive_rate'] * 100:.1f}%), true positives fall "
        f"from {int(first['true_positives']):,} to {int(last['true_positives']):,}, "
        f"false positives fall from {int(first['false_positives']):,} to "
        f"{int(last['false_positives']):,}, true negatives rise from "
        f"{int(first['true_negatives']):,} to {int(last['true_negatives']):,}, "
        f"false negatives rise from {int(first['false_negatives']):,} to "
        f"{int(last['false_negatives']):,}, and recall falls from "
        f"{first['recall']:.4f} to {last['recall']:.4f}. Precision moves from "
        f"{first['precision']:.4f} to {last['precision']:.4f} and F1 from "
        f"{first['f1']:.4f} to {last['f1']:.4f}; neither is assumed to move "
        "monotonically. The single largest step change in flagged complaints "
        f"within this region occurs between {before['threshold']:.2f} and "
        f"{after['threshold']:.2f}, where flagged complaints move from "
        f"{int(before['predicted_positive_count']):,} to "
        f"{int(after['predicted_positive_count']):,} "
        f"({int(after['predicted_positive_count'] - before['predicted_positive_count']):+,}). "
        "Threshold changes alter hard-classification metrics, not the "
        "underlying ranking scores: ROC-AUC and PR-AUC are unaffected."
    )


def _threshold_sweep_plot_data(results: pd.DataFrame) -> pd.DataFrame:
    """Return Phase 4.3 rows in deterministic threshold order for plotting."""
    ordered = results.sort_values("threshold", kind="mergesort").reset_index(drop=True)
    if not ordered["threshold"].is_monotonic_increasing:
        raise RuntimeError("Threshold sweep rows must be ordered by threshold.")
    if len(ordered) != len(results):
        raise RuntimeError("Threshold sweep plotting lost one or more rows.")
    return ordered


def _phase_4_4_figure_paths() -> tuple[Path, ...]:
    """Return deterministic Phase 4.4 figure artifact paths."""
    filenames = (
        "logistic_regression_validation_threshold_precision_recall_f1.png",
        "logistic_regression_validation_threshold_flagged_rate.png",
        "logistic_regression_validation_threshold_classification_counts.png",
    )
    return tuple(ROOT_FIGURES_DIR / filename for filename in filenames)


def _write_threshold_tradeoff_figures(results: pd.DataFrame) -> tuple[Path, ...]:
    """Plot Phase 4.4 threshold trade-offs from existing sweep rows only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _threshold_sweep_plot_data(results)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_figure, metric_axis = plt.subplots(figsize=(8, 5), dpi=150)
    metric_axis.plot(data["threshold"], data["precision"], label="Precision")
    metric_axis.plot(data["threshold"], data["recall"], label="Recall")
    metric_axis.plot(data["threshold"], data["f1"], label="F1")
    metric_axis.set_xlabel("Classification threshold")
    metric_axis.set_ylabel("Metric value")
    metric_axis.set_title("Threshold vs Precision, Recall, and F1")
    metric_axis.set_xlim(0, 1)
    metric_axis.set_ylim(0, 1)
    metric_axis.legend()
    metric_figure.tight_layout()

    workload_figure, workload_axis = plt.subplots(figsize=(8, 5), dpi=150)
    workload_axis.plot(
        data["threshold"],
        data["predicted_positive_rate"],
        color="C3",
        label="Predicted-positive rate",
    )
    workload_axis.set_xlabel("Classification threshold")
    workload_axis.set_ylabel("Predicted-positive rate")
    workload_axis.set_title("Threshold vs Predicted-Positive Rate")
    workload_axis.set_xlim(0, 1)
    workload_axis.set_ylim(0, 1)
    workload_axis.legend()
    workload_figure.tight_layout()

    counts_figure, counts_axis = plt.subplots(figsize=(8, 5), dpi=150)
    counts_axis.plot(
        data["threshold"], data["true_positives"], label="True positives"
    )
    counts_axis.plot(
        data["threshold"], data["false_positives"], label="False positives"
    )
    counts_axis.plot(
        data["threshold"], data["false_negatives"], label="False negatives"
    )
    counts_axis.set_xlabel("Classification threshold")
    counts_axis.set_ylabel("Complaint count")
    counts_axis.set_title("Threshold vs Classification Counts")
    counts_axis.set_xlim(0, 1)
    counts_axis.legend()
    counts_figure.tight_layout()

    figure_artifacts = (
        (
            metric_figure,
            "logistic_regression_validation_threshold_precision_recall_f1.png",
        ),
        (
            workload_figure,
            "logistic_regression_validation_threshold_flagged_rate.png",
        ),
        (
            counts_figure,
            "logistic_regression_validation_threshold_classification_counts.png",
        ),
    )
    for directory in (FIGURES_DIR, ROOT_FIGURES_DIR):
        for figure, filename in figure_artifacts:
            figure.savefig(directory / filename)
    for figure, _ in figure_artifacts:
        plt.close(figure)
    return _phase_4_4_figure_paths()


def _largest_step_change(
    results: pd.DataFrame, column: str
) -> tuple[pd.Series, pd.Series, float]:
    """Return adjacent rows around the largest absolute one-step column change."""
    data = _threshold_sweep_plot_data(results)
    changes = data[column].diff().iloc[1:]
    largest_index = int(changes.abs().idxmax())
    before = data.iloc[largest_index - 1]
    after = data.iloc[largest_index]
    return before, after, float(after[column] - before[column])


def _format_phase_4_4_sensitivity_summary(results: pd.DataFrame) -> str:
    """Describe Phase 4.4 threshold trade-offs using actual sweep values."""
    data = _threshold_sweep_plot_data(results)
    first = data.iloc[0]
    last = data.iloc[-1]
    workload_before, workload_after, workload_delta = _largest_step_change(
        data, "predicted_positive_count"
    )
    recall_before, recall_after, recall_delta = _largest_step_change(data, "recall")
    fn_before, fn_after, fn_delta = _largest_step_change(data, "false_negatives")
    fp_before, fp_after, fp_delta = _largest_step_change(data, "false_positives")
    precision_min = float(data["precision"].min())
    precision_max = float(data["precision"].max())
    f1_min = float(data["f1"].min())
    f1_max = float(data["f1"].max())
    return (
        f"Across the full sweep from {first['threshold']:.2f} to "
        f"{last['threshold']:.2f}, flagged complaints move from "
        f"{int(first['predicted_positive_count']):,} "
        f"({first['predicted_positive_rate'] * 100:.1f}%) to "
        f"{int(last['predicted_positive_count']):,} "
        f"({last['predicted_positive_rate'] * 100:.1f}%). Recall moves from "
        f"{first['recall']:.4f} to {last['recall']:.4f}, true positives from "
        f"{int(first['true_positives']):,} to {int(last['true_positives']):,}, "
        f"false positives from {int(first['false_positives']):,} to "
        f"{int(last['false_positives']):,}, and false negatives from "
        f"{int(first['false_negatives']):,} to "
        f"{int(last['false_negatives']):,}. Precision ranges from "
        f"{precision_min:.4f} to {precision_max:.4f}; F1 ranges from "
        f"{f1_min:.4f} to {f1_max:.4f}. The largest one-step workload change "
        f"occurs between {workload_before['threshold']:.2f} and "
        f"{workload_after['threshold']:.2f}, where flagged complaints change by "
        f"{int(workload_delta):+,}. The largest one-step recall change occurs "
        f"between {recall_before['threshold']:.2f} and "
        f"{recall_after['threshold']:.2f} ({recall_delta:+.4f}). False "
        f"negatives rise most sharply between {fn_before['threshold']:.2f} and "
        f"{fn_after['threshold']:.2f} ({int(fn_delta):+,}), while false "
        f"positives fall most sharply between {fp_before['threshold']:.2f} and "
        f"{fp_after['threshold']:.2f} ({int(fp_delta):+,}). These are "
        "threshold-sensitive regions, not selected or recommended thresholds."
    )


def _write_phase_4_4_outputs(results: pd.DataFrame) -> tuple[Path, ...]:
    """Write Phase 4.4's threshold trade-off figures and report."""
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_4_threshold_tradeoffs.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_paths = _write_threshold_tradeoff_figures(results)
    report = f"""# Phase 4.4 — Threshold Trade-Off Visualizations

Phase 4.4 visualizes the existing Phase 4.3 Logistic Regression validation
threshold sweep. It does not fit a model, regenerate probabilities, recompute
threshold metrics independently, score the protected test set, rank thresholds,
or select an operating point.

Threshold movement does not change model coefficients, probability scores,
score ordering, ROC-AUC, or PR-AUC. It does change hard predictions,
TP/FP/TN/FN, precision, recall, F1, and predicted-positive workload.

## Figures

- Precision / Recall / F1 vs Threshold:
  `reports/figures/{figure_paths[0].name}`
- Predicted-Positive Rate vs Threshold:
  `reports/figures/{figure_paths[1].name}`
- Classification Counts vs Threshold:
  `reports/figures/{figure_paths[2].name}`

## Threshold Sensitivity

{_format_phase_4_4_sensitivity_summary(results)}

The flagged-rate plot is the workload view: it shows the share of validation
complaints that would be sent for attention at each hard-classification
threshold. The classification-count plot shows the operational trade-off behind
recall: as fewer complaints are flagged, true positives and false positives
fall, while missed actual positives become false negatives. The figures make
threshold-sensitive regions visible, but they do not decide which threshold
should be used.

Full sweep data: `reports/tables/logistic_regression_validation_threshold_sweep.csv`
"""
    (BASELINE_REPORT_DIR / "phase_4_4_threshold_tradeoffs.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return figure_paths


def _format_phase_4_5_policy_interpretation(results: pd.DataFrame) -> str:
    """Describe candidate threshold policies without choosing between them."""
    lines: list[str] = []
    for _, row in results.iterrows():
        if not bool(row["constraint_satisfied"]):
            lines.append(
                f"- {row['policy_name']}: no candidate threshold satisfies "
                f"`{row['constraint_name']}` = {row['constraint_value']:.4f}. "
                f"{row['selection_reason']}"
            )
            continue
        lines.append(
            f"- {row['policy_name']}: candidate threshold "
            f"{row['candidate_threshold']:.2f}; precision "
            f"{row['precision']:.4f}, recall {row['recall']:.4f}, F1 "
            f"{row['f1']:.4f}, flagged rate "
            f"{row['predicted_positive_rate']:.4f}, FP "
            f"{int(row['false_positives']):,}, FN "
            f"{int(row['false_negatives']):,}. {row['selection_reason']}"
        )
    return "\n".join(lines)


def _write_phase_4_5_outputs(results: pd.DataFrame) -> None:
    """Write Phase 4.5's validation-only policy candidate comparison."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_5_threshold_policy_candidates.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    filename = "logistic_regression_validation_threshold_policy_candidates.csv"
    for directory in (TABLES_DIR, ROOT_TABLES_DIR):
        results.to_csv(directory / filename, index=False)

    report = f"""# Phase 4.5 — Threshold-Selection Policy Candidates

Phase 4.5 applies explicit candidate threshold-selection policies to the
existing Phase 4.3 Logistic Regression validation sweep. It does not retrain a
model, regenerate probabilities, rebuild the sweep, score the protected test
set, choose a final policy, freeze a threshold, or declare one policy superior.

Different policies answer different questions, so there is no universally best
threshold without first choosing the decision policy that reflects the
operational objective.

Tie-breaking is deterministic: after the policy objective is evaluated, exact
ties prefer the highest threshold. This generally keeps equal-objective
candidates at equal or lower workload.

{_format_phase_4_5_table(results)}

## Policy Interpretation

{_format_phase_4_5_policy_interpretation(results)}

The Max-F1 policy is a metric-balance reference. The minimum-recall policy asks
which threshold maintains detection coverage. The minimum-precision policy asks
which threshold maintains enough trust in flags. The workload policy asks which
threshold keeps the flagged share within a fixed validation percentage while
capturing as many actual misses as possible.

Policy data: `reports/tables/{filename}`
"""
    (BASELINE_REPORT_DIR / "phase_4_5_threshold_policy_candidates.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")


def write_frozen_threshold_decision(
    decision: FrozenThresholdDecision,
    *,
    path: Path = FROZEN_THRESHOLD_DECISION_PATH,
) -> Path:
    """Persist the authoritative Phase 4.6 frozen threshold JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_frozen_threshold_decision(
    path: Path = FROZEN_THRESHOLD_DECISION_PATH,
) -> FrozenThresholdDecision:
    """Load the authoritative frozen threshold decision for later phases."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FrozenThresholdDecision(**payload)


def _write_phase_4_6_outputs(
    decision: FrozenThresholdDecision,
    *,
    decision_path: Path = FROZEN_THRESHOLD_DECISION_PATH,
) -> Path:
    """Write Phase 4.6's frozen threshold artifact and decision report."""
    artifact_path = write_frozen_threshold_decision(decision, path=decision_path)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_6_frozen_threshold_decision.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Phase 4.6 — Frozen Threshold Decision

Phase 4.6 adopts the approved workload-limited policy and freezes the
validation-selected threshold for later evaluation. The policy is chosen first;
the threshold is the validation result produced by that policy.

Approved policy:

- Constraint: `predicted_positive_rate <= {decision.constraint_value:.2f}`
- Secondary objective: `{decision.secondary_objective}`
- Selected on split: `{decision.selected_on_split}`
- Frozen: `{str(decision.frozen).lower()}`

The project adopts the workload-limited policy because this baseline is intended
to prioritize a manageable subset of complaints rather than flag nearly the
entire population. Under that policy, Phase 4.5 selected threshold
`{decision.selected_threshold:.2f}` on validation.

## Validation Evidence

| metric | value |
| --- | ---: |
| selected_threshold | {decision.selected_threshold:.4f} |
| precision | {decision.precision:.4f} |
| recall | {decision.recall:.4f} |
| f1 | {decision.f1:.4f} |
| true_positives | {decision.true_positives} |
| false_positives | {decision.false_positives} |
| true_negatives | {decision.true_negatives} |
| false_negatives | {decision.false_negatives} |
| predicted_positive_count | {decision.predicted_positive_count} |
| predicted_positive_rate | {decision.predicted_positive_rate:.4f} |

## Policy Context

Other Phase 4.5 candidates were not adopted as the operating policy:

- Max F1 selected `0.34` and flagged about `99.66%` of validation complaints.
- Recall >= 0.70 selected `0.43` and flagged about `70.36%`.
- Precision >= 0.50 selected `0.49`, the same candidate as the workload policy.

Threshold `0.49` is not universally optimal and is not guaranteed to be best in
production. It is the validation-selected threshold under the approved
workload-limited policy. Phase 4.7 can load the frozen JSON artifact and apply
this threshold without rerunning threshold selection.

Machine-readable decision artifact:
`{artifact_path.relative_to(PROJECT_ROOT)}`
"""
    (BASELINE_REPORT_DIR / "phase_4_6_frozen_threshold_decision.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return artifact_path


def _write_phase_4_3_outputs(results: pd.DataFrame) -> None:
    """Write Phase 4.3's deterministic validation-only threshold sweep."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/phase_4_3_threshold_sweep.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    filename = "logistic_regression_validation_threshold_sweep.csv"
    for directory in (TABLES_DIR, ROOT_TABLES_DIR):
        results.to_csv(directory / filename, index=False)

    first_row = results.iloc[0]
    report = f"""# Phase 4.3 — Deterministic Threshold Sweep

The same Logistic Regression validation probability vector used in Phase 4.1
and Phase 4.2 is evaluated across a deterministic threshold grid from
`{SWEEP_CLASSIFICATION_THRESHOLDS[0]:.2f}` to
`{SWEEP_CLASSIFICATION_THRESHOLDS[-1]:.2f}` in steps of `0.01`, producing
`{len(SWEEP_CLASSIFICATION_THRESHOLDS)}` rows in ascending threshold order.
Model probabilities, ranking, ROC-AUC, and PR-AUC do not change; only the
hard-classification operating point changes. This sweep is descriptive only:
no threshold is ranked, optimized, recommended, or selected.

Validation population: `{int(first_row["sample_count"]):,}` samples,
`{int(first_row["actual_positive_count"]):,}` actual positives
(`{first_row["actual_positive_rate"] * 100:.2f}%` prevalence).

## Phase 4.2 Checkpoints Within The Sweep

{_format_phase_4_3_checkpoint_table(results)}

## 0.40-0.50 Transition Region

{_format_phase_4_3_focus_table(results)}

{_format_phase_4_3_transition_summary(results)}

Full sweep data: `reports/tables/{filename}`
"""
    (BASELINE_REPORT_DIR / "phase_4_3_threshold_sweep.md").write_text(
        report,
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")


def _write_tables(
    *,
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    subgroup_results: pd.DataFrame,
    calibration_results: pd.DataFrame,
    validation_calibration_results: pd.DataFrame,
    roc_curve_results: pd.DataFrame,
    pr_curve_results: pd.DataFrame,
) -> None:
    """Persist root-compatible and phase-local machine-readable outputs."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (TABLES_DIR, ROOT_TABLES_DIR):
        validation_results.to_csv(directory / "baseline_validation_results.csv", index=False)
        test_results.to_csv(directory / "baseline_test_results.csv", index=False)
        subgroup_results.to_csv(directory / "baseline_subgroup_results.csv", index=False)
        calibration_results.to_csv(
            directory / "logistic_regression_calibration.csv",
            index=False,
        )
        validation_calibration_results.to_csv(
            directory / "baseline_validation_calibration_curve.csv",
            index=False,
        )
        roc_curve_results.to_csv(
            directory / "baseline_validation_roc_curve.csv",
            index=False,
        )
        pr_curve_results.to_csv(
            directory / "baseline_validation_pr_curve.csv",
            index=False,
        )


def build_selected_calibration(
    *,
    inputs: FrozenBaselineInputs,
    selected_model_name: str,
    validation_score: object,
    test_score: object,
) -> pd.DataFrame:
    """Build deterministic validation/test calibration-bin evidence."""
    rows: list[pd.DataFrame] = []
    for split, scores in (
        ("validation", validation_score),
        ("test", test_score),
    ):
        table = build_calibration_table(inputs.targets[split], scores, n_bins=10)
        table.insert(0, "split", split)
        table.insert(1, "model", selected_model_name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _write_calibration_figure(calibration_results: pd.DataFrame) -> None:
    """Write a deterministic calibration diagnostic plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4), dpi=150)
    axis.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Ideal")
    for split in ("validation", "test"):
        table = calibration_results.loc[
            calibration_results["split"].eq(split)
            & calibration_results["row_count"].gt(0)
        ]
        axis.plot(
            table["mean_predicted_risk"],
            table["observed_positive_rate"],
            marker="o",
            linewidth=1.5,
            label=split.title(),
        )
    axis.set_xlabel("Mean predicted risk")
    axis.set_ylabel("Observed missed-target rate")
    axis.set_title("Logistic regression calibration bins")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    for directory in (FIGURES_DIR, ROOT_FIGURES_DIR):
        figure.savefig(directory / "logistic_regression_calibration.png")
    plt.close(figure)


def _write_validation_calibration_figure(
    calibration_results: pd.DataFrame,
) -> None:
    """Write all-baseline validation calibration points and ideal reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5), dpi=150)
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="0.5",
        linestyle="--",
        linewidth=1,
        label="Ideal calibration",
    )
    for model_name in calibration_results["model"].drop_duplicates():
        model_points = calibration_results.loc[
            calibration_results["model"].eq(model_name)
        ]
        brier_score = float(model_points["brier_score"].iloc[0])
        axis.plot(
            model_points["mean_predicted_probability"],
            model_points["observed_positive_rate"],
            marker="o",
            linewidth=1.5,
            label=f"{model_name} (Brier={brier_score:.4f})",
        )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed missed-target rate")
    axis.set_title("Validation calibration curves")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for directory in (FIGURES_DIR, ROOT_FIGURES_DIR):
        figure.savefig(directory / "baseline_validation_calibration_curve.png")
    plt.close(figure)


def _write_ranking_figures(
    *,
    validation_results: pd.DataFrame,
    roc_curve_results: pd.DataFrame,
    pr_curve_results: pd.DataFrame,
) -> None:
    """Write deterministic validation ROC and precision-recall figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_order = validation_results["model"].tolist()

    roc_figure, roc_axis = plt.subplots(figsize=(7, 5), dpi=150)
    for model_name in model_order:
        curve = roc_curve_results.loc[roc_curve_results["model"].eq(model_name)]
        roc_auc = float(curve["roc_auc"].iloc[0])
        roc_axis.plot(
            curve["false_positive_rate"],
            curve["true_positive_rate"],
            linewidth=1.5,
            label=f"{model_name} (ROC-AUC={roc_auc:.4f})",
        )
    roc_axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="0.5",
        linestyle="--",
        linewidth=1,
        label="No skill (ROC-AUC=0.5000)",
    )
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate (recall)")
    roc_axis.set_title("Validation ROC curves")
    roc_axis.set_xlim(0, 1)
    roc_axis.set_ylim(0, 1)
    roc_axis.legend(fontsize=8)
    roc_figure.tight_layout()

    pr_figure, pr_axis = plt.subplots(figsize=(7, 5), dpi=150)
    positive_rate = float(validation_results["positive_rate"].iloc[0])
    for model_name in model_order:
        curve = pr_curve_results.loc[pr_curve_results["model"].eq(model_name)]
        pr_auc = float(curve["pr_auc"].iloc[0])
        pr_axis.plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.5,
            label=f"{model_name} (AP={pr_auc:.4f})",
        )
    pr_axis.axhline(
        positive_rate,
        color="0.5",
        linestyle="--",
        linewidth=1,
        label=f"No skill (prevalence={positive_rate:.4f})",
    )
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title("Validation precision-recall curves")
    pr_axis.set_xlim(0, 1)
    pr_axis.set_ylim(0, 1)
    pr_axis.legend(fontsize=8)
    pr_figure.tight_layout()

    for directory in (FIGURES_DIR, ROOT_FIGURES_DIR):
        roc_figure.savefig(directory / "baseline_validation_roc_curve.png")
        pr_figure.savefig(directory / "baseline_validation_pr_curve.png")
    plt.close(roc_figure)
    plt.close(pr_figure)


def _format_metrics_table(results: pd.DataFrame) -> str:
    """Render a compact Markdown metric table."""
    columns = [
        "model",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "recall_at_10_percent",
    ]
    if "score_unique_count" in results.columns:
        columns.append("score_unique_count")
    display = results.loc[:, columns].copy()
    return _dataframe_to_markdown(display)


def _format_phase_1_table(results: pd.DataFrame) -> str:
    """Render the standardized Phase 1 classification comparison."""
    columns = [
        "model",
        "evaluated_split",
        "row_count",
        "positive_count",
        "negative_count",
        "positive_rate",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    return _dataframe_to_markdown(results.loc[:, columns].copy())


def _format_phase_2_table(results: pd.DataFrame) -> str:
    """Render the standardized Phase 2 validation ranking comparison."""
    columns = [
        "model",
        "evaluated_split",
        "positive_rate",
        "roc_auc",
        "pr_auc",
    ]
    return _dataframe_to_markdown(results.loc[:, columns].copy())


def _format_phase_3_table(results: pd.DataFrame) -> str:
    """Render the standardized Phase 3 validation probability summary."""
    columns = [
        "model",
        "evaluated_split",
        "positive_rate",
        "brier_score",
    ]
    return _dataframe_to_markdown(results.loc[:, columns].copy())


def _format_phase_4_1_table(results: pd.DataFrame) -> str:
    """Render the one-row Phase 4.1 Logistic Regression operating point."""
    columns = [
        "model",
        "evaluated_split",
        "threshold",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "predicted_positive_count",
        "predicted_positive_rate",
        "sample_count",
        "actual_positive_count",
        "actual_positive_rate",
    ]
    return _dataframe_to_markdown(results.loc[:, columns].copy())


def _format_phase_4_2_table(results: pd.DataFrame) -> str:
    """Render Phase 4.2 using the established threshold-result columns."""
    return _format_phase_4_1_table(results)


def _format_phase_4_5_table(results: pd.DataFrame) -> str:
    """Render Phase 4.5 policy candidates in a compact comparison table."""
    columns = [
        "policy_name",
        "constraint_name",
        "constraint_value",
        "constraint_satisfied",
        "candidate_threshold",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "predicted_positive_count",
        "predicted_positive_rate",
    ]
    return _dataframe_to_markdown(results.loc[:, columns].copy())


def _format_confusion_matrices(results: pd.DataFrame) -> str:
    """Render readable predicted-by-actual confusion matrices for each model."""
    sections: list[str] = []
    for _, row in results.iterrows():
        sections.extend(
            [
                f"### {row['model']}",
                "",
                "| Predicted \\ Actual | Miss (1) | On-time (0) |",
                "| --- | ---: | ---: |",
                (
                    f"| Miss (1) | TP = {row['true_positive']} | "
                    f"FP = {row['false_positive']} |"
                ),
                (
                    f"| On-time (0) | FN = {row['false_negative']} | "
                    f"TN = {row['true_negative']} |"
                ),
                "",
            ]
        )
    return "\n".join(sections).rstrip()


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored Markdown without tabulate."""
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rendered: list[str] = []
        for value in row.tolist():
            if pd.isna(value):
                rendered.append("—")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_reports(
    *,
    inputs: FrozenBaselineInputs,
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    subgroup_results: pd.DataFrame,
    calibration_results: pd.DataFrame,
    logistic_validation_threshold_result: pd.DataFrame,
    logistic_validation_manual_threshold_results: pd.DataFrame,
    logistic_validation_sweep_results: pd.DataFrame,
    logistic_validation_policy_results: pd.DataFrame,
    frozen_threshold_decision: FrozenThresholdDecision,
    frozen_threshold_decision_path: Path,
    threshold_tradeoff_figure_paths: tuple[Path, ...],
    selected_model_name: str,
    selected_threshold: float,
    artifact_path: Path,
) -> None:
    """Write human-readable baseline and Month 1 closure reports."""
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected_row = validation_results.loc[
        validation_results["model"].eq(selected_model_name)
    ].iloc[0]
    threshold_row = logistic_validation_threshold_result.iloc[0]
    test_row = test_results.iloc[0]
    baseline_summary = f"""# Baseline Results

## Validation Comparison

{_format_metrics_table(validation_results)}

## Phase 1 — Basic Classification Evaluation

Positive class `1` means missed resolution target; class `0` means on time.
Metrics use the baselines' existing predictions and fixed decision mechanisms.
Precision, recall, and F1 use an explicit zero-division value of `0.0`.

{_format_phase_1_table(validation_results)}

### Human-Readable Confusion Matrices

Rows are predicted outcomes and columns are actual outcomes. A false negative
is an actual missed-target complaint that the model failed to flag.

{_format_confusion_matrices(validation_results)}

## Phase 2 — Ranking Evaluation

Higher scores mean greater predicted risk of missing the resolution target.
ROC and precision-recall curves use the baselines' existing continuous
validation scores and sklearn's score-derived thresholds; they do not select or
change a classification threshold. In this project, `pr_auc` means sklearn
Average Precision (`average_precision_score`), not trapezoidal PR-curve area.
The PR no-skill reference is validation positive prevalence, not `0.5`.

{_format_phase_2_table(validation_results)}

Curve data:

- `reports/tables/baseline_validation_roc_curve.csv`
- `reports/tables/baseline_validation_pr_curve.csv`

Figures:

- `reports/figures/baseline_validation_roc_curve.png`
- `reports/figures/baseline_validation_pr_curve.png`

## Phase 3 — Probability Calibration

The Brier score is sklearn `brier_score_loss`; lower values indicate less
overall probability error, with `0.0` representing perfect probabilities. It
reflects both calibration and probabilistic discrimination/resolution, so it is
not a pure calibration-only statistic and is not a ranking metric.

Calibration curves use 10 uniform probability bins. Points show populated bins
only: x is mean predicted missed-target probability and y is observed
missed-target rate. Points near the ideal `y=x` diagonal are locally calibrated;
points below it indicate overprediction in that bin, while points above it
indicate underprediction. A local bin does not characterize the entire model.

{_format_phase_3_table(validation_results)}

Calibration data:

- `reports/tables/baseline_validation_calibration_curve.csv`

Figure:

- `reports/figures/baseline_validation_calibration_curve.png`

## Phase 4.1 — Default Threshold 0.50

Logistic Regression validation probabilities are converted using
`score >= 0.50` as positive. This is one descriptive operating point; no
threshold search, selection, or optimization occurs.

{_format_phase_4_1_table(logistic_validation_threshold_result)}

At threshold {threshold_row["threshold"]:.2f},
{int(threshold_row["predicted_positive_count"]):,} of
{int(threshold_row["sample_count"]):,} validation complaints
({threshold_row["predicted_positive_rate"] * 100:.1f}%) are flagged as at risk.
The flags capture {int(threshold_row["true_positives"]):,} of
{int(threshold_row["actual_positive_count"]):,} actual missed-target complaints
({threshold_row["recall"] * 100:.1f}% recall). Of the flagged complaints,
{int(threshold_row["true_positives"]):,} of
{int(threshold_row["predicted_positive_count"]):,} actually miss their target
({threshold_row["precision"] * 100:.1f}% precision). These values do not
establish that `0.50` is good, bad, optimal, or selected.

Phase 4.1 data: `reports/tables/logistic_regression_validation_threshold_050.csv`

## Phase 4.2 — Manual Threshold Comparison

The existing Logistic Regression validation probabilities are evaluated at
`0.30`, `0.40`, `0.50`, `0.60`, and `0.70`, in that order. The probability
vector and ranking metrics remain unchanged; only the hard-classification
operating point changes. No threshold is ranked, optimized, recommended, or
selected.

{_format_phase_4_2_table(logistic_validation_manual_threshold_results)}

### Comparisons With 0.50

{_format_phase_4_2_comparisons(logistic_validation_manual_threshold_results)}

Phase 4.2 data: `reports/tables/logistic_regression_validation_manual_thresholds.csv`

## Phase 4.3 — Deterministic Threshold Sweep

The same Logistic Regression validation probability vector is evaluated across
a deterministic `{SWEEP_CLASSIFICATION_THRESHOLDS[0]:.2f}` to
`{SWEEP_CLASSIFICATION_THRESHOLDS[-1]:.2f}` threshold grid in steps of `0.01`
(`{len(SWEEP_CLASSIFICATION_THRESHOLDS)}` thresholds). Ranking metrics are
unchanged; only the hard-classification operating point changes. No threshold
is ranked, optimized, recommended, or selected.

### 0.40-0.50 Transition Region

{_format_phase_4_3_focus_table(logistic_validation_sweep_results)}

{_format_phase_4_3_transition_summary(logistic_validation_sweep_results)}

Full sweep data: `reports/tables/logistic_regression_validation_threshold_sweep.csv`
Phase 4.3 report: `reports/phase_4_3_threshold_sweep.md`

## Phase 4.4 — Threshold Trade-Off Visualizations

The Phase 4.3 sweep rows are visualized directly. No model fitting, probability
generation, threshold recomputation, threshold ranking, threshold optimization,
or test-set threshold evaluation occurs in this visualization layer.

Figures:

- `reports/figures/{threshold_tradeoff_figure_paths[0].name}`
- `reports/figures/{threshold_tradeoff_figure_paths[1].name}`
- `reports/figures/{threshold_tradeoff_figure_paths[2].name}`

{_format_phase_4_4_sensitivity_summary(logistic_validation_sweep_results)}

Phase 4.4 report: `reports/phase_4_4_threshold_tradeoffs.md`

## Phase 4.5 — Threshold-Selection Policy Candidates

The existing Phase 4.3 validation sweep is evaluated under four explicit
candidate policies. Each successful candidate comes from the existing sweep
grid; no new thresholds, interpolation, model fitting, probability generation,
test scoring, final policy choice, or threshold freeze occurs here.

{_format_phase_4_5_table(logistic_validation_policy_results)}

### Policy Interpretation

{_format_phase_4_5_policy_interpretation(logistic_validation_policy_results)}

Phase 4.5 report: `reports/phase_4_5_threshold_policy_candidates.md`
Phase 4.5 data: `reports/tables/logistic_regression_validation_threshold_policy_candidates.csv`

## Phase 4.6 — Frozen Threshold Decision

The approved operating policy is workload-limited: constrain
`predicted_positive_rate <= {frozen_threshold_decision.constraint_value:.2f}`,
then maximize recall on validation. Phase 4.6 reuses the Phase 4.5 workload
policy result, verifies that the candidate satisfies the workload constraint,
and freezes threshold `{frozen_threshold_decision.selected_threshold:.2f}`.

Validation evidence at the frozen threshold: precision
{frozen_threshold_decision.precision:.4f}, recall
{frozen_threshold_decision.recall:.4f}, F1
{frozen_threshold_decision.f1:.4f}, TP
{frozen_threshold_decision.true_positives:,}, FP
{frozen_threshold_decision.false_positives:,}, TN
{frozen_threshold_decision.true_negatives:,}, FN
{frozen_threshold_decision.false_negatives:,}, flagged rate
{frozen_threshold_decision.predicted_positive_rate:.4f}. This threshold is not
globally optimal; it is the validation-selected threshold under the approved
policy.

Frozen threshold artifact: `{frozen_threshold_decision_path.relative_to(PROJECT_ROOT)}`
Phase 4.6 report: `reports/phase_4_6_frozen_threshold_decision.md`

## Baseline Selection

Selected baseline: `{selected_model_name}`

Selection rule: {SELECTION_RULE}

Selected validation Recall@10%: {selected_row["recall_at_10_percent"]:.4f}
Selected validation PR-AUC: {selected_row["pr_auc"]:.4f}
Selected threshold: {selected_threshold:.4f}

## FINAL TEST RESULTS

{_format_metrics_table(test_results)}

{_format_phase_1_table(test_results)}

### Human-Readable Confusion Matrix

{_format_confusion_matrices(test_results)}

Selected artifact: `{artifact_path.relative_to(PROJECT_ROOT)}`
"""
    (BASELINE_REPORT_DIR / "baseline_results.md").write_text(
        baseline_summary,
        encoding="utf-8",
    )
    (PROJECT_ROOT / "reports/baseline_results.md").write_text(
        baseline_summary,
        encoding="utf-8",
    )

    scope_rows = sum(inputs.phase_9_contract.row_counts_by_split.values())
    target_prevalence = (
        inputs.phase_9_contract.train_positive_class_prevalence
    )
    subgroup_summary = _subgroup_summary(subgroup_results)
    calibration_summary = _calibration_summary(calibration_results)
    validation_pr_auc_drop = selected_row["pr_auc"] - test_row["pr_auc"]
    validation_roc_auc_drop = selected_row["roc_auc"] - test_row["roc_auc"]
    validation_recall_drop = selected_row["recall"] - test_row["recall"]
    month_1 = f"""# Month 1 Baseline Report

## 1. Executive Summary

MONTH 1 COMPLETE. The selected Month 1 baseline is `{selected_model_name}` with
the frozen threshold {selected_threshold:.4f}. The untouched final test
ROC-AUC is {test_row["roc_auc"]:.4f}, PR-AUC is {test_row["pr_auc"]:.4f},
Recall@10% is {test_row["recall_at_10_percent"]:.4f}, and Brier score is
{test_row["brier_score"]:.4f}. These results establish a reproducible baseline,
not a production-quality model.

## 2. Business Problem

The project predicts whether a newly created NYC 311 complaint will miss its
expected resolution target. The target is `missed_resolution_target`, prediction
occurs immediately after complaint creation, and operational use is prioritizing
or monitoring complaints by risk. The authoritative business framing is
`docs/business/business_problem.md`.

## 3. Selected Scope

- Agency: DSNY
- Complaint type/scope: Graffiti
- Date range: 2024-01-01 to 2025-12-31
- Eligible record count: {scope_rows}
- Training target prevalence: {target_prevalence:.4f}
- Split ID: {inputs.split_id}

The authoritative scope decision is `docs/scope_decision.md`.

## 4. Target Definition

Target: `missed_resolution_target`; positive class `1` means the expected
resolution target was missed. The target contract and leakage boundary are
defined in `docs/target_definition.md`.

## 5. Dataset / Eligibility Summary

Eligible record count: {scope_rows}. Training rows:
{inputs.phase_9_contract.row_counts_by_split["train"]}; validation rows:
{inputs.phase_9_contract.row_counts_by_split["validation"]}; test rows:
{inputs.phase_9_contract.row_counts_by_split["test"]}. Training positive rate:
{target_prevalence:.4f}.

## 6. Data Quality Findings

Inputs reuse the governed Step 7 cleaning output and Step 8 chronological split.
Known limitations from earlier evidence remain in force: temporal target
prevalence is not stable across the selected period, source data is a snapshot,
and only a small creation-time temporal feature set is approved for Month 1.

## 7. Leakage Policy

Models fit only on training data. Validation is used for comparison and
threshold selection. Test is evaluated only after the baseline selection is
frozen.

## 8. Feature Policy

Approved model features: {", ".join(inputs.feature_names)}.

Conditional geography fields such as `borough`, `location_type`,
`incident_zip`, `latitude`, and `longitude` are not in the frozen Month 1 model
matrix.

## 9. Chronological Split

Train, validation, and test splits retain chronological order and disjoint
complaint identifiers. The selected split policy is documented in
`docs/time_split_policy.md`.

## 10. Frozen Preprocessing

The modelling workflow consumes verified CSR float64 matrices from the existing
Phase 2-8 preprocessing path. No preprocessing is refit inside baseline models.
The frozen feature order is: {", ".join(inputs.feature_names)}.

## 11. Baseline Models

- Majority Class: train majority-class classifier with train prevalence score.
- Historical Rate: train-only `created_month` target rate with global fallback.
- Rule Based: historical-risk score threshold selected on validation F1.
- Logistic Regression: sparse logistic regression over frozen matrix columns.

## 12. Validation Results

{_format_metrics_table(validation_results)}

## 13. Baseline Selection Decision

Selected baseline: `{selected_model_name}`.

Rule: {SELECTION_RULE}

Majority has a constant score and therefore no meaningful ranking signal.
Historical Rate has weak discrimination and ranking. Rule Based achieves high
validation recall by predicting every validation row positive, but it does not
improve discrimination or ranking. Logistic Regression is the strongest Month 1
baseline because it has the best non-constant validation PR-AUC
({selected_row["pr_auc"]:.4f}), ROC-AUC ({selected_row["roc_auc"]:.4f}), Brier
score ({selected_row["brier_score"]:.4f}), and meaningful validation Recall@10%
({selected_row["recall_at_10_percent"]:.4f}) among the non-constant models. This
does not mean it is a strong predictive model.

## 14. Final Test Results

FINAL TEST RESULTS:

{_format_metrics_table(test_results)}

Test ROC-AUC is {test_row["roc_auc"]:.4f}. The model has weak discrimination on
the final chronological test period and performs only slightly above random
ranking by ROC-AUC.

## 15. Threshold Interpretation

At the frozen threshold {selected_threshold:.4f}, final test precision is
{test_row["precision"]:.4f}, recall is {test_row["recall"]:.4f}, and F1 is
{test_row["f1"]:.4f}. This means the classifier identifies only
{test_row["recall"] * 100:.1f}% of actual missed-target complaints. The frozen
0.5 threshold is not operationally useful as a high-recall binary alert
threshold. It remains frozen because test data is final evaluation only and must
not be used for threshold retuning.

## 16. Operational Top-K Evaluation

- Precision@5%: {test_row["precision_at_5_percent"]:.4f}
- Recall@5%: {test_row["recall_at_5_percent"]:.4f}
- Precision@10%: {test_row["precision_at_10_percent"]:.4f}
- Recall@10%: {test_row["recall_at_10_percent"]:.4f}
- Precision@20%: {test_row["precision_at_20_percent"]:.4f}
- Recall@20%: {test_row["recall_at_20_percent"]:.4f}

If operations reviews the highest-risk 20% of complaints, the baseline captures
approximately {test_row["recall_at_20_percent"] * 100:.1f}% of all eventual
missed-target complaints. Precision@20% is {test_row["precision_at_20_percent"]:.4f},
meaning roughly {test_row["precision_at_20_percent"] * 100:.1f}% of complaints
selected in the top-risk 20% actually missed their resolution target. The model
is more useful as a weak prioritization/ranking baseline than as a binary
classifier at threshold 0.5.

## 17. Calibration Evaluation

Validation Brier score is {selected_row["brier_score"]:.4f}; test Brier score is
{test_row["brier_score"]:.4f}. Brier score alone does not establish that the
model is well calibrated. The deterministic calibration table compares fixed
predicted-risk bins with observed missed-target frequency and is saved at
`reports/tables/logistic_regression_calibration.csv`; the companion plot is
`reports/figures/logistic_regression_calibration.png`.

{calibration_summary}

No Platt scaling, isotonic regression, or other calibration model is fitted in
Month 1.

## 18. Validation-to-Test Generalization

- Validation PR-AUC: {selected_row["pr_auc"]:.4f}; test PR-AUC:
  {test_row["pr_auc"]:.4f}; change: -{validation_pr_auc_drop:.4f}
- Validation ROC-AUC: {selected_row["roc_auc"]:.4f}; test ROC-AUC:
  {test_row["roc_auc"]:.4f}; change: -{validation_roc_auc_drop:.4f}
- Validation recall: {selected_row["recall"]:.4f}; test recall:
  {test_row["recall"]:.4f}; change: -{validation_recall_drop:.4f}

This is meaningful performance degradation on the later chronological holdout.
The report treats it as temporal generalization weakness. It does not claim a
new drift diagnosis beyond the split-policy evidence already documented in
`docs/time_split_policy.md`.

## 19. Subgroup / Error Analysis

{subgroup_summary}

Several sufficiently populated temporal subgroups show zero recall at the
frozen 0.5 threshold. This is descriptive error analysis only and does not imply
that those temporal groups cause missed targets.

## 20. Limitations

The baseline does not use unresolved conditional geography fields, NLP, gradient
boosting, SHAP, forecasting, resolution-time regression, text classification, or
calibration fitting. Absolute predictive performance is weak, especially for
binary alerts at threshold 0.5.

## 21. Month 1 Conclusion

The Month 1 Logistic Regression baseline provides measurable but limited
predictive signal from the frozen temporal feature set. Its final chronological
test performance is weak: ROC-AUC = {test_row["roc_auc"]:.4f} and recall at
threshold 0.5 = {test_row["recall"]:.4f}. Risk ranking is more useful than hard
classification: the highest-risk 20% captures {test_row["recall_at_20_percent"] * 100:.1f}%
of actual missed-target complaints. These results establish a valid reproducible
baseline but do not justify production deployment.

## 22. Month 2 Readiness

MONTH 2 READY, conditional on preserving the frozen Month 1 evaluation contract.

- Target frozen: yes
- Scope frozen: yes
- Chronological split frozen: yes
- Preprocessing frozen: yes
- Baseline evaluation complete: yes
- Final test result recorded: yes
- Month 1 reports reproducible from workflow outputs: yes
- Full test suite passes in the verified local run: yes
- Unresolved blockers: none
"""
    (PROJECT_ROOT / "reports/month_1_baseline_report.md").write_text(
        month_1,
        encoding="utf-8",
    )


def _subgroup_summary(subgroup_results: pd.DataFrame) -> str:
    """Return a concise descriptive subgroup summary."""
    if subgroup_results.empty:
        return "No subgroup met the minimum count threshold."
    worst_recall = subgroup_results.sort_values(
        ["recall", "row_count"],
        ascending=[True, False],
        kind="mergesort",
    ).head(5)
    lines = [
        "Lowest observed recall groups meeting the minimum count threshold:",
        "",
        worst_recall.loc[
            :,
            [
                "split",
                "group_column",
                "group_value",
                "row_count",
                "actual_positive_rate",
                "mean_predicted_risk",
                "precision",
                "recall",
                "false_positives",
                "false_negatives",
            ],
        ].pipe(_dataframe_to_markdown),
    ]
    return "\n".join(lines)


def _calibration_summary(calibration_results: pd.DataFrame) -> str:
    """Return a compact calibration evidence table for report text."""
    table = calibration_results.loc[
        calibration_results["split"].eq("test")
        & calibration_results["row_count"].gt(0),
        [
            "split",
            "bin_index",
            "lower_bound",
            "upper_bound",
            "row_count",
            "mean_predicted_risk",
            "observed_positive_rate",
            "positive_count",
        ],
    ].copy()
    if table.empty:
        return "No non-empty calibration bins were available for the test split."
    return (
        "Final test calibration bins with observations:\n\n"
        + _dataframe_to_markdown(table)
    )


def _persist_selected_model(
    *,
    selected_model_name: str,
    model_objects: dict[str, object],
    selected_threshold: float,
    inputs: FrozenBaselineInputs,
) -> Path:
    """Persist the selected frozen baseline artifact and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / "selected_month_1_baseline.joblib"
    payload = {
        "selected_model_name": selected_model_name,
        "model": model_objects[selected_model_name],
        "selected_threshold": selected_threshold,
        "feature_names": inputs.feature_names,
        "phase_9_contract": inputs.phase_9_contract.to_dict(),
        "selection_rule": SELECTION_RULE,
    }
    joblib.dump(payload, artifact_path)
    metadata = {
        "selected_model_name": selected_model_name,
        "selected_threshold": selected_threshold,
        "feature_names": list(inputs.feature_names),
        "phase_9_contract_fingerprint": inputs.phase_9_contract.fingerprint,
        "selection_rule": SELECTION_RULE,
    }
    (MODEL_DIR / "selected_month_1_baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def run_baseline_workflow(
    *,
    split_run_path: Path | None = None,
) -> BaselineWorkflowResult:
    """Run validation comparison, freeze selection, and evaluate test once."""
    inputs = load_frozen_baseline_inputs(split_run_path=split_run_path)
    (
        validation_results,
        model_objects,
        validation_subgroups,
        ranking_evaluations,
        calibration_evaluations,
        logistic_threshold_metrics,
        logistic_manual_threshold_metrics,
        logistic_sweep_metrics,
    ) = _fit_and_evaluate_validation(inputs)
    roc_curve_results, pr_curve_results = build_ranking_curve_tables(
        ranking_evaluations,
        split_name="validation",
    )
    validation_calibration_results = build_validation_calibration_table(
        calibration_evaluations
    )
    logistic_validation_threshold_result = (
        build_logistic_validation_threshold_table(logistic_threshold_metrics)
    )
    logistic_validation_manual_threshold_results = (
        build_logistic_validation_manual_threshold_table(
            logistic_manual_threshold_metrics
        )
    )
    logistic_validation_sweep_results = build_logistic_validation_sweep_table(
        logistic_sweep_metrics
    )
    logistic_validation_policy_results = build_logistic_validation_policy_table(
        evaluate_threshold_selection_policies(logistic_validation_sweep_results)
    )
    frozen_threshold_decision = freeze_workload_limited_threshold(
        logistic_validation_policy_results
    )
    selected_model_name = select_baseline(validation_results)
    _, validation_score, selected_threshold = _split_predictions(
        selected_model_name,
        model_objects,
        inputs,
        split="validation",
    )
    test_pred, test_score, selected_threshold = _split_predictions(
        selected_model_name,
        model_objects,
        inputs,
        split="test",
    )
    test_metrics = _evaluate(
        selected_model_name,
        inputs.targets["test"],
        test_pred,
        test_score,
    )
    test_results = pd.DataFrame(
        [metrics_row(selected_model_name, test_metrics, evaluated_split="test")]
    )
    test_subgroups = build_subgroup_analysis(
        frame=inputs.frames["test"],
        y_true=inputs.targets["test"],
        y_pred=test_pred,
        y_score=test_score,
        split_name="test",
        model_name=selected_model_name,
    )
    subgroup_results = pd.concat(
        [validation_subgroups, test_subgroups],
        ignore_index=True,
    )
    calibration_results = build_selected_calibration(
        inputs=inputs,
        selected_model_name=selected_model_name,
        validation_score=validation_score,
        test_score=test_score,
    )
    artifact_path = _persist_selected_model(
        selected_model_name=selected_model_name,
        model_objects=model_objects,
        selected_threshold=selected_threshold,
        inputs=inputs,
    )
    _write_tables(
        validation_results=validation_results,
        test_results=test_results,
        subgroup_results=subgroup_results,
        calibration_results=calibration_results,
        validation_calibration_results=validation_calibration_results,
        roc_curve_results=roc_curve_results,
        pr_curve_results=pr_curve_results,
    )
    _write_calibration_figure(calibration_results)
    _write_validation_calibration_figure(validation_calibration_results)
    _write_phase_4_1_outputs(logistic_validation_threshold_result)
    _write_phase_4_2_outputs(logistic_validation_manual_threshold_results)
    _write_phase_4_3_outputs(logistic_validation_sweep_results)
    threshold_tradeoff_figure_paths = _write_phase_4_4_outputs(
        logistic_validation_sweep_results
    )
    _write_phase_4_5_outputs(logistic_validation_policy_results)
    frozen_threshold_decision_path = _write_phase_4_6_outputs(
        frozen_threshold_decision
    )
    _write_ranking_figures(
        validation_results=validation_results,
        roc_curve_results=roc_curve_results,
        pr_curve_results=pr_curve_results,
    )
    _write_reports(
        inputs=inputs,
        validation_results=validation_results,
        test_results=test_results,
        subgroup_results=subgroup_results,
        calibration_results=calibration_results,
        logistic_validation_threshold_result=logistic_validation_threshold_result,
        logistic_validation_manual_threshold_results=(
            logistic_validation_manual_threshold_results
        ),
        logistic_validation_sweep_results=logistic_validation_sweep_results,
        logistic_validation_policy_results=logistic_validation_policy_results,
        frozen_threshold_decision=frozen_threshold_decision,
        frozen_threshold_decision_path=frozen_threshold_decision_path,
        threshold_tradeoff_figure_paths=threshold_tradeoff_figure_paths,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        artifact_path=artifact_path,
    )
    return BaselineWorkflowResult(
        validation_results=validation_results,
        test_results=test_results,
        subgroup_results=subgroup_results,
        calibration_results=calibration_results,
        validation_calibration_results=validation_calibration_results,
        logistic_validation_threshold_result=logistic_validation_threshold_result,
        logistic_validation_manual_threshold_results=(
            logistic_validation_manual_threshold_results
        ),
        logistic_validation_sweep_results=logistic_validation_sweep_results,
        logistic_validation_policy_results=logistic_validation_policy_results,
        frozen_threshold_decision=frozen_threshold_decision,
        frozen_threshold_decision_path=frozen_threshold_decision_path,
        threshold_tradeoff_figure_paths=threshold_tradeoff_figure_paths,
        roc_curve_results=roc_curve_results,
        pr_curve_results=pr_curve_results,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        selected_artifact_path=artifact_path,
    )


def _parser() -> argparse.ArgumentParser:
    """Build the baseline workflow command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-run", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the baseline workflow from the command line."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    result = run_baseline_workflow(split_run_path=args.split_run)
    LOGGER.info(
        "Baseline workflow complete selected=%s threshold=%.4f",
        result.selected_model_name,
        result.selected_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
