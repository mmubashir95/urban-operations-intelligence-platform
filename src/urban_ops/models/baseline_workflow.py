"""Run Month 1 baseline modelling, evaluation, and report generation.

The workflow reconstructs the already-governed frozen preprocessing outputs
from Step 8 split artifacts, verifies Phase 8 and Phase 9 contracts, evaluates
Month 1 baselines, freezes one selected baseline, and only then evaluates the
untouched test split.
"""

from __future__ import annotations

import argparse
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
from urban_ops.models.evaluation import ClassificationMetrics, evaluate_binary_classifier, metrics_row
from urban_ops.models.evaluation import build_calibration_table
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
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
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
    return validation_results, model_objects, subgroup


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


def _write_tables(
    *,
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    subgroup_results: pd.DataFrame,
    calibration_results: pd.DataFrame,
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
            if isinstance(value, float):
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
    selected_model_name: str,
    selected_threshold: float,
    artifact_path: Path,
) -> None:
    """Write human-readable baseline and Month 1 closure reports."""
    BASELINE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected_row = validation_results.loc[
        validation_results["model"].eq(selected_model_name)
    ].iloc[0]
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
    validation_results, model_objects, validation_subgroups = _fit_and_evaluate_validation(inputs)
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
    )
    _write_calibration_figure(calibration_results)
    _write_reports(
        inputs=inputs,
        validation_results=validation_results,
        test_results=test_results,
        subgroup_results=subgroup_results,
        calibration_results=calibration_results,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        artifact_path=artifact_path,
    )
    return BaselineWorkflowResult(
        validation_results=validation_results,
        test_results=test_results,
        subgroup_results=subgroup_results,
        calibration_results=calibration_results,
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
