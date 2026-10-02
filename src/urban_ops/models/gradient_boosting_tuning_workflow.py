"""Run controlled Phase 3 selection, persist its freeze, then evaluate TEST once.

All candidates use the verified Phase 2 matrices and the existing XGBoost
wrapper. Final evaluation reuses the selected TRAIN-fitted model without refit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH, FrozenBaselineInputs
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel, load_gradient_boosting_config
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.gradient_boosting_reporting import (
    FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH, MONTH_2_REPORT_DIR,
    load_frozen_logistic_regression_evidence_snapshot,
)
from urban_ops.models.gradient_boosting_selection import (
    CandidateEvaluation, FrozenGradientBoostingDecision, METRIC_NAMES, SELECTION_POLICY,
    evaluate_scores, freeze_decision, select_candidate, verify_persisted_decision,
)
from urban_ops.models.gradient_boosting_tuning import (
    TUNING_CONFIG_PATH, GradientBoostingTuningError, load_tuning_contract, train_candidate,
)
from urban_ops.utils.paths import PROJECT_ROOT

LR_TEST_PATH = PROJECT_ROOT / "reports/12_baseline_modelling/tables/baseline_test_results.csv"
LR_METADATA_PATH = PROJECT_ROOT / "models/baselines/selected_month_1_baseline_metadata.json"
DISPLAY_METRICS = ("PR-AUC", "ROC-AUC", "Brier Score", "Recall@5%", "Recall@10%", "Recall@20%", "Precision@5%", "Precision@10%", "Precision@20%")


@dataclass(frozen=True)
class GradientBoostingSelectionResult:
    """Selection outputs; only the chosen model survives to final evaluation."""

    decision: FrozenGradientBoostingDecision
    model: GradientBoostedRiskModel
    evaluations: tuple[CandidateEvaluation, ...]
    validation_scores: tuple[tuple[str, np.ndarray], ...]
    decision_path: Path
    output_directory: Path


def _read_phase2_metrics(directory: Path, *, inputs: FrozenBaselineInputs) -> dict[str, float]:
    """Load frozen Phase 2 validation evidence for comparison and baseline audit."""
    validation = pd.read_csv(directory / "phase_2_gradient_boosting_validation.csv", float_precision="round_trip")
    capacity = pd.read_csv(directory / "phase_2_gradient_boosting_capacity.csv", float_precision="round_trip")
    if len(validation) != 1 or validation.iloc[0]["evaluated_split"] != "validation":
        raise GradientBoostingTuningError("Phase 2 evidence must contain one validation result.")
    row = validation.iloc[0]
    if int(row["row_count"]) != len(inputs.targets["validation"]) or int(row["positive_count"]) != int(inputs.targets["validation"].sum()):
        raise GradientBoostingTuningError("Phase 2 evidence population differs from frozen validation.")
    values = {name: float(row[name]) for name in METRIC_NAMES[:3]}
    if capacity["capacity"].tolist() != [0.05, 0.1, 0.2] or not capacity["evaluated_split"].eq("validation").all():
        raise GradientBoostingTuningError("Phase 2 capacity evidence differs from frozen policy.")
    for record in capacity.itertuples():
        suffix = f"{round(record.capacity * 100):02d}"
        values[f"recall_at_{suffix}"] = float(record.recall_at_k)
        values[f"precision_at_{suffix}"] = float(record.precision_at_k)
    return values


def _comparison_text(selected: dict[str, float], reference: dict[str, float], label: str) -> str:
    """Describe every improvement and regression with Brier's direction respected."""
    improved, regressed = [], []
    for name in METRIC_NAMES:
        difference = selected[name] - reference[name]
        if difference == 0:
            continue
        destination = improved if (difference < 0 if name == "brier_score" else difference > 0) else regressed
        destination.append(name)
    return (f"Against {label}: PR-AUC difference = {selected['pr_auc'] - reference['pr_auc']:+.8f}. "
            f"Improved: {', '.join(improved) or 'none'}. Regressed: {', '.join(regressed) or 'none'}.\n")


def run_candidate_selection(*, inputs: FrozenBaselineInputs, tuning_config_path: Path | str = TUNING_CONFIG_PATH,
                            output_directory: Path | str = MONTH_2_REPORT_DIR,
                            frozen_evidence_path: Path | str = FROZEN_LOGISTIC_REGRESSION_EVIDENCE_PATH,
                            phase2_directory: Path | str = MONTH_2_REPORT_DIR) -> GradientBoostingSelectionResult:
    """Fit TRAIN, score VALIDATION, compare frozen evidence, and write one freeze.

    This function never retrieves TEST matrices or targets. Existing freeze
    artifacts prevent re-selection, including after final TEST evaluation.
    """
    directory = Path(output_directory)
    decision_path = directory / "phase_3_selected_configuration.json"
    if decision_path.exists() or (directory / "phase_3_test_access.json").exists():
        raise GradientBoostingTuningError("Phase 3 is frozen; candidate selection cannot be rerun.")
    contract = load_tuning_contract(tuning_config_path)
    contract.verify_inputs(inputs)
    lr = load_frozen_logistic_regression_evidence_snapshot(
        frozen_evidence_path, expected_split_id=inputs.split_id,
        expected_phase_9_contract_fingerprint=inputs.phase_9_contract.fingerprint,
    )
    if lr.row_count != len(inputs.targets["validation"]) or lr.positive_count != int(inputs.targets["validation"].sum()):
        raise GradientBoostingTuningError("LR validation population differs from frozen inputs.")
    phase2 = _read_phase2_metrics(Path(phase2_directory), inputs=inputs)
    evaluations, scores, models = [], [], {}
    for candidate in contract.candidates:
        model = train_candidate(candidate, inputs=inputs)
        vector = model.predict_score(inputs.matrices["validation"])
        vector.setflags(write=False)
        evaluations.append(CandidateEvaluation(candidate, evaluate_scores(inputs.targets["validation"], vector)))
        scores.append((candidate.candidate_id, vector))
        models[candidate.candidate_id] = model
    reference = next(row for row in evaluations if row.candidate.candidate_id == "phase2_baseline")
    if not all(np.isclose(dict(reference.metrics)[name], phase2[name], rtol=0, atol=1e-12) for name in METRIC_NAMES):
        raise GradientBoostingTuningError("Phase 2 reference did not reproduce frozen metrics; stop before selection.")
    selected = select_candidate(tuple(evaluations), tolerance=contract.pr_auc_tolerance)
    lr_values = {name: lr.metric_values()[display] for name, display in zip(METRIC_NAMES, DISPLAY_METRICS)}
    directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row.record() for row in evaluations])
    table.to_csv(directory / "phase_3_candidate_comparison.csv", index=False)
    comparison = pd.DataFrame([
        {"model": "Frozen LR", **lr_values}, {"model": "Frozen Phase 2 XGB", **phase2},
        *[{"model": row.candidate.candidate_id, **dict(row.metrics)} for row in evaluations],
        {"model": "Selected XGB", **dict(selected.metrics)},
    ])
    comparison.to_csv(directory / "phase_3_model_comparison.csv", index=False)
    decision = freeze_decision(selected, contract=contract, feature_names=inputs.feature_names, path=decision_path)
    best_recall = max(evaluations, key=lambda row: (dict(row.metrics)["recall_at_10"], row.candidate.candidate_id))
    selected_values = dict(selected.metrics)
    report = (
        "# Phase 3 — controlled model selection\n\n"
        "Tuning was needed because Phase 2 XGB did not outperform frozen LR.\n\n"
        f"Phase 2 configuration: `{json.dumps(load_gradient_boosting_config().model_parameters, sort_keys=True)}`.\n\n"
        "Only eight XGBoost hyperparameters may change. Target, feature order, preprocessing, chronology, "
        "split dates, leakage rules, seed, metric implementations, LR and its threshold remain frozen.\n"
        f"Features: {', '.join(inputs.feature_names)}. Split: `{inputs.split_id}`. "
        f"Fingerprint: `{inputs.phase_9_contract.fingerprint}`.\n\n"
        "Every candidate fitted on TRAIN only and was scored and evaluated on VALIDATION only. "
        "The verified input loader structurally checks TEST; no TEST probabilities or selection metrics were generated.\n\n"
        f"Selection policy: {SELECTION_POLICY} PR-AUC tolerance = {contract.pr_auc_tolerance}.\n\n"
        "Simplicity means lower depth, then fewer trees; final ties use lexical candidate ID. "
        "Sampling and regularization defaults stay fixed except where explicitly listed.\n\n"
        + "\n".join(f"- {candidate.candidate_id}: {candidate.purpose} Parameters: `{json.dumps(candidate.parameters(), sort_keys=True)}`." for candidate in contract.candidates)
        + f"\n\nHighest PR-AUC: {max(evaluations, key=lambda row: dict(row.metrics)['pr_auc']).candidate.candidate_id}. "
        f"Highest Recall@10%: {best_recall.candidate.candidate_id}. Selected: **{selected.candidate.candidate_id}**.\n\n"
        + _comparison_text(selected_values, phase2, "frozen Phase 2 XGB") + "\n"
        + _comparison_text(selected_values, lr_values, "frozen LR")
        + f"\nValidation PR-AUC favors {'XGBoost' if selected_values['pr_auc'] > lr_values['pr_auc'] else 'Logistic Regression'}. "
        "Selecting the best XGB candidate does not replace or force promotion over LR.\n\n"
        "Selection is persisted and FROZEN before TEST. No candidate additions, parameter changes, "
        "threshold search, calibration fitting, or validation-driven retuning are allowed. "
        "Final evaluation preserves Month 1's TRAIN-only fit convention and reuses this fitted model.\n"
    )
    (directory / "phase_3_selection_decision.md").write_text(report, encoding="utf-8")
    (directory / "phase_3_model_selection.md").write_text(report + "\nFinal TEST evaluation pending.\n", encoding="utf-8")
    return GradientBoostingSelectionResult(decision, models[selected.candidate.candidate_id], tuple(evaluations), tuple(scores), decision_path, directory)


def _load_lr_test_evidence(*, inputs: FrozenBaselineInputs, path: Path, metadata_path: Path) -> dict[str, float]:
    """Read the existing frozen TRAIN-fitted LR final results without retraining."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata["selected_model_name"] != "Logistic Regression"
        or tuple(metadata["feature_names"]) != inputs.feature_names
        or metadata["phase_9_contract_fingerprint"] != inputs.phase_9_contract.fingerprint):
        raise GradientBoostingTuningError("LR test artifact metadata differs from frozen inputs.")
    table = pd.read_csv(path, float_precision="round_trip")
    if len(table) != 1 or table.iloc[0]["model"] != "Logistic Regression" or table.iloc[0]["evaluated_split"] != "test":
        raise GradientBoostingTuningError("Expected one frozen LR TEST result.")
    row = table.iloc[0]
    if int(row["row_count"]) != len(inputs.targets["test"]) or int(row["positive_count"]) != int(inputs.targets["test"].sum()):
        raise GradientBoostingTuningError("LR TEST population differs from frozen inputs.")
    values = {}
    for name in METRIC_NAMES:
        column = name
        if "_at_" in name:
            prefix, suffix = name.split("_at_")
            column = f"{prefix}_at_{int(suffix)}_percent"
        values[name] = float(row[column])
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise GradientBoostingTuningError("LR TEST evidence has invalid metrics.")
    return values


def run_final_test_evaluation(selection: GradientBoostingSelectionResult, *, inputs: FrozenBaselineInputs,
                              lr_test_path: Path | str = LR_TEST_PATH,
                              lr_metadata_path: Path | str = LR_METADATA_PATH) -> pd.DataFrame:
    """Score the selected TRAIN-fitted model once, strictly after persisted freeze."""
    decision = selection.decision
    verify_persisted_decision(decision, selection.decision_path)
    if (decision.split_id != inputs.split_id or decision.feature_fingerprint != inputs.phase_9_contract.fingerprint
        or decision.feature_names != inputs.feature_names
        or selection.model.feature_names_ != inputs.feature_names
        or selection.model.config != decision.candidate.model_config(load_gradient_boosting_config())):
        raise GradientBoostingTuningError("Selected model or inputs changed after freeze.")
    directory = selection.output_directory
    # Exclusive access marker is written BEFORE retrieving TEST. A failed final
    # evaluation also cannot silently repeat the protected scoring operation.
    with (directory / "phase_3_test_access.json").open("x", encoding="utf-8") as stream:
        json.dump({"candidate_id": decision.candidate.candidate_id, "selection_timestamp": decision.selection_timestamp,
                   "split_id": decision.split_id, "status": "test_access_started"}, stream, indent=2)
    lr = _load_lr_test_evidence(inputs=inputs, path=Path(lr_test_path), metadata_path=Path(lr_metadata_path))
    scores = selection.model.predict_score(inputs.matrices["test"])
    metrics = dict(evaluate_scores(inputs.targets["test"], scores))
    pd.DataFrame([{"candidate_id": decision.candidate.candidate_id, "evaluated_split": "test", **metrics}]).to_csv(directory / "phase_3_final_test_results.csv", index=False)
    comparison = pd.DataFrame([{"metric": name, "frozen_lr": lr[name], "selected_xgb": metrics[name],
                                "difference_xgb_minus_lr": metrics[name] - lr[name]} for name in METRIC_NAMES])
    comparison.to_csv(directory / "phase_3_final_test_comparison.csv", index=False)
    with (directory / "phase_3_frozen_lr_test_evidence.json").open("x", encoding="utf-8") as stream:
        json.dump({"split_id": decision.split_id, "feature_fingerprint": decision.feature_fingerprint,
                   "source": str(lr_test_path), "metrics": lr, "frozen": True}, stream, indent=2)
    report_path = directory / "phase_3_model_selection.md"
    report = (directory / "phase_3_selection_decision.md").read_text(encoding="utf-8")
    report += "\n## Final TEST evidence\n"
    # CSV is the full-precision authoritative comparison; Markdown stays dependency-free.
    report += "\n| Metric | Frozen LR | Selected XGB | XGB minus LR |\n|---|---:|---:|---:|\n"
    for row in comparison.itertuples():
        report += f"| {row.metric} | {row.frozen_lr:.8f} | {row.selected_xgb:.8f} | {row.difference_xgb_minus_lr:+.8f} |\n"
    report += "\n" + _comparison_text(metrics, lr, "frozen LR on TEST")
    report += (f"\nFinal TEST PR-AUC favors {'XGBoost' if metrics['pr_auc'] > lr['pr_auc'] else 'Logistic Regression'}. "
               "This is generalization evidence only. No changes or retuning occurred after TEST. "
               "Phase 3 selection remains formally frozen; LR benchmark and threshold are unchanged.\n")
    report_path.write_text(report, encoding="utf-8")
    (directory / "phase_3_completion_gate.md").write_text(
        "# Phase 3 completion gate\n\n"
        "Selection and final evaluation are FROZEN. Repository verification must be recorded separately.\n\n"
        "- [x] Frozen features/order, split, preprocessing and leakage policy reused.\n"
        "- [x] Six validated deliberate candidates; Phase 2 reference reproduced.\n"
        "- [x] Every candidate fitted on TRAIN and evaluated on VALIDATION only.\n"
        "- [x] TEST untouched during selection; persisted freeze preceded TEST access.\n"
        "- [x] Shared PR-AUC, ROC-AUC, Brier, Recall and Precision at 5/10/20% evaluated.\n"
        "- [x] Candidate, frozen Phase 2, and frozen LR comparisons generated.\n"
        "- [x] Predefined policy selected one configuration and persisted its immutable freeze.\n"
        "- [x] Selected TRAIN-fitted model evaluated on TEST once; final LR comparison generated.\n"
        "- [x] No parameter, candidate, threshold, or feature changes after TEST.\n"
        "- [x] Selection and final modelling reports produced.\n"
        "- [ ] Unit and integration test evidence recorded.\n"
        "- [ ] Full repository suite and static checks recorded.\n",
        encoding="utf-8",
    )
    return comparison


def run_gradient_boosting_tuning_workflow(*, output_directory: Path | str = MONTH_2_REPORT_DIR,
                                         tuning_config_path: Path | str = TUNING_CONFIG_PATH,
                                         eda_config_path: Path | str = EDA_CONFIG_PATH,
                                         split_run_path: Path | None = None) -> GradientBoostingSelectionResult:
    """Run the verified data foundation, validation freeze, and final evaluation."""
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=eda_config_path, split_run_path=split_run_path)
    selection = run_candidate_selection(inputs=inputs, output_directory=output_directory, tuning_config_path=tuning_config_path)
    run_final_test_evaluation(selection, inputs=inputs)
    return selection


def main() -> None:
    """Execute Phase 3 once from the command line."""
    parser = argparse.ArgumentParser(description="Run controlled Phase 3 XGBoost selection and final TEST.")
    parser.add_argument("--output-directory", type=Path, default=MONTH_2_REPORT_DIR)
    parser.add_argument("--tuning-config", type=Path, default=TUNING_CONFIG_PATH)
    parser.add_argument("--split-run", type=Path)
    args = parser.parse_args()
    run_gradient_boosting_tuning_workflow(output_directory=args.output_directory, tuning_config_path=args.tuning_config, split_run_path=args.split_run)


if __name__ == "__main__":
    main()
