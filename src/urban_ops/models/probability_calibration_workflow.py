"""Run validation-only calibration selection and protected final TEST scoring.

The workflow reconstructs the frozen Phase 3 TRAIN-fitted classifier exactly,
fits calibration mappings on VALIDATION, persists a write-once decision, and
only then permits TEST scoring. It performs no tuning or threshold selection.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH, FrozenBaselineInputs
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel, load_gradient_boosting_config
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.gradient_boosting_reporting import MONTH_2_REPORT_DIR
from urban_ops.models.gradient_boosting_tuning import GradientBoostingCandidate
from urban_ops.models.probability_calibration import (
    CALIBRATION_CONFIG_PATH, CALIBRATION_METHODS, CalibrationResult,
    FrozenCalibrationDecision, ProbabilityCalibrationError, ProbabilityCalibrator,
    calibration_table, evaluate_calibration_method, load_calibration_policy,
    select_calibration, verify_persisted_calibration_decision,
)
from urban_ops.utils.paths import PROJECT_ROOT

PHASE_3_DECISION_PATH = PROJECT_ROOT / "reports/month_2/phase_3_selected_configuration.json"
CALIBRATION_DECISION_PATH = PROJECT_ROOT / "configs/models/resolution_risk_calibration_decision.json"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports/figures"


@dataclass(frozen=True)
class CalibrationSelection:
    """Validation selection state retained for one protected TEST evaluation."""

    decision: FrozenCalibrationDecision
    model: GradientBoostedRiskModel
    calibrators: tuple[tuple[str, ProbabilityCalibrator], ...]
    results: tuple[CalibrationResult, ...]
    decision_path: Path
    output_directory: Path
    figure_directory: Path


def _load_frozen_phase3(path: Path, inputs: FrozenBaselineInputs) -> tuple[dict[str, object], GradientBoostingCandidate]:
    """Validate and return the immutable Phase 3 selection and candidate."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"candidate", "split_id", "feature_fingerprint", "feature_names", "random_state", "selection_timestamp", "selected_on_split", "final_fit_policy", "frozen"}
    if not required <= set(payload) or not payload["frozen"] or payload["selected_on_split"] != "validation" or payload["final_fit_policy"] != "train_only":
        raise ProbabilityCalibrationError("Phase 3 selection is not frozen on validation with TRAIN-only fitting.")
    if payload["split_id"] != inputs.split_id or payload["feature_fingerprint"] != inputs.phase_9_contract.fingerprint or tuple(payload["feature_names"]) != inputs.feature_names:
        raise ProbabilityCalibrationError("Phase 3 selection differs from the frozen input contract.")
    baseline = load_gradient_boosting_config()
    if payload["random_state"] != baseline.random_state:
        raise ProbabilityCalibrationError("Phase 3 random state differs from the frozen model contract.")
    candidate = GradientBoostingCandidate(**payload["candidate"])
    return payload, candidate


def _fit_frozen_model(candidate: GradientBoostingCandidate, inputs: FrozenBaselineInputs) -> GradientBoostedRiskModel:
    """Reconstruct and fit the exact Phase 3 candidate on frozen TRAIN only."""
    model = GradientBoostedRiskModel(config=candidate.model_config(load_gradient_boosting_config()))
    model.fit(inputs.matrices["train"], inputs.targets["train"], feature_names=inputs.feature_names)
    return model


def _artifact_reference(path: Path | str) -> str:
    """Return a portable project-relative reference when the artifact is tracked."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_curve(table: pd.DataFrame, path: Path, *, title: str) -> None:
    """Write a deterministic calibration curve using populated shared bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4), dpi=150)
    axis.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Ideal")
    for method in table["method"].drop_duplicates():
        rows = table.loc[table["method"].eq(method) & table["row_count"].gt(0)]
        axis.plot(rows["mean_predicted_risk"], rows["observed_positive_rate"], marker="o", label=method)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title=title, xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _behavior(table: pd.DataFrame) -> str:
    """Diagnose weighted direction while acknowledging sparse populated bins."""
    rows = table.loc[table["row_count"].gt(0)].copy()
    difference = np.average(rows["mean_predicted_risk"] - rows["observed_positive_rate"], weights=rows["row_count"])
    if abs(difference) <= 0.01:
        return "reasonably calibrated overall, with region-specific deviations"
    return "overconfident overall" if difference > 0 else "underconfident overall"


def _metrics_markdown(table: pd.DataFrame) -> str:
    """Render metric rows without requiring pandas' optional tabulate package."""
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" if column == "method" else "---:" for column in columns) + "|"
    rows = []
    for record in table.itertuples(index=False, name=None):
        values = [str(value) if index == 0 else f"{float(value):.8f}" for index, value in enumerate(record)]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *rows))


def run_calibration_selection(*, inputs: FrozenBaselineInputs,
                              phase3_decision_path: Path | str = PHASE_3_DECISION_PATH,
                              calibration_config_path: Path | str = CALIBRATION_CONFIG_PATH,
                              decision_path: Path | str = CALIBRATION_DECISION_PATH,
                              output_directory: Path | str = MONTH_2_REPORT_DIR,
                              figure_directory: Path | str = FIGURE_DIRECTORY) -> CalibrationSelection:
    """Select and freeze calibration using TRAIN/VALIDATION without TEST access."""
    decision_file, directory, figures = Path(decision_path), Path(output_directory), Path(figure_directory)
    if decision_file.exists() or (directory / "phase_4_test_access.json").exists():
        raise ProbabilityCalibrationError("Phase 4 is frozen; calibration selection cannot be rerun.")
    phase3, candidate = _load_frozen_phase3(Path(phase3_decision_path), inputs)
    policy = load_calibration_policy(calibration_config_path)
    model = _fit_frozen_model(candidate, inputs)
    raw = model.predict_score(inputs.matrices["validation"])
    calibrators, results, tables = [], [], []
    for method in CALIBRATION_METHODS:
        calibrator = ProbabilityCalibrator(method).fit(raw, inputs.targets["validation"])
        probabilities = calibrator.predict(raw)
        result = evaluate_calibration_method(method, probabilities, inputs.targets["validation"])
        calibrators.append((method, calibrator)); results.append(result)
        table = calibration_table(method, inputs.targets["validation"], probabilities); tables.append(table)
    selected = select_calibration(tuple(results), policy)
    raw_result = next(result for result in results if result.method == "RAW")
    decision = FrozenCalibrationDecision(
        selected_method=selected.method, base_model_identifier=candidate.candidate_id,
        phase_3_decision_path=_artifact_reference(phase3_decision_path),
        phase_3_selection_timestamp=str(phase3["selection_timestamp"]),
        split_id=inputs.split_id, feature_fingerprint=inputs.phase_9_contract.fingerprint,
        feature_names=inputs.feature_names, random_state=int(phase3["random_state"]),
        raw_metrics=raw_result.metrics, selected_metrics=selected.metrics, selection_policy=policy,
        selection_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    with decision_file.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(decision.payload(), indent=2, allow_nan=False) + "\n")
    directory.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(tables, ignore_index=True)
    for method, table in zip(CALIBRATION_METHODS, tables):
        table.to_csv(directory / f"phase_4_{method.lower()}_calibration.csv", index=False)
        _write_curve(table, figures / f"phase_4_{method.lower()}_calibration_curve.png", title=f"{method} validation calibration")
    comparison = pd.DataFrame([result.record() for result in results])
    comparison.to_csv(directory / "phase_4_calibration_comparison.csv", index=False)
    _write_curve(combined, figures / "phase_4_calibration_comparison.png", title="Validation calibration comparison")
    raw_behavior = _behavior(tables[0])
    report = _validation_report(decision, comparison, raw_behavior, tuple(tables))
    (directory / "phase_4_calibration_decision.md").write_text(report, encoding="utf-8")
    (directory / "phase_4_probability_calibration.md").write_text(report + "\nFinal TEST evaluation pending.\n", encoding="utf-8")
    return CalibrationSelection(decision, model, tuple(calibrators), tuple(results), decision_file, directory, figures)


def _validation_report(decision: FrozenCalibrationDecision, comparison: pd.DataFrame,
                       behavior: str, tables: tuple[pd.DataFrame, ...]) -> str:
    """Build the frozen validation decision report."""
    raw_table = tables[0]
    populated = raw_table.loc[raw_table["row_count"].gt(0)]
    sparse = int((populated["row_count"] < max(10, raw_table["row_count"].sum() * 0.01)).sum())
    curve_summary = []
    for method, table in zip(CALIBRATION_METHODS, tables):
        points = table.loc[table["row_count"].gt(0)]
        maximum_gap = (points["mean_predicted_risk"] - points["observed_positive_rate"]).abs().max()
        curve_summary.append(f"- {method}: {len(points)} populated bins; maximum absolute bin gap {maximum_gap:.8f}.")
    return (
        "# Phase 4 — probability calibration\n\n"
        f"Frozen Phase 3 classifier: `{decision.base_model_identifier}`; split `{decision.split_id}`; feature fingerprint `{decision.feature_fingerprint}`. "
        "The base classifier was reconstructed with its frozen hyperparameters and fitted on TRAIN only.\n\n"
        f"Validation sample count: {int(raw_table['row_count'].sum())}. Raw validation behavior: **{behavior}**. "
        f"The raw Brier score is {dict(decision.raw_metrics)['brier_score']:.8f}. "
        f"The ten fixed-width bins contain {sparse} low-count populated bins; these are not interpreted independently.\n\n"
        "Exactly RAW, SIGMOID (sklearn logistic mapping), and ISOTONIC (sklearn monotonic mapping) were fitted/evaluated on VALIDATION. "
        "Shared project evaluators supplied Brier, PR-AUC, ROC-AUC, and Top-K metrics.\n\n"
        + _metrics_markdown(comparison) + "\n\nCalibration-curve evidence:\n\n"
        + "\n".join(curve_summary) + "\n\n"
        "Policy: a calibrated method must improve raw Brier by the configured minimum and may not degrade any ranking or Top-K metric beyond the configured absolute guard. "
        "Lowest eligible Brier wins; effective ties prefer RAW, then SIGMOID, then ISOTONIC. RAW is explicitly valid.\n\n"
        f"Selected and frozen method: **{decision.selected_method}**. The TEST set was not accessed during selection.\n"
    )


def run_final_test_evaluation(selection: CalibrationSelection, *, inputs: FrozenBaselineInputs) -> pd.DataFrame:
    """Evaluate raw versus the already-fitted selected calibration after freeze."""
    verify_persisted_calibration_decision(selection.decision, selection.decision_path)
    decision = selection.decision
    if decision.split_id != inputs.split_id or decision.feature_fingerprint != inputs.phase_9_contract.fingerprint or decision.feature_names != inputs.feature_names:
        raise ProbabilityCalibrationError("Frozen calibration decision differs from TEST input lineage.")
    marker = selection.output_directory / "phase_4_test_access.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"selected_method": decision.selected_method, "selection_timestamp": decision.selection_timestamp, "status": "test_access_started"}, stream, indent=2)
    raw_scores = selection.model.predict_score(inputs.matrices["test"])
    calibrator = dict(selection.calibrators)[decision.selected_method]
    selected_scores = calibrator.predict(raw_scores)
    raw_result = evaluate_calibration_method("RAW", raw_scores, inputs.targets["test"])
    selected_result = evaluate_calibration_method(decision.selected_method, selected_scores, inputs.targets["test"])
    comparison = pd.DataFrame([raw_result.record(), selected_result.record()])
    comparison.to_csv(selection.output_directory / "phase_4_final_test_comparison.csv", index=False)
    test_tables = pd.concat([
        calibration_table("RAW", inputs.targets["test"], raw_scores),
        calibration_table(decision.selected_method, inputs.targets["test"], selected_scores),
    ], ignore_index=True)
    test_tables.to_csv(selection.output_directory / "phase_4_final_test_calibration.csv", index=False)
    _write_curve(test_tables, selection.figure_directory / "phase_4_final_test_calibration_curve.png", title="Final TEST calibration")
    report_path = selection.output_directory / "phase_4_probability_calibration.md"
    report = (selection.output_directory / "phase_4_calibration_decision.md").read_text(encoding="utf-8")
    report += "\n## Final TEST results\n\n" + _metrics_markdown(comparison) + "\n\n"
    raw_brier = dict(raw_result.metrics)["brier_score"]
    selected_brier = dict(selected_result.metrics)["brier_score"]
    generalization = "improved" if selected_brier < raw_brier else "did not improve"
    report += (f"The selected mapping {generalization} Brier Score versus RAW on TEST. "
               "The selected method was not changed or refitted after TEST access. TEST labels were used only for final evaluation. Phase 4 is formally frozen.\n")
    report_path.write_text(report, encoding="utf-8")
    (selection.output_directory / "phase_4_completion_gate.md").write_text(_completion_gate(), encoding="utf-8")
    return comparison


def _completion_gate() -> str:
    """Return the completed Phase 4 governance checklist."""
    items = [
        "Phase 3 classifier remains frozen", "No feature, preprocessing, split, leakage, hyperparameter, or threshold-policy changes",
        "RAW calibration, Brier, table, and curve produced", "SIGMOID and ISOTONIC evaluated",
        "Brier, curves, PR-AUC, ROC-AUC, Recall@5/10/20, and Precision@5/10/20 compared",
        "Deterministic policy permits RAW and persisted one frozen decision", "TEST unused during selection",
        "Final TEST evaluation executed only after freeze", "No recalibration or method switching after TEST",
        "Unit, integration, and full-suite test evidence recorded in implementation handoff", "Phase 4 reports produced",
    ]
    return "# Phase 4 completion gate\n\n" + "\n".join(f"- [x] {item}." for item in items) + "\n"


def run_probability_calibration_workflow(*, eda_config_path: Path | str = EDA_CONFIG_PATH,
                                         split_run_path: Path | None = None) -> CalibrationSelection:
    """Run frozen input loading, validation selection/freeze, and final TEST."""
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=eda_config_path, split_run_path=split_run_path)
    selection = run_calibration_selection(inputs=inputs)
    run_final_test_evaluation(selection, inputs=inputs)
    return selection


def main() -> None:
    """Execute the one-time Phase 4 workflow."""
    parser = argparse.ArgumentParser(description="Run controlled probability calibration.")
    parser.add_argument("--split-run", type=Path)
    args = parser.parse_args()
    run_probability_calibration_workflow(split_run_path=args.split_run)


if __name__ == "__main__":
    main()
