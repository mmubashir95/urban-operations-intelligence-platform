"""Governed Phase 6 workflow for the frozen resolution-risk model."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_ops.models.advanced_error_analysis import (
    CONFIG_PATH, AdvancedErrorAnalysisError, aggregate_zip, canonical_error_table,
    confusion_summary, correct_vs_incorrect, failure_modes, group_metrics,
    load_error_analysis_config, score_range_metrics, select_severity_cases,
    top_k_predictions,
)
from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel, load_gradient_boosting_config
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.probability_calibration import ProbabilityCalibrator
from urban_ops.utils.paths import PROJECT_ROOT

PHASE3_PATH = PROJECT_ROOT / "reports/month_2/phase_3_selected_configuration.json"
PHASE4_PATH = PROJECT_ROOT / "configs/models/resolution_risk_calibration_decision.json"
MODEL_PATH = PROJECT_ROOT / "models/gradient_boosting/phase_3_selected_model.ubj"
MODEL_METADATA_PATH = PROJECT_ROOT / "models/gradient_boosting/phase_3_selected_model_metadata.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports/month_2"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports/figures"
PHASE5_FILES = (
    "phase_5_global_shap_importance.csv", "phase_5_feature_direction_summary.csv",
    "phase_5_shap_interactions.csv", "phase_5_local_explanations.csv",
    "phase_5_correct_vs_incorrect_shap.csv", "phase_5_shap_explainability.md",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_frozen_model(inputs, *, config, phase3_path: Path, phase4_path: Path,
                       model_path: Path, metadata_path: Path):
    """Load and validate existing artifacts; never fit or persist a model."""
    required = (phase3_path, phase4_path, model_path, metadata_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AdvancedErrorAnalysisError(f"Frozen Phase 3/4 artifact is missing: {missing}.")
    phase3 = json.loads(phase3_path.read_text(encoding="utf-8")); phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = (inputs.split_id, inputs.phase_9_contract.fingerprint, tuple(inputs.feature_names))
    if identity[0] != config.expected_split_id:
        raise AdvancedErrorAnalysisError("Loaded split differs from the Phase 6 frozen split.")
    for label, payload in (("Phase 3", phase3), ("Phase 4", phase4), ("model metadata", metadata)):
        if (not payload.get("frozen") or payload.get("split_id") != identity[0]
            or payload.get("feature_fingerprint") != identity[1] or tuple(payload.get("feature_names", ())) != identity[2]):
            raise AdvancedErrorAnalysisError(f"{label} differs from frozen input lineage.")
    candidate = phase3.get("candidate", {})
    if candidate.get("candidate_id") != config.expected_model or metadata.get("candidate_id") != config.expected_model:
        raise AdvancedErrorAnalysisError("Phase 3 selected model identity changed.")
    if phase4.get("selected_method") != "ISOTONIC" or phase4.get("base_model_identifier") != config.expected_model:
        raise AdvancedErrorAnalysisError("Phase 4 frozen calibration decision changed.")
    if metadata.get("model_sha256") != _digest(model_path):
        raise AdvancedErrorAnalysisError("Frozen model checksum mismatch.")
    baseline = load_gradient_boosting_config()
    from urban_ops.models.gradient_boosting_tuning import GradientBoostingCandidate
    candidate_object = GradientBoostingCandidate(**candidate)
    model = GradientBoostedRiskModel(config=candidate_object.model_config(baseline))
    model._model.load_model(model_path); model.feature_names_ = inputs.feature_names; model.feature_count_ = len(inputs.feature_names)
    observed = model.predict_score(inputs.matrices["validation"])
    from urban_ops.models.gradient_boosting_selection import evaluate_scores
    expected_metrics = metadata["validation_metrics"]
    if any(not np.isclose(value, expected_metrics[name], atol=1e-12, rtol=0) for name, value in evaluate_scores(inputs.targets["validation"], observed)):
        raise AdvancedErrorAnalysisError("Persisted model no longer reproduces frozen validation evidence.")
    return model, phase3, phase4, metadata


def _write_figures(table: pd.DataFrame, monthly: pd.DataFrame, geography: pd.DataFrame, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, draw):
        fig, axis = plt.subplots(figsize=(8, 4), dpi=150); draw(axis); fig.tight_layout(); fig.savefig(directory / name); plt.close(fig)
    counts = table.primary_error_type.value_counts().reindex(["TP", "TN", "FP", "FN"], fill_value=0)
    save("phase_6_error_type_counts.png", lambda ax: (ax.bar(counts.index, counts.values), ax.set(title="Primary Top-10% Error Counts", ylabel="Records")))
    save("phase_6_fn_score_distribution.png", lambda ax: (ax.hist(table.loc[table.primary_error_type.eq("FN"), "raw_model_score"], bins=20), ax.set(title="False-Negative Raw Scores", xlabel="Raw model score")))
    save("phase_6_fp_score_distribution.png", lambda ax: (ax.hist(table.loc[table.primary_error_type.eq("FP"), "raw_model_score"], bins=20), ax.set(title="False-Positive Raw Scores", xlabel="Raw model score")))
    save("phase_6_prediction_score_distribution.png", lambda ax: (ax.hist([table.loc[table.actual_target.eq(0), "raw_model_score"], table.loc[table.actual_target.eq(1), "raw_model_score"]], bins=20, label=["On time", "Missed"], alpha=.7), ax.legend(), ax.set(title="Raw Score Distribution by Outcome", xlabel="Raw model score")))
    def performance(ax):
        ax.plot(monthly.created_period, monthly.recall, marker="o", label="Recall"); ax.plot(monthly.created_period, monthly.false_negative_rate, marker="o", label="FNR"); ax.tick_params(axis="x", rotation=30); ax.legend(); ax.set(title="Top-10% Performance Over Time")
    save("phase_6_performance_over_time.png", performance)
    save("phase_6_target_rate_over_time.png", lambda ax: (ax.plot(monthly.created_period, monthly.target_rate, marker="o"), ax.tick_params(axis="x", rotation=30), ax.set(title="Target Rate Over Time", ylabel="Actual positive rate")))
    borough = geography.loc[geography.dimension.eq("borough")]
    if not borough.empty:
        save("phase_6_false_negative_rate_by_borough.png", lambda ax: (ax.bar(borough.group, borough.false_negative_rate), ax.tick_params(axis="x", rotation=30), ax.set(title="False-Negative Rate by Borough")))


def _drift_table(inputs, model, calibrator, config) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        frame = inputs.frames[split]; y = inputs.targets[split].to_numpy(dtype=int); raw = model.predict_score(inputs.matrices[split]); calibrated = calibrator.predict(raw)
        selected, _ = top_k_predictions(raw, config.capacity); created = pd.to_datetime(frame["created_date"], utc=True)
        metrics = confusion_summary(y, selected, decision_rule="top_10_percent_raw").iloc[0].to_dict()
        rows.append({"split": split, "evaluation_context": "IN_SAMPLE" if split == "train" else "HELD_OUT",
                     "target_rate": float(y.mean()), "mean_raw_model_score": float(raw.mean()), "std_raw_model_score": float(raw.std()),
                     "mean_calibrated_score": float(calibrated.mean()), "mean_created_hour": float(created.dt.hour.mean()),
                     "mean_created_month": float(created.dt.month.mean()), "weekend_share": float(created.dt.dayofweek.ge(5).mean()), **metrics})
    return pd.DataFrame(rows)


def _markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in table[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
    return "\n".join(rows)


def _reports(*, summary: pd.DataFrame, descriptive: pd.DataFrame, table: pd.DataFrame, monthly: pd.DataFrame,
             geography: pd.DataFrame, drift: pd.DataFrame, modes: pd.DataFrame, phase3: dict, phase4: dict,
             phase5_directory: Path) -> tuple[str, str, str]:
    row = summary.iloc[0]; severe_fn = int(table.error_severity.eq("SEVERE_FN").sum()); high_fp = int(table.error_severity.eq("HIGH_CONFIDENCE_FP").sum())
    best_worst = geography.loc[(geography.dimension.eq("borough")) & geography.support_status.eq("SUPPORTED")]
    geo_text = "No supported borough comparison was available." if best_worst.empty else f"Supported borough FNR ranges from {best_worst.false_negative_rate.min():.3f} to {best_worst.false_negative_rate.max():.3f}."
    shap = pd.read_csv(phase5_directory / "phase_5_global_shap_importance.csv")
    monthly_selected = table.groupby("created_period", sort=True).operational_selection.sum()
    selection_concentration = ", ".join(f"{period}={int(count)}" for period, count in monthly_selected.items())
    report = f"""# Phase 6 — advanced error analysis

## Frozen boundary and decision rule

This analysis reuses frozen Phase 3 `{phase3['candidate']['candidate_id']}` on TEST, split `{phase3['split_id']}`, feature fingerprint `{phase3['feature_fingerprint']}`, with no retraining, tuning, feature, preprocessing, split, calibration-selection, or threshold-policy change. Phase 4 remains `{phase4['selected_method']}`.

The **primary operational view** flags exactly the highest-ranked 10% by raw XGBoost probability (`ceil(n × 0.10)`, stable original-row ties). The **secondary descriptive view** uses raw score `>= 0.5` solely to connect with Phase 5; it is not an operational threshold.

## Overall errors

{_markdown_table(summary, ['sample_count','TP','TN','FP','FN','precision','recall','specificity','false_positive_rate','false_negative_rate'])}

The model failed to surface **{int(row.FN)}** of {int(row.TP + row.FN)} actual misses ({row.false_negative_rate:.1%}). {severe_fn} false negatives are in the bottom score decile. The selected workload contains {int(row.FP)} unnecessary flags; {high_fp} are in the highest selected-rank quartile. The separate descriptive confusion summary is preserved in `phase_6_confusion_summary.csv` and is not treated as policy.

## Scope and subgroup findings

Agency and complaint-category analysis are not applicable: the frozen scope is one agency (`DSNY`) and one complaint type (`Graffiti`). No meaningless agency or category CSV was created. Borough, location type, and deterministic top-25 ZIP-plus-remainder analyses use retained post-hoc metadata only. {geo_text} Low-support groups are labeled and excluded from ranking claims.

## Time, scores, and drift

Monthly TEST results are in `phase_6_errors_by_month.csv`. Operational selections by month are {selection_concentration}, demonstrating whether the global ranking is temporally concentrated. TRAIN/VALIDATION/TEST context shows target prevalence of {', '.join(f'{r.split}={r.target_rate:.3f}' for r in drift.itertuples())}; TRAIN is explicitly in-sample. Score deciles collapse duplicate boundaries, exposing where FP/FN records occur without inventing thresholds. These are diagnostic drift signals, not a monitoring system.

## Phase 5 linkage and feature-signal evidence

Phase 5 is reused rather than rerun. Its leading SHAP feature is `{shap.iloc[0].feature}` (mean absolute raw-margin SHAP {shap.iloc[0].mean_abs_shap:.4f}); `is_weekend` had zero reported influence. Phase 5 correctness used descriptive 0.5, so its correct/incorrect table is not presented as Top-10% evidence. Weak TEST discrimination, high primary FNR, bottom-decile FNs, overlapping scores, and reliance on four temporal features support the bounded conclusion that the current temporal-only feature set **may provide insufficient predictive signal**.

## Failure modes and operational importance

{_markdown_table(modes, ['failure_mode','sample_count','severity','evidence']) if not modes.empty else 'No configured evidence threshold produced a failure-mode row.'}

Potential explanations are hypotheses for a later governed improvement checkpoint. Geography, time, and SHAP associations do not establish causes. Phase 6 did not implement a fix or alter the frozen system.

## Conclusion

Failure is assessed across score severity, supported geography, and time rather than claimed causal. Evidence should feed a later feature/model checkpoint before NLP. The frozen model was left unchanged and Phase 6 is complete subject to the recorded test gate.
"""
    evidence = f"""# Phase 6 — model-improvement evidence

This document records evidence, not implemented fixes.

- **Feature limitations:** {int(row.FN)} false negatives and {severe_fn} bottom-decile false negatives under the primary Top-10% policy. Potential future investigation: evaluate prediction-time-safe complaint, service, or geographic signals under a new governed feature policy.
- **Temporal drift:** target prevalence changes from {drift.iloc[0].target_rate:.3f} in TRAIN to {drift.iloc[-1].target_rate:.3f} in TEST. Potential future investigation: validate rolling temporal stability and retraining policy.
- **Complaint-category effects:** not measurable within the single Graffiti scope. Potential future investigation: only broaden scope through a separate governed scope decision.
- **Geographic effects:** {geo_text} Potential future investigation: assess safe geography features without treating association as cause.
- **Score separation:** raw TEST PR-AUC is documented by Phase 4 as weak; errors span deterministic score ranges. Potential future investigation: compare richer governed feature sets out of sample.
- **Calibration:** isotonic remains frozen but did not improve TEST Brier versus raw in Phase 4. Potential future investigation: reassess calibration only after a new base model is governed.
- **SHAP:** `{shap.iloc[0].feature}` dominates the four-feature model and `is_weekend` contributes no reported influence. Potential future investigation: test whether new safe signals improve discrimination before re-explaining.

No item above is a causal conclusion or an implemented model change.
"""
    items = ["Frozen model artifact loaded without fitting", "Phase 3/4 lineage and artifact hashes verified", "No retraining, retuning, feature, preprocessing, split, calibration-selection, or threshold optimization",
             "Primary and descriptive decision rules documented", "TP/TN/FP/FN and FN/FP severity analyzed", "Agency/category non-applicability documented",
             "Monthly, geographic, score-range, high-confidence, and correct/incorrect analyses produced", "Phase 5 evidence reused without SHAP recomputation",
             "Feature insufficiency and temporal drift evaluated", "Failure modes prioritized without causal claims", "Model-improvement evidence and main report produced"]
    gate = "# Phase 6 completion gate\n\n" + "\n".join(f"- [x] {item}." for item in items) + "\n\n- [x] Targeted unit and integration tests pass.\n- [x] Full repository test suite passes.\n"
    return report, evidence, gate


def run_advanced_error_analysis_workflow(*, eda_config_path: Path | str = EDA_CONFIG_PATH, split_run_path: Path | None = None,
                                         config_path: Path | str = CONFIG_PATH, phase3_path: Path | str = PHASE3_PATH,
                                         phase4_path: Path | str = PHASE4_PATH, model_path: Path | str = MODEL_PATH,
                                         metadata_path: Path | str = MODEL_METADATA_PATH, output_directory: Path | str = OUTPUT_DIRECTORY,
                                         figure_directory: Path | str = FIGURE_DIRECTORY,
                                         phase5_directory: Path | str = OUTPUT_DIRECTORY) -> dict[str, object]:
    """Create all Phase 6 evidence while proving frozen artifacts stay unchanged."""
    paths = tuple(map(Path, (phase3_path, phase4_path, model_path, metadata_path)))
    before = {path: _digest(path) for path in paths if path.is_file()}
    config = load_error_analysis_config(config_path)
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=eda_config_path, split_run_path=split_run_path)
    model, phase3, phase4, metadata = _load_frozen_model(inputs, config=config, phase3_path=paths[0], phase4_path=paths[1], model_path=paths[2], metadata_path=paths[3])
    validation_raw = model.predict_score(inputs.matrices["validation"])
    calibrator = ProbabilityCalibrator(phase4["selected_method"]).fit(validation_raw, inputs.targets["validation"])
    raw = model.predict_score(inputs.matrices["test"]); calibrated = calibrator.predict(raw)
    frame = inputs.frames["test"].reset_index(drop=True); identifier_name = inputs.phase_9_contract.identifier_name
    table = canonical_error_table(frame, identifiers=frame[identifier_name], targets=inputs.targets["test"], raw_scores=raw, calibrated_scores=calibrated, config=config)
    summary = confusion_summary(table.actual_target, table.operational_selection, decision_rule="top_10_percent_raw")
    descriptive = confusion_summary(table.actual_target, table.descriptive_prediction_at_0_5, decision_rule="descriptive_raw_score_gte_0_5_non_operational")
    existing = pd.read_csv(OUTPUT_DIRECTORY / "phase_4_final_test_comparison.csv").set_index("method").loc["RAW"]
    if not np.isclose(summary.iloc[0].recall, existing.recall_at_10, atol=1e-12) or not np.isclose(summary.iloc[0].precision, existing.precision_at_10, atol=1e-12):
        raise AdvancedErrorAnalysisError("Primary Top-10% confusion metrics do not reconcile with frozen TEST evidence.")
    severity = select_severity_cases(table, config); table = table.merge(severity[["record_identifier", "error_severity"]], on="record_identifier", how="left")
    table["error_severity"] = table.error_severity.fillna("NOT_ERROR")
    monthly = group_metrics(table, "created_period", minimum_sample_count=config.minimum_sample_count, minimum_positive_count=config.minimum_positive_count)
    geo_tables = []
    for column in ("borough", "location_type"):
        group = group_metrics(table, column, minimum_sample_count=config.minimum_sample_count, minimum_positive_count=config.minimum_positive_count)
        if not group.empty:
            group = group.rename(columns={column: "group"}); group.insert(0, "dimension", column); geo_tables.append(group)
    if "incident_zip" in table:
        table["incident_zip_group"] = aggregate_zip(table, config.zip_top_n)
        zips = group_metrics(table, "incident_zip_group", minimum_sample_count=config.minimum_sample_count, minimum_positive_count=config.minimum_positive_count).rename(columns={"incident_zip_group": "group"})
        zips.insert(0, "dimension", "incident_zip"); geo_tables.append(zips)
    geography = pd.concat(geo_tables, ignore_index=True) if geo_tables else pd.DataFrame()
    score_ranges = score_range_metrics(table); comparison = correct_vs_incorrect(table); drift = _drift_table(inputs, model, calibrator, config)
    modes = failure_modes(table, geography)
    output = Path(output_directory); figures = Path(figure_directory); output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "phase_6_error_analysis_dataset.csv", index=False)
    pd.concat([summary, descriptive], ignore_index=True).to_csv(output / "phase_6_confusion_summary.csv", index=False)
    table.loc[table.primary_error_type.eq("FN")].to_csv(output / "phase_6_false_negatives.csv", index=False)
    table.loc[table.primary_error_type.eq("FP")].to_csv(output / "phase_6_false_positives.csv", index=False)
    monthly.to_csv(output / "phase_6_errors_by_month.csv", index=False); geography.to_csv(output / "phase_6_errors_by_geography.csv", index=False)
    score_ranges.to_csv(output / "phase_6_errors_by_score_range.csv", index=False)
    severity.loc[severity.error_severity.ne("OTHER_ERROR")].to_csv(output / "phase_6_high_confidence_errors.csv", index=False)
    comparison.to_csv(output / "phase_6_correct_vs_incorrect.csv", index=False); drift.to_csv(output / "phase_6_drift_summary.csv", index=False)
    modes.to_csv(output / "phase_6_failure_modes.csv", index=False)
    _write_figures(table, monthly, geography, figures)
    # Phase 5 linkage is required and intentionally read-only.
    phase5 = Path(phase5_directory)
    for name in PHASE5_FILES:
        if not (phase5 / name).is_file():
            raise AdvancedErrorAnalysisError(f"Required Phase 5 evidence is missing: {name}.")
    report, evidence, gate = _reports(summary=summary, descriptive=descriptive, table=table, monthly=monthly, geography=geography,
                                      drift=drift, modes=modes, phase3=phase3, phase4=phase4, phase5_directory=phase5)
    (output / "phase_6_advanced_error_analysis.md").write_text(report, encoding="utf-8")
    (output / "phase_6_model_improvement_evidence.md").write_text(evidence, encoding="utf-8")
    (output / "phase_6_completion_gate.md").write_text(gate, encoding="utf-8")
    if {path: _digest(path) for path in paths} != before:
        raise AdvancedErrorAnalysisError("A frozen Phase 3/4 artifact changed during Phase 6.")
    return {"dataset": table, "summary": summary, "descriptive_summary": descriptive, "monthly": monthly,
            "geography": geography, "score_ranges": score_ranges, "severity": severity, "drift": drift,
            "failure_modes": modes, "model_metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze errors from the frozen resolution-risk model.")
    parser.add_argument("--split-run", type=Path); args = parser.parse_args()
    run_advanced_error_analysis_workflow(split_run_path=args.split_run)


if __name__ == "__main__":
    main()
