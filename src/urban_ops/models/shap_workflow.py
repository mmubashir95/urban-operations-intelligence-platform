"""Orchestrate frozen-model reconstruction and read-only Phase 5 explanations."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH, FrozenBaselineInputs
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel, load_gradient_boosting_config
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.gradient_boosting_reporting import MONTH_2_REPORT_DIR
from urban_ops.models.gradient_boosting_selection import METRIC_NAMES, evaluate_scores
from urban_ops.models.gradient_boosting_tuning import GradientBoostingCandidate
from urban_ops.models.shap_explainability import (
    SHAP_CONFIG_PATH, ShapExplainabilityError, additivity_summary, correct_vs_incorrect,
    deterministic_sample, direction_summary, explain_tree_model, global_importance,
    interaction_summary, load_shap_config, select_local_cases, stability_summary,
)
from urban_ops.models.shap_reporting import (
    build_local_explanations, build_report, completion_gate, write_dependence_figures,
    write_global_figures, write_waterfalls,
)
from urban_ops.utils.paths import PROJECT_ROOT

PHASE3_PATH = PROJECT_ROOT / "reports/month_2/phase_3_selected_configuration.json"
PHASE4_PATH = PROJECT_ROOT / "configs/models/resolution_risk_calibration_decision.json"
MODEL_DIRECTORY = PROJECT_ROOT / "models/gradient_boosting"
MODEL_PATH = MODEL_DIRECTORY / "phase_3_selected_model.ubj"
MODEL_METADATA_PATH = MODEL_DIRECTORY / "phase_3_selected_model_metadata.json"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports/figures"


def _file_hash(path: Path) -> str:
    """Return the SHA-256 digest of an immutable artifact."""
    return sha256(path.read_bytes()).hexdigest()


def _artifact_reference(path: Path) -> str:
    """Return a project-relative reference when possible, absolute in tests."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_frozen_artifacts(inputs: FrozenBaselineInputs, phase3_path: Path, phase4_path: Path) -> tuple[dict[str, object], dict[str, object], GradientBoostingCandidate]:
    """Validate Phase 3/4 identity, lineage, and frozen state."""
    phase3 = json.loads(phase3_path.read_text(encoding="utf-8")); phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
    if (not phase3.get("frozen") or phase3.get("selected_on_split") != "validation" or phase3.get("final_fit_policy") != "train_only"
        or phase3.get("split_id") != inputs.split_id or phase3.get("feature_fingerprint") != inputs.phase_9_contract.fingerprint
        or tuple(phase3.get("feature_names", ())) != inputs.feature_names or phase3.get("random_state") != 20260806):
        raise ShapExplainabilityError("Phase 3 frozen model differs from verified inputs or policy.")
    if (not phase4.get("frozen") or phase4.get("selected_method") != "ISOTONIC" or phase4.get("base_model_identifier") != phase3["candidate"]["candidate_id"]
        or phase4.get("split_id") != inputs.split_id or phase4.get("feature_fingerprint") != inputs.phase_9_contract.fingerprint
        or tuple(phase4.get("feature_names", ())) != inputs.feature_names):
        raise ShapExplainabilityError("Phase 4 frozen calibration decision differs from Phase 3 lineage.")
    candidate = GradientBoostingCandidate(**phase3["candidate"])
    if candidate.candidate_id != "shallow":
        raise ShapExplainabilityError("Phase 5 requires the frozen shallow candidate.")
    return phase3, phase4, candidate


def _reconstruct_and_persist(inputs: FrozenBaselineInputs, phase3: dict[str, object], candidate: GradientBoostingCandidate,
                             *, model_path: Path, metadata_path: Path) -> tuple[GradientBoostedRiskModel, dict[str, object]]:
    """Reconstruct once or validate and load the existing frozen artifact."""
    if model_path.exists() != metadata_path.exists():
        raise ShapExplainabilityError("Phase 3 model and metadata must either both exist or both be absent.")
    config = candidate.model_config(load_gradient_boosting_config())
    if model_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (not metadata.get("frozen") or metadata.get("candidate_id") != candidate.candidate_id
            or metadata.get("split_id") != inputs.split_id or metadata.get("feature_fingerprint") != inputs.phase_9_contract.fingerprint
            or tuple(metadata.get("feature_names", ())) != inputs.feature_names or metadata.get("model_sha256") != _file_hash(model_path)):
            raise ShapExplainabilityError("Persisted Phase 3 model metadata or checksum differs from frozen lineage.")
        model = GradientBoostedRiskModel(config=config)
        model._model.load_model(model_path); model.feature_names_ = inputs.feature_names; model.feature_count_ = len(inputs.feature_names)
        observed = dict(evaluate_scores(inputs.targets["validation"], model.predict_score(inputs.matrices["validation"])))
        expected = dict(phase3["validation_metrics"])
        if any(not np.isclose(observed[name], expected[name], rtol=0, atol=1e-12) for name in METRIC_NAMES):
            raise ShapExplainabilityError("Persisted model does not reproduce frozen Phase 3 validation evidence.")
        return model, metadata
    model = GradientBoostedRiskModel(config=config).fit(inputs.matrices["train"], inputs.targets["train"], feature_names=inputs.feature_names)
    validation_scores = model.predict_score(inputs.matrices["validation"])
    observed = dict(evaluate_scores(inputs.targets["validation"], validation_scores))
    expected = dict(phase3["validation_metrics"])
    if tuple(observed) != METRIC_NAMES or any(not np.isclose(observed[name], expected[name], rtol=0, atol=1e-12) for name in METRIC_NAMES):
        raise ShapExplainabilityError("Reconstructed model does not reproduce frozen Phase 3 validation evidence.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model._model.save_model(model_path)
    reloaded = GradientBoostedRiskModel(config=config)
    reloaded._model.load_model(model_path); reloaded.feature_names_ = inputs.feature_names; reloaded.feature_count_ = len(inputs.feature_names)
    reloaded_scores = reloaded.predict_score(inputs.matrices["validation"])
    if not np.array_equal(validation_scores, reloaded_scores):
        raise ShapExplainabilityError("Reloaded Phase 3 artifact changed validation scores.")
    metadata = {"artifact_version": 1, "candidate_id": candidate.candidate_id, "candidate_parameters": candidate.parameters(),
                "split_id": inputs.split_id, "feature_fingerprint": inputs.phase_9_contract.fingerprint,
                "feature_names": list(inputs.feature_names), "random_state": phase3["random_state"],
                "phase_3_selection_timestamp": phase3["selection_timestamp"], "phase_3_decision": "reports/month_2/phase_3_selected_configuration.json",
                "fit_policy": "exact_frozen_reconstruction_on_train_only", "validation_metrics": observed,
                "model_path": _artifact_reference(model_path), "model_sha256": _file_hash(model_path), "frozen": True}
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reloaded, metadata


def run_shap_explainability_workflow(*, eda_config_path: Path | str = EDA_CONFIG_PATH, split_run_path: Path | None = None,
                                     config_path: Path | str = SHAP_CONFIG_PATH, phase3_path: Path | str = PHASE3_PATH,
                                     phase4_path: Path | str = PHASE4_PATH, model_path: Path | str = MODEL_PATH,
                                     metadata_path: Path | str = MODEL_METADATA_PATH, output_directory: Path | str = MONTH_2_REPORT_DIR,
                                     figure_directory: Path | str = FIGURE_DIRECTORY) -> dict[str, object]:
    """Generate all governed Phase 5 evidence without changing earlier phases."""
    phase3_file, phase4_file = Path(phase3_path), Path(phase4_path)
    original_hashes = {_path: _file_hash(_path) for _path in (phase3_file, phase4_file)}
    config = load_shap_config(config_path)
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=eda_config_path, split_run_path=split_run_path)
    phase3, phase4, candidate = _load_frozen_artifacts(inputs, phase3_file, phase4_file)
    model, metadata = _reconstruct_and_persist(inputs, phase3, candidate, model_path=Path(model_path), metadata_path=Path(metadata_path))

    test_matrix = inputs.matrices[config.explanation_split]; test_targets = inputs.targets[config.explanation_split].to_numpy(dtype=int)
    test_frame = inputs.frames[config.explanation_split]; identifiers = test_frame[inputs.phase_9_contract.identifier_name].astype(str).to_numpy()
    scores = model.predict_score(test_matrix)
    roles = select_local_cases(scores, test_targets, identifiers, cutoff=config.descriptive_reference_cutoff)
    positions = deterministic_sample(len(scores), sample_size=config.sample_size, seed=config.random_state, reserved=tuple(roles.values()))
    sample_matrix = test_matrix[list(positions)]; sample_scores = scores[list(positions)]; sample_targets = test_targets[list(positions)]
    result = explain_tree_model(model._model.get_booster(), sample_matrix, feature_names=inputs.feature_names,
                                row_positions=positions, identifiers=tuple(identifiers[list(positions)]), raw_scores=sample_scores,
                                additivity_tolerance=config.additivity_tolerance)
    additivity = additivity_summary(result, tolerance=config.additivity_tolerance)
    importance = global_importance(result); direction = direction_summary(result)
    interactions = interaction_summary(model._model.get_booster(), result)
    stability_table, stability = stability_summary(result, seed=config.random_state, half_size=config.stability_half_size,
                                                   minimum_spearman=config.stability_minimum_spearman)
    comparison = correct_vs_incorrect(result, sample_targets, cutoff=config.descriptive_reference_cutoff)
    local = build_local_explanations(result, test_targets, roles, positions, cutoff=config.descriptive_reference_cutoff)

    output, figures = Path(output_directory), Path(figure_directory); output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output / "phase_5_global_shap_importance.csv", index=False)
    direction.to_csv(output / "phase_5_feature_direction_summary.csv", index=False)
    interactions.to_csv(output / "phase_5_shap_interactions.csv", index=False)
    local.to_csv(output / "phase_5_local_explanations.csv", index=False)
    comparison.to_csv(output / "phase_5_correct_vs_incorrect_shap.csv", index=False)
    stability_table.to_csv(output / "phase_5_shap_stability.csv", index=False)
    (output / "phase_5_additivity_verification.json").write_text(json.dumps(additivity, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "phase_5_explanation_sample.json").write_text(json.dumps({"split": "test", "population_size": len(scores), "sample_size": len(positions),
        "seed": config.random_state, "positions": positions, "case_positions": roles, "descriptive_reference_cutoff": config.descriptive_reference_cutoff}, indent=2) + "\n", encoding="utf-8")
    write_global_figures(result, importance, figures); write_dependence_figures(result, figures)
    write_waterfalls(result, local, figures / "phase_5_local")
    report = build_report(model_metadata=metadata, phase4=phase4, result=result, additivity=additivity, importance=importance,
                          direction=direction, interactions=interactions, local=local, comparison=comparison, stability=stability)
    (output / "phase_5_shap_explainability.md").write_text(report, encoding="utf-8")
    (output / "phase_5_completion_gate.md").write_text(completion_gate(interactions_completed=True), encoding="utf-8")
    if any(_file_hash(path) != digest for path, digest in original_hashes.items()):
        raise ShapExplainabilityError("A frozen Phase 3 or Phase 4 artifact changed during explanation.")
    return {"result": result, "importance": importance, "direction": direction, "interactions": interactions,
            "local": local, "comparison": comparison, "stability": stability, "metadata": metadata}


def main() -> None:
    """Run the one-time Phase 5 explanation workflow."""
    parser = argparse.ArgumentParser(description="Explain the frozen Phase 3 XGBoost model with Tree SHAP.")
    parser.add_argument("--split-run", type=Path)
    args = parser.parse_args(); run_shap_explainability_workflow(split_run_path=args.split_run)


if __name__ == "__main__":
    main()
