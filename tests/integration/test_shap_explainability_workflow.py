"""Integration coverage for frozen Phase 5 Tree SHAP orchestration."""
from hashlib import sha256

import pytest

from urban_ops.models import shap_workflow
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.shap_explainability import EXPECTED_FEATURES


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def test_real_workflow_reuses_frozen_lineage_and_generates_complete_evidence(tmp_path, monkeypatch):
    frozen_before = {path: digest(path) for path in (shap_workflow.PHASE3_PATH, shap_workflow.PHASE4_PATH)}
    original_fit = GradientBoostedRiskModel.fit; fits = []
    def fit(model, matrix, target, *, feature_names):
        fits.append((matrix, target, tuple(feature_names)))
        return original_fit(model, matrix, target, feature_names=feature_names)
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", fit)
    output, figures, models = tmp_path / "reports", tmp_path / "figures", tmp_path / "models"
    result = shap_workflow.run_shap_explainability_workflow(
        model_path=models / "model.ubj", metadata_path=models / "metadata.json",
        output_directory=output, figure_directory=figures,
    )
    assert len(fits) == 1 and fits[0][2] == EXPECTED_FEATURES
    assert result["metadata"]["candidate_id"] == "shallow"
    assert result["metadata"]["fit_policy"] == "exact_frozen_reconstruction_on_train_only"
    assert result["result"].feature_names == EXPECTED_FEATURES
    assert result["result"].shap_values.shape == (2000, 4)
    assert result["stability"]["status"] in {"REASONABLY_STABLE", "UNSTABLE"}
    for name in ("phase_5_global_shap_importance.csv", "phase_5_feature_direction_summary.csv", "phase_5_shap_interactions.csv",
                 "phase_5_local_explanations.csv", "phase_5_correct_vs_incorrect_shap.csv", "phase_5_shap_stability.csv",
                 "phase_5_shap_explainability.md", "phase_5_completion_gate.md", "phase_5_additivity_verification.json",
                 "phase_5_explanation_sample.json"):
        assert (output / name).is_file()
    for name in ("phase_5_global_shap_importance.png", "phase_5_shap_summary.png", *[f"phase_5_dependence_{feature}.png" for feature in EXPECTED_FEATURES]):
        assert (figures / name).is_file()
    assert len(tuple((figures / "phase_5_local").glob("*_waterfall.png"))) == 6
    report = (output / "phase_5_shap_explainability.md").read_text()
    assert "does not establish causal relationships" in report
    assert "weak TEST discrimination" in report
    assert "not attributed" in report and "KernelExplainer was not used" in report
    assert {path: digest(path) for path in frozen_before} == frozen_before
    repeated = shap_workflow.run_shap_explainability_workflow(model_path=models / "model.ubj", metadata_path=models / "metadata.json",
                                                              output_directory=output, figure_directory=figures)
    assert len(fits) == 1
    assert repeated["metadata"] == result["metadata"]
