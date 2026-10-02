"""Verify selection and final TEST boundaries against real frozen-input fixtures."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from urban_ops.models import gradient_boosting_selection as selection_module
from urban_ops.models import gradient_boosting_tuning_workflow as workflow
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.gradient_boosting_tuning import TUNING_CONFIG_PATH, GradientBoostingTuningError
from urban_ops.models.gradient_boosting_workflow import run_gradient_boosting_workflow
from tests.integration.test_gradient_boosting_workflow import ProtectedSplits, _write_matching_frozen_evidence
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


@pytest.fixture
def experiment(tmp_path):
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=fixture.config)
    evidence = _write_matching_frozen_evidence(tmp_path, fixture, inputs)
    phase2 = tmp_path / "phase2"
    run_gradient_boosting_workflow(eda_config_path=fixture.config, output_directory=phase2, frozen_evidence_path=evidence)
    config = yaml.safe_load(TUNING_CONFIG_PATH.read_text())
    config["frozen_input_reference"] = str(evidence)
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return inputs, evidence, phase2, config_path


def run_selection(experiment, output_directory, inputs=None):
    original, evidence, phase2, config = experiment
    return workflow.run_candidate_selection(inputs=inputs or original, tuning_config_path=config,
        output_directory=output_directory, frozen_evidence_path=evidence, phase2_directory=phase2)


def test_selection_uses_train_and_validation_only_and_shared_evaluators(experiment, tmp_path, monkeypatch):
    inputs, _, _, _ = experiment
    matrices = ProtectedSplits(inputs.matrices)
    targets = ProtectedSplits(inputs.targets)
    protected = replace(inputs, matrices=matrices, targets=targets)
    fits, scored, evaluations = [], [], []
    original_fit = GradientBoostedRiskModel.fit
    original_score = GradientBoostedRiskModel.predict_score
    def fit(model, matrix, target, *, feature_names):
        fits.append((matrix, target, feature_names))
        return original_fit(model, matrix, target, feature_names=feature_names)
    def score(model, matrix):
        scored.append(matrix)
        return original_score(model, matrix)
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", fit)
    monkeypatch.setattr(GradientBoostedRiskModel, "predict_score", score)
    for name in ("evaluate_ranking", "evaluate_calibration", "compare_capacity_levels"):
        original = getattr(selection_module, name)
        def spy(target, scores, evaluator=original, label=name):
            evaluations.append((label, target))
            return evaluator(target, scores)
        monkeypatch.setattr(selection_module, name, spy)
    result = run_selection(experiment, tmp_path / "selection", protected)
    assert len(fits) == len(scored) == 6
    assert all(matrix is inputs.matrices["train"] and target is inputs.targets["train"] and names == inputs.feature_names for matrix, target, names in fits)
    assert all(matrix is inputs.matrices["validation"] for matrix in scored)
    assert len(evaluations) == 18
    assert all(target is inputs.targets["validation"] for _, target in evaluations)
    assert "test" not in matrices.accessed + targets.accessed
    assert result.decision.split_id == inputs.split_id
    assert result.decision.feature_fingerprint == inputs.phase_9_contract.fingerprint
    assert result.decision.feature_names == inputs.feature_names
    assert result.decision.frozen
    assert result.decision_path.is_file()
    table = pd.read_csv(result.output_directory / "phase_3_candidate_comparison.csv")
    assert "phase2_baseline" in table.candidate_id.tolist()
    assert np.isfinite(table[list(selection_module.METRIC_NAMES)].to_numpy()).all()
    assert not tuple(result.output_directory.glob("phase_3_*test*"))
    second = run_selection(experiment, tmp_path / "second")
    assert second.decision.candidate == result.decision.candidate
    for (_, first), (_, other) in zip(result.validation_scores, second.validation_scores):
        np.testing.assert_array_equal(first, other)
    with pytest.raises(GradientBoostingTuningError, match="frozen"):
        run_selection(experiment, result.output_directory)


def test_rejects_wrong_split_fingerprint_and_phase2_evidence(experiment, tmp_path):
    inputs, _, phase2, _ = experiment
    with pytest.raises(GradientBoostingTuningError, match="split ID"):
        run_selection(experiment, tmp_path / "wrong", replace(inputs, split_id="wrong"))
    changed_contract = replace(inputs.phase_9_contract, preprocessing_schema_fingerprint="wrong")
    with pytest.raises(GradientBoostingTuningError, match="fingerprint"):
        run_selection(experiment, tmp_path / "wrong", replace(inputs, phase_9_contract=changed_contract))
    path = phase2 / "phase_2_gradient_boosting_validation.csv"
    table = pd.read_csv(path)
    table.loc[0, "pr_auc"] = 0.99
    table.to_csv(path, index=False)
    with pytest.raises(GradientBoostingTuningError, match="reproduce"):
        run_selection(experiment, tmp_path / "wrong")
    assert not (tmp_path / "wrong" / "phase_3_selected_configuration.json").exists()


def lr_test_fixture(tmp_path, inputs):
    values = {name: 0.5 for name in selection_module.METRIC_NAMES[:3]}
    for metric in ("precision", "recall"):
        for capacity in (5, 10, 20):
            values[f"{metric}_at_{capacity}_percent"] = 0.5
    path = tmp_path / "lr_test.csv"
    pd.DataFrame([{"model": "Logistic Regression", "evaluated_split": "test",
        "row_count": len(inputs.targets["test"]), "positive_count": int(inputs.targets["test"].sum()), **values}]).to_csv(path, index=False)
    metadata = tmp_path / "lr_metadata.json"
    metadata.write_text(json.dumps({"selected_model_name": "Logistic Regression", "feature_names": inputs.feature_names,
        "phase_9_contract_fingerprint": inputs.phase_9_contract.fingerprint}))
    return path, metadata


def test_final_test_requires_persisted_freeze_and_scores_only_selected_once(experiment, tmp_path, monkeypatch):
    inputs, _, _, _ = experiment
    result = run_selection(experiment, tmp_path / "selection")
    lr_path, metadata = lr_test_fixture(tmp_path, inputs)
    original_score = GradientBoostedRiskModel.predict_score
    calls = []
    def score(model, matrix):
        assert result.decision_path.exists()
        assert (result.output_directory / "phase_3_test_access.json").exists()
        calls.append((model, matrix))
        return original_score(model, matrix)
    monkeypatch.setattr(GradientBoostedRiskModel, "predict_score", score)
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", lambda *args, **kwargs: pytest.fail("Final evaluation refitted a model"))
    payload = result.decision_path.read_text()
    result.decision_path.unlink()
    with pytest.raises(GradientBoostingTuningError, match="missing"):
        workflow.run_final_test_evaluation(result, inputs=inputs, lr_test_path=lr_path, lr_metadata_path=metadata)
    assert calls == []
    result.decision_path.write_text(payload)
    comparison = workflow.run_final_test_evaluation(result, inputs=inputs, lr_test_path=lr_path, lr_metadata_path=metadata)
    assert len(calls) == 1 and calls[0][0] is result.model and calls[0][1] is inputs.matrices["test"]
    assert result.decision_path.read_text() == payload
    assert comparison.metric.tolist() == list(selection_module.METRIC_NAMES)
    final = pd.read_csv(result.output_directory / "phase_3_final_test_results.csv")
    assert final.candidate_id.tolist() == [result.decision.candidate.candidate_id]
    with pytest.raises(FileExistsError):
        workflow.run_final_test_evaluation(result, inputs=inputs, lr_test_path=lr_path, lr_metadata_path=metadata)
    with pytest.raises(GradientBoostingTuningError, match="frozen"):
        run_selection(experiment, result.output_directory)
    changed = replace(result, decision=replace(result.decision, candidate=replace(result.decision.candidate, max_depth=9)))
    with pytest.raises(GradientBoostingTuningError, match="changed"):
        workflow.run_final_test_evaluation(changed, inputs=inputs, lr_test_path=lr_path, lr_metadata_path=metadata)
    assert len(calls) == 1
