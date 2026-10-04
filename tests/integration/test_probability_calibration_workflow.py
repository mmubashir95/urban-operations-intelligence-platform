"""Integration coverage for frozen Phase 4 selection and TEST protection."""
from dataclasses import replace

import pandas as pd
import pytest
import yaml

from urban_ops.models import probability_calibration_workflow as workflow
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.gradient_boosting_inputs import load_and_verify_gradient_boosting_inputs
from urban_ops.models.gradient_boosting_tuning import TUNING_CONFIG_PATH
from urban_ops.models.gradient_boosting_tuning_workflow import run_candidate_selection
from urban_ops.models.gradient_boosting_workflow import run_gradient_boosting_workflow
from urban_ops.models.probability_calibration import CALIBRATION_METHODS, ProbabilityCalibrationError, ProbabilityCalibrator
from tests.integration.test_gradient_boosting_tuning_workflow import ProtectedSplits
from tests.integration.test_gradient_boosting_workflow import _write_matching_frozen_evidence
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


@pytest.fixture
def phase4_experiment(tmp_path):
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(eda_config_path=fixture.config)
    evidence = _write_matching_frozen_evidence(tmp_path, fixture, inputs)
    phase2 = tmp_path / "phase2"
    run_gradient_boosting_workflow(eda_config_path=fixture.config, output_directory=phase2, frozen_evidence_path=evidence)
    config = yaml.safe_load(TUNING_CONFIG_PATH.read_text())
    config["frozen_input_reference"] = str(evidence)
    config_path = tmp_path / "tuning.yaml"; config_path.write_text(yaml.safe_dump(config))
    phase3_dir = tmp_path / "phase3"
    phase3 = run_candidate_selection(inputs=inputs, tuning_config_path=config_path, output_directory=phase3_dir,
                                     frozen_evidence_path=evidence, phase2_directory=phase2)
    return inputs, phase3.decision_path


def select(phase4_experiment, tmp_path, inputs=None):
    original, phase3_path = phase4_experiment
    return workflow.run_calibration_selection(
        inputs=inputs or original, phase3_decision_path=phase3_path,
        decision_path=tmp_path / "calibration_decision.json",
        output_directory=tmp_path / "reports", figure_directory=tmp_path / "figures",
    )


def test_selection_reuses_phase3_train_validation_and_never_test(phase4_experiment, tmp_path, monkeypatch):
    inputs, _ = phase4_experiment
    matrices, targets = ProtectedSplits(inputs.matrices), ProtectedSplits(inputs.targets)
    protected = replace(inputs, matrices=matrices, targets=targets)
    fit_calls, score_calls = [], []
    original_fit, original_score = GradientBoostedRiskModel.fit, GradientBoostedRiskModel.predict_score
    def fit(model, matrix, target, *, feature_names):
        fit_calls.append((matrix, target, feature_names)); return original_fit(model, matrix, target, feature_names=feature_names)
    def score(model, matrix):
        score_calls.append(matrix); return original_score(model, matrix)
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", fit)
    monkeypatch.setattr(GradientBoostedRiskModel, "predict_score", score)
    result = select(phase4_experiment, tmp_path, protected)
    assert len(fit_calls) == 1
    assert fit_calls[0][0] is inputs.matrices["train"] and fit_calls[0][1] is inputs.targets["train"] and fit_calls[0][2] == inputs.feature_names
    assert score_calls == [inputs.matrices["validation"]]
    assert "test" not in matrices.accessed + targets.accessed
    assert result.decision.base_model_identifier == "shallow"
    assert result.decision.split_id == inputs.split_id
    assert result.decision.feature_fingerprint == inputs.phase_9_contract.fingerprint
    assert result.decision.feature_names == inputs.feature_names
    assert result.decision.frozen and result.decision_path.is_file()
    assert {row.method for row in result.results} == set(CALIBRATION_METHODS)
    comparison = pd.read_csv(result.output_directory / "phase_4_calibration_comparison.csv")
    assert comparison.method.tolist() == list(CALIBRATION_METHODS)
    assert not tuple(result.output_directory.glob("phase_4_*test*"))


def test_selection_rejects_changed_phase3_lineage(phase4_experiment, tmp_path):
    inputs, _ = phase4_experiment
    with pytest.raises(ProbabilityCalibrationError, match="input contract"):
        select(phase4_experiment, tmp_path, replace(inputs, split_id="wrong"))


def test_final_test_requires_freeze_and_does_not_refit_or_switch(phase4_experiment, tmp_path, monkeypatch):
    inputs, _ = phase4_experiment
    selection = select(phase4_experiment, tmp_path)
    original_predict = ProbabilityCalibrator.predict
    methods = []
    def predict(calibrator, scores):
        methods.append(calibrator.method); return original_predict(calibrator, scores)
    monkeypatch.setattr(ProbabilityCalibrator, "predict", predict)
    monkeypatch.setattr(ProbabilityCalibrator, "fit", lambda *args, **kwargs: pytest.fail("TEST evaluation refitted calibration"))
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", lambda *args, **kwargs: pytest.fail("TEST evaluation refitted XGBoost"))
    payload = selection.decision_path.read_text(); selection.decision_path.unlink()
    with pytest.raises(ProbabilityCalibrationError, match="missing"):
        workflow.run_final_test_evaluation(selection, inputs=inputs)
    selection.decision_path.write_text(payload)
    comparison = workflow.run_final_test_evaluation(selection, inputs=inputs)
    assert methods == [selection.decision.selected_method]
    assert comparison.method.tolist() == ["RAW", selection.decision.selected_method]
    assert (selection.output_directory / "phase_4_test_access.json").is_file()
    with pytest.raises(FileExistsError):
        workflow.run_final_test_evaluation(selection, inputs=inputs)
