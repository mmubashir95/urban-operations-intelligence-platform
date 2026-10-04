"""Behavioral tests for controlled probability calibration and selection."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from urban_ops.models.gradient_boosting_selection import METRIC_NAMES
from urban_ops.models.probability_calibration import (
    CALIBRATION_METHODS, CalibrationPolicy, CalibrationResult,
    FrozenCalibrationDecision, ProbabilityCalibrationError, ProbabilityCalibrator,
    calibration_table, evaluate_calibration_method, select_calibration,
    verify_persisted_calibration_decision,
)


def result(method, *, brier=0.2, pr=0.6, roc=0.7, recall=0.5, precision=0.5):
    values = dict.fromkeys(METRIC_NAMES, 0.5)
    values.update(brier_score=brier, pr_auc=pr, roc_auc=roc)
    for suffix in ("05", "10", "20"):
        values[f"recall_at_{suffix}"] = recall
        values[f"precision_at_{suffix}"] = precision
    return CalibrationResult(method, np.array([0.2, 0.8]), tuple(values.items()))


@pytest.fixture
def policy():
    return CalibrationPolicy(0.001, 0.0001, 0.01, CALIBRATION_METHODS)


def test_invalid_method_is_rejected():
    with pytest.raises(ProbabilityCalibrationError, match="Unsupported"):
        ProbabilityCalibrator("BETA")


def test_raw_preserves_probabilities_and_row_count():
    scores = np.array([0.1, 0.4, 0.9])
    output = ProbabilityCalibrator("RAW").fit(scores, [0, 1, 1]).predict(scores)
    np.testing.assert_array_equal(output, scores)
    assert output is not scores and len(output) == len(scores)


@pytest.mark.parametrize("method", ["SIGMOID", "ISOTONIC"])
def test_calibrated_probabilities_are_bounded_deterministic_and_aligned(method):
    scores = np.array([0.05, 0.2, 0.3, 0.65, 0.8, 0.95])
    target = np.array([0, 0, 1, 0, 1, 1])
    outputs = [ProbabilityCalibrator(method).fit(scores, target).predict(scores) for _ in range(2)]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert len(outputs[0]) == len(scores)
    assert np.isfinite(outputs[0]).all() and ((outputs[0] >= 0) & (outputs[0] <= 1)).all()


@pytest.mark.parametrize("method", ["SIGMOID", "ISOTONIC"])
def test_calibrated_predict_before_fit_fails(method):
    with pytest.raises(ProbabilityCalibrationError, match="fitted"):
        ProbabilityCalibrator(method).predict([0.2, 0.8])


def test_shared_calibration_bins_cover_samples_and_brier_is_finite():
    target = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.6, 0.8, 0.95])
    table = calibration_table("RAW", target, scores)
    evaluated = evaluate_calibration_method("RAW", scores, target)
    assert table.row_count.sum() == len(scores)
    assert np.isfinite(dict(evaluated.metrics)["brier_score"])


def test_selection_is_deterministic_and_raw_can_win(policy):
    rows = (result("RAW", brier=0.2), result("SIGMOID", brier=0.2), result("ISOTONIC", brier=0.21))
    assert select_calibration(rows, policy).method == "RAW"
    assert select_calibration(tuple(reversed(rows)), policy).method == "RAW"


def test_selection_prefers_simpler_method_when_brier_effectively_tied(policy):
    rows = (result("RAW", brier=0.25), result("SIGMOID", brier=0.19995), result("ISOTONIC", brier=0.1999))
    assert select_calibration(rows, policy).method == "SIGMOID"


def test_material_ranking_regression_guard_rejects_lower_brier(policy):
    rows = (result("RAW", brier=0.25, pr=0.6), result("SIGMOID", brier=0.2, pr=0.58), result("ISOTONIC", brier=0.24, pr=0.6))
    assert select_calibration(rows, policy).method == "ISOTONIC"


def test_frozen_decision_is_immutable_and_tamper_checked(tmp_path, policy):
    raw = result("RAW")
    decision = FrozenCalibrationDecision(
        "RAW", "shallow", "phase3.json", "timestamp", "split", "fingerprint",
        ("a",), 20260806, raw.metrics, raw.metrics, policy, "timestamp",
    )
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision.payload()))
    assert decision.frozen
    with pytest.raises(FrozenInstanceError):
        decision.frozen = False
    verify_persisted_calibration_decision(decision, path)
    payload = json.loads(path.read_text()); payload["frozen"] = False; path.write_text(json.dumps(payload))
    with pytest.raises(ProbabilityCalibrationError, match="changed"):
        verify_persisted_calibration_decision(decision, path)
