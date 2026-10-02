"""Validate bounded candidates, frozen boundaries, and wrapper reproducibility."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest
import yaml
from scipy.sparse import csr_matrix

from urban_ops.models.gradient_boosting import GradientBoostedRiskModel, load_gradient_boosting_config
from urban_ops.models.gradient_boosting_tuning import (
    GradientBoostingCandidate, GradientBoostingTuningError, TUNING_CONFIG_PATH,
    TUNABLE_PARAMETERS, load_tuning_contract,
)
from urban_ops.models.gradient_boosting_selection import (
    CandidateEvaluation, METRIC_NAMES, evaluate_scores, select_candidate,
    freeze_decision, verify_persisted_decision,
)


def candidate(**changes):
    return replace(GradientBoostingCandidate("example", "test", 10, 0.1, 3), **changes)


@pytest.mark.parametrize("name,value", [
    ("n_estimators", 0), ("n_estimators", True), ("n_estimators", 1.5),
    ("max_depth", 0), ("max_depth", -1), ("learning_rate", 0),
    ("learning_rate", 1.1), ("learning_rate", float("nan")),
    ("learning_rate", True), ("subsample", 0), ("subsample", 1.1),
    ("colsample_bytree", 0), ("colsample_bytree", 1.1),
    ("min_child_weight", -1), ("reg_alpha", -1), ("reg_lambda", -1),
    ("reg_lambda", float("inf")), ("candidate_id", ""), ("purpose", ""),
])
def test_candidate_rejects_invalid_values(name, value):
    with pytest.raises(GradientBoostingTuningError):
        candidate(**{name: value})


@pytest.mark.parametrize("mutation", ["duplicate", "unsupported", "reference", "seed", "split", "policy", "size"])
def test_contract_rejects_invalid_definitions(tmp_path, mutation):
    root = yaml.safe_load(TUNING_CONFIG_PATH.read_text())
    if mutation == "duplicate":
        root["candidates"][1]["candidate_id"] = root["candidates"][0]["candidate_id"]
    elif mutation == "unsupported":
        root["candidates"][0]["objective"] = "other"
    elif mutation == "reference":
        root["candidates"][0]["n_estimators"] += 1
    elif mutation == "seed":
        root["random_state"] += 1
    elif mutation == "split":
        root["selection_split"] = "test"
    elif mutation == "policy":
        root["selection_policy"]["pr_auc_tolerance"] = -1
    else:
        root["candidates"] = root["candidates"][:2]
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(root))
    with pytest.raises(GradientBoostingTuningError):
        load_tuning_contract(path)


def test_contract_retains_reference_and_deterministic_order():
    contract = load_tuning_contract()
    assert contract.allowed_tunable_parameters == TUNABLE_PARAMETERS
    ids = [row.candidate_id for row in contract.candidates]
    assert ids == sorted(ids)
    reference = next(row for row in contract.candidates if row.candidate_id == "phase2_baseline")
    config = load_gradient_boosting_config()
    assert (reference.n_estimators, reference.learning_rate, reference.max_depth) == (config.n_estimators, config.learning_rate, config.max_depth)


def test_candidates_preserve_sparse_input_and_seed_reproducibility():
    matrix = csr_matrix(np.arange(80, dtype=np.float64).reshape(20, 4))
    target = np.array([0, 1] * 10)
    names = ("a", "b", "c", "d")
    predictions = []
    for _ in range(2):
        model = GradientBoostedRiskModel(config=candidate().model_config(load_gradient_boosting_config()))
        model.fit(matrix, target, feature_names=names)
        assert model.feature_names_ == names
        scores = model.predict_score(matrix)
        assert scores.shape == (20,)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        predictions.append(scores)
    np.testing.assert_array_equal(*predictions)
    metrics = evaluate_scores(target, predictions[0])
    assert tuple(name for name, _ in metrics) == METRIC_NAMES
    assert all(np.isfinite(value) for _, value in metrics)


def evaluation(identifier, pr_auc=0.5, recall=0.2, brier=0.25, depth=3, trees=10):
    values = dict.fromkeys(METRIC_NAMES, 0.5)
    values.update(pr_auc=pr_auc, recall_at_10=recall, brier_score=brier)
    return CandidateEvaluation(candidate(candidate_id=identifier, max_depth=depth, n_estimators=trees), tuple(values.items()))


@pytest.mark.parametrize("winner,loser,tolerance", [
    (evaluation("high", pr_auc=0.7), evaluation("low", pr_auc=0.6, recall=0.9), 1e-6),
    (evaluation("recall", recall=0.8), evaluation("other", pr_auc=0.5000005), 1e-6),
    (evaluation("brier", brier=0.1), evaluation("other"), 1e-6),
    (evaluation("shallow", depth=2), evaluation("other"), 1e-6),
    (evaluation("fewer", trees=5), evaluation("other"), 1e-6),
    (evaluation("a"), evaluation("z"), 1e-6),
])
def test_selection_policy_and_order_independence(winner, loser, tolerance):
    assert select_candidate((winner, loser), tolerance=tolerance) == winner
    assert select_candidate((loser, winner), tolerance=tolerance) == winner


def test_tolerance_uses_global_maximum_not_pairwise_chain():
    rows = (evaluation("a", pr_auc=0.5, recall=0.9), evaluation("b", pr_auc=0.5000009, recall=0.8), evaluation("c", pr_auc=0.5000018, recall=0.7))
    assert select_candidate(rows, tolerance=1e-6).candidate.candidate_id == "b"


def test_freeze_is_immutable_write_once_and_tamper_checked(tmp_path):
    contract = load_tuning_contract()
    row = CandidateEvaluation(contract.candidates[0], evaluation("x").metrics)
    path = tmp_path / "decision.json"
    decision = freeze_decision(row, contract=contract, feature_names=("a",), path=path)
    assert decision.frozen
    with pytest.raises(FrozenInstanceError):
        decision.frozen = False
    with pytest.raises(FrozenInstanceError):
        decision.candidate.max_depth = 99
    verify_persisted_decision(decision, path)
    with pytest.raises(FileExistsError):
        freeze_decision(row, contract=contract, feature_names=("a",), path=path)
    payload = json.loads(path.read_text())
    payload["candidate"]["max_depth"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(GradientBoostingTuningError, match="changed"):
        verify_persisted_decision(decision, path)
