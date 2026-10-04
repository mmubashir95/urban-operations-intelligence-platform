"""Behavioral tests for native Tree SHAP contracts and summaries."""
from dataclasses import replace

import numpy as np
import pytest
from xgboost import DMatrix, train

from urban_ops.models.shap_explainability import (
    EXPECTED_FEATURES, ShapExplainabilityError, additivity_summary,
    correct_vs_incorrect, deterministic_sample, direction_summary,
    explain_tree_model, global_importance, interaction_summary,
    select_local_cases, stability_summary,
)


@pytest.fixture
def explanation():
    rng = np.random.default_rng(7)
    values = rng.integers(0, 7, size=(40, 4)).astype(float)
    target = ((values[:, 0] + values[:, 2]) > 6).astype(int)
    booster = train({"objective": "binary:logistic", "max_depth": 2, "eta": 0.2, "seed": 7}, DMatrix(values, label=target), num_boost_round=8)
    scores = booster.predict(DMatrix(values))
    result = explain_tree_model(booster, values, feature_names=EXPECTED_FEATURES, row_positions=tuple(range(40)),
                                identifiers=tuple(f"id-{i}" for i in range(40)), raw_scores=scores, additivity_tolerance=1e-5)
    return booster, target, result


def test_unsupported_model_empty_input_and_feature_mismatch_are_rejected(explanation):
    booster, _, result = explanation
    with pytest.raises(ShapExplainabilityError, match="Booster"):
        explain_tree_model(object(), result.feature_values, feature_names=EXPECTED_FEATURES, row_positions=result.row_positions,
                           identifiers=result.identifiers, raw_scores=result.raw_scores, additivity_tolerance=1e-5)
    with pytest.raises(ShapExplainabilityError, match="non-empty"):
        explain_tree_model(booster, np.empty((0, 4)), feature_names=EXPECTED_FEATURES, row_positions=(), identifiers=(), raw_scores=[], additivity_tolerance=1e-5)
    with pytest.raises(ShapExplainabilityError, match="ordering"):
        explain_tree_model(booster, result.feature_values, feature_names=tuple(reversed(EXPECTED_FEATURES)), row_positions=result.row_positions,
                           identifiers=result.identifiers, raw_scores=result.raw_scores, additivity_tolerance=1e-5)


def test_tree_shap_preserves_shapes_order_finiteness_and_output_space(explanation):
    _, _, result = explanation
    assert result.feature_names == EXPECTED_FEATURES
    assert result.shap_values.shape == result.feature_values.shape == (40, 4)
    assert result.base_values.shape == result.raw_margins.shape == (40,)
    assert np.isfinite(result.shap_values).all() and np.isfinite(result.base_values).all()
    assert result.output_space == "raw_margin"
    assert result.explainer_type == "xgboost_native_tree_shap"


def test_additivity_passes_and_detects_excess_residual(explanation):
    _, _, result = explanation
    evidence = additivity_summary(result, tolerance=1e-5)
    assert evidence["status"] == "PASS" and evidence["rows_checked"] == 40
    changed = replace(result, raw_margins=result.raw_margins + 0.1)
    assert additivity_summary(changed, tolerance=1e-5)["status"] == "FAIL"


def test_global_direction_and_interaction_summaries_are_deterministic(explanation):
    booster, _, result = explanation
    for builder in (global_importance, direction_summary):
        first, second = builder(result), builder(result)
        assert first.equals(second)
    importance = global_importance(result)
    assert importance["rank"].tolist() == [1, 2, 3, 4]
    interactions = interaction_summary(booster, result)
    assert len(interactions) == 6 and np.isfinite(interactions.mean_abs_interaction).all()


def test_local_selection_and_sampling_are_deterministic():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])
    targets = np.array([1, 0, 1, 1, 0, 0]); identifiers = np.array(list("abcdef"))
    expected = {"highest_risk": 0, "lowest_risk": 5, "true_positive": 0, "true_negative": 5, "false_positive": 1, "false_negative": 3}
    assert select_local_cases(scores, targets, identifiers, cutoff=0.5) == expected
    first = deterministic_sample(20, sample_size=10, seed=7, reserved=tuple(expected.values()))
    assert first == deterministic_sample(20, sample_size=10, seed=7, reserved=tuple(expected.values()))
    assert set(expected.values()) <= set(first)


def test_stability_and_correctness_comparison_are_deterministic(explanation):
    _, targets, result = explanation
    first_table, first_status = stability_summary(result, seed=7, half_size=20, minimum_spearman=0.0)
    second_table, second_status = stability_summary(result, seed=7, half_size=20, minimum_spearman=0.0)
    assert first_table.equals(second_table) and first_status == second_status
    comparison = correct_vs_incorrect(result, targets, cutoff=0.5)
    assert comparison.feature.tolist() == list(EXPECTED_FEATURES)
    assert np.isfinite(comparison.select_dtypes(include="number")).all().all()
