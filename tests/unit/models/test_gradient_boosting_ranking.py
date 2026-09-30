"""Unit tests for shared ranking evaluation of XGBoost validation scores."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from urban_ops.models import gradient_boosting_ranking
from urban_ops.models.evaluation import RankingEvaluation, evaluate_ranking
from urban_ops.models.gradient_boosting_ranking import (
    evaluate_gradient_boosting_validation_ranking,
)


class TestLabelsProtected(dict):
    """Allow validation labels while failing on test-label access."""

    __test__ = False

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Phase 2.7 attempted to access test labels.")
        return super().__getitem__(key)


def test_existing_evaluate_ranking_receives_unchanged_validation_inputs(
    monkeypatch,
) -> None:
    y_validation = pd.Series([0, 1, 0, 1])
    validation_scores = np.asarray([0.08, 0.74, 0.21, 0.91])
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=y_validation),
    )
    observed = {}
    expected = evaluate_ranking(y_validation, validation_scores)

    def spy_evaluate_ranking(y_true, y_score):
        observed["y_true"] = y_true
        observed["y_score"] = y_score
        return expected

    monkeypatch.setattr(
        gradient_boosting_ranking,
        "evaluate_ranking",
        spy_evaluate_ranking,
    )

    result = evaluate_gradient_boosting_validation_ranking(
        inputs,
        validation_scores,
    )

    assert result is expected
    assert observed["y_true"] is y_validation
    assert observed["y_score"] is validation_scores


def test_ranking_returns_existing_result_with_pr_auc_and_roc_auc() -> None:
    inputs = SimpleNamespace(targets={"validation": pd.Series([0, 1, 0, 1])})
    scores = np.asarray([0.1, 0.8, 0.2, 0.9])

    result = evaluate_gradient_boosting_validation_ranking(inputs, scores)

    assert isinstance(result, RankingEvaluation)
    assert result.metrics.pr_auc == pytest.approx(1.0)
    assert result.metrics.roc_auc == pytest.approx(1.0)


def test_continuous_reversed_ranking_is_worse_than_perfect_ranking() -> None:
    inputs = SimpleNamespace(targets={"validation": pd.Series([0, 1, 0, 1])})

    perfect = evaluate_gradient_boosting_validation_ranking(
        inputs, np.asarray([0.1, 0.8, 0.2, 0.9])
    )
    reversed_ranking = evaluate_gradient_boosting_validation_ranking(
        inputs, np.asarray([0.9, 0.2, 0.8, 0.1])
    )

    assert perfect.metrics.pr_auc > reversed_ranking.metrics.pr_auc
    assert perfect.metrics.roc_auc > reversed_ranking.metrics.roc_auc


def test_month_2_helper_matches_shared_month_1_evaluator_exactly() -> None:
    y_validation = pd.Series([1, 0, 1, 0, 1, 0])
    scores = np.asarray([0.72, 0.61, 0.55, 0.33, 0.27, 0.12])
    inputs = SimpleNamespace(targets={"validation": y_validation})

    result = evaluate_gradient_boosting_validation_ranking(inputs, scores)

    assert result == evaluate_ranking(y_validation, scores)


def test_row_count_mismatch_fails_through_shared_evaluator() -> None:
    inputs = SimpleNamespace(targets={"validation": pd.Series([0, 1, 0, 1])})

    with pytest.raises(ValueError, match="lengths must match"):
        evaluate_gradient_boosting_validation_ranking(
            inputs,
            np.asarray([0.1, 0.8, 0.2]),
        )


def test_test_labels_are_not_accessed() -> None:
    inputs = SimpleNamespace(
        targets=TestLabelsProtected(validation=pd.Series([0, 1, 0, 1]))
    )

    result = evaluate_gradient_boosting_validation_ranking(
        inputs,
        np.asarray([0.1, 0.8, 0.2, 0.9]),
    )

    assert result.metrics.row_count == 4
