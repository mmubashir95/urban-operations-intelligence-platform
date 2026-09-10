"""Unit tests for Month 1 baseline model behavior and leakage boundaries."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from urban_ops.models.baselines import (
    HistoricalRateBaseline,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    RuleBasedHistoricalRateBaseline,
)


def test_majority_class_learns_from_train_only_and_predicts_deterministically() -> None:
    """Validation/test labels cannot affect the learned majority class."""
    X_train = sparse.csr_matrix(np.ones((5, 2), dtype=np.float64))
    y_train = pd.Series([0, 0, 1, 0, 1])
    baseline = MajorityClassBaseline().fit(X_train, y_train)

    assert baseline.majority_class_ == 0
    assert baseline.class_distribution_ == {0: 3, 1: 2}
    np.testing.assert_array_equal(baseline.predict(sparse.csr_matrix((3, 2))), [0, 0, 0])
    assert set(baseline.predict(sparse.csr_matrix((3, 2))).tolist()) <= {0, 1}

    changed_validation_labels = pd.Series([1, 1, 1, 1, 1])
    assert changed_validation_labels.mean() == 1.0
    repeated = MajorityClassBaseline().fit(X_train, y_train)
    assert repeated.majority_class_ == baseline.majority_class_
    np.testing.assert_allclose(
        repeated.predict_proba(sparse.csr_matrix((2, 2))),
        baseline.predict_proba(sparse.csr_matrix((2, 2))),
    )


def test_historical_rate_uses_train_aggregates_and_global_unseen_fallback() -> None:
    """Historical scores come from train groups with finite fallback scores."""
    train = pd.DataFrame({"created_month": [1, 1, 2, 2, 2]})
    y_train = pd.Series([0, 1, 1, 1, 0])
    baseline = HistoricalRateBaseline("created_month").fit(train, y_train)

    scores = baseline.predict_score(pd.DataFrame({"created_month": [1, 2, 3]}))
    np.testing.assert_allclose(scores, [0.5, 2 / 3, 3 / 5])
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()

    changed_validation_labels = pd.Series([0, 0, 0])
    assert changed_validation_labels.mean() == 0.0
    repeated = HistoricalRateBaseline("created_month").fit(train, y_train)
    assert repeated.group_rates_ == baseline.group_rates_
    assert repeated.global_rate_ == baseline.global_rate_


def test_rule_based_threshold_selection_is_validation_only_and_deterministic() -> None:
    """Rule threshold selection chooses the highest threshold on F1 ties."""
    train = pd.DataFrame({"created_month": [1, 1, 2, 2]})
    y_train = pd.Series([0, 0, 1, 1])
    historical = HistoricalRateBaseline("created_month").fit(train, y_train)
    rule = RuleBasedHistoricalRateBaseline(historical)

    validation = pd.DataFrame({"created_month": [1, 2, 3, 2]})
    y_validation = pd.Series([0, 1, 0, 1])
    selection = rule.select_threshold(validation, y_validation, candidates=[0.4, 0.5, 0.6])

    assert selection.selected_threshold == 0.6
    assert selection.selection_metric == "validation_f1"
    predictions = rule.predict(validation)
    assert set(predictions.tolist()) <= {0, 1}

    changed_test_labels = pd.Series([1, 1, 1, 1])
    assert changed_test_labels.sum() == 4
    repeated = RuleBasedHistoricalRateBaseline(historical)
    assert repeated.select_threshold(validation, y_validation, candidates=[0.4, 0.5, 0.6]) == selection


def test_logistic_regression_accepts_sparse_input_and_bounds_probabilities() -> None:
    """Logistic baseline fits CSR matrices and aligns coefficients to features."""
    X_train = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    model = LogisticRegressionBaseline(max_iter=200).fit(
        X_train,
        y_train,
        feature_names=("a", "b"),
    )

    probabilities = model.predict_proba(X_train)
    assert probabilities.shape == (6, 2)
    assert model.coefficient_count_ == 2
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6))


def test_logistic_regression_rejects_feature_mismatch() -> None:
    """Frozen feature ordering/width must match the fitted coefficient contract."""
    X_train = sparse.csr_matrix(np.ones((4, 2), dtype=np.float64))
    y_train = pd.Series([0, 0, 1, 1])

    with pytest.raises(ValueError, match="Feature count"):
        LogisticRegressionBaseline().fit(X_train, y_train, feature_names=("a",))
