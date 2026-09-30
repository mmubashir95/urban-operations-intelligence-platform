"""Unit tests for the project-facing XGBoost risk-model wrapper."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from xgboost import XGBClassifier

from urban_ops.models.gradient_boosting import (
    GradientBoostedRiskModel,
    GradientBoostingModelError,
)


FEATURE_NAMES = ("first_signal", "second_signal", "third_signal")


def _training_data() -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix = sparse.csr_matrix(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    return matrix, np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)


def _fitted_model() -> tuple[GradientBoostedRiskModel, sparse.csr_matrix]:
    matrix, target = _training_data()
    model = GradientBoostedRiskModel().fit(
        matrix,
        target,
        feature_names=FEATURE_NAMES,
    )
    return model, matrix


def test_model_constructs_with_encapsulated_xgb_classifier() -> None:
    model = GradientBoostedRiskModel()

    assert isinstance(model._model, XGBClassifier)


def test_valid_sparse_binary_fit_preserves_exact_feature_order() -> None:
    matrix, target = _training_data()
    names = ("third", "first", "second")

    model = GradientBoostedRiskModel().fit(matrix, target, feature_names=names)

    assert model.feature_names_ == names
    assert model.feature_count_ == matrix.shape[1]
    assert sparse.isspmatrix_csr(matrix)


def test_fit_rejects_row_count_mismatch() -> None:
    matrix, target = _training_data()

    with pytest.raises(GradientBoostingModelError, match="row count"):
        GradientBoostedRiskModel().fit(
            matrix,
            target[:-1],
            feature_names=FEATURE_NAMES,
        )


def test_fit_rejects_feature_count_mismatch() -> None:
    matrix, target = _training_data()

    with pytest.raises(GradientBoostingModelError, match="feature count"):
        GradientBoostedRiskModel().fit(matrix, target, feature_names=("one", "two"))


@pytest.mark.parametrize(
    "feature_names",
    [(), ("valid", None, "third"), ("valid", " ", "third")],
)
def test_fit_rejects_empty_null_or_blank_feature_names(feature_names) -> None:
    matrix, target = _training_data()

    with pytest.raises(GradientBoostingModelError, match="feature_names"):
        GradientBoostedRiskModel().fit(
            matrix,
            target,
            feature_names=feature_names,
        )


def test_fit_rejects_duplicate_feature_names() -> None:
    matrix, target = _training_data()

    with pytest.raises(GradientBoostingModelError, match="unique"):
        GradientBoostedRiskModel().fit(
            matrix,
            target,
            feature_names=("duplicate", "duplicate", "third"),
        )


def test_fit_rejects_invalid_target_classes() -> None:
    matrix, _ = _training_data()

    with pytest.raises(GradientBoostingModelError, match="only 0/1"):
        GradientBoostedRiskModel().fit(
            matrix,
            [0, 0, 1, 1, 0, 1, 2, 1],
            feature_names=FEATURE_NAMES,
        )


def test_fit_rejects_single_class_target() -> None:
    matrix, _ = _training_data()

    with pytest.raises(GradientBoostingModelError, match="both target classes"):
        GradientBoostedRiskModel().fit(
            matrix,
            np.zeros(matrix.shape[0], dtype=int),
            feature_names=FEATURE_NAMES,
        )


@pytest.mark.parametrize("method_name", ["predict_score", "predict_proba", "predict"])
def test_prediction_before_fit_fails(method_name: str) -> None:
    matrix, _ = _training_data()
    method = getattr(GradientBoostedRiskModel(), method_name)

    with pytest.raises(GradientBoostingModelError, match="must be fitted"):
        method(matrix)


def test_sparse_predict_score_is_finite_bounded_and_row_aligned() -> None:
    model, matrix = _fitted_model()

    scores = model.predict_score(matrix)

    assert sparse.isspmatrix_csr(matrix)
    assert scores.shape == (matrix.shape[0],)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_predict_proba_is_finite_bounded_binary_output() -> None:
    model, matrix = _fitted_model()

    probabilities = model.predict_proba(matrix)

    assert probabilities.shape == (matrix.shape[0], 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_predict_uses_scores_and_inclusive_caller_threshold() -> None:
    model, matrix = _fitted_model()
    scores = model.predict_score(matrix)

    predictions = model.predict(matrix, threshold=0.5)

    np.testing.assert_array_equal(predictions, (scores >= 0.5).astype(int))
    assert set(predictions.tolist()).issubset({0, 1})


def test_threshold_zero_and_one_are_valid_boundaries() -> None:
    model, matrix = _fitted_model()
    scores = model.predict_score(matrix)

    np.testing.assert_array_equal(model.predict(matrix, threshold=0.0), np.ones(len(scores)))
    np.testing.assert_array_equal(
        model.predict(matrix, threshold=1.0),
        (scores >= 1.0).astype(int),
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, -np.inf, True])
def test_invalid_threshold_fails(threshold: object) -> None:
    model, matrix = _fitted_model()

    with pytest.raises(GradientBoostingModelError, match="threshold"):
        model.predict(matrix, threshold=threshold)


def test_prediction_feature_count_mismatch_fails() -> None:
    model, matrix = _fitted_model()
    wrong_width = sparse.csr_matrix((matrix.shape[0], matrix.shape[1] + 1))

    with pytest.raises(GradientBoostingModelError, match="Prediction feature count"):
        model.predict_score(wrong_width)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((8, 1)), "shape"),
        (np.full((8, 2), np.nan), "finite"),
        (np.full((8, 2), 1.1), "within"),
    ],
)
def test_invalid_probability_output_fails(output: np.ndarray, message: str) -> None:
    model, matrix = _fitted_model()
    model._model.predict_proba = lambda X: output

    with pytest.raises(GradientBoostingModelError, match=message):
        model.predict_proba(matrix)


def test_same_deterministic_configuration_reproduces_scores() -> None:
    matrix, target = _training_data()
    first = GradientBoostedRiskModel().fit(
        matrix, target, feature_names=FEATURE_NAMES
    )
    second = GradientBoostedRiskModel().fit(
        matrix, pd.Series(target), feature_names=FEATURE_NAMES
    )

    np.testing.assert_array_equal(
        first.predict_score(matrix),
        second.predict_score(matrix),
    )
