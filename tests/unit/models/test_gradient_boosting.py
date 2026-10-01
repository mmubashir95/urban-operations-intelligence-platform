"""Unit tests for the project-facing XGBoost risk-model wrapper."""

from inspect import signature

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import yaml
from xgboost import XGBClassifier

from urban_ops.models.gradient_boosting import (
    GRADIENT_BOOSTING_CONFIG_PATH,
    GradientBoostedRiskModel,
    GradientBoostingModelError,
    load_gradient_boosting_config,
)
from urban_ops.models.baselines import DEFAULT_RANDOM_STATE


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


def test_authoritative_phase_2_4_configuration_loads() -> None:
    config = load_gradient_boosting_config()

    assert GRADIENT_BOOSTING_CONFIG_PATH.is_file()
    assert config.implementation == "xgboost"
    assert config.estimator == "XGBClassifier"
    assert config.model_parameters == {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": DEFAULT_RANDOM_STATE,
        "n_jobs": 1,
    }


def test_default_model_uses_approved_phase_2_4_configuration() -> None:
    model = GradientBoostedRiskModel()
    estimator_parameters = model._model.get_params()

    assert model.config == load_gradient_boosting_config()
    for name, expected in model.config.model_parameters.items():
        assert estimator_parameters[name] == expected


def test_wrapper_public_method_signatures_remain_stable() -> None:
    assert tuple(signature(GradientBoostedRiskModel.fit).parameters) == (
        "self",
        "X_train",
        "y_train",
        "feature_names",
    )
    assert tuple(signature(GradientBoostedRiskModel.predict_score).parameters) == (
        "self",
        "X",
    )
    assert tuple(signature(GradientBoostedRiskModel.predict_proba).parameters) == (
        "self",
        "X",
    )
    assert tuple(signature(GradientBoostedRiskModel.predict).parameters) == (
        "self",
        "X",
        "threshold",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unexpected": True}), "fields"),
        (
            lambda payload: payload["starting_configuration"].update(
                {"objective": "multi:softprob"}
            ),
            "objective",
        ),
        (
            lambda payload: payload["starting_configuration"].update(
                {"eval_metric": "auc"}
            ),
            "eval_metric",
        ),
        (
            lambda payload: payload["starting_configuration"].update(
                {"n_estimators": 0}
            ),
            "n_estimators",
        ),
        (
            lambda payload: payload["starting_configuration"].update(
                {"learning_rate": 0.0}
            ),
            "learning_rate",
        ),
        (
            lambda payload: payload["starting_configuration"].update(
                {"max_depth": -1}
            ),
            "max_depth",
        ),
        (
            lambda payload: payload["starting_configuration"].update(
                {"random_state": 42}
            ),
            "project seed",
        ),
        (
            lambda payload: payload["starting_configuration"].update({"n_jobs": 0}),
            "n_jobs",
        ),
    ],
)
def test_invalid_project_configuration_fails_clearly(
    tmp_path, mutation, message: str
) -> None:
    payload = yaml.safe_load(
        GRADIENT_BOOSTING_CONFIG_PATH.read_text(encoding="utf-8")
    )
    mutation(payload)
    path = tmp_path / "invalid-gradient-boosting.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(GradientBoostingModelError, match=message):
        load_gradient_boosting_config(path)


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


@pytest.mark.parametrize("invalid_label", [-1, 2, "missed"])
def test_fit_rejects_invalid_target_classes(invalid_label: object) -> None:
    matrix, _ = _training_data()
    target = np.asarray([0, 0, 1, 1, 0, 1, invalid_label, 1], dtype=object)

    with pytest.raises(GradientBoostingModelError, match="only 0/1"):
        GradientBoostedRiskModel().fit(
            matrix,
            target,
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

    np.testing.assert_allclose(
        first.predict_score(matrix),
        second.predict_score(matrix),
        rtol=1e-12,
        atol=1e-12,
    )
