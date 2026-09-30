"""Unit tests for Phase 2.6 validation risk-score generation."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.gradient_boosting_training import (
    train_gradient_boosted_risk_model,
)
from urban_ops.models.gradient_boosting_validation import (
    GradientBoostingValidationScoreError,
    generate_gradient_boosting_validation_scores,
)
from tests.unit.models.test_gradient_boosting_training import _inputs


class LabelAccessForbidden:
    """Fail if score generation attempts to consult any target labels."""

    def __getitem__(self, key):
        raise AssertionError(f"Validation score generation accessed {key} labels.")


def test_fitted_model_produces_one_probability_per_sparse_validation_row() -> None:
    inputs = _inputs()
    training = train_gradient_boosted_risk_model(inputs)

    scores = generate_gradient_boosting_validation_scores(training.model, inputs)

    assert sparse.isspmatrix_csr(inputs.matrices["validation"])
    assert scores.shape == (inputs.matrices["validation"].shape[0],)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_helper_uses_predict_score_only_and_preserves_row_order() -> None:
    validation = sparse.csr_matrix(np.eye(3, dtype=np.float64))
    expected = np.asarray([0.73, 0.12, 0.51])
    calls = []

    class SpyModel:
        def predict_score(self, X):
            calls.append(("predict_score", X))
            return expected

        def predict(self, X):
            raise AssertionError("Hard predict must not be used.")

        def fit(self, *args, **kwargs):
            raise AssertionError("Score generation must not fit the model.")

    inputs = SimpleNamespace(
        matrices={"validation": validation},
        targets=LabelAccessForbidden(),
    )

    scores = generate_gradient_boosting_validation_scores(SpyModel(), inputs)

    assert calls == [("predict_score", validation)]
    assert scores is expected
    np.testing.assert_array_equal(scores, [0.73, 0.12, 0.51])


class TestProtectedMatrices(dict):
    """Mapping that fails if Phase 2.6 attempts to access the test split."""

    __test__ = False

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("Phase 2.6 attempted to access the test matrix.")
        return super().__getitem__(key)


def test_validation_labels_and_test_split_are_not_accessed() -> None:
    validation = sparse.csr_matrix((2, 3), dtype=np.float64)
    inputs = SimpleNamespace(
        matrices=TestProtectedMatrices(validation=validation),
        targets=LabelAccessForbidden(),
    )
    model = SimpleNamespace(
        predict_score=lambda X: np.asarray([0.25, 0.75]),
    )

    scores = generate_gradient_boosting_validation_scores(model, inputs)

    np.testing.assert_array_equal(scores, [0.25, 0.75])


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.asarray([[0.2, 0.8], [0.7, 0.3]]), "one-dimensional"),
        (np.asarray([0.2]), "score count"),
        (np.asarray([np.nan, 0.2]), "finite"),
        (np.asarray([np.inf, 0.2]), "finite"),
        (np.asarray([-0.1, 0.2]), "within"),
        (np.asarray([0.2, 1.1]), "within"),
    ],
)
def test_invalid_score_output_fails_clearly(output: np.ndarray, message: str) -> None:
    inputs = SimpleNamespace(
        matrices={"validation": sparse.csr_matrix((2, 3), dtype=np.float64)}
    )
    model = SimpleNamespace(predict_score=lambda X: output)

    with pytest.raises(GradientBoostingValidationScoreError, match=message):
        generate_gradient_boosting_validation_scores(model, inputs)


def test_unfitted_model_fails_through_wrapper_contract() -> None:
    inputs = SimpleNamespace(
        matrices={"validation": sparse.csr_matrix((2, 3), dtype=np.float64)}
    )

    with pytest.raises(ValueError, match="must be fitted"):
        generate_gradient_boosting_validation_scores(
            GradientBoostedRiskModel(), inputs
        )


def test_score_generation_does_not_mutate_fitted_model() -> None:
    inputs = _inputs()
    model = train_gradient_boosted_risk_model(inputs).model
    feature_names_before = model.feature_names_
    feature_count_before = model.feature_count_
    parameters_before = model._model.get_params().copy()
    booster_before = bytes(model._model.get_booster().save_raw())

    generate_gradient_boosting_validation_scores(model, inputs)

    assert model.feature_names_ == feature_names_before
    assert model.feature_count_ == feature_count_before
    assert model._model.get_params() == parameters_before
    assert bytes(model._model.get_booster().save_raw()) == booster_before
