"""Compatibility tests for the frozen Phase 2.2 XGBoost choice."""

from pathlib import Path

import numpy as np
from scipy import sparse
import yaml
from xgboost import XGBClassifier

from urban_ops.models.baselines import DEFAULT_RANDOM_STATE


SELECTION_CONFIG_PATH = Path(
    "configs/models/resolution_risk_gradient_boosting.yaml"
)
SMOKE_PARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 8,
    "max_depth": 2,
    "learning_rate": 0.2,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 1,
    "tree_method": "hist",
}


def _synthetic_binary_data() -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return tiny synthetic CSR data containing both binary target classes."""
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
    target = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int64)
    return matrix, target


def _fit_smoke_model() -> tuple[XGBClassifier, sparse.csr_matrix]:
    """Fit the selected estimator only on the tiny synthetic smoke fixture."""
    matrix, target = _synthetic_binary_data()
    model = XGBClassifier(**SMOKE_PARAMETERS)
    model.fit(matrix, target)
    return model, matrix


def test_xgb_classifier_imports_and_instantiates() -> None:
    model = XGBClassifier(random_state=DEFAULT_RANDOM_STATE, n_jobs=1)

    assert isinstance(model, XGBClassifier)


def test_selection_config_freezes_xgboost_classifier() -> None:
    selection = yaml.safe_load(SELECTION_CONFIG_PATH.read_text(encoding="utf-8"))

    assert selection["selection_version"] == 1
    assert selection["implementation"] == "xgboost"
    assert selection["estimator"] == "XGBClassifier"
    assert isinstance(selection["starting_configuration"], dict)


def test_xgb_classifier_fits_tiny_sparse_csr_matrix() -> None:
    model, matrix = _fit_smoke_model()

    assert sparse.isspmatrix_csr(matrix)
    assert model.classes_.tolist() == [0, 1]


def test_predict_proba_returns_finite_binary_probabilities_for_sparse_input() -> None:
    model, matrix = _fit_smoke_model()

    probabilities = model.predict_proba(matrix)
    positive_scores = probabilities[:, 1]

    assert probabilities.shape == (matrix.shape[0], 2)
    assert positive_scores.shape == (matrix.shape[0],)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_deterministic_smoke_configuration_reproduces_scores() -> None:
    first, matrix = _fit_smoke_model()
    second, _ = _fit_smoke_model()

    np.testing.assert_array_equal(
        first.predict_proba(matrix),
        second.predict_proba(matrix),
    )
