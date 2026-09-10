"""Integration coverage for frozen preprocessing outputs entering baselines."""

import numpy as np
from scipy import sparse

from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.baseline_workflow import load_frozen_baseline_inputs
from urban_ops.models.evaluation import evaluate_binary_classifier
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


def test_frozen_inputs_to_logistic_validation_evaluation(tmp_path) -> None:
    """Run frozen input creation, sparse baseline fit, scoring, and evaluation."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_frozen_baseline_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    X_train = inputs.matrices["train"]
    X_validation = inputs.matrices["validation"]
    y_train = inputs.targets["train"]
    y_validation = inputs.targets["validation"]

    assert sparse.isspmatrix_csr(X_train)
    assert sparse.isspmatrix_csr(X_validation)
    assert inputs.feature_names == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )
    assert X_train.shape[1] == len(inputs.feature_names)

    model = LogisticRegressionBaseline(max_iter=200).fit(
        X_train,
        y_train,
        feature_names=inputs.feature_names,
    )
    scores = model.predict_score(X_validation)
    predictions = model.predict(X_validation)
    metrics = evaluate_binary_classifier(y_validation, predictions, scores)

    assert len(scores) == X_validation.shape[0] == len(y_validation)
    assert np.isfinite(scores).all()
    assert metrics.row_count == X_validation.shape[0]
    repeated_scores = model.predict_score(X_validation)
    np.testing.assert_allclose(scores, repeated_scores)
