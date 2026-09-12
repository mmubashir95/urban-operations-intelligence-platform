"""Integration coverage for frozen preprocessing outputs entering baselines."""

import numpy as np
from scipy import sparse

from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.baseline_workflow import (
    _fit_and_evaluate_validation,
    _format_confusion_matrices,
    load_frozen_baseline_inputs,
)
from urban_ops.models.evaluation import evaluate_basic_classifier
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
    metrics = evaluate_basic_classifier(y_validation, predictions)

    assert len(scores) == X_validation.shape[0] == len(y_validation)
    assert np.isfinite(scores).all()
    assert metrics.row_count == X_validation.shape[0]
    assert metrics.true_positive + metrics.false_negative == metrics.positive_count
    assert metrics.true_negative + metrics.false_positive == metrics.negative_count
    assert (
        metrics.true_positive
        + metrics.false_positive
        + metrics.false_negative
        + metrics.true_negative
        == metrics.row_count
    )
    repeated_scores = model.predict_score(X_validation)
    np.testing.assert_allclose(scores, repeated_scores)


def test_all_validation_baselines_share_phase_1_schema_and_readable_counts(
    tmp_path,
) -> None:
    """Existing baseline predictions produce one consistent Phase 1 report shape."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_frozen_baseline_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )

    results, _, _ = _fit_and_evaluate_validation(inputs)
    confusion_tables = _format_confusion_matrices(results)

    assert results["model"].tolist() == [
        "Majority Class",
        "Historical Rate",
        "Rule Based",
        "Logistic Regression",
    ]
    assert results["evaluated_split"].eq("validation").all()
    assert results["row_count"].eq(len(inputs.targets["validation"])).all()
    assert (
        results["true_positive"]
        + results["false_positive"]
        + results["false_negative"]
        + results["true_negative"]
    ).eq(results["row_count"]).all()
    assert "Predicted \\ Actual" in confusion_tables
    assert "FN =" in confusion_tables
