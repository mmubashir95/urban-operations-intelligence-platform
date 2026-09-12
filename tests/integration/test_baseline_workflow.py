"""Integration coverage for frozen preprocessing outputs entering baselines."""

import numpy as np
from scipy import sparse

from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.baseline_workflow import (
    _fit_and_evaluate_validation,
    _format_confusion_matrices,
    _write_ranking_figures,
    build_ranking_curve_tables,
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

    results, _, _, ranking_evaluations = _fit_and_evaluate_validation(inputs)
    confusion_tables = _format_confusion_matrices(results)
    roc_table, pr_table = build_ranking_curve_tables(
        ranking_evaluations,
        split_name="validation",
    )

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
    assert set(roc_table["model"]) == set(results["model"])
    assert set(pr_table["model"]) == set(results["model"])
    assert roc_table["split"].eq("validation").all()
    assert pr_table["split"].eq("validation").all()
    assert pr_table["positive_rate"].eq(results["positive_rate"].iloc[0]).all()
    assert pr_table["pr_auc_definition"].eq("average_precision_score").all()
    for model_name, evaluation in ranking_evaluations.items():
        row = results.loc[results["model"].eq(model_name)].iloc[0]
        assert row["roc_auc"] == evaluation.metrics.roc_auc
        assert row["pr_auc"] == evaluation.metrics.pr_auc


def test_ranking_figures_use_validation_curves_and_prevalence_reference(
    tmp_path,
    monkeypatch,
) -> None:
    """Validation curve tables render deterministic ROC and PR image artifacts."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_frozen_baseline_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    results, _, _, ranking_evaluations = _fit_and_evaluate_validation(inputs)
    roc_table, pr_table = build_ranking_curve_tables(
        ranking_evaluations,
        split_name="validation",
    )
    phase_figures = tmp_path / "phase_figures"
    root_figures = tmp_path / "root_figures"
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.FIGURES_DIR",
        phase_figures,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.ROOT_FIGURES_DIR",
        root_figures,
    )

    _write_ranking_figures(
        validation_results=results,
        roc_curve_results=roc_table,
        pr_curve_results=pr_table,
    )

    for directory in (phase_figures, root_figures):
        assert (directory / "baseline_validation_roc_curve.png").stat().st_size > 0
        assert (directory / "baseline_validation_pr_curve.png").stat().st_size > 0
