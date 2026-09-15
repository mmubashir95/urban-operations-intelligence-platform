"""Integration coverage for frozen preprocessing outputs entering baselines."""

import numpy as np
from scipy import sparse

from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.baseline_workflow import (
    _fit_and_evaluate_validation,
    _format_confusion_matrices,
    _write_phase_4_1_outputs,
    _write_ranking_figures,
    _write_validation_calibration_figure,
    build_logistic_validation_threshold_table,
    build_ranking_curve_tables,
    build_validation_calibration_table,
    load_frozen_baseline_inputs,
)
from urban_ops.models.evaluation import evaluate_basic_classifier, evaluate_threshold
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
    threshold_metrics = evaluate_threshold(y_validation, scores, threshold=0.5)

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
    np.testing.assert_array_equal(
        predictions,
        (scores >= threshold_metrics.threshold).astype(int),
    )
    assert threshold_metrics.sample_count == len(y_validation)


def test_all_validation_baselines_share_phase_1_schema_and_readable_counts(
    tmp_path,
) -> None:
    """Existing baseline predictions produce one consistent Phase 1 report shape."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_frozen_baseline_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )

    (
        results,
        _,
        _,
        ranking_evaluations,
        calibration_evaluations,
        threshold_metrics,
    ) = _fit_and_evaluate_validation(inputs)
    confusion_tables = _format_confusion_matrices(results)
    roc_table, pr_table = build_ranking_curve_tables(
        ranking_evaluations,
        split_name="validation",
    )
    calibration_table = build_validation_calibration_table(
        calibration_evaluations
    )
    threshold_table = build_logistic_validation_threshold_table(threshold_metrics)

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
    assert set(calibration_table["model"]) == set(results["model"])
    assert calibration_table["split"].eq("validation").all()
    assert calibration_table["requested_bin_count"].eq(10).all()
    assert calibration_table["strategy"].eq("uniform").all()
    for model_name, evaluation in calibration_evaluations.items():
        row = results.loc[results["model"].eq(model_name)].iloc[0]
        assert row["brier_score"] == evaluation.metrics.brier_score
    logistic_row = results.loc[
        results["model"].eq("Logistic Regression")
    ].iloc[0]
    threshold_row = threshold_table.iloc[0]
    assert threshold_row["model"] == "Logistic Regression"
    assert threshold_row["evaluated_split"] == "validation"
    assert threshold_row["threshold"] == 0.5
    assert threshold_row["sample_count"] == len(inputs.targets["validation"])
    assert threshold_row["precision"] == logistic_row["precision"]
    assert threshold_row["recall"] == logistic_row["recall"]
    assert threshold_row["f1"] == logistic_row["f1"]


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
    results, _, _, ranking_evaluations, _, _ = _fit_and_evaluate_validation(inputs)
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


def test_validation_calibration_figure_uses_all_baseline_probability_points(
    tmp_path,
    monkeypatch,
) -> None:
    """All validation baselines render against the ideal calibration diagonal."""
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_frozen_baseline_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    results, _, _, _, evaluations, _ = _fit_and_evaluate_validation(inputs)
    calibration_table = build_validation_calibration_table(evaluations)
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

    _write_validation_calibration_figure(calibration_table)

    assert set(calibration_table["model"]) == set(results["model"])
    for directory in (phase_figures, root_figures):
        artifact = directory / "baseline_validation_calibration_curve.png"
        assert artifact.stat().st_size > 0


def test_phase_4_1_outputs_use_full_precision_csv_and_four_decimal_markdown(
    tmp_path,
    monkeypatch,
) -> None:
    """The validation operating point follows established report conventions."""
    metrics = evaluate_threshold(
        [0, 1, 1, 0],
        [0.18, 0.42, 0.63, 0.91],
        threshold=0.5,
    )
    results = build_logistic_validation_threshold_table(metrics)
    phase_report_dir = tmp_path / "phase"
    phase_tables_dir = phase_report_dir / "tables"
    root_tables_dir = tmp_path / "root_tables"
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.BASELINE_REPORT_DIR",
        phase_report_dir,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.TABLES_DIR",
        phase_tables_dir,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.ROOT_TABLES_DIR",
        root_tables_dir,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.PROJECT_ROOT",
        project_root,
    )

    _write_phase_4_1_outputs(results)

    filename = "logistic_regression_validation_threshold_050.csv"
    phase_csv = (phase_tables_dir / filename).read_text(encoding="utf-8")
    report = (phase_report_dir / "phase_4_1_default_threshold.md").read_text(
        encoding="utf-8"
    )
    normalized_report = " ".join(report.split())
    assert phase_csv == (root_tables_dir / filename).read_text(encoding="utf-8")
    assert "0.5" in phase_csv
    assert (
        "| Logistic Regression | validation | 0.5000 | 0.5000 | 0.5000 | 0.5000 |"
        in report
    )
    assert "2 of 4 validation complaints (50.0%) are flagged" in normalized_report
    assert "does not select or optimize a threshold" in report
    assert report == (
        project_root / "reports/phase_4_1_default_threshold.md"
    ).read_text(encoding="utf-8")
