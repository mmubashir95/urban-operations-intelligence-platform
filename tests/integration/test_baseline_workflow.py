"""Integration coverage for frozen preprocessing outputs entering baselines."""

import numpy as np
import pandas as pd
from scipy import sparse

from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.baseline_workflow import (
    _fit_and_evaluate_validation,
    _format_confusion_matrices,
    _format_phase_4_3_focus_table,
    _format_phase_4_3_transition_summary,
    _format_phase_4_4_sensitivity_summary,
    _phase_4_3_focus_region,
    _threshold_sweep_plot_data,
    _write_phase_4_4_outputs,
    _write_phase_4_1_outputs,
    _write_phase_4_2_outputs,
    _write_phase_4_3_outputs,
    _write_threshold_tradeoff_figures,
    _write_phase_4_5_outputs,
    _write_phase_4_6_outputs,
    _write_ranking_figures,
    _write_validation_calibration_figure,
    build_logistic_validation_policy_table,
    build_logistic_validation_threshold_table,
    build_logistic_validation_manual_threshold_table,
    build_logistic_validation_sweep_table,
    build_ranking_curve_tables,
    build_validation_calibration_table,
    load_frozen_threshold_decision,
    load_frozen_baseline_inputs,
    write_frozen_threshold_decision,
)
from urban_ops.models.evaluation import (
    MANUAL_CLASSIFICATION_THRESHOLDS,
    SWEEP_CLASSIFICATION_THRESHOLDS,
    evaluate_basic_classifier,
    evaluate_manual_thresholds,
    evaluate_threshold,
    evaluate_threshold_selection_policies,
    evaluate_threshold_sweep,
    freeze_workload_limited_threshold,
)
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
    manual_threshold_metrics = evaluate_manual_thresholds(y_validation, scores)
    sweep_metrics = evaluate_threshold_sweep(y_validation, scores)

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
    assert manual_threshold_metrics[2] == threshold_metrics
    assert manual_threshold_metrics == tuple(
        evaluate_threshold(y_validation, scores, threshold=threshold)
        for threshold in MANUAL_CLASSIFICATION_THRESHOLDS
    )
    assert len(sweep_metrics) == 91
    assert sweep_metrics[0].threshold == 0.05
    assert sweep_metrics[-1].threshold == 0.95
    assert tuple(result.threshold for result in sweep_metrics) == (
        SWEEP_CLASSIFICATION_THRESHOLDS
    )
    sweep_at_050 = next(
        result for result in sweep_metrics if result.threshold == 0.5
    )
    assert sweep_at_050 == threshold_metrics


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
        manual_threshold_metrics,
        sweep_metrics,
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
    manual_threshold_table = build_logistic_validation_manual_threshold_table(
        manual_threshold_metrics
    )
    sweep_table = build_logistic_validation_sweep_table(sweep_metrics)
    policy_table = build_logistic_validation_policy_table(
        evaluate_threshold_selection_policies(sweep_table)
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
    assert manual_threshold_table["threshold"].tolist() == list(
        MANUAL_CLASSIFICATION_THRESHOLDS
    )
    assert manual_threshold_table["evaluated_split"].eq("validation").all()
    assert len(manual_threshold_table) == 5
    for column in (
        "predicted_positive_count",
        "predicted_positive_rate",
        "true_positives",
        "false_positives",
        "recall",
    ):
        assert manual_threshold_table[column].is_monotonic_decreasing
    for column in ("true_negatives", "false_negatives"):
        assert manual_threshold_table[column].is_monotonic_increasing
    manual_reference = manual_threshold_table.iloc[2]
    for field in threshold_metrics.to_dict():
        assert manual_reference[field] == threshold_row[field]
    assert len(sweep_table) == 91
    assert sweep_table["threshold"].tolist() == list(SWEEP_CLASSIFICATION_THRESHOLDS)
    assert sweep_table["evaluated_split"].eq("validation").all()
    assert sweep_table["sample_count"].eq(len(inputs.targets["validation"])).all()
    for column in (
        "predicted_positive_count",
        "predicted_positive_rate",
        "true_positives",
        "false_positives",
        "recall",
    ):
        assert sweep_table[column].is_monotonic_decreasing
    for column in ("true_negatives", "false_negatives"):
        assert sweep_table[column].is_monotonic_increasing
    for checkpoint in MANUAL_CLASSIFICATION_THRESHOLDS:
        sweep_row = sweep_table.loc[sweep_table["threshold"].eq(checkpoint)].iloc[0]
        manual_row = manual_threshold_table.loc[
            manual_threshold_table["threshold"].eq(checkpoint)
        ].iloc[0]
        for field in threshold_metrics.to_dict():
            assert sweep_row[field] == manual_row[field]
    assert policy_table["policy_name"].tolist() == [
        "Max F1",
        "Recall >= 0.70",
        "Precision >= 0.50",
        "Flagged rate <= 0.30",
    ]
    assert policy_table["evaluated_split"].eq("validation").all()
    assert policy_table["constraint_satisfied"].all()
    for _, row in policy_table.iterrows():
        candidate = sweep_table.loc[
            sweep_table["threshold"].eq(row["candidate_threshold"])
        ]
        assert len(candidate) == 1
        for column in (
            "precision",
            "recall",
            "f1",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "predicted_positive_count",
            "predicted_positive_rate",
        ):
            assert row[column] == candidate.iloc[0][column]
    decision = freeze_workload_limited_threshold(policy_table)
    workload_row = policy_table.loc[
        policy_table["policy_name"].eq("Flagged rate <= 0.30")
    ].iloc[0]
    assert decision.selected_threshold == workload_row["candidate_threshold"]
    assert decision.selected_threshold in set(sweep_table["threshold"])
    assert decision.selected_on_split == "validation"
    assert decision.frozen is True
    assert decision.predicted_positive_rate <= 0.30
    assert decision.precision == workload_row["precision"]
    assert decision.recall == workload_row["recall"]
    assert decision.f1 == workload_row["f1"]


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
    results, _, _, ranking_evaluations, _, _, _, _ = _fit_and_evaluate_validation(
        inputs
    )
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
    results, _, _, _, evaluations, _, _, _ = _fit_and_evaluate_validation(inputs)
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


def test_phase_4_2_outputs_preserve_manual_order_and_report_comparisons(
    tmp_path,
    monkeypatch,
) -> None:
    """The five validation rows are persisted in manual, not metric, order."""
    metrics = evaluate_manual_thresholds(
        [1, 0, 1, 0, 1],
        [0.80, 0.65, 0.45, 0.35, 0.25],
    )
    results = build_logistic_validation_manual_threshold_table(metrics)
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

    _write_phase_4_2_outputs(results)

    filename = "logistic_regression_validation_manual_thresholds.csv"
    persisted = np.genfromtxt(
        phase_tables_dir / filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    report = (
        phase_report_dir / "phase_4_2_manual_threshold_comparison.md"
    ).read_text(encoding="utf-8")
    normalized_report = " ".join(report.split())
    assert persisted["threshold"].tolist() == list(MANUAL_CLASSIFICATION_THRESHOLDS)
    assert (persisted["evaluated_split"] == "validation").all()
    assert (root_tables_dir / filename).read_bytes() == (
        phase_tables_dir / filename
    ).read_bytes()
    for threshold in MANUAL_CLASSIFICATION_THRESHOLDS:
        assert f"{threshold:.4f}" in report
    assert "At 0.30 versus 0.50" in report
    assert "At 0.70 versus 0.50" in report
    assert "not assumed to be monotonic" in report
    assert (
        "No threshold is ranked, optimized, recommended, or selected"
        in normalized_report
    )
    assert report == (
        project_root / "reports/phase_4_2_manual_threshold_comparison.md"
    ).read_text(encoding="utf-8")


def test_phase_4_3_outputs_produce_deterministic_91_row_sweep_and_focus_region(
    tmp_path,
    monkeypatch,
) -> None:
    """The full sweep persists 91 ascending rows and reports the 0.40-0.50 focus."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]
    metrics = evaluate_threshold_sweep(y_true, y_score)
    results = build_logistic_validation_sweep_table(metrics)
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

    _write_phase_4_3_outputs(results)

    filename = "logistic_regression_validation_threshold_sweep.csv"
    persisted = np.genfromtxt(
        phase_tables_dir / filename,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert len(persisted) == 91
    assert persisted["threshold"].tolist() == list(SWEEP_CLASSIFICATION_THRESHOLDS)
    assert (persisted["evaluated_split"] == "validation").all()
    assert (root_tables_dir / filename).read_bytes() == (
        phase_tables_dir / filename
    ).read_bytes()

    report = (
        phase_report_dir / "phase_4_3_threshold_sweep.md"
    ).read_text(encoding="utf-8")
    normalized_report = " ".join(report.split())
    assert "91" in report
    assert "0.05" in report and "0.95" in report
    assert "no threshold is ranked, optimized, recommended, or selected" in report
    assert "ROC-AUC and PR-AUC are unaffected" in report
    for checkpoint in MANUAL_CLASSIFICATION_THRESHOLDS:
        assert f"{checkpoint:.4f}" in report
    for step in range(40, 51):
        assert f"{step / 100:.4f}" in report
    assert "single largest step change" in normalized_report
    assert report == (
        project_root / "reports/phase_4_3_threshold_sweep.md"
    ).read_text(encoding="utf-8")


def test_phase_4_3_focus_region_and_transition_summary_use_actual_sweep_values() -> (
    None
):
    """The 0.40-0.50 helpers read directly from the sweep table, not hard-coding."""
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_score = [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90]
    metrics = evaluate_threshold_sweep(y_true, y_score)
    results = build_logistic_validation_sweep_table(metrics)

    focus = _phase_4_3_focus_region(results)

    assert len(focus) == 11
    assert focus["threshold"].tolist() == [round(0.40 + step * 0.01, 2) for step in range(11)]
    assert focus["predicted_positive_count"].iloc[0] == 6
    assert focus["predicted_positive_count"].iloc[-1] == 2

    focus_table_text = _format_phase_4_3_focus_table(results)
    for step in range(40, 51):
        assert f"{step / 100:.4f}" in focus_table_text

    summary = _format_phase_4_3_transition_summary(results)
    assert "From threshold 0.40 to 0.50" in summary
    assert "ROC-AUC and PR-AUC are unaffected" in summary


def test_phase_4_4_plot_data_uses_all_sweep_fields_in_threshold_order() -> None:
    """Plot inputs come directly from the Phase 4.3 sweep table."""
    metrics = evaluate_threshold_sweep(
        [1, 0, 1, 0, 1],
        [0.80, 0.65, 0.45, 0.35, 0.25],
        thresholds=(0.30, 0.40, 0.50, 0.60, 0.70),
    )
    results = build_logistic_validation_sweep_table(metrics)

    plot_data = _threshold_sweep_plot_data(results)

    assert plot_data["threshold"].tolist() == [0.30, 0.40, 0.50, 0.60, 0.70]
    assert plot_data["precision"].tolist() == results["precision"].tolist()
    assert plot_data["recall"].tolist() == results["recall"].tolist()
    assert plot_data["f1"].tolist() == results["f1"].tolist()
    assert plot_data["predicted_positive_rate"].tolist() == (
        results["predicted_positive_rate"].tolist()
    )
    assert plot_data["true_positives"].tolist() == results["true_positives"].tolist()
    assert plot_data["false_positives"].tolist() == results["false_positives"].tolist()
    assert plot_data["false_negatives"].tolist() == results["false_negatives"].tolist()
    assert len(plot_data) == len(results)


def test_phase_4_4_figures_are_deterministic_and_reported(
    tmp_path,
    monkeypatch,
) -> None:
    """Phase 4.4 writes three fixed figure artifacts and one report."""
    metrics = evaluate_threshold_sweep(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90],
    )
    results = build_logistic_validation_sweep_table(metrics)
    phase_report_dir = tmp_path / "phase"
    phase_figures = phase_report_dir / "figures"
    root_figures = tmp_path / "root_figures"
    project_root = tmp_path / "project"
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.BASELINE_REPORT_DIR",
        phase_report_dir,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.FIGURES_DIR",
        phase_figures,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.ROOT_FIGURES_DIR",
        root_figures,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.PROJECT_ROOT",
        project_root,
    )

    figure_paths = _write_phase_4_4_outputs(results)

    filenames = [
        "logistic_regression_validation_threshold_precision_recall_f1.png",
        "logistic_regression_validation_threshold_flagged_rate.png",
        "logistic_regression_validation_threshold_classification_counts.png",
    ]
    assert [path.name for path in figure_paths] == filenames
    for directory in (phase_figures, root_figures):
        for filename in filenames:
            assert (directory / filename).stat().st_size > 0
    report = (phase_report_dir / "phase_4_4_threshold_tradeoffs.md").read_text(
        encoding="utf-8"
    )
    assert "Phase 4.3 Logistic Regression validation" in report
    assert "threshold sweep" in report
    assert "does not fit a model" in report
    assert "threshold-sensitive regions" in report
    for filename in filenames:
        assert f"reports/figures/{filename}" in report
    assert report == (
        project_root / "reports/phase_4_4_threshold_tradeoffs.md"
    ).read_text(encoding="utf-8")

    repeated_paths = _write_threshold_tradeoff_figures(results)
    assert [path.name for path in repeated_paths] == filenames


def test_phase_4_4_sensitivity_summary_uses_actual_sweep_values() -> None:
    """Sensitivity prose reports observed changes without selecting a threshold."""
    metrics = evaluate_threshold_sweep(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90],
    )
    results = build_logistic_validation_sweep_table(metrics)

    summary = _format_phase_4_4_sensitivity_summary(results)

    assert "Across the full sweep from 0.05 to 0.95" in summary
    assert "flagged complaints move from" in summary
    assert "largest one-step workload change" in summary
    assert "False negatives rise most sharply" in summary
    assert "false positives fall most sharply" in summary
    assert "not selected or recommended thresholds" in summary


def test_phase_4_5_outputs_compare_policy_candidates_without_selecting_winner(
    tmp_path,
    monkeypatch,
) -> None:
    """Policy candidates are persisted from the sweep without ranking winners."""
    metrics = evaluate_threshold_sweep(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90],
    )
    sweep = build_logistic_validation_sweep_table(metrics)
    results = build_logistic_validation_policy_table(
        evaluate_threshold_selection_policies(sweep)
    )
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

    _write_phase_4_5_outputs(results)

    filename = "logistic_regression_validation_threshold_policy_candidates.csv"
    persisted = pd.read_csv(phase_tables_dir / filename)
    report = (
        phase_report_dir / "phase_4_5_threshold_policy_candidates.md"
    ).read_text(encoding="utf-8")
    normalized_report = " ".join(report.split())
    assert persisted["policy_name"].tolist() == [
        "Max F1",
        "Recall >= 0.70",
        "Precision >= 0.50",
        "Flagged rate <= 0.30",
    ]
    assert (root_tables_dir / filename).read_bytes() == (
        phase_tables_dir / filename
    ).read_bytes()
    assert "existing Phase 4.3 Logistic Regression validation sweep" in report
    assert "does not retrain" in normalized_report
    assert "freeze a threshold" in report
    assert "Tie-breaking is deterministic" in report
    assert "Different policies answer different questions" in report
    assert "winner" not in normalized_report.lower()
    assert report == (
        project_root / "reports/phase_4_5_threshold_policy_candidates.md"
    ).read_text(encoding="utf-8")


def test_phase_4_6_outputs_freeze_workload_policy_decision(
    tmp_path,
    monkeypatch,
) -> None:
    """The frozen threshold artifact is the approved Phase 4.5 workload result."""
    metrics = evaluate_threshold_sweep(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0.10, 0.20, 0.42, 0.44, 0.46, 0.48, 0.80, 0.90],
    )
    sweep = build_logistic_validation_sweep_table(metrics)
    policy_table = build_logistic_validation_policy_table(
        evaluate_threshold_selection_policies(sweep)
    )
    decision = freeze_workload_limited_threshold(policy_table)
    phase_report_dir = tmp_path / "phase"
    project_root = tmp_path / "project"
    decision_path = project_root / "configs/models/threshold_decision.json"
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.BASELINE_REPORT_DIR",
        phase_report_dir,
    )
    monkeypatch.setattr(
        "urban_ops.models.baseline_workflow.PROJECT_ROOT",
        project_root,
    )

    artifact_path = _write_phase_4_6_outputs(decision, decision_path=decision_path)

    assert artifact_path == decision_path
    loaded = load_frozen_threshold_decision(decision_path)
    assert loaded == decision
    rewritten_path = write_frozen_threshold_decision(loaded, path=decision_path)
    assert load_frozen_threshold_decision(rewritten_path) == decision
    report = (phase_report_dir / "phase_4_6_frozen_threshold_decision.md").read_text(
        encoding="utf-8"
    )
    normalized_report = " ".join(report.split())
    assert "predicted_positive_rate <=" in report
    assert "maximize_recall" in report
    assert "validation" in report
    assert "Frozen: `true`" in report
    assert f"{decision.selected_threshold:.2f}" in report
    assert f"{decision.precision:.4f}" in report
    assert f"{decision.recall:.4f}" in report
    assert "not universally optimal" in report
    assert "Phase 4.7 can load the frozen JSON artifact" in report
    assert "test precision" not in normalized_report.lower()
    assert "test recall" not in normalized_report.lower()
    assert "production performance" not in normalized_report.lower()
    assert report == (
        project_root / "reports/phase_4_6_frozen_threshold_decision.md"
    ).read_text(encoding="utf-8")
