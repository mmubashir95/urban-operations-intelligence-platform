"""Unit tests for the dedicated Gradient Boosting workflow composition."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from urban_ops.models import gradient_boosting_workflow
from urban_ops.models.evaluation import (
    build_calibration_table,
    compare_capacity_levels,
    evaluate_calibration,
    evaluate_ranking,
    get_standard_capacity_levels,
)
from urban_ops.models.gradient_boosting_reporting import (
    FrozenLogisticRegressionValidationEvidence,
    GradientBoostingReportArtifacts,
)
from urban_ops.models.gradient_boosting_workflow import (
    GradientBoostingWorkflowResult,
    run_gradient_boosting_workflow,
)


def test_workflow_reuses_one_validation_score_array_across_evaluators(
    monkeypatch,
    tmp_path,
) -> None:
    y_validation = pd.Series([0, 1, 0, 1])
    validation_scores = np.asarray([0.10, 0.80, 0.20, 0.90])
    inputs = SimpleNamespace(
        targets={"validation": y_validation},
        feature_names=("feature_a", "feature_b"),
        split_id="frozen-split",
    )
    training = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(model_parameters={"random_state": 20260806})
        ),
        metadata=SimpleNamespace(
            training_row_count=8,
            feature_count=2,
            implementation="XGBoost",
            model_class="GradientBoostedRiskModel",
            configuration_version=1,
        ),
    )
    calls = []
    ranking = evaluate_ranking(y_validation, validation_scores)
    calibration = evaluate_calibration(y_validation, validation_scores)
    calibration_table = build_calibration_table(y_validation, validation_scores)
    levels = get_standard_capacity_levels(len(y_validation))
    capacity = compare_capacity_levels(y_validation, validation_scores)
    capacity_table = pd.DataFrame(
        {
            "capacity": [0.05, 0.10, 0.20],
            "selected_count": [1, 1, 1],
        }
    )
    frozen_evidence = FrozenLogisticRegressionValidationEvidence(
        model_name="Logistic Regression",
        evaluated_split="validation",
        frozen=True,
        validation_source=tmp_path / "frozen_validation.csv",
        capacity_source=tmp_path / "frozen_capacity.csv",
        row_count=4,
        positive_count=2,
        metrics=(),
    )
    artifacts = GradientBoostingReportArtifacts(
        validation_path=tmp_path / "validation.csv",
        calibration_path=tmp_path / "calibration.csv",
        capacity_path=tmp_path / "capacity.csv",
        comparison_path=tmp_path / "comparison.csv",
        markdown_path=tmp_path / "report.md",
        frozen_logistic_regression=frozen_evidence,
        comparison=pd.DataFrame(),
    )

    monkeypatch.setattr(
        gradient_boosting_workflow,
        "load_and_verify_gradient_boosting_inputs",
        lambda **kwargs: inputs,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "train_gradient_boosted_risk_model",
        lambda observed: training,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "generate_gradient_boosting_validation_scores",
        lambda model, observed: validation_scores,
    )

    def spy_ranking(y_true, y_score):
        calls.append(("ranking", y_true, y_score))
        return ranking

    def spy_calibration(y_true, y_score):
        calls.append(("calibration", y_true, y_score))
        return calibration

    def spy_calibration_table(y_true, y_score, *, n_bins):
        calls.append(("calibration_table", y_true, y_score, n_bins))
        return calibration_table

    def spy_capacity(y_true, y_score, *, capacities):
        calls.append(("capacity", y_true, y_score, capacities))
        return capacity

    monkeypatch.setattr(gradient_boosting_workflow, "evaluate_ranking", spy_ranking)
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "evaluate_calibration",
        spy_calibration,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "build_calibration_table",
        spy_calibration_table,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "get_standard_capacity_levels",
        lambda count: levels,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "compare_capacity_levels",
        spy_capacity,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "build_gradient_boosting_capacity_table",
        lambda rows: capacity_table,
    )
    monkeypatch.setattr(
        gradient_boosting_workflow,
        "write_gradient_boosting_reports",
        lambda **kwargs: artifacts,
    )

    result = run_gradient_boosting_workflow(output_directory=tmp_path)

    assert isinstance(result, GradientBoostingWorkflowResult)
    assert result.validation_scores is validation_scores
    assert result.ranking is ranking
    assert result.calibration is calibration
    assert result.capacity is capacity
    assert result.feature_names == inputs.feature_names
    assert result.frozen_logistic_regression is frozen_evidence
    assert result.comparison is artifacts.comparison
    assert calls == [
        ("ranking", y_validation, validation_scores),
        ("calibration", y_validation, validation_scores),
        ("calibration_table", y_validation, validation_scores, 10),
        ("capacity", y_validation, validation_scores, (0.05, 0.10, 0.20)),
    ]


def test_workflow_module_does_not_import_baseline_runner() -> None:
    source = Path(gradient_boosting_workflow.__file__).read_text(encoding="utf-8")

    assert "run_baseline_workflow" not in source
    assert "0.49" not in source
    assert "threshold=" not in source
