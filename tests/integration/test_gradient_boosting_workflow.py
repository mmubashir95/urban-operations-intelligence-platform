"""Integration tests for the dedicated Month 2 Gradient Boosting workflow."""

from dataclasses import replace

import numpy as np
import pandas as pd

from urban_ops.models import gradient_boosting_workflow
from urban_ops.models.baselines import LogisticRegressionBaseline
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)
from urban_ops.models.gradient_boosting_workflow import (
    run_gradient_boosting_workflow,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


class ProtectedSplits(dict):
    """Record train/validation access and reject all test access."""

    def __init__(self, values):
        super().__init__(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "test":
            raise AssertionError("Phase 2.11 accessed the test split.")
        return super().__getitem__(key)


def _write_matching_frozen_evidence(tmp_path, inputs):
    """Write structured frozen evidence aligned to the synthetic validation split."""
    y_validation = inputs.targets["validation"]
    row_count = len(y_validation)
    positive_count = int(y_validation.sum())
    validation_path = tmp_path / "frozen_validation.csv"
    capacity_path = tmp_path / "frozen_capacity.csv"
    pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "evaluated_split": "validation",
                "row_count": row_count,
                "positive_count": positive_count,
                "pr_auc": 0.50,
                "roc_auc": 0.50,
                "brier_score": 0.25,
            }
        ]
    ).to_csv(validation_path, index=False)
    pd.DataFrame(
        {
            "model": ["Logistic Regression"] * 3,
            "evaluated_split": ["validation"] * 3,
            "capacity": [0.05, 0.10, 0.20],
            "selected_count": [1, 1, 2],
            "captured_positive_count": [1, 1, 1],
            "precision_at_k": [1.0, 1.0, 0.5],
            "recall_at_k": [1 / positive_count] * 3,
        }
    ).to_csv(capacity_path, index=False)
    return validation_path, capacity_path


def test_dedicated_workflow_uses_train_and_validation_only(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    matrices = ProtectedSplits(inputs.matrices)
    targets = ProtectedSplits(inputs.targets)
    protected_inputs = replace(inputs, matrices=matrices, targets=targets)
    frozen_validation_path, frozen_capacity_path = _write_matching_frozen_evidence(
        tmp_path,
        inputs,
    )
    load_calls = []

    def load_inputs(**kwargs):
        load_calls.append(kwargs)
        return protected_inputs

    monkeypatch.setattr(
        gradient_boosting_workflow,
        "load_and_verify_gradient_boosting_inputs",
        load_inputs,
    )
    monkeypatch.setattr(
        LogisticRegressionBaseline,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Phase 2.12 retrained Logistic Regression.")
        ),
    )

    result = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "reports" / "month_2",
        frozen_validation_path=frozen_validation_path,
        frozen_capacity_path=frozen_capacity_path,
    )

    assert load_calls == [
        {"eda_config_path": fixture.config, "split_run_path": None}
    ]
    assert matrices.accessed == ["train", "validation"]
    assert targets.accessed == ["train", "validation"]
    assert result.training.metadata.training_split == "train"
    assert result.training.metadata.feature_names == inputs.feature_names
    assert result.feature_names == inputs.feature_names
    assert len(result.validation_scores) == len(inputs.targets["validation"])
    assert result.ranking.metrics.row_count == len(result.validation_scores)
    assert result.calibration.metrics.row_count == len(result.validation_scores)
    assert [row.capacity for row in result.capacity] == [0.05, 0.10, 0.20]
    assert result.report_artifacts.validation_path.is_file()
    assert result.report_artifacts.calibration_path.is_file()
    assert result.report_artifacts.capacity_path.is_file()
    assert result.report_artifacts.comparison_path.is_file()
    assert result.report_artifacts.markdown_path.is_file()
    assert result.frozen_logistic_regression.frozen is True
    assert result.frozen_logistic_regression.evaluated_split == "validation"
    assert result.frozen_logistic_regression.row_count == len(
        result.validation_scores
    )
    assert result.comparison["metric"].tolist() == [
        "PR-AUC",
        "ROC-AUC",
        "Brier Score",
        "Precision@5%",
        "Recall@5%",
        "Precision@10%",
        "Recall@10%",
        "Precision@20%",
        "Recall@20%",
    ]
    expected_difference = (
        result.comparison["gradient_boosting"]
        - result.comparison["logistic_regression"]
    )
    np.testing.assert_allclose(
        result.comparison["difference_gb_minus_lr"],
        expected_difference,
    )


def test_workflow_is_deterministic_for_fixed_configuration(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    frozen_validation_path, frozen_capacity_path = _write_matching_frozen_evidence(
        tmp_path,
        inputs,
    )
    first = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "first",
        frozen_validation_path=frozen_validation_path,
        frozen_capacity_path=frozen_capacity_path,
    )
    second = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "second",
        frozen_validation_path=frozen_validation_path,
        frozen_capacity_path=frozen_capacity_path,
    )

    np.testing.assert_array_equal(first.validation_scores, second.validation_scores)
    assert first.ranking == second.ranking
    assert first.calibration == second.calibration
    assert first.capacity == second.capacity
    assert first.feature_names == second.feature_names
    assert first.report_artifacts.comparison.equals(
        second.report_artifacts.comparison
    )
