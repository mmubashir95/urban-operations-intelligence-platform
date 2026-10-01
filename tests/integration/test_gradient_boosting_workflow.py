"""Integration tests for the dedicated Month 2 Gradient Boosting workflow."""

from dataclasses import replace

import numpy as np

from urban_ops.models import gradient_boosting_workflow
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
    load_calls = []

    def load_inputs(**kwargs):
        load_calls.append(kwargs)
        return protected_inputs

    monkeypatch.setattr(
        gradient_boosting_workflow,
        "load_and_verify_gradient_boosting_inputs",
        load_inputs,
    )

    result = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "reports" / "month_2",
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


def test_workflow_is_deterministic_for_fixed_configuration(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    first = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "first",
    )
    second = run_gradient_boosting_workflow(
        eda_config_path=fixture.config,
        output_directory=tmp_path / "second",
    )

    np.testing.assert_array_equal(first.validation_scores, second.validation_scores)
    assert first.ranking == second.ranking
    assert first.calibration == second.calibration
    assert first.capacity == second.capacity
    assert first.feature_names == second.feature_names
    assert first.report_artifacts.comparison.equals(
        second.report_artifacts.comparison
    )
