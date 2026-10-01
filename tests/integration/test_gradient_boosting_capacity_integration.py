"""Integration coverage for Gradient Boosting validation capacity evaluation."""

from dataclasses import replace

import numpy as np

from urban_ops.models.gradient_boosting_calibration import (
    evaluate_gradient_boosting_validation_calibration,
)
from urban_ops.models.gradient_boosting_capacity import (
    evaluate_gradient_boosting_validation_capacity,
    write_gradient_boosting_capacity_table,
)
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)
from urban_ops.models.gradient_boosting_ranking import (
    evaluate_gradient_boosting_validation_ranking,
)
from urban_ops.models.gradient_boosting_training import (
    train_gradient_boosted_risk_model,
)
from urban_ops.models.gradient_boosting_validation import (
    generate_gradient_boosting_validation_scores,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


class RecordingTargets(dict):
    """Record target access and reject test labels."""

    def __init__(self, values):
        super().__init__(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "test":
            raise AssertionError("Phase 2.9 attempted to evaluate test labels.")
        return super().__getitem__(key)


def test_real_pipeline_evaluates_and_persists_validation_capacity(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    training = train_gradient_boosted_risk_model(inputs)
    validation_scores = generate_gradient_boosting_validation_scores(
        training.model,
        inputs,
    )
    targets = RecordingTargets(inputs.targets)
    evaluation_inputs = replace(inputs, targets=targets)
    booster_before = bytes(training.model._model.get_booster().save_raw())
    training.model.fit = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Phase 2.9 must not refit or threshold the model.")
    )

    ranking = evaluate_gradient_boosting_validation_ranking(
        evaluation_inputs,
        validation_scores,
    )
    calibration = evaluate_gradient_boosting_validation_calibration(
        evaluation_inputs,
        validation_scores,
    )
    capacity = evaluate_gradient_boosting_validation_capacity(
        evaluation_inputs,
        validation_scores,
    )
    output_path = write_gradient_boosting_capacity_table(
        capacity,
        tmp_path / "reports" / "month_2",
    )

    assert targets.accessed == ["validation", "validation", "validation"]
    assert len(validation_scores) == len(inputs.targets["validation"])
    assert np.isfinite(validation_scores).all()
    assert ((validation_scores >= 0.0) & (validation_scores <= 1.0)).all()
    assert ranking.metrics.row_count == len(validation_scores)
    assert calibration.calibration.metrics.row_count == len(validation_scores)
    assert [level.capacity for level in capacity.capacity_levels] == [
        0.05,
        0.10,
        0.20,
    ]
    assert [level.k for level in capacity.capacity_levels] == [1, 1, 2]
    assert [row.selected_count for row in capacity.comparisons] == [1, 1, 2]
    assert all(0.0 <= row.precision <= 1.0 for row in capacity.comparisons)
    assert all(0.0 <= row.recall <= 1.0 for row in capacity.comparisons)
    assert output_path.is_file()
    assert output_path.name == "phase_2_gradient_boosting_capacity.csv"
    assert bytes(training.model._model.get_booster().save_raw()) == booster_before
