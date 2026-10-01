"""Integration coverage for raw validation calibration evaluation."""

from dataclasses import replace

import numpy as np

from urban_ops.models.gradient_boosting_calibration import (
    evaluate_gradient_boosting_validation_calibration,
    write_gradient_boosting_calibration_table,
)
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
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
            raise AssertionError("Phase 2.8 attempted to evaluate test labels.")
        return super().__getitem__(key)


def test_real_pipeline_evaluates_and_persists_raw_validation_calibration(
    tmp_path,
) -> None:
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
        AssertionError("Phase 2.8 must not refit or calibrate the model.")
    )

    result = evaluate_gradient_boosting_validation_calibration(
        evaluation_inputs,
        validation_scores,
    )
    output_path = write_gradient_boosting_calibration_table(
        result,
        tmp_path / "reports" / "month_2",
    )

    assert targets.accessed == ["validation"]
    assert result.calibration.metrics.row_count == len(inputs.targets["validation"])
    assert np.isfinite(result.calibration.metrics.brier_score)
    assert 0.0 <= result.calibration.metrics.brier_score <= 1.0
    assert int(result.table["row_count"].sum()) == len(
        inputs.targets["validation"]
    )
    assert output_path.is_file()
    assert output_path.name == "phase_2_gradient_boosting_calibration.csv"
    assert bytes(training.model._model.get_booster().save_raw()) == booster_before
