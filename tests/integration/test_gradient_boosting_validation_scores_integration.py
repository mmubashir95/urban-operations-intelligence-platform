"""Integration coverage for frozen validation risk-score generation."""

from dataclasses import replace

import numpy as np
from scipy import sparse

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


class RecordingMatrices(dict):
    """Record the sole split requested by Phase 2.6."""

    def __init__(self, values):
        super().__init__(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)


class LabelsForbidden(dict):
    """Fail if Phase 2.6 consults labels after train-only fitting."""

    def __getitem__(self, key):
        raise AssertionError(f"Phase 2.6 accessed {key} labels.")


def test_fitted_phase_2_5_model_scores_only_frozen_validation(tmp_path) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )
    training = train_gradient_boosted_risk_model(inputs)
    validation_matrix = inputs.matrices["validation"]
    recording_matrices = RecordingMatrices(inputs.matrices)
    scoring_inputs = replace(
        inputs,
        matrices=recording_matrices,
        targets=LabelsForbidden(),
    )
    feature_names_before = training.model.feature_names_
    booster_before = bytes(training.model._model.get_booster().save_raw())

    scores = generate_gradient_boosting_validation_scores(
        training.model,
        scoring_inputs,
    )

    assert recording_matrices.accessed == ["validation"]
    assert sparse.isspmatrix_csr(validation_matrix)
    assert scores.ndim == 1
    assert scores.shape == (validation_matrix.shape[0],)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert training.model.feature_count_ == validation_matrix.shape[1]
    assert training.model.feature_names_ == feature_names_before
    assert bytes(training.model._model.get_booster().save_raw()) == booster_before
