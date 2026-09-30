"""Integration test for train-only fitting through frozen Month 1 inputs."""

import numpy as np
from scipy import sparse

from urban_ops.models import gradient_boosting_training
from urban_ops.models.gradient_boosting_training import (
    load_and_train_gradient_boosted_risk_model,
    train_gradient_boosted_risk_model,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


def test_real_frozen_input_path_fits_train_only(tmp_path, monkeypatch) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    original_model = gradient_boosting_training.GradientBoostedRiskModel
    original_loader = gradient_boosting_training.load_and_verify_gradient_boosting_inputs
    observed = {}

    def tracked_loader(**kwargs):
        inputs = original_loader(**kwargs)
        observed["inputs"] = inputs
        return inputs

    class SpyModel(original_model):
        def fit(self, X_train, y_train, *, feature_names):
            observed["X"] = X_train
            observed["y"] = y_train
            observed["feature_names"] = feature_names
            return super().fit(X_train, y_train, feature_names=feature_names)

    monkeypatch.setattr(gradient_boosting_training, "GradientBoostedRiskModel", SpyModel)
    monkeypatch.setattr(
        gradient_boosting_training,
        "load_and_verify_gradient_boosting_inputs",
        tracked_loader,
    )

    result = load_and_train_gradient_boosted_risk_model(
        eda_config_path=fixture.config,
        split_run_path=None,
    )

    assert sparse.isspmatrix_csr(observed["X"])
    assert observed["X"] is observed["inputs"].matrices["train"]
    assert observed["y"] is observed["inputs"].targets["train"]
    assert observed["feature_names"] is observed["inputs"].feature_names
    assert observed["X"] is not observed["inputs"].matrices["validation"]
    assert observed["X"] is not observed["inputs"].matrices["test"]
    assert observed["X"].shape[0] == result.metadata.training_row_count
    assert observed["X"].shape[1] == result.metadata.feature_count
    assert len(observed["y"]) == result.metadata.training_row_count
    assert observed["feature_names"] == result.metadata.feature_names
    assert result.model.feature_names_ == result.metadata.feature_names
    assert result.model.feature_count_ == result.metadata.feature_count
    assert result.metadata.training_split == "train"
    assert result.metadata.split_id == fixture.split_run.name.removeprefix("split_id=")

    repeated = train_gradient_boosted_risk_model(observed["inputs"])
    fixed_train_slice = observed["inputs"].matrices["train"][:3]
    np.testing.assert_allclose(
        result.model.predict_score(fixed_train_slice),
        repeated.model.predict_score(fixed_train_slice),
        rtol=0.0,
        atol=0.0,
    )
