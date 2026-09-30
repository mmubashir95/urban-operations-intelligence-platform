"""Unit tests for train-only Gradient Boosting orchestration."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from urban_ops.models import gradient_boosting_training
from urban_ops.models.gradient_boosting import (
    GradientBoostedRiskModel,
    load_gradient_boosting_config,
)
from urban_ops.models.gradient_boosting_training import (
    load_and_train_gradient_boosted_risk_model,
    train_gradient_boosted_risk_model,
)


FEATURE_NAMES = ("first", "second", "third")


def _inputs():
    train = sparse.csr_matrix(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        matrices={
            "train": train,
            "validation": sparse.csr_matrix((3, 3), dtype=np.float64),
            "test": sparse.csr_matrix((2, 3), dtype=np.float64),
        },
        targets={
            "train": pd.Series([0, 0, 0, 1, 0, 1, 1, 1]),
            "validation": pd.Series([1, 1, 1]),
            "test": pd.Series([0, 0]),
        },
        feature_names=FEATURE_NAMES,
        split_id="unit-training-split",
    )


def test_valid_verified_inputs_train_and_record_metadata() -> None:
    inputs = _inputs()

    result = train_gradient_boosted_risk_model(inputs)

    assert isinstance(result.model, GradientBoostedRiskModel)
    assert result.model.feature_names_ == FEATURE_NAMES
    assert result.metadata.training_split == "train"
    assert result.metadata.training_row_count == 8
    assert result.metadata.feature_count == 3
    assert result.metadata.negative_class_count == 4
    assert result.metadata.positive_class_count == 4
    assert result.metadata.positive_class_rate == 0.5
    assert result.metadata.random_state == load_gradient_boosting_config().random_state


def test_orchestration_passes_only_train_objects_and_unchanged_features(
    monkeypatch,
) -> None:
    inputs = _inputs()
    observed = {}

    class SpyModel(GradientBoostedRiskModel):
        def fit(self, X_train, y_train, *, feature_names):
            observed["X"] = X_train
            observed["y"] = y_train
            observed["feature_names"] = feature_names
            return super().fit(X_train, y_train, feature_names=feature_names)

    monkeypatch.setattr(gradient_boosting_training, "GradientBoostedRiskModel", SpyModel)

    result = train_gradient_boosted_risk_model(inputs)

    assert observed["X"] is inputs.matrices["train"]
    assert observed["y"] is inputs.targets["train"]
    assert observed["feature_names"] is inputs.feature_names
    assert observed["X"] is not inputs.matrices["validation"]
    assert observed["X"] is not inputs.matrices["test"]
    assert observed["X"].shape[0] == 8
    assert observed["X"].shape[0] != 8 + 3
    assert observed["X"].shape[0] != 8 + 2
    assert sparse.isspmatrix_csr(observed["X"])
    assert result.model.config == load_gradient_boosting_config()


def test_load_and_train_begins_with_phase_2_1_gate(monkeypatch) -> None:
    inputs = _inputs()
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return inputs

    monkeypatch.setattr(
        gradient_boosting_training,
        "load_and_verify_gradient_boosting_inputs",
        fake_loader,
    )

    result = load_and_train_gradient_boosted_risk_model(
        eda_config_path="fixture-eda.yaml"
    )

    assert calls == [
        {"eda_config_path": "fixture-eda.yaml", "split_run_path": None}
    ]
    assert result.metadata.training_row_count == len(inputs.targets["train"])


@pytest.mark.parametrize(
    ("collection", "message"),
    [("matrices", "train matrix"), ("targets", "train target")],
)
def test_missing_training_split_fails_clearly(collection: str, message: str) -> None:
    inputs = _inputs()
    del getattr(inputs, collection)["train"]

    with pytest.raises(ValueError, match=message):
        train_gradient_boosted_risk_model(inputs)


def test_phase_2_1_verification_failure_is_preserved(monkeypatch) -> None:
    expected = RuntimeError("frozen input contract failed")

    def failing_loader(**kwargs):
        raise expected

    monkeypatch.setattr(
        gradient_boosting_training,
        "load_and_verify_gradient_boosting_inputs",
        failing_loader,
    )

    with pytest.raises(RuntimeError, match="frozen input contract failed") as caught:
        load_and_train_gradient_boosted_risk_model()

    assert caught.value is expected


def test_returned_model_is_fitted_and_usable_on_small_train_slice() -> None:
    inputs = _inputs()
    result = train_gradient_boosted_risk_model(inputs)

    scores = result.model.predict_score(inputs.matrices["train"][:2])

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_repeated_train_only_fit_is_deterministic() -> None:
    inputs = _inputs()

    first = train_gradient_boosted_risk_model(inputs).model
    second = train_gradient_boosted_risk_model(inputs).model
    fixed_slice = inputs.matrices["train"][:3]

    np.testing.assert_allclose(
        first.predict_score(fixed_slice),
        second.predict_score(fixed_slice),
        rtol=0.0,
        atol=0.0,
    )
