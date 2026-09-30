"""Integration coverage for shared validation ranking evaluation."""

from dataclasses import replace

import numpy as np

from urban_ops.models import gradient_boosting_ranking
from urban_ops.models.evaluation import evaluate_ranking
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
    """Record target split access and reject test labels."""

    def __init__(self, values):
        super().__init__(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "test":
            raise AssertionError("Phase 2.7 attempted to evaluate test labels.")
        return super().__getitem__(key)


def test_real_pipeline_reuses_shared_validation_ranking_evaluator(
    tmp_path, monkeypatch
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
    calls = []

    def spy_evaluate_ranking(y_true, y_score):
        calls.append((y_true, y_score))
        return evaluate_ranking(y_true, y_score)

    monkeypatch.setattr(
        gradient_boosting_ranking,
        "evaluate_ranking",
        spy_evaluate_ranking,
    )
    training.model.fit = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Phase 2.7 must not refit the model.")
    )

    result = evaluate_gradient_boosting_validation_ranking(
        evaluation_inputs,
        validation_scores,
    )

    assert targets.accessed == ["validation"]
    assert calls == [(inputs.targets["validation"], validation_scores)]
    assert result.metrics.row_count == len(inputs.targets["validation"])
    assert np.isfinite(result.metrics.pr_auc)
    assert np.isfinite(result.metrics.roc_auc)
    assert 0.0 <= result.metrics.pr_auc <= 1.0
    assert 0.0 <= result.metrics.roc_auc <= 1.0
    assert bytes(training.model._model.get_booster().save_raw()) == booster_before
