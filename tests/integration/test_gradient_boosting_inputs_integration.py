"""Integration coverage for Phase 2.1 using the real Month 1 loading path."""

from scipy import sparse

from urban_ops.models import gradient_boosting_inputs
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


def test_real_frozen_loader_is_reused_without_test_evaluation(
    tmp_path, monkeypatch
) -> None:
    fixture = build_eda_fixture(tmp_path, make_eda_frame())
    original_loader = gradient_boosting_inputs.load_frozen_baseline_inputs
    calls = []

    def tracked_loader(**kwargs):
        calls.append(kwargs)
        return original_loader(**kwargs)

    monkeypatch.setattr(
        gradient_boosting_inputs, "load_frozen_baseline_inputs", tracked_loader
    )

    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=fixture.config,
        split_run_path=None,
    )

    assert len(calls) == 1
    assert tuple(inputs.matrices) == ("train", "validation", "test")
    assert tuple(inputs.targets) == ("train", "validation", "test")
    assert all(sparse.issparse(inputs.matrices[split]) for split in inputs.matrices)
    assert all(
        inputs.matrices[split].shape[0] == len(inputs.targets[split])
        for split in inputs.matrices
    )
    assert all(
        inputs.matrices[split].shape[1] == len(inputs.feature_names)
        for split in inputs.matrices
    )
    assert all(
        set(inputs.targets[split].unique()).issubset({0, 1})
        for split in inputs.targets
    )
    assert inputs.phase_9_contract.ordered_feature_names == inputs.feature_names
    assert inputs.phase_9_contract.status == "MODEL_INPUTS_VERIFIED"
    assert not any(
        name.startswith(("predict", "score", "metric"))
        for name in vars(gradient_boosting_inputs)
    )
