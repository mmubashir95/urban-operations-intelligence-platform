"""Tests for the Phase 9 baseline modelling input contract."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_verification import VerifiedPreprocessingContract
from urban_ops.models.baseline_contract import (
    BaselineModellingContractError,
    VerifiedBaselineModellingContract,
    build_baseline_modelling_contract_evidence,
    load_baseline_modelling_contract_config,
    verify_baseline_modelling_contract,
)


CONFIG_PATH = Path("configs/models/resolution_risk_baseline_modelling_contract.yaml")
POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")
FEATURE_NAMES = (
    "created_hour",
    "created_day_of_week",
    "created_month",
    "is_weekend",
)
SPLITS = ("train", "validation", "test")


def _phase_8_contract(
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    row_count: int = 4,
    schema_fingerprint: str = "phase-8-schema",
    status: str = "FROZEN_MODEL_READY",
) -> VerifiedPreprocessingContract:
    return VerifiedPreprocessingContract(
        verification_version=1,
        policy_version=1,
        cardinality_fingerprint="phase-4",
        categorical_encoder_fingerprint="phase-5",
        numeric_preprocessor_fingerprint="phase-6",
        composition_fingerprint="phase-7",
        combined_feature_names=feature_names,
        combined_feature_count=len(feature_names),
        matrix_type="csr_matrix",
        output_dtype="float64",
        train_row_count=row_count,
        validation_row_count=row_count,
        test_row_count=row_count,
        target_name="missed_resolution_target",
        identifier_name="unique_key",
        timestamp_name="created_date",
        split_names=SPLITS,
        schema_fingerprint=schema_fingerprint,
        verification_status=status,
    )


def _valid_inputs(
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    row_count: int = 4,
):
    width = len(feature_names)
    base = np.arange(1, 1 + row_count * width, dtype=np.float64).reshape(
        row_count, width
    )
    matrices = {
        split: sparse.csr_matrix(base + offset, dtype=np.float64)
        for split, offset in zip(SPLITS, (0, 100, 200))
    }
    targets = {split: pd.Series([0, 1, 0, 1][:row_count]) for split in SPLITS}
    identifiers = {
        split: pd.Series([f"{split}-{index}" for index in range(row_count)])
        for split in SPLITS
    }
    timestamps = {
        "train": pd.Series(pd.date_range("2022-01-01", periods=row_count, tz="UTC")),
        "validation": pd.Series(
            pd.date_range("2022-02-01", periods=row_count, tz="UTC")
        ),
        "test": pd.Series(pd.date_range("2022-03-01", periods=row_count, tz="UTC")),
    }
    return matrices, targets, identifiers, timestamps


def _verify(
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    phase_8_contract: VerifiedPreprocessingContract | None = None,
    matrices=None,
    targets=None,
    identifiers=None,
    timestamps=None,
    expected_phase_8_fingerprint: str | None = None,
    expected_schema_fingerprint: str | None = None,
    use_policy: bool = True,
    use_default_phase_8: bool = True,
) -> VerifiedBaselineModellingContract:
    defaults = _valid_inputs(feature_names=feature_names)
    resolved_phase_8_contract = (
        _phase_8_contract(feature_names=feature_names)
        if phase_8_contract is None and use_default_phase_8
        else phase_8_contract
    )
    return verify_baseline_modelling_contract(
        matrices=matrices or defaults[0],
        targets=targets or defaults[1],
        identifiers=identifiers or defaults[2],
        timestamps=timestamps or defaults[3],
        feature_names=feature_names,
        phase_8_contract=resolved_phase_8_contract,
        config=load_baseline_modelling_contract_config(CONFIG_PATH),
        expected_phase_8_fingerprint=expected_phase_8_fingerprint,
        expected_schema_fingerprint=expected_schema_fingerprint,
        policy=load_feature_policy(POLICY_PATH) if use_policy else None,
    )


def test_valid_inputs_produce_contract_with_stable_fingerprint_and_round_trip() -> None:
    phase_8 = _phase_8_contract()
    first = _verify(
        phase_8_contract=phase_8,
        expected_phase_8_fingerprint=phase_8.fingerprint,
        expected_schema_fingerprint=phase_8.schema_fingerprint,
    )
    second = _verify(
        phase_8_contract=phase_8,
        expected_phase_8_fingerprint=phase_8.fingerprint,
        expected_schema_fingerprint=phase_8.schema_fingerprint,
    )
    restored = VerifiedBaselineModellingContract.from_dict(
        json.loads(json.dumps(first.to_dict()))
    )

    assert first == second
    assert first.fingerprint == second.fingerprint == restored.fingerprint
    assert restored == first
    assert first.status == "MODEL_INPUTS_VERIFIED"
    assert first.phase_8_contract_fingerprint == phase_8.fingerprint
    assert first.ordered_feature_names == FEATURE_NAMES
    assert first.feature_count == 4
    assert first.train_positive_class_count == 2
    assert first.train_negative_class_count == 2
    assert first.train_positive_class_prevalence == 0.5
    assert first.train_majority_class == 0


def test_evidence_table_is_metadata_only_and_split_ordered() -> None:
    contract = _verify()
    evidence = build_baseline_modelling_contract_evidence(contract)

    assert evidence["split"].tolist() == list(SPLITS)
    assert evidence["status"].eq("MODEL_INPUTS_VERIFIED").all()
    assert evidence["matrix_type"].eq("csr_matrix").all()
    assert evidence["matrix_dtype"].eq("float64").all()
    assert "matrix" not in evidence.columns


@pytest.mark.parametrize(
    ("phase_8", "kwargs", "message"),
    [
        (None, {}, "required"),
        (_phase_8_contract(status="DRAFT"), {}, "status"),
        (_phase_8_contract(), {"expected_phase_8_fingerprint": "wrong"}, "fingerprint"),
        (_phase_8_contract(), {"expected_schema_fingerprint": "wrong"}, "schema"),
    ],
)
def test_phase_8_lineage_failures(
    phase_8: VerifiedPreprocessingContract | None,
    kwargs: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(
            phase_8_contract=phase_8,
            use_default_phase_8=phase_8 is not None,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("feature_names", "message"),
    [
        (("created_day_of_week", "created_hour", "created_month", "is_weekend"), "schema"),
        (("created_hour", "created_day_of_week", "created_month"), "schema"),
        (("created_hour", "created_hour", "created_month", "is_weekend"), "schema"),
    ],
)
def test_modified_feature_names_order_or_count_fail(
    feature_names: tuple[str, ...], message: str
) -> None:
    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(feature_names=feature_names, phase_8_contract=_phase_8_contract())


def test_wrong_phase_8_schema_fingerprint_in_contract_fails_when_expected() -> None:
    phase_8 = _phase_8_contract(schema_fingerprint="changed")

    with pytest.raises(BaselineModellingContractError, match="schema fingerprint"):
        _verify(
            phase_8_contract=phase_8,
            expected_schema_fingerprint="phase-8-schema",
        )


def test_missing_or_unexpected_splits_fail() -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    del matrices["test"]
    timestamps["holdout"] = timestamps["test"].copy()

    with pytest.raises(BaselineModellingContractError, match="missing=.*test"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)

    matrices, targets, identifiers, timestamps = _valid_inputs()
    matrices["holdout"] = matrices["test"].copy()
    with pytest.raises(BaselineModellingContractError, match="unexpected=.*holdout"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


@pytest.mark.parametrize(
    ("matrix_factory", "message"),
    [
        (lambda m: m.toarray(), "csr_matrix"),
        (lambda m: sparse.csc_matrix(m), "csr_matrix"),
        (lambda m: sparse.csr_matrix(m.toarray(), dtype=np.float32), "dtype"),
        (lambda m: sparse.csr_matrix((0, m.shape[1]), dtype=np.float64), "greater than zero"),
        (lambda m: sparse.csr_matrix((m.shape[0], m.shape[1] - 1), dtype=np.float64), "feature count"),
    ],
)
def test_matrix_contract_failures(matrix_factory, message: str) -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    matrices["validation"] = matrix_factory(matrices["validation"])

    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_matrix_non_finite_values_fail(bad_value: float) -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    matrices["test"].data[0] = bad_value

    with pytest.raises(BaselineModellingContractError, match="non-finite"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


def test_row_count_must_match_phase_8_contract() -> None:
    with pytest.raises(BaselineModellingContractError, match="differs from Phase 8"):
        _verify(phase_8_contract=_phase_8_contract(row_count=5))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (pd.Series([0, 1, 0]), "count mismatch"),
        (pd.Series([0, 1, None, 1]), "null"),
        (pd.Series([0, 1, 2, 1]), "outside"),
        (pd.DataFrame({"y": [0, 1, 0, 1]}), "one-dimensional"),
    ],
)
def test_target_contract_failures(target: object, message: str) -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    targets["validation"] = target

    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


def test_training_target_must_contain_both_classes() -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    targets["train"] = pd.Series([0, 0, 0, 0])

    with pytest.raises(BaselineModellingContractError, match="both target classes"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        (pd.Series(["a", "b", "c"]), "count mismatch"),
        (pd.Series(["a", "b", None, "d"]), "null"),
        (pd.Series(["a", "b", "b", "d"]), "duplicate"),
    ],
)
def test_identifier_contract_failures(identifier: object, message: str) -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    identifiers["validation"] = identifier

    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


def test_identifier_overlap_across_splits_fails() -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    identifiers["test"].iloc[0] = identifiers["train"].iloc[0]

    with pytest.raises(BaselineModellingContractError, match="Split overlap"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        (pd.Series(pd.date_range("2022-02-01", periods=3, tz="UTC")), "count mismatch"),
        (pd.Series(["not-a-date", "2022-02-02", "2022-02-03", "2022-02-04"]), "invalid"),
        (pd.Series(pd.date_range("2022-02-01", periods=4)), "timezone-aware"),
        (pd.Series(pd.to_datetime(["2022-02-03", "2022-02-01", "2022-02-02", "2022-02-04"], utc=True)), "chronological"),
    ],
)
def test_chronology_contract_failures(timestamp: object, message: str) -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    timestamps["validation"] = timestamp

    with pytest.raises(BaselineModellingContractError, match=message):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


def test_split_chronology_violation_fails() -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    timestamps["validation"] = pd.Series(pd.date_range("2021-12-01", periods=4, tz="UTC"))

    with pytest.raises(BaselineModellingContractError, match="Chronology violation"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)


def test_row_index_alignment_violation_fails_without_repair() -> None:
    matrices, targets, identifiers, timestamps = _valid_inputs()
    original_index = identifiers["train"].index.copy()
    identifiers["train"].index = [10, 11, 12, 13]

    with pytest.raises(BaselineModellingContractError, match="alignment"):
        _verify(matrices=matrices, targets=targets, identifiers=identifiers, timestamps=timestamps)
    assert list(original_index) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "blocked_feature",
    [
        "missed_resolution_target",
        "unique_key",
        "created_date",
        "closed_date",
        "status",
        "borough",
        "incident_zip",
        "latitude",
    ],
)
def test_feature_isolation_rejects_forbidden_policy_fields(blocked_feature: str) -> None:
    feature_names = (blocked_feature,)
    matrices, targets, identifiers, timestamps = _valid_inputs(feature_names=feature_names)
    phase_8 = _phase_8_contract(feature_names=feature_names)

    with pytest.raises(
        BaselineModellingContractError,
        match="Forbidden modelling fields|Feature schema contains",
    ):
        _verify(
            feature_names=feature_names,
            phase_8_contract=phase_8,
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
        )


def test_malformed_persisted_contract_fails() -> None:
    contract = _verify()
    payload = contract.to_dict()
    payload["feature_count"] = 99

    with pytest.raises(BaselineModellingContractError, match="feature count"):
        VerifiedBaselineModellingContract.from_dict(payload)


def test_unsupported_runtime_config_fails() -> None:
    config = replace(load_baseline_modelling_contract_config(CONFIG_PATH), output_dtype="float32")
    matrices, targets, identifiers, timestamps = _valid_inputs()

    with pytest.raises(BaselineModellingContractError, match="runtime configuration"):
        verify_baseline_modelling_contract(
            matrices=matrices,
            targets=targets,
            identifiers=identifiers,
            timestamps=timestamps,
            feature_names=FEATURE_NAMES,
            phase_8_contract=_phase_8_contract(),
            config=config,
        )
