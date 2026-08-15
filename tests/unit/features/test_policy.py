"""Tests for the frozen Notebook 11 baseline feature policy."""

from pathlib import Path

import pytest
import yaml

from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.features.policy import PolicyStatus, load_feature_policy


POLICY_PATH = Path("configs/features/resolution_risk_baseline.yaml")


def _payload() -> dict[str, object]:
    """Return a mutable copy of the repository policy payload."""
    payload = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_policy(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one intentionally modified policy for validation tests."""
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _entry(payload: dict[str, object], feature_name: str) -> dict[str, object]:
    """Return one mutable raw feature entry."""
    features = payload["features"]
    assert isinstance(features, list)
    return next(item for item in features if item["feature_name"] == feature_name)


def test_valid_policy_loads_and_reconciles_notebook_10_evidence() -> None:
    policy = load_feature_policy(POLICY_PATH)

    assert policy.policy_version == 1
    assert policy.policy_stage == "feature_policy_freeze"
    assert len(policy.features) == 54
    assert policy.notebook_10_handoff == {
        "split_id": "20260806T135114Z_9d945cb2da0eecfc",
        "integrity": "PASS",
        "reconciliation": "PASS",
        "step_9a_decision": "COMPLETE",
        "model_ready": False,
    }
    assert policy.feature_creation_allow_list == (
        "created_hour",
        "created_day_of_week",
        "created_month",
        "is_weekend",
    )


def test_unknown_status_fails(tmp_path: Path) -> None:
    payload = _payload()
    _entry(payload, "created_hour")["policy_status"] = "MAYBE"

    with pytest.raises(FeaturePolicyError, match="unknown policy_status"):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)


def test_duplicate_feature_fails(tmp_path: Path) -> None:
    payload = _payload()
    features = payload["features"]
    assert isinstance(features, list)
    features.append(dict(features[0]))

    with pytest.raises(FeaturePolicyError, match="duplicates"):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)


def test_missing_required_entry_field_fails(tmp_path: Path) -> None:
    payload = _payload()
    del _entry(payload, "created_hour")["reason"]

    with pytest.raises(FeaturePolicyError, match="missing required fields"):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)


@pytest.mark.parametrize(
    "feature_name",
    [
        "closed_date",
        "due_date",
        "missed_resolution_target",
        "unique_key",
        "descriptor_2",
        "agency",
        "agency_name",
        "complaint_type",
        "descriptor",
        "open_data_channel_type",
    ],
)
def test_governed_blocked_or_unusable_fields_cannot_be_approved(
    tmp_path: Path, feature_name: str
) -> None:
    payload = _payload()
    entry = _entry(payload, feature_name)
    entry["policy_status"] = "APPROVED_CANDIDATE"
    entry["phase_2_allowed"] = True

    with pytest.raises(FeaturePolicyError, match=feature_name):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)


@pytest.mark.parametrize(
    ("feature_name", "expected_status"),
    [
        ("created_hour", PolicyStatus.APPROVED_CANDIDATE),
        ("created_day_of_week", PolicyStatus.APPROVED_CANDIDATE),
        ("created_month", PolicyStatus.APPROVED_CANDIDATE),
        ("is_weekend", PolicyStatus.APPROVED_CANDIDATE),
        ("created_day_name", PolicyStatus.ALTERNATIVE_REPRESENTATION),
        ("created_month_name", PolicyStatus.ALTERNATIVE_REPRESENTATION),
        ("created_quarter", PolicyStatus.REVIEW_REDUNDANCY),
        ("created_week_of_year", PolicyStatus.REVIEW_REDUNDANCY),
        ("created_day_of_month", PolicyStatus.REVIEW),
        ("created_year", PolicyStatus.CONDITIONAL),
        ("created_date", PolicyStatus.SOURCE_ONLY),
    ],
)
def test_temporal_statuses_follow_notebook_10_eda_status(
    feature_name: str, expected_status: PolicyStatus
) -> None:
    entry = load_feature_policy(POLICY_PATH).by_name[feature_name]

    assert entry.policy_status is expected_status
    assert entry.phase_2_allowed is (expected_status is PolicyStatus.APPROVED_CANDIDATE)


@pytest.mark.parametrize(
    "feature_name",
    ["borough", "location_type", "incident_zip", "latitude", "longitude"],
)
def test_unresolved_geography_remains_conditional(feature_name: str) -> None:
    entry = load_feature_policy(POLICY_PATH).by_name[feature_name]

    assert entry.policy_status is PolicyStatus.CONDITIONAL
    assert entry.prediction_time_status == "UNRESOLVED"
    assert entry.phase_2_allowed is False


def test_exclusion_groups_match_notebook_10_evidence_precedence() -> None:
    policy = load_feature_policy(POLICY_PATH)
    grouped = {
        status: {
            entry.feature_name for entry in policy.features if entry.policy_status is status
        }
        for status in (
            PolicyStatus.EXCLUDE_IDENTIFIER,
            PolicyStatus.EXCLUDE_ALL_NULL,
            PolicyStatus.EXCLUDE_ZERO_VARIANCE,
        )
    }

    assert grouped[PolicyStatus.EXCLUDE_IDENTIFIER] == {"unique_key"}
    assert grouped[PolicyStatus.EXCLUDE_ALL_NULL] == {"descriptor_2"}
    assert grouped[PolicyStatus.EXCLUDE_ZERO_VARIANCE] == {
        "agency",
        "agency_name",
        "complaint_type",
        "descriptor",
        "open_data_channel_type",
    }


def test_implementation_boundary_declares_no_preprocessing_or_modelling() -> None:
    boundary = load_feature_policy(POLICY_PATH).implementation_boundary

    prohibited_flags = {
        "transformations_implemented",
        "missing_values_imputed",
        "rare_categories_grouped",
        "encoder_fitted",
        "scaler_fitted",
        "column_transformer_built",
        "feature_matrix_created",
        "model_trained",
    }
    assert prohibited_flags.issubset(boundary)
    assert all(boundary[flag] is False for flag in prohibited_flags)


def test_incomplete_notebook_10_handoff_fails(tmp_path: Path) -> None:
    payload = _payload()
    payload["notebook_10_handoff"]["integrity"] = "FAIL"

    with pytest.raises(FeaturePolicyError, match="handoff"):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)


def test_preprocessing_or_modelling_boundary_flag_cannot_be_enabled(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["implementation_boundary"]["encoder_fitted"] = True

    with pytest.raises(FeaturePolicyError, match="encoder_fitted"):
        load_feature_policy(_write_policy(tmp_path, payload), validate_evidence=False)
