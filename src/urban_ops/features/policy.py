"""Load and validate the frozen baseline feature policy.

This module decides feature eligibility only. It does not derive features, fit
preprocessing, create model matrices, or train models.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
import yaml

from urban_ops.features.feature_roles import FIELD_ROLES, FeatureRole
from urban_ops.features.leakage import FeaturePolicyError
from urban_ops.utils.paths import PROJECT_ROOT


class PolicyStatus(StrEnum):
    """Non-overlapping feature eligibility decisions."""

    APPROVED_CANDIDATE = "APPROVED_CANDIDATE"
    SOURCE_ONLY = "SOURCE_ONLY"
    ALTERNATIVE_REPRESENTATION = "ALTERNATIVE_REPRESENTATION"
    REVIEW_REDUNDANCY = "REVIEW_REDUNDANCY"
    CONDITIONAL = "CONDITIONAL"
    REVIEW = "REVIEW"
    EXCLUDE_LEAKAGE = "EXCLUDE_LEAKAGE"
    EXCLUDE_IDENTIFIER = "EXCLUDE_IDENTIFIER"
    EXCLUDE_ALL_NULL = "EXCLUDE_ALL_NULL"
    EXCLUDE_ZERO_VARIANCE = "EXCLUDE_ZERO_VARIANCE"


PREDICTION_TIME_STATUSES: Final = frozenset(
    {"AVAILABLE", "UNAVAILABLE", "UNRESOLVED"}
)
LEAKAGE_STATUSES: Final = frozenset({"SAFE", "CONDITIONAL", "BLOCKED"})
EDA_STATUSES: Final = frozenset(
    {
        "CANDIDATE",
        "ALTERNATIVE_REPRESENTATION",
        "REVIEW_REDUNDANCY",
        "CONDITIONAL",
        "REVIEW",
        "EXCLUDE",
        "NOT_ELIGIBLE",
    }
)
VARIATION_STATUSES: Final = frozenset(
    {"HAS_VARIATION", "ZERO_VARIANCE", "ALL_NULL"}
)
REDUNDANCY_STATUSES: Final = frozenset(
    {
        "NONE",
        "SOURCE_FOR_DERIVATIONS",
        "EXACT_ALTERNATIVE",
        "OVERLAPS_PREFERRED",
        "NOT_APPLICABLE",
    }
)
REQUIRED_ENTRY_FIELDS: Final = frozenset(
    {
        "feature_name",
        "source_column",
        "policy_status",
        "prediction_time_status",
        "leakage_status",
        "eda_status",
        "variation_status",
        "redundancy_status",
        "reason",
        "evidence_source",
        "phase_2_allowed",
    }
)


@dataclass(frozen=True)
class FeaturePolicyEntry:
    """One evidence-backed feature eligibility decision."""

    feature_name: str
    source_column: str
    policy_status: PolicyStatus
    prediction_time_status: str
    leakage_status: str
    eda_status: str
    variation_status: str
    redundancy_status: str
    reason: str
    evidence_source: tuple[str, ...]
    phase_2_allowed: bool
    preferred_counterpart: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the stable row used by Notebook 11's policy table."""
        return {
            "feature_name": self.feature_name,
            "source_column": self.source_column,
            "policy_status": self.policy_status.value,
            "prediction_time_status": self.prediction_time_status,
            "leakage_status": self.leakage_status,
            "eda_status": self.eda_status,
            "variation_status": self.variation_status,
            "redundancy_status": self.redundancy_status,
            "reason": self.reason,
            "phase_2_allowed": self.phase_2_allowed,
            "preferred_counterpart": self.preferred_counterpart,
            "evidence_source": "; ".join(self.evidence_source),
        }


@dataclass(frozen=True)
class FeaturePolicy:
    """Validated, immutable baseline feature policy and its provenance."""

    policy_version: int
    policy_name: str
    policy_stage: str
    prediction_moment: dict[str, object]
    authority_hierarchy: tuple[dict[str, object], ...]
    notebook_10_handoff: dict[str, object]
    evidence_tables: dict[str, str]
    implementation_boundary: dict[str, object]
    features: tuple[FeaturePolicyEntry, ...]

    @property
    def by_name(self) -> dict[str, FeaturePolicyEntry]:
        """Return entries indexed by unique feature name."""
        return {entry.feature_name: entry for entry in self.features}

    @property
    def feature_creation_allow_list(self) -> tuple[str, ...]:
        """Return the deterministic feature-creation allow-list."""
        return tuple(
            entry.feature_name for entry in self.features if entry.phase_2_allowed
        )


def _required_mapping(value: object, name: str) -> dict[str, object]:
    """Return a string-keyed mapping or raise a readable policy error."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise FeaturePolicyError(f"{name} must be a mapping.")
    return value


def _required_text(value: object, field: str, feature_name: str) -> str:
    """Return a non-empty text field from one policy entry."""
    if not isinstance(value, str) or not value.strip():
        raise FeaturePolicyError(
            f"Feature {feature_name!r} requires non-empty field {field!r}."
        )
    return value.strip()


def _parse_entry(raw: object, index: int) -> FeaturePolicyEntry:
    """Parse and locally validate one YAML feature entry."""
    item = _required_mapping(raw, f"features[{index}]")
    missing = sorted(REQUIRED_ENTRY_FIELDS.difference(item))
    if missing:
        raise FeaturePolicyError(f"features[{index}] is missing required fields: {missing}")
    feature_name = _required_text(item["feature_name"], "feature_name", f"index {index}")
    status_text = _required_text(item["policy_status"], "policy_status", feature_name)
    try:
        policy_status = PolicyStatus(status_text)
    except ValueError as error:
        raise FeaturePolicyError(
            f"Feature {feature_name!r} has unknown policy_status {status_text!r}."
        ) from error

    bounded_fields = {
        "prediction_time_status": PREDICTION_TIME_STATUSES,
        "leakage_status": LEAKAGE_STATUSES,
        "eda_status": EDA_STATUSES,
        "variation_status": VARIATION_STATUSES,
        "redundancy_status": REDUNDANCY_STATUSES,
    }
    parsed: dict[str, str] = {}
    for field, allowed in bounded_fields.items():
        parsed[field] = _required_text(item[field], field, feature_name)
        if parsed[field] not in allowed:
            raise FeaturePolicyError(
                f"Feature {feature_name!r} has unknown {field} {parsed[field]!r}."
            )
    evidence = item["evidence_source"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(source, str) and source.strip() for source in evidence)
    ):
        raise FeaturePolicyError(
            f"Feature {feature_name!r} requires a non-empty evidence_source list."
        )
    if not isinstance(item["phase_2_allowed"], bool):
        raise FeaturePolicyError(
            f"Feature {feature_name!r} phase_2_allowed must be boolean."
        )
    counterpart = item.get("preferred_counterpart")
    if counterpart is not None:
        counterpart = _required_text(counterpart, "preferred_counterpart", feature_name)
    return FeaturePolicyEntry(
        feature_name=feature_name,
        source_column=_required_text(item["source_column"], "source_column", feature_name),
        policy_status=policy_status,
        prediction_time_status=parsed["prediction_time_status"],
        leakage_status=parsed["leakage_status"],
        eda_status=parsed["eda_status"],
        variation_status=parsed["variation_status"],
        redundancy_status=parsed["redundancy_status"],
        reason=_required_text(item["reason"], "reason", feature_name),
        evidence_source=tuple(str(source).strip() for source in evidence),
        phase_2_allowed=item["phase_2_allowed"],
        preferred_counterpart=counterpart,
    )


def _validate_entry_relationships(entries: tuple[FeaturePolicyEntry, ...]) -> None:
    """Validate cross-entry references and Phase 2 governance invariants."""
    names = [entry.feature_name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise FeaturePolicyError(f"Feature names must be unique; duplicates: {duplicates}")
    by_name = {entry.feature_name: entry for entry in entries}
    for entry in entries:
        if entry.source_column not in by_name:
            raise FeaturePolicyError(
                f"Feature {entry.feature_name!r} references unknown source_column "
                f"{entry.source_column!r}."
            )
        should_be_allowed = entry.policy_status is PolicyStatus.APPROVED_CANDIDATE
        if entry.phase_2_allowed != should_be_allowed:
            raise FeaturePolicyError(
                f"Feature {entry.feature_name!r} phase_2_allowed must be true only "
                "for APPROVED_CANDIDATE."
            )
        if should_be_allowed:
            blockers = []
            if entry.prediction_time_status != "AVAILABLE":
                blockers.append("prediction-time availability")
            if entry.leakage_status != "SAFE":
                blockers.append("leakage")
            if entry.variation_status != "HAS_VARIATION":
                blockers.append("usable variation")
            if entry.eda_status != "CANDIDATE":
                blockers.append("EDA candidate status")
            if entry.redundancy_status != "NONE":
                blockers.append("approved representation")
            if blockers:
                raise FeaturePolicyError(
                    f"Feature {entry.feature_name!r} cannot be APPROVED_CANDIDATE; "
                    f"unresolved blockers: {', '.join(blockers)}."
                )
        if entry.variation_status == "ALL_NULL" and (
            entry.policy_status is not PolicyStatus.EXCLUDE_ALL_NULL
        ):
            raise FeaturePolicyError(
                f"All-null feature {entry.feature_name!r} must be EXCLUDE_ALL_NULL."
            )
        if entry.variation_status == "ZERO_VARIANCE" and entry.policy_status not in {
            PolicyStatus.EXCLUDE_ZERO_VARIANCE,
            PolicyStatus.EXCLUDE_LEAKAGE,
        }:
            raise FeaturePolicyError(
                f"Zero-variance feature {entry.feature_name!r} cannot enter feature creation."
            )
        if entry.policy_status in {
            PolicyStatus.ALTERNATIVE_REPRESENTATION,
            PolicyStatus.REVIEW_REDUNDANCY,
        }:
            preferred = by_name.get(entry.preferred_counterpart or "")
            if preferred is None or preferred.policy_status is not PolicyStatus.APPROVED_CANDIDATE:
                raise FeaturePolicyError(
                    f"Feature {entry.feature_name!r} must point to an approved preferred_counterpart."
                )


def _validate_step_4_roles(entries: tuple[FeaturePolicyEntry, ...]) -> None:
    """Ensure the frozen policy cannot override Step 4 blocked field roles."""
    expected_exclusion = {
        FeatureRole.IDENTIFIER: PolicyStatus.EXCLUDE_IDENTIFIER,
        FeatureRole.TARGET_INPUT: PolicyStatus.EXCLUDE_LEAKAGE,
        FeatureRole.POST_CREATION_FIELD: PolicyStatus.EXCLUDE_LEAKAGE,
        FeatureRole.TARGET_DERIVED: PolicyStatus.EXCLUDE_LEAKAGE,
        FeatureRole.EXCLUDED: PolicyStatus.EXCLUDE_LEAKAGE,
    }
    for entry in entries:
        role = FIELD_ROLES.get(entry.feature_name)
        if role is None or role.role not in expected_exclusion:
            continue
        expected = expected_exclusion[role.role]
        if entry.policy_status is not expected:
            raise FeaturePolicyError(
                f"Step 4 blocks {entry.feature_name!r} as {role.role.value}; "
                f"required policy status is {expected.value}."
            )


def _read_evidence_rows(path_text: str, table_name: str) -> list[dict[str, str]]:
    """Read a configured Notebook 10 CSV without modifying it."""
    path = Path(path_text)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FeaturePolicyError(f"Notebook 10 evidence table is missing: {resolved}")
    try:
        with resolved.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise FeaturePolicyError(
            f"Could not read Notebook 10 evidence table {table_name!r}: {resolved}"
        ) from error


def validate_feature_policy_evidence(policy: FeaturePolicy) -> None:
    """Cross-check frozen decisions against authoritative Notebook 10 CSVs."""
    required_tables = {"baseline_recommendation", "leakage_audit", "all_null", "zero_variance"}
    missing_tables = sorted(required_tables.difference(policy.evidence_tables))
    if missing_tables:
        raise FeaturePolicyError(f"evidence_tables is missing: {missing_tables}")
    tables = {
        name: _read_evidence_rows(policy.evidence_tables[name], name)
        for name in required_tables
    }
    by_name = policy.by_name
    baseline_names = {row.get("feature_name", "") for row in tables["baseline_recommendation"]}
    if baseline_names != set(by_name):
        missing = sorted(baseline_names.difference(by_name))
        unexpected = sorted(set(by_name).difference(baseline_names))
        raise FeaturePolicyError(
            "Frozen policy must cover the complete Notebook 10 baseline inventory; "
            f"missing={missing}, unexpected={unexpected}."
        )

    leakage_rows = {row.get("column_name", ""): row for row in tables["leakage_audit"]}
    for name, row in leakage_rows.items():
        entry = by_name.get(name)
        if entry is None:
            continue
        if row.get("authoritative_role") == "IDENTIFIER":
            expected = PolicyStatus.EXCLUDE_IDENTIFIER
        elif row.get("leakage_status") == "BLOCKED":
            expected = PolicyStatus.EXCLUDE_LEAKAGE
        else:
            continue
        if entry.policy_status is not expected:
            raise FeaturePolicyError(
                f"Notebook 10 leakage evidence requires {name!r} to be {expected.value}."
            )

    all_null_names = {row.get("feature_name", "") for row in tables["all_null"]}
    for name in all_null_names:
        if by_name[name].policy_status is not PolicyStatus.EXCLUDE_ALL_NULL:
            raise FeaturePolicyError(
                f"Notebook 10 all-null evidence requires {name!r} to be EXCLUDE_ALL_NULL."
            )

    zero_variance_names = {
        row.get("feature_name", "") for row in tables["zero_variance"]
    }
    higher_priority = {
        PolicyStatus.EXCLUDE_LEAKAGE,
        PolicyStatus.EXCLUDE_IDENTIFIER,
        PolicyStatus.EXCLUDE_ALL_NULL,
    }
    for name in zero_variance_names:
        status = by_name[name].policy_status
        if status not in higher_priority and status is not PolicyStatus.EXCLUDE_ZERO_VARIANCE:
            raise FeaturePolicyError(
                f"Notebook 10 zero-variance evidence requires {name!r} to be excluded."
            )
    for entry in policy.features:
        if entry.policy_status is PolicyStatus.EXCLUDE_ALL_NULL and entry.feature_name not in all_null_names:
            raise FeaturePolicyError(
                f"Feature {entry.feature_name!r} lacks Notebook 10 all-null evidence."
            )
        if entry.policy_status is PolicyStatus.EXCLUDE_ZERO_VARIANCE and entry.feature_name not in zero_variance_names:
            raise FeaturePolicyError(
                f"Feature {entry.feature_name!r} lacks Notebook 10 zero-variance evidence."
            )


def _validate_completion_and_boundary(
    handoff: dict[str, object], boundary: dict[str, object]
) -> None:
    """Require a complete EDA handoff and a strict policy-only boundary."""
    expected_handoff = {
        "integrity": "PASS",
        "reconciliation": "PASS",
        "step_9a_decision": "COMPLETE",
        "model_ready": False,
    }
    mismatches = {
        key: (handoff.get(key), expected)
        for key, expected in expected_handoff.items()
        if handoff.get(key) != expected
    }
    if mismatches:
        raise FeaturePolicyError(
            f"Notebook 10 handoff is not eligible for policy freezing: {mismatches}"
        )
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
    missing = sorted(prohibited_flags.difference(boundary))
    if missing:
        raise FeaturePolicyError(
            f"implementation_boundary is missing required flags: {missing}"
        )
    enabled = sorted(flag for flag in prohibited_flags if boundary[flag] is not False)
    if enabled:
        raise FeaturePolicyError(
            "Feature-policy freeze cannot include preprocessing or modelling; "
            f"enabled flags: {enabled}"
        )


def load_feature_policy(
    path: Path | str,
    *,
    validate_evidence: bool = True,
) -> FeaturePolicy:
    """Load and fully validate one frozen baseline feature policy.

    Raises:
        FeaturePolicyError: If YAML structure, governance, references, or
            Notebook 10 evidence reconciliation is invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FeaturePolicyError(f"Feature policy file does not exist: {config_path}")
    try:
        root = _required_mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "policy root"
        )
    except (OSError, yaml.YAMLError) as error:
        raise FeaturePolicyError(f"Feature policy YAML is invalid: {config_path}") from error
    required_root = {
        "policy_version",
        "policy_name",
        "policy_stage",
        "prediction_moment",
        "authority_hierarchy",
        "notebook_10_handoff",
        "evidence_tables",
        "implementation_boundary",
        "features",
    }
    missing_root = sorted(required_root.difference(root))
    if missing_root:
        raise FeaturePolicyError(f"Feature policy root is missing: {missing_root}")
    if root["policy_version"] != 1:
        raise FeaturePolicyError("Only policy_version 1 is supported.")
    if not isinstance(root["features"], list) or not root["features"]:
        raise FeaturePolicyError("features must be a non-empty list.")
    entries = tuple(_parse_entry(raw, index) for index, raw in enumerate(root["features"]))
    _validate_entry_relationships(entries)
    _validate_step_4_roles(entries)
    handoff = _required_mapping(root["notebook_10_handoff"], "notebook_10_handoff")
    boundary = _required_mapping(
        root["implementation_boundary"], "implementation_boundary"
    )
    _validate_completion_and_boundary(handoff, boundary)
    policy = FeaturePolicy(
        policy_version=1,
        policy_name=_required_text(root["policy_name"], "policy_name", "policy root"),
        policy_stage=_required_text(
            root["policy_stage"], "policy_stage", "policy root"
        ),
        prediction_moment=_required_mapping(root["prediction_moment"], "prediction_moment"),
        authority_hierarchy=tuple(
            _required_mapping(item, "authority_hierarchy entry")
            for item in root["authority_hierarchy"]
        ),
        notebook_10_handoff=handoff,
        evidence_tables={
            str(key): _required_text(value, str(key), "evidence_tables")
            for key, value in _required_mapping(root["evidence_tables"], "evidence_tables").items()
        },
        implementation_boundary=boundary,
        features=entries,
    )
    if validate_evidence:
        validate_feature_policy_evidence(policy)
    return policy


def feature_policy_table(policy: FeaturePolicy) -> pd.DataFrame:
    """Return the ordered, human-readable baseline policy table."""
    return pd.DataFrame(entry.to_dict() for entry in policy.features)
