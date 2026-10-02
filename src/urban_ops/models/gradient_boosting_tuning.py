"""Validate a small candidate set and fit the existing wrapper on TRAIN only.

Frozen identifiers are resolved from the tracked Month 1 evidence snapshot.
No search engine, preprocessing, threshold selection, or TEST scoring lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math

import yaml

from urban_ops.models.baseline_workflow import FrozenBaselineInputs
from urban_ops.models.gradient_boosting import (
    GradientBoostedRiskModel, GradientBoostingConfig, load_gradient_boosting_config,
)
from urban_ops.utils.paths import PROJECT_ROOT

TUNING_CONFIG_PATH = PROJECT_ROOT / "configs/models/resolution_risk_gradient_boosting_tuning.yaml"
TUNABLE_PARAMETERS = (
    "n_estimators", "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
)


class GradientBoostingTuningError(ValueError):
    """Raised when a candidate or tuning boundary violates the frozen contract."""


@dataclass(frozen=True)
class GradientBoostingCandidate:
    """One deliberate, immutable XGBoost configuration with validated ranges."""

    candidate_id: str
    purpose: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise GradientBoostingTuningError("candidate_id must be nonblank text.")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise GradientBoostingTuningError("Candidate purpose must be documented.")
        for name in ("n_estimators", "max_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GradientBoostingTuningError(f"{name} must be a positive integer.")
        for name in TUNABLE_PARAMETERS[1:]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GradientBoostingTuningError(f"{name} must be finite numeric data.")
            if name in ("learning_rate", "subsample", "colsample_bytree"):
                valid = 0 < value <= 1
            else:
                valid = value >= 0
            if not valid:
                raise GradientBoostingTuningError(f"Invalid {name} range.")

    def parameters(self) -> dict[str, int | float]:
        """Return only the eight explicitly allowed tuning parameters."""
        return {name: getattr(self, name) for name in TUNABLE_PARAMETERS}

    def model_config(self, baseline: GradientBoostingConfig) -> GradientBoostingConfig:
        """Preserve objective, seed, implementation, and threading from Phase 2."""
        return replace(baseline, **self.parameters())


@dataclass(frozen=True)
class GradientBoostingTuningContract:
    """Resolved tuning boundary and predefined deterministic selection policy."""

    version: int
    candidates: tuple[GradientBoostingCandidate, ...]
    frozen_split_id: str
    frozen_feature_fingerprint: str
    random_state: int
    pr_auc_tolerance: float
    selection_split: str = "validation"
    test_access_policy: str = "after_persisted_freeze_only"
    final_fit_policy: str = "train_only"
    allowed_tunable_parameters: tuple[str, ...] = TUNABLE_PARAMETERS

    def verify_inputs(self, inputs: FrozenBaselineInputs) -> None:
        """Reject another split or feature/preprocessing contract before fitting."""
        if inputs.split_id != self.frozen_split_id:
            raise GradientBoostingTuningError("Frozen split ID differs from tuning contract.")
        if inputs.phase_9_contract.fingerprint != self.frozen_feature_fingerprint:
            raise GradientBoostingTuningError("Frozen feature fingerprint differs from tuning contract.")
        if inputs.feature_names != inputs.phase_9_contract.ordered_feature_names:
            raise GradientBoostingTuningError("Feature names/order differ from the frozen contract.")


def load_tuning_contract(path: Path | str = TUNING_CONFIG_PATH) -> GradientBoostingTuningContract:
    """Strictly parse the bounded candidate set, rejecting unsupported keys."""
    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"version", "frozen_input_reference", "allowed_tunable_parameters", "random_state",
                "selection_split", "test_access_policy", "final_fit_policy", "selection_policy", "candidates"}
    if not isinstance(root, dict) or set(root) != expected:
        raise GradientBoostingTuningError("Unsupported or missing tuning contract keys.")
    baseline = load_gradient_boosting_config()
    if (root["version"] != 1 or isinstance(root["version"], bool)
        or root["allowed_tunable_parameters"] != list(TUNABLE_PARAMETERS)
        or root["random_state"] != baseline.random_state
        or isinstance(root["random_state"], bool)
        or root["selection_split"] != "validation"
        or root["test_access_policy"] != "after_persisted_freeze_only"
        or root["final_fit_policy"] != "train_only"):
        raise GradientBoostingTuningError("Tuning boundary may change only allowed hyperparameters.")
    policy = root["selection_policy"]
    if (not isinstance(policy, dict) or set(policy) != {"pr_auc_tolerance", "tie_breakers"}
        or policy["tie_breakers"] != ["recall_at_10", "brier_score", "complexity", "candidate_id"]):
        raise GradientBoostingTuningError("Unsupported selection policy.")
    tolerance = policy["pr_auc_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise GradientBoostingTuningError("PR-AUC tolerance must be finite in [0, 1].")
    definitions = root["candidates"]
    if not isinstance(definitions, list) or not 6 <= len(definitions) <= 12:
        raise GradientBoostingTuningError("Define between 6 and 12 deliberate candidates.")
    candidates = []
    for definition in definitions:
        required = {"candidate_id", "purpose", "n_estimators", "learning_rate", "max_depth"}
        if not isinstance(definition, dict) or not required <= set(definition) or set(definition) - (required | set(TUNABLE_PARAMETERS)):
            raise GradientBoostingTuningError("Unsupported or missing candidate keys.")
        candidates.append(GradientBoostingCandidate(**definition))
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise GradientBoostingTuningError("Duplicate candidate IDs.")
    reference = [candidate for candidate in candidates if candidate.candidate_id == "phase2_baseline"]
    expected_reference = GradientBoostingCandidate("phase2_baseline", "reference", baseline.n_estimators, baseline.learning_rate, baseline.max_depth)
    if len(reference) != 1 or reference[0].parameters() != expected_reference.parameters():
        raise GradientBoostingTuningError("Phase 2 configuration must be retained exactly.")
    evidence_path = Path(root["frozen_input_reference"])
    if not evidence_path.is_absolute():
        evidence_path = PROJECT_ROOT / evidence_path
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    return GradientBoostingTuningContract(
        version=1, candidates=tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id)),
        frozen_split_id=evidence["split_id"],
        frozen_feature_fingerprint=evidence["phase_9_contract_fingerprint"],
        random_state=baseline.random_state, pr_auc_tolerance=float(tolerance),
    )


def train_candidate(candidate: GradientBoostingCandidate, *, inputs: FrozenBaselineInputs) -> GradientBoostedRiskModel:
    """Fit one candidate using the unchanged frozen TRAIN matrix and feature order."""
    model = GradientBoostedRiskModel(config=candidate.model_config(load_gradient_boosting_config()))
    model.fit(inputs.matrices["train"], inputs.targets["train"], feature_names=inputs.feature_names)
    return model
