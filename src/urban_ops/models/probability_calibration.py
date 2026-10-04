"""Fit and select a controlled probability layer for frozen model scores.

Only RAW, sklearn logistic (sigmoid), and sklearn isotonic mappings are
supported. This module accepts existing scores; it never fits a base model,
loads a split, chooses a threshold, or accesses TEST data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from urban_ops.models.baselines import DEFAULT_RANDOM_STATE
from urban_ops.models.evaluation import build_calibration_table
from urban_ops.models.gradient_boosting_selection import METRIC_NAMES, evaluate_scores
from urban_ops.utils.paths import PROJECT_ROOT

CALIBRATION_CONFIG_PATH = PROJECT_ROOT / "configs/models/resolution_risk_probability_calibration.yaml"
CALIBRATION_METHODS = ("RAW", "SIGMOID", "ISOTONIC")


class ProbabilityCalibrationError(ValueError):
    """Raised when a calibration contract, mapping, or decision is unsafe."""


def _probabilities(values: object, *, name: str) -> np.ndarray:
    """Return one finite, non-empty probability vector without mutating input."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ProbabilityCalibrationError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ProbabilityCalibrationError(f"{name} must contain finite probabilities in [0, 1].")
    return result


def _targets(values: object, *, expected_rows: int) -> np.ndarray:
    """Return binary targets aligned to the supplied score vector."""
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != expected_rows or set(result.tolist()) != {0, 1}:
        raise ProbabilityCalibrationError("Calibration targets must align and contain both classes {0, 1}.")
    return result.astype(int)


class ProbabilityCalibrator:
    """Apply one validated calibration method to existing base probabilities."""

    def __init__(self, method: str) -> None:
        if method not in CALIBRATION_METHODS:
            raise ProbabilityCalibrationError(f"Unsupported calibration method: {method!r}.")
        self.method = method
        self._estimator: LogisticRegression | IsotonicRegression | None = None

    def fit(self, raw_probabilities: object, y_true: object) -> ProbabilityCalibrator:
        """Fit a mapping on validation probabilities and labels only."""
        scores = _probabilities(raw_probabilities, name="raw_probabilities")
        targets = _targets(y_true, expected_rows=len(scores))
        if self.method == "SIGMOID":
            estimator = LogisticRegression(random_state=DEFAULT_RANDOM_STATE, solver="lbfgs")
            estimator.fit(scores.reshape(-1, 1), targets)
            self._estimator = estimator
        elif self.method == "ISOTONIC":
            estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            estimator.fit(scores, targets)
            self._estimator = estimator
        return self

    def predict(self, raw_probabilities: object) -> np.ndarray:
        """Return calibrated positive-class probabilities preserving row count."""
        scores = _probabilities(raw_probabilities, name="raw_probabilities")
        if self.method == "RAW":
            return scores.copy()
        if self._estimator is None:
            raise ProbabilityCalibrationError("Calibrator must be fitted before prediction.")
        if self.method == "SIGMOID":
            result = self._estimator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            result = self._estimator.predict(scores)
        return _probabilities(result, name="calibrated_probabilities")


@dataclass(frozen=True)
class CalibrationPolicy:
    """Explicit validation-only calibration selection tolerances."""

    minimum_brier_improvement: float
    brier_tie_tolerance: float
    maximum_metric_degradation: float
    simpler_method_order: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationResult:
    """Deterministic validation evidence for one calibration method."""

    method: str
    probabilities: np.ndarray
    metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS or tuple(name for name, _ in self.metrics) != METRIC_NAMES:
            raise ProbabilityCalibrationError("Calibration result schema differs from the controlled contract.")
        _probabilities(self.probabilities, name="probabilities")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for _, value in self.metrics):
            raise ProbabilityCalibrationError("Calibration metrics must be finite in [0, 1].")

    def record(self) -> dict[str, object]:
        """Return a flat comparison row."""
        return {"method": self.method, **dict(self.metrics)}


def evaluate_calibration_method(method: str, probabilities: object, y_true: object) -> CalibrationResult:
    """Evaluate one score vector through the shared metric stack."""
    scores = _probabilities(probabilities, name="probabilities")
    metrics = evaluate_scores(y_true, scores)
    scores.setflags(write=False)
    return CalibrationResult(method, scores, metrics)


def select_calibration(results: tuple[CalibrationResult, ...], policy: CalibrationPolicy) -> CalibrationResult:
    """Select lower Brier only when all ranking and Top-K safety guards pass."""
    if {result.method for result in results} != set(CALIBRATION_METHODS) or len(results) != len(CALIBRATION_METHODS):
        raise ProbabilityCalibrationError("Selection requires exactly RAW, SIGMOID, and ISOTONIC results.")
    raw = next(result for result in results if result.method == "RAW")
    raw_metrics = dict(raw.metrics)
    protected = tuple(name for name in METRIC_NAMES if name != "brier_score")
    eligible = [raw]
    for result in results:
        if result.method == "RAW":
            continue
        metrics = dict(result.metrics)
        improves = raw_metrics["brier_score"] - metrics["brier_score"] >= policy.minimum_brier_improvement
        safe = all(raw_metrics[name] - metrics[name] <= policy.maximum_metric_degradation for name in protected)
        if improves and safe:
            eligible.append(result)
    best_brier = min(dict(result.metrics)["brier_score"] for result in eligible)
    tied = [result for result in eligible if dict(result.metrics)["brier_score"] - best_brier <= policy.brier_tie_tolerance]
    order = {method: index for index, method in enumerate(policy.simpler_method_order)}
    return min(tied, key=lambda result: order[result.method])


def load_calibration_policy(path: Path | str = CALIBRATION_CONFIG_PATH) -> CalibrationPolicy:
    """Strictly load the bounded method set and deterministic selection policy."""
    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"version", "methods", "selection_split", "test_access_policy", "selection_policy"}
    if not isinstance(root, dict) or set(root) != expected or root["version"] != 1:
        raise ProbabilityCalibrationError("Unsupported calibration configuration schema.")
    if tuple(root["methods"]) != CALIBRATION_METHODS or root["selection_split"] != "validation" or root["test_access_policy"] != "after_persisted_freeze_only":
        raise ProbabilityCalibrationError("Calibration boundaries or method set changed.")
    definition = root["selection_policy"]
    fields = {"minimum_brier_improvement", "brier_tie_tolerance", "maximum_metric_degradation", "simpler_method_order"}
    if not isinstance(definition, dict) or set(definition) != fields or tuple(definition["simpler_method_order"]) != CALIBRATION_METHODS:
        raise ProbabilityCalibrationError("Unsupported calibration selection policy.")
    values = [definition[name] for name in fields - {"simpler_method_order"}]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ProbabilityCalibrationError("Calibration tolerances must be finite values in [0, 1].")
    return CalibrationPolicy(
        minimum_brier_improvement=float(definition["minimum_brier_improvement"]),
        brier_tie_tolerance=float(definition["brier_tie_tolerance"]),
        maximum_metric_degradation=float(definition["maximum_metric_degradation"]),
        simpler_method_order=tuple(definition["simpler_method_order"]),
    )


@dataclass(frozen=True)
class FrozenCalibrationDecision:
    """Immutable calibration selection linked to the frozen Phase 3 model."""

    selected_method: str
    base_model_identifier: str
    phase_3_decision_path: str
    phase_3_selection_timestamp: str
    split_id: str
    feature_fingerprint: str
    feature_names: tuple[str, ...]
    random_state: int
    raw_metrics: tuple[tuple[str, float], ...]
    selected_metrics: tuple[tuple[str, float], ...]
    selection_policy: CalibrationPolicy
    selection_timestamp: str
    selected_on_split: str = "validation"
    calibration_fit_split: str = "validation"
    base_model_fit_policy: str = "train_only"
    test_access_policy: str = "after_persisted_freeze_only"
    frozen: bool = True

    def payload(self) -> dict[str, object]:
        """Return a JSON-safe immutable decision mapping."""
        return asdict(self)


def verify_persisted_calibration_decision(decision: FrozenCalibrationDecision, path: Path) -> None:
    """Reject a missing, edited, or unfrozen decision before TEST access."""
    expected = json.loads(json.dumps(decision.payload()))
    if not decision.frozen or decision.selected_on_split != "validation" or decision.calibration_fit_split != "validation":
        raise ProbabilityCalibrationError("Final evaluation requires a frozen validation decision.")
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ProbabilityCalibrationError("Persisted calibration decision is missing or changed.")


def calibration_table(method: str, y_true: object, probabilities: object):
    """Return the shared ten-bin calibration table with method identity."""
    table = build_calibration_table(y_true, probabilities)
    table.insert(0, "method", method)
    return table
