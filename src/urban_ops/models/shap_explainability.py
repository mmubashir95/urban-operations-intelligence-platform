"""Compute validated native Tree SHAP explanations for frozen XGBoost models.

XGBoost's ``pred_contribs`` and ``pred_interactions`` APIs implement Tree SHAP
in raw-margin space. This module performs no fitting, tuning, calibration,
feature transformation, split selection, or threshold selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import xgboost
import yaml
from scipy.stats import spearmanr
from xgboost import Booster, DMatrix

from urban_ops.utils.paths import PROJECT_ROOT

SHAP_CONFIG_PATH = PROJECT_ROOT / "configs/models/resolution_risk_shap_explainability.yaml"
EXPECTED_FEATURES = ("created_hour", "created_day_of_week", "created_month", "is_weekend")
CASE_ROLES = ("highest_risk", "lowest_risk", "true_positive", "true_negative", "false_positive", "false_negative")


class ShapExplainabilityError(ValueError):
    """Raised when frozen-model explanation inputs or outputs are unsafe."""


@dataclass(frozen=True)
class ShapConfig:
    """Strict deterministic Phase 5 explanation contract."""

    explanation_split: str
    sample_size: int
    random_state: int
    additivity_tolerance: float
    descriptive_reference_cutoff: float
    stability_half_size: int
    stability_minimum_spearman: float
    model_output: str
    explainer: str


@dataclass(frozen=True)
class ShapExplanationResult:
    """Immutable metadata with validated aligned Tree SHAP arrays."""

    feature_names: tuple[str, ...]
    shap_values: np.ndarray
    base_values: np.ndarray
    feature_values: np.ndarray
    row_positions: tuple[int, ...]
    identifiers: tuple[str, ...]
    raw_scores: np.ndarray
    raw_margins: np.ndarray
    output_space: str
    explainer_type: str
    xgboost_version: str

    def __post_init__(self) -> None:
        rows, columns = self.feature_values.shape if self.feature_values.ndim == 2 else (-1, -1)
        if not self.feature_names or columns != len(self.feature_names):
            raise ShapExplainabilityError("Feature names must align with explanation columns.")
        if self.shap_values.shape != (rows, columns):
            raise ShapExplainabilityError("SHAP row/column shape differs from explanation data.")
        if self.base_values.shape != (rows,) or self.raw_scores.shape != (rows,) or self.raw_margins.shape != (rows,):
            raise ShapExplainabilityError("Base values and model outputs must align by row.")
        if len(self.row_positions) != rows or len(self.identifiers) != rows or rows == 0:
            raise ShapExplainabilityError("Explanation row metadata must be non-empty and aligned.")
        for values in (self.shap_values, self.base_values, self.feature_values, self.raw_scores, self.raw_margins):
            if not np.isfinite(values).all():
                raise ShapExplainabilityError("Explanation values must be finite.")
        if self.output_space != "raw_margin" or self.explainer_type != "xgboost_native_tree_shap":
            raise ShapExplainabilityError("Tree SHAP output metadata differs from the Phase 5 contract.")


def load_shap_config(path: Path | str = SHAP_CONFIG_PATH) -> ShapConfig:
    """Load the exact Phase 5 explanation contract and reject drift."""
    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"version", "explanation_split", "sample_size", "random_state", "additivity_tolerance",
                "descriptive_reference_cutoff", "stability_half_size", "stability_minimum_spearman",
                "model_output", "explainer"}
    if not isinstance(root, dict) or set(root) != expected or root["version"] != 1:
        raise ShapExplainabilityError("Unsupported SHAP configuration schema.")
    if root["explanation_split"] != "test" or root["model_output"] != "raw_margin" or root["explainer"] != "xgboost_native_tree_shap":
        raise ShapExplainabilityError("SHAP split, output space, or explainer changed.")
    config = ShapConfig(**{key: root[key] for key in expected if key != "version"})
    if (config.sample_size != 2000 or config.stability_half_size != 1000 or config.random_state != 20260806
        or not 0 < config.additivity_tolerance <= 1e-4 or not 0 <= config.descriptive_reference_cutoff <= 1
        or not 0 <= config.stability_minimum_spearman <= 1):
        raise ShapExplainabilityError("SHAP deterministic policy differs from the approved contract.")
    return config


def _dense_matrix(matrix: object, *, feature_names: tuple[str, ...]) -> np.ndarray:
    """Return a small finite dense explanation matrix in frozen feature order."""
    if hasattr(matrix, "toarray"):
        values = np.asarray(matrix.toarray(), dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(feature_names):
        raise ShapExplainabilityError("Explanation matrix must be non-empty and match frozen feature count.")
    if not np.isfinite(values).all():
        raise ShapExplainabilityError("Explanation matrix must contain finite values.")
    return values


def explain_tree_model(booster: Booster, matrix: object, *, feature_names: tuple[str, ...],
                       row_positions: tuple[int, ...], identifiers: tuple[str, ...],
                       raw_scores: object, additivity_tolerance: float) -> ShapExplanationResult:
    """Generate native Tree SHAP contributions and verify raw-margin additivity."""
    if not isinstance(booster, Booster):
        raise ShapExplainabilityError("Only a fitted XGBoost Booster is supported.")
    if tuple(feature_names) != EXPECTED_FEATURES:
        raise ShapExplainabilityError("Feature ordering differs from the frozen Phase 3 contract.")
    values = _dense_matrix(matrix, feature_names=feature_names)
    scores = np.asarray(raw_scores, dtype=float)
    if scores.shape != (len(values),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ShapExplainabilityError("Raw scores must be aligned finite probabilities.")
    data = DMatrix(values)
    contributions = np.asarray(booster.predict(data, pred_contribs=True, validate_features=False), dtype=float)
    margins = np.asarray(booster.predict(data, output_margin=True, validate_features=False), dtype=float)
    if contributions.shape != (len(values), len(feature_names) + 1):
        raise ShapExplainabilityError("Tree SHAP output shape differs from features plus base value.")
    result = ShapExplanationResult(
        tuple(feature_names), contributions[:, :-1], contributions[:, -1], values,
        tuple(row_positions), tuple(str(value) for value in identifiers), scores, margins,
        "raw_margin", "xgboost_native_tree_shap", xgboost.__version__,
    )
    residual = np.abs(result.base_values + result.shap_values.sum(axis=1) - result.raw_margins)
    if float(residual.max()) > additivity_tolerance:
        raise ShapExplainabilityError(f"Tree SHAP additivity residual {residual.max()} exceeds tolerance {additivity_tolerance}.")
    return result


def additivity_summary(result: ShapExplanationResult, *, tolerance: float) -> dict[str, object]:
    """Return explicit all-row raw-margin additivity evidence."""
    residual = np.abs(result.base_values + result.shap_values.sum(axis=1) - result.raw_margins)
    return {"rows_checked": len(residual), "tolerance": tolerance, "maximum_residual": float(residual.max()),
            "status": "PASS" if float(residual.max()) <= tolerance else "FAIL"}


def global_importance(result: ShapExplanationResult) -> pd.DataFrame:
    """Rank features by deterministic mean absolute raw-margin SHAP value."""
    table = pd.DataFrame({"feature": result.feature_names, "mean_abs_shap": np.abs(result.shap_values).mean(axis=0)})
    table = table.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def direction_summary(result: ShapExplanationResult) -> pd.DataFrame:
    """Summarize contribution direction for each exact temporal feature value."""
    rows = []
    for column, feature in enumerate(result.feature_names):
        frame = pd.DataFrame({"feature_value": result.feature_values[:, column], "shap": result.shap_values[:, column]})
        for value, group in frame.groupby("feature_value", sort=True):
            rows.append({"feature": feature, "feature_value": value, "sample_count": len(group),
                         "mean_shap": group.shap.mean(), "median_shap": group.shap.median(),
                         "mean_abs_shap": group.shap.abs().mean(), "positive_shap_rate": (group.shap > 0).mean(),
                         "negative_shap_rate": (group.shap < 0).mean()})
    return pd.DataFrame(rows)


def select_local_cases(scores: object, targets: object, identifiers: object, *, cutoff: float) -> dict[str, int]:
    """Select six local roles with score and original-position tie breaking."""
    score = np.asarray(scores, dtype=float); target = np.asarray(targets, dtype=int); ids = np.asarray(identifiers, dtype=str)
    if score.ndim != 1 or len(score) == 0 or target.shape != score.shape or ids.shape != score.shape:
        raise ShapExplainabilityError("Local-case arrays must be non-empty and aligned.")
    predicted = score >= cutoff
    positions = np.arange(len(score))
    def choose(mask: np.ndarray, *, highest: bool) -> int:
        candidates = positions[mask]
        if len(candidates) == 0:
            raise ShapExplainabilityError("A required descriptive confusion category is absent.")
        order = sorted(candidates.tolist(), key=lambda index: ((-score[index] if highest else score[index]), index, ids[index]))
        return int(order[0])
    return {
        "highest_risk": choose(np.ones(len(score), dtype=bool), highest=True),
        "lowest_risk": choose(np.ones(len(score), dtype=bool), highest=False),
        "true_positive": choose((target == 1) & predicted, highest=True),
        "true_negative": choose((target == 0) & ~predicted, highest=False),
        "false_positive": choose((target == 0) & predicted, highest=True),
        "false_negative": choose((target == 1) & ~predicted, highest=False),
    }


def deterministic_sample(row_count: int, *, sample_size: int, seed: int, reserved: tuple[int, ...]) -> tuple[int, ...]:
    """Reserve local cases, seed-sample remaining positions, then restore order."""
    if row_count < sample_size or sample_size <= 0 or any(index < 0 or index >= row_count for index in reserved):
        raise ShapExplainabilityError("Invalid deterministic sample boundary.")
    fixed = tuple(dict.fromkeys(reserved))
    pool = np.setdiff1d(np.arange(row_count), np.asarray(fixed, dtype=int), assume_unique=False)
    chosen = np.random.default_rng(seed).choice(pool, size=sample_size - len(fixed), replace=False)
    return tuple(sorted((*fixed, *(int(value) for value in chosen))))


def interaction_summary(booster: Booster, result: ShapExplanationResult) -> pd.DataFrame:
    """Return all distinct feature-pair mean absolute native Tree SHAP interactions."""
    values = np.asarray(booster.predict(DMatrix(result.feature_values), pred_interactions=True, validate_features=False), dtype=float)
    expected = (len(result.feature_values), len(result.feature_names) + 1, len(result.feature_names) + 1)
    if values.shape != expected or not np.isfinite(values).all():
        raise ShapExplainabilityError("Tree SHAP interaction output is invalid.")
    rows = []
    for left in range(len(result.feature_names)):
        for right in range(left + 1, len(result.feature_names)):
            rows.append({"feature_1": result.feature_names[left], "feature_2": result.feature_names[right],
                         "mean_abs_interaction": float(np.abs(values[:, left, right]).mean())})
    table = pd.DataFrame(rows).sort_values(["mean_abs_interaction", "feature_1", "feature_2"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def stability_summary(result: ShapExplanationResult, *, seed: int, half_size: int, minimum_spearman: float) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare global importance across two seeded, disjoint equal halves."""
    if len(result.feature_values) != 2 * half_size:
        raise ShapExplainabilityError("Stability analysis requires exactly two configured halves.")
    order = np.random.default_rng(seed).permutation(len(result.feature_values))
    means = [np.abs(result.shap_values[order[start:start + half_size]]).mean(axis=0) for start in (0, half_size)]
    ranks = [pd.Series(-values).rank(method="first").astype(int).to_numpy() for values in means]
    table = pd.DataFrame({"feature": result.feature_names, "mean_abs_shap_half_a": means[0], "rank_half_a": ranks[0],
                          "mean_abs_shap_half_b": means[1], "rank_half_b": ranks[1]})
    table["absolute_difference"] = (table.mean_abs_shap_half_a - table.mean_abs_shap_half_b).abs()
    correlation = float(spearmanr(ranks[0], ranks[1]).statistic)
    top_a = set(table.nsmallest(2, "rank_half_a").feature); top_b = set(table.nsmallest(2, "rank_half_b").feature)
    stable = table.loc[table.rank_half_a.eq(1), "feature"].iloc[0] == table.loc[table.rank_half_b.eq(1), "feature"].iloc[0] and top_a == top_b and correlation >= minimum_spearman
    return table, {"spearman_rank_correlation": correlation, "top_feature_matches": bool(table.loc[table.rank_half_a.eq(1), "feature"].iloc[0] == table.loc[table.rank_half_b.eq(1), "feature"].iloc[0]),
                   "top_two_set_matches": top_a == top_b, "minimum_spearman": minimum_spearman, "status": "REASONABLY_STABLE" if stable else "UNSTABLE"}


def correct_vs_incorrect(result: ShapExplanationResult, targets: object, *, cutoff: float) -> pd.DataFrame:
    """Compare feature reliance for descriptively correct and incorrect rows."""
    target = np.asarray(targets, dtype=int)
    if target.shape != result.raw_scores.shape:
        raise ShapExplainabilityError("Correctness targets must align with explanations.")
    correct = (result.raw_scores >= cutoff).astype(int) == target
    if not correct.any() or correct.all():
        raise ShapExplainabilityError("Correct-vs-incorrect comparison requires both groups.")
    correct_mean = np.abs(result.shap_values[correct]).mean(axis=0); incorrect_mean = np.abs(result.shap_values[~correct]).mean(axis=0)
    return pd.DataFrame({"feature": result.feature_names, "mean_abs_shap_correct": correct_mean,
                         "mean_abs_shap_incorrect": incorrect_mean, "difference_incorrect_minus_correct": incorrect_mean - correct_mean})
