"""Evaluate binary baseline classifiers and risk rankings.

This module centralizes metric calculation for Month 1 baselines. It accepts
already produced labels, binary predictions, and risk scores; it does not fit
models, choose features, tune thresholds, or inspect preprocessing internals.

The positive-class contract is fixed: ``1`` means a complaint missed its
resolution target and ``0`` means it resolved on time. Consequently, a false
negative is an actual missed-target complaint that the model failed to flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Final

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


TOP_K_FRACTIONS: Final = (0.05, 0.10, 0.20)
NEGATIVE_LABEL: Final = 0
POSITIVE_LABEL: Final = 1
ZERO_DIVISION: Final = 0
PR_AUC_DEFINITION: Final = "average_precision_score"
CALIBRATION_N_BINS: Final = 10
CALIBRATION_STRATEGY: Final = "uniform"


class EvaluationError(ValueError):
    """Raised when metric inputs are malformed or unsafe to score."""


@dataclass(frozen=True)
class TopKMetrics:
    """Precision and recall for a deterministic highest-risk slice."""

    fraction: float
    selected_count: int
    precision: float
    recall: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class BasicClassificationMetrics:
    """Phase 1 metrics at an already-established binary decision threshold.

    Count meanings use positive label ``1`` (missed target): true positives are
    misses correctly flagged, false positives are on-time complaints flagged as
    misses, false negatives are misses not flagged, and true negatives are
    on-time complaints correctly classified.
    """

    row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, object]:
        """Return a flat JSON-safe Phase 1 metric mapping."""
        return asdict(self)


@dataclass(frozen=True)
class RankingMetrics:
    """Phase 2 ranking summary calculated from continuous missed-risk scores.

    ``pr_auc`` is the project's established name for sklearn Average Precision
    (``average_precision_score``); it is not trapezoidal PR-curve area.
    """

    row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    roc_auc: float
    pr_auc: float

    def to_dict(self) -> dict[str, object]:
        """Return a flat JSON-safe Phase 2 summary mapping."""
        return asdict(self)


@dataclass(frozen=True)
class RankingCurves:
    """Score-derived ROC and precision-recall curve coordinates."""

    roc_false_positive_rate: tuple[float, ...]
    roc_true_positive_rate: tuple[float, ...]
    roc_thresholds: tuple[float, ...]
    pr_precision: tuple[float, ...]
    pr_recall: tuple[float, ...]
    pr_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class RankingEvaluation:
    """Phase 2 ranking summary and curve data for one score vector."""

    metrics: RankingMetrics
    curves: RankingCurves


@dataclass(frozen=True)
class CalibrationMetrics:
    """Phase 3 probability-quality summary for missed-target risk scores."""

    row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    brier_score: float

    def to_dict(self) -> dict[str, object]:
        """Return a flat JSON-safe Phase 3 summary mapping."""
        return asdict(self)


@dataclass(frozen=True)
class CalibrationCurve:
    """Populated uniform-bin calibration coordinates from sklearn."""

    mean_predicted_probability: tuple[float, ...]
    observed_positive_rate: tuple[float, ...]
    requested_bin_count: int
    strategy: str


@dataclass(frozen=True)
class CalibrationEvaluation:
    """Phase 3 scalar metrics and populated calibration-curve points."""

    metrics: CalibrationMetrics
    curve: CalibrationCurve


@dataclass(frozen=True)
class ClassificationMetrics(BasicClassificationMetrics):
    """Backward-compatible Month 1 classification and ranking metrics."""

    roc_auc: float
    pr_auc: float
    brier_score: float
    precision_at_5_percent: float
    recall_at_5_percent: float
    precision_at_10_percent: float
    recall_at_10_percent: float
    precision_at_20_percent: float
    recall_at_20_percent: float


@dataclass(frozen=True)
class CalibrationBin:
    """Observed outcome frequency for one fixed predicted-risk interval."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    row_count: int
    mean_predicted_risk: float
    observed_positive_rate: float
    positive_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return asdict(self)


def _as_1d_array(values: object, *, name: str) -> np.ndarray:
    """Return a one-dimensional numpy array or fail clearly."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise EvaluationError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise EvaluationError(f"{name} must not be empty.")
    return array


def _validate_binary(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate binary class labels/predictions and return integer values."""
    if np.any(pd.isna(values)):
        raise EvaluationError(f"{name} must not contain null values.")
    observed = set(values.tolist())
    if not observed.issubset({0, 1, False, True}):
        raise EvaluationError(f"{name} must contain only 0/1 values.")
    return values.astype(int)


def _validate_scores(values: np.ndarray) -> np.ndarray:
    """Validate finite probability/risk scores in the closed unit interval."""
    scores = values.astype(float)
    if not np.isfinite(scores).all():
        raise EvaluationError("y_score must contain only finite values.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise EvaluationError("y_score values must be in [0, 1].")
    return scores


def _uniform_calibration_curve(
    true: np.ndarray,
    score: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sklearn observed rates and means for populated uniform bins."""
    observed_positive_rate, mean_predicted_probability = calibration_curve(
        true,
        score,
        pos_label=POSITIVE_LABEL,
        n_bins=n_bins,
        strategy=CALIBRATION_STRATEGY,
    )
    return observed_positive_rate, mean_predicted_probability


def _validate_inputs(
    y_true: object, y_pred: object, y_score: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align evaluation arrays."""
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    pred = _validate_binary(_as_1d_array(y_pred, name="y_pred"), name="y_pred")
    score = _validate_scores(_as_1d_array(y_score, name="y_score"))
    if not (len(true) == len(pred) == len(score)):
        raise EvaluationError("y_true, y_pred, and y_score lengths must match.")
    return true, pred, score


def evaluate_basic_classifier(
    y_true: object,
    y_pred: object,
) -> BasicClassificationMetrics:
    """Calculate Phase 1 classification metrics from existing predictions.

    The confusion matrix uses ``labels=[0, 1]`` explicitly, whose sklearn
    layout is ``[[TN, FP], [FN, TP]]``. Precision, recall, and F1 use positive
    label ``1`` and return ``0.0`` when their denominator is zero. Single-class
    actual targets are permitted and therefore remain deterministic.
    """
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    pred = _validate_binary(_as_1d_array(y_pred, name="y_pred"), name="y_pred")
    if len(true) != len(pred):
        raise EvaluationError("y_true and y_pred lengths must match.")

    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(
            true,
            pred,
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ).ravel()
    )
    positive_count = int(true.sum())
    row_count = len(true)
    negative_count = int(row_count - positive_count)
    return BasicClassificationMetrics(
        row_count=row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=float(positive_count / row_count),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        accuracy=float(accuracy_score(true, pred)),
        precision=float(
            precision_score(
                true,
                pred,
                pos_label=POSITIVE_LABEL,
                zero_division=ZERO_DIVISION,
            )
        ),
        recall=float(
            recall_score(
                true,
                pred,
                pos_label=POSITIVE_LABEL,
                zero_division=ZERO_DIVISION,
            )
        ),
        f1=float(
            f1_score(
                true,
                pred,
                pos_label=POSITIVE_LABEL,
                zero_division=ZERO_DIVISION,
            )
        ),
    )


def evaluate_ranking(
    y_true: object,
    y_score: object,
) -> RankingEvaluation:
    """Calculate Phase 2 ranking metrics and curves from existing risk scores.

    Higher scores must mean greater risk of positive class ``1`` (missed
    target). Both target classes are required because ROC ranking is undefined
    for a single-class population. This function does not create predictions,
    choose a threshold, train a model, or modify the supplied scores.
    """
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    score = _validate_scores(_as_1d_array(y_score, name="y_score"))
    if len(true) != len(score):
        raise EvaluationError("y_true and y_score lengths must match.")
    if set(true.tolist()) != {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise EvaluationError(
            "Ranking evaluation requires y_true to contain both 0 and 1."
        )

    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        true,
        score,
        pos_label=POSITIVE_LABEL,
    )
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        true,
        score,
        pos_label=POSITIVE_LABEL,
    )
    positive_count = int(true.sum())
    row_count = len(true)
    metrics = RankingMetrics(
        row_count=row_count,
        positive_count=positive_count,
        negative_count=int(row_count - positive_count),
        positive_rate=float(positive_count / row_count),
        roc_auc=float(roc_auc_score(true, score)),
        pr_auc=float(average_precision_score(true, score)),
    )
    curves = RankingCurves(
        roc_false_positive_rate=tuple(float(value) for value in false_positive_rate),
        roc_true_positive_rate=tuple(float(value) for value in true_positive_rate),
        roc_thresholds=tuple(float(value) for value in roc_thresholds),
        pr_precision=tuple(float(value) for value in pr_precision),
        pr_recall=tuple(float(value) for value in pr_recall),
        pr_thresholds=tuple(float(value) for value in pr_thresholds),
    )
    return RankingEvaluation(metrics=metrics, curves=curves)


def evaluate_calibration(
    y_true: object,
    y_score: object,
) -> CalibrationEvaluation:
    """Evaluate existing positive-class probabilities without changing them.

    Scores must be finite probabilities in ``[0, 1]`` where larger values mean
    greater risk of missed target (class ``1``). The curve uses ten uniform bins
    and sklearn omits bins with no observations. Single-class targets are
    accepted because Brier loss and observed-frequency estimates remain defined.
    No estimator fitting, recalibration, or threshold operation occurs here.
    """
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    score = _validate_scores(_as_1d_array(y_score, name="y_score"))
    if len(true) != len(score):
        raise EvaluationError("y_true and y_score lengths must match.")

    observed_positive_rate, mean_predicted_probability = _uniform_calibration_curve(
        true,
        score,
        n_bins=CALIBRATION_N_BINS,
    )
    positive_count = int(true.sum())
    row_count = len(true)
    metrics = CalibrationMetrics(
        row_count=row_count,
        positive_count=positive_count,
        negative_count=int(row_count - positive_count),
        positive_rate=float(positive_count / row_count),
        brier_score=float(brier_score_loss(true, score, pos_label=POSITIVE_LABEL)),
    )
    curve = CalibrationCurve(
        mean_predicted_probability=tuple(
            float(value) for value in mean_predicted_probability
        ),
        observed_positive_rate=tuple(
            float(value) for value in observed_positive_rate
        ),
        requested_bin_count=CALIBRATION_N_BINS,
        strategy=CALIBRATION_STRATEGY,
    )
    return CalibrationEvaluation(metrics=metrics, curve=curve)


def top_k_metrics(
    y_true: object, y_score: object, *, fraction: float
) -> TopKMetrics:
    """Compute metrics for the highest-risk fraction of rows.

    The selected row count is `max(1, ceil(n * fraction))`. Ties are resolved by
    original row order after sorting by score descending, using stable mergesort.
    """
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    score = _validate_scores(_as_1d_array(y_score, name="y_score"))
    if len(true) != len(score):
        raise EvaluationError("y_true and y_score lengths must match.")
    if not 0.0 < fraction <= 1.0:
        raise EvaluationError("fraction must be in the interval (0, 1].")
    selected_count = max(1, int(ceil(len(true) * fraction)))
    order = np.argsort(-score, kind="mergesort")
    selected = order[:selected_count]
    selected_true = true[selected]
    selected_positive = int(selected_true.sum())
    total_positive = int(true.sum())
    precision = float(selected_positive / selected_count)
    recall = float(selected_positive / total_positive) if total_positive else 0.0
    return TopKMetrics(
        fraction=float(fraction),
        selected_count=selected_count,
        precision=precision,
        recall=recall,
    )


def evaluate_binary_classifier(
    y_true: object,
    y_pred: object,
    y_score: object,
) -> ClassificationMetrics:
    """Calculate the legacy Month 1 metric set, reusing Phase 1 metrics."""
    true, pred, score = _validate_inputs(y_true, y_pred, y_score)
    basic = evaluate_basic_classifier(true, pred)
    if basic.positive_count == 0 or basic.negative_count == 0:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        ranking = evaluate_ranking(true, score)
        roc_auc = ranking.metrics.roc_auc
        pr_auc = ranking.metrics.pr_auc
    calibration = evaluate_calibration(true, score)
    top_k = {
        fraction: top_k_metrics(true, score, fraction=fraction)
        for fraction in TOP_K_FRACTIONS
    }
    return ClassificationMetrics(
        **basic.to_dict(),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=calibration.metrics.brier_score,
        precision_at_5_percent=top_k[0.05].precision,
        recall_at_5_percent=top_k[0.05].recall,
        precision_at_10_percent=top_k[0.10].precision,
        recall_at_10_percent=top_k[0.10].recall,
        precision_at_20_percent=top_k[0.20].precision,
        recall_at_20_percent=top_k[0.20].recall,
    )


def metrics_row(
    model_name: str,
    metrics: BasicClassificationMetrics,
    *,
    evaluated_split: str | None = None,
) -> dict[str, object]:
    """Return the standard report row for one evaluated model and split."""
    row = metrics.to_dict()
    identity: dict[str, object] = {"model": model_name}
    if evaluated_split is not None:
        identity["evaluated_split"] = evaluated_split
    return {**identity, **row}


def build_calibration_table(
    y_true: object,
    y_score: object,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predicted risks and compare them with observed positive rates.

    Bins are deterministic, equal-width intervals over `[0, 1]`. The first bin
    includes zero and all bins include their right edge, so a score of `1.0`
    belongs to the final bin. Empty bins are retained with zero row count and
    NaN rate fields to keep a stable report shape.
    """
    true = _validate_binary(_as_1d_array(y_true, name="y_true"), name="y_true")
    score = _validate_scores(_as_1d_array(y_score, name="y_score"))
    if len(true) != len(score):
        raise EvaluationError("y_true and y_score lengths must match.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise EvaluationError("n_bins must be an integer greater than one.")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_indexes = np.digitize(score, edges[1:-1], right=True)
    observed_rates, predicted_means = _uniform_calibration_curve(
        true,
        score,
        n_bins=n_bins,
    )
    populated_index = 0
    rows: list[CalibrationBin] = []
    for bin_index in range(int(n_bins)):
        mask = bin_indexes == bin_index
        row_count = int(mask.sum())
        positives = int(true[mask].sum()) if row_count else 0
        mean_predicted_risk = (
            float(predicted_means[populated_index])
            if row_count
            else float("nan")
        )
        observed_positive_rate = (
            float(observed_rates[populated_index])
            if row_count
            else float("nan")
        )
        if row_count:
            populated_index += 1
        rows.append(
            CalibrationBin(
                bin_index=bin_index,
                lower_bound=float(edges[bin_index]),
                upper_bound=float(edges[bin_index + 1]),
                row_count=row_count,
                mean_predicted_risk=mean_predicted_risk,
                observed_positive_rate=observed_positive_rate,
                positive_count=positives,
            )
        )
    return pd.DataFrame([row.to_dict() for row in rows])
