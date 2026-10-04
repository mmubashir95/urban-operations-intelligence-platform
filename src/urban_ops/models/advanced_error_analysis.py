"""Pure, deterministic Phase 6 advanced error-analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from urban_ops.utils.paths import PROJECT_ROOT

CONFIG_PATH: Final = PROJECT_ROOT / "configs/models/resolution_risk_advanced_error_analysis.yaml"


class AdvancedErrorAnalysisError(ValueError):
    """Raised when Phase 6 input or configuration violates its frozen contract."""


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    analysis_split: str
    expected_split_id: str
    expected_model: str
    capacity: float
    descriptive_threshold: float
    minimum_sample_count: int
    minimum_positive_count: int
    zip_top_n: int
    score_quantiles: int
    borderline_fn_capacity: float
    severe_fn_bottom_quantile: float
    selected_rank_quartile: float


def load_error_analysis_config(path: Path | str = CONFIG_PATH) -> ErrorAnalysisConfig:
    """Load the strict, bounded Phase 6 policy."""
    root = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"version", "analysis_split", "expected_split_id", "expected_model", "primary_decision",
                "secondary_decision", "minimum_support", "geography", "score_bins", "severity"}
    if not isinstance(root, dict) or set(root) != expected or root["version"] != 1:
        raise AdvancedErrorAnalysisError("Unsupported Phase 6 configuration schema.")
    primary, secondary = root["primary_decision"], root["secondary_decision"]
    if primary != {"kind": "top_k", "capacity": primary.get("capacity"), "score": "raw_model_score",
                   "tie_break": "stable_original_row_order"}:
        raise AdvancedErrorAnalysisError("Primary decision must be stable raw-score Top-K.")
    if secondary != {"kind": "descriptive_threshold", "threshold": secondary.get("threshold"), "score": "raw_model_score"}:
        raise AdvancedErrorAnalysisError("Secondary decision must be a raw-score descriptive threshold.")
    support, geography, bins, severity = root["minimum_support"], root["geography"], root["score_bins"], root["severity"]
    config = ErrorAnalysisConfig(
        str(root["analysis_split"]), str(root["expected_split_id"]), str(root["expected_model"]),
        float(primary["capacity"]), float(secondary["threshold"]), int(support["sample_count"]),
        int(support["positive_count"]), int(geography["zip_top_n"]), int(bins["quantiles"]),
        float(severity["borderline_fn_capacity"]), float(severity["severe_fn_bottom_quantile"]),
        float(severity["selected_rank_quartile"]),
    )
    if (config.analysis_split != "test" or not 0 < config.capacity <= 1 or not 0 <= config.descriptive_threshold <= 1
        or config.minimum_sample_count < 1 or config.minimum_positive_count < 1 or config.zip_top_n < 1
        or config.score_quantiles < 2 or not 0 < config.borderline_fn_capacity <= 1
        or not 0 < config.severe_fn_bottom_quantile < 1 or not 0 < config.selected_rank_quartile <= .5):
        raise AdvancedErrorAnalysisError("Phase 6 configuration values are outside supported bounds.")
    return config


def classify_errors(actual: object, predicted: object) -> np.ndarray:
    """Return TP/TN/FP/FN labels for aligned binary arrays."""
    y, p = np.asarray(actual), np.asarray(predicted)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p) or not set(y.tolist()) <= {0, 1} or not set(p.tolist()) <= {0, 1}:
        raise AdvancedErrorAnalysisError("Actual and predicted values must be aligned non-empty binary vectors.")
    return np.select([(y == 1) & (p == 1), (y == 0) & (p == 0), (y == 0) & (p == 1)], ["TP", "TN", "FP"], default="FN")


def top_k_predictions(scores: object, capacity: float) -> tuple[np.ndarray, np.ndarray]:
    """Rank descending with stable original-order ties and select ceil(n*capacity)."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or not 0 < capacity <= 1:
        raise AdvancedErrorAnalysisError("Top-K requires finite scores and capacity in (0, 1].")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int); ranks[order] = np.arange(1, len(values) + 1)
    return (ranks <= ceil(len(values) * capacity)).astype(int), ranks


def confusion_summary(actual: object, predicted: object, *, decision_rule: str) -> pd.DataFrame:
    """Build a one-row confusion and rate summary with safe denominators."""
    labels = classify_errors(actual, predicted); counts = {name: int(np.sum(labels == name)) for name in ("TP", "TN", "FP", "FN")}
    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")
    return pd.DataFrame([{"decision_rule": decision_rule, "sample_count": len(labels), **counts,
        "precision": rate(counts["TP"], counts["TP"] + counts["FP"]),
        "recall": rate(counts["TP"], counts["TP"] + counts["FN"]),
        "specificity": rate(counts["TN"], counts["TN"] + counts["FP"]),
        "false_positive_rate": rate(counts["FP"], counts["FP"] + counts["TN"]),
        "false_negative_rate": rate(counts["FN"], counts["FN"] + counts["TP"])}])


def _metric_record(group: pd.DataFrame, *, minimum_sample_count: int, minimum_positive_count: int) -> dict[str, object]:
    y = group["actual_target"].to_numpy(dtype=int); p = group["operational_selection"].to_numpy(dtype=int)
    base = confusion_summary(y, p, decision_rule="top_10_percent_raw").iloc[0].to_dict()
    positives = int(y.sum()); supported = len(group) >= minimum_sample_count and positives >= minimum_positive_count
    both = len(np.unique(y)) == 2 and supported
    base.update({"actual_positive_count": positives, "target_rate": float(y.mean()),
                 "mean_raw_model_score": float(group.raw_model_score.mean()),
                 "mean_calibrated_score": float(group.calibrated_score.mean()),
                 "support_status": "SUPPORTED" if supported else "LOW SUPPORT",
                 "pr_auc": float(average_precision_score(y, group.raw_model_score)) if both else np.nan,
                 "roc_auc": float(roc_auc_score(y, group.raw_model_score)) if both else np.nan,
                 "auc_status": "AVAILABLE" if both else "UNAVAILABLE_INSUFFICIENT_SUPPORT_OR_SINGLE_CLASS"})
    return base


def group_metrics(table: pd.DataFrame, column: str, *, minimum_sample_count: int, minimum_positive_count: int) -> pd.DataFrame:
    """Compute stable subgroup metrics and preserve low-support rows."""
    if column not in table:
        return pd.DataFrame()
    rows = []
    values = table[column].astype("string").fillna("MISSING")
    for value in sorted(values.unique().tolist()):
        record = {column: value}; record.update(_metric_record(table.loc[values.eq(value)], minimum_sample_count=minimum_sample_count,
                                                               minimum_positive_count=minimum_positive_count)); rows.append(record)
    return pd.DataFrame(rows)


def add_score_bins(table: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Add duplicate-safe quantile score ranges with explicit observed edges."""
    result = table.copy()
    probabilities = np.linspace(0, 1, quantiles + 1)
    edges = np.unique(np.quantile(result.raw_model_score.to_numpy(dtype=float), probabilities))
    if len(edges) < 2:
        result["score_range"] = "ALL_SCORES_EQUAL"; return result
    edges[0] = -np.inf; edges[-1] = np.inf
    result["score_range"] = pd.cut(result.raw_model_score, bins=edges, include_lowest=True, duplicates="drop").astype("string")
    return result


def score_range_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize target and primary error counts for each populated score bin."""
    rows = []
    for label, group in table.groupby("score_range", sort=False, observed=True):
        counts = group.primary_error_type.value_counts()
        rows.append({"score_range": label, "score_min": float(group.raw_model_score.min()), "score_max": float(group.raw_model_score.max()),
                     "sample_count": len(group), "actual_positive_rate": float(group.actual_target.mean()),
                     **{name: int(counts.get(name, 0)) for name in ("TP", "TN", "FP", "FN")}})
    return pd.DataFrame(rows)


def select_severity_cases(table: pd.DataFrame, config: ErrorAnalysisConfig) -> pd.DataFrame:
    """Select deterministic borderline and high-severity primary errors."""
    result = table.loc[table.primary_error_type.isin(["FP", "FN"])].copy(); result["error_severity"] = "OTHER_ERROR"
    selected_count = int(table.operational_selection.sum()); below_count = ceil(len(table) * config.borderline_fn_capacity)
    bottom_start = len(table) - ceil(len(table) * config.severe_fn_bottom_quantile) + 1
    result.loc[result.primary_error_type.eq("FN") & result.risk_rank.between(selected_count + 1, selected_count + below_count), "error_severity"] = "BORDERLINE_FN"
    result.loc[result.primary_error_type.eq("FN") & result.risk_rank.ge(bottom_start), "error_severity"] = "SEVERE_FN"
    quartile_count = max(1, ceil(selected_count * config.selected_rank_quartile))
    result.loc[result.primary_error_type.eq("FP") & result.risk_rank.le(quartile_count), "error_severity"] = "HIGH_CONFIDENCE_FP"
    result.loc[result.primary_error_type.eq("FP") & result.risk_rank.gt(selected_count - quartile_count), "error_severity"] = "BORDERLINE_FP"
    return result.sort_values(["error_severity", "risk_rank"], kind="stable")


def canonical_error_table(frame: pd.DataFrame, *, identifiers: object, targets: object, raw_scores: object,
                          calibrated_scores: object, config: ErrorAnalysisConfig) -> pd.DataFrame:
    """Create the aligned, non-sensitive TEST error-analysis dataset."""
    y = np.asarray(targets, dtype=int); raw = np.asarray(raw_scores, dtype=float); calibrated = np.asarray(calibrated_scores, dtype=float)
    if not (len(frame) == len(y) == len(raw) == len(calibrated)):
        raise AdvancedErrorAnalysisError("Canonical error inputs are not row aligned.")
    selected, ranks = top_k_predictions(raw, config.capacity); descriptive = (raw >= config.descriptive_threshold).astype(int)
    created = pd.to_datetime(frame["created_date"], utc=True)
    table = pd.DataFrame({"record_identifier": pd.Series(identifiers, dtype="string"), "actual_target": y,
        "raw_model_score": raw, "calibrated_score": calibrated, "risk_rank": ranks,
        "operational_selection": selected, "primary_error_type": classify_errors(y, selected),
        "descriptive_prediction_at_0_5": descriptive, "descriptive_error_type_at_0_5": classify_errors(y, descriptive),
        "created_date": created, "created_hour": created.dt.hour, "created_day_of_week": created.dt.dayofweek,
        "created_month": created.dt.month, "created_period": created.dt.strftime("%Y-%m"),
        "is_weekend": created.dt.dayofweek.ge(5).astype(int)})
    for column in ("agency", "complaint_type", "borough", "incident_zip", "location_type"):
        if column in frame:
            table[column] = frame[column].astype("string").fillna("MISSING").reset_index(drop=True)
    return add_score_bins(table, config.score_quantiles)


def aggregate_zip(table: pd.DataFrame, top_n: int) -> pd.Series:
    """Return deterministic top-volume ZIP, OTHER, and MISSING labels."""
    values = table["incident_zip"].astype("string").fillna("MISSING")
    nonmissing = values.loc[~values.eq("MISSING")]
    counts = nonmissing.value_counts().rename_axis("zip").reset_index(name="count").sort_values(["count", "zip"], ascending=[False, True], kind="stable")
    keep = set(counts.head(top_n).zip.tolist())
    return values.map(lambda value: value if value == "MISSING" or value in keep else "OTHER").astype("string")


def correct_vs_incorrect(table: pd.DataFrame) -> pd.DataFrame:
    """Compare broad primary correctness groups."""
    working = table.assign(prediction_result=np.where(table.primary_error_type.isin(["TP", "TN"]), "CORRECT", "INCORRECT"))
    rows = []
    for label, group in working.groupby("prediction_result", sort=True):
        rows.append({"prediction_result": label, "sample_count": len(group), "actual_positive_count": int(group.actual_target.sum()),
                     "target_rate": float(group.actual_target.mean()), "mean_raw_model_score": float(group.raw_model_score.mean()),
                     "median_raw_model_score": float(group.raw_model_score.median()), "mean_calibrated_score": float(group.calibrated_score.mean()),
                     "mean_created_hour": float(group.created_hour.mean()), "weekend_share": float(group.is_weekend.mean())})
    return pd.DataFrame(rows)


def failure_modes(table: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    """Generate only deterministic, evidence-backed diagnostic failure modes."""
    rows = []
    severe = table.primary_error_type.eq("FN") & table.error_severity.eq("SEVERE_FN")
    if severe.any():
        rows.append(("Very-low-score false negatives", int(severe.sum()), "Actual misses occur in the bottom raw-score decile.", "Inspect missing prediction-time signal."))
    high_fp = table.primary_error_type.eq("FP") & table.error_severity.eq("HIGH_CONFIDENCE_FP")
    if high_fp.any():
        rows.append(("High-ranked false positives", int(high_fp.sum()), "On-time cases occur in the highest selected-rank quartile.", "Inspect temporal patterns and omitted safe metadata."))
    supported = geography.loc[geography.support_status.eq("SUPPORTED")] if not geography.empty else geography
    if len(supported) > 1 and supported.false_negative_rate.max() - supported.false_negative_rate.min() >= .10:
        rows.append(("Geographic subgroup weakness", int(supported.loc[supported.false_negative_rate.idxmax(), "FN"]),
                     "Supported geography groups differ in false-negative rate by at least 0.10.", "Evaluate geography as a governed future feature candidate."))
    periods = table.groupby("created_period", sort=True).agg(selected=("operational_selection", "sum"), false_negatives=("primary_error_type", lambda values: int(values.eq("FN").sum())))
    if len(periods) > 1 and periods.selected.eq(0).any() and periods.selected.gt(0).any():
        affected = int(periods.loc[periods.selected.eq(0), "false_negatives"].sum())
        rows.append(("Temporal selection concentration", affected,
                     "The global Top-10% policy selects no records in one or more TEST months while selecting records in another.",
                     "Investigate temporal stability and month-feature dependence under a future governed iteration."))
    if not rows:
        return pd.DataFrame(columns=["failure_mode", "sample_count", "severity", "evidence", "likely_next_investigation"])
    output = []
    for name, count, evidence, investigation in rows:
        severity = "HIGH" if count >= 100 else "MEDIUM" if count >= 25 else "LOW"
        output.append({"failure_mode": name, "sample_count": count, "severity": severity, "evidence": evidence, "likely_next_investigation": investigation})
    return pd.DataFrame(output).sort_values(["severity", "sample_count"], ascending=[True, False], kind="stable")
