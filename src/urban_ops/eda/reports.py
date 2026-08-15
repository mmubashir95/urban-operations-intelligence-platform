"""Write and validate the complete rollback-safe Step 9A report inventory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_ops.eda.figures import REQUIRED_FIGURES, write_eda_figures
from urban_ops.eda.models import SourceSplitEvidence


REQUIRED_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "source_split_verification.csv": ("artifact_name", "artifact_path", "sha256", "expected_sha256", "mtime_ns", "row_count", "expected_row_count", "status", "split_id"),
    "column_inventory.csv": ("column_name", "source_dtype", "train_non_null_count", "train_missing_count", "train_missing_rate", "train_unique_count", "feature_role", "prediction_time_available", "leakage_status", "variance_status", "eda_treatment", "reason"),
    "train_target_distribution.csv": ("on_time_count", "missed_count", "total_count", "on_time_share", "missed_target_rate", "class_ratio_missed_to_on_time", "majority_class", "majority_class_accuracy", "constant_probability_baseline", "class_weighting_recommendation"),
    "train_missingness.csv": ("column_name", "missing_count", "missing_rate", "non_null_count", "missingness_band"),
    "missingness_by_target.csv": ("feature_name", "target_value", "row_count", "missing_count", "missing_rate", "absolute_difference_between_targets"),
    "missingness_by_month.csv": ("feature_name", "month", "row_count", "missing_count", "missing_rate"),
    "monthly_volume.csv": ("month", "row_count", "minimum_created_date", "maximum_created_date", "row_share", "month_to_month_count_change", "month_to_month_count_pct_change", "support_flag"),
    "monthly_target_rate.csv": ("month", "row_count", "missed_count", "on_time_count", "missed_target_rate", "overall_train_rate", "rate_difference_from_train", "month_to_month_rate_change", "support_flag"),
    "day_of_week_target_rate.csv": ("day_order", "created_day_name", "row_count", "missed_count", "on_time_count", "missed_target_rate"),
    "hour_of_day_target_rate.csv": ("created_hour", "row_count", "missed_count", "on_time_count", "missed_target_rate"),
    "quarterly_target_rate.csv": ("created_quarter", "row_count", "missed_count", "on_time_count", "missed_target_rate"),
    "weekend_target_rate.csv": ("is_weekend", "row_count", "missed_count", "on_time_count", "missed_target_rate"),
    "categorical_cardinality.csv": ("feature_name", "unique_count", "missing_count", "missing_rate", "most_common_value", "most_common_count", "most_common_share", "top_5_share", "top_10_share", "singleton_category_count", "cardinality_band"),
    "categorical_feature_profile.csv": ("feature_name", "category_value", "row_count", "row_share", "rank_by_count"),
    "categorical_target_rates.csv": ("feature_name", "category_value", "row_count", "row_share", "on_time_count", "missed_count", "missed_target_rate", "difference_from_train_rate", "minimum_support_passed", "stability_warning"),
    "rare_category_analysis.csv": ("feature_name", "threshold_type", "threshold_value", "rare_category_count", "train_rows_affected", "train_row_share", "validation_rows_affected", "test_rows_affected", "recommendation"),
    "numeric_feature_summary.csv": ("feature_name", "count", "missing_count", "missing_rate", "mean", "standard_deviation", "minimum", "maximum", "q1", "q3", "iqr", "infinite_count", "p01", "p05", "p25", "median", "p75", "p95", "p99"),
    "numeric_summary_by_target.csv": ("feature_name", "target_value", "count", "missing_rate", "mean", "median", "standard_deviation", "q1", "q3", "iqr", "minimum", "maximum"),
    "geographic_outlier_analysis.csv": ("finding", "outlier_category", "row_count", "row_rate", "recommended_action", "governed_bounds"),
    "numeric_outlier_analysis.csv": ("feature_name", "q1", "q3", "iqr", "lower_iqr_fence", "upper_iqr_fence", "below_fence_count", "above_fence_count", "total_iqr_outlier_count", "iqr_outlier_rate", "p01_threshold", "p99_threshold", "below_p01_count", "above_p99_count", "domain_invalid_count", "recommended_action", "reason"),
    "temporal_outlier_analysis.csv": ("metric", "period", "metric_value", "row_count", "classification", "recommended_action"),
    "zero_variance_columns.csv": ("feature_name", "baseline_decision", "decision_reason"),
    "all_null_columns.csv": ("feature_name", "baseline_decision", "decision_reason"),
    "split_target_comparison.csv": ("split_name", "row_count", "on_time_count", "missed_count", "missed_target_rate", "governance_use"),
    "split_missingness_comparison.csv": ("feature_name", "train_missing_rate", "validation_missing_rate", "test_missing_rate", "train_validation_difference", "train_test_difference", "maximum_difference", "interpretation"),
    "unseen_categories.csv": ("feature_name", "comparison_split", "unseen_category_count", "unseen_categories", "rows_affected", "row_share_affected", "recommended_unknown_policy"),
    "numeric_range_drift.csv": ("feature_name", "split_name", "minimum", "maximum", "p01", "median", "p99", "iqr", "missing_rate", "governance_use"),
    "structural_drift_summary.csv": ("area", "feature_name", "comparison", "metric", "train_value", "later_split_value", "absolute_difference", "warning_status", "interpretation", "decision_impact"),
    "leakage_audit.csv": ("column_name", "source_column", "authoritative_role", "prediction_time_available", "allowed_for_baseline", "leakage_status", "decision_status", "decision_reason"),
    "baseline_feature_recommendation.csv": ("feature_name", "source_column", "feature_type", "derivation_rule", "prediction_time_available", "train_missing_rate", "train_unique_count", "cardinality_band", "zero_variance", "all_null", "validation_unseen_count", "validation_unseen_rows", "test_unseen_count", "test_unseen_rows", "outlier_status", "leakage_status", "baseline_decision", "decision_reason", "recommended_missing_policy", "recommended_encoding", "recommended_scaling", "recommended_outlier_policy", "variation_status", "redundancy_status", "temporal_risk", "eda_status", "status_reason"),
    "transformation_recommendation.csv": ("feature_name", "data_type", "recommended_missing_strategy", "recommended_encoding", "recommended_scaling", "recommended_rare_category_policy", "recommended_unknown_category_policy", "recommended_outlier_treatment", "fit_on", "apply_to", "reason"),
    "outlier_recommendation.csv": ("feature_name", "outlier_type", "diagnostic_rule", "lower_threshold", "upper_threshold", "outlier_count", "outlier_rate", "domain_invalid_count", "recommended_action", "decision_reason", "fit_or_derive_on", "apply_to"),
    "eda_integrity_checks.csv": ("check_id", "area", "status", "observed_value", "expected_value", "affected_rows", "message"),
    "output_reconciliation.csv": ("check_name", "left_value", "right_value", "status"),
}


REQUIRED_TABLES = tuple(REQUIRED_TABLE_COLUMNS)


def build_eda_summary(source: SourceSplitEvidence, tables: dict[str, pd.DataFrame]) -> str:
    """Build the evidence-based feature and transformation handoff summary."""
    target = tables["train_target_distribution.csv"].iloc[0]
    volume = tables["monthly_volume.csv"]
    monthly = tables["monthly_target_rate.csv"]
    baseline = tables["baseline_feature_recommendation.csv"]
    included = baseline.loc[baseline["baseline_decision"].eq("INCLUDE"), "feature_name"].tolist()
    conditional = baseline.loc[baseline["baseline_decision"].eq("CONDITIONAL"), "feature_name"].tolist()
    blocked = baseline.loc[baseline["baseline_decision"].eq("EXCLUDE_LEAKAGE"), "feature_name"].tolist()
    all_null = tables["all_null_columns.csv"]["feature_name"].tolist()
    zero = tables["zero_variance_columns.csv"]["feature_name"].tolist()
    unseen = tables["unseen_categories.csv"]
    domain_invalid = int(tables["geographic_outlier_analysis.csv"].loc[
        tables["geographic_outlier_analysis.csv"]["outlier_category"].eq("DOMAIN_INVALID"), "row_count"
    ].sum())
    return f"""# Step 9A Split-Aware EDA Summary

- Source split: `{source.split_id}`
- Train: `{source.artifacts['train'].path}` ({len(source.train):,} rows; SHA-256 `{source.artifacts['train'].sha256}`)
- Validation: `{source.artifacts['validation'].path}` ({len(source.validation):,} rows; SHA-256 `{source.artifacts['validation'].sha256}`)
- Test: `{source.artifacts['test'].path}` ({len(source.test):,} rows; SHA-256 `{source.artifacts['test'].sha256}`)

## Governance boundary

Train is the sole authority for feature discovery, missingness policy,
cardinality, rare-category diagnostics, outlier thresholds, and recommendations.
Validation and test are used only for governed structural disclosure. Test
prevalence is not used for feature or transformation selection. No rows or
values were changed, no preprocessing was fitted, no feature matrix was built,
and no model was trained or evaluated.

## Target and temporal evidence

Training contains {int(target['on_time_count']):,} on-time and
{int(target['missed_count']):,} missed complaints ({target['missed_target_rate']:.2%}
missed). Accuracy alone is insufficient; Notebook 12 must beat both the
majority-class accuracy ({target['majority_class_accuracy']:.2%}) and the
constant-probability baseline ({target['constant_probability_baseline']:.4f}).
Class weighting is an experiment, not automatic.

Training spans {len(volume)} calendar months. Highest volume is
{volume.loc[volume['row_count'].idxmax(), 'month']}; lowest volume is
{volume.loc[volume['row_count'].idxmin(), 'month']}. Monthly missed rates range
from {monthly['missed_target_rate'].min():.2%} to
{monthly['missed_target_rate'].max():.2%}. These are associations and
operational variation, not causal or guaranteed predictive effects.

## Data quality, categories, and outliers

- All-null fields: {', '.join(all_null) if all_null else 'None'}
- Zero-variance fields: {', '.join(zero) if zero else 'None'}
- Later-split unseen-category findings affecting rows: {int(unseen['rows_affected'].gt(0).sum())}
- Domain-invalid coordinate findings: {domain_invalid}
- Statistical numeric outliers: retained and monitored; no clipping or removal
- Temporal operational outliers: retained for investigation

Missingness bands are descriptive and do not automatically approve or exclude
features. `Unspecified` borough and missing categories are preserved. Incident
ZIP remains string-valued and is never treated as continuous or target encoded.

## Baseline and Notebook 11 handoff

- Include: {', '.join(included) if included else 'None'}
- Conditional pending creation-time governance: {', '.join(conditional) if conditional else 'None'}
- Leakage blocked: {', '.join(blocked)}

Notebook 11 may evaluate explicit missing tokens, train-median candidates,
unknown-safe one-hot encoding, train-fitted rare grouping, and scaling for
linear models. Every fitted transformation must be fit on train and applied
unchanged to validation and test. The outputs here are recommendations and
evidence—not model-ready matrices.

## Limitations and completion decision

The source is a later-state operational snapshot, temporal prevalence shifts,
and several geographic/intake fields remain conditional because creation-time
availability and correction behavior are unresolved. Integrity and output
reconciliation pass, so Step 9A analysis is complete within this boundary.
"""


def write_eda_report(
    report_root: Path,
    source: SourceSplitEvidence,
    tables: dict[str, pd.DataFrame],
    train: pd.DataFrame,
) -> tuple[str, ...]:
    """Write all tables, figures, and the summary to a staged report directory."""
    table_root = report_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_TABLES:
        tables[filename].to_csv(table_root / filename, index=False)
    figures = write_eda_figures(report_root / "figures", tables, train)
    (report_root / "eda_summary.md").write_text(
        build_eda_summary(source, tables), encoding="utf-8"
    )
    return figures


def validate_eda_report(
    report_root: Path, source: SourceSplitEvidence
) -> None:
    """Read back required schemas, source evidence, figures, and passing checks."""
    summary_path = report_root / "eda_summary.md"
    if not summary_path.is_file() or not summary_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("EDA summary is missing or empty.")
    summary = summary_path.read_text(encoding="utf-8")
    if source.split_id not in summary:
        raise RuntimeError("EDA summary does not reference the source split ID.")
    for name in ("train", "validation", "test"):
        if source.artifacts[name].sha256 not in summary:
            raise RuntimeError(f"EDA summary does not reference the {name} source hash.")
    loaded = {}
    for filename, expected_columns in REQUIRED_TABLE_COLUMNS.items():
        path = report_root / "tables" / filename
        if not path.is_file():
            raise RuntimeError(f"Required EDA table is missing: {filename}")
        table = pd.read_csv(path)
        if tuple(table.columns) != expected_columns:
            raise RuntimeError(f"EDA table schema is invalid: {filename}")
        loaded[filename] = table
    for figure in REQUIRED_FIGURES:
        path = report_root / "figures" / figure
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required EDA figure is missing or empty: {figure}")
    if loaded["eda_integrity_checks.csv"]["status"].eq("FAIL").any():
        raise RuntimeError("EDA integrity report contains a failure.")
    reconciliation = loaded["output_reconciliation.csv"]
    if reconciliation.empty or not reconciliation["status"].eq("PASS").all():
        raise RuntimeError("EDA output reconciliation contains a failure.")
