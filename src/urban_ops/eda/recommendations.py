"""Leakage-aware feature, transformation, and outlier recommendations for Step 9A."""

from __future__ import annotations

import pandas as pd

from urban_ops.eda.models import EDAConfig, SourceSplitEvidence
from urban_ops.features.leakage import build_leakage_audit


DERIVED_TEMPORAL_SOURCE = "created_date"

# Exact human-readable duplicates of a preferred machine-friendly temporal field.
TEMPORAL_ALTERNATIVE_REPRESENTATIONS: dict[str, str] = {
    "created_day_name": "created_day_of_week",
    "created_month_name": "created_month",
}

# Overlapping or coarser seasonal representations of preferred temporal candidates.
TEMPORAL_REVIEW_REDUNDANCY: dict[str, str] = {
    "created_week_of_year": (
        "Seasonal representation that overlaps strongly with month and quarter."
    ),
    "created_quarter": (
        "Coarser seasonal representation that overlaps with created_month."
    ),
}

TEMPORAL_CONDITIONAL_RISK: dict[str, str] = {
    "created_year": (
        "Technically available at creation time but may encode temporal drift "
        "and fail to generalize to future years."
    ),
}

TEMPORAL_REVIEW_REASONS: dict[str, str] = {
    "created_date": (
        "Raw timestamp is not directly model-ready and may encode long-term "
        "time progression."
    ),
    "created_day_of_month": (
        "Creation-time safe but operational usefulness is uncertain and may "
        "be weak."
    ),
}

TEMPORAL_CANDIDATE_REASONS: dict[str, str] = {
    "created_hour": (
        "Creation-time safe, valid, interpretable, and useful for testing "
        "time-of-day patterns."
    ),
    "created_day_of_week": (
        "Creation-time safe and the preferred machine-friendly weekday "
        "representation."
    ),
    "created_month": "Preferred interpretable seasonal candidate.",
    "is_weekend": (
        "Simple, creation-time-safe operational grouping; partially overlaps "
        "with day of week but remains interpretable."
    ),
}


def assign_temporal_eda_status(
    feature_name: str,
    *,
    prediction_time_available: bool,
    leakage_status: str,
    all_null: bool,
    zero_variance: bool,
    is_temporal_feature: bool,
) -> dict[str, str]:
    """Assign deterministic EDA-only temporal feature statuses.

    These statuses describe eligibility and risk for later testing. They do not
    replace ``baseline_decision`` and are not final model-selection decisions.
    Priority: exclusion → alternative representation → redundancy →
    conditional risk → candidate → review.
    """
    variation_status = (
        "ALL_NULL" if all_null else "ZERO_VARIANCE" if zero_variance else "HAS_VARIATION"
    )

    if not is_temporal_feature:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NOT_ASSESSED",
            "temporal_risk": "NOT_APPLICABLE",
            "eda_status": "NOT_TEMPORAL",
            "status_reason": "EDA status framework currently applies to created_date temporal features.",
        }

    if leakage_status == "BLOCKED" or prediction_time_available is False:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NOT_APPLICABLE",
            "temporal_risk": "LEAKAGE_OR_UNAVAILABLE",
            "eda_status": "EXCLUDE",
            "status_reason": (
                "Feature is unavailable at prediction time or blocked for leakage."
            ),
        }
    if all_null or zero_variance:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NOT_APPLICABLE",
            "temporal_risk": "NONE",
            "eda_status": "EXCLUDE",
            "status_reason": (
                "Feature is all-null or has zero variance in the training split."
            ),
        }
    if feature_name in TEMPORAL_ALTERNATIVE_REPRESENTATIONS:
        preferred = TEMPORAL_ALTERNATIVE_REPRESENTATIONS[feature_name]
        return {
            "variation_status": variation_status,
            "redundancy_status": f"EXACT_ALTERNATIVE_OF:{preferred}",
            "temporal_risk": "NONE",
            "eda_status": "ALTERNATIVE_REPRESENTATION",
            "status_reason": (
                f"Exact human-readable representation of {preferred}; normally "
                "choose one, not both."
            ),
        }
    if feature_name in TEMPORAL_REVIEW_REDUNDANCY:
        return {
            "variation_status": variation_status,
            "redundancy_status": "OVERLAPPING_SEASONAL",
            "temporal_risk": "NONE",
            "eda_status": "REVIEW_REDUNDANCY",
            "status_reason": TEMPORAL_REVIEW_REDUNDANCY[feature_name],
        }
    if feature_name in TEMPORAL_CONDITIONAL_RISK:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NONE",
            "temporal_risk": "TEMPORAL_GENERALIZATION",
            "eda_status": "CONDITIONAL",
            "status_reason": TEMPORAL_CONDITIONAL_RISK[feature_name],
        }
    if feature_name in TEMPORAL_CANDIDATE_REASONS:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NONE",
            "temporal_risk": "NONE",
            "eda_status": "CANDIDATE",
            "status_reason": TEMPORAL_CANDIDATE_REASONS[feature_name],
        }
    if feature_name in TEMPORAL_REVIEW_REASONS:
        return {
            "variation_status": variation_status,
            "redundancy_status": "NONE",
            "temporal_risk": (
                "LONG_TERM_PROGRESSION" if feature_name == DERIVED_TEMPORAL_SOURCE else "NONE"
            ),
            "eda_status": "REVIEW",
            "status_reason": TEMPORAL_REVIEW_REASONS[feature_name],
        }
    return {
        "variation_status": variation_status,
        "redundancy_status": "NONE",
        "temporal_risk": "UNRESOLVED",
        "eda_status": "REVIEW",
        "status_reason": "Evidence is not strong enough for a candidate recommendation.",
    }


def build_leakage_table(columns: tuple[str, ...], derived: tuple[str, ...]) -> pd.DataFrame:
    """Reuse Step 4 roles and append explicitly safe created-time derivations."""
    raw_columns = tuple(column for column in columns if column not in derived)
    authoritative = build_leakage_audit(raw_columns)
    rows = [{
        "column_name": row["column_name"],
        "source_column": row.get("source_column", row["column_name"]),
        "authoritative_role": row.get("role", "UNKNOWN"),
        "prediction_time_available": row.get("available_at_creation"),
        "allowed_for_baseline": bool(row.get("allowed_for_baseline", False)),
        "leakage_status": row.get("leakage_status", "BLOCKED"),
        "decision_status": row.get("decision_status", "UNKNOWN"),
        "decision_reason": row.get("decision_reason", "Unknown field is blocked."),
    } for row in authoritative.to_dict("records")]
    rows.extend({
        "column_name": feature, "source_column": DERIVED_TEMPORAL_SOURCE,
        "authoritative_role": "SAFE_FEATURE", "prediction_time_available": True,
        "allowed_for_baseline": True, "leakage_status": "SAFE",
        "decision_status": "APPROVED",
        "decision_reason": "Deterministically derived from created_date available at prediction time.",
    } for feature in derived)
    return pd.DataFrame(rows)


def _cardinality_band(unique_count: int) -> str:
    """Return the shared descriptive cardinality band."""
    if unique_count <= 1:
        return "ZERO_VARIANCE"
    if unique_count <= 15:
        return "LOW"
    if unique_count <= 50:
        return "MODERATE"
    return "HIGH"


def build_column_inventory(
    train: pd.DataFrame, config: EDAConfig, leakage: pd.DataFrame
) -> pd.DataFrame:
    """Inventory source and temporary features with train-only evidence and roles."""
    leakage_by_name = leakage.set_index("column_name")
    rows = []
    for column in train.columns:
        series = train[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        governance = leakage_by_name.loc[column]
        all_null = missing == len(train)
        zero_variance = unique <= 1 and not all_null
        if governance["leakage_status"] == "BLOCKED":
            role = "BLOCKED_LEAKAGE"
        elif all_null:
            role = "ALL_NULL"
        elif zero_variance:
            role = "ZERO_VARIANCE"
        elif governance["leakage_status"] == "SAFE":
            role = "SAFE_CANDIDATE"
        elif governance["leakage_status"] == "CONDITIONAL":
            role = "CONDITIONAL_CANDIDATE"
        else:
            role = "AUDIT_ONLY"
        if column == config.identifier_column:
            role = "IDENTIFIER"
        elif column == config.target_column:
            role = "TARGET"
        rows.append({
            "column_name": column, "source_dtype": str(series.dtype),
            "train_non_null_count": len(train) - missing,
            "train_missing_count": missing, "train_missing_rate": missing / len(train),
            "train_unique_count": unique, "feature_role": role,
            "prediction_time_available": governance["prediction_time_available"],
            "leakage_status": governance["leakage_status"],
            "variance_status": "ALL_NULL" if all_null else "ZERO_VARIANCE" if zero_variance else "HAS_VARIATION",
            "eda_treatment": "DESCRIBE_ONLY" if role in {"IDENTIFIER", "TARGET", "BLOCKED_LEAKAGE", "AUDIT_ONLY"} else "PROFILE_FOR_RECOMMENDATION",
            "reason": governance["decision_reason"],
        })
    return pd.DataFrame(rows)


def _unseen_lookup(unseen: pd.DataFrame, feature: str, split: str, field: str) -> int:
    """Read one unseen-category metric, returning zero for non-categorical features."""
    match = unseen.loc[
        unseen["feature_name"].eq(feature) & unseen["comparison_split"].eq(split), field
    ]
    return int(match.iloc[0]) if len(match) else 0


def build_recommendation_tables(
    source: SourceSplitEvidence,
    train: pd.DataFrame,
    config: EDAConfig,
    inventory: pd.DataFrame,
    leakage: pd.DataFrame,
    analysis_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply deterministic train-governed recommendation priorities."""
    leakage_by_name = leakage.set_index("column_name")
    unseen = analysis_tables["unseen_categories.csv"]
    numeric_outliers = analysis_tables["numeric_outlier_analysis.csv"].set_index("feature_name")
    categorical = set(config.categorical_features) | {"created_day_name", "created_month_name"}
    numeric = set(config.numeric_features) | {
        "created_hour", "created_day_of_week", "created_day_of_month",
        "created_week_of_year", "created_month", "created_quarter", "created_year",
        "is_weekend",
    }
    baseline_rows = []
    for record in inventory.to_dict("records"):
        feature = record["column_name"]
        governance = leakage_by_name.loc[feature]
        all_null = record["variance_status"] == "ALL_NULL"
        zero_variance = record["variance_status"] == "ZERO_VARIANCE"
        if governance["leakage_status"] == "BLOCKED":
            decision = "EXCLUDE_LEAKAGE"
        elif all_null:
            decision = "EXCLUDE_ALL_NULL"
        elif zero_variance:
            decision = "EXCLUDE_ZERO_VARIANCE"
        elif feature == config.timestamp_column:
            decision = "REVIEW"
        elif governance["leakage_status"] == "CONDITIONAL":
            decision = "CONDITIONAL"
        elif governance["leakage_status"] == "SAFE":
            decision = "INCLUDE"
        else:
            decision = "REVIEW"
        feature_type = "categorical" if feature in categorical else "numeric" if feature in numeric else "audit"
        is_numeric = feature_type == "numeric"
        missing_policy = (
            "NONE" if record["train_missing_rate"] == 0
            else "TRAIN_MEDIAN_CANDIDATE" if is_numeric else "EXPLICIT___MISSING___CANDIDATE"
        )
        encoding = (
            "TRAIN_ONE_HOT_UNKNOWN_SAFE" if feature_type == "categorical" else "NONE"
        )
        scaling = "TRAIN_FITTED_FOR_LINEAR_MODELS" if is_numeric else "NONE"
        outlier_status = (
            "IQR_DIAGNOSTIC_AVAILABLE" if feature in numeric_outliers.index else "NOT_APPLICABLE"
        )
        is_temporal_feature = (
            feature == config.timestamp_column
            or feature in config.derived_temporal_features
        )
        eda_assessment = assign_temporal_eda_status(
            feature,
            prediction_time_available=bool(governance["prediction_time_available"]),
            leakage_status=str(governance["leakage_status"]),
            all_null=all_null,
            zero_variance=zero_variance,
            is_temporal_feature=is_temporal_feature,
        )
        baseline_rows.append({
            "feature_name": feature,
            "source_column": DERIVED_TEMPORAL_SOURCE if feature in config.derived_temporal_features else feature,
            "feature_type": feature_type,
            "derivation_rule": f"derive from {DERIVED_TEMPORAL_SOURCE}" if feature in config.derived_temporal_features else "source value unchanged",
            "prediction_time_available": governance["prediction_time_available"],
            "train_missing_rate": record["train_missing_rate"],
            "train_unique_count": record["train_unique_count"],
            "cardinality_band": _cardinality_band(record["train_unique_count"]),
            "zero_variance": zero_variance, "all_null": all_null,
            "validation_unseen_count": _unseen_lookup(unseen, feature, "validation", "unseen_category_count"),
            "validation_unseen_rows": _unseen_lookup(unseen, feature, "validation", "rows_affected"),
            "test_unseen_count": _unseen_lookup(unseen, feature, "test", "unseen_category_count"),
            "test_unseen_rows": _unseen_lookup(unseen, feature, "test", "rows_affected"),
            "outlier_status": outlier_status, "leakage_status": governance["leakage_status"],
            "baseline_decision": decision,
            "decision_reason": (
                f"Deterministic priority: {decision}; {governance['decision_reason']}"
            ),
            "recommended_missing_policy": missing_policy,
            "recommended_encoding": encoding,
            "recommended_scaling": scaling,
            "recommended_outlier_policy": "RETAIN_AND_MONITOR" if feature in config.numeric_features else "NOT_APPLICABLE",
            "variation_status": eda_assessment["variation_status"],
            "redundancy_status": eda_assessment["redundancy_status"],
            "temporal_risk": eda_assessment["temporal_risk"],
            "eda_status": eda_assessment["eda_status"],
            "status_reason": eda_assessment["status_reason"],
        })
    baseline = pd.DataFrame(baseline_rows)

    transformations = []
    for row in baseline.loc[baseline["baseline_decision"].isin({"INCLUDE", "CONDITIONAL", "REVIEW"})].to_dict("records"):
        high_cardinality = row["feature_type"] == "categorical" and row["cardinality_band"] == "HIGH"
        transformations.append({
            "feature_name": row["feature_name"], "data_type": row["feature_type"],
            "recommended_missing_strategy": row["recommended_missing_policy"],
            "recommended_encoding": row["recommended_encoding"],
            "recommended_scaling": row["recommended_scaling"],
            "recommended_rare_category_policy": (
                "EVALUATE_TRAIN_FITTED_RARE_GROUPING" if high_cardinality else "NONE"
            ),
            "recommended_unknown_category_policy": (
                "MAP_TO___UNKNOWN__" if row["feature_type"] == "categorical" else "NOT_APPLICABLE"
            ),
            "recommended_outlier_treatment": row["recommended_outlier_policy"],
            "fit_on": "train", "apply_to": "validation,test",
            "reason": "Recommendation only; Notebook 11 must fit any state on train.",
        })

    outlier_rows = []
    for row in analysis_tables["numeric_outlier_analysis.csv"].to_dict("records"):
        outlier_rows.append({
            "feature_name": row["feature_name"], "outlier_type": "IQR_AND_PERCENTILE",
            "diagnostic_rule": f"train IQR multiplier={config.iqr_multiplier}; train p01/p99",
            "lower_threshold": row["lower_iqr_fence"], "upper_threshold": row["upper_iqr_fence"],
            "outlier_count": row["total_iqr_outlier_count"],
            "outlier_rate": row["iqr_outlier_rate"],
            "domain_invalid_count": row["domain_invalid_count"],
            "recommended_action": row["recommended_action"],
            "decision_reason": row["reason"], "fit_or_derive_on": "train",
            "apply_to": "validation,test without row removal",
        })
    for feature in config.categorical_features:
        outlier_rows.append({
            "feature_name": feature, "outlier_type": "RARE_CATEGORY",
            "diagnostic_rule": "configured count/share candidates; no threshold selected",
            "lower_threshold": pd.NA, "upper_threshold": pd.NA,
            "outlier_count": pd.NA, "outlier_rate": pd.NA, "domain_invalid_count": 0,
            "recommended_action": "RETAIN_AND_MONITOR",
            "decision_reason": "Rare groups remain diagnostic until train-fitted evaluation.",
            "fit_or_derive_on": "train", "apply_to": "validation,test unknown-safe",
        })
    return {
        "baseline_feature_recommendation.csv": baseline,
        "transformation_recommendation.csv": pd.DataFrame(transformations),
        "outlier_recommendation.csv": pd.DataFrame(outlier_rows),
        "zero_variance_columns.csv": baseline.loc[baseline["zero_variance"], ["feature_name", "baseline_decision", "decision_reason"]].reset_index(drop=True),
        "all_null_columns.csv": baseline.loc[baseline["all_null"], ["feature_name", "baseline_decision", "decision_reason"]].reset_index(drop=True),
    }
