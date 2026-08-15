"""Deterministic train-first EDA and restricted later-split structural comparisons."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from urban_ops.eda.models import EDAConfig, SourceSplitEvidence


MISSING_CATEGORY = "__MISSING__"


def derive_temporal_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Return an EDA-only copy with governed creation-time calendar features."""
    result = frame.copy(deep=True)
    created = result[timestamp_column]
    result["created_hour"] = created.dt.hour.astype("Int8")
    result["created_day_of_week"] = created.dt.dayofweek.astype("Int8")
    result["created_day_name"] = created.dt.day_name().astype("string")
    result["created_day_of_month"] = created.dt.day.astype("Int8")
    result["created_week_of_year"] = created.dt.isocalendar().week.astype("Int8")
    result["created_month"] = created.dt.month.astype("Int8")
    result["created_month_name"] = created.dt.month_name().astype("string")
    result["created_quarter"] = created.dt.quarter.astype("Int8")
    result["created_year"] = created.dt.year.astype("Int16")
    result["is_weekend"] = created.dt.dayofweek.ge(5)
    return result


def build_target_tables(
    source: SourceSplitEvidence, *, target_column: str
) -> dict[str, pd.DataFrame]:
    """Build train baseline evidence and governed split target disclosure."""
    train_target = source.train[target_column].astype(int)
    missed = int(train_target.sum())
    on_time = len(train_target) - missed
    majority_value = 1 if missed > on_time else 0
    distribution = pd.DataFrame([{
        "on_time_count": on_time,
        "missed_count": missed,
        "total_count": len(train_target),
        "on_time_share": on_time / len(train_target),
        "missed_target_rate": missed / len(train_target),
        "class_ratio_missed_to_on_time": missed / on_time if on_time else pd.NA,
        "majority_class": majority_value,
        "majority_class_accuracy": max(on_time, missed) / len(train_target),
        "constant_probability_baseline": missed / len(train_target),
        "class_weighting_recommendation": (
            "EXPERIMENT_NOT_AUTOMATIC; both classes have substantial support"
        ),
    }])
    comparison = []
    for name in ("train", "validation", "test"):
        target = getattr(source, name)[target_column].astype(int)
        count_missed = int(target.sum())
        comparison.append({
            "split_name": name, "row_count": len(target),
            "on_time_count": len(target) - count_missed,
            "missed_count": count_missed,
            "missed_target_rate": float(target.mean()),
            "governance_use": (
                "feature discovery authority" if name == "train"
                else "structural disclosure only; not feature selection"
            ),
        })
    return {
        "train_target_distribution.csv": distribution,
        "split_target_comparison.csv": pd.DataFrame(comparison),
    }


def _grouped_target_rate(
    frame: pd.DataFrame,
    group_column: str,
    target_column: str,
    *,
    all_values: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Return support and binary-target counts for one ordered grouping."""
    grouped = frame.groupby(group_column, observed=False)[target_column].agg(["size", "sum"])
    if all_values is not None:
        grouped = grouped.reindex(list(all_values), fill_value=0)
    grouped.index.name = group_column
    result = grouped.reset_index().rename(columns={"size": "row_count", "sum": "missed_count"})
    result["missed_count"] = result["missed_count"].astype(int)
    result["on_time_count"] = result["row_count"] - result["missed_count"]
    result["missed_target_rate"] = result["missed_count"].div(
        result["row_count"].replace(0, pd.NA)
    )
    return result


def build_temporal_tables(
    train: pd.DataFrame, config: EDAConfig
) -> dict[str, pd.DataFrame]:
    """Build ordered train-only temporal volume, target, and extrema evidence."""
    timestamp = config.timestamp_column
    target = config.target_column
    local = train.assign(
        month=train[timestamp].dt.tz_localize(None).dt.to_period("M").astype(str)
    )
    month_index = pd.period_range(
        train[timestamp].min().tz_localize(None).to_period("M"),
        train[timestamp].max().tz_localize(None).to_period("M"), freq="M",
    ).astype(str)
    volume_grouped = local.groupby("month")[timestamp].agg(
        row_count="size", minimum_created_date="min", maximum_created_date="max"
    ).reindex(month_index)
    volume_grouped.index.name = "month"
    volume = volume_grouped.reset_index()
    volume["row_count"] = volume["row_count"].fillna(0).astype(int)
    volume["row_share"] = volume["row_count"] / len(train)
    volume["month_to_month_count_change"] = volume["row_count"].diff()
    volume["month_to_month_count_pct_change"] = volume["row_count"].pct_change().replace(
        [np.inf, -np.inf], pd.NA
    )
    volume["support_flag"] = np.where(volume["row_count"].gt(0), "SUPPORTED", "NO_SUPPORT")

    monthly_rate = _grouped_target_rate(local, "month", target, all_values=month_index)
    overall = float(train[target].astype(int).mean())
    monthly_rate["overall_train_rate"] = overall
    monthly_rate["rate_difference_from_train"] = monthly_rate["missed_target_rate"] - overall
    monthly_rate["month_to_month_rate_change"] = monthly_rate["missed_target_rate"].diff()
    monthly_rate["support_flag"] = np.where(
        monthly_rate["row_count"].ge(config.minimum_support_for_target_rate),
        "SUPPORTED", "LOW_SUPPORT",
    )

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_rate = _grouped_target_rate(train, "created_day_name", target, all_values=days)
    day_rate.insert(0, "day_order", range(7))
    hour_rate = _grouped_target_rate(train, "created_hour", target, all_values=range(24))
    quarter_rate = _grouped_target_rate(train, "created_quarter", target, all_values=range(1, 5))
    weekend_rate = _grouped_target_rate(train, "is_weekend", target, all_values=(False, True))

    def row_for(metric: str, row: pd.Series, value_column: str, interpretation: str) -> dict[str, object]:
        return {
            "metric": metric, "period": row.iloc[0],
            "metric_value": row[value_column], "row_count": row.get("row_count", pd.NA),
            "classification": interpretation,
            "recommended_action": "RETAIN_PERIOD_AND_INVESTIGATE_CONTEXT",
        }

    volume_change = volume["month_to_month_count_change"].abs()
    rate_change = monthly_rate["month_to_month_rate_change"].abs()
    operational = pd.DataFrame([
        row_for("highest_volume_month", volume.loc[volume["row_count"].idxmax()], "row_count", "valid operational variation"),
        row_for("lowest_volume_month", volume.loc[volume["row_count"].idxmin()], "row_count", "valid operational variation"),
        row_for("largest_month_to_month_volume_change", volume.loc[volume_change.idxmax()], "month_to_month_count_change", "possible collection or policy change"),
        row_for("highest_target_rate_month", monthly_rate.loc[monthly_rate["missed_target_rate"].idxmax()], "missed_target_rate", "valid operational variation"),
        row_for("lowest_target_rate_month", monthly_rate.loc[monthly_rate["missed_target_rate"].idxmin()], "missed_target_rate", "valid operational variation"),
        row_for("largest_month_to_month_target_rate_change", monthly_rate.loc[rate_change.idxmax()], "month_to_month_rate_change", "requires investigation"),
    ])
    return {
        "monthly_volume.csv": volume,
        "monthly_target_rate.csv": monthly_rate,
        "day_of_week_target_rate.csv": day_rate,
        "hour_of_day_target_rate.csv": hour_rate,
        "quarterly_target_rate.csv": quarter_rate,
        "weekend_target_rate.csv": weekend_rate,
        "temporal_outlier_analysis.csv": operational,
    }


def missingness_band(rate: float, bands: dict[str, float]) -> str:
    """Classify one descriptive missingness rate without making a feature decision."""
    if rate == bands["complete_max"]:
        return "COMPLETE"
    if rate < bands["low_max"]:
        return "LOW"
    if rate < bands["moderate_max"]:
        return "MODERATE"
    if rate < bands["high_max"]:
        return "HIGH"
    if rate < bands["very_high_max"]:
        return "VERY_HIGH"
    return "ALL_NULL"


def build_missingness_tables(
    source: SourceSplitEvidence, train: pd.DataFrame, config: EDAConfig
) -> dict[str, pd.DataFrame]:
    """Build train missingness evidence and restricted cross-split comparisons."""
    train_rows = []
    for column in train.columns:
        missing = int(train[column].isna().sum())
        rate = missing / len(train)
        train_rows.append({
            "column_name": column, "missing_count": missing, "missing_rate": rate,
            "non_null_count": len(train) - missing,
            "missingness_band": missingness_band(rate, config.missingness_bands),
        })
    train_missing = pd.DataFrame(train_rows)
    candidates = tuple(dict.fromkeys(
        (*config.categorical_features, *config.numeric_features, *config.derived_temporal_features)
    ))
    by_target = []
    for feature in candidates:
        rates = {}
        for target_value in (0, 1):
            subset = train.loc[train[config.target_column].astype(int).eq(target_value)]
            missing = int(subset[feature].isna().sum())
            rates[target_value] = missing / len(subset) if len(subset) else 0.0
            by_target.append({
                "feature_name": feature, "target_value": target_value,
                "row_count": len(subset), "missing_count": missing,
                "missing_rate": rates[target_value], "absolute_difference_between_targets": pd.NA,
            })
        for row in by_target[-2:]:
            row["absolute_difference_between_targets"] = abs(rates[1] - rates[0])
    local = train.assign(month=train[config.timestamp_column].dt.tz_localize(None).dt.to_period("M").astype(str))
    by_month = []
    for feature in candidates:
        for month, subset in local.groupby("month", sort=True):
            missing = int(subset[feature].isna().sum())
            by_month.append({
                "feature_name": feature, "month": month, "row_count": len(subset),
                "missing_count": missing, "missing_rate": missing / len(subset),
            })
    comparison = []
    raw_candidates = tuple(dict.fromkeys((*config.categorical_features, *config.numeric_features)))
    for feature in raw_candidates:
        rates = {
            name: float(getattr(source, name)[feature].isna().mean())
            for name in ("train", "validation", "test")
        }
        maximum = max(abs(rates["validation"] - rates["train"]), abs(rates["test"] - rates["train"]))
        comparison.append({
            "feature_name": feature,
            "train_missing_rate": rates["train"],
            "validation_missing_rate": rates["validation"],
            "test_missing_rate": rates["test"],
            "train_validation_difference": rates["validation"] - rates["train"],
            "train_test_difference": rates["test"] - rates["train"],
            "maximum_difference": maximum,
            "interpretation": "Structural disclosure only; train governs policy.",
        })
    return {
        "train_missingness.csv": train_missing,
        "missingness_by_target.csv": pd.DataFrame(by_target),
        "missingness_by_month.csv": pd.DataFrame(by_month),
        "split_missingness_comparison.csv": pd.DataFrame(comparison),
    }


def _category_values(series: pd.Series) -> pd.Series:
    """Return stable string categories while preserving source nulls as an EDA label."""
    return series.astype("string").fillna(MISSING_CATEGORY)


def build_categorical_tables(
    source: SourceSplitEvidence, train: pd.DataFrame, config: EDAConfig
) -> dict[str, pd.DataFrame]:
    """Build train cardinality/rareness evidence and later-split unknown disclosure."""
    features = tuple(dict.fromkeys((
        *config.categorical_features, "created_day_name", "created_month_name",
        "agency", "agency_name", "complaint_type", "descriptor", "descriptor_2",
        "open_data_channel_type",
    )))
    cardinality_rows, profile_rows, target_rows = [], [], []
    overall_rate = float(train[config.target_column].astype(int).mean())
    for feature in features:
        values = _category_values(train[feature])
        counts = values.value_counts(dropna=False).sort_index(kind="stable")
        ranked = values.value_counts(dropna=False)
        unique = int(values.nunique(dropna=False))
        cardinality_rows.append({
            "feature_name": feature, "unique_count": unique,
            "missing_count": int(train[feature].isna().sum()),
            "missing_rate": float(train[feature].isna().mean()),
            "most_common_value": ranked.index[0], "most_common_count": int(ranked.iloc[0]),
            "most_common_share": float(ranked.iloc[0] / len(train)),
            "top_5_share": float(ranked.head(5).sum() / len(train)),
            "top_10_share": float(ranked.head(10).sum() / len(train)),
            "singleton_category_count": int(ranked.eq(1).sum()),
            "cardinality_band": (
                "ZERO_VARIANCE" if unique == 1 else "LOW" if unique <= 15
                else "MODERATE" if unique <= 50 else "HIGH"
            ),
        })
        for category, count in counts.items():
            subset = train.loc[values.eq(category)]
            missed = int(subset[config.target_column].astype(int).sum())
            rank = int(ranked.index.get_loc(category) + 1)
            if rank <= config.top_n_categories:
                profile_rows.append({
                    "feature_name": feature, "category_value": category,
                    "row_count": int(count), "row_share": count / len(train),
                    "rank_by_count": rank,
                })
            support = int(count) >= config.minimum_support_for_target_rate
            rate = missed / count
            target_rows.append({
                "feature_name": feature, "category_value": category,
                "row_count": int(count), "row_share": count / len(train),
                "on_time_count": int(count) - missed, "missed_count": missed,
                "missed_target_rate": rate, "difference_from_train_rate": rate - overall_rate,
                "minimum_support_passed": support,
                "stability_warning": "NONE" if support else "LOW_SUPPORT_DO_NOT_OVERINTERPRET",
            })

    rare_rows = []
    raw_features = config.categorical_features
    for feature in raw_features:
        train_values = _category_values(source.train[feature])
        counts = train_values.value_counts(dropna=False)
        for threshold_type, thresholds in (
            ("count", config.rare_count_candidates), ("share", config.rare_share_candidates)
        ):
            for threshold in thresholds:
                rare = set(counts.index[counts.le(threshold if threshold_type == "count" else threshold * len(train))])
                affected = {
                    name: int(_category_values(getattr(source, name)[feature]).isin(rare).sum())
                    for name in ("train", "validation", "test")
                }
                rare_rows.append({
                    "feature_name": feature, "threshold_type": threshold_type,
                    "threshold_value": threshold, "rare_category_count": len(rare),
                    "train_rows_affected": affected["train"],
                    "train_row_share": affected["train"] / len(source.train),
                    "validation_rows_affected": affected["validation"],
                    "test_rows_affected": affected["test"],
                    "recommendation": "DIAGNOSTIC_ONLY; evaluate train-fitted grouping in Notebook 11",
                })
    unseen_rows = []
    for feature in raw_features:
        known = set(_category_values(source.train[feature]).unique())
        for split_name in ("validation", "test"):
            values = _category_values(getattr(source, split_name)[feature])
            unseen = sorted(set(values.unique()).difference(known))
            mask = values.isin(unseen)
            unseen_rows.append({
                "feature_name": feature, "comparison_split": split_name,
                "unseen_category_count": len(unseen), "unseen_categories": "|".join(unseen),
                "rows_affected": int(mask.sum()), "row_share_affected": float(mask.mean()),
                "recommended_unknown_policy": "TRAIN_FITTED_UNKNOWN_SAFE_ENCODING",
            })
    return {
        "categorical_cardinality.csv": pd.DataFrame(cardinality_rows),
        "categorical_feature_profile.csv": pd.DataFrame(profile_rows),
        "categorical_target_rates.csv": pd.DataFrame(target_rows),
        "rare_category_analysis.csv": pd.DataFrame(rare_rows),
        "unseen_categories.csv": pd.DataFrame(unseen_rows),
    }


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce an EDA view to numeric without changing source values."""
    return pd.to_numeric(series, errors="coerce")


def _numeric_stats(series: pd.Series, percentiles: tuple[float, ...]) -> dict[str, object]:
    """Calculate configured numeric distribution statistics on finite values."""
    numeric = _numeric(series)
    finite = numeric.loc[np.isfinite(numeric)]
    quantiles = finite.quantile(percentiles) if len(finite) else pd.Series(dtype=float)
    q1 = float(finite.quantile(0.25)) if len(finite) else pd.NA
    q3 = float(finite.quantile(0.75)) if len(finite) else pd.NA
    result: dict[str, object] = {
        "count": len(finite), "missing_count": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()),
        "mean": float(finite.mean()) if len(finite) else pd.NA,
        "standard_deviation": float(finite.std()) if len(finite) > 1 else pd.NA,
        "minimum": float(finite.min()) if len(finite) else pd.NA,
        "maximum": float(finite.max()) if len(finite) else pd.NA,
        "q1": q1, "q3": q3,
        "iqr": q3 - q1 if pd.notna(q1) and pd.notna(q3) else pd.NA,
        "infinite_count": int(np.isinf(numeric).sum()),
    }
    for percentile in percentiles:
        label = "median" if percentile == 0.5 else f"p{int(percentile * 100):02d}"
        result[label] = float(quantiles.get(percentile, np.nan))
    return result


def build_numeric_tables(
    source: SourceSplitEvidence, train: pd.DataFrame, config: EDAConfig
) -> dict[str, pd.DataFrame]:
    """Build train numeric summaries, outlier diagnostics, geography, and range drift."""
    features = tuple(dict.fromkeys((
        *config.numeric_features, "created_day_of_month", "created_week_of_year"
    )))
    summary_rows, target_rows = [], []
    for feature in features:
        summary_rows.append({"feature_name": feature, **_numeric_stats(train[feature], config.percentiles)})
        for target_value in (0, 1):
            subset = train.loc[train[config.target_column].astype(int).eq(target_value), feature]
            stats = _numeric_stats(subset, (0.25, 0.5, 0.75))
            target_rows.append({
                "feature_name": feature, "target_value": target_value,
                "count": stats["count"], "missing_rate": stats["missing_rate"],
                "mean": stats["mean"], "median": stats["median"],
                "standard_deviation": stats["standard_deviation"],
                "q1": stats["q1"], "q3": stats["q3"], "iqr": stats["iqr"],
                "minimum": stats["minimum"], "maximum": stats["maximum"],
            })

    outlier_rows = []
    for feature in features:
        numeric = _numeric(train[feature])
        finite = numeric.loc[np.isfinite(numeric)]
        applicable = feature in config.numeric_features
        q1, q3 = finite.quantile([0.25, 0.75]) if len(finite) else (np.nan, np.nan)
        iqr = q3 - q1
        lower, upper = q1 - config.iqr_multiplier * iqr, q3 + config.iqr_multiplier * iqr
        p01, p99 = finite.quantile([0.01, 0.99]) if len(finite) else (np.nan, np.nan)
        below = int(finite.lt(lower).sum()) if applicable else 0
        above = int(finite.gt(upper).sum()) if applicable else 0
        outlier_rows.append({
            "feature_name": feature, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_iqr_fence": lower if applicable else pd.NA,
            "upper_iqr_fence": upper if applicable else pd.NA,
            "below_fence_count": below, "above_fence_count": above,
            "total_iqr_outlier_count": below + above,
            "iqr_outlier_rate": (below + above) / len(train),
            "p01_threshold": p01 if applicable else pd.NA,
            "p99_threshold": p99 if applicable else pd.NA,
            "below_p01_count": int(finite.lt(p01).sum()) if applicable else 0,
            "above_p99_count": int(finite.gt(p99).sum()) if applicable else 0,
            "domain_invalid_count": 0,
            "recommended_action": "RETAIN" if applicable else "NOT_APPLICABLE",
            "reason": (
                "Statistical extremes remain valid diagnostic observations."
                if applicable else "IQR is not applied to cyclic or category-coded values."
            ),
        })

    lat_raw, lon_raw = train["latitude"], train["longitude"]
    lat, lon = _numeric(lat_raw), _numeric(lon_raw)
    lat_nonfinite = lat.notna() & ~np.isfinite(lat)
    lon_nonfinite = lon.notna() & ~np.isfinite(lon)
    lat_malformed = lat_raw.notna() & lat.isna()
    lon_malformed = lon_raw.notna() & lon.isna()
    partial_lat = lat.notna() & lon.isna()
    partial_lon = lon.notna() & lat.isna()
    lat_world = lat.notna() & ~lat.between(*config.latitude_range)
    lon_world = lon.notna() & ~lon.between(*config.longitude_range)
    min_lat, max_lat, min_lon, max_lon = config.nyc_bounds
    valid_pair = lat.notna() & lon.notna() & ~lat_world & ~lon_world & ~lat_nonfinite & ~lon_nonfinite
    outside_nyc = valid_pair & ~(lat.between(min_lat, max_lat) & lon.between(min_lon, max_lon))
    geographic_masks = (
        ("latitude_below_world_bound", lat.lt(config.latitude_range[0]), "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("latitude_above_world_bound", lat.gt(config.latitude_range[1]), "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("longitude_below_world_bound", lon.lt(config.longitude_range[0]), "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("longitude_above_world_bound", lon.gt(config.longitude_range[1]), "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("latitude_malformed", lat_malformed, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("longitude_malformed", lon_malformed, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("latitude_non_finite", lat_nonfinite, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("longitude_non_finite", lon_nonfinite, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("latitude_present_longitude_missing", partial_lat, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("longitude_present_latitude_missing", partial_lon, "DOMAIN_INVALID", "INVESTIGATE_UPSTREAM"),
        ("coordinate_pair_outside_nyc_bounds", outside_nyc, "STATISTICAL_OR_OPERATIONAL", "RETAIN_AND_MONITOR"),
    )
    geographic = pd.DataFrame([{
        "finding": name, "outlier_category": category,
        "row_count": int(mask.fillna(False).sum()),
        "row_rate": float(mask.fillna(False).mean()),
        "recommended_action": action,
        "governed_bounds": (
            f"lat={config.latitude_range}; lon={config.longitude_range}; "
            f"nyc={config.nyc_bounds}"
        ),
    } for name, mask, category, action in geographic_masks])
    domain_invalid = int(geographic.loc[geographic["outlier_category"].eq("DOMAIN_INVALID"), "row_count"].sum())
    for row in outlier_rows:
        if row["feature_name"] in {"latitude", "longitude"}:
            row["domain_invalid_count"] = domain_invalid

    drift_rows = []
    for feature in features:
        for split_name in ("train", "validation", "test"):
            stats = _numeric_stats(getattr(source, split_name)[feature] if feature in source.train else derive_temporal_features(getattr(source, split_name), config.timestamp_column)[feature], config.percentiles)
            drift_rows.append({
                "feature_name": feature, "split_name": split_name,
                "minimum": stats["minimum"], "maximum": stats["maximum"],
                "p01": stats.get("p01", pd.NA), "median": stats.get("median", pd.NA),
                "p99": stats.get("p99", pd.NA), "iqr": stats["iqr"],
                "missing_rate": stats["missing_rate"],
                "governance_use": "train authority" if split_name == "train" else "structural disclosure only",
            })
    return {
        "numeric_feature_summary.csv": pd.DataFrame(summary_rows),
        "numeric_summary_by_target.csv": pd.DataFrame(target_rows),
        "geographic_outlier_analysis.csv": geographic,
        "numeric_outlier_analysis.csv": pd.DataFrame(outlier_rows),
        "numeric_range_drift.csv": pd.DataFrame(drift_rows),
    }


def build_structural_drift_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize later-split evidence without imposing hard drift rejection thresholds."""
    rows = []
    targets = tables["split_target_comparison.csv"].set_index("split_name")
    train_rate = float(targets.loc["train", "missed_target_rate"])
    for split_name in ("validation", "test"):
        later = float(targets.loc[split_name, "missed_target_rate"])
        rows.append({
            "area": "target", "feature_name": "missed_resolution_target",
            "comparison": f"train_to_{split_name}", "metric": "missed_target_rate",
            "train_value": train_rate, "later_split_value": later,
            "absolute_difference": abs(later - train_rate), "warning_status": "WARN",
            "interpretation": "Temporal prevalence drift disclosed; not feature evidence.",
            "decision_impact": "No feature or transformation decision from later outcomes.",
        })
    for record in tables["split_missingness_comparison.csv"].to_dict("records"):
        for split_name, value_key in (("validation", "validation_missing_rate"), ("test", "test_missing_rate")):
            rows.append({
                "area": "missingness", "feature_name": record["feature_name"],
                "comparison": f"train_to_{split_name}", "metric": "missing_rate",
                "train_value": record["train_missing_rate"],
                "later_split_value": record[value_key],
                "absolute_difference": abs(record[value_key] - record["train_missing_rate"]),
                "warning_status": "INFO", "interpretation": "Structural disclosure without a hard threshold.",
                "decision_impact": "Train remains the policy authority.",
            })
    for record in tables["unseen_categories.csv"].to_dict("records"):
        rows.append({
            "area": "unseen_category", "feature_name": record["feature_name"],
            "comparison": f"train_to_{record['comparison_split']}",
            "metric": "row_share_affected", "train_value": 0.0,
            "later_split_value": record["row_share_affected"],
            "absolute_difference": record["row_share_affected"],
            "warning_status": "WARN" if record["rows_affected"] else "INFO",
            "interpretation": "Later categories require unknown-safe handling.",
            "decision_impact": "Encoder policy only; never later-split feature selection.",
        })
    numeric = tables["numeric_range_drift.csv"]
    for feature, feature_rows in numeric.groupby("feature_name", sort=True):
        indexed = feature_rows.set_index("split_name")
        for split_name in ("validation", "test"):
            for metric in ("minimum", "maximum", "p01", "median", "p99", "iqr"):
                train_value = indexed.loc["train", metric]
                later_value = indexed.loc[split_name, metric]
                difference = (
                    abs(float(later_value) - float(train_value))
                    if pd.notna(train_value) and pd.notna(later_value) else pd.NA
                )
                rows.append({
                    "area": "numeric_range", "feature_name": feature,
                    "comparison": f"train_to_{split_name}", "metric": metric,
                    "train_value": train_value, "later_split_value": later_value,
                    "absolute_difference": difference, "warning_status": "INFO",
                    "interpretation": "Numeric range drift disclosed without a hard rejection threshold.",
                    "decision_impact": "Train remains the threshold and recommendation authority.",
                })
    return pd.DataFrame(rows)
