"""Inline visualizations for the split-aware EDA notebook.

The functions in this module only consume the governed aggregate tables produced
by :mod:`urban_ops.eda.pipeline`.  They do not reload or mutate split artifacts,
fit preprocessing state, or use validation/test outcomes for feature discovery.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


EDA_COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea")


def _finish(
    figure: Figure,
    *,
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Apply consistent notebook styling and return a renderable figure."""
    figure.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    if subtitle:
        figure.text(0.01, 0.94, subtitle, fontsize=9, color="#4b5563")
    for axis in figure.axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.90 if subtitle else 0.93))
    return figure


def _percent_axis(axis: Axes) -> None:
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def plot_target_distribution(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show training class counts and shares."""
    row = tables["train_target_distribution.csv"].iloc[0]
    counts = [int(row["on_time_count"]), int(row["missed_count"])]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(["On time", "Missed target"], counts, color=EDA_COLORS[:2])
    axis.bar_label(
        bars,
        labels=[f"{count:,}\n({count / sum(counts):.1%})" for count in counts],
        padding=4,
    )
    axis.set_ylabel("Training complaints")
    axis.set_ylim(0, max(counts) * 1.18)
    return _finish(
        figure,
        title="Training target distribution",
        subtitle="Both outcomes have substantial support; bars show count and share.",
    )


def plot_monthly_volume(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show complaint volume through the training window."""
    data = tables["monthly_volume.csv"]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(data["month"], data["row_count"], marker="o", color=EDA_COLORS[0])
    axis.fill_between(data["month"], data["row_count"], alpha=0.12, color=EDA_COLORS[0])
    highest = data.loc[data["row_count"].idxmax()]
    lowest = data.loc[data["row_count"].idxmin()]
    for row, label in ((highest, "Highest"), (lowest, "Lowest")):
        axis.annotate(
            f"{label}: {int(row['row_count']):,}",
            (row["month"], row["row_count"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Complaints")
    return _finish(figure, title="Monthly training complaint volume")


def plot_monthly_target_rate(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show monthly missed-target rate against the train-wide baseline."""
    data = tables["monthly_target_rate.csv"]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(data["month"], data["missed_target_rate"], marker="o", color=EDA_COLORS[1])
    baseline = float(data["overall_train_rate"].iloc[0])
    axis.axhline(baseline, color="#374151", linestyle="--", linewidth=1.2, label=f"Train overall: {baseline:.1%}")
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Missed-target rate")
    _percent_axis(axis)
    return _finish(figure, title="Monthly missed-target rate")


def plot_day_of_week(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show training missed-target rate by creation weekday."""
    data = tables["day_of_week_target_rate.csv"].sort_values("day_order")
    figure, axis = plt.subplots(figsize=(9, 4))
    rates = pd.to_numeric(data["missed_target_rate"], errors="coerce").fillna(0)
    bars = axis.bar(data["created_day_name"], rates, color=EDA_COLORS[0])
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in rates], padding=3)
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylabel("Missed-target rate")
    _percent_axis(axis)
    return _finish(figure, title="Missed-target rate by creation weekday")


def plot_hour_of_day(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show training missed-target rate and volume by creation hour."""
    data = tables["hour_of_day_target_rate.csv"].sort_values("created_hour")
    figure, axis = plt.subplots(figsize=(10, 4))
    rates = pd.to_numeric(data["missed_target_rate"], errors="coerce")
    axis.plot(data["created_hour"], rates, marker="o", color=EDA_COLORS[1])
    axis.set_xticks(range(0, 24, 2))
    axis.set_xlabel("Hour (UTC)")
    axis.set_ylabel("Missed-target rate")
    _percent_axis(axis)
    return _finish(figure, title="Missed-target rate by creation hour")


def plot_period_patterns(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare quarterly and weekday/weekend target patterns."""
    quarter = tables["quarterly_target_rate.csv"]
    weekend = tables["weekend_target_rate.csv"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    quarter_rates = pd.to_numeric(quarter["missed_target_rate"], errors="coerce").fillna(0)
    quarter_bars = axes[0].bar(
        [f"Q{value}" for value in quarter["created_quarter"]],
        quarter_rates,
        color=EDA_COLORS[0],
    )
    axes[0].bar_label(quarter_bars, labels=[f"{v:.1%}" for v in quarter_rates], padding=3)
    axes[0].set_ylabel("Missed-target rate")
    axes[0].set_title("Quarter")
    labels = weekend["is_weekend"].map({False: "Weekday", True: "Weekend"})
    weekend_rates = pd.to_numeric(weekend["missed_target_rate"], errors="coerce").fillna(0)
    weekend_bars = axes[1].bar(labels, weekend_rates, color=EDA_COLORS[2])
    axes[1].bar_label(weekend_bars, labels=[f"{v:.1%}" for v in weekend_rates], padding=3)
    axes[1].set_title("Weekday vs weekend")
    for axis in axes:
        _percent_axis(axis)
    return _finish(figure, title="Training calendar patterns")


def plot_training_missingness(tables: Mapping[str, pd.DataFrame], *, limit: int = 15) -> Figure:
    """Show the highest training missingness rates."""
    data = tables["train_missingness.csv"].nlargest(limit, "missing_rate").sort_values("missing_rate")
    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.barh(data["column_name"], data["missing_rate"], color=EDA_COLORS[1])
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in data["missing_rate"]], padding=3)
    axis.set_xlabel("Missing share")
    axis.set_xlim(0, 1.08)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return _finish(figure, title="Highest training feature missingness")


def plot_missingness_by_target(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare candidate-feature missingness between target classes."""
    data = tables["missingness_by_target.csv"].pivot(
        index="feature_name", columns="target_value", values="missing_rate"
    )
    data = data.rename(columns={0: "On time", 1: "Missed target"})
    figure, axis = plt.subplots(figsize=(9, 5))
    data.plot(kind="barh", ax=axis, color=EDA_COLORS[:2])
    axis.set_xlabel("Missing share")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False)
    return _finish(figure, title="Candidate-feature missingness by target")


def plot_missingness_over_time(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show monthly missingness trajectories for candidate features."""
    data = tables["missingness_by_month.csv"]
    pivot = data.pivot(index="month", columns="feature_name", values="missing_rate")
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(ax=axis, marker="o")
    axis.set_ylabel("Missing share")
    axis.set_xlabel("Month")
    axis.tick_params(axis="x", rotation=45)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(title="Feature", frameon=False, ncol=2)
    return _finish(figure, title="Candidate-feature missingness over training time")


def plot_cardinality(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show categorical unique counts on a log scale."""
    data = tables["categorical_cardinality.csv"].sort_values("unique_count")
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(data["feature_name"], data["unique_count"], color=EDA_COLORS[2])
    axis.bar_label(bars, labels=[f"{int(value):,}" for value in data["unique_count"]], padding=3)
    axis.set_xscale("log")
    axis.set_xlabel("Unique categories (log scale)")
    return _finish(figure, title="Training categorical cardinality")


def plot_categorical_target_rate(
    tables: Mapping[str, pd.DataFrame], feature: str, *, limit: int = 15
) -> Figure:
    """Show supported training category rates, ordered by complaint count."""
    data = tables["categorical_target_rates.csv"]
    data = data.loc[
        data["feature_name"].eq(feature) & data["minimum_support_passed"]
    ].nlargest(limit, "row_count")
    data = data.sort_values("missed_target_rate")
    figure, axis = plt.subplots(figsize=(9, max(3.5, len(data) * 0.42)))
    bars = axis.barh(data["category_value"].astype(str), data["missed_target_rate"], color=EDA_COLORS[0])
    axis.bar_label(
        bars,
        labels=[f"{rate:.1%}  (n={count:,})" for rate, count in zip(data["missed_target_rate"], data["row_count"])],
        padding=3,
        fontsize=8,
    )
    axis.set_xlabel("Missed-target rate")
    axis.set_xlim(0, min(1.0, max(0.6, float(data["missed_target_rate"].max()) + 0.18)))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    title_feature = feature.replace("_", " ").title()
    return _finish(
        figure,
        title=f"{title_feature}: supported training target rates",
        subtitle="Labels show rate and complaint support; low-support categories are excluded.",
    )


def plot_numeric_by_target(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare numeric medians and interquartile ranges by target."""
    data = tables["numeric_summary_by_target.csv"]
    features = data["feature_name"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), squeeze=False)
    for axis, feature in zip(axes[0], features):
        subset = data.loc[data["feature_name"].eq(feature)].sort_values("target_value")
        labels = subset["target_value"].map({0: "On time", 1: "Missed"})
        medians = subset["median"].to_numpy()
        lower = medians - subset["q1"].to_numpy()
        upper = subset["q3"].to_numpy() - medians
        axis.errorbar(
            labels,
            medians,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=7,
            color=EDA_COLORS[0],
        )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_ylabel("Median with interquartile range")
    return _finish(figure, title="Numeric distributions by target")


def plot_outlier_diagnostics(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare IQR and percentile diagnostic counts without removing rows."""
    data = tables["numeric_outlier_analysis.csv"].set_index("feature_name")
    counts = pd.DataFrame(
        {
            "IQR diagnostic": data["total_iqr_outlier_count"],
            "Outside p01–p99": data["below_p01_count"] + data["above_p99_count"],
            "Domain invalid": data["domain_invalid_count"],
        }
    )
    figure, axis = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=axis, color=EDA_COLORS[:3])
    axis.set_ylabel("Training rows flagged")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(frameon=False)
    return _finish(
        figure,
        title="Numeric outlier diagnostics",
        subtitle="Flags are descriptive only; valid extreme rows remain in the governed split.",
    )


def plot_split_target_rate(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Disclose target prevalence across governed temporal splits."""
    data = tables["split_target_comparison.csv"]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(data["split_name"].str.title(), data["missed_target_rate"], color=EDA_COLORS[:3])
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in data["missed_target_rate"]], padding=4)
    axis.set_ylabel("Missed-target rate")
    _percent_axis(axis)
    return _finish(
        figure,
        title="Target prevalence across temporal splits",
        subtitle="Validation/test are disclosure only; training remains the feature-policy authority.",
    )


def plot_split_missingness(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare candidate missingness across splits."""
    data = tables["split_missingness_comparison.csv"].set_index("feature_name")
    rates = data[["train_missing_rate", "validation_missing_rate", "test_missing_rate"]]
    rates.columns = ["Train", "Validation", "Test"]
    figure, axis = plt.subplots(figsize=(10, 5))
    rates.plot(kind="bar", ax=axis, color=EDA_COLORS[:3])
    axis.set_ylabel("Missing share")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=30)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False)
    return _finish(figure, title="Candidate-feature missingness drift")


def plot_unseen_categories(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show rows affected by categories absent from training."""
    data = tables["unseen_categories.csv"].copy()
    pivot = data.pivot(index="feature_name", columns="comparison_split", values="rows_affected").fillna(0)
    figure, axis = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=axis, color=EDA_COLORS[1:3])
    axis.set_ylabel("Rows with unseen category")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=30)
    axis.legend(title="Comparison split", frameon=False)
    return _finish(figure, title="Validation/test unseen-category exposure")


def plot_numeric_drift(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Show split medians as per-feature small multiples."""
    data = tables["numeric_range_drift.csv"]
    features = data["feature_name"].drop_duplicates().tolist()
    columns = min(3, len(features))
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.5 * rows), squeeze=False)
    for axis, feature in zip(axes.flat, features):
        subset = data.loc[data["feature_name"].eq(feature)]
        axis.plot(subset["split_name"].str.title(), subset["median"], marker="o", color=EDA_COLORS[0])
        axis.set_title(feature.replace("_", " ").title())
        axis.set_ylabel("Median")
    for axis in axes.flat[len(features):]:
        axis.remove()
    return _finish(
        figure,
        title="Numeric median drift by feature",
        subtitle="Each panel has its own scale; later splits are structural disclosure only.",
    )


def plot_recommendation_decisions(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Summarize deterministic baseline feature decisions."""
    data = tables["baseline_feature_recommendation.csv"]
    counts = data["baseline_decision"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(counts.index.str.replace("_", " ").str.title(), counts.values, color=EDA_COLORS[3])
    axis.bar_label(bars, padding=3)
    axis.set_xlabel("Features")
    return _finish(figure, title="Baseline feature recommendation summary")
