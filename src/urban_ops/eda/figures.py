"""Create the small governed Step 9A figure inventory from verified EDA evidence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "urban_ops_matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_FIGURES = (
    "train_target_distribution.png",
    "monthly_complaint_volume.png",
    "monthly_missed_target_rate.png",
    "missed_rate_by_day_of_week.png",
    "missed_rate_by_created_hour.png",
    "missed_rate_by_borough.png",
    "top_location_types.png",
    "location_type_target_rates.png",
    "feature_missingness.png",
    "split_target_rate_comparison.png",
    "split_missingness_drift.png",
    "latitude_distribution.png",
    "longitude_distribution.png",
)


def _save(path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    """Label, lay out, save, and close the active figure."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def write_eda_figures(
    figure_root: Path, tables: dict[str, pd.DataFrame], train: pd.DataFrame
) -> tuple[str, ...]:
    """Write required non-leakage figures and close every matplotlib figure."""
    figure_root.mkdir(parents=True, exist_ok=True)
    target = tables["train_target_distribution.csv"].iloc[0]
    plt.figure(figsize=(6, 4))
    plt.bar(["On time", "Missed"], [target["on_time_count"], target["missed_count"]])
    _save(figure_root / REQUIRED_FIGURES[0], title="Training target distribution", xlabel="Outcome", ylabel="Complaints")

    volume = tables["monthly_volume.csv"]
    plt.figure(figsize=(9, 4)); plt.plot(volume["month"], volume["row_count"], marker="o"); plt.xticks(rotation=45)
    _save(figure_root / REQUIRED_FIGURES[1], title="Monthly training complaint volume", xlabel="Month", ylabel="Complaints")
    monthly = tables["monthly_target_rate.csv"]
    plt.figure(figsize=(9, 4)); plt.plot(monthly["month"], monthly["missed_target_rate"], marker="o"); plt.xticks(rotation=45); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[2], title="Monthly training missed-target rate", xlabel="Month", ylabel="Missed-target rate")
    days = tables["day_of_week_target_rate.csv"]
    plt.figure(figsize=(8, 4)); plt.bar(days["created_day_name"], days["missed_target_rate"].fillna(0)); plt.xticks(rotation=35); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[3], title="Training missed rate by creation weekday", xlabel="Day", ylabel="Missed-target rate")
    hours = tables["hour_of_day_target_rate.csv"]
    plt.figure(figsize=(8, 4)); plt.plot(hours["created_hour"], hours["missed_target_rate"], marker="o"); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[4], title="Training missed rate by creation hour", xlabel="Hour", ylabel="Missed-target rate")

    categorical = tables["categorical_target_rates.csv"]
    borough = categorical.loc[categorical["feature_name"].eq("borough")].sort_values("row_count", ascending=False)
    plt.figure(figsize=(8, 4)); plt.bar(borough["category_value"], borough["missed_target_rate"]); plt.xticks(rotation=35); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[5], title="Training missed rate by borough", xlabel="Borough", ylabel="Missed-target rate")
    profile = tables["categorical_feature_profile.csv"]
    locations = profile.loc[profile["feature_name"].eq("location_type")].nlargest(10, "row_count")
    plt.figure(figsize=(9, 4)); plt.bar(locations["category_value"], locations["row_count"]); plt.xticks(rotation=45, ha="right")
    _save(figure_root / REQUIRED_FIGURES[6], title="Top training location types", xlabel="Location type", ylabel="Complaints")
    location_rates = categorical.loc[
        categorical["feature_name"].eq("location_type") & categorical["minimum_support_passed"]
    ].nlargest(15, "row_count")
    plt.figure(figsize=(10, 4)); plt.bar(location_rates["category_value"], location_rates["missed_target_rate"]); plt.xticks(rotation=45, ha="right"); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[7], title="Supported location-type missed rates", xlabel="Location type", ylabel="Missed-target rate")

    missing = tables["train_missingness.csv"].nlargest(15, "missing_rate").sort_values("missing_rate")
    plt.figure(figsize=(8, 6)); plt.barh(missing["column_name"], missing["missing_rate"]); plt.xlim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[8], title="Highest training feature missingness", xlabel="Missing rate", ylabel="Column")
    split_target = tables["split_target_comparison.csv"]
    plt.figure(figsize=(6, 4)); plt.bar(split_target["split_name"], split_target["missed_target_rate"]); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[9], title="Governed split target-rate disclosure", xlabel="Split", ylabel="Missed-target rate")
    split_missing = tables["split_missingness_comparison.csv"].set_index("feature_name")[["train_missing_rate", "validation_missing_rate", "test_missing_rate"]]
    plt.figure(figsize=(9, 5)); split_missing.plot(kind="bar", ax=plt.gca()); plt.xticks(rotation=35, ha="right"); plt.ylim(0, 1)
    _save(figure_root / REQUIRED_FIGURES[10], title="Candidate-feature missingness by split", xlabel="Feature", ylabel="Missing rate")

    for index, feature in enumerate(("latitude", "longitude"), start=11):
        values = pd.to_numeric(train[feature], errors="coerce").dropna()
        plt.figure(figsize=(7, 4)); plt.hist(values, bins=40)
        _save(figure_root / REQUIRED_FIGURES[index], title=f"Training {feature} distribution", xlabel=feature.title(), ylabel="Complaints")
    return REQUIRED_FIGURES
