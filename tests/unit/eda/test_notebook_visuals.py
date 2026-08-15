"""Tests for governed inline EDA notebook visualizations."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from urban_ops.eda.notebook_visuals import (
    plot_cardinality,
    plot_categorical_target_rate,
    plot_day_of_week,
    plot_hour_of_day,
    plot_missingness_by_target,
    plot_missingness_over_time,
    plot_monthly_target_rate,
    plot_monthly_volume,
    plot_numeric_by_target,
    plot_numeric_drift,
    plot_outlier_diagnostics,
    plot_period_patterns,
    plot_recommendation_decisions,
    plot_split_missingness,
    plot_split_target_rate,
    plot_target_distribution,
    plot_training_missingness,
    plot_unseen_categories,
)
from urban_ops.eda.pipeline import run_split_aware_eda


def test_notebook_plotters_return_figures(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    plotters = [
        plot_target_distribution,
        plot_monthly_volume,
        plot_monthly_target_rate,
        plot_day_of_week,
        plot_hour_of_day,
        plot_period_patterns,
        plot_training_missingness,
        plot_missingness_by_target,
        plot_missingness_over_time,
        plot_cardinality,
        plot_numeric_by_target,
        plot_outlier_diagnostics,
        plot_split_target_rate,
        plot_split_missingness,
        plot_unseen_categories,
        plot_numeric_drift,
        plot_recommendation_decisions,
    ]
    for plotter in plotters:
        figure = plotter(tables)
        assert isinstance(figure, Figure)
        assert figure.axes
        plt.close(figure)


def test_categorical_plotter_supports_governed_features(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    for feature in ("borough", "location_type", "incident_zip"):
        figure = plot_categorical_target_rate(tables, feature)
        assert isinstance(figure, Figure)
        assert figure.axes[0].patches
        plt.close(figure)
