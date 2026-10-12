"""Synthetic integration coverage for the complete Phase 7.4 workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from urban_ops.models import duration_regression_workflow as workflow
from urban_ops.splitting.models import SplitBoundaries


def test_regression_workflow_reuses_populations_policy_and_frozen_baseline(
    tmp_path, monkeypatch
) -> None:
    boundaries = SplitBoundaries(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-09-01", tz="UTC"),
        pd.Timestamp("2024-09-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"),
    )
    config = SimpleNamespace(
        processed_root=tmp_path / "processed", latest_pointer=tmp_path / "latest.json",
        timestamp_column="created_date", boundaries=boundaries,
    )
    created = pd.to_datetime([
        "2024-01-01T00:00:00Z", "2024-01-02T06:00:00Z",
        "2024-02-03T12:00:00Z", "2024-03-04T18:00:00Z",
        "2024-03-20T00:00:00Z", "2024-04-02T03:00:00Z",
        "2024-05-03T09:00:00Z", "2024-06-08T15:00:00Z",
        "2024-08-20T00:00:00Z", "2024-09-01T00:00:00Z",
    ], utc=True)
    duration_hours = [0, 10, 30, 50, 400, 5, 20, 45, 400, 1]
    closed = created + pd.to_timedelta(duration_hours, unit="h")
    # These two closures are after their split-specific observation cutoffs.
    closed = pd.Series(closed)
    closed.iloc[4] = pd.Timestamp("2024-04-02", tz="UTC")
    closed.iloc[8] = pd.Timestamp("2024-09-02", tz="UTC")
    frame = pd.DataFrame({
        "unique_key": [f"id-{index}" for index in range(10)],
        "created_date": created, "closed_date": closed, "status": ["Closed"] * 10,
        "resolution_description": ["blocked"] * 10,
    })
    original = frame.copy(deep=True)
    monkeypatch.setattr(workflow, "load_split_config", lambda _: config)
    monkeypatch.setattr(workflow, "resolve_cleaning_run", lambda **_: tmp_path)
    monkeypatch.setattr(
        workflow, "load_verified_cleaned_source",
        lambda _: (frame, tmp_path / "cleaned_service_requests.parquet"),
    )
    baseline_path = tmp_path / "phase_7_3.csv"
    pd.DataFrame([
        ("training_eligible_count", 4, "complaints"),
        ("validation_eligible_count", 3, "complaints"),
        ("validation_mae_hours", 25.0, "hours"),
        ("validation_rmse_hours", 30.0, "hours"),
    ], columns=("metric", "value", "unit")).to_csv(baseline_path, index=False)
    baseline_bytes = baseline_path.read_bytes()
    report_root = tmp_path / "reports"

    result = workflow.run_duration_regression_workflow(
        split_config_path=tmp_path / "splits.yaml",
        baseline_metrics_path=baseline_path, report_root=report_root,
    )

    assert len(result.training.eligible_durations) == 4
    assert len(result.validation.eligible_durations) == 3
    assert result.matrices.train.shape == (4, 4)
    assert result.matrices.validation.shape == (3, 4)
    assert result.matrices.feature_names == (
        "created_hour", "created_day_of_week", "created_month", "is_weekend"
    )
    assert result.comparison["model"].tolist() == [
        "Median Duration Baseline", "Linear Regression", "XGBRegressor"
    ]
    assert result.comparison["validation_sample_count"].eq(3).all()
    assert result.comparison[["validation_mae_hours", "validation_rmse_hours"]].notna().all().all()
    assert result.report_path.is_file() and result.comparison_path.is_file()
    assert set(result.model_predictions) == {"Linear Regression", "XGBRegressor"}
    assert all(
        predictions.raw.shape == (3,)
        for predictions in result.model_predictions.values()
    )
    assert baseline_path.read_bytes() == baseline_bytes
    assert "Test data or targets used: No" in result.report_path.read_text(encoding="utf-8")
    assert "resolution_description" not in result.matrices.feature_names
    pd.testing.assert_frame_equal(frame, original)


def test_frozen_baseline_population_mismatch_fails(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame([
        ("training_eligible_count", 99, "complaints"),
        ("validation_eligible_count", 3, "complaints"),
        ("validation_mae_hours", 1, "hours"),
        ("validation_rmse_hours", 2, "hours"),
    ], columns=("metric", "value", "unit")).to_csv(path, index=False)
    try:
        workflow.load_frozen_baseline_metrics(
            path, expected_training_count=4, expected_validation_count=3
        )
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("Population mismatch should fail.")
