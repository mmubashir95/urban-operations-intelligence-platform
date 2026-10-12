"""Synthetic integration tests for the Phase 7.3 duration baseline workflow."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from urban_ops.models import duration_baseline_workflow as workflow
from urban_ops.models.duration_baseline_reporting import METRICS_FILENAME, REPORT_FILENAME
from urban_ops.splitting.models import SplitBoundaries


def test_duration_baseline_uses_train_only_and_split_specific_cutoffs(
    tmp_path, monkeypatch
) -> None:
    boundaries = SplitBoundaries(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2024-09-01", tz="UTC"),
        pd.Timestamp("2024-09-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    )
    config = SimpleNamespace(
        processed_root=tmp_path / "processed",
        latest_pointer=tmp_path / "latest.json",
        timestamp_column="created_date",
        boundaries=boundaries,
    )
    frame = pd.DataFrame({
        "unique_key": [
            "train-zero", "train-mid", "train-long", "train-unobserved",
            "validation-low", "validation-mid", "validation-high",
            "validation-unobserved", "test-never-accessed",
        ],
        "created_date": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z", "2024-03-20T00:00:00Z",
            "2024-04-02T00:00:00Z", "2024-04-03T00:00:00Z",
            "2024-04-04T00:00:00Z", "2024-08-20T00:00:00Z",
            "2024-09-01T00:00:00Z",
        ], utc=True),
        "closed_date": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-02T10:00:00Z",
            "2024-01-03T20:00:00Z", "2024-04-02T00:00:00Z",
            "2024-04-02T05:00:00Z", "2024-04-03T12:00:00Z",
            "2024-04-05T01:00:00Z", "2024-09-02T00:00:00Z",
            "2024-09-01T01:00:00Z",
        ], utc=True),
        "status": ["Closed"] * 9,
    })
    original = frame.copy(deep=True)
    source_path = tmp_path / "cleaned_service_requests.parquet"
    monkeypatch.setattr(workflow, "load_split_config", lambda _: config)
    monkeypatch.setattr(workflow, "resolve_cleaning_run", lambda **_: tmp_path)
    monkeypatch.setattr(
        workflow, "load_verified_cleaned_source", lambda _: (frame, source_path)
    )
    report_root = tmp_path / "reports" / "month_2"

    result = workflow.run_duration_baseline_workflow(
        split_config_path=tmp_path / "splits.yaml", report_root=report_root
    )

    assert result.training.historical["unique_key"].tolist() == [
        "train-zero", "train-mid", "train-long", "train-unobserved"
    ]
    assert result.training.eligible_durations.tolist() == [0.0, 10.0, 20.0]
    assert result.validation.historical["unique_key"].tolist() == [
        "validation-low", "validation-mid", "validation-high", "validation-unobserved"
    ]
    assert result.validation.eligible_durations.tolist() == [5.0, 12.0, 25.0]
    assert result.fitted_median_hours == 10.0
    np.testing.assert_array_equal(result.validation_predictions, [10.0, 10.0, 10.0])
    assert result.validation_predictions.shape == result.validation_targets.shape
    assert result.validation_metrics.mae_hours == 22 / 3
    assert result.validation_metrics.rmse_hours == np.sqrt(254 / 3)
    assert result.report_path == report_root / REPORT_FILENAME
    assert result.metrics_path == report_root / METRICS_FILENAME
    assert result.report_path.is_file() and result.metrics_path.is_file()
    metrics = pd.read_csv(result.metrics_path)
    assert tuple(metrics.columns) == ("metric", "value", "unit")
    assert metrics.loc[
        metrics["metric"].eq("fitted_training_median_hours"), "value"
    ].item() == 10.0
    report = result.report_path.read_text(encoding="utf-8")
    assert "Test data accessed: No" in report
    assert "2024-09-01T00:00:00+00:00" in report
    assert "Validation MAE" in report and "Validation RMSE" in report
    pd.testing.assert_frame_equal(frame, original)
