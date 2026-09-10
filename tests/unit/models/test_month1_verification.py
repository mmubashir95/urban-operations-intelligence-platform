"""Tests for Month 1 artifact and report consistency verification."""

import pandas as pd
import pytest

from urban_ops.models.month1_verification import (
    Month1VerificationError,
    verify_report_consistency,
)


def test_verify_report_consistency_accepts_generated_metric_values(tmp_path) -> None:
    """Report verification checks key CSV-derived values in the Markdown text."""
    validation = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "pr_auc": 0.465888831,
                "roc_auc": 0.556797613,
            }
        ]
    )
    test = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "precision": 0.448863636,
                "recall": 0.040996367,
                "f1": 0.075130766,
                "pr_auc": 0.389849318,
                "roc_auc": 0.519168142,
                "brier_score": 0.229829429,
                "recall_at_10_percent": 0.131811105,
                "recall_at_20_percent": 0.245978204,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {"split": "validation", "bin_index": 0, "row_count": 1},
            {"split": "test", "bin_index": 0, "row_count": 1},
        ]
    )
    report = """
MONTH 1 COMPLETE
MONTH 2 READY
Selected baseline: `Logistic Regression`
threshold 0.5000
0.4659 0.5568 0.4489 0.0410 0.0751 0.3898 0.5192 0.2298 0.1318 0.2460
"""
    report_path = tmp_path / "report.md"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    calibration_path = tmp_path / "calibration.csv"
    report_path.write_text(report, encoding="utf-8")
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    verify_report_consistency(
        report_path=report_path,
        validation_results_path=validation_path,
        test_results_path=test_path,
        calibration_path=calibration_path,
    )


def test_verify_report_consistency_rejects_stale_report_value(tmp_path) -> None:
    """A missing rounded metric causes the consistency check to fail."""
    validation = pd.DataFrame(
        [{"model": "Logistic Regression", "pr_auc": 0.4659, "roc_auc": 0.5568}]
    )
    test = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "precision": 0.4489,
                "recall": 0.0410,
                "f1": 0.0751,
                "pr_auc": 0.3898,
                "roc_auc": 0.5192,
                "brier_score": 0.2298,
                "recall_at_10_percent": 0.1318,
                "recall_at_20_percent": 0.2460,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {"split": "validation", "bin_index": 0, "row_count": 1},
            {"split": "test", "bin_index": 0, "row_count": 1},
        ]
    )
    report_path = tmp_path / "report.md"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    calibration_path = tmp_path / "calibration.csv"
    report_path.write_text(
        "MONTH 1 COMPLETE\nMONTH 2 READY\nSelected baseline: `Logistic Regression`\nthreshold 0.5000\n",
        encoding="utf-8",
    )
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    with pytest.raises(Month1VerificationError, match="validation PR-AUC"):
        verify_report_consistency(
            report_path=report_path,
            validation_results_path=validation_path,
            test_results_path=test_path,
            calibration_path=calibration_path,
        )
