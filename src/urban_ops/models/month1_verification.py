"""Verify Month 1 closure artifacts and report consistency.

This module checks generated baseline outputs without training a new model. It
is intended for the `make verify-month1` completion gate after the baseline
workflow has regenerated reports and artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

import joblib
import pandas as pd

from urban_ops.utils.paths import PROJECT_ROOT


EXPECTED_SELECTED_MODEL: Final = "Logistic Regression"
EXPECTED_THRESHOLD: Final = 0.5
EXPECTED_FEATURE_NAMES: Final = (
    "created_hour",
    "created_day_of_week",
    "created_month",
    "is_weekend",
)


class Month1VerificationError(RuntimeError):
    """Raised when Month 1 artifacts are missing or internally inconsistent."""


def _require_file(path: Path) -> None:
    """Raise if a required artifact does not exist."""
    if not path.is_file():
        raise Month1VerificationError(f"Required Month 1 artifact is missing: {path}")


def _contains(report: str, text: str) -> None:
    """Require exact text to appear in the generated Month 1 report."""
    if text not in report:
        raise Month1VerificationError(f"Month 1 report is missing expected text: {text}")


def _contains_metric(report: str, value: float, *, label: str) -> None:
    """Require a rounded metric value to appear in the generated report."""
    rendered = f"{value:.4f}"
    if rendered not in report:
        raise Month1VerificationError(
            f"Month 1 report is missing {label} value {rendered}."
        )


def verify_selected_model_artifact(artifact_path: Path) -> None:
    """Load and validate the selected baseline artifact contract."""
    _require_file(artifact_path)
    payload = joblib.load(artifact_path)
    selected_model = payload.get("selected_model_name")
    threshold = float(payload.get("selected_threshold"))
    feature_names = tuple(payload.get("feature_names", ()))
    if selected_model != EXPECTED_SELECTED_MODEL:
        raise Month1VerificationError(
            f"Selected model mismatch: expected {EXPECTED_SELECTED_MODEL}, "
            f"observed {selected_model}."
        )
    if threshold != EXPECTED_THRESHOLD:
        raise Month1VerificationError(
            f"Selected threshold mismatch: expected {EXPECTED_THRESHOLD}, "
            f"observed {threshold}."
        )
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise Month1VerificationError(
            f"Feature order mismatch: expected {EXPECTED_FEATURE_NAMES}, "
            f"observed {feature_names}."
        )
    contract = payload.get("phase_9_contract", {})
    if tuple(contract.get("ordered_feature_names", ())) != EXPECTED_FEATURE_NAMES:
        raise Month1VerificationError("Artifact Phase 9 feature order is inconsistent.")


def verify_report_consistency(
    *,
    report_path: Path,
    validation_results_path: Path,
    test_results_path: Path,
    calibration_path: Path,
) -> None:
    """Verify key generated report values against structured CSV outputs."""
    for path in (report_path, validation_results_path, test_results_path, calibration_path):
        _require_file(path)
    report = report_path.read_text(encoding="utf-8")
    validation = pd.read_csv(validation_results_path)
    test = pd.read_csv(test_results_path)
    calibration = pd.read_csv(calibration_path)
    selected = validation.loc[validation["model"].eq(EXPECTED_SELECTED_MODEL)]
    if len(selected) != 1:
        raise Month1VerificationError("Validation CSV must contain one selected row.")
    selected_row = selected.iloc[0]
    test_row = test.iloc[0]

    _contains(report, "MONTH 1 COMPLETE")
    _contains(report, "MONTH 2 READY")
    _contains(report, f"Selected baseline: `{EXPECTED_SELECTED_MODEL}`")
    _contains(report, f"threshold {EXPECTED_THRESHOLD:.4f}")
    for label, value in (
        ("validation PR-AUC", selected_row["pr_auc"]),
        ("validation ROC-AUC", selected_row["roc_auc"]),
        ("test precision", test_row["precision"]),
        ("test recall", test_row["recall"]),
        ("test F1", test_row["f1"]),
        ("test PR-AUC", test_row["pr_auc"]),
        ("test ROC-AUC", test_row["roc_auc"]),
        ("test Brier", test_row["brier_score"]),
        ("test Recall@10%", test_row["recall_at_10_percent"]),
        ("test Recall@20%", test_row["recall_at_20_percent"]),
    ):
        _contains_metric(report, float(value), label=label)
    if calibration.empty or not {"validation", "test"}.issubset(set(calibration["split"])):
        raise Month1VerificationError("Calibration CSV must include validation and test bins.")


def verify_month1_artifacts(project_root: Path = PROJECT_ROOT) -> None:
    """Verify all required Month 1 closure artifacts."""
    required = [
        project_root / "models/baselines/selected_month_1_baseline.joblib",
        project_root / "reports/baseline_results.md",
        project_root / "reports/month_1_baseline_report.md",
        project_root / "reports/tables/baseline_validation_results.csv",
        project_root / "reports/tables/baseline_test_results.csv",
        project_root / "reports/tables/baseline_subgroup_results.csv",
        project_root / "reports/tables/logistic_regression_calibration.csv",
        project_root / "reports/figures/logistic_regression_calibration.png",
    ]
    for path in required:
        _require_file(path)
    verify_selected_model_artifact(
        project_root / "models/baselines/selected_month_1_baseline.joblib"
    )
    verify_report_consistency(
        report_path=project_root / "reports/month_1_baseline_report.md",
        validation_results_path=(
            project_root / "reports/tables/baseline_validation_results.csv"
        ),
        test_results_path=project_root / "reports/tables/baseline_test_results.csv",
        calibration_path=(
            project_root / "reports/tables/logistic_regression_calibration.csv"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the Month 1 verification command-line interface."""
    return argparse.ArgumentParser(description=__doc__)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Month 1 artifact verification from the command line."""
    _parser().parse_args(argv)
    verify_month1_artifacts()
    print("MONTH 1 COMPLETE")
    print("MONTH 2 READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
