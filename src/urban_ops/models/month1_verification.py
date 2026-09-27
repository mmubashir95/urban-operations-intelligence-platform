"""Verify Month 1 closure artifacts and report consistency.

This module checks generated baseline outputs without training a new model. It
is intended for the `make verify-month1` completion gate after the baseline
workflow has regenerated reports and artifacts.

The governed Logistic Regression threshold is never a constant here: the gate
loads and validates the frozen threshold decision artifact and requires every
governed artifact to agree with ``decision.selected_threshold``. The gate only
checks consistency; it never selects or tunes a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Final, Sequence

import joblib
import pandas as pd

from urban_ops.models.baseline_workflow import (
    FROZEN_THRESHOLD_DECISION_PATH,
    load_frozen_threshold_decision,
)
from urban_ops.models.evaluation import (
    EvaluationError,
    FrozenThresholdDecision,
    validate_frozen_threshold_decision,
)
from urban_ops.utils.paths import PROJECT_ROOT


EXPECTED_SELECTED_MODEL: Final = "Logistic Regression"
# Matches text that presents a value as the selected or frozen threshold, e.g.
# "frozen threshold 0.4900", "Selected threshold: 0.4900", "frozen threshold `0.49`".
# Descriptive wording such as "default/reference cutoff 0.50" is not matched.
GOVERNED_THRESHOLD_CLAIM: Final = re.compile(
    r"(?:frozen|selected)\s+threshold\s*[:=]?\s*`?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CONFUSION_COLUMNS: Final = (
    ("true_positive", "true_positives"),
    ("false_positive", "false_positives"),
    ("true_negative", "true_negatives"),
    ("false_negative", "false_negatives"),
)
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


def load_governed_threshold_decision(
    project_root: Path = PROJECT_ROOT,
) -> FrozenThresholdDecision:
    """Load and validate the authoritative frozen threshold decision."""
    path = project_root / FROZEN_THRESHOLD_DECISION_PATH.relative_to(PROJECT_ROOT)
    _require_file(path)
    try:
        return validate_frozen_threshold_decision(
            load_frozen_threshold_decision(path)
        )
    except (EvaluationError, TypeError, ValueError) as exc:
        raise Month1VerificationError(
            f"Frozen threshold decision is invalid: {exc}"
        ) from exc


def _require_threshold(observed: object, governed: float, *, source: str) -> None:
    """Require a governed artifact to carry the frozen decision threshold."""
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise Month1VerificationError(
            f"{source} threshold must be numeric, observed {observed!r}."
        )
    if not math.isclose(float(observed), governed, rel_tol=0.0, abs_tol=1e-12):
        raise Month1VerificationError(
            f"{source} threshold mismatch: frozen decision is {governed}, "
            f"observed {observed}."
        )


def verify_governed_threshold_claims(
    text: str,
    governed_threshold: float,
    *,
    source: str,
    require_claim: bool = True,
) -> None:
    """Require every selected/frozen threshold statement to match the decision.

    Descriptive default/reference cutoffs are not selected or frozen claims and
    are therefore not constrained.
    """
    claims = GOVERNED_THRESHOLD_CLAIM.findall(text)
    if require_claim and not claims:
        raise Month1VerificationError(
            f"{source} does not state the frozen threshold."
        )
    for claim in claims:
        if not math.isclose(float(claim), governed_threshold, abs_tol=1e-12):
            raise Month1VerificationError(
                f"{source} labels {claim} as the selected/frozen threshold; "
                f"the frozen decision is {governed_threshold}."
            )


def verify_selected_model_artifact(
    artifact_path: Path,
    *,
    governed_threshold: float,
) -> None:
    """Load and validate the selected baseline artifact contract."""
    _require_file(artifact_path)
    payload = joblib.load(artifact_path)
    selected_model = payload.get("selected_model_name")
    threshold = payload.get("selected_threshold")
    feature_names = tuple(payload.get("feature_names", ()))
    if selected_model != EXPECTED_SELECTED_MODEL:
        raise Month1VerificationError(
            f"Selected model mismatch: expected {EXPECTED_SELECTED_MODEL}, "
            f"observed {selected_model}."
        )
    _require_threshold(threshold, governed_threshold, source="Selected model artifact")
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise Month1VerificationError(
            f"Feature order mismatch: expected {EXPECTED_FEATURE_NAMES}, "
            f"observed {feature_names}."
        )
    contract = payload.get("phase_9_contract", {})
    if tuple(contract.get("ordered_feature_names", ())) != EXPECTED_FEATURE_NAMES:
        raise Month1VerificationError("Artifact Phase 9 feature order is inconsistent.")


def verify_selected_model_metadata(
    metadata_path: Path,
    *,
    governed_threshold: float,
) -> None:
    """Require the selected-model metadata to restate the frozen decision."""
    _require_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("selected_model_name") != EXPECTED_SELECTED_MODEL:
        raise Month1VerificationError("Selected model metadata names the wrong model.")
    _require_threshold(
        metadata.get("selected_threshold"),
        governed_threshold,
        source="Selected model metadata",
    )


def verify_governed_test_metrics(
    *,
    test_results_path: Path,
    frozen_test_results_path: Path,
    decision: FrozenThresholdDecision,
) -> None:
    """Require final test metrics to equal the Phase 4.7 frozen evaluation."""
    for path in (test_results_path, frozen_test_results_path):
        _require_file(path)
    final_row = pd.read_csv(test_results_path).iloc[0]
    frozen_row = pd.read_csv(frozen_test_results_path).iloc[0]
    _require_threshold(
        float(frozen_row["threshold"]),
        decision.selected_threshold,
        source="Phase 4.7 frozen test evidence",
    )
    if frozen_row["selection_split"] != decision.selected_on_split:
        raise Month1VerificationError(
            "Phase 4.7 evidence does not record validation selection."
        )
    for final_column, frozen_column in (
        *_CONFUSION_COLUMNS,
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
    ):
        if not math.isclose(
            float(final_row[final_column]),
            float(frozen_row[frozen_column]),
            abs_tol=1e-12,
        ):
            raise Month1VerificationError(
                f"Final test {final_column} does not match the frozen-threshold "
                "evaluation."
            )


def verify_governed_subgroup_metrics(
    *,
    subgroup_results_path: Path,
    test_results_path: Path,
    decision: FrozenThresholdDecision,
) -> None:
    """Require subgroup errors to come from the frozen-threshold classifications.

    Any subgroup column whose groups cover the whole split must sum to that
    split's governed false positives and false negatives: validation totals
    come from the frozen decision and test totals from the final test row.
    """
    for path in (subgroup_results_path, test_results_path):
        _require_file(path)
    subgroups = pd.read_csv(subgroup_results_path)
    test_row = pd.read_csv(test_results_path).iloc[0]
    expected = {
        "validation": (
            decision.true_positives
            + decision.false_positives
            + decision.true_negatives
            + decision.false_negatives,
            decision.false_positives,
            decision.false_negatives,
        ),
        "test": (
            int(test_row["row_count"]),
            int(test_row["false_positive"]),
            int(test_row["false_negative"]),
        ),
    }
    for split, (row_count, false_positives, false_negatives) in expected.items():
        totals = (
            subgroups.loc[subgroups["split"].eq(split)]
            .groupby("group_column")[["row_count", "false_positives", "false_negatives"]]
            .sum()
        )
        complete = totals.loc[totals["row_count"].eq(row_count)]
        if complete.empty:
            raise Month1VerificationError(
                f"No {split} subgroup column covers every {split} row."
            )
        for column, row in complete.iterrows():
            if (
                int(row["false_positives"]) != false_positives
                or int(row["false_negatives"]) != false_negatives
            ):
                raise Month1VerificationError(
                    f"{split} subgroup errors for {column} do not match the "
                    "frozen-threshold classifications."
                )


def verify_report_consistency(
    *,
    report_path: Path,
    validation_results_path: Path,
    test_results_path: Path,
    calibration_path: Path,
    governed_threshold: float,
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
    _contains(report, f"frozen threshold {governed_threshold:.4f}")
    verify_governed_threshold_claims(
        report,
        governed_threshold,
        source="Month 1 report",
    )
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
    decision = load_governed_threshold_decision(project_root)
    governed_threshold = decision.selected_threshold
    verify_selected_model_artifact(
        project_root / "models/baselines/selected_month_1_baseline.joblib",
        governed_threshold=governed_threshold,
    )
    verify_selected_model_metadata(
        project_root / "models/baselines/selected_month_1_baseline_metadata.json",
        governed_threshold=governed_threshold,
    )
    verify_governed_test_metrics(
        test_results_path=project_root / "reports/tables/baseline_test_results.csv",
        frozen_test_results_path=(
            project_root / "reports/tables/logistic_regression_test_frozen_threshold.csv"
        ),
        decision=decision,
    )
    verify_governed_subgroup_metrics(
        subgroup_results_path=(
            project_root / "reports/tables/baseline_subgroup_results.csv"
        ),
        test_results_path=project_root / "reports/tables/baseline_test_results.csv",
        decision=decision,
    )
    for path, source in (
        (project_root / "reports/baseline_results.md", "Baseline report"),
        (project_root / "README.md", "README"),
    ):
        _require_file(path)
        verify_governed_threshold_claims(
            path.read_text(encoding="utf-8"),
            governed_threshold,
            source=source,
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
        governed_threshold=governed_threshold,
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
