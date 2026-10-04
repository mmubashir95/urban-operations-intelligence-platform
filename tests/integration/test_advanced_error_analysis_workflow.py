"""Integration coverage for governed Phase 6 artifact generation."""
from hashlib import sha256

import pandas as pd

from urban_ops.models import advanced_error_analysis_workflow as workflow
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def test_real_workflow_loads_frozen_model_and_writes_applicable_evidence(tmp_path, monkeypatch):
    frozen = (workflow.PHASE3_PATH, workflow.PHASE4_PATH, workflow.MODEL_PATH, workflow.MODEL_METADATA_PATH)
    before = {path: _digest(path) for path in frozen}
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Phase 6 must not fit the base model.")
    monkeypatch.setattr(GradientBoostedRiskModel, "fit", forbidden_fit)
    output, figures = tmp_path / "reports", tmp_path / "figures"
    result = workflow.run_advanced_error_analysis_workflow(output_directory=output, figure_directory=figures,
                                                            phase5_directory=workflow.OUTPUT_DIRECTORY)
    assert result["model_metadata"]["candidate_id"] == "shallow"
    assert len(result["dataset"]) == 5499
    summary = result["summary"].iloc[0]
    assert summary.TP + summary.TN + summary.FP + summary.FN == summary.sample_count
    expected_csv = ("phase_6_error_analysis_dataset.csv", "phase_6_confusion_summary.csv", "phase_6_false_negatives.csv",
                    "phase_6_false_positives.csv", "phase_6_errors_by_month.csv", "phase_6_errors_by_geography.csv",
                    "phase_6_errors_by_score_range.csv", "phase_6_high_confidence_errors.csv",
                    "phase_6_correct_vs_incorrect.csv", "phase_6_drift_summary.csv", "phase_6_failure_modes.csv")
    assert all((output / name).is_file() for name in expected_csv)
    assert all((output / name).is_file() for name in ("phase_6_advanced_error_analysis.md", "phase_6_model_improvement_evidence.md", "phase_6_completion_gate.md"))
    assert not (output / "phase_6_errors_by_agency.csv").exists()
    assert not (output / "phase_6_errors_by_complaint_category.csv").exists()
    report = (output / "phase_6_advanced_error_analysis.md").read_text()
    assert "not applicable" in report and "not an operational threshold" in report and "do not establish causes" in report
    confusion = pd.read_csv(output / "phase_6_confusion_summary.csv")
    assert confusion.decision_rule.tolist() == ["top_10_percent_raw", "descriptive_raw_score_gte_0_5_non_operational"]
    assert {path: _digest(path) for path in frozen} == before
