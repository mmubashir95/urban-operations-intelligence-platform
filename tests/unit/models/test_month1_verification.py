"""Tests for Month 1 artifact and report consistency verification."""

import json
import shutil

import joblib
import pandas as pd
import pytest

from urban_ops.models.baseline_workflow import FROZEN_THRESHOLD_DECISION_PATH
from urban_ops.models.month1_verification import (
    EXPECTED_FEATURE_NAMES,
    Month1VerificationError,
    load_governed_threshold_decision,
    verify_governed_subgroup_metrics,
    verify_governed_test_metrics,
    verify_governed_threshold_claims,
    verify_report_consistency,
    verify_selected_model_artifact,
    verify_selected_model_metadata,
)
from urban_ops.utils.paths import PROJECT_ROOT


LEGACY_THRESHOLD = 0.50


@pytest.fixture
def decision():
    """The real validated repository frozen threshold decision."""
    return load_governed_threshold_decision()


def _write_report_inputs(tmp_path, report: str):
    """Write a report plus minimal CSVs matching the report's metric values."""
    validation = pd.DataFrame(
        [{"model": "Logistic Regression", "pr_auc": 0.4659, "roc_auc": 0.5568}]
    )
    test = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "precision": 0.4511,
                "recall": 0.0550,
                "f1": 0.0981,
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
    paths = {
        "report_path": tmp_path / "report.md",
        "validation_results_path": tmp_path / "validation.csv",
        "test_results_path": tmp_path / "test.csv",
        "calibration_path": tmp_path / "calibration.csv",
    }
    paths["report_path"].write_text(report, encoding="utf-8")
    validation.to_csv(paths["validation_results_path"], index=False)
    test.to_csv(paths["test_results_path"], index=False)
    calibration.to_csv(paths["calibration_path"], index=False)
    return paths


def _report(threshold_text: str, *, extra: str = "") -> str:
    return (
        "MONTH 1 COMPLETE\nMONTH 2 READY\n"
        "Selected baseline: `Logistic Regression`\n"
        f"with the frozen threshold {threshold_text}.\n"
        "0.4659 0.5568 0.4511 0.0550 0.0981 0.3898 0.5192 0.2298 0.1318 0.2460\n"
        f"{extra}"
    )


def test_gate_loads_real_validated_frozen_decision(decision) -> None:
    """The gate's threshold authority is the validated repository JSON."""
    payload = json.loads(FROZEN_THRESHOLD_DECISION_PATH.read_text(encoding="utf-8"))

    assert decision.selected_threshold == payload["selected_threshold"]
    assert decision.selected_on_split == "validation"
    assert decision.frozen is True


def test_gate_rejects_invalid_frozen_decision(tmp_path) -> None:
    """A tampered decision artifact cannot become the gate's authority."""
    relative = FROZEN_THRESHOLD_DECISION_PATH.relative_to(PROJECT_ROOT)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    payload = json.loads(FROZEN_THRESHOLD_DECISION_PATH.read_text(encoding="utf-8"))
    payload["selected_threshold"] = LEGACY_THRESHOLD
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Month1VerificationError, match="Frozen threshold decision is invalid"):
        load_governed_threshold_decision(tmp_path)


def test_verify_report_consistency_accepts_frozen_threshold(tmp_path, decision) -> None:
    """A report stating the decision threshold and CSV metrics passes."""
    paths = _write_report_inputs(
        tmp_path,
        _report(
            f"{decision.selected_threshold:.4f}",
            extra="The descriptive default/reference cutoff 0.50 is for comparison.\n",
        ),
    )

    verify_report_consistency(
        **paths,
        governed_threshold=decision.selected_threshold,
    )


def test_verify_report_consistency_rejects_frozen_legacy_threshold(
    tmp_path,
    decision,
) -> None:
    """A report claiming frozen 0.50 fails against the frozen decision."""
    paths = _write_report_inputs(
        tmp_path,
        _report(
            f"{decision.selected_threshold:.4f}",
            extra=f"At the frozen threshold {LEGACY_THRESHOLD:.4f}, recall is low.\n",
        ),
    )

    with pytest.raises(Month1VerificationError, match="selected/frozen threshold"):
        verify_report_consistency(
            **paths,
            governed_threshold=decision.selected_threshold,
        )


def test_verify_report_consistency_rejects_stale_report_value(
    tmp_path,
    decision,
) -> None:
    """A missing rounded metric causes the consistency check to fail."""
    paths = _write_report_inputs(tmp_path, _report("0.4900"))
    paths["report_path"].write_text(
        "MONTH 1 COMPLETE\nMONTH 2 READY\nSelected baseline: `Logistic Regression`\n"
        f"frozen threshold {decision.selected_threshold:.4f}\n",
        encoding="utf-8",
    )

    with pytest.raises(Month1VerificationError, match="validation PR-AUC"):
        verify_report_consistency(
            **paths,
            governed_threshold=decision.selected_threshold,
        )


@pytest.mark.parametrize(
    "text",
    [
        "Phase 4.1 evaluates the default/reference cutoff 0.50.",
        "The Precision >= 0.50 policy selects the same candidate.",
        "The sweep includes a 0.50 row.",
    ],
)
def test_descriptive_legacy_references_do_not_fail_the_gate(text, decision) -> None:
    """Descriptive 0.50 references are not selected/frozen threshold claims."""
    verify_governed_threshold_claims(
        f"frozen threshold `{decision.selected_threshold}`\n{text}",
        decision.selected_threshold,
        source="README",
    )


def test_readme_claiming_frozen_legacy_threshold_fails(decision) -> None:
    """README governed wording must follow the frozen decision."""
    with pytest.raises(Month1VerificationError, match="README labels 0.5"):
        verify_governed_threshold_claims(
            "Logistic Regression is selected with frozen threshold `0.5`;",
            decision.selected_threshold,
            source="README",
        )


def _write_selected_artifacts(tmp_path, *, joblib_threshold, metadata_threshold):
    payload = {
        "selected_model_name": "Logistic Regression",
        "selected_threshold": joblib_threshold,
        "feature_names": EXPECTED_FEATURE_NAMES,
        "phase_9_contract": {"ordered_feature_names": list(EXPECTED_FEATURE_NAMES)},
    }
    artifact_path = tmp_path / "selected.joblib"
    joblib.dump(payload, artifact_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selected_model_name": "Logistic Regression",
                "selected_threshold": metadata_threshold,
            }
        ),
        encoding="utf-8",
    )
    return artifact_path, metadata_path


def test_selected_artifacts_at_frozen_threshold_pass(tmp_path, decision) -> None:
    """Persisted artifacts carrying the decision threshold are accepted."""
    artifact_path, metadata_path = _write_selected_artifacts(
        tmp_path,
        joblib_threshold=decision.selected_threshold,
        metadata_threshold=decision.selected_threshold,
    )

    verify_selected_model_artifact(
        artifact_path, governed_threshold=decision.selected_threshold
    )
    verify_selected_model_metadata(
        metadata_path, governed_threshold=decision.selected_threshold
    )


def test_selected_artifact_at_legacy_threshold_fails(tmp_path, decision) -> None:
    """A selected joblib persisted at 0.50 fails the frozen-decision check."""
    artifact_path, _ = _write_selected_artifacts(
        tmp_path,
        joblib_threshold=LEGACY_THRESHOLD,
        metadata_threshold=decision.selected_threshold,
    )

    with pytest.raises(Month1VerificationError, match="Selected model artifact threshold"):
        verify_selected_model_artifact(
            artifact_path, governed_threshold=decision.selected_threshold
        )


def test_selected_metadata_mismatch_fails(tmp_path, decision) -> None:
    """Selected-model metadata cannot disagree with the frozen decision."""
    _, metadata_path = _write_selected_artifacts(
        tmp_path,
        joblib_threshold=decision.selected_threshold,
        metadata_threshold=LEGACY_THRESHOLD,
    )

    with pytest.raises(Month1VerificationError, match="Selected model metadata threshold"):
        verify_selected_model_metadata(
            metadata_path, governed_threshold=decision.selected_threshold
        )


def _governed_tables(tmp_path, decision, *, final_false_negatives=None):
    """Write consistent final, Phase 4.7, and subgroup tables for one scenario."""
    tp, fp, tn, fn = 10, 5, 70, 15
    final = pd.DataFrame(
        [
            {
                "row_count": tp + fp + tn + fn,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn if final_false_negatives is None else final_false_negatives,
                "precision": 10 / 15,
                "recall": 10 / 25,
                "f1": 0.5,
            }
        ]
    )
    frozen = pd.DataFrame(
        [
            {
                "threshold": decision.selected_threshold,
                "selection_split": decision.selected_on_split,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": 10 / 15,
                "recall": 10 / 25,
                "f1": 0.5,
            }
        ]
    )
    validation_rows = (
        decision.true_positives
        + decision.false_positives
        + decision.true_negatives
        + decision.false_negatives
    )
    subgroups = pd.DataFrame(
        [
            {
                "split": "validation",
                "group_column": "is_weekend",
                "row_count": validation_rows,
                "false_positives": decision.false_positives,
                "false_negatives": decision.false_negatives,
            },
            {
                "split": "test",
                "group_column": "is_weekend",
                "row_count": tp + fp + tn + fn,
                "false_positives": fp,
                "false_negatives": fn,
            },
        ]
    )
    paths = {
        "final": tmp_path / "final.csv",
        "frozen": tmp_path / "frozen.csv",
        "subgroups": tmp_path / "subgroups.csv",
    }
    final.to_csv(paths["final"], index=False)
    frozen.to_csv(paths["frozen"], index=False)
    subgroups.to_csv(paths["subgroups"], index=False)
    return paths


def test_governed_metrics_and_subgroups_pass_when_consistent(tmp_path, decision) -> None:
    """Final and subgroup errors agreeing with the frozen evaluation pass."""
    paths = _governed_tables(tmp_path, decision)

    verify_governed_test_metrics(
        test_results_path=paths["final"],
        frozen_test_results_path=paths["frozen"],
        decision=decision,
    )
    verify_governed_subgroup_metrics(
        subgroup_results_path=paths["subgroups"],
        test_results_path=paths["final"],
        decision=decision,
    )


def test_final_metrics_from_another_threshold_fail(tmp_path, decision) -> None:
    """Final test counts that differ from Phase 4.7 frozen evidence fail."""
    paths = _governed_tables(tmp_path, decision, final_false_negatives=16)

    with pytest.raises(Month1VerificationError, match="false_negative"):
        verify_governed_test_metrics(
            test_results_path=paths["final"],
            frozen_test_results_path=paths["frozen"],
            decision=decision,
        )


def test_phase_4_7_evidence_at_legacy_threshold_fails(tmp_path, decision) -> None:
    """Phase 4.7 evidence must record the frozen decision threshold."""
    paths = _governed_tables(tmp_path, decision)
    frozen = pd.read_csv(paths["frozen"]).assign(threshold=LEGACY_THRESHOLD)
    frozen.to_csv(paths["frozen"], index=False)

    with pytest.raises(Month1VerificationError, match="Phase 4.7"):
        verify_governed_test_metrics(
            test_results_path=paths["final"],
            frozen_test_results_path=paths["frozen"],
            decision=decision,
        )


def test_subgroups_from_another_threshold_fail(tmp_path, decision) -> None:
    """Validation subgroup errors must equal the frozen decision's errors."""
    paths = _governed_tables(tmp_path, decision)
    subgroups = pd.read_csv(paths["subgroups"])
    subgroups.loc[subgroups["split"].eq("validation"), "false_positives"] += 1
    subgroups.to_csv(paths["subgroups"], index=False)

    with pytest.raises(Month1VerificationError, match="validation subgroup errors"):
        verify_governed_subgroup_metrics(
            subgroup_results_path=paths["subgroups"],
            test_results_path=paths["final"],
            decision=decision,
        )


def test_gate_copy_of_repository_decision_is_accepted(tmp_path, decision) -> None:
    """The gate resolves the decision relative to the given project root."""
    target = tmp_path / FROZEN_THRESHOLD_DECISION_PATH.relative_to(PROJECT_ROOT)
    target.parent.mkdir(parents=True)
    shutil.copy(FROZEN_THRESHOLD_DECISION_PATH, target)

    assert load_governed_threshold_decision(tmp_path) == decision
