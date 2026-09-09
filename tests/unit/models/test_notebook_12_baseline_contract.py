"""Notebook safety checks for Phase 9 baseline modelling contract evidence."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/12_baseline_modelling_contract.ipynb")


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_contains_phase_9_contract_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## A. Notebook Setup",
        "## B. Governed Configuration",
        "## C. Load Authoritative Chronological Splits",
        "## D. Rebuild Phase 8 Verified Outputs In Memory",
        "## E. Frozen Feature Schema",
        "## F. Phase 8 Readiness Evidence",
        "## G. Phase 9 Verification Gate",
        "## H. Split And Train-Only Target Evidence",
        "## I. Fingerprint And Serialization Evidence",
        "## J. Immutability And Scope Evidence",
        "## K. Phase 9 Completion Decision",
        "## STOP — Baseline Training Boundary",
    ):
        assert heading in all_text
    for token in (
        "load_verified_split",
        "derive_split_temporal_features",
        "verify_preprocessing_contract",
        "verify_baseline_modelling_contract",
        "VerifiedBaselineModellingContract.from_dict",
        "build_baseline_modelling_contract_evidence",
        "build_preprocessing_verification_evidence",
        "build_training_feature_variance_evidence",
        "MODEL_INPUTS_VERIFIED",
        "verified_preprocessing_contract",
        "phase_8_contract_fingerprint",
        "preprocessing_schema_fingerprint",
        "train_positive_class_prevalence",
        "source_artifacts_unchanged",
        "model_training_implemented",
        "model_evaluation_implemented",
        "preprocessing_modified",
        "Phase 10 — Baseline Model Training",
    ):
        assert token in all_text


def test_notebook_does_not_train_predict_or_calculate_metrics() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    forbidden_tokens = (
        "LogisticRegression(",
        "DummyClassifier(",
        ".fit(",
        ".fit_transform(",
        ".predict(",
        ".predict_proba(",
        "roc_auc",
        "average_precision",
        "precision_recall",
        "confusion_matrix",
        "classification_report",
    )
    assert all(token not in code for token in forbidden_tokens)
