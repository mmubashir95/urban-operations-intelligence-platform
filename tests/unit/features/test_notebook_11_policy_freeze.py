"""Regression tests for Notebook 11's strict policy-freeze boundary."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/11_feature_engineering_and_preprocessing.ipynb")


def _notebook() -> dict[str, object]:
    """Load Notebook 11 as its inspectable JSON document."""
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_notebook_contains_complete_policy_freeze_learning_sequence() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    for heading in (
        "## 1. Objective and scope boundary",
        "## 2. Notebook 10 handoff authority",
        "## 3. Feature-policy decision rules",
        "## 4. Approved first-pass candidates",
        "## 5. Alternative and redundant representations",
        "## 6. Conditional features",
        "## 7. Excluded features",
        "## 8. Frozen feature policy",
        "## 9. Policy validation",
        "## 10. Completion decision",
    ):
        assert heading in markdown
    assert "Approved candidate" in markdown or "approved candidate" in markdown
    assert "not" in markdown.lower() and "final model feature" in markdown.lower()
    legacy_stage_label = "Phase" + " 1"
    assert legacy_stage_label not in markdown


def test_notebook_code_does_not_implement_preprocessing_or_modelling() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    forbidden_code = (
        "SimpleImputer",
        "StandardScaler",
        "ColumnTransformer",
        "LogisticRegression",
        ".fit(",
        ".fit_transform(",
        ".transform(",
        "read_parquet(",
        "to_parquet(",
        "to_csv(",
    )
    assert all(token not in code for token in forbidden_code)
    assert "load_feature_policy" in code
    assert "validate_feature_policy_evidence" in code
    assert "fit_categorical_encoder" in code
    assert "build_preprocessing_composition" in code
    assert "before_states == after_states" in code
    legacy_identifier = "phase" + "_1"
    assert legacy_identifier not in code


def test_notebook_contains_deterministic_creation_sections_and_next_boundary() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Deterministic Feature Creation",
        "### A. Frozen policy handoff",
        "### B. Approved derivation allow-list",
        "### C. Source-column contract",
        "### D. Temporal derivation rules",
        "### E. Training derivation preview",
        "### F. Split consistency and feature-domain validation",
        "### G. Row and target reconciliation",
        "### H. Source immutability and implementation boundary",
        "### I. Completion decision",
    ):
        assert heading in all_text
    assert "derive_split_temporal_features" in all_text
    assert "build_temporal_validation_table" in all_text
    assert "build_feature_reconciliation_table" in all_text
    assert "Categorical Missing-Value Handling" in all_text
    assert "policy_decision" in all_text
    assert "creation_decision" in all_text
    assert "model_ready" in all_text


def test_notebook_contains_categorical_missing_learning_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Categorical Missing-Value Handling",
        "### A. Scope boundary",
        "### B. Why missingness remains explicit",
        "### C. Active categorical policy status",
        "### D. Missingness before transformation",
        "### E. Canonical `__MISSING__` rule",
        "### F. Training demonstration",
        "### G. Split consistency and reconciliation",
        "### H. Token-collision validation",
        "### I. Source immutability",
        "### J. Deferred numeric missingness",
        "### K. Completion decision",
    ):
        assert heading in all_text
    assert "replace_split_categorical_missing" in all_text
    assert "build_categorical_missing_evidence" in all_text
    assert "incident_zip" in all_text
    assert "blank normalization" in all_text
    assert "in-memory" in all_text
    assert "serialization/reload" in all_text
    assert "prediction-time status `UNRESOLVED`" in all_text
    assert "missing-value handling status `DEFERRED`" in all_text
    assert "fit imputation values on training data only" in all_text
    assert "No coordinate imputation" in all_text
    assert "Handle Rare and Unseen Categories" in all_text


def test_notebook_contains_rare_unseen_training_only_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Rare and Unseen Categorical Handling",
        "### A. Scope and production decision",
        "### B. Training-only cardinality and threshold evidence",
        "### C. Separate missing, rare, and unseen meanings",
        "### D. Explicit training-only fitted state",
        "### E. Governed no-op evidence and reconciliation",
        "### F. Phase 4 completion decision",
    ):
        assert heading in all_text
    assert "fit_rare_unseen_handler" in all_text
    assert "transform_split_rare_unseen" in all_text
    assert "build_rare_unseen_evidence" in all_text
    assert "COMPLETE — GOVERNED NO-OP" in all_text
    assert "TRAIN ONLY" in all_text
    assert "Categorical encoding" in all_text
    assert "categorical_encoding_implemented': False" in all_text


def test_notebook_contains_categorical_encoding_training_only_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Categorical Encoding",
        "### A. Scope and production decision",
        "### B. One-hot strategy and token vocabulary",
        "### C. Training-only fitted encoder state",
        "### D. Schema, sparsity, and reconciliation evidence",
        "### E. Phase 5 completion decision",
    ):
        assert heading in all_text
    assert "fit_categorical_encoder" in all_text
    assert "transform_split_categorical_encoder" in all_text
    assert "build_categorical_encoding_evidence" in all_text
    assert "OneHotEncoder" in all_text
    assert "handle_unknown=\"ignore\"" in all_text
    assert "drop=None" in all_text
    assert "sparse output" in all_text
    assert "explicit `__MISSING__`, `__RARE__`, and `__UNKNOWN__` columns" in all_text
    assert "COMPLETE -- GOVERNED NO-OP" in all_text
    assert "Numeric preprocessing" in all_text
    assert "numeric_preprocessing_implemented': False" in all_text
    assert "model_training_implemented': False" in all_text


def test_notebook_contains_numeric_preprocessing_pass_through_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Numeric Preprocessing",
        "### A. Scope and production decision",
        "### B. Per-feature strategy and cyclic semantics",
        "### C. Explicit numeric state and pass-through matrix",
        "### D. Domain validation, leakage exclusion, and row alignment",
        "### E. Phase 6 completion decision",
    ):
        assert heading in all_text
    assert "fit_numeric_preprocessor" in all_text
    assert "transform_split_numeric_preprocessor" in all_text
    assert "build_numeric_preprocessing_evidence" in all_text
    assert "created_hour" in all_text
    assert "created_day_of_week" in all_text
    assert "created_month" in all_text
    assert "is_weekend" in all_text
    assert "deterministic pass-through" in all_text
    assert "learned_statistics_absent" in all_text
    assert "latitude_longitude_deferred" in all_text
    assert "coordinate_imputation_implemented': False" in all_text
    assert "phase_7_composition_implemented': False" in all_text
    assert "Preprocessing Pipeline Composition" in all_text


def test_notebook_contains_preprocessing_composition_sequence() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for heading in (
        "## Preprocessing Pipeline Composition",
        "### A. Scope and upstream block contracts",
        "### B. Frozen composition state",
        "### C. Compose train, validation, and test matrices",
        "### D. Schema, fingerprint, leakage, and immutability checks",
        "### E. Phase 7 completion decision",
    ):
        assert heading in all_text
    assert "build_preprocessing_composition" in all_text
    assert "compose_split_preprocessing_blocks" in all_text
    assert "build_preprocessing_composition_evidence" in all_text
    assert "categorical columns first" in all_text
    assert "numeric columns second" in all_text
    assert "combined_feature_names" in all_text
    assert "upstream_fingerprints_recorded" in all_text
    assert "source_artifacts_unchanged" in all_text
    assert "conditional_geography_activated': False" in all_text
    assert "model_training_implemented': False" in all_text
    assert "model_evaluation_implemented': False" in all_text
    assert "Final Preprocessing Verification" in all_text


def test_notebook_contains_final_preprocessing_verification_and_stop_boundary() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for heading in (
        "## Final Preprocessing Verification",
        "### A. Final X / y / identifier separation",
        "### B. Frozen contract and authoritative feature schema",
        "### C. Per-split readiness and train-only variance evidence",
        "### D. Fingerprint, determinism, leakage, and immutability checks",
        "### E. Phase 8 completion decision",
        "## STOP — Baseline Modelling Boundary",
    ):
        assert heading in all_text
    for token in (
        "verify_preprocessing_contract",
        "build_final_feature_schema_evidence",
        "build_preprocessing_verification_evidence",
        "build_training_feature_variance_evidence",
        "VerifiedPreprocessingContract.from_dict",
        "X_train",
        "y_train",
        "identifier_train",
        "source_artifacts_unchanged",
        "FROZEN / MODEL-READY",
        "Baseline Modelling",
    ):
        assert token in all_text
    forbidden_model_calls = (
        "LogisticRegression(",
        "DummyClassifier(",
        ".predict(",
        ".predict_proba(",
        ".fit(",
        ".fit_transform(",
    )
    assert all(token not in code for token in forbidden_model_calls)
