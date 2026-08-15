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


def test_notebook_code_does_not_implement_feature_creation_or_modelling() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    forbidden_code = (
        "sklearn",
        "SimpleImputer",
        "OneHotEncoder",
        "StandardScaler",
        "ColumnTransformer",
        "LogisticRegression",
        "X_train",
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
    assert "before_states == after_states" in code
    legacy_identifier = "phase" + "_1"
    assert legacy_identifier not in code


def test_notebook_declares_deterministic_feature_creation_as_next_work() -> None:
    notebook = _notebook()
    all_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "Deterministic Feature Creation" in all_text
    assert "policy_decision" in all_text
    assert "model_ready" in all_text
