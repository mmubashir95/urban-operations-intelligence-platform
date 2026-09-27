# Phase 6.3 — Selected-Model Metadata Alignment

## 1. Objective

Align the persisted selected-model joblib and JSON metadata with the validated,
frozen Month 1 Logistic Regression threshold decision. This phase changes only
selected-model persistence; it does not migrate binary classification, reports,
README text, or the Month 1 completion gate.

## 2. Repository Sources Inspected

- `docs/ai-workflow/CODEX_IMPLEMENTATION.md`
- `configs/models/month1_logistic_regression_threshold_decision.json`
- `src/urban_ops/models/evaluation.py`
- `src/urban_ops/models/baseline_workflow.py`
- `src/urban_ops/models/month1_verification.py`
- `tests/unit/models/`
- `tests/integration/test_baseline_workflow.py`
- `models/baselines/selected_month_1_baseline.joblib`
- `models/baselines/selected_month_1_baseline_metadata.json`
- generated baseline and Month 1 reports

## 3. Previous Persisted-Threshold Inconsistency

`run_baseline_workflow` obtained `selected_threshold` from
`_split_predictions`. For Logistic Regression, that helper returns the generic
`model.predict(...)` result and the legacy default cutoff `0.5`. The workflow
then passed that scalar to `_persist_selected_model`, which copied it into both
selected-model artifacts.

The same workflow separately loaded and validated the authoritative frozen
decision before final test evaluation, but did not previously use that decision
for selected-model persistence.

## 4. Authoritative Threshold Source

The single authority remains:

```text
configs/models/month1_logistic_regression_threshold_decision.json
selected_threshold = 0.49
selected_on_split = validation
frozen = true
```

The artifact is loaded by `load_frozen_threshold_decision` and accepted only
after `validate_frozen_threshold_decision` verifies its complete provenance.
It was not changed in Phase 6.3.

## 5. Persistence Flow Before

```text
_split_predictions (Logistic Regression default = 0.50)
    -> selected_threshold scalar
    -> _persist_selected_model
    -> joblib selected_threshold = 0.50
    -> metadata JSON selected_threshold = 0.50
```

## 6. Persistence Flow After

```text
frozen threshold JSON
    -> load_frozen_threshold_decision
    -> validate_frozen_threshold_decision
    -> validated FrozenThresholdDecision
    -> _persist_selected_model
    -> decision.selected_threshold
    -> joblib selected_threshold = 0.49
    -> metadata JSON selected_threshold = 0.49
```

`_persist_selected_model` now accepts the already-loaded and validated decision
object from `run_baseline_workflow`. It does not reload the JSON, create another
parser, or define an independent `0.49` threshold.

The frozen decision governs Logistic Regression only, so persistence raises
`EvaluationError` if the selected model is anything else rather than pairing
another model with the Logistic Regression threshold.

## 7. Selected Joblib Result

After regeneration through `make baseline-resolution-risk`, the payload in
`models/baselines/selected_month_1_baseline.joblib` contains:

```text
selected_model_name = Logistic Regression
selected_threshold = 0.49
```

The Logistic Regression coefficient/intercept fingerprint before and after
regeneration was unchanged:

```text
8135d6eb396b54f7c9f5fd31825553a2cee0adf07dd83797b2bdb714e57fbbdc
```

## 8. Selected Metadata JSON Result

After the same workflow regeneration,
`models/baselines/selected_month_1_baseline_metadata.json` contains:

```text
selected_threshold = 0.49
```

Its model name, feature names, Phase 9 contract fingerprint, and selection rule
are unchanged.

## 9. Artifact Consistency Check

| Artifact | Threshold | Result |
| --- | ---: | --- |
| Frozen decision JSON | 0.49 | Authoritative |
| Selected joblib payload | 0.49 | Consistent |
| Selected metadata JSON | 0.49 | Consistent |

## 10. Tests

The new regression test persists both outputs to an isolated directory and
compares their threshold fields directly with the validated repository
decision.

| Check | Result |
| --- | --- |
| Focused persistence regressions | 3 passed (frozen authority, non-LR guard, committed artifacts) |
| `pytest tests/unit/models/ -v` | 250 passed |
| `pytest tests/integration/test_baseline_workflow.py -v` | 30 passed |
| `pytest tests/integration/ -v` | 64 passed |
| Full `pytest` | 878 passed |
| `git diff --check` | Passed |
| Month 1 verification module | Failed as expected: legacy gate expected 0.50, observed 0.49 |

The verifier failure is a known authority violation in
`month1_verification.py`, not a persistence failure. Changing that gate is
outside Phase 6.3.

## 11. Files Changed

- `src/urban_ops/models/baseline_workflow.py`
- `tests/integration/test_baseline_workflow.py`
- `models/baselines/selected_month_1_baseline.joblib`
- `models/baselines/selected_month_1_baseline_metadata.json`
- `reports/phase_6_3_selected_model_metadata.md`

## 12. Remaining Cleanup

The following governed consumers still expose or apply the legacy threshold
and belong to later cleanup phases:

- Logistic Regression binary predictions returned by `_split_predictions`,
  including threshold-dependent final test metrics and test subgroup results;
- validation subgroup predictions in `_fit_and_evaluate_validation`
  (`logistic.predict(X_validation)`), a separate code site whose output is the
  validation half of `baseline_subgroup_results.csv`;
- `BaselineWorkflowResult.selected_threshold` and the selected-threshold values
  passed into generated report rendering;
- governed selected/final sections of `reports/baseline_results.md` and
  `reports/month_1_baseline_report.md`;
- the Month 1 threshold statement in `README.md`; and
- the report-template threshold strings in `baseline_workflow.py` that render
  those reports;
- `month1_verification.py:EXPECTED_THRESHOLD` and its report assertion;
- the legacy gate fixtures in `tests/unit/models/test_month1_verification.py`
  that expect `threshold 0.5000`; and
- any consumer of the persisted joblib, which must apply
  `payload["selected_threshold"]` rather than the model's generic
  `predict()` default of `0.50`.

Descriptive Phase 4.1/4.2/4.3 comparisons at `0.50`, the minimum-precision
constraint of `0.50`, and generic helper defaults remain intentionally
unchanged.

## 13. Known Transitional State

Until the next phase lands, `main` is intentionally inconsistent:

- the selected-model joblib and metadata store `0.49`;
- the workflow log, `BaselineWorkflowResult.selected_threshold`, final test
  and subgroup metrics, and generated report text still use `0.50`; and
- `make verify-month1` fails with
  `Selected threshold mismatch: expected 0.5, observed 0.49.`

The generated reports' `MONTH 1 COMPLETE` statement is therefore stale and
must not be relied on until the gate passes at `0.49`. Persistence must not be
reverted to `0.50` to make the legacy gate pass.

## 14. Phase 6.3 Verdict

**PHASE 6.3 COMPLETE**

Both selected-model persistence artifacts now obtain `0.49` from the validated
frozen decision. No threshold selection, test-driven choice, feature,
preprocessing, model coefficient, or Top-K behavior changed.
