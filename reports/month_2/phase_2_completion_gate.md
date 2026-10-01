# Phase 2.16 — Gradient Boosting Completion Gate

## Verdict

**PHASE 2 COMPLETE — FROZEN AND READY FOR PHASE 3**

Phase 2 changes only the model implementation over the frozen Month 1
foundation. It uses one deterministic XGBoost configuration, fits on train,
evaluates on validation, compares with frozen Logistic Regression evidence,
and leaves test untouched. This gate performs verification only; it does not
change or select a model.

## Completion Gate

| Requirement | Status | Evidence |
|---|---|---|
| Frozen Month 1 inputs reused | PASS | `load_and_verify_gradient_boosting_inputs()` delegates to `load_frozen_baseline_inputs()` and re-verifies the frozen contracts. |
| No feature-policy change | PASS | Phase 8 and Phase 9 ordered feature names must exactly match; training preserves the same names and matrix width. |
| No split change | PASS | No split creation occurs in the Gradient Boosting path; train fits, validation evaluates, and test remains protected. |
| No leakage-policy change | PASS | The input gate requires every frozen feature to remain available, leakage-safe, approved, and Phase 2 allowed. |
| One Gradient Boosting implementation selected | PASS | Configuration version 1 selects `xgboost` / `XGBClassifier` only. |
| Model wrapper implemented | PASS | `GradientBoostedRiskModel` validates inputs and fitted state and exposes `fit`, `predict_score`, `predict_proba`, and `predict`. |
| Model fitted using train only | PASS | Training and integration identity checks use only `matrices["train"]` and `targets["train"]`. |
| Validation probabilities generated | PASS | One finite, bounded positive-class score is produced per validation row. |
| PR-AUC calculated | PASS | Shared `evaluate_ranking()` result and validation CSV contain PR-AUC. |
| ROC-AUC calculated | PASS | Shared `evaluate_ranking()` result and validation CSV contain ROC-AUC. |
| Brier calculated | PASS | Shared `evaluate_calibration()` evaluates raw validation probabilities. |
| Recall@5% calculated | PASS | Shared capacity evaluation and capacity CSV contain the 5% result. |
| Recall@10% calculated | PASS | Shared capacity evaluation and capacity CSV contain the 10% result. |
| Recall@20% calculated | PASS | Shared capacity evaluation and capacity CSV contain the 20% result. |
| Logistic Regression comparison produced | PASS | `phase_2_model_comparison.csv` loads frozen Month 1 validation and capacity artifacts; Logistic Regression is not retrained. |
| Test set remains untouched | PASS | Fail-on-access integration guards prove no test matrix scoring or test-target evaluation; no test report exists. |
| Unit tests pass | PASS | Required wrapper suite: 41 passed. |
| Integration tests pass | PASS | Required workflow suite: 2 passed. |
| Phase 2 report produced | PASS | Validation, calibration, capacity, comparison, and Markdown artifacts exist and pass schema/content checks. |
| No hyperparameter tuning performed | PASS | One fixed YAML configuration; no parameter grid, sweep, search library, candidate loop, or `best_params_` path exists. |

## Frozen Configuration

```text
implementation = xgboost
estimator       = XGBClassifier
objective       = binary:logistic
eval_metric     = logloss
n_estimators    = 100
learning_rate   = 0.1
max_depth       = 3
random_state    = 20260806
n_jobs          = 1
```

No hyperparameter tuning, probability calibration, threshold selection, new
feature engineering, preprocessing change, split change, or test evaluation
occurred in Phase 2.

## Verified Validation Evidence

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | 0.45137514203766316 |
| ROC-AUC | 0.5441074890718789 |
| Brier Score | 0.2441231896662757 |
| Precision@5% | 0.39823008849557523 |
| Recall@5% | 0.04684247050659265 |
| Precision@10% | 0.4505169867060561 |
| Recall@10% | 0.10582928521859819 |
| Precision@20% | 0.4523281596452328 |
| Recall@20% | 0.21235253296322 |

The values above are copied from the full-precision validation and capacity
CSVs. The artifact audit reconciled the 5%, 10%, and 20% selected counts with
the 6,762-row validation population using the shared ceiling rule.

## Frozen Logistic Regression Comparison

The comparison reads:

- `reports/tables/baseline_validation_results.csv`
- `reports/tables/logistic_regression_validation_capacity_comparison.csv`

It compares PR-AUC, ROC-AUC, Brier Score, Precision@5/10/20%, and
Recall@5/10/20%, with every difference defined as Gradient Boosting minus
Logistic Regression. Logistic Regression is neither retrained nor rescored.

## Test-Set Protection

```text
TRAIN       -> fit Gradient Boosting
VALIDATION  -> ranking, raw calibration assessment, Top-K, frozen comparison
TEST        -> not scored, not evaluated, not reported
```

The frozen input verifier retains and structurally validates all three Month 1
splits. After that verification boundary, the Phase 2 workflow retrieves only
train for fitting and validation for scoring/evaluation. Its integration test
raises on any test-split retrieval and asserts that no Phase 2 test artifact is
created.

## Verification Results

| Command | Result |
|---|---|
| `pytest -q tests/unit/models/test_gradient_boosting.py` | 41 passed |
| `pytest -q tests/integration/test_gradient_boosting_workflow.py` | 2 passed |
| Required unit and integration files together | 43 passed |
| All Gradient Boosting unit and integration tests | 115 passed |
| Full `pytest -q` | 1011 passed |
| Phase 2 artifact schema/content audit | Passed |
| `compileall` and `git diff --check` | Passed |

No failed, skipped, or xfailed tests and no pytest warnings were reported.
The artifact audit emitted only non-failing platform CPU-cache discovery
messages from the Arrow runtime.

## Freeze Statement

Phase 2 is frozen. Its initial deterministic Gradient Boosting configuration,
validation evidence, frozen Logistic Regression comparison, protected test
boundary, and reports form the starting point for:

```text
Phase 3 — Advanced Model Selection & Tuning
```

Phase 3 work must continue to use train for fitting and validation for model
selection while leaving test untouched until a boosted configuration is
formally frozen.
