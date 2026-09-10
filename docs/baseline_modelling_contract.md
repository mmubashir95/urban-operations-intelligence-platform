# Baseline Modelling Contract

## Production Decision

Phase 9 creates the strict metadata boundary between the Phase 8 frozen
preprocessing output and later baseline modelling phases.

**Phase 9 DOES NOT TRAIN A MODEL.**

It does not calculate ROC-AUC, PR-AUC, accuracy, calibration, thresholds, test
metrics, or any other model evaluation output. It does not modify
preprocessing, create features, sort rows, repair alignment, or persist feature
matrices.

## Phase 8 Dependency

Phase 9 accepts only the already verified Phase 8 preprocessing contract with
status `FROZEN_MODEL_READY`. The Phase 8 contract remains authoritative for:

- ordered feature names;
- feature count;
- matrix type and dtype;
- target, identifier, and timestamp names;
- split order and row counts;
- preprocessing schema fingerprint;
- full Phase 4 through Phase 7 lineage fingerprint.

If any supplied Phase 9 input differs from Phase 8 metadata, the verification
gate fails closed.

## Allowed Model Inputs

For each authoritative split, `train`, `validation`, and `test`, later baseline
models may receive only:

- `X`: two-dimensional SciPy CSR matrix, `float64`, finite, with the exact
  Phase 8 feature order;
- `y`: one-dimensional `missed_resolution_target`;
- `unique_key`: one-dimensional traceability identifier, separate from `X`;
- `created_date`: one-dimensional UTC chronology series, separate from `X`.

The current Phase 8 schema is:

| Column | Feature |
| ---: | --- |
| 0 | created_hour |
| 1 | created_day_of_week |
| 2 | created_month |
| 3 | is_weekend |

Phase 9 reads this schema from the Phase 8 contract rather than redefining a
second production schema.

## Forbidden Inputs

The following must not enter `X` or model fitting through Phase 9:

- `missed_resolution_target`;
- `unique_key`;
- `created_date`;
- leakage fields such as closure timestamps and final status;
- excluded fields;
- unresolved conditional geography fields;
- any post-creation field;
- any field not approved by the existing feature policy.

Identifier and chronology fields remain available only for traceability,
alignment, temporal integrity, later evaluation joins, and audit evidence.

## Split Semantics

The only accepted split set is exactly:

1. `train`
2. `validation`
3. `test`

Missing or unexpected split names are rejected. Split row counts must match the
Phase 8 contract. Chronology must preserve the existing project rule: training
ends before validation starts, and validation ends before test starts.

## Target Semantics

All targets must be one-dimensional, non-null, binary, and restricted to
`{0, 1}`. Training must contain both classes.

Phase 9 records train-only descriptive target metadata:

- training row count;
- positive-class count;
- negative-class count;
- positive-class prevalence;
- majority class;
- observed training classes.

Validation and test labels are checked only for structural validity. They are
not used to fit or choose a baseline.

## Identifier And Chronology Semantics

`unique_key` must be one-dimensional, non-null, unique within each split, and
disjoint across splits.

`created_date` must be valid, timezone-aware UTC chronology. Phase 9 verifies
that row order and pandas indexes remain aligned across `y`, `unique_key`, and
`created_date`. It does not reset indexes, sort records, merge frames, or
repair rows.

## Fingerprint Lineage

`VerifiedBaselineModellingContract` records a deterministic fingerprint derived
from metadata only:

- Phase 8 contract fingerprint;
- Phase 9 contract and policy versions;
- ordered feature names and count;
- matrix type and dtype;
- target, identifier, and chronology names;
- split order and row counts;
- train-only class counts and prevalence;
- preprocessing schema fingerprint;
- Phase 9 status.

Full sparse matrix payloads are not hashed. The Phase 9 fingerprint is intended
for contract lineage, not dataset-content fingerprinting.

## Serialization

The contract provides stable `to_dict()` and `from_dict()` methods. Serialized
payloads are JSON-safe and contain no sparse matrices, pandas objects, mutable
unordered state, models, thresholds, or metrics.

## Scope Boundary

Phase 9 does not implement majority-class baseline, historical-rate baseline,
logistic regression, decision trees, gradient boosting, prediction methods,
probabilities, threshold selection, calibration, feature selection, scaling,
imputation, encoding, sklearn pipelines, model persistence, SHAP, top-K
evaluation, or subgroup evaluation.

## Month 1 Baseline Modelling Contract

Target: `missed_resolution_target`.

Positive class: `1 = missed expected resolution target`.

Training policy: models may fit only on training data.

Validation policy: validation may be used for model comparison, threshold
selection, and approved baseline-level decisions.

Test policy: test data must remain untouched until final model selection is
frozen.

Preprocessing policy: baseline models must use the existing frozen
preprocessing outputs. They must not refit, duplicate, or redesign
preprocessing.

Split policy: the existing chronological train/validation/test split is the
only accepted split design. Random re-splitting is prohibited.

Leakage policy: no target-derived, post-creation, future, validation-derived,
or test-derived information may enter training features or train-derived
aggregates.

Determinism: any estimator with stochastic behavior must use an explicit random
seed. Current Month 1 estimators are deterministic under the frozen inputs.

Probability policy: probability or risk scores are preserved separately from
binary predictions. Ranking metrics use scores, not hard predictions.

Threshold policy: threshold selection may use validation only. The rule-based
historical-rate baseline selects the validation-F1 maximizing threshold from a
deterministic candidate grid. Ties choose the highest threshold.

Final test policy: final test evaluation runs only after baseline choice,
configuration, threshold, metric definitions, and selection logic are frozen.
Test results must not be used to change features, preprocessing, solvers,
thresholds, fallback rules, regularization, or model selection.

## Implemented Month 1 Metrics

The common evaluator implements:

- precision;
- recall;
- F1;
- ROC-AUC;
- PR-AUC;
- Brier score;
- confusion matrix counts: true negative, false positive, false negative,
  true positive;
- Precision@5% and Recall@5%;
- Precision@10% and Recall@10%;
- Precision@20% and Recall@20%.

Top-K metrics rank rows by predicted risk descending and select
`max(1, ceil(n * fraction))` rows. Ranking ties preserve original row order via
stable sorting.

For single-class `y_true`, ROC-AUC and PR-AUC are reported as `NaN`; threshold,
confusion, Brier, precision, recall, F1, and top-K metrics remain defined.
Empty inputs, mismatched lengths, non-binary labels, non-finite scores, and
scores outside `[0, 1]` raise explicit errors.

## Implemented Baselines

- Majority Class: learns the most common training label, with deterministic
  tie-breaking to class `0`, and reports train empirical class probabilities as
  constant scores.
- Historical Rate: learns `mean(missed_resolution_target)` by approved
  creation-time group `created_month` on training rows only, with fallback to
  global training prevalence for unseen groups.
- Rule Based: thresholds the historical-rate score using a validation-only
  threshold selected for F1.
- Logistic Regression: fits sparse-compatible sklearn logistic regression
  directly on the frozen CSR matrix with no preprocessing refit.

Month 2 models such as gradient boosting, SHAP, forecasting, resolution-time
regression, and NLP classification remain outside this contract.
