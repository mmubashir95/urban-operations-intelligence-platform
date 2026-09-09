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

Next work: **Phase 10 — Baseline Model Training**.
