# Baseline Feature Policy — Notebook 11

## Purpose and boundary

The policy freeze converts Notebook 10 evidence into one explicit eligibility decision
before any feature creation or preprocessing is written. The machine-readable
source of truth is `configs/features/resolution_risk_baseline.yaml`; this
document explains that policy but does not redefine it.

An `APPROVED_CANDIDATE` is allowed to enter Phase 2 feature-engineering design.
It is not automatically a final model feature. Final inclusion requires later
validation evidence.

```text
Notebook 10: What does the data tell us?
        ↓
Notebook 11 policy freeze: Which features are allowed to proceed?
        ↓
Notebook 11 Phase 2: How are approved features created?
```

The policy freeze fits no imputer, encoder, scaler, selector, or model. It creates no
`ColumnTransformer`, model matrix, transformed split, or model artifact. It
does not apply Notebook 10's missing-value, rare-category, scaling, or outlier
recommendations.

## Prediction moment and authority

Prediction occurs immediately after complaint creation. Feature eligibility is
resolved in this order:

1. Step 4 leakage and prediction-time availability authority in
   `docs/leakage_policy.md` and `urban_ops.features.feature_roles`.
2. Notebook 10's leakage audit.
3. Notebook 10's newer `eda_status` decisions.
4. Notebook 10's train-only missingness, cardinality, variation, and outlier evidence.
5. Notebook 10's older baseline recommendation.

This ordering is deliberate. A target-rate difference cannot approve a
feature. Availability, leakage safety, usable structure, an approved
representation, and the absence of an unresolved blocker must all be
satisfied first.

## Deterministic statuses

| Status | Meaning in the frozen policy |
| --- | --- |
| `APPROVED_CANDIDATE` | Safe enough to enter Phase 2 design; not guaranteed final model inclusion |
| `SOURCE_ONLY` | Needed to derive another feature, but not a direct first-pass model input |
| `ALTERNATIVE_REPRESENTATION` | Same essential information as a preferred candidate |
| `REVIEW_REDUNDANCY` | Strong overlap with a preferred candidate |
| `CONDITIONAL` | Potentially useful but blocked by unresolved governance or generalization risk |
| `REVIEW` | Evidence is insufficient for first-pass promotion |
| `EXCLUDE_LEAKAGE` | Leakage, target, target-derived, post-creation, or ungoverned audit field |
| `EXCLUDE_IDENTIFIER` | Traceability identifier, not a model feature |
| `EXCLUDE_ALL_NULL` | Entirely null in the training evidence |
| `EXCLUDE_ZERO_VARIANCE` | No useful variation in the training evidence |

Only `APPROVED_CANDIDATE` implies `phase_2_allowed: true`.

## Frozen first-pass decisions

The approved candidates are `created_hour`, `created_day_of_week`,
`created_month`, and `is_weekend`. All four will eventually be derived from
`created_date`; no derivation occurs during the policy freeze.

`created_date` is `SOURCE_ONLY`: it is required for deterministic temporal
derivation, while its raw timestamp is not approved as a direct first-pass
model feature.

`created_day_name` and `created_month_name` are alternative representations of
`created_day_of_week` and `created_month`. `created_quarter` and
`created_week_of_year` remain under redundancy review because they overlap with
`created_month`. `created_day_of_month` remains under general review.

`created_year` is conditional because it may encode time progression or an
operational regime rather than a stable relationship that generalizes to
future years.

## Conditional geography

`borough`, `location_type`, `incident_zip`, `latitude`, and `longitude` remain
conditional. Step 4 and Notebook 10 do not prove that their recorded values are
available and immutable at complaint creation. Descriptive target patterns,
missing-value recommendations, valid geographic outliers, and apparent
statistical usefulness do not resolve that governance question.

No missing category, median imputation, rare ZIP threshold, `__RARE__`, or
`__UNKNOWN__` behavior is selected during the policy freeze. Notebook 10's valid statistical
geographic outliers remain retain-and-monitor evidence; they are not clipped
or deleted.

## Exclusions

The frozen policy assigns reason-specific exclusions and reconciles them to the
Notebook 10 tables when loaded:

- `unique_key` is `EXCLUDE_IDENTIFIER` and remains usable only for traceability.
- `descriptor_2` is `EXCLUDE_ALL_NULL`.
- `agency`, `agency_name`, `complaint_type`, `descriptor`, and
  `open_data_channel_type` are `EXCLUDE_ZERO_VARIANCE` based on train-only
  evidence.
- Closure, due-date, final-status, resolution, target, eligibility,
  target-derived, post-creation, and ungoverned audit fields are
  `EXCLUDE_LEAKAGE`. `due_date` remains blocked because its creation-time
  availability and mutability have not been proven.

`urban_ops.features.policy.load_feature_policy` validates allowed statuses,
required fields, unique names, source and counterpart references, the exact
Phase 2 allow-list rule, Step 4 blocked roles, and reconciliation with Notebook
10's leakage, all-null, zero-variance, and baseline inventories.

## Why Notebook 10 is not model-ready

Notebook 10 correctly reports `model_ready: false`: it supplies governed EDA
evidence and recommendations, but it neither freezes eligibility nor builds a
training-only preprocessing contract. This work resolves eligibility only.
Deterministic feature creation begins in Phase 2 after review of this policy.
