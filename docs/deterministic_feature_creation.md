# Deterministic Feature Creation

## Purpose

Notebook 11 consumes the frozen baseline policy and creates only the approved
calendar features. Eligibility remains owned by
`configs/features/resolution_risk_baseline.yaml`; feature code does not maintain
a separate allow-list.

The operation is stateless and deterministic. It learns no training statistics,
uses no target values, randomness, locale, current time, or external service,
and requires no fitting. Returned DataFrames are deep copies with their original
index, row order, identifiers, targets, and source values preserved.

## Timestamp contract

`created_date` must already be a non-null timezone-aware UTC datetime series.
This is the representation published by cleaning and enforced by splitting.
The builder does not parse strings, coerce errors, infer timezones, or convert
timestamps to the machine's local timezone. Contract violations raise
`TemporalFeatureError`.

## Approved derivations

| Feature | Rule | Dtype | Domain |
| --- | --- | --- | --- |
| `created_hour` | UTC hour component | `Int8` | 0–23 |
| `created_day_of_week` | Monday=0 through Sunday=6 | `Int8` | 0–6 |
| `created_month` | Calendar month number | `Int8` | 1–12 |
| `is_weekend` | Weekday value is 5 or 6 | `bool` | `False`, `True` |

`created_date` remains source-only. Alternative, redundant, review, and
conditional calendar fields are not created. Conditional geography remains
unchanged source data and is not promoted or transformed.

If a requested derived column already exists, it must exactly match the
canonical values and dtype. A conflicting pre-existing column raises an error;
correct existing values are safely reproducible.

## Implementation boundary

`urban_ops.features.temporal` provides the reusable builder, domain evidence,
and row/target reconciliation helpers. Notebook 11 applies the same builder to
train, validation, and test in memory and verifies repeatability and source
immutability.

This work does not implement missing-value treatment, category grouping,
unknown-category behavior, outlier transformation, encoding, scaling, combined
preprocessing, persisted model matrices, or modelling.
