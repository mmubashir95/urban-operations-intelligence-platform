# Target Definition

## Status and selected scope

This is the authoritative Month 1 target contract. The machine-readable scope
is loaded from `reports/05_temporal_stability/tables/selected_scope_summary.csv`;
the values currently resolve to DSNY / Graffiti, 2024-01-01 through 2025-12-31
inclusive. Counts are snapshot evidence, not hardcoded source truth.

The prediction is generated immediately after complaint creation using only
information available then.

## Target contract

- Name: `missed_resolution_target`
- Formula: `closed_date > due_date`
- `1`: complaint closed after its due date
- `0`: complaint closed on or before its due date
- Equality: `closed_date == due_date` is target `0`
- Dtype: nullable `Int8`
- Ineligible rows: `pd.NA`, never `0`
- Target inputs: `closed_date` and `due_date`

## Eligibility

A record is eligible only when it is within the selected agency, complaint
type, and inclusive created-date interval; has parseable `created_date`,
`due_date`, and `closed_date`; satisfies `due_date >= created_date` and
`closed_date >= created_date`; has normalized status `closed`; has a mature
outcome; and is the unambiguous canonical record for its complaint ID.

`outcome_mature = due_date.notna() & (due_date <= extraction_timestamp)`.
Maturity does not by itself make a row eligible.

Open complaints remain ineligible and unlabeled even after the due date.
Cancelled complaints are excluded because cancellation is not approved as a
successful operational resolution. Other non-closed and unknown statuses are
also excluded. Approved status: `closed`. Explicitly excluded statuses include
`assigned`, `cancelled`, `duplicate`, `open`, `pending`, and `started`.

Exact duplicate rows keep one deterministic canonical record. Redundant rows
are marked and excluded. When a complaint ID has conflicting target inputs,
status, or scope values, every member is excluded as
`conflicting_duplicate_unique_key`.

## Exclusion precedence

The canonical order is:

`outside_selected_scope, missing_created_date, missing_due_date, missing_closed_date, due_before_created, closed_before_created, excluded_status, conflicting_duplicate_unique_key, exact_duplicate_unique_key, outcome_not_mature, eligible`

Individual flags are retained even when another failure wins precedence.

## Leakage and limitations

The target inputs, final status, resolution fields, eligibility fields, and
target-derived fields are not creation-time model features. `due_date` is
approved for target construction only; its creation-time availability,
business semantics, source versioning, and mutability are not proven.

The selected population still has temporal-volatility limitations documented
in `docs/scope_decision.md`. Rebuild labels from the current selected
population and extraction timestamp rather than reusing historic counts.

## Source of truth

- Scope loading: `urban_ops.data.selected_scope`
- Eligibility and precedence: `urban_ops.features.eligibility`
- Target formula and dtype: `urban_ops.features.target`
- Reports: `urban_ops.analysis.target_and_leakage`
- Leakage roles: `urban_ops.features.feature_roles`
- Enforcement: `urban_ops.features.leakage`
