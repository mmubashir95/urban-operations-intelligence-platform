# Creation-Time Feature Leakage Policy

## Boundary

Target leakage is any model input that contains the eventual outcome or
information not available when prediction occurs. Prediction occurs
immediately after complaint creation. Every training and inference pipeline
must call `validate_feature_columns` before accepting its feature list.

## Roles

- Safe features: `created_date` (prefer derived calendar components), `agency`,
  `agency_name`, `complaint_type`, and `open_data_channel_type`. Scope-constant
  fields add no within-scope variation.
- Conditional features: descriptors, geography, address, facility, and
  coordinate fields. They require explicit pipeline approval after intake-time
  availability and later correction behavior are verified.
- Identifier: `unique_key` is not approved as a feature.
- Target inputs: `closed_date` and `due_date` are blocked from model features.
- Post-creation fields: final `status`, resolution description, resolution
  action timestamps, and actual-resolution duration are blocked.
- Target-derived fields: `missed_resolution_target`, `target_eligible`,
  `outcome_mature`, duplicate flags, and `primary_exclusion_reason` are blocked.
- Unknown fields are rejected pending governance.

## Due-date decision

`due_date` may be used to construct the target. It is **not approved** as a
model feature, and neither is a derived `days_until_due`, until repository or
source evidence establishes that it exists at creation, whether it changes,
whether the API preserves original or latest values, and its operational
semantics.

## Historical features

Historical features must use only information strictly before each complaint's
creation time. Full-period target rates, future complaint volume, future
category miss rates, post-period aggregates, and aggregates using validation
or test outcomes are blocked. Time-split preprocessing may not use later
outcomes.

## Enforcement

`urban_ops.features.feature_roles` owns the inventory. Safe fields pass,
conditional fields require an explicit approval set, and identifiers,
target-input, post-creation, target-derived, excluded, and unknown fields raise
`FeaturePolicyError`. `build_leakage_audit` emits the reviewable inventory.

The unresolved timing questions above must be answered before expanding the
baseline-safe list.
