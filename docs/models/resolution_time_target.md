# Resolution-Time Target (Phase 7.1)

## Contract

The business objective is to predict a complaint's eventual resolution duration
at the moment it is created. The target is `resolution_duration_hours`, defined
without rounding as:

```text
(closed_date - created_date).total_seconds() / 3600.0
```

Timestamps use the existing UTC normalization policy. Fractional hours, valid
zero-hour durations, and extremely long positive durations are retained. Phase
7.1 does not cap or analyze the duration distribution.

## Eligibility and deterministic precedence

Only normalized status `closed` is accepted as completed, matching the existing
classification governance. `cancelled` is excluded; `open` is unresolved; null
or any other status is ambiguous and excluded. Status text is trimmed and
case-folded, but no unsupported status is reinterpreted.

Every row is retained in the aligned output and receives at most one reason.
The first matching reason in this order wins:

1. `missing_created_date`
2. `invalid_created_date`
3. `missing_closed_date`
4. `invalid_closed_date`
5. `invalid_timestamp_order`
6. `cancelled_complaint`
7. `open_complaint`
8. `ambiguous_completion_status`
9. `resolution_not_observed_by_cutoff`
10. `invalid_calculated_duration`
11. `eligible`

Ineligible targets are nullable, never zero-filled or imputed. The companion
category is `invalid`, except missing/unresolved closure and closure after the
cutoff, which are `not_observed`. This distinguishes permanently invalid labels
from outcomes unavailable at a historical observation time without adding a
separate audit framework.

## Historical maturity and leakage

`build_resolution_time_target(frame, observation_cutoff=...)` accepts a UTC
observation cutoff. A closure before or exactly at the cutoff is observable; a
closure after it receives `resolution_not_observed_by_cutoff` and no label.
Phase 7.2 must supply the cutoff appropriate to each historical development
window. In particular, labels for complaints assigned by creation date to the
frozen train interval must not use closures after that interval's observation
cutoff. Phase 7.1 does not change the globally frozen creation-date boundaries.

The target, closure date, final status, resolution text/action timestamps, and
all resolution-time eligibility fields are outcome-derived and blocked from
creation-time features by the central feature-role policy. The original NYC
Open Data extract is a later-state snapshot, so historical corrections before
extraction cannot be reconstructed; supplied cutoffs prevent known-after-cutoff
closures but cannot recover prior versions of mutable source fields.

## Phase 7.2 handoff

The reusable builder returns a DataFrame aligned to the input index with:

- `resolution_time_eligible`
- `resolution_duration_hours`
- `resolution_time_exclusion_reason`
- `resolution_time_exclusion_category`

Phase 7.2 can join these columns by index (while preserving `unique_key` from
the source), filter on `resolution_time_eligible`, and analyze the unrounded
target. No standalone CLI or full-dataset report is introduced in Phase 7.1.
