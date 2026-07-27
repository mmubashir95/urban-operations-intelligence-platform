# Target Definition

## Status and Source of Truth

This document is the source of truth for Month 1 target governance. The target
is approved for feasibility analysis and downstream modelling preparation only
within the selected DSNY Graffiti scope, with the limitations below.

## Target

- Name: `missed_resolution_target`
- Prediction moment: shortly after complaint creation
- Positive class: complaint closed after its expected due date
- Negative class: complaint closed on or before its expected due date

```python
outcome_mature = due_date.notna() & (due_date <= extraction_timestamp)
eligible_for_target = (
    created_date.notna()
    & closed_date.notna()
    & due_date.notna()
    & outcome_mature
)
missed_resolution_target = (closed_date > due_date).astype("int8")
```

The comparison is strictly `>`: closure exactly at the due date is on time.
The formula has not been changed from the established target definition;
outcome maturity is an additional observation-window eligibility requirement.

## Final Scope

- Agency: DSNY
- Complaint type: Graffiti
- Created-date range: 2024-01-01 through 2025-12-31, inclusive

The target and model scope do not generalise to all DSNY or NYC 311 records.
See `docs/scope_decision.md` for snapshot metrics and candidate comparison.

## Eligibility and Open Complaints

Required fields are `unique_key`, `created_date`, `closed_date`, `due_date`,
`agency`, `complaint_type`, and `status`. Timestamps must be parseable and are
normalised to UTC for comparison. Source nulls remain null; no timestamp is
imputed.

Open complaints without `closed_date` are excluded from historical label
construction and reported separately. They are not inferred to be on time or
late from status. Complaints whose due date is later than the extraction
timestamp are outcome-immature and deferred.

## Historical DOT Evidence

DOT was evaluated first and rejected, not silently removed. Its due-date
coverage was 0% and it had zero target-eligible records, so this target could
not be constructed for the DOT extract. The DOT investigation also observed a
separate closure-before-creation chronology concern. DOT is not the selected
scope and its infeasibility does not veto the independently feasible DSNY
Graffiti subgroup.

The detailed historical snapshot remains in
`reports/03_target_feasibility/` and `notebooks/03_target_feasibility.ipynb`.

## Invalid Substitutes

Final status, a missing closure date, arbitrary resolution-duration cutoffs,
`resolution_action_updated_date`, and `resolution_description` are not valid
substitutes for the expected due-date target.

## Leakage Boundary

`closed_date`, final status, `resolution_description`,
`resolution_action_updated_date`, actual resolution duration, and other
post-creation updates may be used only for feasibility analysis or label
construction. They must not be model features at the creation-time prediction
moment.

`due_date` is necessary for label construction, but its availability and
mutability at prediction time have not been proven. It must not be treated as a
safe feature until that operational contract is confirmed.

## Remaining Governance Limitations

- Confirm the business semantics and lifecycle of DSNY Graffiti `due_date`.
- Confirm creation-time availability and whether due dates can later change.
- Approve explicit cancelled and duplicate complaint eligibility rules.
- Preserve the extraction-time outcome-maturity cutoff in every label build.
- Reassess this contract if coverage, category definitions, or target prevalence
  drifts materially.
