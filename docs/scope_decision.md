# Modelling Scope Decision

## Decision Status

**Approved with limitations**

## Executive Summary

The API-backed comparison produced status **Approved with limitations** for the narrow selected
population. Data-scope feasibility is established separately from final target
governance.

## Business Problem

Predict whether a newly created NYC 311 complaint will miss its expected resolution target.

## Prediction Target

`missed_resolution_target`: 1 when `closed_date > due_date`, otherwise 0, only for target-eligible historical records.

## Target Definition Reference

The eligibility and label rules match `notebooks/03_target_feasibility.ipynb` and `docs/target_definition_draft.md`.

## Data Source

Official NYC Open Data dataset `erm2-nwe9` at `https://data.cityofnewyork.us/resource/erm2-nwe9.json`.

## Extraction Summary

- Extraction timestamp: 2026-07-26T04:37:44.092054+00:00
- Requested period: 2020-01-01 through live snapshot
- Aggregated records represented: 21,663,984
- Latest available date: 2026-07-25T01:51:18+00:00
- Latest complete month: 2026-06

## Selection Criteria

Critical rules cover eligible volume, due-date and closure coverage, class balance, temporal continuity, and measurable timestamp consistency. Score weights are 30%, 20%, 15%, 15%, 10%, and 10% respectively; scoring does not override failed critical gates.

## Selected Agency

DSNY — Department of Sanitation

## Selected Scope

- Agency: DSNY
- Agency name: Department of Sanitation
- Complaint types: Graffiti
- Date range: 2022-01-01 through 2026-06-30

## Selected Date Range

2022-01-01 through 2026-06-30.

## Selected Complaint Types

Graffiti

## Why This Scope Was Selected

DSNY has only 3.09% agency-wide due-date coverage
because most of its complaint types do not use this field. The selected
complaint-type population is sufficiently large, temporally continuous, and
target-feasible, with strong due-date and closure coverage. The initial model
is therefore complaint-type-specific and is not intended to generalise to all
DSNY service requests.

## Eligible Population

Records with non-null parseable `created_date`, `closed_date`, and `due_date`, matching Notebook 03. Ineligible records retain a null target.

## Target Distribution

- Eligible records: 66,340
- Missed: 34,271
- On time: 32,069
- Missed-target rate: 51.66%

## Data Quality Summary

- Due-date coverage: 91.20%
- Closed-date coverage: 98.62%
- Eligibility rate: 89.83%

## Temporal Stability

The current incomplete month was excluded from recommendation and monthly comparisons. The selected candidate covers 54 active months with 0 missing months.

## Sensitivity Analysis

The focused threshold analysis rebuilds the included complaint types and final
candidate for each threshold or date-range scenario. Agency-wide due-date
coverage remains diagnostic and does not veto a qualifying scoped population.

## Included Records

73,851 records are represented by the selected scope; 66,340 can receive the target.

## Exclusion Rules

Rows without any required target timestamp are target-ineligible. The incomplete current month is excluded from stability and final date-range decisions. No status-only proxy target is used.

## Rejected Alternatives

| candidate_id | candidate_name | agency | date_range | complaint_type_count | eligible_target_count | due_date_coverage | closed_date_coverage | missed_target_rate | decision | rejection_reasons | scope_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C5 | Recent complete calendar years | DSNY | 2022-01-01 to 2025-12-31 | 1 | 60281 | 0.9108137736330736 | 0.9943686083705006 | 0.533451668021433 | Rejected alternative | Lower scope score than selected feasible candidate | 0.9054731416736128 |
| C1 | Full available period | DSNY | 2020-01-01 to 2026-06-30 | 1 | 74519 | 0.9124660756287317 | 0.9862493215125746 | 0.5660972369462821 | Rejected alternative | Lower scope score than selected feasible candidate | 0.8973159092360414 |
| C3 | Recent 4-year period | DSNY | 2023-01-01 to 2026-06-30 | 1 | 54486 | 0.9103692995137337 | 0.984656328032593 | 0.4275593730499578 | Rejected alternative | Lower scope score than selected feasible candidate | 0.890390481183796 |
| C4 | Recent 3-year period | DSNY | 2024-01-01 to 2026-06-30 | 1 | 42025 | 0.908242908814011 | 0.980603676206189 | 0.4242712671029149 | Rejected alternative | Lower scope score than selected feasible candidate | 0.8845715254119572 |

## Known Limitations

- The initial scope contains one complaint type, so complaint_type is constant and cannot be a varying model feature.
- The business meaning and creation-time semantics of due_date require confirmation.
- Prediction-time availability and post-creation changes to due_date require confirmation.
- Status eligibility and cancelled/duplicate complaint treatment remain to be approved.
- A scoped row-level extraction must validate identifiers, parsing, chronology, target construction, and status distributions before modelling.
- Operational relevance has no objective repository-backed measure and is scored neutrally.

## Downstream Implications

Before modelling, retrieve the selected agency, complaint types, and date range
at row level. Validate identifier uniqueness, conflicting duplicates, timestamp
parsing, creation-to-due and creation-to-closure chronology, target
construction, status distribution, cancelled and duplicate treatment,
prediction-time availability of `due_date`, and whether `due_date` changes
after creation. Because this scope contains one complaint type,
`complaint_type` is constant and is not useful as a varying initial-model
feature.

## Reproducibility Information

Generated by `notebooks/04_scope_selection.ipynb` from deterministic server-side API aggregations ordered by stable grouping keys.

## Approval Status

Approved with limitations. No model is trained and no train/validation/test split is created in this analysis.
