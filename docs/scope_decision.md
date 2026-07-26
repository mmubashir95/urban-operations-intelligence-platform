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

- Extraction timestamp: 2026-07-26T17:22:05.446045+00:00
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

Complaint-type inclusion evidence covers the full analysis history (2020-01-01 through 2026-06-30), which is wider than the selected date range below; the final modelling population is limited further to that narrower date range.

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

- Monthly eligible-volume coefficient of variation: 0.47
- Monthly eligible volume range: 325 to 3,692
- Monthly missed-target rate range: 85.59% (largest single-month change: 35.82%)
- Monthly due-date coverage range: 19.51%
- Monthly closed-date coverage range: 15.18%

The monthly missed-target rate varies substantially across the selected window (range 85.59%, largest single-month change 35.82%). Zero missing months establishes temporal continuity only; it does not establish stable target behaviour. A time-aware validation design and an explicit drift check are required before modelling, and whether the full date range belongs in one modelling population should be revisited in light of this variation.

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
| C5 | Recent complete calendar years (48 months: 2022-01 to 2025-12) | DSNY | 2022-01-01 to 2025-12-31 | 1 | 60281 | 91.08% | 99.44% | 53.35% | Rejected alternative | Lower scope score than selected feasible candidate | 90.55% |
| C1 | Full available period (78 months: 2020-01 to 2026-06) | DSNY | 2020-01-01 to 2026-06-30 | 1 | 74519 | 91.25% | 98.62% | 56.61% | Rejected alternative | Lower scope score than selected feasible candidate | 89.73% |
| C3 | Recent 4-year period (42 months: 2023-01 to 2026-06) | DSNY | 2023-01-01 to 2026-06-30 | 1 | 54486 | 91.04% | 98.47% | 42.76% | Rejected alternative | Lower scope score than selected feasible candidate | 89.04% |
| C4 | Recent 3-year period (30 months: 2024-01 to 2026-06) | DSNY | 2024-01-01 to 2026-06-30 | 1 | 42025 | 90.82% | 98.06% | 42.43% | Rejected alternative | Lower scope score than selected feasible candidate | 88.46% |

## Known Limitations

- The initial scope contains one complaint type, so complaint_type is constant and cannot be a varying model feature.
- The business meaning and creation-time semantics of due_date require confirmation.
- Prediction-time availability and post-creation changes to due_date require confirmation.
- Status eligibility and cancelled/duplicate complaint treatment remain to be approved.
- A scoped row-level extraction must validate identifiers, parsing, chronology, target construction, and status distributions before modelling.
- Operational relevance has no objective repository-backed measure and is scored neutrally.
- Monthly missed-target rate varies by 86% across the selected window (temporal continuity does not imply temporal stability); a time-aware validation design and drift check are required before modelling.

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
