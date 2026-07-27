# Modelling Scope Decision

## Decision Status

**APPROVED_WITH_LIMITATIONS**

Month 1 — Step 3 is finalised for a narrow modelling population. The model
scope applies to **DSNY Graffiti complaints only**. It must not be interpreted
as applying to all DSNY complaints or all NYC 311 complaints.

## Selected Scope

- Agency: DSNY — Department of Sanitation
- Complaint type: Graffiti
- Start date: 2024-01-01
- End date: 2025-12-31
- Target: `missed_resolution_target`
- Target rule: `closed_date > due_date`
- Extraction timestamp: 2026-07-27T16:41:16.401819+00:00
- Source: NYC Open Data dataset `erm2-nwe9`

The live source can change. Counts below describe this extraction snapshot and
must be recomputed by the notebooks for a later snapshot.

## Decision Authority

Notebook 04's general scorecard establishes candidate feasibility; its
`scope_score` does not select the final date range. Notebook 05 is the
authoritative volatility-aware temporal decision because it additionally
evaluates outcome maturity, complete-calendar-year boundaries, and recent
relevance. Notebook 04 reconciles its candidate table to the single selected
Notebook 05 artifact and fails if identifiers, dates, counts, or rates disagree.

## Selected-Scope Statistics

- Total records: 40,017
- Outcome-mature target-eligible records: 35,966 (89.88%)
- Missed complaints: 15,716
- On-time complaints: 20,250
- Missed-target rate: 43.70%
- Due-date coverage: 90.55%
- Closed-date coverage: 99.32%
- Active months: 24
- Expected months: 24
- Missing months: 0
- Outcome-maturity rate among records with a due date: 100.00%
- Mature open complaints excluded from label construction: 270
- Invalid timestamp-sequence rate: 0.0150%

Small differences from earlier reports are expected when the live extraction
timestamp changes. The selected complete-calendar period itself had no
outcome-immature due dates in this snapshot.

## Outcome-Maturity Rule

`outcome_mature = due_date.notna() & (due_date <= extraction_timestamp)`.
Target evidence additionally requires non-null, parseable `created_date`,
`closed_date`, and `due_date`. Therefore an open complaint is excluded from
scope-selection labels and deferred; it is not silently classified as on time
or late. Records with due dates after extraction are also excluded from
eligible evidence.

## Temporal Continuity

Every expected month in the selected period contains eligible records. This
establishes continuity only. Zero missing months is not evidence of stable
target behaviour.

## Temporal Stability

For the selected period:

- Monthly eligible-volume range: 1,219 to 2,367
- Mean monthly eligible volume: 1,498.58
- Monthly volume standard deviation: 252.25
- Monthly volume coefficient of variation: 0.1683
- Monthly missed-target-rate range: 30.23 percentage points
- Monthly target-rate standard deviation: 7.62 percentage points
- Largest absolute month-to-month rate change: 21.56 percentage points
- Yearly missed-target rate: 44.12% in 2024 and 43.19% in 2025

The range and largest monthly change cross the configured warning thresholds,
but not the severe thresholds. The data indicates a possible operational or
data-generation regime change across the wider history. The available dataset
does not establish the underlying cause.

## Candidate-Period Comparison

| Candidate | Eligible | Target rate | Closed coverage | Target-rate range | Rate std. dev. | Volume CV | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-01-01–2026-06-30 | 66,290 | 51.70% | 98.62% | 84.78% | 24.13% | 0.4674 | REQUIRES_NARROWER_RANGE |
| 2022-01-01–2025-12-31 | 60,281 | 53.35% | 99.44% | 72.36% | 24.27% | 0.4771 | REQUIRES_NARROWER_RANGE |
| 2023-01-01–2026-06-30 | 54,436 | 42.80% | 98.47% | 59.04% | 11.01% | 0.3011 | REQUIRES_NARROWER_RANGE |
| 2023-01-01–2025-12-31 | 48,427 | 43.74% | 99.46% | 46.62% | 10.66% | 0.2929 | APPROVED_WITH_LIMITATIONS |
| 2024-01-01–2026-06-30 | 41,975 | 42.48% | 98.07% | 45.93% | 8.67% | 0.2226 | APPROVED_WITH_LIMITATIONS |
| **2024-01-01–2025-12-31** | **35,966** | **43.70%** | **99.32%** | **30.23%** | **7.62%** | **0.1683** | **SELECTED — APPROVED_WITH_LIMITATIONS** |

The exact machine-readable comparison, thresholds, and maturity counts are in
`reports/05_temporal_stability/tables/temporal_stability_summary.csv`.

## Final Selection Rationale

The selected range retains two recent complete calendar years, more than
35,000 eligible records, both target classes, full monthly continuity, strong
coverage, and the best stability of the required candidates. It is preferred
to the longer ranges because 2022 contributes a markedly different target
regime and 2026 is an incomplete calendar year with weaker closure completeness
and 61 due dates that were still after the extraction timestamp.

The 2023–2025 range was rejected because its target-rate range and standard
deviation are materially higher. The 2024–2026 range was rejected because it
adds an incomplete recent year and weaker outcome observation without improving
stability. No range shorter than two complete calendar years was added: it
would reduce seasonal representation below the configured rule and is not
needed to achieve a non-severe candidate.

## Agency and Complaint-Type Feasibility

Agency screening is diagnostic and does not veto a viable subgroup merely
because other complaint types lack due dates. DSNY Graffiti independently
passes the complaint-type volume, coverage, target-balance, continuity, recent
activity, and timestamp-consistency gates. This corrects the misleading
agency-wide veto interpretation while preserving agency-wide evidence.

## Rejected Alternatives

- **DOT:** rejected. Due-date coverage is 0% and target-eligible records are 0.
  DOT remains useful historical target-feasibility evidence, but it is not the
  selected scope.
- **All DSNY complaints:** rejected. Most DSNY complaint types do not provide
  the deadline field required by this target.
- **All NYC 311 complaints:** rejected. Target-field meaning and coverage are
  not sufficiently uniform across agencies and complaint types.
- **Longer DSNY Graffiti ranges:** rejected or required to narrow for the
  temporal reasons reported above.

## Decision Thresholds

Thresholds are centralised in
`urban_ops.analysis.temporal_stability.TemporalStabilityThresholds`. Core
rules require at least 10,000 eligible records, 70% due-date coverage, 80%
closed-date coverage, both classes (5%–95% target rate), 24 active continuous
months, at least 250 eligible records per month, two complete calendar years,
98% outcome maturity, and no more than 5% invalid timestamp sequences. Severe
volatility is flagged at a 50-point rate range, 15-point monthly rate standard
deviation, 25-point largest monthly change, or 0.60 volume CV. Warning
thresholds are lower and prevent unconditional approval.

## Known Limitations

- Monthly target behaviour is improved but not stationary; the selected range
  still triggers a volatility warning.
- The business meaning, creation-time availability, and mutability of
  `due_date` require confirmation. It must not be assumed to be a safe feature.
- Cancelled and duplicate complaint eligibility remains a governance decision.
- `complaint_type` is constant in this narrow population and is not useful as a
  varying model feature.
- The analysed `descriptor` field remained constant (`Graffiti`, 100% of rows)
  across the evaluated years and therefore does not explain the temporal
  change. The possible regime signal is supported by changes in missed-target
  rate, coverage, and complaint volume. The available data does not establish
  the underlying operational cause.

## Downstream Requirements

- Use chronological evaluation; do not randomly split this time-dependent
  population.
- Report performance by month and year, run explicit drift checks, and reassess
  the scope if target prevalence or field coverage changes.
- Use `closed_date`, final status, resolution descriptions, actual duration,
  resolution-action timestamps, and other post-creation updates only for
  feasibility or label construction, never as creation-time features.
- Confirm `due_date` availability and immutability at prediction time before
  any feature use.
- Preserve the outcome-maturity cutoff when rebuilding labels.

No model is trained and no train/validation/test split is created in this
scope-selection work.
