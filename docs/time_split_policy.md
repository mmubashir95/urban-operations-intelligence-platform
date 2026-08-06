# Time-Based Split Policy

## Purpose and source

This is the authoritative Month 1 Step 8 partition contract for the DSNY /
Graffiti resolution-risk population. The source is
`eligible_service_requests.parquet` from the latest successful Step 7 run,
verified against its cleaning metadata, output hash, row count, target
contract, and Step 3 scope authority.

Step 8 only assigns already eligible analytical rows to chronological
partitions. It does not clean data, construct targets, engineer features, fit
preprocessing, select a threshold, evaluate a model, or train a model.

## Why chronology is mandatory

Random splitting is prohibited because it would allow later complaints to
inform development evaluated on earlier complaints. Prediction occurs when a
complaint is created, so `created_date` is the only authoritative split
timestamp. Closure, due, and resolution-action timestamps must never control
assignment.

All intervals are half-open: `[start, end)`. Starts are inclusive and ends are
exclusive. This makes a boundary timestamp belong to exactly one partition.

## Candidate-boundary evaluation

Three complete, contiguous policies were evaluated against the verified
35,960-row snapshot:

| Candidate | Train | Validation | Test | Rows (train / validation / test) | Missed rates | Maximum difference |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2024-01-01–2025-07-01 | 2025-07-01–2025-10-01 | 2025-10-01–2026-01-01 | 27,809 / 3,969 / 4,182 | 45.86% / 39.51% / 33.38% | 12.47 points |
| B | 2024-01-01–2025-04-01 | 2025-04-01–2025-09-01 | 2025-09-01–2026-01-01 | 23,699 / 6,762 / 5,499 | 46.02% / 42.62% / 35.04% | 10.98 points |
| C | 2024-01-01–2025-01-01 | 2025-01-01–2025-07-01 | 2025-07-01–2026-01-01 | 19,573 / 8,236 / 8,151 | 44.12% / 49.99% / 36.36% | 13.62 points |

Counts are reproducible snapshot evidence and are recomputed by the pipeline;
they are not hardcoded acceptance values. All three candidates pass the
configured minimums and contain both classes. Candidate B is selected because
it has the smallest split-level target-rate spread, retains 23,699 training
examples, and provides larger multi-month validation and test samples than
Candidate A. Deterministic ranking first filters integrity/minimum failures,
then minimizes target-rate spread, then favors the larger smaller holdout.

## Selected boundaries

- Train: `[2024-01-01T00:00:00Z, 2025-04-01T00:00:00Z)`
- Validation: `[2025-04-01T00:00:00Z, 2025-09-01T00:00:00Z)`
- Test: `[2025-09-01T00:00:00Z, 2026-01-01T00:00:00Z)`

Every split must contain at least 1,000 rows and at least 100 observations of
each target class. Train must precede validation, validation must precede test,
every row must be assigned once, and no complaint identifier may cross splits.

## Temporal drift interpretation

Target prevalence is not stable across the selected period. Candidate B's
missed rate falls from 46.02% in train to 42.62% in validation and 35.04% in
test. Monthly rates range much more widely. This is operationally relevant
drift evidence, not a reason to manufacture balanced random partitions or
reject realistic future data. Downstream evaluation must report time and
segment behavior honestly.

## Test-set governance

> The test set must not be used for feature selection, preprocessing design,
> category-policy decisions, threshold selection, hyperparameter tuning, or
> model selection.

The test set is reserved for one final evaluation after development decisions
are fixed. Validation supports development and model selection. Future
preprocessing—including imputation, scaling, encoding, rare-category grouping,
frequency mappings, and feature selection—must be fit on the training split
only and then applied unchanged to validation and test. Target encoding and
future target aggregates remain prohibited.

## Reproducibility and outputs

Configuration lives in `configs/data/splits.yaml`. Every run records the
source cleaning/raw lineage, eligible file hash and modification time, exact
UTC boundaries, config hash, counts, class balance, date ranges, output paths,
and hashes. Rows are sorted by `created_date`, then `unique_key` using stable
sorting. Outputs are written to a temporary sibling directory, read back,
validated, and atomically finalized; successful runs are immutable and the
latest pointer updates only after success.

Train, validation, and test preserve the Step 7 eligible schema. This supports
auditing but does not make outcome or leakage-prone fields acceptable model
features.

## Known limitations

- The NYC Open Data source is a later-state snapshot and may reflect historical corrections.
- Monthly and split-level target prevalence is temporally unstable.
- Due-date creation-time availability and mutability remain unresolved.
- Two years provide limited evidence about longer-term regimes.
- The split policy is fixed before model development but must be reassessed for a materially changed source period.
