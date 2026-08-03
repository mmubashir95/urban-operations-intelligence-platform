# Data Cleaning Policy

## Purpose and authority

This is the authoritative Month 1 Step 7 cleaning contract for the DSNY /
Graffiti resolution-risk population. The input is the latest successful
immutable Step 5 raw run, reconciled to the Step 6 evidence under
`reports/08_data_validation` and to the Step 3 selected-scope authority.

Step 6 measures data-quality issues. Step 7 applies only the transformations
approved here and records their effects. It does not perform feature
engineering, data splitting, exploratory modelling analysis, or training.

## Raw immutability and auditability

The Step 5 `service_requests.parquet` file is never rewritten. Its SHA-256 and
modification time are checked around cleaning. Processed outputs are written to
an atomic immutable run containing rule and provenance snapshots, counts,
hashes, action reports, and explicit exclusion reasons.

## Timestamp policy

`created_date`, `closed_date`, `due_date`, and
`resolution_action_updated_date` become UTC-aware processed timestamps.
Timezone-naive source strings are interpreted as UTC, matching Step 6.
Nulls remain null. Invalid values become `NaT` only in the processed copy and
receive parse-failure flags; the original text remains in the raw artifact.
Dates are never guessed, imputed, or changed to repair chronology.

## Missing-value policy

- Missing identifiers and scope fields are never imputed and prevent a safe
  cleaning run through the critical Step 6 gate.
- Missing `due_date` or `closed_date` is preserved, remains target-ineligible,
  and receives a nullable target.
- Approved blank values in `descriptor`, `descriptor_2`, `borough`, and
  `location_type` become null after configured whitespace trimming.
- Missing coordinates and ZIP codes remain missing. ZIP values remain strings,
  including leading zeroes. No geography is invented.

## Category policy

Only configured columns are trimmed. Repeated-space collapsing and category
mapping occur only when explicitly configured. Automatic title-casing,
case-folding, fuzzy merging, and inferred geography are prohibited. Literal
`UNKNOWN` in `open_data_channel_type` and `Unspecified` borough values remain
explicit categories.

## Duplicate policy

Duplicate groups use the governed material-column authority shared with Step 4
and Step 6. Exact duplicate copies retain one deterministic canonical record
selected by a stable source-value fingerprint; redundant copies are audited.
Every member of a conflicting `unique_key` group remains in the cleaned data
but is excluded from the eligible output. No conflicting winner is selected.

## Chronology and status policy

`due_date >= created_date` and `closed_date >= created_date` are required for
target eligibility. Violations remain in cleaned and excluded datasets with
unchanged processed timestamps and the Step 4 exclusion reason. Step 4 owns the
allowed/excluded status policy; Step 7 does not add or silently map statuses.

## Eligibility and target

Step 7 calls `urban_ops.features.eligibility.evaluate_target_eligibility` and
does not define another eligibility formula or exclusion precedence. It calls
`urban_ops.features.target.build_missed_resolution_target`; late closure is
`1`, closure on or before due is `0`, and ineligible rows are nullable `NA`.

## Outputs

Each processed run contains cleaned all-records, eligible, and excluded
Parquet datasets, cleaning metadata, and an exact cleaning-rule snapshot. The
cleaned dataset retains outcome and audit columns for governed downstream use;
it is not itself a model feature matrix.

## Leakage policy

The Step 4 feature-role inventory and leakage validator remain authoritative.
Identifiers, target inputs, final status, resolution fields, eligibility and
target fields are blocked. Conditional fields remain unapproved. All-null
`descriptor_2` and zero-variance `open_data_channel_type` are explicitly
unusable for the baseline even though the latter is otherwise creation-time
safe.

## Known limitations

The source is a later-state API snapshot and can reflect historical updates.
Due-date creation-time availability and mutability remain unproven. Category
trimming does not establish semantic equivalence. Quality flags produced by
cleaning are audit fields, not automatically approved model features.
