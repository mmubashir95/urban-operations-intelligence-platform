# Split-Aware EDA and Outlier Policy

## Purpose and source authority

Step 9A profiles the governed DSNY / Graffiti resolution-risk population and
produces evidence for baseline feature, transformation, and outlier decisions.
It resolves the latest successful Step 8 run through
`data/splits/resolution_risk/latest.json`, or accepts an explicit split-run
override. Metadata, rules, Parquet hashes, schemas, row and target counts,
identifiers, UTC chronology, and split boundaries are verified before use.

## Split governance

Training data is the sole authority for detailed target analysis, feature
discovery, missingness recommendations, cardinality and rare-category
diagnostics, statistical-outlier thresholds, and feature decisions.

Validation is limited to target-prevalence disclosure, missingness drift,
unknown-category coverage, numeric-range comparison, and structural drift.
Test use is restricted to the same governed structural disclosures. Test
observations and outcomes must never select features, missingness rules,
category thresholds, encoders, clipping thresholds, models, hyperparameters,
or decision thresholds.

## Allowed temporary features

The pipeline derives calendar fields such as creation hour, weekday, calendar
month, quarter, year, and weekend status on in-memory copies only. The source
Parquets are never rewritten and no derived Parquet or final model matrix is
created.

## Missingness, categories, and numeric analysis

Missingness is reported overall in train, by target, by training month, and
across splits. Descriptive bands do not automatically approve or reject a
feature. Categories are profiled for cardinality, support, target association,
rare-threshold sensitivity, and later unknowns. `Unspecified` borough and null
categories remain explicit; incident ZIP remains string-valued and is not
target encoded or treated as continuous.

Numeric analysis reports finite-value distributions, configured percentiles,
target-segment summaries, and split range drift. Descriptive target differences
are associations, not causal findings or proof of predictive value.

## Outlier categories

Domain-invalid findings include malformed/non-finite coordinates, impossible
world coordinates, and partial coordinate pairs. Step 6 world and NYC bounds
are reused without introducing a conflicting authority. Domain-valid geographic
extremes remain observations to retain and monitor.

IQR and percentile findings are statistical diagnostics. They are not applied
to ZIP strings or cyclic/category-coded calendar fields. Temporal extrema are
operational findings that may reflect valid demand, collection, or policy
variation. Step 9A never removes rows, clips values, or imputes outliers.

## Leakage and prediction-time availability

The Step 4 `urban_ops.features.feature_roles` inventory remains authoritative.
Targets, target inputs, final status, resolution fields, eligibility fields,
and outcome-derived fields are blocked. Due-date fields remain unapproved.
Location type, incident ZIP, and coordinates remain conditional until their
creation-time availability and later correction behavior are governed.
Creation-time calendar derivations are safe candidates. Conditional fields are
never silently promoted.

## Recommendations and Notebook 11 boundary

Recommendation priority is deterministic: blocked leakage, all-null, and
zero-variance exclusions precede inclusion or conditional review. Later-split
target associations cannot approve a feature. Suggested missing tokens,
train-median candidates, one-hot encoding, rare grouping, unknown handling,
scaling, or clipping candidates are recommendations only.

Notebook 11 must fit any approved state on train and apply it unchanged to
validation and test. Step 9A fits no imputer, encoder, scaler, selector, or
outlier transformer; creates no model-ready feature matrix; and trains or
evaluates no model.

## Immutable inputs and report publication

Hashes and modification times are captured for all three Parquets, split
metadata, rules snapshot, and Step 8 latest pointer, then rechecked after
analysis and report generation. Reports and figures are written to a temporary
sibling directory, validated, and published using the rollback-safe Step 8
report replacement helpers. Failure preserves prior authoritative reports and
removes temporary paths.

## Known limitations

- The NYC source is a later-state operational snapshot and may contain historical corrections.
- Target prevalence and operational volume vary over time.
- Creation-time availability remains unresolved for several geographic fields.
- Rare-category and outlier thresholds are diagnostics, not selected transformations.
