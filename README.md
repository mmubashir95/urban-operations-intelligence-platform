# urban-operations-intelligence-platform

Production-grade machine learning and data engineering platform for NYC 311 service-request forecasting, resolution-risk prediction, categorization, anomaly detection, and operational monitoring.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies from `requirements.txt`:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The Month 1 preprocessing notebooks require `scikit-learn` for the governed
categorical encoding phase.

Launch JupyterLab:

```bash
jupyter lab
```

Execute the Month 1 scope notebooks reproducibly from the repository root:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/05_temporal_stability_analysis.ipynb --inplace --ExecutePreprocessor.timeout=600
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/04_scope_selection.ipynb --inplace --ExecutePreprocessor.timeout=600
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/06_target_and_leakage_definition.ipynb --inplace --ExecutePreprocessor.timeout=600
```

Notebook 04 establishes agency and complaint-type feasibility. Notebook 05 is
the independent source of the final outcome-maturity and date-range decision;
it does not read Notebook 04 outputs. The final Notebook 04 execution reconciles
its general feasibility scorecard to Notebook 05's selected-scope artifact, so
there is no circular dependency. If broad screening evidence is refreshed,
generate Notebook 04's general evidence first, run Notebook 05, then rerun
Notebook 04 for final reconciliation. Notebook 06 then loads Notebook 05's
machine-readable scope authority, rebuilds the current selected population,
and produces the governed Step 4 target and leakage reports.

When opening a notebook in JupyterLab or Visual Studio Code, select the Python
kernel from the `.venv` virtual environment.

## Step 5: reproducible raw API ingestion

Step 5 downloads and preserves the immutable NYC 311 population selected for
resolution-risk modelling. It requires the completed Step 3 scope authority at
`reports/05_temporal_stability/tables/selected_scope_summary.csv`. A Socrata
application token is optional; copy `.env.example` and expose
`NYC_OPEN_DATA_APP_TOKEN` in the environment if one is used.

Validate the resolved scope, query, ordering, and output location without
making count or data-page HTTP requests:

```bash
make ingest-resolution-risk-dry-run
```

Run the complete ingestion:

```bash
make ingest-resolution-risk
```

The equivalent direct commands use `PYTHONPATH=src` and
`python -m urban_ops.data.ingest --config configs/ingestion/resolution_risk.yaml`.
Each successful run creates an immutable partition under
`data/raw/nyc_311/extraction_date=YYYY-MM-DD/run_id=.../` containing:

- `service_requests.parquet`: uncleaned source values as returned;
- `metadata.json`: scope, counts, timing, pagination, integrity, and provenance;
- `query.sql`: the credential-free count and page-query audit.

The latest-run report is generated under `reports/07_api_ingestion/`, including
`ingestion_summary.md`, `tables/page_summary.csv`, and
`tables/extraction_validation.csv`. Raw runs are never manually edited or
silently overwritten. The query and ordering are deterministic and every run
stores its own provenance, but NYC Open Data is live and may receive historical
corrections. The saved raw extraction is the immutable downstream source for
that snapshot.

## Step 6: raw data validation

Step 6 finds, measures, and reports data-quality issues in the latest successful
Step 5 extraction. It does not clean records. In particular, it never rewrites
`service_requests.parquet`, fills missing values, standardizes saved categories,
removes duplicates, creates a target, or writes a cleaned dataset.

Run the default validation threshold:

```bash
make validate-resolution-risk
```

The equivalent direct command is:

```bash
PYTHONPATH=src .venv/bin/python -m urban_ops.validation.pipeline \
  --config configs/data/validation_rules.yaml
```

The command locates the latest run whose validated `metadata.json` has
`completion_status: success`; it does not select a directory by timestamp
alone. Reports are written to `reports/08_data_validation/`. The main evidence
is `validation_summary.md` plus the CSV tables for schema, scope, missingness,
timestamps, chronology, duplicates, categories, status, geography, provisional
target readiness, severities, and proposed Step 7 actions.

Socrata timestamp strings without an explicit offset are interpreted as UTC in
temporary validation views. Raw timestamp strings are preserved unchanged, and
timezone-naive versus timezone-aware counts are reported separately.

The default command exits non-zero only for `CRITICAL` findings, meaning the raw
run cannot safely proceed to cleaning. `ERROR` findings require an explicit
cleaning rule, `WARNING` findings may affect later modelling, and `INFO` findings
are profiles or expected conditions. Use `--fail-on-error` through
`make validate-resolution-risk-strict`, or pass `--fail-on-warning` directly,
for stricter automation.

Validation and cleaning are deliberately separate:

- Validation finds, measures, and reports issues.
- Cleaning applies approved corrections in Step 7.

Execute the evidence notebook after validation with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/07_data_validation.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
```

## Step 7: governed data cleaning

Step 6 measured the issues. Step 7 applies approved, auditable transformations.
It requires matching Step 6 evidence for the same raw run and hash, zero critical
validation findings, and agreement with the Step 3 scope authority.

The cleaning policy is documented in `docs/cleaning_policy.md`. In summary:

- timestamps become UTC-aware in processed outputs, without guessing invalid values;
- missing due and closure dates are not imputed and remain target-ineligible;
- configured categories are trimmed and configured true blanks become null;
- literal `UNKNOWN` channels and `Unspecified` boroughs remain explicit;
- missing geography is preserved and never inferred;
- one deterministic canonical exact duplicate is retained, while complete
  conflicting groups are preserved but excluded;
- Step 4 remains the authority for eligibility, exclusion precedence, target
  construction, status governance, and leakage prevention.

Verify inputs, counts, and the proposed output without writing artifacts:

```bash
make clean-resolution-risk-dry-run
```

Run cleaning:

```bash
make clean-resolution-risk
```

The equivalent direct command is
`PYTHONPATH=src .venv/bin/python -m urban_ops.cleaning.pipeline --config configs/data/cleaning_rules.yaml`.
Each successful immutable run is written under
`data/processed/resolution_risk/run_id=<timestamp>_<cleaning-hash>/` with:

- `cleaned_service_requests.parquet` for every deterministically retained row;
- `eligible_service_requests.parquet` for governed binary targets;
- `excluded_service_requests.parquet` for nullable targets and explicit reasons;
- `cleaning_metadata.json` for lineage, counts, hashes, and decisions;
- `cleaning_rules_snapshot.yaml` for the exact applied configuration.

`data/processed/resolution_risk/latest.json` changes only after a successful run.
Reports are generated under `reports/09_data_cleaning/`. Raw data is never
rewritten, and processed analytical data is not yet a feature matrix or split.

Execute the cleaning evidence notebook with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/08_data_cleaning.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
```

## Step 8: time-based splitting

Step 8 partitions the latest successful Step 7 eligible dataset into governed
chronological train, validation, and test data. Random splitting is prohibited:
prediction occurs at complaint creation, so `created_date` is the only split
timestamp. All ranges use half-open `[start, end)` intervals.

Three plausible boundary policies are evaluated against actual monthly volume,
class counts, coverage, and target-rate drift. Candidate B is selected because
it passes all minimums, provides substantial validation and test windows, and
has the smallest split-level target-rate spread:

- Train: `[2024-01-01, 2025-04-01)`
- Validation: `[2025-04-01, 2025-09-01)`
- Test: `[2025-09-01, 2026-01-01)`

Each split must contain at least 1,000 rows and 100 rows per target class.
Temporal drift is reported rather than treated as automatic rejection. The
test set is protected from feature selection, preprocessing design, category
decisions, threshold selection, tuning, and model selection.

Verify the source and complete assignment without writing artifacts:

```bash
make split-resolution-risk-dry-run
```

Create an immutable split run:

```bash
make split-resolution-risk
```

Outputs are written under `data/splits/resolution_risk/split_id=.../` as
`train.parquet`, `validation.parquet`, `test.parquet`, `split_metadata.json`,
and an exact rules snapshot. Reports are written under
`reports/10_time_based_splitting/`. Split files and reports are prepared and
validated in temporary sibling directories, then the Step 7 source hash and
modification time are rechecked before publication. The pipeline provides
rollback-safe publication across the split-run directory, report directory,
and `latest.json`: a failure removes the new split, restores or preserves the
previous reports and pointer, and removes temporary paths. `latest.json`
changes only after the split and reports have been published successfully.

These outputs preserve the complete eligible analytical schema for auditing;
they are not final feature matrices. No imputer, encoder, scaler, selector, or
model is fitted. Future preprocessing must be fit on train only and applied
unchanged to validation and test.

Execute the evidence notebook with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/09_time_based_splitting.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
```

## Step 9A: split-aware EDA and outlier analysis

Step 9A verifies the latest successful Step 8 split and builds governed
train-first exploratory evidence. Train is the authority for feature discovery,
missingness, cardinality, rare-category and statistical-outlier diagnostics,
and recommendations. Validation and test are restricted to target-prevalence,
missingness, unknown-category, numeric-range, and structural-drift disclosure;
test evidence never selects a feature or transformation.

Run all analysis without publishing reports:

```bash
make eda-resolution-risk-dry-run
```

Publish the validated report inventory:

```bash
make eda-resolution-risk
```

The input is the split referenced by
`data/splits/resolution_risk/latest.json`. Reports are published rollback-safely
under `reports/11_split_aware_eda/` with a Markdown summary, 34 stable-schema
tables, and 13 focused figures. Source split hashes and modification times are
rechecked, and train, validation, test, split metadata, rules, and the latest
pointer remain unchanged.

Step 9A derives creation-time calendar fields only in memory. It does not remove
rows, clip or impute values, fit preprocessing, create a final feature matrix,
train a model, evaluate a model, or select a threshold. The complete governance
and Notebook 11 handoff are documented in
`docs/split_aware_eda_policy.md`.

Execute Notebook 10 from the repository root:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/10_split_aware_eda.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
```

Notebook 10 defaults to `PUBLISH_EDA_REPORTS = False`, so ordinary exploratory
execution does not replace authoritative reports. Set the flag deliberately
only when notebook-driven publication is intended.

## Step 9B: Notebook 11 feature-policy freeze

Notebook 11 freezes the baseline feature eligibility policy before any
feature creation or preprocessing. Its machine-readable authority is
`configs/features/resolution_risk_baseline.yaml`, validated against Step 4
leakage governance and the existing Notebook 10 evidence tables.

Execute the policy notebook from the repository root:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/11_feature_engineering_and_preprocessing.ipynb \
  --inplace --ExecutePreprocessor.timeout=600
```

The frozen policy approves only `created_hour`, `created_day_of_week`,
`created_month`, and `is_weekend` to proceed to deterministic creation. It does
not create those fields, fit preprocessing, create a feature matrix, train a
model, or modify the Step 8 splits or Notebook 10 reports. The full rationale
and scope boundary are in `docs/baseline_feature_policy.md`.

## Step 9C: deterministic feature creation

Notebook 11 now consumes the frozen baseline policy and creates the approved
calendar fields from the UTC `created_date` source: `created_hour`,
`created_day_of_week`, `created_month`, and `is_weekend`. The reusable builder
is `urban_ops.features.temporal.derive_approved_temporal_features`.

Creation is stateless and non-mutating. It uses the same rules for train,
validation, and test, learns no statistics, writes no feature datasets, and
performs no missing-value handling, encoding, scaling, or modelling. The source
contract and exact formulas are documented in
`docs/deterministic_feature_creation.md`.

## Categorical missing-value handling

Notebook 11 now defines the reusable constant rule `null → __MISSING__` for
categorical values. The token is configured in
`configs/features/resolution_risk_categorical_missing.yaml`, and the production
implementation is `urban_ops.features.categorical_missing`.

The frozen feature policy remains authoritative: `borough`, `location_type`,
and `incident_zip` are still conditional, so configuration support does not
activate or transform them in the current baseline. See
`docs/categorical_missing_values.md` for the contract and deferred boundaries.

## Rare and unseen categorical handling

Phase 4 adds an explicit training-only fit/transform lifecycle in
`urban_ops.features.rare_unseen`. Known low-count training categories map to
`__RARE__`, later values absent from the frozen training vocabulary map to
`__UNKNOWN__`, and `__MISSING__` remains unchanged. The minimum-count candidate
and its Notebook 10 evidence are recorded in
`configs/features/resolution_risk_categorical_cardinality.yaml`.

The real baseline has no active categorical fields, so Phase 4 is a governed
production no-op and no conditional geography is transformed. Notebook 11
records the decision and evidence. No categorical encoding or model training
is implemented. See `docs/rare_and_unseen_categories.md`.
