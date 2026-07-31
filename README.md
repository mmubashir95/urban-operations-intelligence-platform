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
