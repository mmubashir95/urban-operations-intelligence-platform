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
