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

When opening a notebook in JupyterLab or Visual Studio Code, select the Python
kernel from the `.venv` virtual environment.
