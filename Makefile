PYTHON ?= .venv/bin/python
INGESTION_CONFIG ?= configs/ingestion/resolution_risk.yaml
VALIDATION_CONFIG ?= configs/data/validation_rules.yaml

.PHONY: ingest-resolution-risk ingest-resolution-risk-dry-run validate-resolution-risk validate-resolution-risk-strict

ingest-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG)

ingest-resolution-risk-dry-run:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG) --dry-run

validate-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.validation.pipeline --config $(VALIDATION_CONFIG)

validate-resolution-risk-strict:
	PYTHONPATH=src $(PYTHON) -m urban_ops.validation.pipeline --config $(VALIDATION_CONFIG) --fail-on-error
