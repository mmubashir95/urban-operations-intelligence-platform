PYTHON ?= .venv/bin/python
INGESTION_CONFIG ?= configs/ingestion/resolution_risk.yaml
VALIDATION_CONFIG ?= configs/data/validation_rules.yaml
CLEANING_CONFIG ?= configs/data/cleaning_rules.yaml
SPLIT_CONFIG ?= configs/data/splits.yaml

.PHONY: ingest-resolution-risk ingest-resolution-risk-dry-run validate-resolution-risk validate-resolution-risk-strict clean-resolution-risk clean-resolution-risk-dry-run split-resolution-risk split-resolution-risk-dry-run

ingest-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG)

ingest-resolution-risk-dry-run:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG) --dry-run

validate-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.validation.pipeline --config $(VALIDATION_CONFIG)

validate-resolution-risk-strict:
	PYTHONPATH=src $(PYTHON) -m urban_ops.validation.pipeline --config $(VALIDATION_CONFIG) --fail-on-error

clean-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.cleaning.pipeline --config $(CLEANING_CONFIG)

clean-resolution-risk-dry-run:
	PYTHONPATH=src $(PYTHON) -m urban_ops.cleaning.pipeline --config $(CLEANING_CONFIG) --dry-run

split-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.splitting.pipeline --config $(SPLIT_CONFIG)

split-resolution-risk-dry-run:
	PYTHONPATH=src $(PYTHON) -m urban_ops.splitting.pipeline --config $(SPLIT_CONFIG) --dry-run
