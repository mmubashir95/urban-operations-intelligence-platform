PYTHON ?= .venv/bin/python
INGESTION_CONFIG ?= configs/ingestion/resolution_risk.yaml

.PHONY: ingest-resolution-risk ingest-resolution-risk-dry-run

ingest-resolution-risk:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG)

ingest-resolution-risk-dry-run:
	PYTHONPATH=src $(PYTHON) -m urban_ops.data.ingest --config $(INGESTION_CONFIG) --dry-run

