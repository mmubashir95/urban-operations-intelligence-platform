# Urban Operations Intelligence Platform — Final Project Structure

```text
urban-operations-intelligence/
├── README.md
├── pyproject.toml
├── Makefile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── data-quality.yml
│       └── docker-build.yml
│
├── configs/
│   ├── base.yaml
│   │
│   ├── ingestion/
│   │   ├── discovery.yaml
│   │   ├── development.yaml
│   │   └── production.yaml
│   │
│   ├── data/
│   │   ├── schema.yaml
│   │   ├── validation_rules.yaml
│   │   └── splits.yaml
│   │
│   ├── features/
│   │   ├── common.yaml
│   │   └── resolution_risk.yaml
│   │
│   ├── models/
│   │   ├── baselines.yaml
│   │   ├── resolution_risk.yaml
│   │   ├── resolution_time.yaml
│   │   ├── volume_forecasting.yaml
│   │   ├── text_classification.yaml
│   │   └── anomaly_detection.yaml
│   │
│   ├── evaluation/
│   │   └── metrics.yaml
│   │
│   └── monitoring/
│       └── drift.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── quarantine/
│   │   └── .gitkeep
│   └── external/
│       └── .gitkeep
│
├── docs/
│   ├── business/
│   │   ├── business_problem.md
│   │   ├── business_metrics.md
│   │   └── scope_decision.md
│   │
│   ├── data/
│   │   ├── data_dictionary.md
│   │   ├── data_contract.md
│   │   ├── target_definition.md
│   │   ├── feature_dictionary.md
│   │   └── exclusion_rules.md
│   │
│   ├── modelling/
│   │   ├── baseline_methodology.md
│   │   ├── evaluation_strategy.md
│   │   ├── leakage_analysis.md
│   │   └── model_cards/
│   │
│   ├── architecture/
│   │   ├── system_overview.md
│   │   ├── data_flow.md
│   │   ├── prediction_flow.md
│   │   └── database_design.md
│   │
│   ├── runbooks/
│   │   ├── ingestion.md
│   │   ├── training.md
│   │   ├── batch_prediction.md
│   │   ├── api_service.md
│   │   └── monitoring.md
│   │
│   └── adr/
│       └── README.md
│
├── notebooks/
│   ├── 01_month_1_data_and_baselines/
│   │   ├── 01_data_overview.ipynb
│   │   ├── 02_schema_and_quality_analysis.ipynb
│   │   ├── 03_target_feasibility.ipynb
│   │   ├── 04_scope_selection.ipynb
│   │   ├── 05_exploratory_analysis.ipynb
│   │   └── 06_baseline_evaluation.ipynb
│   │
│   ├── 02_month_2_modelling/
│   │   ├── 01_resolution_risk_modelling.ipynb
│   │   ├── 02_resolution_time_modelling.ipynb
│   │   ├── 03_volume_forecasting.ipynb
│   │   ├── 04_text_classification.ipynb
│   │   ├── 05_anomaly_detection.ipynb
│   │   ├── 06_probability_calibration.ipynb
│   │   └── 07_explainability_and_error_analysis.ipynb
│   │
│   └── 03_month_3_production_validation/
│       ├── 01_batch_prediction_validation.ipynb
│       ├── 02_api_prediction_validation.ipynb
│       └── 03_monitoring_validation.ipynb
│
├── src/
│   └── urban_ops/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── api_client.py
│       │   ├── query_builder.py
│       │   ├── ingestion.py
│       │   ├── schema.py
│       │   ├── validation.py
│       │   ├── cleaning.py
│       │   ├── quality.py
│       │   ├── splitting.py
│       │   └── repositories.py
│       │
│       ├── labels/
│       │   ├── __init__.py
│       │   ├── resolution_risk.py
│       │   └── resolution_time.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── common.py
│       │   ├── temporal.py
│       │   ├── categorical.py
│       │   ├── geospatial.py
│       │   ├── historical.py
│       │   ├── text.py
│       │   └── feature_store.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   │
│       │   ├── common/
│       │   │   ├── __init__.py
│       │   │   ├── interfaces.py
│       │   │   ├── persistence.py
│       │   │   └── registry.py
│       │   │
│       │   ├── resolution_risk/
│       │   │   ├── __init__.py
│       │   │   ├── baselines.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   │
│       │   ├── resolution_time/
│       │   │   ├── __init__.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   │
│       │   ├── volume_forecasting/
│       │   │   ├── __init__.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   │
│       │   ├── text_classification/
│       │   │   ├── __init__.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   │
│       │   └── anomaly_detection/
│       │       ├── __init__.py
│       │       ├── train.py
│       │       └── detect.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── classification.py
│       │   ├── regression.py
│       │   ├── forecasting.py
│       │   ├── calibration.py
│       │   ├── ranking.py
│       │   ├── subgroup.py
│       │   └── reports.py
│       │
│       ├── explainability/
│       │   ├── __init__.py
│       │   ├── global_explanations.py
│       │   └── local_explanations.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── ingest_pipeline.py
│       │   ├── validation_pipeline.py
│       │   ├── feature_pipeline.py
│       │   ├── training_pipeline.py
│       │   ├── evaluation_pipeline.py
│       │   └── batch_prediction_pipeline.py
│       │
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── dependencies.py
│       │   ├── schemas.py
│       │   ├── prediction_service.py
│       │   └── model_loader.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py
│       │   ├── models.py
│       │   └── prediction_repository.py
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── data_drift.py
│       │   ├── prediction_drift.py
│       │   ├── model_performance.py
│       │   └── alerts.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── paths.py
│           ├── datetime.py
│           └── exceptions.py
│
├── apps/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── resolution_risk.py
│   │       ├── resolution_time.py
│   │       └── forecasts.py
│   │
│   └── dashboard/
│       ├── app.py
│       ├── pages/
│       └── components/
│
├── db/
│   ├── migrations/
│   ├── seeds/
│   └── README.md
│
├── docker/
│   ├── api.Dockerfile
│   ├── dashboard.Dockerfile
│   └── pipeline.Dockerfile
│
├── scripts/
│   ├── bootstrap.sh
│   ├── ingest_data.sh
│   ├── validate_data.sh
│   ├── train_baselines.sh
│   ├── run_batch_prediction.sh
│   └── start_services.sh
│
├── tests/
│   ├── unit/
│   │   ├── data/
│   │   ├── labels/
│   │   ├── features/
│   │   ├── models/
│   │   ├── evaluation/
│   │   └── monitoring/
│   │
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_training_pipeline.py
│   │   ├── test_database.py
│   │   └── test_batch_prediction.py
│   │
│   ├── contract/
│   │   ├── test_source_schema.py
│   │   └── test_api_contract.py
│   │
│   ├── e2e/
│   │   └── test_prediction_workflow.py
│   │
│   └── fixtures/
│       ├── sample_311.csv
│       ├── valid_records.json
│       ├── invalid_records.json
│       └── model_inputs.json
│
├── reports/
│   ├── month_1/
│   │   ├── data_overview.md
│   │   ├── data_quality_report.md
│   │   ├── eda_report.md
│   │   ├── baseline_results.md
│   │   ├── month_1_baseline_report.md
│   │   ├── figures/
│   │   └── tables/
│   │
│   ├── month_2/
│   │   ├── modelling_report.md
│   │   ├── error_analysis.md
│   │   ├── explainability_report.md
│   │   ├── figures/
│   │   └── tables/
│   │
│   └── month_3/
│       ├── production_readiness_report.md
│       ├── load_test_report.md
│       ├── monitoring_report.md
│       ├── figures/
│       └── tables/
│
├── artifacts/
│   ├── models/
│   ├── preprocessors/
│   ├── metrics/
│   ├── predictions/
│   └── explanations/
│
└── load_tests/
    ├── locustfile.py
    └── scenarios/
```
