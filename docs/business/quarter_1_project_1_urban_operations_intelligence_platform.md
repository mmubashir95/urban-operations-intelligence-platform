# Quarter 1 — Production Machine Learning and Data Engineering

## Project 1: Urban Operations Intelligence Platform

### Duration

**Months 1–3**

## Problem

Build an intelligent platform for a city operations team that:

- Predicts incoming service-request volume by location and category.
- Predicts whether a complaint will miss its expected resolution target.
- Predicts resolution time.
- Automatically categorizes complaint descriptions.
- Identifies unusual spikes in complaints.
- Explains which factors influenced each prediction.
- Provides batch predictions, an online API and an operations dashboard.

This is much stronger than a generic churn or house-price project because it demonstrates operational forecasting, tabular ML, NLP, geospatial data, uncertainty and production engineering in one system.

## Public Data Source

Use the official **NYC 311 Service Requests dataset**, which is updated regularly and is accessible through the NYC Open Data API. It includes agency, complaint type, location, timestamps, status and resolution-related fields.

Dataset:

https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9

Use a limited time range and selected agencies rather than loading the entire historical dataset.

## AI Areas Included

| Area | Coverage |
|---|---|
| Data Science | Strong |
| Traditional ML | Strong |
| NLP | Moderate |
| Deep Learning | Optional |
| Computer Vision | None |
| MLOps | Strong |

# Implementation Stages

## Month 1: Data and Baseline System

- Define business metrics before model metrics.
- Build reproducible API ingestion.
- Validate schemas and missing values.
- Detect duplicate and inconsistent records.
- Create time-based train, validation and test splits.
- Establish simple baselines:
  - Historical average
  - Linear/logistic regression
  - Rule-based category baseline
- Build initial exploratory analysis.

## Month 2: Modelling

- Gradient-boosted trees for resolution-risk prediction.
- Regression or survival-style formulation for resolution time.
- Time-series forecasting for complaint volume.
- TF-IDF plus linear model for complaint-text classification.
- Optional transformer comparison.
- Probability calibration.
- SHAP-based local and global explanations.
- Error analysis by agency, complaint category and location.

## Month 3: Production

- FastAPI prediction service.
- Batch prediction pipeline.
- Docker and Docker Compose.
- PostgreSQL feature and prediction storage.
- MLflow experiment tracking and model registry.
- Data validation tests.
- Unit, integration and API contract tests.
- GitHub Actions CI.
- Drift and prediction-distribution monitoring.
- Load and latency testing.
- Dashboard showing volume, risk and model health.

# Track B Theory During Months 1–3

## Mathematics

Learn only what supports the project:

- Vectors, matrices and matrix multiplication
- Mean, variance, covariance and correlation
- Conditional probability and Bayes’ theorem
- Probability distributions
- Maximum likelihood
- Derivatives, gradients and the chain rule
- Loss functions
- Regularization
- Confidence intervals
- Hypothesis testing
- Time-series stationarity and autocorrelation

## Classical ML Concepts

- Linear and logistic regression
- Decision trees
- Random forests
- Gradient boosting
- Bias versus variance
- Underfitting versus overfitting
- Cross-validation
- Data leakage
- Class imbalance
- Precision, recall, F1, ROC-AUC and PR-AUC
- Calibration
- Feature importance versus causality
- Time-based evaluation
- Distribution drift

## Minimal From-Scratch Exercises

Do not build every algorithm from scratch. Implement only:

- Linear regression with gradient descent
- Logistic regression with cross-entropy
- One decision-tree split using Gini or entropy
- K-fold and time-based validation logic
- Core evaluation metrics

# “Why” Questions You Must Answer

- Why is random splitting dangerous for time-dependent data?
- Why can accuracy be misleading?
- Why might a calibrated weaker model be more useful?
- Why do boosted trees often work well on tabular data?
- Why is feature importance not evidence of causation?
- Why can production data drift while model code remains unchanged?
- Why should preprocessing be stored with the model?

# Roles You Can Start Targeting After Project 1

- Machine Learning Engineer
- Applied Data Scientist
- Junior or mid-level MLOps Engineer
- AI Backend Engineer
- Product Data Scientist
- Software Engineer — AI/ML

Because of your existing engineering experience, present yourself as:

> **Senior Software Engineer transitioning into Applied AI**

Do not position yourself as a beginner developer.
