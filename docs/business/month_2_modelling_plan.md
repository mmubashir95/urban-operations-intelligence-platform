# Month 2 — Modelling Plan
## Project 1: Urban Operations Intelligence Platform

## Month 2 Goal

After Month 1 is complete, the project moves from **data foundation + baseline evaluation** into **advanced modelling and model understanding**.

The main goal for Month 2 is:

> **Move beyond Logistic Regression and simple baselines into stronger models for the different Urban Operations prediction problems, while keeping evaluation operationally meaningful.**

Month 2 covers:

1. Gradient-boosted trees for resolution-risk prediction.
2. Resolution-time prediction.
3. Complaint-volume forecasting.
4. Complaint-text classification.
5. Optional transformer comparison.
6. Probability calibration.
7. SHAP explanations.
8. Detailed error analysis.

---

# Phase 1 — Freeze the Month 1 Baseline

Before advanced modelling, preserve the Month 1 result.

The existing resolution-risk problem remains:

```text
Complaint created
        ↓
Creation-time features
        ↓
Predict:
Will this complaint miss its resolution target?
```

The Month 1 Logistic Regression becomes the benchmark.

Track the frozen baseline metrics:

```text
Logistic Regression

PR-AUC
ROC-AUC
Recall
Precision
Brier Score
Recall@5%
Recall@10%
Recall@20%
```

Do not keep changing the baseline every time the advanced model changes.

The key question becomes:

> Does the Month 2 model actually improve on the Month 1 system?

---

# Phase 2 — Gradient-Boosted Trees for Resolution Risk

This should be the first major Month 2 modelling task.

Conceptually:

```text
Month 1

Features
   ↓
Logistic Regression
   ↓
Resolution-risk probability


Month 2

Same prediction problem
Same chronological split
Same leakage policy
       ↓
Gradient-Boosted Trees
       ↓
Resolution-risk probability
```

A practical implementation can use a suitable gradient-boosting library or implementation such as:

- XGBoost
- LightGBM
- HistGradientBoosting

The project should not depend on exhaustive model comparison. Pick one suitable implementation first and learn the modelling concepts properly.

Key learning sequence:

```text
Decision Trees
        ↓
Boosting
        ↓
Sequential Error Correction
        ↓
Gradient Boosting
```

Compare the boosted model directly against Month 1 models.

Example comparison:

| Model | PR-AUC | Brier | Recall@10% |
|---|---:|---:|---:|
| Historical Rate | Baseline | Baseline | Baseline |
| Logistic Regression | Month 1 | Month 1 | Month 1 |
| Gradient Boosting | Month 2 | Month 2 | Month 2 |

Learning goal:

> Understand why boosted trees often work well on tabular data.

---

# Phase 3 — Advanced Model Selection and Tuning

Once the first boosted-tree model works, improve it carefully.

Do not immediately move to NLP or forecasting.

Recommended flow:

```text
Baseline Tree Model
      ↓
Hyperparameter Candidates
      ↓
Validation-Period Evaluation
      ↓
Model Comparison
      ↓
Freeze Chosen Configuration
      ↓
Final Test Evaluation
```

Important parameters may include:

```text
Number of Trees
Learning Rate
Maximum Tree Depth
Minimum Samples
Regularization
```

The purpose is not to perform a huge hyperparameter search.

The main learning objective is:

> How do we improve a model without leaking information from validation or test data?

Continue using the chronological train / validation / test discipline established in Month 1.

---

# Phase 4 — Probability Calibration

Once the stronger classifier is stable, evaluate whether its probabilities are trustworthy.

Example:

```text
Complaint A → 0.80 risk
Complaint B → 0.20 risk
```

We do not only want:

```text
A > B
```

We want the probabilities themselves to have operational meaning.

For example:

```text
Among complaints predicted around 80% risk,
approximately 80% should actually miss their target.
```

Compare:

```text
Raw Probabilities
       ↓
Calibration Curve
Brier Score

vs.

Calibrated Probabilities
       ↓
Calibration Curve
Brier Score
```

Learning goal:

> Understand why a calibrated weaker model can sometimes be more useful operationally than a stronger but poorly calibrated model.

---

# Phase 5 — SHAP Explainability

After the resolution-risk model is stable, explain its predictions.

There are two explanation levels.

## Global Explanations

Understand which features generally influence predictions across the dataset.

Example features:

```text
created_hour
created_day_of_week
created_month
is_weekend
...
```

Question:

> Which features influence the model most overall?

## Local Explanations

Explain an individual prediction.

Example:

```text
Complaint #123

Predicted Risk = 0.73

Factors Increasing Risk:
Feature A
Feature B

Factors Reducing Risk:
Feature C
```

Learning goal:

> Understand the difference between model feature importance and causality.

Feature importance should not be interpreted as proof that a feature causes the outcome.

---

# Phase 6 — Advanced Error Analysis

Month 1 introduced evaluation.

Month 2 should go deeper into where the model succeeds and fails.

Investigate:

```text
False Positive:
Predicted High Risk
Actually Completed On Time

False Negative:
Predicted Low Risk
Actually Missed Target
```

False negatives are especially important operationally because those are complaints the model failed to surface for intervention.

Analyze errors across:

```text
Complaint Category
Agency, if applicable
Location
Time Period
Available Geographic Grouping
Prediction-Score Range
```

The goal moves beyond:

> Our PR-AUC is X.

Toward:

> Where does the model work, where does it fail, and why?

---

# Phase 7 — Resolution-Time Prediction

Introduce the second major machine-learning problem.

The existing classification problem asks:

```text
Will the complaint miss its target?

YES / NO
```

Resolution-time prediction asks:

```text
How long will this complaint take to resolve?
```

Conceptually:

```text
Complaint Created
       ↓
Creation-Time Features
       ↓
Regression Model
       ↓
Expected Resolution Duration

Example:
37 hours
4.2 days
8 days
```

Recommended breakdown:

```text
7.1 Define the Resolution-Time Target

7.2 Study the Duration Distribution

7.3 Establish a Simple Duration Baseline

7.4 Train a Regression Model

7.5 Evaluate the Regression Model

7.6 Consider Whether Survival Modelling Adds Value
```

Start with standard regression before introducing survival analysis unless the data or business problem clearly justifies it.

---

# Phase 8 — Complaint-Volume Forecasting

Next introduce the time-series problem.

Instead of one complaint producing one prediction:

```text
Complaint #123
       ↓
Risk = 72%
```

Forecasting works at an operational aggregate level:

```text
Historical Complaint Volume
        ↓
Time-Series Model
        ↓
Expected Future Complaint Volume
```

Example:

```text
Tomorrow:
120 complaints

Next Monday:
155 complaints
```

Forecasts may eventually be produced by:

```text
Overall Volume
Location
Complaint Category
```

Topics to study:

```text
Trend
Seasonality
Lags
Rolling Statistics
Autocorrelation
Stationarity
Time-Series Backtesting
```

The focus should be on proper time-based evaluation rather than random train/test splitting.

---

# Phase 9 — Complaint Text Classification

Now introduce NLP into the project.

Use a traditional NLP baseline first:

```text
Complaint Text
      ↓
Text Preprocessing
      ↓
TF-IDF
      ↓
Linear Classifier
      ↓
Predicted Complaint Category
```

Example:

```text
"The traffic light at Main Street has stopped working"
                ↓
             TF-IDF
                ↓
          Linear Classifier
                ↓
        Traffic Signal Issue
```

This phase should teach an end-to-end traditional NLP workflow before introducing transformer models.

---

# Phase 10 — Optional Transformer Comparison

Only after the TF-IDF system is reliable should a transformer be considered.

Recommended progression:

```text
TF-IDF + Linear Model
        ↓
Reliable Benchmark
        ↓
Optional Pretrained Transformer
        ↓
Compare
```

Compare:

```text
Prediction Quality
Inference Speed
Memory Usage
Implementation Complexity
Production Complexity
```

Questions to answer:

> Does the transformer meaningfully improve quality?

> How much slower is it?

> How much more memory does it require?

> Would the improvement justify the production complexity?

This phase is optional and should not consume most of Month 2.

---

# Recommended Month 2 Execution Order

```text
MONTH 2 — MODELLING
│
├── Phase 1
│   Freeze Month 1 Baseline
│
├── Phase 2
│   Gradient-Boosted Resolution-Risk Model
│
├── Phase 3
│   Model Tuning and Selection
│
├── Phase 4
│   Probability Calibration
│
├── Phase 5
│   SHAP Explanations
│
├── Phase 6
│   Advanced Error Analysis
│
├── Phase 7
│   Resolution-Time Prediction
│
├── Phase 8
│   Complaint-Volume Forecasting
│
├── Phase 9
│   TF-IDF Complaint-Text Classification
│
└── Phase 10
    Optional Transformer Comparison
```

---

# Suggested Four-Week Breakdown

## Week 1 — Advanced Resolution-Risk Modelling

Focus:

```text
Gradient Boosting
      ↓
Model Tuning
      ↓
Validation Comparison
      ↓
Comparison with Logistic Regression
```

Expected output:

- first gradient-boosted risk model;
- validation comparison with Month 1 baselines;
- selected model configuration;
- frozen model-selection decision.

---

## Week 2 — Model Reliability and Explainability

Focus:

```text
Probability Calibration
        ↓
SHAP
        ↓
Advanced Error Analysis
```

Expected output:

- calibration report;
- calibrated probability comparison;
- SHAP global explanations;
- SHAP local explanations;
- subgroup and failure-case analysis.

---

## Week 3 — New Operational Prediction Problems

Focus:

```text
Resolution-Time Regression
        +
Complaint-Volume Forecasting
```

Expected output:

- resolution-time target definition;
- regression baseline;
- advanced regression model;
- regression evaluation;
- volume-forecasting dataset;
- forecasting baseline;
- time-series evaluation.

---

## Week 4 — NLP and Month 2 Consolidation

Focus:

```text
Complaint Text
      ↓
TF-IDF
      ↓
Linear Model
      ↓
Evaluation
```

Then optionally:

```text
TF-IDF Benchmark
      ↓
Transformer Comparison
```

Expected output:

- TF-IDF classifier;
- text-classification evaluation;
- optional transformer comparison;
- consolidated Month 2 modelling report;
- Month 3 production-readiness decisions.

---

# What Not to Start in Month 2

Do not start production engineering too early.

The following belong to Month 3:

```text
FastAPI Prediction Service
Batch Prediction Pipeline
Docker
Docker Compose
PostgreSQL
MLflow
GitHub Actions CI
Drift Monitoring
Prediction-Distribution Monitoring
Load Testing
Latency Testing
Operations Dashboard
```

Month 2 should focus on producing reliable, evaluated, explainable models first.

---

# Immediate Next Step

After Month 1 is completely frozen, the next task should be:

> **Month 2 — Phase 1: Understand what gradient-boosted trees add beyond the frozen Logistic Regression baseline before implementing them.**

Recommended workflow for each Month 2 phase:

```text
1. Understand the phase
       ↓
2. Divide it into small sub-parts
       ↓
3. Prepare implementation plan
       ↓
4. Cursor implementation
       ↓
5. Codex testing + review
       ↓
6. Claude final validation
       ↓
7. Freeze the phase
       ↓
8. Move to the next phase
```

This keeps Month 2 consistent with the disciplined workflow used during Month 1.
