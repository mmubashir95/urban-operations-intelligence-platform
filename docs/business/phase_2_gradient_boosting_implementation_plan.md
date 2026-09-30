# Phase 2 — Gradient-Boosted Trees for Resolution Risk
## Implementation Plan

## Objective

Build the first gradient-boosted classifier for:

```text
Complaint created
        ↓
Creation-time features
        ↓
Gradient-Boosted Tree
        ↓
Probability complaint misses resolution target
```

The purpose is **not tuning yet**.

The purpose is to answer:

> Does a basic gradient-boosted model improve ranking and operational usefulness over the frozen Month 1 Logistic Regression?

Hyperparameter tuning belongs to Phase 3.

---

# Phase 2.1 — Reuse and Verify the Frozen Month 1 Inputs

Do not create new features or new data splits.

Reuse:

```text
train
validation
test
```

and the existing Month 1 feature/preprocessing contract.

Your current architecture already exposes this through:

```text
load_frozen_baseline_inputs(...)
```

in:

```text
src/urban_ops/models/baseline_workflow.py
```

The returned `FrozenBaselineInputs` already contains things such as:

```text
inputs.matrices["train"]
inputs.matrices["validation"]
inputs.matrices["test"]

inputs.targets["train"]
inputs.targets["validation"]
inputs.targets["test"]

inputs.feature_names
```

Therefore Phase 2 should **not** rebuild:

```text
data cleaning
feature creation
categorical encoding
numeric preprocessing
time splitting
target generation
```

Those remain frozen.

### Validation Checks

Before fitting the boosted model, verify:

```text
X_train rows == y_train rows
X_validation rows == y_validation rows

feature count == frozen feature-name count

train < validation < test chronologically

target contains only 0/1

no leakage fields have entered the matrix
```

### Important

Phase 2 should initially operate only on:

```text
TRAIN → model fitting
VALIDATION → evaluation
```

Do not use the test labels to decide whether Gradient Boosting is better.

---

# Phase 2.2 — Select One Gradient Boosting Implementation

The Month 2 plan allows:

```text
XGBoost
LightGBM
HistGradientBoosting
```

For this project, start Phase 2 with:

```text
XGBoost
```

provided it is added explicitly to the project dependency set.

Why it fits this project:

```text
binary classification
tabular ML
probability output
sparse feature matrix support
strong boosted-tree implementation
later SHAP compatibility
later regularization/tuning support
```

Do **not** compare XGBoost against LightGBM in this phase.

---

# Phase 2.3 — Add a Gradient Boosting Model Wrapper

Create:

```text
src/urban_ops/models/gradient_boosting.py
```

Conceptually:

```python
class GradientBoostedRiskModel:
    def __init__(...):
        ...

    def fit(
        self,
        X_train,
        y_train,
        *,
        feature_names,
    ):
        ...

    def predict_score(self, X):
        ...

    def predict_proba(self, X):
        ...

    def predict(self, X, *, threshold=0.5):
        ...
```

The public interface should deliberately resemble the existing:

```text
LogisticRegressionBaseline
```

which currently exposes:

```text
fit(...)
predict_score(...)
predict(...)
predict_proba(...)
```

This allows reuse of the existing evaluation architecture instead of special-casing XGBoost everywhere.

---

# Phase 2.4 — Start with a Deliberately Simple Model Configuration

Do not perform hyperparameter tuning here.

For example, configure one deterministic starting model approximately like:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=20260806,
    n_jobs=...,
)
```

Potentially provide a few explicit conservative parameters so the experiment is reproducible.

However:

```text
NO grid search
NO random search
NO Optuna
NO dozens of parameter combinations
NO validation-driven tuning
```

Those belong to Phase 3.

This phase should answer:

```text
Can one sensible Gradient Boosting model outperform
our frozen Logistic Regression baseline?
```

not:

```text
What is the absolute best possible XGBoost configuration?
```

---

# Phase 2.5 — Train Using Training Data Only

Flow:

```text
Frozen X_train
+
Frozen y_train
        ↓
GradientBoostedRiskModel.fit()
        ↓
Fitted Gradient Boosting model
```

Never:

```text
train + validation → fit
```

at this point.

Validation remains independent.

---

# Phase 2.6 — Generate Validation Risk Probabilities

Run:

```text
X_validation
      ↓
Gradient Boosting
      ↓
positive-class probabilities
```

Output conceptually:

```text
Complaint 1 → 0.81
Complaint 2 → 0.17
Complaint 3 → 0.63
Complaint 4 → 0.39
```

The important output is:

```python
y_score
```

not only:

```python
0 / 1 predictions
```

because most of the Month 1 evaluation framework is already built around ranking/risk scores.

---

# Phase 2.7 — Reuse the Existing Ranking Evaluation

Your repository already contains:

```python
evaluate_ranking(...)
```

inside:

```text
src/urban_ops/models/evaluation.py
```

Use the same evaluation function for the boosted model.

Calculate:

```text
PR-AUC
ROC-AUC
```

These are directly comparable with Logistic Regression.

Do not create different metric formulas for Month 2.

---

# Phase 2.8 — Reuse Calibration Evaluation

You also already have:

```python
evaluate_calibration(...)
```

Use it on the raw Gradient Boosting probabilities.

Calculate at least:

```text
Brier Score
calibration bins/table
```

Do not calibrate the model yet.

At this stage, ask:

> How calibrated are its raw probabilities?

Actual calibration methods belong to Month 2 Phase 4.

---

# Phase 2.9 — Reuse the Operational Top-K Evaluation

The existing `evaluation.py` already contains:

```text
rank_by_risk(...)
top_k_metrics(...)
compare_capacity_levels(...)
get_standard_capacity_levels(...)
```

Use those directly.

Evaluate:

```text
Recall@5%
Recall@10%
Recall@20%

Precision@5%
Precision@10%
Precision@20%
```

Conceptually:

```text
Gradient Boosting scores
        ↓
Rank validation complaints highest → lowest
        ↓
Top 5%
Top 10%
Top 20%
        ↓
How many actual misses were captured?
```

This is operationally more useful than focusing only on a small ROC-AUC improvement.

---

# Phase 2.10 — Do Not Introduce a New Threshold Policy Yet

The Month 1 Logistic Regression currently has a frozen operational threshold:

```text
0.49
```

Do not automatically apply:

```text
Gradient Boosting threshold = 0.49
```

because probability distributions from different models are not necessarily comparable.

Also do not perform another large threshold optimization during this first Phase 2 experiment.

For the first comparison, focus on threshold-independent and ranking metrics:

```text
PR-AUC
ROC-AUC
Brier
Recall@K
Precision@K
```

A simple `0.5` threshold may be reported descriptively if useful, but it should not become the new frozen operational decision policy.

Threshold/model-selection decisions can be handled when Phase 3 selects the final boosted configuration and Phase 4 handles calibration.

---

# Phase 2.11 — Build a Dedicated Month 2 Workflow

Do **not** modify `run_baseline_workflow()` to train the boosted model.

That workflow represents the frozen Month 1 system.

Instead create:

```text
src/urban_ops/models/gradient_boosting_workflow.py
```

Possible workflow:

```python
def run_gradient_boosting_workflow(...):
    inputs = load_frozen_baseline_inputs(...)

    model = GradientBoostedRiskModel(...)

    model.fit(
        inputs.matrices["train"],
        inputs.targets["train"],
        feature_names=inputs.feature_names,
    )

    validation_scores = model.predict_score(
        inputs.matrices["validation"]
    )

    ranking = evaluate_ranking(
        inputs.targets["validation"],
        validation_scores,
    )

    calibration = evaluate_calibration(
        inputs.targets["validation"],
        validation_scores,
    )

    capacity = compare_capacity_levels(
        inputs.targets["validation"],
        validation_scores,
    )

    ...
```

This keeps a clean architectural boundary:

```text
Month 1
baseline_workflow.py
        ↓
Frozen benchmark

Month 2
gradient_boosting_workflow.py
        ↓
Advanced model experiment
```

---

# Phase 2.12 — Load the Frozen Month 1 Comparison Numbers

Do not retrain Logistic Regression just because Gradient Boosting is being trained.

Month 1 has already been frozen.

For comparison, use the frozen Month 1 validation evidence/report artifacts wherever practical.

The comparison should be something like:

| Metric | Logistic Regression | Gradient Boosting | Difference |
|---|---:|---:|---:|
| PR-AUC | frozen | new | GB − LR |
| ROC-AUC | frozen | new | GB − LR |
| Brier | frozen | new | GB − LR |
| Recall@5% | frozen | new | GB − LR |
| Recall@10% | frozen | new | GB − LR |
| Recall@20% | frozen | new | GB − LR |

---

# Phase 2.13 — Produce Reports

Recommended:

```text
reports/month_2/
├── phase_2_gradient_boosting_validation.csv
├── phase_2_gradient_boosting_capacity.csv
├── phase_2_model_comparison.csv
└── phase_2_gradient_boosting.md
```

The Markdown report should answer:

```text
1. What model was trained?
2. What frozen data/preprocessing was reused?
3. Which split was used for fitting?
4. Which split was used for evaluation?
5. What configuration was used?
6. What were its PR-AUC / ROC-AUC / Brier results?
7. What were Recall@5/10/20?
8. How does it compare with Logistic Regression?
9. Did ranking materially improve?
10. Are raw probabilities well calibrated?
11. Were any test labels used?
12. Is the model ready for Phase 3 tuning?
```

---

# Phase 2.14 — Add Tests

Create:

```text
tests/unit/models/test_gradient_boosting.py
```

Test at least:

```text
model rejects invalid targets
model rejects row-count mismatch
feature count matches feature_names
predict before fit fails
predict_score returns one score per row
scores lie inside [0, 1]
same random state gives deterministic result
model accepts the frozen matrix representation
```

Also create:

```text
tests/integration/test_gradient_boosting_workflow.py
```

Integration tests should verify:

```text
workflow loads frozen Month 1 inputs
training uses train only
validation evaluation uses validation only
test set is not evaluated
same feature names/order are retained
expected report files are produced
comparison includes frozen Logistic Regression
expected metrics are finite
top-K populations/capacities are correct
```

---

# Phase 2.15 — Protect the Untouched Test Set

This is a hard requirement.

Phase 2:

```text
TRAIN       → fit
VALIDATION  → compare
TEST        → untouched
```

Phase 3:

```text
TRAIN       → fit configurations
VALIDATION  → choose configuration
TEST        → still untouched
```

Only after the boosted configuration is frozen should we do:

```text
Frozen configuration
        ↓
Final TEST evaluation
```

---

# Phase 2.16 — Completion Gate

Phase 2 is complete when all of these are true:

```text
[ ] Frozen Month 1 inputs reused
[ ] No feature-policy change
[ ] No split change
[ ] No leakage-policy change

[ ] One Gradient Boosting implementation selected
[ ] Model wrapper implemented
[ ] Model fitted using train only

[ ] Validation probabilities generated
[ ] PR-AUC calculated
[ ] ROC-AUC calculated
[ ] Brier calculated

[ ] Recall@5% calculated
[ ] Recall@10% calculated
[ ] Recall@20% calculated

[ ] Logistic Regression comparison produced
[ ] Test set remains untouched

[ ] Unit tests pass
[ ] Integration tests pass
[ ] Phase 2 report produced

[ ] No hyperparameter tuning performed
```

Then:

```text
First Gradient Boosting model works
               ↓
Validation comparison understood
               ↓
Phase 2 frozen
               ↓
Move to Phase 3
Advanced Model Selection & Tuning
```

---

# Recommended File Changes

Expected Phase 2 changes:

```text
NEW
src/urban_ops/models/gradient_boosting.py

NEW
src/urban_ops/models/gradient_boosting_workflow.py

NEW
tests/unit/models/test_gradient_boosting.py

NEW
tests/integration/test_gradient_boosting_workflow.py

NEW
reports/month_2/phase_2_gradient_boosting.md

NEW
reports/month_2/phase_2_gradient_boosting_validation.csv

NEW
reports/month_2/phase_2_gradient_boosting_capacity.csv

NEW
reports/month_2/phase_2_model_comparison.csv

POSSIBLY NEW
configs/models/resolution_risk_gradient_boosting.yaml

UPDATE
requirements.txt / dependency configuration
```

Month 1 components should largely remain unchanged:

```text
baseline_workflow.py
baselines.py
evaluation.py

feature preprocessing modules
split logic
target logic

month1_logistic_regression_threshold_decision.json
selected_month_1_baseline.joblib
```

---

# Architecture Principle

We are **not building another ML pipeline from zero**.

```text
                 FROZEN MONTH 1 FOUNDATION
                          │
          ┌───────────────┴────────────────┐
          │                                │
 Logistic Regression              Gradient Boosting
    Month 1 benchmark                Month 2 model
          │                                │
          └───────────────┬────────────────┘
                          ↓
                SAME EVALUATION LAYER
                          ↓
             PR-AUC / ROC-AUC / Brier
               Recall@5/10/20%
                          ↓
                   FAIR COMPARISON
```

This is the practical Phase 2 implementation to complete before moving to Phase 3.
