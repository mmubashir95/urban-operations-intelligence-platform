# Phase 2 — Gradient Boosting Validation Report

## 1. Experiment Objective

This first Gradient Boosting experiment tests whether a deterministic boosted
tree model improves missed-target risk ranking over the frozen Month 1 Logistic
Regression benchmark. Phase 2 establishes reproducible validation evidence; it
does not select a production model or operational policy.

## 2. Model Trained

The workflow trained the existing `XGBoost` binary classifier
through the `GradientBoostedRiskModel` project wrapper. Class `1` represents a complaint
that misses its expected resolution target.

## 3. Frozen Inputs Reused

The dedicated Month 2 workflow reused frozen Month 1 inputs (`20260806T135114Z_9d945cb2da0eecfc`),
including the chronological split membership, target definition, fitted
preprocessing outputs, sparse feature matrices, and ordered feature names.
The matrices contain 4 features. Preprocessing was not refitted
and feature columns were not reordered.

Month 1 inputs remained frozen; only the model implementation changed.

## 4. Evaluation Boundary

- **TRAIN:** fitted the model on 23,699 rows.
- **VALIDATION:** evaluated 6,762 rows using ranking,
  raw-probability calibration, Top-K capacity, and frozen-model comparison.
- **TEST:** untouched; no test scores or labels were accessed.

The same raw, uncalibrated validation score array feeds ranking, calibration,
and operational Top-K evaluation. No classification threshold is applied; in
particular, the frozen Logistic Regression threshold `0.49` is not transferred
to Gradient Boosting.

## 5. Model Configuration

Configuration version: `1`. Only explicitly configured
parameters are shown.

| Parameter | Value |
|---|---|
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `n_estimators` | `100` |
| `learning_rate` | `0.1` |
| `max_depth` | `3` |
| `random_state` | `20260806` |
| `n_jobs` | `1` |

No hyperparameter tuning was performed in Phase 2.

## 6. Validation Metrics

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | 0.4513751420 |
| ROC-AUC | 0.5441074891 |
| Brier Score | 0.2441231897 |

Higher PR-AUC and ROC-AUC indicate stronger ranking. Lower Brier Score indicates
lower probability prediction error.

Machine-readable values are in
`reports/month_2/phase_2_gradient_boosting_validation.csv`.

## 7. Raw Probability Calibration

The Brier Score and ten uniform calibration bins use raw validation
probabilities. Empty bins remain present in the CSV; populated bins are shown
below.

| Probability bin | Count | Mean predicted risk | Observed miss rate |
|---:|---:|---:|---:|
| 0.1–0.2 | 1 | 0.1990 | 0.0000 |
| 0.2–0.3 | 64 | 0.2434 | 0.0312 |
| 0.3–0.4 | 2,506 | 0.3672 | 0.3911 |
| 0.4–0.5 | 2,562 | 0.4508 | 0.4500 |
| 0.5–0.6 | 1,627 | 0.5311 | 0.4591 |
| 0.6–0.7 | 2 | 0.6091 | 0.0000 |

Differences between predicted and observed rates show region-specific under-
or over-prediction. No probability calibration transformation or qualitative
pass/fail rule is applied.

The evidence does not support a binary "well calibrated" label: the raw
probabilities show region-specific under- and over-prediction. No calibration
method was applied. Full-precision bins are in
`reports/month_2/phase_2_gradient_boosting_calibration.csv`.

## 8. Operational Top-K Evaluation

Capacity is a percentage of the 6,762-complaint validation
population. Counts use the shared `ceil(n × capacity)` rule.

| Capacity | Reviews | Actual misses captured | Precision@K | Recall@K |
|---:|---:|---:|---:|---:|
| 5% | 339 | 135 | 0.3982 | 0.0468 |
| 10% | 677 | 305 | 0.4505 | 0.1058 |
| 20% | 1,353 | 612 | 0.4523 | 0.2124 |

Expanding the reviewed prefix increases captured misses and recall. These
figures describe validation evidence only; no capacity or staffing policy is
selected.

Full-precision capacity evidence is in
`reports/month_2/phase_2_gradient_boosting_capacity.csv`.

## 9. Frozen Logistic Regression Comparison

The Logistic Regression values come from frozen Month 1 validation CSV
evidence. Logistic Regression is not retrained. Gradient Boosting values come
from the current Month 2 validation workflow, and every difference is
`Gradient Boosting - Logistic Regression`.

- Validation metrics source: `reports/tables/baseline_validation_results.csv`
- Capacity metrics source: `reports/tables/logistic_regression_validation_capacity_comparison.csv`
- Frozen model: `Logistic Regression`
- Frozen split: `validation`

| Metric | Logistic Regression | Gradient Boosting | Difference (GB − LR) |
|---|---:|---:|---:|
| PR-AUC | 0.4659 | 0.4514 | -0.0145 |
| ROC-AUC | 0.5568 | 0.5441 | -0.0127 |
| Brier Score | 0.2440 | 0.2441 | +0.0001 |
| Precision@5% | 0.4808 | 0.3982 | -0.0826 |
| Recall@5% | 0.0566 | 0.0468 | -0.0097 |
| Precision@10% | 0.4786 | 0.4505 | -0.0281 |
| Recall@10% | 0.1124 | 0.1058 | -0.0066 |
| Precision@20% | 0.4915 | 0.4523 | -0.0392 |
| Recall@20% | 0.2307 | 0.2124 | -0.0184 |

Full-precision differences are in
`reports/month_2/phase_2_model_comparison.csv`.

## 10. Ranking Improvement Assessment

The initial Gradient Boosting configuration has lower validation PR-AUC and ROC-AUC than the frozen Logistic Regression benchmark. Precision@K and Recall@K are also lower at every shared capacity. Therefore, the initial untuned
Gradient Boosting configuration
did not materially improve validation ranking over the frozen Logistic
Regression benchmark. This does not establish that Gradient Boosting is
universally worse or unusable.

Gradient Boosting also has a slightly higher Brier Score, indicating slightly higher validation probability error.

## 11. Threshold Policy Status

Gradient Boosting operational threshold selection remains deferred. The frozen
Logistic Regression threshold `0.49` was not transferred to Gradient Boosting,
and no `0.5` or other threshold is selected by this report.

## 12. Test-Set Protection

No test labels were used, no test probabilities were generated, and no test
evaluation was performed. Phase 2 reporting contains validation evidence only.

## 13. Phase 3 Readiness

Yes—the experiment is technically ready to proceed to Phase 3 model
selection/tuning because the train-only fit, validation evaluation stack,
frozen comparison, and deterministic reports work end to end while the test set
remains protected. This does not mean the model is production-ready.

## 14. Conclusion

The initial deterministic Gradient Boosting pipeline is reproducible, but its
current validation ranking, Top-K performance, and Brier Score do not improve
on the frozen Logistic Regression evidence. Phase 2 changes no features,
preprocessing, threshold policy, calibration method, or test-set boundary.
