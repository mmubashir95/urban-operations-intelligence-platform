# Phase 2 Gradient Boosting Validation Workflow

## Workflow boundary

The dedicated Month 2 workflow reused frozen Month 1 inputs (`20260806T135114Z_9d945cb2da0eecfc`),
fitted the deterministic XGBoost configuration on 23,699
training rows and 4 frozen features, then generated one raw
positive-class probability per validation complaint. Preprocessing was not
refitted. Test scores and labels were not accessed.

The same raw, uncalibrated validation score array feeds ranking, calibration,
and operational Top-K evaluation. No classification threshold is applied; in
particular, the frozen Logistic Regression threshold `0.49` is not transferred
to Gradient Boosting.

## Validation metrics

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | 0.4513751420 |
| ROC-AUC | 0.5441074891 |
| Brier Score | 0.2441231897 |

## Raw-probability calibration

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

## Operational capacity

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

## Frozen Logistic Regression comparison

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

The initial Gradient Boosting configuration has lower validation PR-AUC and ROC-AUC than the frozen Logistic Regression benchmark. Gradient Boosting also has a slightly higher Brier Score, indicating slightly higher validation probability error. This evidence does not establish
that either model is universally better. No tuning, probability calibration
transformation, threshold selection, or test-set evaluation occurs in this
workflow.
