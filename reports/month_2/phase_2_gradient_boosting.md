# Phase 2 Gradient Boosting — Raw Validation Calibration

## Evaluation boundary

The initial XGBoost classifier was fitted on the frozen Month 1 training split.
This report evaluates the same 6,762 positive-class validation probabilities
created in Phase 2.6. They are raw, uncalibrated probabilities: no Platt
scaling, isotonic regression, sigmoid transformation, threshold, or other
probability calibration method was applied. Test scores and test labels were
not used.

## Validation metrics

| Metric | Gradient Boosting |
|---|---:|
| PR-AUC | 0.4513751420 |
| ROC-AUC | 0.5441074891 |
| Brier Score | 0.2441231897 |

The Brier Score was calculated by the shared Month 1
`evaluate_calibration()` evaluator from validation labels and raw validation
probabilities. Lower Brier Score means lower probability prediction error, but
Brier alone does not establish universal model quality.

## Calibration-bin findings

The canonical calibration table uses ten uniform probability bins and retains
empty bins in the CSV. Most observations lie in three bins:

| Probability bin | Count | Mean predicted risk | Observed miss rate | Evidence |
|---|---:|---:|---:|---|
| 0.3–0.4 | 2,506 | 0.3672 | 0.3911 | Raw risk under-predicts observed misses in this region. |
| 0.4–0.5 | 2,562 | 0.4508 | 0.4500 | Predicted and observed rates are closely aligned. |
| 0.5–0.6 | 1,627 | 0.5311 | 0.4591 | Raw risk over-predicts observed misses in this region. |

The 0.1–0.2 and 0.6–0.7 bins contain only one and two observations,
respectively, so their gaps do not support broad conclusions. The 0.2–0.3 bin
contains 64 observations and also shows over-prediction. Overall, the raw
probabilities are close in the 0.4–0.5 region but exhibit region-specific
under- and over-prediction; no unsupported qualitative calibration label is
assigned.

The complete table is in
`reports/month_2/phase_2_gradient_boosting_calibration.csv`.

## Frozen Logistic Regression comparison

The Logistic Regression values below come from the frozen Month 1 validation
artifact; that model was not retrained.

| Metric | Logistic Regression | Gradient Boosting | Difference (GB − LR) |
|---|---:|---:|---:|
| PR-AUC | 0.4658888310 | 0.4513751420 | -0.0145136890 |
| ROC-AUC | 0.5567976133 | 0.5441074891 | -0.0126901243 |
| Brier Score | 0.2440254783 | 0.2441231897 | +0.0000977113 |

Gradient Boosting has a slightly higher validation Brier Score than the frozen
Logistic Regression benchmark. The difference is small and is comparative
evidence only; it does not by itself establish that either model is universally
better.

## Operational Top-K capacity evaluation

The operational evaluation ranks the same raw Phase 2.6 validation
probabilities from highest to lowest predicted missed-target risk. Equal scores
retain original validation-row order. Capacity means a percentage of all 6,762
validation complaints, and exact reviewed counts use the shared
`ceil(validation rows × capacity)` rule. No probability or classification
threshold is applied.

There are 2,882 actual missed-target complaints in validation. The shared
Top-K results are:

| Capacity | Complaints reviewed | Actual misses captured | Precision@K | Recall@K |
|---:|---:|---:|---:|---:|
| 5% | 339 | 135 | 0.3982 | 0.0468 |
| 10% | 677 | 305 | 0.4505 | 0.1058 |
| 20% | 1,353 | 612 | 0.4523 | 0.2124 |

At 5% capacity, operations would review 339 complaints and capture 135 actual
misses. Approximately 39.82% of that queue would be actual misses, while the
queue would cover 4.68% of all validation misses.

At 10% capacity, 677 reviews capture 305 misses. This adds 338 reviews and 170
captured misses relative to 5%, with Precision@10% of 45.05% and Recall@10% of
10.58%.

At 20% capacity, 1,353 reviews capture 612 misses. This adds 676 reviews and
307 captured misses relative to 10%, with Precision@20% of 45.23% and
Recall@20% of 21.24%.

Recall increases as the reviewed prefix expands. Precision also increases in
these observed results, so the data does not support claiming the usual
precision-decreases-with-capacity pattern here. No preferred capacity or
staffing policy is selected.

The full-precision artifact is
`reports/month_2/phase_2_gradient_boosting_capacity.csv`.

## Frozen Top-K comparison

The Logistic Regression values are reused from the frozen Month 1 validation
capacity artifact; Logistic Regression was not retrained.

| Metric | Logistic Regression | Gradient Boosting | Difference (GB − LR) |
|---|---:|---:|---:|
| Precision@5% | 0.4808 | 0.3982 | -0.0826 |
| Recall@5% | 0.0566 | 0.0468 | -0.0097 |
| Precision@10% | 0.4786 | 0.4505 | -0.0281 |
| Recall@10% | 0.1124 | 0.1058 | -0.0066 |
| Precision@20% | 0.4915 | 0.4523 | -0.0392 |
| Recall@20% | 0.2307 | 0.2124 | -0.0184 |

At each shared review capacity, the frozen Logistic Regression ranking captures
more actual misses and produces a more precise queue than this initial Gradient
Boosting configuration. This is comparative validation evidence only and does
not select a final production model or capacity policy.
