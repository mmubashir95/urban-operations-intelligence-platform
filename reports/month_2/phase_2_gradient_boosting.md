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
