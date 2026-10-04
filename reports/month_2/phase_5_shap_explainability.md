# Phase 5 — SHAP explainability

> **Interpretation guardrail:** SHAP explains the behavior of the current model. It does not establish that the model is accurate. A feature can have high SHAP influence inside a weak model without being strongly predictive in the real world.

SHAP attributes model output to model inputs; it does not establish causal relationships.

## Frozen boundary

The explained model is Phase 3 `shallow` using the frozen TRAIN-only fit, split `20260806T135114Z_9d945cb2da0eecfc`, and feature fingerprint `d6620d99301f80884487140bde487479f58fe72c1e209328524fa27e8b5a2271`. Phase 4 remains frozen as `ISOTONIC` and is not attributed: calibration is post-processing. No model selection, retuning, calibration fitting, feature change, preprocessing change, split change, or threshold optimization occurred.

The TEST split is used only for post-freeze descriptive explanation. A seeded sample of 2000 rows was drawn from 5,499 TEST records with required local cases reserved. Seed: `20260806`.

## Explainer and correctness

Environment: XGBoost `3.4.1`. The Python SHAP package could not be installed in this Python 3.13 macOS x86_64 environment because its required `llvmlite` wheel is unavailable and local LLVM is absent. The supported tree-specific equivalent, XGBoost native Tree SHAP (`pred_contribs`), was therefore used; KernelExplainer was not used.

SHAP values are additive contributions in **raw margin/log-odds space**, not percentage-point probability changes. Additivity checked 2000 rows at tolerance 1e-05; maximum residual `0.0000007302`; status **PASS**.

## Global model influence

| rank | feature | mean_abs_shap |
|---|---|---|
| 1 | created_month | 0.339197 |
| 2 | created_hour | 0.180867 |
| 3 | created_day_of_week | 0.078702 |
| 4 | is_weekend | 0.000000 |

The fitted model relies most on `created_month` among the four available temporal features. This describes model reliance, not a real-world cause of missed targets.

## Feature direction

Exact feature-value groups with the largest absolute mean contribution are:

| feature | feature_value | mean_shap | sample_count |
|---|---|---|---|
| created_hour | 23.000000 | 0.432804 | 11 |
| created_hour | 22.000000 | 0.364974 | 19 |
| created_month | 10.000000 | -0.362941 | 545 |
| created_month | 11.000000 | -0.360856 | 444 |
| created_day_of_week | 6.000000 | 0.154370 | 132 |
| created_day_of_week | 2.000000 | -0.108291 | 340 |
| is_weekend | 0.000000 | 0.000000 | 1744 |
| is_weekend | 1.000000 | 0.000000 | 256 |

Positive SHAP values raise the raw model margin; negative values lower it. Sparse value groups must be interpreted cautiously.

## Interactions

Native Tree SHAP interaction analysis completed. Strongest pairs:

| rank | feature_1 | feature_2 | mean_abs_interaction |
|---|---|---|---|
| 1 | created_hour | created_month | 0.042422 |
| 2 | created_day_of_week | created_month | 0.022585 |
| 3 | created_hour | created_day_of_week | 0.015579 |
| 4 | created_day_of_week | is_weekend | 0.000000 |
| 5 | created_hour | is_weekend | 0.000000 |
| 6 | created_month | is_weekend | 0.000000 |

## Representative local cases

Cases use the wrapper's existing `0.5` cutoff only as a descriptive reference; it is not an operational threshold.

| case_role | record_identifier | raw_model_score | feature | shap_contribution_raw_margin |
|---|---|---|---|---|
| false_negative | 66416359 | 0.266917 | created_month | -0.418990 |
| false_positive | 67312708 | 0.609985 | created_hour | 0.240006 |
| highest_risk | 67074101 | 0.609985 | created_hour | 0.240006 |
| lowest_risk | 66416359 | 0.266917 | created_month | -0.418990 |
| true_negative | 66409248 | 0.266917 | created_month | -0.418990 |
| true_positive | 67074101 | 0.609985 | created_hour | 0.240006 |

Waterfalls show the base raw margin, feature contributions, and final raw model margin for highest/lowest risk, TP, TN, FP, and FN roles. Duplicate record identifiers across roles are allowed when one deterministic record satisfies multiple roles.

## Correct versus incorrect patterns

| feature | mean_abs_shap_correct | mean_abs_shap_incorrect | difference_incorrect_minus_correct |
|---|---|---|---|
| created_hour | 0.183390 | 0.176778 | -0.006612 |
| created_day_of_week | 0.077658 | 0.080395 | 0.002737 |
| created_month | 0.342267 | 0.334220 | -0.008047 |
| is_weekend | 0.000000 | 0.000000 | 0.000000 |

This is bounded descriptive evidence for the reference cutoff, not Phase 6 subgroup error analysis.

## Stability

Two seeded, disjoint 1,000-row halves produced Spearman rank correlation `1.000000`. Top feature match: `True`. Top-two set match: `True`. Status: **REASONABLY_STABLE**.

## What SHAP can and cannot say

SHAP can describe how this fitted tree model distributes its raw output among the four supplied features, globally and for selected records. It cannot prove causality, establish model accuracy, validate data-generating mechanisms, or explain the separate isotonic mapping. The model's known weak TEST discrimination and failed calibration generalization materially limit how strongly these patterns should be interpreted.

Phase 6 should investigate the false-positive and false-negative patterns surfaced here, but Phase 5 does not perform that broader error analysis. Phase 5 is formally complete subject to its recorded test gate.
