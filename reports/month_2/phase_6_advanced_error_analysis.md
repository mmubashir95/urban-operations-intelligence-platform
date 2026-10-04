# Phase 6 — advanced error analysis

## Frozen boundary and decision rule

This analysis reuses frozen Phase 3 `shallow` on TEST, split `20260806T135114Z_9d945cb2da0eecfc`, feature fingerprint `d6620d99301f80884487140bde487479f58fe72c1e209328524fa27e8b5a2271`, with no retraining, tuning, feature, preprocessing, split, calibration-selection, or threshold-policy change. Phase 4 remains `ISOTONIC`.

The **primary operational view** flags exactly the highest-ranked 10% by raw XGBoost probability (`ceil(n × 0.10)`, stable original-row ties). The **secondary descriptive view** uses raw score `>= 0.5` solely to connect with Phase 5; it is not an operational threshold.

## Overall errors

| sample_count | TP | TN | FP | FN | precision | recall | specificity | false_positive_rate | false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|
| 5499 | 222 | 3244 | 328 | 1705 | 0.4036 | 0.1152 | 0.9082 | 0.0918 | 0.8848 |

The model failed to surface **1705** of 1927 actual misses (88.5%). 181 false negatives are in the bottom score decile. The selected workload contains 328 unnecessary flags; 72 are in the highest selected-rank quartile. The separate descriptive confusion summary is preserved in `phase_6_confusion_summary.csv` and is not treated as policy.

## Scope and subgroup findings

Agency and complaint-category analysis are not applicable: the frozen scope is one agency (`DSNY`) and one complaint type (`Graffiti`). No meaningless agency or category CSV was created. Borough, location type, and deterministic top-25 ZIP-plus-remainder analyses use retained post-hoc metadata only. Supported borough FNR ranges from 0.830 to 0.946. Low-support groups are labeled and excluded from ranking claims.

## Time, scores, and drift

Monthly TEST results are in `phase_6_errors_by_month.csv`. Operational selections by month are 2025-09=0, 2025-10=0, 2025-11=0, 2025-12=550, demonstrating whether the global ranking is temporally concentrated. TRAIN/VALIDATION/TEST context shows target prevalence of train=0.460, validation=0.426, test=0.350; TRAIN is explicitly in-sample. Score deciles collapse duplicate boundaries, exposing where FP/FN records occur without inventing thresholds. These are diagnostic drift signals, not a monitoring system.

## Phase 5 linkage and feature-signal evidence

Phase 5 is reused rather than rerun. Its leading SHAP feature is `created_month` (mean absolute raw-margin SHAP 0.3392); `is_weekend` had zero reported influence. Phase 5 correctness used descriptive 0.5, so its correct/incorrect table is not presented as Top-10% evidence. Weak TEST discrimination, high primary FNR, bottom-decile FNs, overlapping scores, and reliance on four temporal features support the bounded conclusion that the current temporal-only feature set **may provide insufficient predictive signal**.

## Failure modes and operational importance

| failure_mode | sample_count | severity | evidence |
|---|---|---|---|
| Temporal selection concentration | 1457 | HIGH | The global Top-10% policy selects no records in one or more TEST months while selecting records in another. |
| Very-low-score false negatives | 181 | HIGH | Actual misses occur in the bottom raw-score decile. |
| High-ranked false positives | 72 | MEDIUM | On-time cases occur in the highest selected-rank quartile. |
| Geographic subgroup weakness | 40 | MEDIUM | Supported geography groups differ in false-negative rate by at least 0.10. |

Potential explanations are hypotheses for a later governed improvement checkpoint. Geography, time, and SHAP associations do not establish causes. Phase 6 did not implement a fix or alter the frozen system.

## Conclusion

Failure is assessed across score severity, supported geography, and time rather than claimed causal. Evidence should feed a later feature/model checkpoint before NLP. The frozen model was left unchanged and Phase 6 is complete subject to the recorded test gate.
