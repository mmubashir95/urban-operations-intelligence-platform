# Phase 6.11 — Month 1 Freeze

This report closes Phases 6.5 through 6.11: README update, verification
hardening, artifact regeneration, focused and full test runs, the completion
gate, and the Month 1 freeze.

## 1. Objective

Make every governed Month 1 artifact and completion check agree with the
validated frozen Logistic Regression threshold decision, then freeze Month 1.
No threshold was reselected, no test result was used for selection, and the
model was not changed.

## 2. Authoritative Threshold Contract

```text
configs/models/month1_logistic_regression_threshold_decision.json
selected_threshold = 0.49
selected_on_split  = validation
policy             = max_flagged_rate (predicted_positive_rate <= 0.30, then maximize recall)
frozen             = true
SHA-1              = beff4df155a52314c0a3a53022a388bf822454fd (unchanged)
```

```text
validation selection -> frozen JSON -> load_frozen_threshold_decision
  -> validate_frozen_threshold_decision -> decision.selected_threshold
  -> governed predictions -> persistence -> final/subgroup metrics
  -> Month 1 report -> README -> month1_verification -> completion gate
```

## 3. Governed Artifacts Reconciled

| Phase | Change |
| --- | --- |
| 6.5 README | "frozen threshold `0.5`" became "frozen threshold `0.49`, selected on validation under the approved workload-limited policy and recorded in the frozen decision JSON". |
| 6.6 Gate | `month1_verification.py` no longer has `EXPECTED_THRESHOLD`. It loads and validates the frozen JSON and checks every governed artifact against `decision.selected_threshold`. |
| 6.7 Regeneration | `make baseline-resolution-risk`: both copies of `baseline_results.md` gained the default/reference-cutoff label on Validation Comparison. The Month 1 report's generalization line was reworded so it is not read as a threshold claim. No other output changed. |

### Completion gate checks (`verify_month1_artifacts`)

1. The frozen decision is loaded and validated. An invalid or tampered decision
   fails with `Frozen threshold decision is invalid`.
2. The selected joblib `selected_threshold` equals the decision.
3. The selected metadata JSON `selected_threshold` equals the decision.
4. The Phase 4.7 frozen test evidence records the decision threshold and
   validation selection.
5. The final test TP/FP/TN/FN, precision, recall, and F1 equal the Phase 4.7
   frozen evaluation.
6. For every subgroup column that covers a whole split, summed FP/FN equal the
   decision's validation FP/FN and the final test FP/FN.
7. Every "selected/frozen threshold" claim in the Month 1 report, baseline
   report, and README equals the decision value.
8. The Month 1 report states `frozen threshold {decision}` and the CSV metric
   values (unchanged existing checks).

The gate only checks consistency. It never evaluates or selects thresholds.

## 4. Descriptive 0.50 Paths Preserved

- `DEFAULT_CLASSIFICATION_THRESHOLD = 0.5` and the Phase 4.1 default/reference report.
- `MANUAL_CLASSIFICATION_THRESHOLDS` including `0.50` (Phase 4.2).
- The `0.50` sweep row and the `0.40-0.50` transition region (Phases 4.3 and 4.4).
- The `Precision >= 0.50` policy constraint (`THRESHOLD_POLICY_MIN_PRECISION`).
- The `LogisticRegressionBaseline.predict(threshold=0.5)` generic default.
- Validation Comparison LR precision/recall/F1 at the default cutoff, now
  labelled descriptive.
- The Majority Class `0.5` in `_split_predictions`, which is not the governed
  Logistic Regression path.
- Plot colours, the "No skill (ROC-AUC=0.5000)" label, and calibration bin bounds.

The gate's claim pattern ignores all of these because none is labelled as the
selected or frozen threshold.

## 5. Final Governed Metrics

Generated `reports/tables/baseline_test_results.csv`, which equals
`logistic_regression_test_frozen_threshold.csv`:

| TP | FP | TN | FN | Precision | Recall | F1 | Flagged | Flagged rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 106 | 129 | 3443 | 1821 | 0.4511 | 0.0550 | 0.0981 | 235 / 5499 | 0.0427 |

Threshold-independent test metrics: ROC-AUC 0.5192, PR-AUC 0.3898, Brier
0.2298, Recall@10% 0.1318, Recall@20% 0.2460.

Subgroups: in `baseline_subgroup_results.csv`, the full-coverage columns
(`created_day_of_week`, `created_month`, `is_weekend`) sum to validation FP/FN
888/1980, which matches the frozen decision, and to test FP/FN 129/1821, which
matches the final test row.

## 6. Test Results

| Check | Result |
| --- | --- |
| `tests/unit/models/test_month1_verification.py` | 17 passed |
| Focused frozen/persistence/report integration tests | 11 passed |
| `tests/unit/models/` | 265 passed |
| `tests/integration/test_baseline_workflow.py` | 33 passed |
| `tests/integration/` | 67 passed |
| Full `pytest` | 896 passed, 0 failed, 0 skipped, 0 xfailed |
| `git diff --check` | Passed |

No linter or type checker is configured in the repository.

## 7. Completion-Gate Result

`make verify-month1`: gate test modules passed, the baseline workflow
completed with `threshold=0.4900`, and the verifier printed:

```text
MONTH 1 COMPLETE
MONTH 2 READY
```

## 8. Artifact Consistency Matrix

| Artifact | Threshold / source | Role | Consistent | Notes |
| --- | --- | --- | --- | --- |
| Frozen decision JSON | 0.49 / validation policy selection | Authority | Yes | Unchanged. |
| Selected joblib | 0.49 / `decision.selected_threshold` | Governed | Yes | Gate check 2. |
| Selected metadata JSON | 0.49 / `decision.selected_threshold` | Governed | Yes | Gate check 3. |
| Phase 4.7 frozen test table | 0.49 / validated decision | Governed evidence | Yes | Gate check 4. |
| Final test classification metrics | 0.49 predictions from `_split_predictions` | Governed | Yes | Gate check 5; workflow guard. |
| Subgroup classification metrics | 0.49 validation and test predictions | Governed | Yes | Gate check 6. |
| `baseline_results.md` | "Selected threshold: 0.4900" | Governed | Yes | Gate check 7; default comparison labelled. |
| `month_1_baseline_report.md` | "frozen threshold 0.4900" | Governed | Yes | Gate checks 7-8. |
| README | "frozen threshold `0.49`" | Governed | Yes | Gate check 7. |
| `month1_verification` expectations | loaded `decision.selected_threshold` | Gate | Yes | No threshold constant. |
| Phase 4.1/4.2/4.3 outputs | 0.50 | Descriptive | N/A | Preserved. |

## 9. Model Fingerprint Verification

SHA-256 of the Logistic Regression `coef_` and `intercept_` bytes is
`8135d6eb396b54f7c9f5fd31825553a2cee0adf07dd83797b2bdb714e57fbbdc`. It is
identical at the pre-Phase-6 commit `d141245` and now. The pickled model object
is byte-identical, and feature names and the Phase 9 contract are equal. Only
`selected_threshold` changed, from 0.5 to 0.49.

Regeneration in Phase 6.7 was compared against a snapshot of all generated
outputs. Only the two labelled report text changes differed. Every table,
figure, Phase 4/5 report, and model artifact was byte-identical.

## 10. Month 1 Freeze Statement

Month 1 is frozen:

- The governed threshold is `0.49`, selected on validation under the approved
  workload-limited policy and frozen in Phase 4.6.
- No threshold reselection occurred in Phase 6. The test split was used only to
  evaluate the frozen threshold.
- Training data, split, features, preprocessing, hyperparameters, and model
  coefficients are unchanged.
- Top-K ranking and capacities are unchanged and independent of the
  classification threshold.
- The completion gate derives its threshold from the frozen decision and passes.

## 11. Month 2 Readiness Statement

MONTH 2 READY. Month 2 work must preserve the frozen Month 1 evaluation
contract and compare against this baseline without editing the frozen decision
artifact.

## 12. Remaining Non-Blocking Notes

- Generated reports and tables under `reports/` are gitignored and reproduced
  by `make baseline-resolution-risk`. Only the Phase 6 reports and model
  artifacts are versioned.
- `reports/phase_6_4_month_1_report_reconciliation.md` quotes the historical
  "frozen threshold 0.5000" text as audit evidence. This is not a live claim,
  and the gate does not scan it.
- The persisted `LogisticRegressionBaseline.predict()` keeps its generic 0.50
  default. Consumers of the joblib must apply `payload["selected_threshold"]`.

## Verdict

**MONTH 1 FROZEN — MONTH 2 READY**
