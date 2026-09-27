# Phase 6.4 — Month 1 Final Report Reconciliation

## 1. Objective

Make `reports/month_1_baseline_report.md` state and use the governed frozen
Logistic Regression threshold. The report is regenerated from governed metrics
computed at the frozen threshold; its text was not edited by hand.

## 2. Repository Sources Inspected

- `docs/ai-workflow/CODEX_IMPLEMENTATION.md`
- `configs/models/month1_logistic_regression_threshold_decision.json`
- `src/urban_ops/models/baseline_workflow.py`: `_fit_and_evaluate_validation`,
  `_split_predictions`, `build_subgroup_analysis`, `run_baseline_workflow`,
  `_write_tables`, `_write_reports`, `_subgroup_summary`
- `src/urban_ops/models/baselines.py`: `predict_from_scores`
- `src/urban_ops/models/evaluation.py`: `evaluate_threshold`,
  `classify_scores_at_threshold`, `ThresholdMetrics`
- `src/urban_ops/models/month1_verification.py`
- `reports/month_1_baseline_report.md`, `reports/baseline_results.md`, and
  `reports/tables/*.csv`
- `tests/integration/test_baseline_workflow.py`

## 3. Previous Month 1 Report Inconsistency

The report labelled `0.5000` as the frozen threshold and every governed
threshold-dependent value came from `0.50` classifications:

| Section | Previous content | Class |
| --- | --- | --- |
| 1 Executive Summary | "the frozen threshold 0.5000" | Governed |
| 14 Final Test Results | precision 0.4489, recall 0.0410, F1 0.0751 (at 0.50) | Governed |
| 15 Threshold Interpretation | "At the frozen threshold 0.5000"; "The frozen 0.5 threshold" | Governed |
| 16 Operational Top-K | "binary classifier at threshold 0.5" | Governed wording |
| 18 Generalization | validation recall 0.2634 (default 0.50) against test recall | Governed, mismatched cutoffs |
| 19 Subgroup Analysis | subgroup precision/recall/FP/FN at 0.50; "frozen 0.5 threshold" | Governed |
| 20 Limitations, 21 Conclusion | "threshold 0.5" | Governed wording |
| 12 Validation Results | Logistic Regression precision/recall/F1 at 0.50, unlabelled | Descriptive, unlabelled |
| 12 Majority ROC-AUC `0.5000`; 17 calibration bin bounds | numeric values | Unrelated |

## 4. Authoritative Threshold Source

```text
configs/models/month1_logistic_regression_threshold_decision.json
selected_threshold = 0.49
selected_on_split = validation
frozen = true
```

The file was not modified. Its SHA-1 before and after regeneration is
`beff4df155a52314c0a3a53022a388bf822454fd`.

## 5. Report Generation Flow Before

```text
_fit_and_evaluate_validation
    -> logistic.predict(X_validation)             (implicit 0.50)
    -> validation subgroup rows
_split_predictions (Logistic Regression)
    -> model.predict(matrix), threshold 0.5       (hard-coded)
    -> test_results, test subgroup rows, selected_threshold
    -> _write_reports -> Month 1 report text ("frozen 0.5")
```

Phase 4.7 separately evaluated test at the frozen `0.49`, so the report showed a
`0.50` final result alongside a `0.49` frozen-threshold result.

## 6. Report Generation Flow After

```text
frozen threshold JSON
    -> load_frozen_threshold_decision
    -> validate_frozen_threshold_decision
    -> reconciled with the validation policy candidate
    -> _split_predictions(..., frozen_threshold_decision=decision)
         Logistic Regression: predict_from_scores(scores, decision.selected_threshold)
    -> validation + test predictions
    -> test_results  (verified equal to the Phase 4.7 frozen evaluation)
    -> validation + test subgroup rows
    -> _write_reports (threshold text from decision.selected_threshold)
    -> reports/month_1_baseline_report.md
```

Changes in `baseline_workflow.py`:

- `_split_predictions` takes the validated decision and classifies Logistic
  Regression scores at `decision.selected_threshold` through the existing
  `predict_from_scores` (`score >= threshold`) helper.
- The hidden `0.50` validation subgroup computation was removed from
  `_fit_and_evaluate_validation`. Validation subgroups are now built in
  `run_baseline_workflow` from the same governed validation predictions as the
  test subgroups, after the decision is validated.
- `_verify_final_test_matches_frozen_threshold_evaluation` requires the final
  test row's confusion counts, precision, recall, and F1 to equal the Phase 4.7
  frozen-threshold evaluation computed independently from the same scores.
- `_write_reports` reads the governed threshold from the decision, refuses to
  render if the selected-model threshold differs, and replaces every governed
  `0.5` phrase with the decision value. No `0.49` literal was introduced.

## 7. Final Governed Threshold

`0.4900`, selected on validation under the workload-limited policy
(`predicted_positive_rate <= 0.30`, then maximize recall).

## 8. Final Governed Test Metrics

Generated `reports/tables/baseline_test_results.csv`, which equals
`reports/tables/logistic_regression_test_frozen_threshold.csv`:

| Metric | Value |
| --- | ---: |
| True positives | 106 |
| False positives | 129 |
| True negatives | 3443 |
| False negatives | 1821 |
| Precision | 0.4511 |
| Recall | 0.0550 |
| F1 | 0.0981 |
| Flagged count | 235 of 5499 |
| Flagged rate | 0.0427 |

Threshold-independent values are unchanged: ROC-AUC 0.5192, PR-AUC 0.3898,
Brier 0.2298, and Precision/Recall@5/10/20%.

## 9. Subgroup Metric Reconciliation

`baseline_subgroup_results.csv` (both `reports/tables/` and
`reports/12_baseline_modelling/tables/`) was regenerated. Validation and test
rows now use the frozen-threshold predictions. Group definitions and the
minimum-count rule are unchanged. The report's subgroup note is now
data-driven: 14 sufficiently populated test subgroups have zero recall at
`0.4900`.

## 10. Preserved Descriptive 0.50 References

- Section 12 Validation Results keeps each baseline's default decision
  mechanism for model comparison. It is now labelled as the descriptive
  default/reference cutoff `0.50`, not the frozen threshold. Baseline selection
  uses threshold-independent ranking metrics.
- Phase 4.1, 4.2, and 4.3 reports and tables (default point, manual grid, sweep
  row) and the `Precision >= 0.50` policy constraint are unchanged.
- Majority ROC-AUC `0.5000` and calibration bin bounds `0.5000` are unrelated
  numeric values.

## 11. Tests

New regressions in `tests/integration/test_baseline_workflow.py`:

- `test_selected_logistic_predictions_use_frozen_threshold` checks that
  validation and test predictions equal `score >= decision.selected_threshold`.
- `test_final_test_guard_rejects_metrics_from_another_threshold` checks that
  metrics computed at `0.50` are rejected as the final frozen result.
- `test_repository_month_1_report_follows_frozen_threshold_decision` checks that
  the generated final test CSV equals the Phase 4.7 frozen CSV, and that the
  report states the decision threshold and restates those metrics. It also
  checks that the report labels the default cutoff as descriptive and never
  labels `0.5` as frozen or selected.

| Check | Result |
| --- | --- |
| Focused Phase 6.4 tests | 3 passed |
| `pytest tests/unit/models/` | 250 passed |
| `pytest tests/integration/test_baseline_workflow.py` | 33 passed |
| `pytest tests/integration/` | 67 passed |
| Full `pytest` | 881 passed |
| `git diff --check` | Passed |
| `make verify-month1` | Gate tests 212 passed; verifier failed: `Selected threshold mismatch: expected 0.5, observed 0.49.` |

The verifier failure is the known legacy `EXPECTED_THRESHOLD = 0.5` gate from
Phase 6.3 and is not caused by this phase.

## 12. Files Changed

Tracked:

- `src/urban_ops/models/baseline_workflow.py`
- `tests/integration/test_baseline_workflow.py`
- `reports/phase_6_4_month_1_report_reconciliation.md`

Regenerated by `make baseline-resolution-risk`. These are gitignored, so they
are local workflow outputs and are not versioned:

- `reports/month_1_baseline_report.md`
- `reports/baseline_results.md` and `reports/12_baseline_modelling/baseline_results.md`
  (selected threshold and final test sections now at `0.49`)
- `reports/tables/baseline_test_results.csv` and the `12_baseline_modelling` copy
- `reports/tables/baseline_subgroup_results.csv` and the `12_baseline_modelling` copy
- the other phase reports and tables the workflow rewrites on every run. Their
  inputs are threshold-independent or already frozen, so they are expected to
  be unchanged. Because they are untracked, git cannot confirm that.

`models/baselines/selected_month_1_baseline.joblib` and its metadata were
rewritten with identical content.

## 13. Remaining Cleanup

- `month1_verification.py:EXPECTED_THRESHOLD = 0.5` and its report assertion
  `threshold 0.5000`.
- Legacy gate fixtures in `tests/unit/models/test_month1_verification.py`.
- `README.md` "frozen threshold `0.5`" status line.
- `reports/baseline_results.md` Validation Comparison still lacks the
  default/reference-cutoff label that the Month 1 report now has.
- The Month 1 report still says `MONTH 1 COMPLETE` while `make verify-month1`
  fails. That claim holds only once the gate is migrated.

## 14. Phase 6.4 Verdict

**PHASE 6.4 COMPLETE**

The Month 1 report identifies `0.49` as the frozen governed threshold. Its final
and subgroup metrics are computed at `0.49` from the validated decision.
Descriptive `0.50` references are labelled. The frozen JSON, model
coefficients, features, preprocessing, threshold selection, and Top-K are
unchanged.
