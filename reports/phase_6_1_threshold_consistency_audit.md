# Phase 6.1 — Logistic Regression Threshold Consistency Audit

## 1. Objective

This audit explains why Month 1 artifacts contain both `0.49` and `0.50` as
Logistic Regression classification thresholds. It identifies the authoritative
decision, traces both values through runtime and reporting paths, checks
test-set isolation, and defines the bounded Phase 6.2 cleanup scope.

This phase does not retrain the model, rerun threshold selection, compare
thresholds to choose a winner on test, alter predictions, modify the frozen
decision, or change Top-K behavior.

## 2. Repository Sources Inspected

The audit inspected:

- `docs/ai-workflow/CODEX_IMPLEMENTATION.md`;
- `configs/models/month1_logistic_regression_threshold_decision.json`;
- `src/urban_ops/models/evaluation.py`;
- `src/urban_ops/models/baselines.py`;
- `src/urban_ops/models/baseline_workflow.py`;
- `src/urban_ops/models/month1_verification.py`;
- `models/baselines/selected_month_1_baseline.joblib`;
- `models/baselines/selected_month_1_baseline_metadata.json`;
- Phase 4, Phase 5, baseline, and Month 1 generated reports and CSV tables;
- `README.md`;
- threshold-related unit and integration tests; and
- Git log, pickaxe history, and blame for the threshold-bearing files.

## 3. Authoritative Frozen Threshold

The authoritative classification threshold is **`0.49`**. The machine-readable
authority is
`configs/models/month1_logistic_regression_threshold_decision.json`, which
records:

| Field | Recorded value |
| --- | --- |
| `selected_threshold` | `0.49` |
| `selected_on_split` | `validation` |
| `policy_name` | `max_flagged_rate` |
| `constraint_name` | `predicted_positive_rate` |
| `constraint_value` | `0.30` |
| `secondary_objective` | `maximize_recall` |
| `frozen` | `true` |

The artifact is internally consistent. Its validation evidence reports 902 true
positives, 888 false positives, 2,992 true negatives, 1,980 false negatives,
precision 0.5039, recall 0.3130, F1 0.3861, and a flagged rate of 0.2647.

`evaluation.py` independently fixes the approved contract through
`FROZEN_THRESHOLD_SELECTED_THRESHOLD = 0.49`. Its
`validate_frozen_threshold_decision` function rejects a decision with threshold
`0.50`, a non-validation selection split, an unfrozen state, or different
policy provenance. The frozen JSON therefore is not merely documentation; it is
the validated Phase 4.6 authority.

## 4. Origin of `0.49`

The `0.49` value came from validation-only threshold analysis:

1. Logistic Regression was fitted on training data.
2. Validation probabilities were evaluated on the fixed `0.05` through `0.95`
   grid in `0.01` increments.
3. Phase 4.5 evaluated candidate policies from that validation sweep.
4. The approved workload-limited policy retained thresholds with
   `predicted_positive_rate <= 0.30`, then maximized validation recall.
5. That policy selected the existing validation sweep row at `0.49`.
6. Phase 4.6 copied the selected row and its provenance into the frozen JSON.
7. Phase 4.7 loaded and validated the JSON and applied `0.49` unchanged to test.

The same validation row also satisfied the separate descriptive
`Precision >= 0.50` candidate policy. The number `0.50` in that policy name is a
minimum-precision constraint, not a selected classification cutoff.

## 5. Origin of `0.50`

The `0.50` value predates the governed Phase 4.6 decision. Commit `ccbc108`
(`Implement Month 1 baseline modelling closure`, 2026-09-10) introduced:

- `LogisticRegressionBaseline.predict(..., threshold=0.5)`;
- the selected-model workflow path returning `0.5` for Logistic Regression;
- selected-model joblib and JSON metadata containing `0.5`; and
- baseline/final test reporting based on those default predictions.

Commit `5df820e` (`Finalize Month 1 readiness verification`, 2026-09-10) then
made the completion gate explicitly expect `0.5` and documented it in the
README and Month 1 closeout report.

Commit `106927a` (`Freeze workload-limited threshold`, 2026-09-16) added the
validation-selected `0.49` decision and its authoritative JSON. Commit `93a750f`
(`Evaluate frozen threshold on final test`, 2026-09-16) added a separate test
evaluation at `0.49` and validation of its provenance. Those commits did not
replace `_split_predictions`, selected-model persistence, the Month 1 summary,
or the verification gate. The present mismatch is therefore an incomplete
integration of the later frozen decision into the earlier closeout path.

## 6. Threshold Occurrence Inventory

### Consistency matrix

| Artifact or path | Threshold | Role | Runtime / documentation | Source of value | Expected authority | Status | Notes |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `configs/models/month1_logistic_regression_threshold_decision.json` | 0.49 | Governed Phase 4.6 decision | Runtime configuration | Validation workload policy | Authoritative | Consistent | Frozen, validation-selected, internally coherent. |
| `evaluation.py:FROZEN_THRESHOLD_SELECTED_THRESHOLD` | 0.49 | Frozen-decision contract validation | Runtime | Approved Phase 4.6 contract | Frozen JSON/approved contract | Consistent | Explicitly rejects 0.50 as a frozen artifact value. |
| `baseline_workflow.py:load_frozen_threshold_decision` | 0.49 | Loads Phase 4.6 decision | Runtime | Frozen JSON | Frozen JSON | Consistent | Loaded decision is also reconciled to the recomputed validation candidate. |
| `baseline_workflow.py:run_baseline_workflow` Phase 4.7 branch | 0.49 | Final frozen-threshold test evaluation | Runtime | Loaded frozen decision | Frozen JSON | Consistent | Calls `evaluate_threshold(..., threshold=decision.selected_threshold)`. |
| `reports/tables/logistic_regression_test_frozen_threshold.csv` | 0.49 | Final Phase 4.7 test operating point | Generated data | Frozen decision | Frozen JSON | Consistent | 235 flagged rows, 106 TP, 129 FP. |
| Phase 4.6/4.7 reports | 0.49 | Decision and final-test evidence | Documentation | Frozen decision | Frozen JSON | Consistent | State validation selection and no retuning. |
| Phase 5 reports | 0.49 reference only | States Top-K independence | Documentation | Frozen decision | Frozen JSON | Consistent | `0.49` does not affect Top-K membership. |
| `evaluation.py:DEFAULT_CLASSIFICATION_THRESHOLD` | 0.50 | Phase 4.1 default/reference cutoff | Runtime helper default | Conventional initial cutoff | Intentionally descriptive | Semantically distinct | Valid for Phase 4.1, not the frozen operational decision. |
| Phase 4.1/4.2/4.3 reports and tables | 0.50 | Default/manual/sweep comparison point | Generated evidence | Fixed comparison grid | Intentionally descriptive | Semantically distinct | Reports explicitly say 0.50 is not selected. |
| `evaluation.py:THRESHOLD_POLICY_MIN_PRECISION` | 0.50 | Minimum precision constraint | Runtime policy parameter | Candidate policy definition | Not a classification cutoff | Semantically distinct | This constraint happens to select cutoff 0.49. |
| `baselines.py:LogisticRegressionBaseline.predict` | 0.50 default | Default binary prediction | Runtime | Original baseline implementation | Should defer to frozen decision in governed use | Inconsistent when used as governed Month 1 output | The generic default can remain useful, but governed callers currently invoke it implicitly. |
| `baseline_workflow.py:_fit_and_evaluate_validation` | 0.50 implicitly | Baseline validation hard predictions | Runtime | `model.predict()` default | Descriptive Phase 1/4.1 | Mixed but traceable | Appropriate as initial/default evidence; it must not be labelled frozen. |
| `baseline_workflow.py:_split_predictions` Logistic branch | 0.50 | Selected validation/test predictions and returned threshold | Runtime | Hard-coded legacy branch | Frozen decision | Inconsistent | Produces the selected-model and Month 1 closeout path at 0.50. |
| `baseline_workflow.py:_persist_selected_model` outputs | 0.50 | Persisted selected-model threshold | Runtime artifact / metadata | `_split_predictions` result | Frozen decision | Inconsistent | Both joblib payload and JSON metadata contain 0.5. |
| `reports/tables/baseline_test_results.csv` | 0.50 implicitly | Main final test classification metrics | Generated data | `_split_predictions` predictions | Frozen decision for final operating point | Inconsistent | 176 flagged rows, 79 TP, 97 FP. |
| `reports/tables/baseline_subgroup_results.csv` | 0.50 implicitly | Selected-model subgroup classifications | Generated data | `_split_predictions` predictions | Frozen decision for governed classifications | Inconsistent | Test/validation subgroup precision and recall use 0.50 predictions. |
| `reports/baseline_results.md` selected threshold | 0.50 | Selected artifact summary | Documentation | Legacy selected-model path | Frozen decision | Inconsistent | The same report separately and correctly documents Phase 4.6/4.7 at 0.49. |
| `reports/month_1_baseline_report.md` | 0.50 | Final Month 1 classification summary | Documentation | Legacy selected-model path | Frozen decision | Inconsistent | Labels 0.50 as frozen and reports 0.50 test/subgroup metrics. |
| `README.md` | 0.50 | Project status | Documentation | September 10 closeout | Frozen decision | Inconsistent | Predates the September 16 freeze. |
| `month1_verification.py:EXPECTED_THRESHOLD` | 0.50 | Completion-gate expectation | Runtime verification | September 10 closeout | Frozen decision | Inconsistent | Does not read or cross-check the frozen JSON. |
| `test_month1_verification.py` | 0.50 | Gate fixture and expectation | Test | Gate constant | Frozen decision | Inconsistent protection | Locks in the legacy closeout value. |

Literal `0.50` values used as metric examples, class prevalence, plot colors,
ROC no-skill references, Top-K test data, or unrelated scope/EDA thresholds are
not Logistic Regression classification-threshold inconsistencies.

Repository-generated reports are duplicated under root `reports/` and
`reports/12_baseline_modelling/`. The corresponding duplicate Phase 4 and
baseline artifacts have the same meanings and consistency status.

## 7. Runtime Threshold Behavior

The workflow currently produces two binary classifications from the same
Logistic Regression probability scores.

### Frozen Phase 4.7 path: `0.49`

`run_baseline_workflow`:

1. computes the workload-policy candidate from validation;
2. loads the frozen JSON;
3. verifies the loaded decision matches the validation candidate;
4. calculates test scores through `logistic_model.predict_score(...)`; and
5. calls `evaluate_threshold(..., threshold=0.49)`.

This path writes `logistic_regression_test_frozen_threshold.csv` and the Phase
4.7 report. It yields 235 positive predictions on test: 106 TP and 129 FP.

### Legacy selected-model path: `0.50`

Later in the same workflow, `_split_predictions` handles Logistic Regression by
calling `model.predict(matrix)` and returning literal `0.5`. The model method
defaults to `score >= 0.5`. This path:

- creates `baseline_test_results.csv`;
- feeds validation/test subgroup analysis;
- supplies `selected_threshold` to selected-model persistence;
- supplies `selected_threshold` and the legacy test row to the Month 1 report;
  and
- returns `BaselineWorkflowResult.selected_threshold = 0.5`.

It yields 176 positive predictions on test: 79 TP and 97 FP.

Therefore the same score in the half-open interval `[0.49, 0.50)` is positive
in the frozen Phase 4.7 path and negative in the selected-model path. On the
current test artifact, the two paths differ for 59 complaints: the 0.49 path
adds 27 true positives and 32 false positives relative to the 0.50 path.

The selected joblib payload contains `selected_threshold: 0.5`, but no current
repository inference service consumes that payload for live prediction. Its
only current loader is the Month 1 verification gate. It nevertheless exposes
an incorrect governed threshold contract to future consumers.

## 8. Selected-Model Metadata Behavior

`_persist_selected_model` does not load the frozen decision. It serializes the
`selected_threshold` supplied by the caller into both:

- `models/baselines/selected_month_1_baseline.joblib`; and
- `models/baselines/selected_month_1_baseline_metadata.json`.

The caller supplies the value returned by `_split_predictions`, which is `0.5`
for Logistic Regression. Both persisted artifacts therefore accurately reflect
the legacy runtime branch but conflict with the later authoritative Phase 4.6
decision. The JSON metadata is not read elsewhere in current source code. The
joblib is read by the verification gate.

## 9. Verification-Gate Behavior

`month1_verification.py` defines `EXPECTED_THRESHOLD = 0.5` directly. The gate:

- loads the selected joblib payload and requires its threshold to equal 0.5;
- requires the Month 1 report to contain `threshold 0.5000`;
- checks legacy baseline test metrics against that report; and
- never requires, loads, or validates
  `configs/models/month1_logistic_regression_threshold_decision.json`.

It does not read the selected-model metadata JSON, compare joblib metadata with
the frozen decision, check the Phase 4.7 CSV, or require the Phase 4.6/4.7
reports. Verification passes because the gate, joblib, legacy test-results CSV,
and Month 1 report are mutually consistent around the older `0.50` path. The
gate checks internal consistency within that subset, not consistency with the
authoritative frozen decision.

## 10. Threshold History and Timeline

| Date / commit | Evidence-supported event |
| --- | --- |
| 2026-09-10, `ccbc108` | Initial Month 1 baseline closure uses the Logistic Regression default cutoff `0.5`, persists it, and evaluates the selected model with it. |
| 2026-09-10, `5df820e` | Month 1 readiness gate, README, and closeout report are built around `0.5`. |
| 2026-09-16, `106927a` | Validation workload policy freezes `0.49` in a new authoritative JSON and Phase 4.6 report. The older selected-model path remains unchanged. |
| 2026-09-16, `93a750f` | Phase 4.7 loads and validates `0.49`, applies it unchanged to test, and writes separate final evidence. The older 0.50 test/report/persistence path still remains. |
| Later Phase 5 commits | Top-K reporting correctly identifies `0.49` as the frozen classification threshold while remaining rank-based and independent of it. |

## 11. Test-Set Isolation Verification

The authoritative threshold-selection path preserves test isolation:

```text
train-only model fit
    -> validation probability sweep
    -> validation policy evaluation
    -> select 0.49
    -> freeze JSON
    -> load and validate frozen decision
    -> apply 0.49 unchanged to test
```

The code computes the frozen candidate exclusively from validation results and
verifies the loaded artifact against that validation candidate before scoring
test. Tests reject an artifact selected on `test`, reject threshold `0.50` as a
frozen decision, verify Phase 4.7 does not rewrite the decision, and require the
test result to carry `selection_split=validation`.

No code evaluates 0.49 and 0.50 on test and then chooses the better threshold.
The workflow does calculate and report both operating points on test through two
parallel output paths, but neither comparison feeds threshold selection or
changes the frozen artifact. This is a reporting/runtime-contract inconsistency,
not evidence of test-driven threshold selection.

## 12. Confirmed Inconsistencies

1. `_split_predictions` still uses and returns `0.5` for selected Logistic
   Regression predictions after Phase 4.6 froze `0.49`.
2. Selected-model joblib and JSON metadata persist `0.5` rather than the frozen
   decision.
3. The main baseline test table and subgroup table are based on `0.50`, while
   the separate Phase 4.7 table is based on `0.49`.
4. `baseline_results.md` simultaneously reports the authoritative 0.49 decision
   and a selected threshold of 0.5000.
5. `month_1_baseline_report.md` and README call `0.50` frozen even though the
   Phase 4.6 authority is `0.49`.
6. The Month 1 verification gate hard-codes and positively enforces `0.50`
   without consulting the frozen artifact.
7. Verification tests protect the older gate contract rather than detecting the
   cross-artifact mismatch.

## 13. Intentionally Different Values

The following `0.50` uses are valid and should not be mechanically replaced:

- the Phase 4.1 default/reference operating point;
- the 0.50 member of Phase 4.2 manual comparisons and Phase 4.3 sweep;
- the minimum-precision value in the `Precision >= 0.50` candidate policy;
- generic threshold helper defaults where no governed decision is claimed;
- ROC no-skill values, metric/prevalence examples, plot colors, Top-K fixture
  values, and unrelated analytical constants.

The inconsistency arises when the legacy default is presented or persisted as
the selected/frozen Month 1 Logistic Regression threshold, or when it supplies
the governed final classification outputs.

## 14. Phase 6.1 Conclusion

**`0.49` is the authoritative Month 1 Logistic Regression classification
threshold.** It was selected on validation under the approved workload-limited
policy, frozen in the Phase 4.6 JSON, validated in code, and applied unchanged in
the Phase 4.7 test evaluation.

**`0.50` originated as the earlier default baseline cutoff.** It remains
intentional as a descriptive Phase 4.1/manual/sweep reference and as a generic
default, but repository history proves that its use in selected-model
persistence, the final Month 1 summary, final baseline classifications, README,
and completion gate predates the later 0.49 freeze and was not reconciled when
Phase 4.6/4.7 were added.

The model, probability scores, validation selection policy, frozen artifact,
and Top-K behavior are not in dispute. The defect is the coexistence of an
authoritative frozen-decision path and an older closeout/persistence path.

## 15. Recommended Phase 6.2 Cleanup Scope

Phase 6.2 should make the smallest reconciliation necessary:

1. Make the governed Logistic Regression branch in
   `baseline_workflow.py` use the loaded frozen decision instead of returning
   literal `0.5` for selected validation/test classifications.
2. Persist the frozen threshold in the selected joblib and metadata JSON.
3. Regenerate affected baseline test, subgroup, baseline summary, and Month 1
   report artifacts from the existing model and frozen scores without
   retraining or threshold reselection.
4. Update README threshold wording to `0.49`.
5. Make `month1_verification.py` load the frozen JSON and cross-check the frozen
   decision, selected artifact, metadata, reports, and relevant final-test
   evidence instead of hard-coding `0.5`.
6. Update focused unit/integration tests to protect the single-authority
   contract and to preserve the intentionally descriptive 0.50 Phase 4.1 path.
7. Rerun the Month 1 completion gate and full suite.

Phase 6.2 must not rerun threshold selection, use test results to choose a
cutoff, modify the frozen JSON to match legacy artifacts, retrain the model, or
change Top-K behavior.
