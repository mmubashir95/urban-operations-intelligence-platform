# Phase 6.2 — Authoritative Month 1 Threshold Source

## 1. Objective

This phase establishes the single source of truth for governed Month 1
Logistic Regression classification. It does not migrate all consumers, change
predictions, rerun threshold selection, retrain the model, or modify Top-K.

The project rule is:

> For governed Month 1 Logistic Regression classification, the validated frozen
> threshold decision artifact is the single authoritative source.

Governed code must obtain the threshold from the validated decision object, not
from a duplicated literal, generic model default, selected-model metadata, or
report text.

## 2. Repository Sources Inspected

The authority review inspected:

- `docs/ai-workflow/CODEX_IMPLEMENTATION.md`;
- `configs/models/month1_logistic_regression_threshold_decision.json`;
- `src/urban_ops/models/evaluation.py`;
- `src/urban_ops/models/baseline_workflow.py`;
- `src/urban_ops/models/baselines.py`;
- `src/urban_ops/models/month1_verification.py`;
- selected-model joblib and JSON metadata;
- baseline, Month 1, Phase 4, and Phase 5 reports and tables;
- `README.md`;
- threshold-related model unit and integration tests; and
- the Phase 6.1 consistency audit.

## 3. Authoritative Artifact

The single authoritative artifact is:

```text
configs/models/month1_logistic_regression_threshold_decision.json
```

It contains the complete frozen decision rather than only a copied scalar:

| Field | Authoritative value |
| --- | --- |
| `selected_threshold` | `0.49` |
| `selected_on_split` | `validation` |
| `policy_name` | `max_flagged_rate` |
| `constraint_name` | `predicted_positive_rate` |
| `constraint_value` | `0.30` |
| `secondary_objective` | `maximize_recall` |
| `frozen` | `true` |

The artifact also records the validation confusion counts, precision, recall,
F1, selected count, and selected rate that support the decision. It therefore
has enough provenance to distinguish the governed decision from a generic
cutoff or copied metadata value.

## 4. Authoritative Threshold Value

The governed threshold is **`0.49`**.

This value was selected on validation under the approved workload-limited
policy and frozen before final test evaluation. The artifact must not be edited
to agree with older `0.50` consumers. Those consumers must instead migrate to
the frozen decision in the next cleanup phase.

## 5. Why This Source Is Authoritative

The JSON is authoritative because it is the only persisted threshold source
that combines all of the following:

- the selected classification threshold;
- validation selection provenance;
- the approved policy identity;
- the workload constraint and its value;
- the secondary objective;
- the frozen state; and
- the validation evidence supporting the selection.

By contrast:

- `LogisticRegressionBaseline.predict(..., threshold=0.5)` is a generic model
  default;
- `DEFAULT_CLASSIFICATION_THRESHOLD = 0.5` is the descriptive Phase 4.1
  reference cutoff;
- selected-model metadata currently copies the older workflow value;
- reports render upstream values but do not govern them; and
- `month1_verification.py:EXPECTED_THRESHOLD` is a legacy assertion, not a
  selection record with policy provenance.

The validator in `evaluation.py` also names the decision authoritative and
rejects any artifact that does not match the approved frozen contract.

## 6. Provenance Chain

The approved chain is:

```text
train-only Logistic Regression fit
    -> validation probability evaluation
    -> validation threshold sweep
    -> workload-limited policy selection
    -> selected validation row at 0.49
    -> frozen threshold JSON
    -> load_frozen_threshold_decision(...)
    -> validate_frozen_threshold_decision(decision)
    -> decision.selected_threshold
    -> governed Month 1 consumers
```

Test data is downstream of the freeze and does not participate in selection.

## 7. Loader and Validation Contract

The existing approved access pattern is:

```python
decision = validate_frozen_threshold_decision(
    load_frozen_threshold_decision()
)
threshold = decision.selected_threshold
```

Responsibilities remain deliberately separated:

- `load_frozen_threshold_decision` reads the configured JSON and constructs a
  `FrozenThresholdDecision`;
- `validate_frozen_threshold_decision` enforces the governed contract; and
- consumers read `selected_threshold` only after validation.

No second JSON parser, threshold registry, or replacement constant was added.
The loader documentation now makes the governed access pattern explicit.

The validator rejects:

- a threshold other than approved `0.49`;
- `selected_on_split` other than `validation`;
- `frozen` other than `true`;
- an unexpected policy name;
- unexpected constraint name or value;
- an unexpected secondary objective;
- a selected rate above the workload constraint; and
- invalid metric/count fields.

The workflow additionally recomputes the approved candidate from validation and
checks that the loaded artifact matches that validation provenance before final
test evaluation.

## 8. Governed Consumers

The following consumers are governed and should obtain their threshold from the
validated decision:

1. final selected-model Logistic Regression binary classification on validation
   and test;
2. selected-model joblib persistence;
3. selected-model metadata JSON;
4. final baseline classification metrics and confusion counts;
5. final subgroup/error metrics that depend on binary predictions;
6. baseline and Month 1 final report threshold text and threshold-dependent
   metrics;
7. README Month 1 frozen-threshold status; and
8. the Month 1 verification/completion gate.

Phase 4.7 is already compliant. The remaining consumers are mapped below rather
than migrated in this phase.

## 9. Descriptive and Non-Governed Threshold Usages

The following `0.50` usages are intentionally not authorities and may remain:

- Phase 4.1's default/reference cutoff;
- the `0.50` comparison row in Phase 4.2;
- the `0.50` row in the Phase 4.3 sweep;
- the `Precision >= 0.50` policy constraint, where `0.50` is a required metric
  value rather than the classification cutoff selected by that policy;
- generic helper/model defaults when no governed Month 1 decision is claimed;
- test fixtures exercising generic threshold equality and validation;
- Top-K fixtures and metrics, because Top-K membership is rank-based; and
- unrelated metric, prevalence, plotting, EDA, and scope constants.

These values must not be mechanically replaced. A violation occurs only when a
descriptive/default value is used or labelled as the selected/frozen governed
Month 1 threshold.

## 10. Consumer Authority Map

| Consumer | Current threshold source/value | Category | Required authoritative source | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Phase 4.7 final test evaluation | Loaded and validated frozen JSON / 0.49 | Governed | Frozen JSON | Compliant | Uses `decision.selected_threshold`. |
| Frozen-decision report and CSV | Frozen decision / 0.49 | Governed evidence | Frozen JSON | Compliant | Records validation selection and unchanged test use. |
| Phase 5 Top-K | Continuous score ranking; 0.49 mentioned only as separate context | Non-threshold operational evaluation | No classification threshold | Compliant | Must remain independent. |
| Phase 4.1 default evaluation | `DEFAULT_CLASSIFICATION_THRESHOLD` / 0.50 | Descriptive | Fixed descriptive default | Compliant | Explicitly not selected. |
| Phase 4.2 manual comparison | Manual grid including 0.50 | Descriptive | Fixed comparison grid | Compliant | Retains 0.50. |
| Phase 4.3 sweep | Sweep row at 0.50 | Descriptive | Fixed sweep grid | Compliant | Retains 0.50. |
| Minimum-precision policy | Constraint `precision >= 0.50` | Descriptive policy candidate | Policy configuration | Compliant | Constraint selects cutoff 0.49. |
| `LogisticRegressionBaseline.predict` default | Generic default / 0.50 | Descriptive/generic | Caller-supplied decision for governed use | Conditionally compliant | Default may remain; governed callers must pass authority. |
| `_fit_and_evaluate_validation` initial baseline row | Implicit model default / 0.50 | Descriptive baseline evidence | Descriptive default | Compliant if labelled default | Must not be called frozen. |
| `_split_predictions` Logistic branch | Hard-coded 0.50 | Governed | Validated frozen decision | Noncompliant | Next-phase migration required. |
| Selected joblib payload | Workflow-returned 0.50 | Governed persistence | Validated frozen decision | Noncompliant | Currently verified as 0.50. |
| Selected metadata JSON | Workflow-returned 0.50 | Governed persistence | Validated frozen decision | Noncompliant | Must not become another authority. |
| `baseline_test_results.csv` | 0.50 predictions | Governed final metrics | Validated frozen decision | Noncompliant | Differs from Phase 4.7 results. |
| `baseline_subgroup_results.csv` | 0.50 predictions | Governed final subgroup metrics | Validated frozen decision | Noncompliant | Threshold-dependent fields need regeneration. |
| `baseline_results.md` selected threshold/final row | Legacy selected path / 0.50 | Governed reporting | Validated frozen decision | Noncompliant | Same report also contains compliant Phase 4.6/4.7 evidence. |
| `month_1_baseline_report.md` | Legacy selected path / 0.50 | Governed reporting | Validated frozen decision | Noncompliant | Calls 0.50 frozen. |
| README project status | Literal 0.5 | Governed documentation | Frozen decision-derived truth | Noncompliant | Predates Phase 4.6. |
| `month1_verification.py` | Hard-coded `EXPECTED_THRESHOLD = 0.5` | Governed gate | Load and validate frozen JSON | Noncompliant | Does not cross-check authority artifacts. |
| Verification tests | 0.50 fixtures/expectations | Governed regression contract | Frozen JSON contract | Noncompliant | Must change with gate migration. |

Root and `reports/12_baseline_modelling/` compatibility copies have the same
authority status as their corresponding report or table.

## 11. Tests

Phase 6.2 adds a focused integration regression test that loads the real
repository artifact through the existing loader, validates it through the
existing validator, and asserts its threshold, validation split, policy,
constraint, secondary objective, and frozen status.

Existing unit tests already protect invalid-artifact rejection, including
wrong threshold `0.50`, test-selected provenance, unfrozen state, wrong policy,
wrong constraint metadata, and wrong secondary objective. Existing integration
tests protect load/write behavior, validation provenance reconciliation, and
unchanged Phase 4.7 test application.

Descriptive `0.50` tests remain intact.

## 12. Remaining Inconsistent Consumers

This authority phase intentionally does not migrate:

- `_split_predictions`;
- selected-model joblib and metadata persistence;
- baseline final test and subgroup artifacts;
- baseline and Month 1 final reports;
- README project-status wording;
- `month1_verification.py`; or
- legacy completion-gate tests.

They remain visible as noncompliant consumers so the next phase can reconcile
them in one controlled change without altering threshold selection.

## 13. Phase 6.2 Conclusion

`configs/models/month1_logistic_regression_threshold_decision.json` is the
single authoritative source for governed Month 1 Logistic Regression
classification. Its authoritative threshold is `0.49`, selected on validation
under the workload-limited policy and frozen before test evaluation.

Governed consumers must use the existing load-then-validate pattern and obtain
the scalar through `decision.selected_threshold`. They must not infer authority
from selected-model metadata, report text, generic `0.50` defaults, or the
legacy verification constant.

The authority contract is now explicit in implementation documentation,
project documentation, and a regression test against the real repository
artifact. Consumer migration remains intentionally separate.

## 14. Recommended Next Cleanup Phase

The next phase should migrate the concrete noncompliant governed consumers in a
single controlled workflow change:

1. pass the validated frozen decision into selected-model Logistic Regression
   classification instead of using literal/default 0.50;
2. persist `decision.selected_threshold` in joblib and metadata;
3. regenerate threshold-dependent final baseline and subgroup artifacts;
4. regenerate the baseline and Month 1 reports and update README;
5. make the completion gate load and validate the JSON and cross-check governed
   artifacts; and
6. update regression tests while preserving descriptive Phase 4.1/4.2/4.3
   `0.50` behavior and all Top-K behavior.

That phase must not rerun selection, compare thresholds on test, retrain the
model, modify the frozen JSON, or change Top-K.
