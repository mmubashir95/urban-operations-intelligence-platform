# Phase 3 completion gate

**PHASE 3 COMPLETE — SELECTION AND FINAL TEST EVIDENCE FROZEN**

## Frozen foundation

- [x] Frozen feature policy and exact order reused: created_hour, created_day_of_week, created_month, is_weekend.
- [x] Frozen chronological split reused: `20260806T135114Z_9d945cb2da0eecfc`.
- [x] Frozen preprocessing reused: CSR / float64, verified through Phase 2 inputs.
- [x] Frozen leakage policy reused; no target, date, feature, or threshold changes.
- [x] Phase 9 fingerprint reused: `d6620d99301f80884487140bde487479f58fe72c1e209328524fa27e8b5a2271`.

## Candidate selection

- [x] Six deliberate candidates defined with documented rationale.
- [x] Phase 2 reference retained and all nine frozen metrics reproduced within 1e-12.
- [x] Candidate values, IDs, allowed keys, and ordering validated.
- [x] Every candidate fitted on TRAIN only.
- [x] Every candidate evaluated on VALIDATION only.
- [x] TEST not scored during selection; fail-on-access integration guards pass.
- [x] PR-AUC (existing Average Precision), ROC-AUC, and raw Brier evaluated.
- [x] Recall@5%, Recall@10%, Recall@20% evaluated.
- [x] Precision@5%, Precision@10%, Precision@20% evaluated.
- [x] Candidate comparison produced.
- [x] Frozen Phase 2 XGB comparison produced.
- [x] Frozen LR validation snapshot reused without retraining or rescoring.
- [x] Deterministic policy defined in advance with absolute PR-AUC tolerance 1e-6.
- [x] One configuration selected: `shallow`, 200 trees, learning rate 0.05, depth 2.
- [x] Immutable decision persisted with all candidate definitions, policy, seed, identifiers, timestamp and version.
- [x] Selection artifact created exclusively; overwrite, changed freeze, and reselection rejected.

## Final evaluation

- [x] Final TEST evaluation occurred only after persisted selection freeze.
- [x] Existing TRAIN-only final fitting convention preserved; selected fitted model reused without refit.
- [x] Only the selected candidate scored on TEST.
- [x] One-time TEST access marker persisted before scoring; repeated final evaluation rejected.
- [x] Frozen LR final evidence reused, checked against model metadata and population, and copied into a bound snapshot.
- [x] All nine final TEST metrics and LR comparison produced.
- [x] No parameter, candidate, feature, threshold, or selection changes after seeing TEST.

## Validation

- [x] Required unit and integration tests pass.
- [x] Existing wrapper compatibility tests pass.
- [x] Full repository suite passes.
- [x] Syntax and whitespace checks pass.
- [x] Artifact consistency audit passes.
- [x] Main report and completion gate produced.

| Command | Result |
|---|---|
| `PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/unit/models/test_gradient_boosting_tuning.py tests/unit/models/test_gradient_boosting.py tests/integration/test_gradient_boosting_tuning_workflow.py` | 80 passed |
| Phase 3 unit and integration tests after final contract guards | 39 passed |
| `PYTHONPATH=src:. .venv/bin/python -m pytest -q` | 1057 passed |
| `PYTHONPATH=src:. .venv/bin/python -m compileall -q src/urban_ops/models tests/unit/models/test_gradient_boosting_tuning.py tests/integration/test_gradient_boosting_tuning_workflow.py` | Passed |
| `git diff --check` | Passed |
| Phase 3 JSON/CSV/Markdown consistency audit | Passed |

No repository lint/type-check configuration or installed formatter was found;
no lint or type-check result is claimed. The real-data workflow emitted
non-failing Arrow CPU-cache discovery messages under the sandbox and exited
successfully. No test failures, skips, or xfails were reported in the final run.

## Outcome

Validation PR-AUC improved from Phase 2's 0.45137514 to 0.46447317, but remained
below frozen LR's 0.46588883. On TEST, selected XGB PR-AUC was 0.36871428 versus
LR's 0.38984932; XGB trailed LR on all nine final metrics. This experiment does
not justify replacing LR. The Month 1 model and threshold remain unchanged.

Phase 3 is formally frozen. No probability calibration, threshold tuning,
SHAP, new features, or later-phase work was introduced. Phase 4 remains future
work.
