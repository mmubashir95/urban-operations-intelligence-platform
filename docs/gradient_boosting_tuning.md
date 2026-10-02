# Phase 3 tuning workflow

`resolution_risk_gradient_boosting_tuning.yaml` defines six candidates, their
rationale, eight allowed parameter names, the project seed, selection policy,
and TEST access policy. The split ID and feature fingerprint resolve from the
existing frozen LR snapshot rather than duplicating features or split dates.
The fingerprint covers the full Phase 9 feature/preprocessing contract.

`gradient_boosting_tuning.py` validates definitions and uses the existing
`GradientBoostedRiskModel` with an optional typed configuration. Calls using
the original `config_path` interface retain Phase 2 behavior. Defaults for
minimum child weight, sampling, and regularization explicitly reproduce the
Phase 2 estimator defaults; its replay must match all nine frozen validation
metrics within 1e-12 before selection proceeds.

`gradient_boosting_selection.py` reuses `evaluate_ranking`,
`evaluate_calibration` (raw Brier assessment only), and
`compare_capacity_levels`. PR-AUC means the existing Average Precision metric.
The PR-AUC tie band is measured from the global maximum with absolute tolerance
1e-6. Within that band, prefer higher Recall@10%, lower Brier, lower depth,
fewer trees, then lexical candidate ID. No composite score is used.

`run_candidate_selection` returns candidate scores and immutable evaluations,
retains the winner's fitted wrapper, and writes its immutable decision with
exclusive creation. The decision includes all candidates, hyperparameters,
validation metrics, policy, UTC timestamp, version, seed, ordered features,
split ID, and fingerprint. The authoritative artifact is
`reports/month_2/phase_3_selected_configuration.json`; no duplicate selected
config is maintained under `configs/models`.

`run_final_test_evaluation` verifies the persisted decision and model config,
then writes an exclusive TEST access marker before reading TEST. It reuses the
TRAIN-fitted winner: Month 1 already established this final evaluation
convention. No TRAIN+VALIDATION refit is introduced. Only the selected candidate
is scored, using the same nine metrics. Existing LR final evidence is read
without model loading, rescoring, or retraining; its model metadata,
fingerprint, ordered features, population counts, and metric ranges are
validated. A bound, write-once LR TEST evidence copy is retained beside the
comparison. The original LR files remain unchanged.

The verified input loader structurally checks all three splits before the
experiment. After that boundary, candidate selection retrieves TRAIN and
VALIDATION only. Tests guard TEST retrieval during selection and assert that
all candidate fits and evaluations receive the original frozen objects.

## Running and verification

```bash
make tune-gradient-boosting-resolution-risk
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

The real-data experiment has already run and is frozen. The first command now
refuses to overwrite the decision. Use the fixture-based tests to verify the
workflow without repeating the production experiment. `--output-directory`,
`--tuning-config`, and `--split-run` are available for explicitly separate
experiments; these do not authorize replacing the frozen Phase 3 decision.

A TEST access marker also prevents repeating scoring if an evaluation fails.
Do not remove the freeze or marker to resume an experiment. A process crash
between freeze and report completion requires inspection of the artifacts;
the workflow deliberately does not silently reopen selection or TEST.

Reports document validation gains and regressions and the final LR comparison.
Selecting an XGB configuration is independent of promoting it over LR. In this
experiment, shallow XGB improves all nine metrics over Phase 2 on validation,
but its PR-AUC remains below LR; it trails LR on all nine final TEST metrics.
The LR benchmark remains unchanged. Calibration fitting and all later phases
remain outside Phase 3.
