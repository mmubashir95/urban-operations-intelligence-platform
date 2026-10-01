# Phase 2.2 Gradient Boosting Implementation Choice

## Frozen choice

- Library: XGBoost
- Estimator: `XGBClassifier`
- Dependency constraint: `xgboost>=3.0.2,<4`
- Selection record: `configs/models/resolution_risk_gradient_boosting.yaml`

The lower bound is a stable release with Python 3.13 package support, matching
the current project runtime, and the upper bound prevents an unreviewed major
upgrade. Local verification resolved XGBoost 3.4.1. On macOS, the XGBoost wheel
also requires the system OpenMP runtime; install it with `brew install libomp`
when it is not already available.

XGBoost was selected as the single implementation for the current
resolution-risk path because it supports binary classification on structured
tabular data, probability output, SciPy sparse CSR matrices, later
regularization and tuning, and later SHAP-based explanation work.

This choice does not claim that XGBoost is universally better than other
gradient-boosting libraries. The future project experiment will compare the
frozen Month 1 Logistic Regression with XGBoost. Phase 2.2 does not compare
XGBoost with LightGBM, HistGradientBoosting, or CatBoost.

## Scope boundary

Phase 2.2 performs only a tiny synthetic compatibility smoke test. No real
project model was trained, no project validation metrics were calculated, no
test labels were used, and no hyperparameter tuning was performed. The smoke
test parameters are test-only and are not the final production XGBoost
configuration.

Phase 2.1 remains the required input gate. The production model wrapper,
training behavior, scoring API, evaluation, model comparison, and persistence
belong to later phases.

## Phase 2.3 project wrapper

`GradientBoostedRiskModel` encapsulates `XGBClassifier` behind the same
project-facing shape used by the Logistic Regression baseline:

- `fit(X_train, y_train, feature_names=...)`
- `predict_score(X)`
- `predict_proba(X)`
- `predict(X, threshold=0.5)`

The wrapper consumes already-preprocessed matrices and preserves the supplied
frozen feature-name order. No preprocessing, data splitting, evaluation,
calibration, hyperparameter tuning, test-set logic, or persistence happens
inside the wrapper. The default threshold of `0.5` is only a technical method
default and is not a frozen operational decision.

## Phase 2.4 initial configuration

The single authoritative starting configuration lives in
`configs/models/resolution_risk_gradient_boosting.yaml`. Default construction
of `GradientBoostedRiskModel` loads and validates that file before constructing
`XGBClassifier`.

| Parameter | Value | Purpose |
| --- | ---: | --- |
| `objective` | `binary:logistic` | Produce probabilities for binary classification. |
| `eval_metric` | `logloss` | Use a probability-oriented training objective report. |
| `n_estimators` | `100` | Start with a modest fixed number of trees. |
| `learning_rate` | `0.1` | Use a conservative contribution from each tree. |
| `max_depth` | `3` | Limit the initial trees to shallow interactions. |
| `random_state` | `20260806` | Reuse the project seed for reproducibility. |
| `n_jobs` | `1` | Keep CPU execution controlled and deterministic. |

These are conservative starting defaults for the first experiment, not an
optimized or final model configuration. No parameter sweep, hyperparameter
tuning, early stopping, or validation-driven selection was performed. No real
project data was trained or evaluated while selecting or verifying these
values; verification uses only tiny synthetic sparse matrices.

## Phase 2.5 train-only fitting

`load_and_train_gradient_boosted_risk_model()` begins with the Phase 2.1 input
gate, constructs the Phase 2.3 wrapper using the unchanged Phase 2.4
configuration, and calls `fit()` with only:

- `inputs.matrices["train"]`
- `inputs.targets["train"]`
- `inputs.feature_names`

The fitted model remains in memory and is not persisted or promoted. Training
metadata records the split identity, training rows, ordered features, class
counts, positive-class rate, configuration version, and seed. It contains no
performance metrics.

Validation is not passed to `fit()` or `eval_set`, and no early stopping is
used. Test data is not used. Phase 2.5 produces no validation or test
predictions and performs no evaluation, tuning, calibration, or explanation.

The verified repository training run recorded descriptive metadata only:

| Field | Value |
| --- | ---: |
| Frozen split ID | `20260806T135114Z_9d945cb2da0eecfc` |
| Training rows | 23,699 |
| Features | 4 |
| Negative-class rows | 12,792 |
| Positive-class rows | 10,907 |
| Positive-class rate | 0.4602303895 |
| Configuration version | 1 |
| Random seed | 20260806 |

These values describe the rows used for fitting; they are not model-quality
metrics. Repeated fits produced identical scores on a fixed small training
slice, confirming the configured training path is reproducible.

## Phase 2.6 validation risk probabilities

`generate_gradient_boosting_validation_scores()` receives the already-fitted
Phase 2.5 `GradientBoostedRiskModel` and the verified frozen inputs. It passes
`inputs.matrices["validation"]` directly to `model.predict_score()` and returns
one one-dimensional positive-class risk score per validation complaint in the
original matrix row order.

The score is the estimated risk of target class `1`, meaning the complaint
misses its resolution target. No threshold is applied, validation labels are
not needed, and the test split is not accessed. The helper performs no
training, ranking, metric calculation, calibration, model comparison, or
artifact persistence.

The verified repository scoring run produced 6,762 aligned validation scores
with shape `(6762,)`. All values were finite and within `[0, 1]`; the observed
contract range was `0.1989542842` to `0.6090874076`. This range is descriptive
output validation only, not a model-quality evaluation. The fitted booster was
unchanged after score generation.

## Phase 2.7 shared ranking evaluation

`evaluate_gradient_boosting_validation_ranking()` passes the frozen
`inputs.targets["validation"]` and the unchanged Phase 2.6 continuous score
vector directly to the existing Month 1 `evaluate_ranking()` function. It
returns the existing `RankingEvaluation`, including PR-AUC (the established
average-precision definition) and ROC-AUC.

No threshold or hard predictions are used, and the Month 1 threshold `0.49`
is not applied. Only validation is evaluated; test scores and labels remain
untouched. This phase adds no Brier, calibration, Top-K, comparison, tuning, or
model-selection logic.

The verified repository validation run evaluated 6,762 aligned rows and
produced PR-AUC `0.4513751420` and ROC-AUC `0.5441074891`. These are Gradient
Boosting validation ranking results only; Phase 2.7 makes no final comparison
or model-selection decision.

## Phase 2.8 shared raw-probability calibration evaluation

`evaluate_gradient_boosting_validation_calibration()` passes the frozen
`inputs.targets["validation"]` and the same unchanged Phase 2.6 probability
vector directly to the existing Month 1 `evaluate_calibration()` function. Its
reporting table is produced by the existing `build_calibration_table()` helper,
preserving the canonical ten uniform bins, column names, boundary convention,
and empty-bin representation.

This phase evaluates raw positive-class probabilities only. It does not apply
Platt scaling, isotonic regression, a sigmoid, a threshold, or any other
probability transformation. Training and test probabilities and labels are not
used. The measured Brier score and bin findings are recorded in
`reports/month_2/phase_2_gradient_boosting.md`, with the canonical table in
`reports/month_2/phase_2_gradient_boosting_calibration.csv`.

## Phase 2.9 shared operational Top-K evaluation

`evaluate_gradient_boosting_validation_capacity()` obtains the established
5%, 10%, and 20% review levels from `get_standard_capacity_levels()` and
passes the frozen validation labels and unchanged Phase 2.6 probability vector
to `compare_capacity_levels()`. That shared evaluator reuses deterministic
descending-risk ranking, stable original-order tie handling, and the existing
`ceil(validation rows × capacity)` selected-count rule.

Top-K selection uses continuous raw risk and applies no classification
threshold, including the frozen Logistic Regression threshold `0.49`. No
capacity is selected or recommended. Test scores and labels remain untouched.
Results are recorded in
`reports/month_2/phase_2_gradient_boosting_capacity.csv` and summarized in the
Phase 2 Markdown report.

## Phase 2.11 dedicated Gradient Boosting workflow

`run_gradient_boosting_workflow()` is the dedicated Month 2 orchestration
boundary. It loads and verifies the frozen Month 1 model inputs, performs the
existing train-only fit, generates validation probabilities exactly once, and
passes that same array directly to the shared ranking, calibration, and
capacity evaluators. It returns a structured `GradientBoostingWorkflowResult`
and writes the complete Month 2 report set.

The workflow loads frozen Logistic Regression validation evidence from CSV for
comparison; it does not retrain Logistic Regression. It never invokes
`run_baseline_workflow()`, never applies the frozen Logistic Regression
threshold, and never scores or evaluates the test split. Run it with:

```bash
make gradient-boosting-resolution-risk
```

## Phase 2.12 frozen Logistic Regression comparison

The Month 2 reporting layer loads Logistic Regression validation metrics from
the authoritative structured Month 1 artifacts:

- `reports/tables/baseline_validation_results.csv` for PR-AUC, ROC-AUC, and
  Brier Score.
- `reports/tables/logistic_regression_validation_capacity_comparison.csv` for
  Precision@K and Recall@K at 5%, 10%, and 20% capacity.

The loader validates artifact existence, required columns, model identity,
validation split provenance, finite unit-interval metrics, standard capacities,
and reconciliation of selected/captured counts with the frozen validation
population. It returns typed frozen evidence with source paths and normalized
metric names. The workflow compares those values with its current validation
results using `Gradient Boosting - Logistic Regression` for every metric,
including Brier Score. Logistic Regression is not retrained or rescored.
