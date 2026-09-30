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
