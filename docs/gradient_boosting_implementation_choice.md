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
