# Phase 3 — controlled model selection

Tuning was needed because Phase 2 XGB did not outperform frozen LR.

Phase 2 configuration: `{"eval_metric": "logloss", "learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "n_jobs": 1, "objective": "binary:logistic", "random_state": 20260806}`.

Only eight XGBoost hyperparameters may change. Target, feature order, preprocessing, chronology, split dates, leakage rules, seed, metric implementations, LR and its threshold remain frozen.
Features: created_hour, created_day_of_week, created_month, is_weekend. Split: `20260806T135114Z_9d945cb2da0eecfc`. Fingerprint: `d6620d99301f80884487140bde487479f58fe72c1e209328524fa27e8b5a2271`.

Every candidate fitted on TRAIN only and was scored and evaluated on VALIDATION only. The verified input loader structurally checks TEST; no TEST probabilities or selection metrics were generated.

Selection policy: Maximize validation PR-AUC; within tolerance of global maximum, maximize Recall@10%, minimize Brier, minimize (max_depth, n_estimators), then lexical candidate ID. PR-AUC tolerance = 1e-06.

Simplicity means lower depth, then fewer trees; final ties use lexical candidate ID. Sampling and regularization defaults stay fixed except where explicitly listed.

- deeper: Test modestly increased interaction capacity with depth four. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 1.0, "n_estimators": 200, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0}`.
- phase2_baseline: Existing Phase 2 reference with XGBoost default regularization and sampling. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "n_estimators": 100, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0}`.
- regularized: Stronger L2 penalty and minimum child weight to limit overfitting. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 5.0, "n_estimators": 200, "reg_alpha": 0.0, "reg_lambda": 5.0, "subsample": 1.0}`.
- shallow: Reduce interaction complexity with depth two. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 1.0, "n_estimators": 200, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0}`.
- slower_learning: Lower learning rate with twice as many trees. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1.0, "n_estimators": 200, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0}`.
- slower_learning_more_trees: Smaller learning steps with similar total boosting budget. Parameters: `{"colsample_bytree": 1.0, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 1.0, "n_estimators": 300, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0}`.

Highest PR-AUC: shallow. Highest Recall@10%: slower_learning_more_trees. Selected: **shallow**.

Against frozen Phase 2 XGB: PR-AUC difference = +0.01309803. Improved: pr_auc, roc_auc, brier_score, recall_at_05, recall_at_10, recall_at_20, precision_at_05, precision_at_10, precision_at_20. Regressed: none.

Against frozen LR: PR-AUC difference = -0.00141566. Improved: brier_score, recall_at_05, precision_at_05. Regressed: pr_auc, roc_auc, recall_at_10, recall_at_20, precision_at_10, precision_at_20.

Validation PR-AUC favors Logistic Regression. Selecting the best XGB candidate does not replace or force promotion over LR.

Selection is persisted and FROZEN before TEST. No candidate additions, parameter changes, threshold search, calibration fitting, or validation-driven retuning are allowed. Final evaluation preserves Month 1's TRAIN-only fit convention and reuses this fitted model.
