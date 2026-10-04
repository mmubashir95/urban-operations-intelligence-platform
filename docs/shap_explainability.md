# Tree SHAP explainability

Month 2 Phase 5 is a read-only explanation layer around the frozen Phase 3
`shallow` XGBoost classifier. Because Phase 3 persisted its configuration but
not a fitted binary, the workflow reconstructs the exact TRAIN-only estimator
once, requires all frozen validation metrics to reproduce within `1e-12`, then
saves and reloads a native XGBoost artifact before explaining it.

The environment uses Python 3.13 on macOS x86_64. SHAP 0.48 and 0.50 both
require `llvmlite`, for which no compatible wheel was available; building it
also required an unavailable local LLVM installation. Phase 5 therefore uses
XGBoost 3.4.1's supported native Tree SHAP APIs (`pred_contribs` and
`pred_interactions`). KernelExplainer is not used and no approximation is
introduced.

The explanation population is a deterministic 2,000-row sample of the already
finalized TEST split. Six local roles are reserved before seeded sampling. The
existing `0.5` wrapper default is used only to label descriptive TP/TN/FP/FN
examples; it is not selected or promoted as an operational threshold.

Attributions are in raw XGBoost margin/log-odds space. They are checked against
the model's raw margin for every explanation row. They do not represent direct
probability-point changes, do not explain Phase 4 isotonic post-processing, and
must not be interpreted as causal effects or evidence that the weak model is
accurate.
