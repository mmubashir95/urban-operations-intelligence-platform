# Probability calibration

Month 2 Phase 4 is a controlled probability layer around the frozen Phase 3
`shallow` XGBoost classifier. The base classifier keeps its exact feature,
preprocessing, chronological split, leakage, random-state, and hyperparameter
contracts and is fitted on TRAIN only.

RAW, SIGMOID, and ISOTONIC are the complete candidate set. Sigmoid uses
scikit-learn logistic regression over the frozen model's validation scores;
isotonic uses scikit-learn's bounded monotonic regression. Both mappings are
fit on VALIDATION. The existing evaluation module supplies Brier Score, ten
uniform calibration bins, PR-AUC, ROC-AUC, and Top-K precision/recall.

The configured selection policy requires an absolute Brier improvement over
RAW and rejects any mapping that degrades a ranking or Top-K metric beyond the
explicit guard. Lowest eligible Brier wins; effective ties prefer RAW, then
SIGMOID, then ISOTONIC. RAW is therefore a valid frozen outcome.

The decision is written before TEST access. Final evaluation applies the
already-fitted selected mapping without refitting, switching methods, or using
TEST labels for calibration. No classification threshold is selected.
